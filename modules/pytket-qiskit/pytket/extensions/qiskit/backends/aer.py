# Copyright 2019-2021 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from collections import defaultdict
from logging import warning
from typing import Dict, Iterable, List, Optional, Tuple, cast, TYPE_CHECKING, Set

import numpy as np  # type: ignore
import qiskit.providers.aer.extensions.snapshot_expectation_value  # type: ignore # pylint: disable=unused-import
from pytket.backends import Backend, CircuitNotRunError, CircuitStatus, ResultHandle
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import BasisOrder, Circuit, Node, OpType, Qubit  # type: ignore
from pytket.device import Device, QubitErrorContainer  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    CliffordSimp,
    CXMappingPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    RebaseCustom,
    RebaseIBM,
    SequencePass,
    SynthesiseIBM,
)
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.predicates import (  # type: ignore
    ConnectivityPredicate,
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.extensions.qiskit.qiskit_convert import (
    tk_to_qiskit,
    _qiskit_gates_1q,
    _qiskit_gates_2q,
    _gate_str_2_optype,
)
from pytket.extensions.qiskit.result_convert import qiskit_result_to_backendresult
from pytket.routing import Architecture, NoiseAwarePlacement  # type: ignore
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.results import KwargTypes, permute_basis_indexing
from qiskit import Aer
from qiskit.compiler import assemble  # type: ignore
from qiskit.providers.aer.noise import NoiseModel  # type: ignore
from qiskit.quantum_info.operators import Pauli as qk_Pauli  # type: ignore

from .ibm_utils import _STATUS_MAP

if TYPE_CHECKING:
    from qiskit.providers.aer import AerJob  # type: ignore
    from qiskit.providers.aer.backends.aerbackend import AerBackend as QiskitAerBackend  # type: ignore


def _default_q_index(q: Qubit) -> int:
    if q.reg_name != "q" or len(q.index) != 1:
        raise ValueError("Non-default qubit register")
    return int(q.index[0])


_required_gates: Set[OpType] = {OpType.CX, OpType.U1, OpType.U2, OpType.U3}
_1q_gates: Set[OpType] = set(_qiskit_gates_1q.values())
_2q_gates: Set[OpType] = set(_qiskit_gates_2q.values())


def _tk1_to_u(a: float, b: float, c: float) -> Circuit:
    circ = Circuit(1)
    circ.add_gate(OpType.U3, [b, a - 0.5, c + 0.5], [0])
    circ.add_phase(-0.5 * (a + c))
    return circ


class _AerBaseBackend(Backend):
    """Common base class for all Aer simulator backends"""

    _persistent_handles = False

    def __init__(self, backend_name: str):
        super().__init__()
        self._backend: "QiskitAerBackend" = Aer.get_backend(backend_name)
        self._gate_set: Set[OpType] = {
            _gate_str_2_optype[gate_str]
            for gate_str in self._backend.configuration().basis_gates
            if gate_str in _gate_str_2_optype
        }
        if not self._gate_set >= _required_gates:
            raise NotImplementedError(
                f"Gate set {self._gate_set} missing at least one of {_required_gates}"
            )
        self._noise_model: Optional[NoiseModel] = None
        self._characterisation: Optional[dict] = None
        self._device: Optional[Device] = None
        self._memory = False

        self._rebase_pass = RebaseCustom(
            self._gate_set & _2q_gates,
            Circuit(2).CX(0, 1),
            self._gate_set & _1q_gates,
            _tk1_to_u,
        )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, int)

    @property
    def characterisation(self) -> Optional[dict]:
        return self._characterisation

    @property
    def device(self) -> Optional[Device]:
        return self._device

    def process_circuits(
        self,
        circuits: Iterable[Circuit],
        n_shots: Optional[int] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        circuit_list = list(circuits)

        if valid_check:
            self._check_all_circuits(circuit_list)

        qcs = [tk_to_qiskit(tkc) for tkc in circuit_list]
        seed = cast(Optional[int], kwargs.get("seed"))
        qobj = assemble(qcs, shots=n_shots, memory=self._memory, seed_simulator=seed)
        job = self._backend.run(qobj, noise_model=self._noise_model)
        jobid = job.job_id()
        handle_list = [ResultHandle(jobid, i) for i in range(len(circuit_list))]
        for handle in handle_list:
            self._cache[handle] = {"job": job}
        return handle_list

    def cancel(self, handle: ResultHandle) -> None:
        job: "AerJob" = self._cache[handle]["job"]
        cancelled = job.cancel()
        if not cancelled:
            warning(f"Unable to cancel job {cast(str, handle[0])}")

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        self._check_handle_type(handle)
        job: "AerJob" = self._cache[handle]["job"]
        ibmstatus = job.status()
        return CircuitStatus(_STATUS_MAP[ibmstatus], ibmstatus.value)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            jobid, _ = handle
            try:
                job: "AerJob" = self._cache[handle]["job"]
            except KeyError:
                raise CircuitNotRunError(handle)

            res = job.result()
            backresults = qiskit_result_to_backendresult(res)
            for circ_index, backres in enumerate(backresults):
                self._cache[ResultHandle(jobid, circ_index)]["result"] = backres

            return cast(BackendResult, self._cache[handle]["result"])

    def _snapshot_expectation_value(
        self,
        circuit: Circuit,
        hamiltonian: List[Tuple[complex, qk_Pauli]],
        valid_check: bool = True,
    ) -> complex:
        if valid_check:
            self._check_all_circuits([circuit], nomeasure_warn=False)

        circ_qbs = circuit.qubits
        q_indices = (_default_q_index(q) for q in circ_qbs)
        if not all(q_ind == i for q_ind, i in zip(q_indices, range(len(circ_qbs)))):
            raise ValueError(
                "Circuit must act on default register Qubits, contiguously from 0"
                + f" onwards. Circuit qubits were: {circ_qbs}"
            )
        qc = tk_to_qiskit(circuit)
        qc.snapshot_expectation_value("snap", hamiltonian, qc.qubits)
        qobj = assemble(qc)
        job = self._backend.run(qobj)
        return cast(
            complex,
            job.result().data(qc)["snapshots"]["expectation_value"]["snap"][0]["value"],
        )

    def get_pauli_expectation_value(
        self,
        state_circuit: Circuit,
        pauli: QubitPauliString,
        valid_check: bool = True,
    ) -> complex:
        """Calculates the expectation value of the given circuit using the built-in Aer
        snapshot functionality
        Requires a simple circuit with default register qubits.

        :param state_circuit: Circuit that generates the desired state
            :math:`\\left|\\psi\\right>`.
        :type state_circuit: Circuit
        :param pauli: Pauli operator
        :type pauli: QubitPauliString
        :param valid_check: Explicitly check that the circuit satisfies all required
            predicates to run on the backend. Defaults to True
        :type valid_check: bool, optional
        :return: :math:`\\left<\\psi | P | \\psi \\right>`
        :rtype: complex
        """
        if not self._supports_expectation:
            raise NotImplementedError("Cannot get expectation value from this backend")

        operator = [(1 + 0j, _sparse_to_qiskit_pauli(pauli, state_circuit.n_qubits))]
        return self._snapshot_expectation_value(state_circuit, operator, valid_check)

    def get_operator_expectation_value(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        valid_check: bool = True,
    ) -> complex:
        """Calculates the expectation value of the given circuit with respect to the
        operator using the built-in Aer snapshot functionality
        Requires a simple circuit with default register qubits.

        :param state_circuit: Circuit that generates the desired state
            :math:`\\left|\\psi\\right>`.
        :type state_circuit: Circuit
        :param operator: Operator :math:`H`.
        :type operator: QubitPauliOperator
        :param valid_check: Explicitly check that the circuit satisfies all required
            predicates to run on the backend. Defaults to True
        :type valid_check: bool, optional
        :return: :math:`\\left<\\psi | H | \\psi \\right>`
        :rtype: complex
        """
        if not self._supports_expectation:
            raise NotImplementedError("Cannot get expectation value from this backend")

        q_operator = []
        for term, coeff in operator._dict.items():
            q_operator.append(
                (coeff, _sparse_to_qiskit_pauli(term, state_circuit.n_qubits))
            )
        return self._snapshot_expectation_value(state_circuit, q_operator, valid_check)


class _AerStateBaseBackend(_AerBaseBackend):
    def __init__(self, *args: str, **kwargs: KwargTypes):
        self._qlists: Dict[ResultHandle, Tuple[int, ...]] = {}
        super().__init__(*args)

    @property
    def required_predicates(self) -> List[Predicate]:
        return [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            GateSetPredicate(
                self._gate_set.union(
                    {
                        OpType.noop,
                        OpType.Unitary1qBox,
                    }
                )
            ),
        ]

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        if optimisation_level == 0:
            return SequencePass([DecomposeBoxes(), RebaseIBM()])
        elif optimisation_level == 1:
            return SequencePass([DecomposeBoxes(), SynthesiseIBM()])
        else:
            return SequencePass([DecomposeBoxes(), FullPeepholeOptimise()])

    def process_circuits(
        self,
        circuits: Iterable[Circuit],
        n_shots: Optional[int] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        handles = super().process_circuits(
            circuits, n_shots=None, valid_check=valid_check, **kwargs
        )
        for handle, circ in zip(handles, circuits):
            perm: Dict[Qubit, Qubit] = circ.implicit_qubit_permutation()
            if not all(key == val for key, val in perm.items()):
                self._cache[handle]["implicit_perm_qubits"] = perm
        return handles

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        if handle in self._cache:
            if "result" in self._cache[handle]:
                return cast(BackendResult, self._cache[handle]["result"])

        self._check_handle_type(handle)
        try:
            job: "AerJob" = self._cache[handle]["job"]
        except KeyError:
            raise CircuitNotRunError(handle)

        res = job.result()
        backresults = qiskit_result_to_backendresult(res)
        for circ_index, backres in enumerate(backresults):
            newhandle = ResultHandle(handle[0], circ_index)
            if "implicit_perm_qubits" in self._cache[newhandle]:
                permed_qbit_map: Dict[Qubit, Qubit] = self._cache[newhandle][
                    "implicit_perm_qubits"
                ]
                original_indexmap = backres.q_bits.copy()
                assert original_indexmap
                # Simultaneous permutation of inputs and outputs of process
                # Handles implicit permutation of outputs for statevector
                backres.q_bits = {
                    permed_qbit_map[qb]: index
                    for qb, index in original_indexmap.items()
                }

                if backres._unitary is not None:
                    # For unitaries, the implicit permutation
                    #  should only be applied to inputs
                    # The above relabelling will permute both inputs and outputs
                    # Correct by applying the inverse
                    # permutation on the inputs (i.e. a column permutation)
                    permutation = [0] * len(original_indexmap)
                    for qb, index in original_indexmap.items():
                        permutation[index] = original_indexmap[permed_qbit_map[qb]]
                    backres._unitary = permute_basis_indexing(
                        backres._unitary.T, tuple(permutation)
                    ).T
            self._cache[newhandle]["result"] = backres

        return cast(BackendResult, self._cache[handle]["result"])


class AerBackend(_AerBaseBackend):
    _supports_shots = True
    _supports_counts = True
    _supports_expectation = True

    def __init__(
        self,
        noise_model: Optional[NoiseModel] = None,
        simulation_method: str = "automatic",
    ):
        """Backend for running simulations on the Qiskit Aer QASM simulator.

        :param noise_model: Noise model to apply during simulation. Defaults to None.
        :type noise_model: Optional[NoiseModel], optional
        :param simulation_method: Simulation method, see
         https://qiskit.org/documentation/stubs/qiskit.providers.aer.QasmSimulator.html
         for available values. Defaults to "automatic".
        :type simulation_method: str
        """
        super().__init__("qasm_simulator")

        if not noise_model or all(
            value == [] for value in noise_model.to_dict().values()
        ):
            self._noise_model = None
        else:
            self._noise_model = noise_model
            self._characterisation = _process_model(noise_model, self._gate_set)

            self._device = Device(
                self._characterisation.get("NodeErrors", {}),
                self._characterisation.get("EdgeErrors", {}),
                self._characterisation.get("Architecture", Architecture([])),
            )
        self._memory = True

        self._backend.set_options(method=simulation_method)

    @property
    def required_predicates(self) -> List[Predicate]:
        pred_list = [
            NoSymbolsPredicate(),
            GateSetPredicate(
                self._gate_set.union(
                    {
                        OpType.Measure,
                        OpType.Reset,
                        OpType.Barrier,
                        OpType.noop,
                        OpType.Unitary1qBox,
                        OpType.RangePredicate,
                    }
                )
            ),
        ]
        if self._noise_model and self._device:
            pred_list.append(ConnectivityPredicate(self._device))
        return pred_list

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passlist = [DecomposeBoxes()]
        if optimisation_level == 0:
            passlist.append(self._rebase_pass)
        elif optimisation_level == 1:
            passlist.append(SynthesiseIBM())
        else:
            passlist.append(FullPeepholeOptimise())
        if self._noise_model and self._device:
            passlist.append(
                CXMappingPass(
                    self._device,
                    NoiseAwarePlacement(self._device),
                    directed_cx=True,
                    delay_measures=False,
                )
            )
            if optimisation_level == 0:
                passlist.append(self._rebase_pass)
            elif optimisation_level == 1:
                passlist.append(SynthesiseIBM())
            else:
                passlist.extend([CliffordSimp(False), SynthesiseIBM()])
        return SequencePass(passlist)

    def process_circuits(
        self,
        circuits: Iterable[Circuit],
        n_shots: Optional[int] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: `seed`.
        """
        if n_shots is None or n_shots < 1:
            raise ValueError(
                "Parameter n_shots is required for this backend for this backend."
            )
        return super().process_circuits(circuits, n_shots, valid_check, **kwargs)

    def get_pauli_expectation_value(
        self,
        state_circuit: Circuit,
        pauli: QubitPauliString,
        valid_check: bool = True,
    ) -> complex:
        """Calculates the expectation value of the given circuit using the built-in Aer
        snapshot functionality.
        Requires a simple circuit with default register qubits, and no noise model.

        :param state_circuit: Circuit that generates the desired state
            :math:`\\left|\\psi\\right>`.
        :type state_circuit: Circuit
        :param pauli: Pauli operator
        :type pauli: QubitPauliString
        :param valid_check: Explicitly check that the circuit satisfies all required
            predicates to run on the backend. Defaults to True
        :type valid_check: bool, optional
        :return: :math:`\\left<\\psi | P | \\psi \\right>`
        :rtype: complex
        """
        if self._noise_model:
            raise RuntimeError(
                (
                    "Snapshot based expectation value not supported with noise model. "
                    "Use shots."
                )
            )

        return super().get_pauli_expectation_value(state_circuit, pauli, valid_check)

    def get_operator_expectation_value(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        valid_check: bool = True,
    ) -> complex:
        """Calculates the expectation value of the given circuit with respect to the
        operator using the built-in Aer snapshot functionality
        Requires a simple circuit with default register qubits, and no noise model.

        :param state_circuit: Circuit that generates the desired state
            :math:`\\left|\\psi\\right>`.
        :type state_circuit: Circuit
        :param operator: Operator :math:`H`.
        :type operator: QubitPauliOperator
        :param valid_check: Explicitly check that the circuit satisfies all required
            predicates to run on the backend. Defaults to True
        :type valid_check: bool, optional
        :return: :math:`\\left<\\psi | H | \\psi \\right>`
        :rtype: complex
        """
        if self._noise_model:
            raise RuntimeError(
                (
                    "Snapshot based expectation value not supported with noise model. "
                    "Use shots."
                )
            )

        return super().get_operator_expectation_value(
            state_circuit, operator, valid_check
        )


class AerStateBackend(_AerStateBaseBackend):
    _supports_state = True
    _supports_expectation = True

    def __init__(self) -> None:
        """Backend for running simulations on the Qiskit Aer Statevector simulator."""
        super().__init__("statevector_simulator")


class AerUnitaryBackend(_AerStateBaseBackend):
    _supports_unitary = True

    def __init__(self) -> None:
        """Backend for running simulations on the Qiskit Aer Unitary simulator."""
        super().__init__("unitary_simulator")

    def get_unitary(
        self,
        circuit: Circuit,
        basis: BasisOrder = BasisOrder.ilo,
        valid_check: bool = True,
    ) -> np.ndarray:
        """
        Calculate the unitary matrix for a circuit.

        :param circuit: Circuit to execute
        :type circuit: Circuit
        :param basis: Toggle between ILO-BE (increasing lexicographic order of
            bit ids, big-endian) and DLO-BE (decreasing lexicographic order,
            big-endian) for column ordering. Defaults to BasisOrder.ilo.
        :type basis: BasisOrder, optional
        :param valid_check: Explicitly check that the circuit satisfies all of
            the required predicates before running. Defaults to True.
        :type valid_check: bool, optional
        :return: Full statevector in encoding given by `basis`.
        :rtype: np.ndarray
        """

        result, _ = self._process(circuit, valid_check=valid_check)
        q_bits = (
            sorted(result.q_bits.keys(), reverse=(basis is not BasisOrder.ilo))
            if result.q_bits
            else None
        )
        return result.get_unitary(q_bits)


def _process_model(noise_model: NoiseModel, gate_set: Set[OpType]) -> dict:
    # obtain approximations for gate errors from noise model by using probability of
    #  "identity" error
    assert OpType.CX in gate_set
    # TODO explicitly check for and separate 1 and 2 qubit gates
    supported_single_optypes = gate_set.difference({OpType.CX})
    supported_single_optypes.add(OpType.Reset)
    errors = [
        e
        for e in noise_model.to_dict()["errors"]
        if e["type"] == "qerror" or e["type"] == "roerror"
    ]
    link_ers_dict: dict = {}
    node_ers_dict: dict = defaultdict(
        lambda: QubitErrorContainer(supported_single_optypes)
    )
    readout_errors_dict: dict = {}
    generic_single_qerrors_dict: dict = defaultdict(lambda: list())
    generic_2q_qerrors_dict: dict = defaultdict(lambda: list())

    node_ers_qubits: set = set()
    link_ers_qubits: set = set()

    coupling_map = []
    for error in errors:
        name = error["operations"]
        if len(name) > 1:
            raise RuntimeWarning("Error applies to multiple gates.")
        if "gate_qubits" not in error:
            raise RuntimeWarning(
                (
                    "Please define NoiseModel without using the"
                    " add_all_qubit_quantum_error()"
                    " or add_all_qubit_readout_error() method."
                )
            )
        name = name[0]

        qubits = error["gate_qubits"][0]
        node_ers_qubits.add(qubits[0])
        gate_fid = error["probabilities"][-1]
        if len(qubits) == 1:
            if error["type"] == "qerror":
                node_ers_dict[qubits[0]].add_error(
                    (_gate_str_2_optype[name], 1 - gate_fid)
                )
                generic_single_qerrors_dict[qubits[0]].append(
                    (error["instructions"], error["probabilities"])
                )
            elif error["type"] == "roerror":
                node_ers_dict[qubits[0]].add_readout(error["probabilities"][0][1])
                readout_errors_dict[qubits[0]] = error["probabilities"]
            else:
                raise RuntimeWarning("Error type not 'qerror' or 'roerror'.")
        elif len(qubits) == 2:
            # note that if multiple multi-qubit errors are added to the CX gate,
            #  the resulting noise channel is composed and reflected in probabilities
            error_cont = QubitErrorContainer({_gate_str_2_optype[name]: 1 - gate_fid})
            link_ers_qubits.add(qubits[0])
            link_ers_qubits.add(qubits[1])
            link_ers_dict[tuple(qubits)] = error_cont
            # to simulate a worse reverse direction square the fidelity
            rev_error_cont = QubitErrorContainer(
                {_gate_str_2_optype[name]: 1 - gate_fid ** 2}
            )
            link_ers_dict[tuple(qubits[::-1])] = rev_error_cont
            generic_2q_qerrors_dict[tuple(qubits)].append(
                (error["instructions"], error["probabilities"])
            )
            coupling_map.append(qubits)

    free_qubits = node_ers_qubits - link_ers_qubits

    for q in free_qubits:
        for lq in link_ers_qubits:
            coupling_map.append([q, lq])
            coupling_map.append([lq, q])

    for pair in itertools.permutations(free_qubits, 2):
        coupling_map.append(pair)

    # convert qubits to architecture Nodes
    characterisation = {}
    node_ers_dict = {Node(q_index): ers for q_index, ers in node_ers_dict.items()}
    link_ers_dict = {
        (Node(q_indices[0]), Node(q_indices[1])): ers
        for q_indices, ers in link_ers_dict.items()
    }

    characterisation["NodeErrors"] = node_ers_dict
    characterisation["EdgeErrors"] = link_ers_dict
    characterisation["ReadoutErrors"] = readout_errors_dict
    characterisation["GenericOneQubitQErrors"] = generic_single_qerrors_dict
    characterisation["GenericTwoQubitQErrors"] = generic_2q_qerrors_dict
    characterisation["Architecture"] = Architecture(coupling_map)

    return characterisation


def _sparse_to_qiskit_pauli(pauli: QubitPauliString, n_qubits: int) -> qk_Pauli:
    empty = np.zeros(n_qubits)
    q_pauli = qk_Pauli(empty, empty)
    for q, p in pauli.to_dict().items():
        i = _default_q_index(q)
        if p in (Pauli.X, Pauli.Y):
            q_pauli._x[i] = True
        if p in (Pauli.Z, Pauli.Y):
            q_pauli._z[i] = True
    return q_pauli
