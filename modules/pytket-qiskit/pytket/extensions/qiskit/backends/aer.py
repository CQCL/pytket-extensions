# Copyright 2019-2022 Cambridge Quantum Computing
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
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    TYPE_CHECKING,
    Set,
)

import numpy as np
from pytket.backends import Backend, CircuitNotRunError, CircuitStatus, ResultHandle
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit, Node, OpType, Qubit  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    CliffordSimp,
    CXMappingPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    SequencePass,
    SynthesiseTket,
    auto_rebase_pass,
    NaivePlacementPass,
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
    _gate_str_2_optype,
    get_avg_characterisation,
)
from pytket.extensions.qiskit.result_convert import qiskit_result_to_backendresult
from pytket.extensions.qiskit._metadata import __extension_version__
from pytket.architecture import Architecture  # type: ignore
from pytket.placement import NoiseAwarePlacement  # type: ignore
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.results import KwargTypes, permute_basis_indexing
from qiskit import Aer  # type: ignore
from qiskit.providers.aer.library import (  # type: ignore # pylint: disable=unused-import
    save_expectation_value,
)
from qiskit.providers.aer.noise import NoiseModel  # type: ignore
from qiskit.quantum_info.operators import Pauli as qk_Pauli  # type: ignore
from qiskit.quantum_info.operators.symplectic.pauli_table import PauliTable  # type: ignore
from qiskit.quantum_info.operators.symplectic.sparse_pauli_op import SparsePauliOp  # type: ignore

from .ibm_utils import _STATUS_MAP, _batch_circuits

if TYPE_CHECKING:
    from qiskit.providers.aer import AerJob  # type: ignore
    from qiskit.providers.aer.backends.aerbackend import AerBackend as QiskitAerBackend  # type: ignore


def _default_q_index(q: Qubit) -> int:
    if q.reg_name != "q" or len(q.index) != 1:
        raise ValueError("Non-default qubit register")
    return int(q.index[0])


_required_gates: Set[OpType] = {OpType.CX, OpType.U1, OpType.U2, OpType.U3}


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
        # special case mapping TK1 to U
        self._gate_set.add(OpType.TK1)
        if not self._gate_set >= _required_gates:
            raise NotImplementedError(
                f"Gate set {self._gate_set} missing at least one of {_required_gates}"
            )
        self._backend_info = BackendInfo(
            type(self).__name__,
            backend_name,
            __extension_version__,
            Architecture([]),
            self._gate_set,
            supports_midcircuit_measurement=True,  # is this correct?
            misc={"characterisation": None},
        )

        self._memory = False
        self._noise_model: Optional[NoiseModel] = None

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, int)

    @property
    def backend_info(self) -> BackendInfo:
        return self._backend_info

    def rebase_pass(self) -> BasePass:
        return auto_rebase_pass(
            self._gate_set,
        )

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=True,
        )

        if valid_check:
            self._check_all_circuits(circuits)

        handle_list: List[Optional[ResultHandle]] = [None] * len(circuits)
        circuit_batches, batch_order = _batch_circuits(circuits, n_shots_list)

        for (n_shots, batch), indices in zip(circuit_batches, batch_order):
            qcs = [tk_to_qiskit(tkc) for tkc in batch]
            if self._backend_info.device_name == "aer_simulator_statevector":
                for qc in qcs:
                    qc.save_state()
            elif self._backend_info.device_name == "aer_simulator_unitary":
                for qc in qcs:
                    qc.save_unitary()
            seed = cast(Optional[int], kwargs.get("seed"))
            job = self._backend.run(
                qcs,
                shots=n_shots,
                memory=self._memory,
                seed_simulator=seed,
                noise_model=self._noise_model,
            )
            jobid = job.job_id()
            for i, ind in enumerate(indices):
                handle = ResultHandle(jobid, i)
                handle_list[ind] = handle
                self._cache[handle] = {"job": job}
        return cast(List[ResultHandle], handle_list)

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
        hamiltonian: Union[SparsePauliOp, qk_Pauli],
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
        qc.save_expectation_value(hamiltonian, qc.qubits, "snap")
        job = self._backend.run(qc)
        return cast(
            complex,
            job.result().data(qc)["snap"],
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

        operator = qk_Pauli(_sparse_to_zx_tup(pauli, state_circuit.n_qubits))
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

        sparse_op = _qubitpauliop_to_sparsepauliop(operator, state_circuit.n_qubits)
        return self._snapshot_expectation_value(state_circuit, sparse_op, valid_check)


class _AerStateBaseBackend(_AerBaseBackend):
    def __init__(self, *args: str):
        self._qlists: Dict[ResultHandle, Tuple[int, ...]] = {}
        super().__init__(*args)

    @property
    def required_predicates(self) -> List[Predicate]:
        return [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            GateSetPredicate(
                self._backend_info.gate_set.union(
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
            return SequencePass([DecomposeBoxes(), self.rebase_pass()])
        elif optimisation_level == 1:
            return SequencePass([DecomposeBoxes(), SynthesiseTket()])
        else:
            return SequencePass([DecomposeBoxes(), FullPeepholeOptimise()])

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
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
    _expectation_allows_nonhermitian = False

    def __init__(
        self,
        noise_model: Optional[NoiseModel] = None,
        simulation_method: str = "automatic",
    ):
        """Backend for running simulations on the Qiskit Aer QASM simulator.

        :param noise_model: Noise model to apply during simulation. Defaults to None.
        :type noise_model: Optional[NoiseModel], optional
        :param simulation_method: Simulation method, see
         https://qiskit.org/documentation/stubs/qiskit.providers.aer.AerSimulator.html
         for available values. Defaults to "automatic".
        :type simulation_method: str
        """
        super().__init__("aer_simulator")

        if not noise_model or all(
            value == [] for value in noise_model.to_dict().values()
        ):
            self._noise_model = None
        else:
            self._noise_model = noise_model
            characterisation = _process_model(noise_model, self._backend_info.gate_set)
            averaged_errors = get_avg_characterisation(characterisation)

            arch = characterisation["Architecture"]
            self._backend_info.architecture = arch
            self._backend_info.all_node_gate_errors = characterisation["NodeErrors"]
            self._backend_info.all_edge_gate_errors = characterisation["EdgeErrors"]
            self._backend_info.all_readout_errors = characterisation["ReadoutErrors"]

            self._backend_info.averaged_node_gate_errors = averaged_errors[
                "node_errors"
            ]
            self._backend_info.averaged_edge_gate_errors = averaged_errors[
                "edge_errors"
            ]
            self._backend_info.averaged_readout_errors = averaged_errors[
                "readout_errors"
            ]

            characterisation_keys = [
                "GenericOneQubitQErrors",
                "GenericTwoQubitQErrors",
            ]
            # filter entries to keep
            characterisation = {
                k: v for k, v in characterisation.items() if k in characterisation_keys
            }
            self._backend_info.misc["characterisation"] = characterisation

        self._memory = True

        self._backend.set_options(method=simulation_method)

    @property
    def required_predicates(self) -> List[Predicate]:
        pred_list = [
            NoSymbolsPredicate(),
            GateSetPredicate(
                self._backend_info.gate_set.union(
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
        arch = self._backend_info.architecture
        if arch.coupling:
            # architecture is non-trivial
            pred_list.append(ConnectivityPredicate(arch))
        return pred_list

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passlist = [DecomposeBoxes()]
        if optimisation_level == 0:
            passlist.append(self.rebase_pass())
        elif optimisation_level == 1:
            passlist.append(SynthesiseTket())
        else:
            passlist.append(FullPeepholeOptimise())
        arch = self._backend_info.architecture
        if arch.coupling and self._backend_info.get_misc("characterisation"):
            # architecture is non-trivial
            passlist.append(
                CXMappingPass(
                    arch,
                    NoiseAwarePlacement(
                        arch,
                        self._backend_info.averaged_node_gate_errors,
                        self._backend_info.averaged_edge_gate_errors,
                        self._backend_info.averaged_readout_errors,
                    ),
                    directed_cx=True,
                    delay_measures=False,
                )
            )
            passlist.append(NaivePlacementPass(arch))
            if optimisation_level == 0:
                passlist.append(self.rebase_pass())
            elif optimisation_level == 1:
                passlist.append(SynthesiseTket())
            else:
                passlist.extend([CliffordSimp(False), SynthesiseTket()])
        return SequencePass(passlist)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: `seed`.
        """
        # discard result but useful to validate n_shots
        Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
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
    _expectation_allows_nonhermitian = False

    def __init__(self) -> None:
        """Backend for running simulations on the Qiskit Aer Statevector simulator."""
        super().__init__("aer_simulator_statevector")


class AerUnitaryBackend(_AerStateBaseBackend):
    _supports_unitary = True

    def __init__(self) -> None:
        """Backend for running simulations on the Qiskit Aer Unitary simulator."""
        super().__init__("aer_simulator_unitary")


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

    link_errors: dict = defaultdict(dict)
    node_errors: dict = defaultdict(dict)
    readout_errors: dict = {}

    generic_single_qerrors_dict: dict = defaultdict(list)
    generic_2q_qerrors_dict: dict = defaultdict(list)

    # remember which qubits have explicit link errors
    link_errors_qubits: set = set()

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
        gate_fid = error["probabilities"][0]
        if len(qubits) == 1:
            [q] = qubits
            optype = _gate_str_2_optype[name]
            if error["type"] == "qerror":
                node_errors[q].update({optype: 1 - gate_fid})
                generic_single_qerrors_dict[q].append(
                    [error["instructions"], error["probabilities"]]
                )
            elif error["type"] == "roerror":
                readout_errors[q] = error["probabilities"]
            else:
                raise RuntimeWarning("Error type not 'qerror' or 'roerror'.")
        elif len(qubits) == 2:
            # note that if multiple multi-qubit errors are added to the CX gate,
            #  the resulting noise channel is composed and reflected in probabilities
            [q0, q1] = qubits
            optype = _gate_str_2_optype[name]
            link_errors[(q0, q1)].update({optype: 1 - gate_fid})
            link_errors_qubits.add(q0)
            link_errors_qubits.add(q1)
            # to simulate a worse reverse direction square the fidelity
            link_errors[(q1, q0)].update({optype: 1 - gate_fid**2})
            generic_2q_qerrors_dict[(q0, q1)].append(
                [error["instructions"], error["probabilities"]]
            )
            coupling_map.append(qubits)

    # free qubits (ie qubits with no link errors) have full connectivity
    free_qubits = set(node_errors).union(set(readout_errors)) - link_errors_qubits

    for q in free_qubits:
        for lq in link_errors_qubits:
            coupling_map.append([q, lq])
            coupling_map.append([lq, q])

    for pair in itertools.permutations(free_qubits, 2):
        coupling_map.append(pair)

    # map type (k1 -> k2) -> v[k1] -> v[k2]
    K1 = TypeVar("K1")
    K2 = TypeVar("K2")
    V = TypeVar("V")
    convert_keys_t = Callable[[Callable[[K1], K2], Dict[K1, V]], Dict[K2, V]]
    # convert qubits to architecture Nodes
    convert_keys: convert_keys_t = lambda f, d: {f(k): v for k, v in d.items()}
    node_errors = convert_keys(lambda q: Node(q), node_errors)
    link_errors = convert_keys(lambda p: (Node(p[0]), Node(p[1])), link_errors)
    readout_errors = convert_keys(lambda q: Node(q), readout_errors)

    characterisation: Dict[str, Any] = {}
    characterisation["NodeErrors"] = node_errors
    characterisation["EdgeErrors"] = link_errors
    characterisation["ReadoutErrors"] = readout_errors
    characterisation["GenericOneQubitQErrors"] = [
        [k, v] for k, v in generic_single_qerrors_dict.items()
    ]
    characterisation["GenericTwoQubitQErrors"] = [
        [list(k), v] for k, v in generic_2q_qerrors_dict.items()
    ]
    characterisation["Architecture"] = Architecture(coupling_map)

    return characterisation


def _sparse_to_zx_tup(
    pauli: QubitPauliString, n_qubits: int
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.zeros(n_qubits, dtype=np.bool8)
    z = np.zeros(n_qubits, dtype=np.bool8)
    for q, p in pauli.map.items():
        i = _default_q_index(q)
        z[i] = p in (Pauli.Z, Pauli.Y)
        x[i] = p in (Pauli.X, Pauli.Y)
    return (z, x)


def _qubitpauliop_to_sparsepauliop(
    operator: QubitPauliOperator, n_qubits: int
) -> SparsePauliOp:
    n_ops = len(operator._dict)
    table_array = np.zeros((n_ops, 2 * n_qubits), dtype=np.bool8)
    coeffs = np.zeros(n_ops, dtype=np.float64)

    for i, (term, coeff) in enumerate(operator._dict.items()):
        coeffs[i] = coeff
        z, x = _sparse_to_zx_tup(term, n_qubits)
        table_array[i, :n_qubits] = x
        table_array[i, n_qubits:] = z

    return SparsePauliOp(PauliTable(table_array), coeffs)
