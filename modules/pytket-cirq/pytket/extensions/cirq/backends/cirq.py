# Copyright 2020-2022 Cambridge Quantum Computing
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

from abc import abstractmethod
from typing import Sequence, cast, Optional, List, Union
from uuid import uuid4
from cirq.sim import (
    CliffordSimulator,
    CliffordSimulatorStepResult,
    DensityMatrixSimulator,
    Simulator,
    StateVectorTrialResult,
)

from cirq import ops
from cirq.value import RANDOM_STATE_OR_SEED_LIKE
from cirq.devices import NOISE_MODEL_LIKE
from cirq.circuits import Circuit as CirqCircuit

from pytket.circuit import Circuit, OpType, Qubit  # type: ignore
from pytket.transform import Transform  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    auto_rebase_pass,
    SequencePass,
    RebaseCustom,
    SquashCustom,
    SynthesiseTket,
    DecomposeBoxes,
    FlattenRegisters,
    RemoveRedundancies,
    FullPeepholeOptimise,
)
from pytket._tket.circuit._library import _TK1_to_PhasedXRz, _CX  # type: ignore
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    Predicate,
)
from pytket.backends import Backend, ResultHandle, CircuitStatus, StatusEnum
from pytket.backends.backendresult import BackendResult
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.utils.results import KwargTypes
from pytket.extensions.cirq.cirq_convert import tk_to_cirq  # type: ignore
from pytket.utils.outcomearray import OutcomeArray
from .cirq_utils import _get_default_uids


class _CirqBaseBackend(Backend):
    """Common base class for all Cirq simulator backends"""

    def __init__(self) -> None:
        super().__init__()
        self._pass_0 = SequencePass(
            [FlattenRegisters(), DecomposeBoxes(), self.rebase_pass()]
        )
        self._pass_1 = SequencePass(
            [
                FlattenRegisters(),
                DecomposeBoxes(),
                SynthesiseTket(),
                self.rebase_pass(),
                RemoveRedundancies(),
                _cirq_squash,
            ]
        )
        self._pass_2 = SequencePass(
            [
                FlattenRegisters(),
                DecomposeBoxes(),
                FullPeepholeOptimise(),
                self.rebase_pass(),
                RemoveRedundancies(),
                _cirq_squash,
            ]
        )
        self._gate_set_predicate = _regular_gate_set_predicate

    def rebase_pass(self) -> BasePass:
        return auto_rebase_pass({OpType.CZ, OpType.PhasedX, OpType.Rz})

    @property
    def required_predicates(self) -> List[Predicate]:
        preds = [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            self._gate_set_predicate,
        ]
        return preds

    @property
    def backend_info(self) -> Optional[BackendInfo]:
        return None

    @property
    def characterisation(self) -> Optional[dict]:
        return None

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        if optimisation_level == 0:
            return self._pass_0
        elif optimisation_level == 1:
            return self._pass_1
        else:
            return self._pass_2

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, int)

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        self._check_handle_type(handle)
        return CircuitStatus(StatusEnum.COMPLETED)


class _CirqSampleBackend(_CirqBaseBackend):
    """Common base class for all Cirq simulator sampling backends"""

    def __init__(self) -> None:
        super().__init__()
        self._supports_shots: bool = True
        self._supports_counts: bool = True
        self._simulator: Union[
            Simulator, DensityMatrixSimulator, CliffordSimulator
        ] = Simulator()

    def _run_circuit(self, circuit: Circuit, n_shots: int) -> BackendResult:
        cirq_circ = tk_to_cirq(circuit)
        bit_to_qubit_map = {b: q for q, b in circuit.qubit_to_bit_map.items()}
        if not cirq_circ.has_measurements():  # type: ignore
            return self.empty_result(circuit, n_shots=n_shots)
        else:
            run = self._simulator.run(cirq_circ, repetitions=n_shots)
            run_dict = run.data.to_dict()
            c_bits = [
                bit
                for key in run_dict.keys()
                for bit in bit_to_qubit_map.keys()
                if str(bit) == key
            ]
            individual_readouts = [
                list(readout.values()) for readout in run_dict.values()
            ]
            shots = OutcomeArray.from_readouts(
                [list(r) for r in zip(*individual_readouts)]
            )
            return BackendResult(shots=shots, c_bits=c_bits)

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
            optional=False,
        )

        if valid_check:
            self._check_all_circuits(circuits)

        handle_list = []
        for i, (circuit, n_shots) in enumerate(zip(circuits, n_shots_list)):
            handle = ResultHandle(str(uuid4()), i)
            handle_list.append(handle)
            backres = self._run_circuit(circuit, n_shots=n_shots)
            self._cache[handle] = {"result": backres}

        return handle_list


class CirqStateSampleBackend(_CirqSampleBackend):
    """Backend for Cirq statevector simulator sampling."""

    def __init__(self, seed: RANDOM_STATE_OR_SEED_LIKE = None) -> None:
        super().__init__()
        self._simulator = Simulator(seed=seed)


class CirqDensityMatrixSampleBackend(_CirqSampleBackend):
    """Backend for Cirq density matrix simulator sampling."""

    def __init__(
        self,
        seed: RANDOM_STATE_OR_SEED_LIKE = None,
        noise_model: NOISE_MODEL_LIKE = None,
    ) -> None:
        super().__init__()
        self._supports_density_matrix = True
        self._simulator = DensityMatrixSimulator(seed=seed, noise=noise_model)


class CirqCliffordSampleBackend(_CirqSampleBackend):
    """Backend for Cirq Clifford simulator sampling."""

    def __init__(self, seed: RANDOM_STATE_OR_SEED_LIKE = None) -> None:
        super().__init__()
        self._simulator = CliffordSimulator(seed=seed)
        self._pass_0 = SequencePass(
            [FlattenRegisters(), DecomposeBoxes(), self.rebase_pass()]
        )
        self._pass_1 = SequencePass(
            [
                FlattenRegisters(),
                DecomposeBoxes(),
                SynthesiseTket(),
                RemoveRedundancies(),
                self.rebase_pass(),
            ]
        )
        self._pass_2 = SequencePass(
            [
                FlattenRegisters(),
                DecomposeBoxes(),
                FullPeepholeOptimise(),
                RemoveRedundancies(),
                self.rebase_pass(),
            ]
        )

        self._gate_set_predicate = _clifford_gate_set_predicate
        self._clifford_only = True

    def rebase_pass(self) -> BasePass:
        return _partial_clifford_rebase


class _CirqSimBackend(_CirqBaseBackend):
    """Common base class for all Cirq simulator matrix/state returning backends."""

    @abstractmethod
    def package_result(
        self, circuit: CirqCircuit, q_bits: Sequence[Qubit]
    ) -> BackendResult:
        """

        :param circuit: The circuit to simulate.
        :type circuit: CirqCircuit
        :param q_bits: ordered pytket Qubit
        :type q_bits: Sequence[Qubit]
        :return: result of simulation
        :rtype: BackendResult
        """
        ...

    def _run_circuit(self, circuit: Circuit) -> BackendResult:
        cirq_circ = tk_to_cirq(circuit, copy_all_qubits=True)
        _, q_bits = _get_default_uids(cirq_circ, circuit)
        return self.package_result(cirq_circ, q_bits)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Optional[Union[int, Sequence[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:

        if n_shots is not None:
            raise ValueError("`n_shots` argument is invalid for _CirqSimBackend")

        if valid_check:
            self._check_all_circuits(circuits)

        handle_list = []
        for i, circuit in enumerate(circuits):
            handle = ResultHandle(str(uuid4()), i)
            handle_list.append(handle)
            backres = self._run_circuit(circuit)
            self._cache[handle] = {"result": backres}

        return handle_list

    @abstractmethod
    def package_results(
        self, circuit: CirqCircuit, q_bits: Sequence[Qubit]
    ) -> List[BackendResult]:
        """

        :param circuit: The circuit to simulate.
        :type circuit: CirqCircuit
        :param q_bits: ordered pytket Qubit
        :type q_bits: Sequence[Qubit]
        :return: sequence of moments from simulator
        :rtype: List[BackendResult]
        """
        ...

    def _run_circuit_moments(self, circuit: Circuit) -> List[BackendResult]:
        cirq_circ = tk_to_cirq(circuit, copy_all_qubits=True)
        _, q_bits = _get_default_uids(cirq_circ, circuit)
        return self.package_results(cirq_circ, q_bits)

    def process_circuit_moments(
        self,
        circuit: Circuit,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> ResultHandle:
        """
        Submit a single circuit to the backend for running. See
        :py:meth:`_CirqSimBackend.process_circuits_moments`.
        """

        return self.process_circuits_moments(
            [circuit], valid_check=valid_check, **kwargs
        )[0]

    def process_circuits_moments(
        self,
        circuits: Sequence[Circuit],
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:

        """
        Submit circuits to the backend for running. The results will be stored
        in the backend's result cache to be retrieved by the corresponding
        get_<data> method. The get_<data> method will return List[BackendResult]
        corresponding to each moment.

        :param circuits: Circuits to process on the backend.
        :type circuits: Iterable[Circuit]
        :param valid_check: Explicitly check that all circuits satisfy all required
            predicates to run on the backend. Defaults to True
        :type valid_check: bool, optional
        :return: Handles to results for each input circuit, as an interable in
            the same order as the circuits.
        :rtype: List[ResultHandle]
        """

        if valid_check:
            self._check_all_circuits(circuits)

        handle_list = []
        for i, circuit in enumerate(circuits):
            handle = ResultHandle(str(uuid4()), i)
            handle_list.append(handle)
            backres = self._run_circuit_moments(circuit)
            self._cache[handle] = {"result": backres}  # type: ignore

        return handle_list


class CirqStateSimBackend(_CirqSimBackend):
    """Backend for Cirq statevector simulator state return."""

    def __init__(self, seed: RANDOM_STATE_OR_SEED_LIKE = None) -> None:
        super().__init__()
        self._simulator = Simulator(seed=seed)
        self._supports_state = True

    def package_result(
        self, circuit: CirqCircuit, q_bits: Sequence[Qubit]
    ) -> BackendResult:
        run = self._simulator.simulate(
            circuit,
            qubit_order=ops.QubitOrder.as_qubit_order(ops.QubitOrder.DEFAULT).order_for(
                circuit.all_qubits()
            ),
        )

        return BackendResult(state=run.final_state_vector, q_bits=q_bits)

    def package_results(
        self, circuit: CirqCircuit, q_bits: Sequence[Qubit]
    ) -> List[BackendResult]:
        moments = self._simulator.simulate_moment_steps(
            circuit,
            qubit_order=ops.QubitOrder.as_qubit_order(ops.QubitOrder.DEFAULT).order_for(
                circuit.all_qubits()
            ),
        )
        all_backres = [
            BackendResult(
                state=run.state_vector(copy=True),
                q_bits=q_bits,
            )
            for run in moments
        ]
        return all_backres


class CirqDensityMatrixSimBackend(_CirqSimBackend):
    """Backend for Cirq density matrix simulator density_matrix return."""

    def __init__(
        self,
        seed: RANDOM_STATE_OR_SEED_LIKE = None,
        noise_model: NOISE_MODEL_LIKE = None,
    ) -> None:
        super().__init__()
        self._supports_density_matrix = True
        self._simulator = DensityMatrixSimulator(seed=seed, noise=noise_model)

    def package_result(
        self, circuit: CirqCircuit, q_bits: Sequence[Qubit]
    ) -> BackendResult:
        run = self._simulator.simulate(circuit)
        return BackendResult(density_matrix=run.final_density_matrix, q_bits=q_bits)

    def package_results(
        self, circuit: CirqCircuit, q_bits: Sequence[Qubit]
    ) -> List[BackendResult]:
        moments = self._simulator.simulate_moment_steps(circuit)
        all_backres = [
            BackendResult(
                density_matrix=run.density_matrix(copy=True),  # type: ignore
                q_bits=q_bits,
            )
            for run in moments
        ]

        return all_backres


class CirqCliffordSimBackend(_CirqSimBackend):
    """Backend for Cirq Clifford simulator state return."""

    def __init__(self, seed: RANDOM_STATE_OR_SEED_LIKE = None) -> None:
        super().__init__()
        self._pass_0 = SequencePass(
            [FlattenRegisters(), DecomposeBoxes(), self.rebase_pass()]
        )
        self._pass_1 = SequencePass(
            [
                FlattenRegisters(),
                DecomposeBoxes(),
                SynthesiseTket(),
                RemoveRedundancies(),
                self.rebase_pass(),
            ]
        )
        self._pass_2 = SequencePass(
            [
                FlattenRegisters(),
                DecomposeBoxes(),
                FullPeepholeOptimise(),
                RemoveRedundancies(),
                self.rebase_pass(),
            ]
        )

        self._gate_set_predicate = _clifford_gate_set_predicate
        self._supports_state = True
        self._simulator = CliffordSimulator(seed=seed)

    def rebase_pass(self) -> BasePass:
        return _partial_clifford_rebase

    def package_result(
        self, circuit: CirqCircuit, q_bits: Sequence[Qubit]
    ) -> BackendResult:
        run = cast(
            StateVectorTrialResult,
            self._simulator.simulate(
                circuit,
                qubit_order=ops.QubitOrder.as_qubit_order(
                    ops.QubitOrder.DEFAULT
                ).order_for(circuit.all_qubits()),
            ),
        )
        return BackendResult(state=run.final_state_vector, q_bits=q_bits)

    def package_results(
        self, circuit: CirqCircuit, q_bits: Sequence[Qubit]
    ) -> List[BackendResult]:
        moments = self._simulator.simulate_moment_steps(
            circuit,
            qubit_order=ops.QubitOrder.as_qubit_order(ops.QubitOrder.DEFAULT).order_for(
                circuit.all_qubits()
            ),
        )
        all_backres = [
            BackendResult(
                state=cast(CliffordSimulatorStepResult, run).state.state_vector(),  # type: ignore
                q_bits=q_bits,
            )
            for run in moments
        ]
        return all_backres


_cirq_squash = SquashCustom(
    {OpType.PhasedX, OpType.Rz, OpType.Rx, OpType.Ry}, _TK1_to_PhasedXRz
)

_regular_gate_set_predicate = GateSetPredicate(
    {
        OpType.Measure,
        OpType.CX,
        OpType.CZ,
        OpType.PhasedX,
        OpType.Rz,
        OpType.Rx,
        OpType.Ry,
        OpType.H,
        OpType.S,
        OpType.SWAP,
        OpType.T,
        OpType.X,
        OpType.Y,
        OpType.Z,
        OpType.noop,
        OpType.CU1,
        OpType.CSWAP,
        OpType.ISWAP,
        OpType.ISWAPMax,
        OpType.FSim,
        OpType.Sycamore,
        OpType.ZZPhase,
        OpType.XXPhase,
        OpType.YYPhase,
        OpType.PhasedISWAP,
    }
)

_clifford_gate_set_predicate = GateSetPredicate(
    {
        OpType.Measure,
        OpType.CX,
        OpType.CZ,
        OpType.SWAP,
        OpType.Z,
        OpType.X,
        OpType.Y,
        OpType.V,
        OpType.Vdg,
        OpType.S,
        OpType.Sdg,
        OpType.H,
    }
)


def _tk1_to_phasedxrz_clifford(a: float, b: float, c: float) -> Circuit:
    circ = _TK1_to_PhasedXRz(a, b, c)
    Transform.RebaseToCliffordSingles().apply(circ)
    return circ


_partial_clifford_rebase = RebaseCustom(
    {
        OpType.PhasedX,
        OpType.Rz,
        OpType.Rx,
        OpType.Ry,
        OpType.Z,
        OpType.X,
        OpType.Y,
        OpType.V,
        OpType.Vdg,
        OpType.S,
        OpType.Sdg,
        OpType.H,
        OpType.CX,
        OpType.CZ,
    },
    _CX(),
    _tk1_to_phasedxrz_clifford,
)
