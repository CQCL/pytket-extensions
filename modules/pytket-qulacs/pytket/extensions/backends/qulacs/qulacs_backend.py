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

"""Methods to allow tket circuits to be ran on the Qulacs simulator
"""

from typing import TYPE_CHECKING, Iterable, List, Optional
from logging import warning
from uuid import uuid4
import numpy as np  # type: ignore
from qulacs import Observable, QuantumState  # type: ignore
from qulacs.observable import create_observable_from_openfermion_text  # type: ignore
from pytket.backends import (
    Backend,
    CircuitNotRunError,
    CircuitStatus,
    ResultHandle,
    StatusEnum,
)
from pytket.backends.backend import KwargTypes
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    SynthesiseIBM,
    SequencePass,
    DecomposeBoxes,
    RebaseIBM,
    FullPeepholeOptimise,
    FlattenRegisters,
)
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    DefaultRegisterPredicate,
    Predicate,
)
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.outcomearray import OutcomeArray
from pytket.extensions.qulacs.qulacs_convert import tk_to_qulacs

_GPU_ENABLED = True
try:
    from qulacs import QuantumStateGpu
except ImportError:
    _GPU_ENABLED = False


if TYPE_CHECKING:
    from pytket.device import Device  # type: ignore


class QulacsBackend(Backend):
    """
    Backend for running simulations on the Qulacs simulator
    """

    _supports_shots = True
    _supports_counts = True
    _supports_state = True
    _supports_expectation = True
    _persistent_handles = False

    def __init__(self) -> None:
        super().__init__()
        self._sim = QuantumState

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    @property
    def device(self) -> Optional["Device"]:
        return None

    @property
    def required_predicates(self) -> List[Predicate]:
        return [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoMidMeasurePredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(
                {
                    OpType.X,
                    OpType.Y,
                    OpType.Z,
                    OpType.H,
                    OpType.S,
                    OpType.Sdg,
                    OpType.T,
                    OpType.Tdg,
                    OpType.Rx,
                    OpType.Ry,
                    OpType.Rz,
                    OpType.CX,
                    OpType.CZ,
                    OpType.SWAP,
                    OpType.U1,
                    OpType.U2,
                    OpType.U3,
                    OpType.Measure,
                    OpType.Barrier,
                }
            ),
            DefaultRegisterPredicate(),
        ]

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        if optimisation_level == 0:
            return SequencePass([DecomposeBoxes(), FlattenRegisters(), RebaseIBM()])
        elif optimisation_level == 1:
            return SequencePass([DecomposeBoxes(), FlattenRegisters(), SynthesiseIBM()])
        else:
            return SequencePass(
                [DecomposeBoxes(), FlattenRegisters(), FullPeepholeOptimise()]
            )

    def process_circuits(
        self,
        circuits: Iterable[Circuit],
        n_shots: Optional[int] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        circuit_list = list(circuits)
        if valid_check:
            self._check_all_circuits(circuit_list, nomeasure_warn=(n_shots is not None))

        handle_list = []
        for circuit in circuit_list:
            qulacs_state = self._sim(circuit.n_qubits)
            qulacs_state.set_zero_state()
            qulacs_circ = tk_to_qulacs(circuit)
            qulacs_circ.update_quantum_state(qulacs_state)
            state = qulacs_state.get_vector()
            qubits = sorted(circuit.qubits, reverse=True)
            shots = None
            bits = None
            if n_shots:

                bits2index = list(
                    (com.bits[0], qubits.index(com.qubits[0]))
                    for com in circuit
                    if com.op.type == OpType.Measure
                )
                if len(bits2index) == 0:
                    bits = circuit.bits
                    shots = OutcomeArray.from_ints([0] * n_shots, len(bits))
                else:
                    bits, choose_indices = zip(*bits2index)

                    samples = qulacs_state.sampling(n_shots)
                    shots = OutcomeArray.from_ints(samples, circuit.n_qubits)
                    shots = shots.choose_indices(choose_indices)
            try:
                phase = float(circuit.phase)
                coeff = np.exp(phase * np.pi * 1j)
                state *= coeff
            except TypeError:
                warning(
                    "Global phase is dependent on a symbolic parameter, so cannot "
                    "adjust for phase"
                )
            implicit_perm = circuit.implicit_qubit_permutation()
            qubits = [implicit_perm[qb] for qb in qubits]
            handle = ResultHandle(str(uuid4()))
            self._cache[handle] = {
                "result": BackendResult(
                    state=state, shots=shots, c_bits=bits, q_bits=qubits
                )
            }
            handle_list.append(handle)
            del qulacs_state
            del qulacs_circ
        return handle_list

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        if handle in self._cache:
            return CircuitStatus(StatusEnum.COMPLETED)
        raise CircuitNotRunError(handle)

    def get_operator_expectation_value(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        n_shots: int = 0,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> complex:
        if valid_check:
            self._check_all_circuits([state_circuit], nomeasure_warn=False)
        observable = create_observable_from_openfermion_text(
            str(operator.to_OpenFermion())
        )
        expectation_value = self._expectation_value(state_circuit, observable)
        del observable
        return expectation_value.real

    def _expectation_value(self, circuit: Circuit, operator: Observable) -> complex:
        state = self._sim(circuit.n_qubits)
        state.set_zero_state()
        ql_circ = tk_to_qulacs(circuit)
        ql_circ.update_quantum_state(state)
        expectation_value = operator.get_expectation_value(state)
        del state
        del ql_circ
        return complex(expectation_value)


if _GPU_ENABLED:

    class QulacsGPUBackend(QulacsBackend):
        """
        Backend for running simulations on the Qulacs GPU simulator
        """

        def __init__(self) -> None:
            super().__init__()
            self._sim = QuantumStateGpu
