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

from typing import TYPE_CHECKING, List, Optional, Sequence, Union, cast
from logging import warning
from uuid import uuid4
import numpy as np
from qulacs import Observable, QuantumState  # type: ignore
from pytket.backends import (
    Backend,
    CircuitNotRunError,
    CircuitStatus,
    ResultHandle,
    StatusEnum,
)
from pytket.backends.backend import KwargTypes
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.extensions.qulacs._metadata import __extension_version__
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
from pytket.circuit import Pauli, Qubit  # type: ignore
from pytket.pauli import QubitPauliString  # type: ignore
from pytket.routing import Architecture  # type: ignore
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
    _expectation_allows_nonhermitian = False
    _persistent_handles = False
    _GATE_SET = {
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

    def __init__(self) -> None:
        super().__init__()
        self._backend_info = BackendInfo(
            type(self).__name__,
            None,
            __extension_version__,
            Architecture([]),
            self._GATE_SET,
        )

        self._sim = QuantumState

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    @property
    def backend_info(self) -> Optional["Device"]:
        return None

    @property
    def required_predicates(self) -> List[Predicate]:
        return [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoMidMeasurePredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(self._GATE_SET),
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
        circuits: Sequence[Circuit],
        n_shots: Optional[Union[int, Sequence[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        circuits = list(circuits)
        if hasattr(n_shots, "__iter__"):
            n_shots_list = list(cast(Sequence[Optional[int]], n_shots))
            if len(n_shots_list) != len(circuits):
                raise ValueError("The length of n_shots and circuits must match")
        else:
            # convert n_shots to a list
            n_shots_list = [cast(Optional[int], n_shots)] * len(circuits)

        if valid_check:
            self._check_all_circuits(circuits, nomeasure_warn=False)

        handle_list = []
        for circuit, n_shots_circ in zip(circuits, n_shots_list):
            qulacs_state = self._sim(circuit.n_qubits)
            qulacs_state.set_zero_state()
            qulacs_circ = tk_to_qulacs(circuit)
            qulacs_circ.update_quantum_state(qulacs_state)
            state = qulacs_state.get_vector()
            qubits = sorted(circuit.qubits, reverse=True)
            shots = None
            bits = None
            if n_shots_circ is not None:
                bits2index = list(
                    (com.bits[0], qubits.index(com.qubits[0]))
                    for com in circuit
                    if com.op.type == OpType.Measure
                )
                if len(bits2index) == 0:
                    bits = circuit.bits
                    shots = OutcomeArray.from_ints([0] * n_shots_circ, len(bits))
                else:
                    bits, choose_indices = zip(*bits2index)

                    samples = qulacs_state.sampling(n_shots_circ)
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

        observable = Observable(state_circuit.n_qubits)
        for (qps, coeff) in operator._dict.items():
            _items = []
            if qps != QubitPauliString():
                for qubit, pauli in qps.to_dict().items():
                    if pauli == Pauli.X:
                        _items.append('X')
                    elif pauli == Pauli.Y:
                        _items.append('Y')
                    elif pauli == Pauli.Z:
                        _items.append('Z')
                    _items.append(str(qubit.index[0]))

            qulacs_qps = " ".join(_items)
            qulacs_coeff = complex(coeff.evalf())
            observable.add_operator(qulacs_coeff, qulacs_qps)

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
            self._backend_info.name = type(self).__name__
            self._sim = QuantumStateGpu
