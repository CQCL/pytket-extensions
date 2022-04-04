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

from typing import cast, List, Optional, Sequence, Union
from uuid import uuid4
import numpy as np
from pysimplex import Simplex  # type: ignore
from pytket.backends import (
    Backend,
    CircuitNotRunError,
    CircuitStatus,
    ResultHandle,
    StatusEnum,
)
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    DecomposeBoxes,
    FlattenRegisters,
    RebaseCustom,
    RemoveRedundancies,
    SequencePass,
)
from pytket.predicates import (  # type: ignore
    DefaultRegisterPredicate,
    GateSetPredicate,
    NoClassicalControlPredicate,
    Predicate,
)
from pytket.utils.outcomearray import OutcomeArray
from pytket.utils.results import KwargTypes

_gateset = {
    OpType.X,
    OpType.Y,
    OpType.Z,
    OpType.H,
    OpType.S,
    OpType.Sdg,
    OpType.CX,
    OpType.CZ,
    OpType.Reset,
    OpType.Measure,
}


def _int_double(x: float) -> int:
    # return (2x) mod 8 if x is close to a half-integer, otherwise error
    y = 2 * x
    n = int(np.round(y))  # type: ignore
    if np.isclose(y, n):
        return n % 8
    else:
        raise ValueError("Non-Clifford angle encountered")


def _tk1_to_cliff(a: float, b: float, c: float) -> Circuit:
    # Convert Clifford tk1(a, b, c) to a circuit composed of Clifford gates
    n_a, n_b, n_c = _int_double(a), _int_double(b), _int_double(c)
    circ = Circuit(1)
    if n_c == 1:
        circ.S(0)
    elif n_c == 2:
        circ.Z(0)
    elif n_c == 3:
        circ.Sdg(0)
    if n_b == 1:
        circ.H(0).S(0).H(0)
    elif n_b == 2:
        circ.H(0).Z(0).H(0)
    elif n_b == 3:
        circ.H(0).Sdg(0).H(0)
    if n_a == 1:
        circ.S(0)
    elif n_a == 2:
        circ.Z(0)
    elif n_a == 3:
        circ.Sdg(0)
    circ.add_phase(-0.25 * (n_a + n_b + n_c))
    return circ


def _process_one_circuit(circ: Circuit, n_shots: int) -> BackendResult:
    n_qubits = circ.n_qubits
    qubits = circ.qubits
    bits = circ.bits
    n_bits = len(circ.bits)
    cmds = circ.get_commands()
    readouts = []
    for _ in range(n_shots):
        S = Simplex(n_qubits)
        measurements = [0] * n_bits
        for cmd in cmds:
            optype = cmd.op.type
            args = cmd.args
            if optype == OpType.Measure:
                qarg, carg = args
                qb = qubits.index(qarg)
                cb = bits.index(carg)
                measurements[cb] = S.MeasZ(qb)
            else:
                qbs = [qubits.index(arg) for arg in args]
                if optype == OpType.X:
                    S.X(*qbs)
                elif optype == OpType.Y:
                    S.Y(*qbs)
                elif optype == OpType.Z:
                    S.Z(*qbs)
                elif optype == OpType.H:
                    S.H(*qbs)
                elif optype == OpType.S:
                    S.S(*qbs)
                elif optype == OpType.Sdg:
                    S.Sdg(*qbs)
                elif optype == OpType.CX:
                    S.CX(*qbs)
                elif optype == OpType.CZ:
                    S.CZ(*qbs)
                elif optype == OpType.Reset:
                    S.ResetZ(*qbs)
                else:
                    raise ValueError(f"Unexpected operation type {optype}")
        readouts.append(measurements)
    return BackendResult(shots=OutcomeArray.from_readouts(readouts))


class SimplexBackend(Backend):
    """
    Backend for simulating Clifford circuits using pysimplex
    """

    _supports_shots = True
    _supports_counts = True

    @property
    def required_predicates(self) -> List[Predicate]:
        return [
            DefaultRegisterPredicate(),
            GateSetPredicate(_gateset),
            NoClassicalControlPredicate(),
        ]

    def rebase_pass(self) -> BasePass:
        return RebaseCustom(_gateset, Circuit(), _tk1_to_cliff)

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        # No optimization.
        return SequencePass(
            [
                DecomposeBoxes(),
                FlattenRegisters(),
                self.rebase_pass(),
                RemoveRedundancies(),
            ]
        )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Optional[Union[int, Sequence[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        circuits = list(circuits)
        n_shots_list: List[int] = []
        if hasattr(n_shots, "__iter__"):
            n_shots_list = cast(List[int], n_shots)
            if len(n_shots_list) != len(circuits):
                raise ValueError("The length of n_shots and circuits must match")
        else:
            # convert n_shots to a list
            n_shots_list = [cast(int, n_shots)] * len(circuits)

        if valid_check:
            self._check_all_circuits(circuits)

        handle_list = []
        for circuit, n_shots_circ in zip(circuits, n_shots_list):
            handle = ResultHandle(str(uuid4()))
            self._cache[handle] = {
                "result": _process_one_circuit(circuit, n_shots_circ)
            }
            handle_list.append(handle)
        return handle_list

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        if handle in self._cache:
            return CircuitStatus(StatusEnum.COMPLETED)
        raise CircuitNotRunError(handle)
