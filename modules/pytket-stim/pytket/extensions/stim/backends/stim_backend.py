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
import numpy as np
import stim  # type: ignore

_gate = {
    OpType.noop: "I",
    OpType.X: "X",
    OpType.Y: "Y",
    OpType.Z: "Z",
    OpType.H: "H",
    OpType.S: "S",
    OpType.SX: "SQRT_X",
    OpType.SXdg: "SQRT_X_DAG",
    OpType.CX: "CX",
    OpType.CY: "CY",
    OpType.CZ: "CZ",
    OpType.ISWAPMax: "ISWAP",
    OpType.SWAP: "SWAP",
    OpType.Measure: "M",
    OpType.Reset: "R",
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
    # Convert Clifford tk1(a, b, c) to a circuit composed of H and S gates
    n_a, n_b, n_c = _int_double(a), _int_double(b), _int_double(c)
    circ = Circuit(1)
    for _ in range(n_c):
        circ.S(0)
    for _ in range(n_b):
        circ.H(0).S(0).H(0)
    for _ in range(n_a):
        circ.S(0)
    circ.add_phase(-0.25 * (n_a + n_b + n_c))
    return circ


def _process_one_circuit(circ: Circuit, n_shots: int) -> BackendResult:
    qubits = circ.qubits
    bits = circ.bits
    c = stim.Circuit()
    readout_bits = []
    for cmd in circ.get_commands():
        optype = cmd.op.type
        args = cmd.args
        if optype == OpType.Measure:
            qb, cb = args
            c.append_operation("M", [qubits.index(qb)])
            readout_bits.append(cb)
        else:
            qbs = [qubits.index(arg) for arg in args]
            c.append_operation(_gate[optype], qbs)
        if len(set(readout_bits)) != len(readout_bits):
            raise ValueError("Measurement overwritten")
    sampler = c.compile_sampler()
    batch = sampler.sample(n_shots)
    # batch[k,:] has the measurements in the order they were added to the stim circuit.
    # We want them to be returned in bit order.
    return BackendResult(
        shots=OutcomeArray.from_readouts(
            [[batch[k, readout_bits.index(cb)] for cb in bits] for k in range(n_shots)]
        )
    )


class StimBackend(Backend):
    """
    Backend for simulating Clifford circuits using Stim
    """

    _supports_shots = True
    _supports_counts = True

    @property
    def required_predicates(self) -> List[Predicate]:
        return [
            DefaultRegisterPredicate(),
            GateSetPredicate(set(_gate.keys())),
            NoClassicalControlPredicate(),
        ]

    def rebase_pass(self) -> BasePass:
        return RebaseCustom({OpType.CX, OpType.H, OpType.S}, Circuit(), _tk1_to_cliff)

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
