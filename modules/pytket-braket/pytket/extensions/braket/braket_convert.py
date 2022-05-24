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

""" Conversion from tket to braket
"""

from typing import (
    cast,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    TYPE_CHECKING,
)
from pytket.circuit import Circuit, OpType  # type: ignore
from braket.circuits import Circuit as BK_Circuit  # type: ignore
from numpy import pi

if TYPE_CHECKING:
    from pytket.circuit import Node  # type: ignore


def tk_to_braket(
    tkcirc: Circuit, allqbs: Optional[Iterable[int]] = None
) -> Tuple[BK_Circuit, Dict[int, int]]:
    """
    Convert a tket :py:class:`Circuit` to a braket circuit.

    If `allqbs` is provided then `tkcirc` must be a circuit having a single
    one-dimensional register of qubits.

    If `allqbs` is not provided then it is taken to be the qubit set of `tkcirc`.

    The resulting circuit will begin with an identity gate on all qubits in `allqbs`.
    This is to work around a quirk in braket where circuits whose qubit set contains
    gaps are rejected.

    :param tkcirc: circuit to be converted
    :param allqbs: superset of indices of tkcirc qubits, if those have been mapped to a
        braked QPU device

    :returns: circuit converted to braket, and dictionary of measurements (from bit
        indices to qubit indices)
    """
    bkcirc = BK_Circuit()
    mapped_qubits = allqbs is not None
    if not mapped_qubits:
        allqbs = range(tkcirc.n_qubits)
    assert allqbs is not None
    for qb in allqbs:
        bkcirc.i(qb)
    measures = {}
    # Add commands
    for cmd in tkcirc.get_commands():
        qbs = [
            qb.index[0] if mapped_qubits else tkcirc.qubits.index(qb)
            for qb in cmd.qubits
        ]
        cbs = [tkcirc.bits.index(cb) for cb in cmd.bits]
        op = cmd.op
        optype = op.type
        params = op.params
        if optype == OpType.CCX:
            bkcirc.ccnot(*qbs)
        elif optype == OpType.CX:
            bkcirc.cnot(*qbs)
        elif optype == OpType.CU1:
            bkcirc.cphaseshift(*qbs, params[0] * pi)
        elif optype == OpType.CSWAP:
            bkcirc.cswap(*qbs)
        elif optype == OpType.CY:
            bkcirc.cy(*qbs)
        elif optype == OpType.CZ:
            bkcirc.cz(*qbs)
        elif optype == OpType.H:
            bkcirc.h(*qbs)
        elif optype == OpType.noop:
            pass
        elif optype == OpType.ISWAPMax:
            bkcirc.iswap(*qbs)
        elif optype == OpType.U1:
            bkcirc.phaseshift(*qbs, params[0] * pi)
        elif optype == OpType.Rx:
            bkcirc.rx(*qbs, params[0] * pi)
        elif optype == OpType.Ry:
            bkcirc.ry(*qbs, params[0] * pi)
        elif optype == OpType.Rz:
            bkcirc.rz(*qbs, params[0] * pi)
        elif optype == OpType.S:
            bkcirc.s(*qbs)
        elif optype == OpType.Sdg:
            bkcirc.si(*qbs)
        elif optype == OpType.SWAP:
            bkcirc.swap(*qbs)
        elif optype == OpType.T:
            bkcirc.t(*qbs)
        elif optype == OpType.Tdg:
            bkcirc.ti(*qbs)
        # V amd Vdg differ by a pi/4 phase from braket according to the get_matrix
        # methods. However, braket circuits do not seem to be phase-aware.
        elif optype == OpType.V:
            bkcirc.v(*qbs)
        elif optype == OpType.Vdg:
            bkcirc.vi(*qbs)
        elif optype == OpType.X:
            bkcirc.x(*qbs)
        elif optype == OpType.XXPhase:
            bkcirc.xx(*qbs, params[0] * pi)
        elif optype == OpType.ISWAP:
            bkcirc.xy(*qbs, params[0] * pi)
        elif optype == OpType.Y:
            bkcirc.y(*qbs)
        elif optype == OpType.YYPhase:
            bkcirc.yy(*qbs, params[0] * pi)
        elif optype == OpType.Z:
            bkcirc.z(*qbs)
        elif optype == OpType.ZZPhase:
            bkcirc.zz(*qbs, params[0] * pi)
        elif optype == OpType.Measure:
            # Not wanted by braket, but must be tracked for final conversion of results.
            measures[cbs[0]] = qbs[0]
        else:
            raise NotImplementedError(f"Cannot convert {op.get_name()} to braket")
    return (bkcirc, measures)


def braket_to_tk(bkcirc: BK_Circuit) -> Circuit:
    """
    Convert a braket circuit to a tket :py:class:`Circuit`

    :param bkcirc: circuit to be converted

    :returns: circuit converted to tket
    """
    n_qbs = len(bkcirc.qubits)
    tkcirc = Circuit(n_qbs)
    for instr in bkcirc.instructions:
        op = instr.operator
        qbs = [bkcirc.qubits.index(qb) for qb in instr.target]
        opname = op.name
        if opname == "CCNot":
            tkcirc.add_gate(OpType.CCX, qbs)
        elif opname == "CNot":
            tkcirc.add_gate(OpType.CX, qbs)
        elif opname == "CPhaseShift":
            tkcirc.add_gate(OpType.CU1, op.angle / pi, qbs)
        elif opname == "CSwap":
            tkcirc.add_gate(OpType.CSWAP, qbs)
        elif opname == "CY":
            tkcirc.add_gate(OpType.CY, qbs)
        elif opname == "CZ":
            tkcirc.add_gate(OpType.CZ, qbs)
        elif opname == "H":
            tkcirc.add_gate(OpType.H, qbs)
        elif opname == "I":
            pass
        elif opname == "ISwap":
            tkcirc.add_gate(OpType.ISWAPMax, qbs)
        elif opname == "PhaseShift":
            tkcirc.add_gate(OpType.U1, op.angle / pi, qbs)
        elif opname == "Rx":
            tkcirc.add_gate(OpType.Rx, op.angle / pi, qbs)
        elif opname == "Ry":
            tkcirc.add_gate(OpType.Ry, op.angle / pi, qbs)
        elif opname == "Rz":
            tkcirc.add_gate(OpType.Rz, op.angle / pi, qbs)
        elif opname == "S":
            tkcirc.add_gate(OpType.S, qbs)
        elif opname == "Si":
            tkcirc.add_gate(OpType.Sdg, qbs)
        elif opname == "Swap":
            tkcirc.add_gate(OpType.SWAP, qbs)
        elif opname == "T":
            tkcirc.add_gate(OpType.T, qbs)
        elif opname == "Ti":
            tkcirc.add_gate(OpType.Tdg, qbs)
        elif opname == "V":
            tkcirc.add_gate(OpType.V, qbs)
            tkcirc.add_phase(0.25)
        elif opname == "Vi":
            tkcirc.add_gate(OpType.Vdg, qbs)
            tkcirc.add_phase(-0.25)
        elif opname == "X":
            tkcirc.add_gate(OpType.X, qbs)
        elif opname == "XX":
            tkcirc.add_gate(OpType.XXPhase, op.angle / pi, qbs)
        elif opname == "XY":
            tkcirc.add_gate(OpType.ISWAP, op.angle / pi, qbs)
        elif opname == "Y":
            tkcirc.add_gate(OpType.Y, qbs)
        elif opname == "YY":
            tkcirc.add_gate(OpType.YYPhase, op.angle / pi, qbs)
        elif opname == "Z":
            tkcirc.add_gate(OpType.Z, qbs)
        elif opname == "ZZ":
            tkcirc.add_gate(OpType.ZZPhase, op.angle / pi, qbs)
        else:
            # The following don't have direct equivalents:
            # - CPhaseShift00, CPhaseShift01, CPhaseShift10: diagonal unitaries with 1s
            # on the diagonal except for a phase e^{ia} in the (0,0), (1,1) or (2,2)
            # position respectively.
            # - PSwap: unitary with 1s at (0,0) and (3,3), a phase e^{ia} at (1,2) and
            # (2,1), and zeros elsewhere.
            # They could be decomposed into pytket gates, but it would be better to add
            # the gate types to tket.
            # The "Unitary" type could be represented as a box in the 1q and 2q cases,
            # but not in general.
            raise NotImplementedError(f"Cannot convert {opname} to tket")
    return tkcirc


def get_avg_characterisation(
    characterisation: Dict[str, Any]
) -> Dict[str, Dict["Node", float]]:
    """
    Convert gate-specific characterisation into readout, one- and two-qubit errors

    Used to convert the stored full `characterisation` into an input
    noise characterisation for NoiseAwarePlacement
    """

    K = TypeVar("K")
    V1 = TypeVar("V1")
    V2 = TypeVar("V2")
    map_values_t = Callable[[Callable[[V1], V2], Dict[K, V1]], Dict[K, V2]]
    map_values: map_values_t = lambda f, d: {k: f(v) for k, v in d.items()}

    node_errors = cast(
        Dict["Node", Dict[OpType, float]], characterisation["NodeErrors"]
    )
    link_errors = cast(
        Dict[Tuple["Node", "Node"], Dict[OpType, float]], characterisation["EdgeErrors"]
    )
    readout_errors = cast(
        Dict["Node", List[List[float]]], characterisation["ReadoutErrors"]
    )

    avg: Callable[[Dict[Any, float]], float] = lambda xs: sum(xs.values()) / len(xs)
    avg_mat: Callable[[List[List[float]]], float] = (
        lambda xs: (xs[0][1] + xs[1][0]) / 2.0
    )
    avg_readout_errors = map_values(avg_mat, readout_errors)
    avg_node_errors = map_values(avg, node_errors)
    avg_link_errors = map_values(avg, link_errors)

    return {
        "node_errors": avg_node_errors,
        "link_errors": avg_link_errors,
        "readout_errors": avg_readout_errors,
    }
