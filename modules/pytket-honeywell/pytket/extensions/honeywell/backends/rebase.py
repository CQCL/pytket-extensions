from typing import Set, Union
from sympy import Expr
from pytket.circuit import OpType, Circuit  # type: ignore
from pytket.passes import BasePass, RebaseCustom  # type: ignore
from pytket.transform import Transform  # type: ignore


class GateSetError(Exception):
    """Gate set required for rebase not available."""


def _approx_0_mod_2(x: Union[float, Expr], eps: float = 1e-10) -> bool:
    if isinstance(x, Expr) and not x.is_constant():
        return False
    x = float(x)
    x %= 2
    return min(x, 2 - x) < eps


def _tk1_to_PhasedXRz(
    a: Union[float, Expr], b: Union[float, Expr], c: Union[float, Expr]
) -> Circuit:
    circ = Circuit(1)
    if _approx_0_mod_2(b - 1):
        # Angles β ∈ {π, 3π}
        circ.add_gate(OpType.PhasedX, [b, (a - c) / 2], [0])
    elif _approx_0_mod_2(b):
        # Angle β ∈ {0, 2π}
        circ.Rz(a + b + c, 0)
    else:
        circ.Rz(a + c, 0)
        circ.add_gate(OpType.PhasedX, [b, a], [0])

    Transform.RemoveRedundancies().apply(circ)
    return circ


def _quantinuum_rebase(gate_set: Set[OpType]) -> BasePass:
    # CX replacement
    c_cx = Circuit(2)
    c_cx.Rz(1.5, 0).Rx(0.5, 1).Rz(1.5, 1).Rx(1.5, 1)
    c_cx.add_gate(OpType.ZZMax, [0, 1])
    c_cx.Rx(1.5, 1).Rz(1.5, 1)
    c_cx.add_phase(0.75)
    if not {OpType.ZZMax, OpType.PhasedX, OpType.Rz}.issubset(gate_set):
        raise GateSetError(
            "ZZMax, PhasedX, Rz are required in gateset to perform rebase."
        )
    multiops = gate_set.intersection({OpType.ZZMax})
    singlops = gate_set.intersection({OpType.PhasedX, OpType.Rz})
    return RebaseCustom(
        multiops,
        c_cx,
        singlops,
        _tk1_to_PhasedXRz,
    )
