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

"""Methods to allow conversion from tket to Q#
"""

from typing import List, Tuple
from math import pi
from pytket.circuit import Circuit, OpType, Op  # type: ignore
from pytket.pauli import Pauli  # type: ignore

qs_pauli = {Pauli.I: "PauliI", Pauli.X: "PauliX", Pauli.Y: "PauliY", Pauli.Z: "PauliZ"}


def cmd_body(op: Op, qbs: List[int]) -> str:
    optype = op.type

    if optype == OpType.CCX:
        return "CCNOT(q[{}], q[{}], q[{}])".format(*qbs)
    elif optype == OpType.CX:
        return "CNOT(q[{}], q[{}])".format(*qbs)
    elif optype == OpType.PauliExpBox:
        paulis = op.get_paulis()
        theta = (-2 / pi) * op.get_phase()
        return "Exp([{}], {}, [{}])".format(
            ", ".join([qs_pauli[p] for p in paulis]),
            theta,
            ", ".join(["q[{}]".format(i) for i in qbs]),
        )
    elif optype == OpType.H:
        return "H(q[{}])".format(*qbs)
    elif optype == OpType.noop:
        pass
    elif optype == OpType.Rx:
        return "Rx({}, q[{}])".format(pi * op.params[0], qbs[0])
    elif optype == OpType.Ry:
        return "Ry({}, q[{}])".format(pi * op.params[0], qbs[0])
    elif optype == OpType.Rz:
        return "Rz({}, q[{}])".format(pi * op.params[0], qbs[0])
    elif optype == OpType.S:
        return "S(q[{}])".format(*qbs)
    elif optype == OpType.SWAP:
        return "SWAP(q[{}], q[{}])".format(*qbs)
    elif optype == OpType.T:
        return "T(q[{}])".format(*qbs)
    elif optype == OpType.X:
        return "X(q[{}])".format(*qbs)
    elif optype == OpType.Y:
        return "Y(q[{}])".format(*qbs)
    elif optype == OpType.Z:
        return "Z(q[{}])".format(*qbs)

    elif optype == OpType.CnX:
        return (
            "ApplyMultiControlledC("
            + ", ".join(
                [
                    "ApplyToFirstTwoQubitsCA(CNOT, _)",
                    "CCNOTop(CCNOT)",
                    "[{}]".format(", ".join(["q[{}]".format(i) for i in qbs[:-1]])),
                    "[{}]".format("q[{}]".format(qbs[-1])),
                ]
            )
            + ")"
        )
    else:
        raise RuntimeError("Unsupported operation {}".format(optype))


def main_body(c: Circuit) -> Tuple[List[str], List[int]]:
    lines = []
    measured_bits: List[int] = []
    for cmd in c:
        qbs = [qb.index[0] for qb in cmd.qubits]
        if cmd.op.type == OpType.Measure:
            cb = cmd.bits[0].index[0]
            measured_bits.append(cb)
            lines.append("        set r w/= {} <- M(q[{}]);".format(cb, qbs[0]))
        else:
            lines.append("        " + cmd_body(cmd.op, qbs) + ";")

    return lines, measured_bits


def operation_body(c: Circuit, sim: bool = True) -> List[str]:
    lines = []
    n_q = c.n_qubits
    n_c = len(c.bits)
    if n_c > 0:
        lines.append("    mutable r = [" + ", ".join(["Zero"] * n_c) + "];")
    else:
        lines.append("    mutable r = new Result[0];")

    lines.append("    use q = Qubit[{}] {{".format(n_q))
    # devices don't support reset yet
    if sim:
        lines.append("        ResetAll(q);")
    main_lines, meas_bits = main_body(c)
    lines.extend(main_lines)
    meas_bits = sorted(meas_bits)
    all_bits = list(range(n_c))
    if meas_bits != all_bits:
        extra_meas_lines = [
            "        set r w/= {} <- Zero;".format(cb)
            for cb in all_bits
            if cb not in meas_bits
        ]
        lines.extend(extra_meas_lines)
    if sim:
        lines.append("        ResetAll(q);")
    lines.append("        return r;")
    lines.append("    }")
    return lines


def tk_to_qsharp(tkcirc: Circuit, sim: bool = True) -> str:
    """Convert a tket :py:class:`Circuit` to a Q# program.

    :param tkcirc: Circuit to be converted

    :return: Converted circuit
    """
    if tkcirc.is_symbolic():
        raise RuntimeError("Cannot convert symbolic circuit to Q#")

    lines = []
    lines.append("open Microsoft.Quantum.Intrinsic;")
    lines.append("open Microsoft.Quantum.Measurement;")
    lines.append("open Microsoft.Quantum.Canon;")
    lines.append("operation TketCircuit() : Result[] {")
    lines.extend(operation_body(tkcirc, sim))
    lines.append("}")
    return "\n".join(lines)
