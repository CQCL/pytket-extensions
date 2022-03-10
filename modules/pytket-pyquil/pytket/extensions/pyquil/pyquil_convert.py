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

"""Methods to allow conversion between pyQuil and tket data types
"""

from collections import defaultdict
from logging import warning
import math
from typing import (
    Any,
    Callable,
    Union,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
    overload,
)
from typing_extensions import Literal

from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.external.rpcq import GateInfo, MeasureInfo
from pyquil.quilatom import (
    Qubit as Qubit_,
    Expression,
    MemoryReference,
    quil_sin,
    quil_cos,
    Add as Add_,
    Sub,
    Mul as Mul_,
    Div,
    Pow as Pow_,
    Function as Function_,
)
from pyquil.quilbase import Declare, Gate, Halt, Measurement, Pragma
from sympy import pi, Expr, Symbol, sin, cos, Number, Add, Mul, Pow

from pytket.circuit import Circuit, Node, OpType, Qubit, Bit  # type: ignore
from pytket.architecture import Architecture  # type: ignore

_known_quil_gate = {
    "X": OpType.X,
    "Y": OpType.Y,
    "Z": OpType.Z,
    "H": OpType.H,
    "S": OpType.S,
    "T": OpType.T,
    "RX": OpType.Rx,
    "RY": OpType.Ry,
    "RZ": OpType.Rz,
    "CZ": OpType.CZ,
    "CNOT": OpType.CX,
    "CCNOT": OpType.CCX,
    "CPHASE": OpType.CU1,
    "PHASE": OpType.U1,
    "SWAP": OpType.SWAP,
    "XY": OpType.ISWAP,
}


_known_quil_gate_rev = {v: k for k, v in _known_quil_gate.items()}


def param_to_pyquil(p: Union[float, Expr]) -> Union[float, Expression]:
    ppi = p * pi
    if len(ppi.free_symbols) == 0:
        return float(ppi.evalf())
    else:

        def to_pyquil(e: Expr) -> Union[float, Expression]:
            if isinstance(e, Number):
                return float(e)
            elif isinstance(e, Symbol):
                return MemoryReference(str(e))
            elif isinstance(e, sin):
                return quil_sin(to_pyquil(e))
            elif isinstance(e, cos):
                return quil_cos(to_pyquil(e))
            elif isinstance(e, Add):
                args = [to_pyquil(a) for a in e.args]
                acc = args[0]
                for a in args[1:]:
                    acc += a
                return acc
            elif isinstance(e, Mul):
                args = [to_pyquil(a) for a in e.args]
                acc = args[0]
                for a in args[1:]:
                    acc *= a
                return acc
            elif isinstance(e, Pow):
                args = Pow_(to_pyquil(e.base), to_pyquil(e.exp))  # type: ignore
            elif e == pi:
                return math.pi
            else:
                raise NotImplementedError(
                    "Sympy expression could not be converted to a Quil expression: "
                    + str(e)
                )

        return to_pyquil(ppi)


def param_from_pyquil(p: Union[float, Expression]) -> Expr:
    def to_sympy(e: Any) -> Union[float, int, Expr, Symbol]:
        if isinstance(e, (float, int)):
            return e
        elif isinstance(e, MemoryReference):
            return Symbol(e.name)  # type: ignore
        elif isinstance(e, Function_):
            if e.name == "SIN":
                return sin(to_sympy(e.expression))  # type: ignore
            elif e.name == "COS":
                return cos(to_sympy(e.expression))  # type: ignore
            else:
                raise NotImplementedError(
                    "Quil expression function "
                    + e.name
                    + " cannot be converted to a sympy expression"
                )
        elif isinstance(e, Add_):
            return to_sympy(e.op1) + to_sympy(e.op2)  # type: ignore
        elif isinstance(e, Sub):
            return to_sympy(e.op1) - to_sympy(e.op2)  # type: ignore
        elif isinstance(e, Mul_):
            return to_sympy(e.op1) * to_sympy(e.op2)  # type: ignore
        elif isinstance(e, Div):
            return to_sympy(e.op1) / to_sympy(e.op2)  # type: ignore
        elif isinstance(e, Pow_):
            return to_sympy(e.op1) ** to_sympy(e.op2)  # type: ignore
        else:
            raise NotImplementedError(
                "Quil expression could not be converted to a sympy expression: "
                + str(e)
            )

    return to_sympy(p) / pi  # type: ignore


def pyquil_to_tk(prog: Program) -> Circuit:
    """
    Convert a :py:class:`pyquil.Program` to a tket :py:class:`Circuit` .
    Note that not all pyQuil operations are currently supported by pytket.

    :param prog: A circuit to be converted

    :return: The converted circuit
    """
    tkc = Circuit()
    qmap = {}
    for q in prog.get_qubits():
        uid = Qubit("q", q)
        tkc.add_qubit(uid)
        qmap.update({q: uid})
    cregmap: Dict = {}
    for i in prog.instructions:
        if isinstance(i, Gate):
            try:
                optype = _known_quil_gate[i.name]
            except KeyError as error:
                raise NotImplementedError(
                    "Operation not supported by tket: " + str(i)
                ) from error
            qubits = [qmap[q.index] for q in i.qubits]
            params = [param_from_pyquil(p) for p in i.params]  # type: ignore
            tkc.add_gate(optype, params, qubits)
        elif isinstance(i, Measurement):
            qubit = qmap[i.qubit.index]
            reg = cregmap[i.classical_reg.name]  # type: ignore
            bit = reg[i.classical_reg.offset]  # type: ignore
            tkc.Measure(qubit, bit)
        elif isinstance(i, Declare):
            if i.memory_type == "BIT":
                new_reg = tkc.add_c_register(i.name, i.memory_size)
                cregmap.update({i.name: new_reg})
            elif i.memory_type == "REAL":
                continue
            else:
                raise NotImplementedError(
                    "Cannot handle memory of type " + i.memory_type
                )
        elif isinstance(i, Pragma):
            continue
        elif isinstance(i, Halt):
            return tkc
        else:
            raise NotImplementedError("PyQuil instruction is not a gate: " + str(i))
    return tkc


@overload
def tk_to_pyquil(
    tkcirc: Circuit, active_reset: bool = False, return_used_bits: Literal[False] = ...
) -> Program:
    ...


@overload
def tk_to_pyquil(
    tkcirc: Circuit, active_reset: bool = False, *, return_used_bits: Literal[True]
) -> Tuple[Program, List[Bit]]:
    ...


@overload
def tk_to_pyquil(
    tkcirc: Circuit, active_reset: bool, return_used_bits: Literal[True]
) -> Tuple[Program, List[Bit]]:
    ...


def tk_to_pyquil(
    tkcirc: Circuit, active_reset: bool = False, return_used_bits: bool = False
) -> Union[Program, Tuple[Program, List[Bit]]]:
    """
       Convert a tket :py:class:`Circuit` to a :py:class:`pyquil.Program` .

    :param tkcirc: A circuit to be converted

    :return: The converted circuit
    """
    p = Program()
    qregs = set()
    for qb in tkcirc.qubits:
        if len(qb.index) != 1:
            raise NotImplementedError("PyQuil registers must use a single index")
        qregs.add(qb.reg_name)
    if len(qregs) > 1:
        raise NotImplementedError(
            "Cannot convert circuit with multiple quantum registers to pyQuil"
        )
    creg_sizes: Dict = {}
    for b in tkcirc.bits:
        if len(b.index) != 1:
            raise NotImplementedError("PyQuil registers must use a single index")
        if (b.reg_name not in creg_sizes) or (b.index[0] >= creg_sizes[b.reg_name]):
            creg_sizes.update({b.reg_name: b.index[0] + 1})
    cregmap = {}
    for reg_name, size in creg_sizes.items():
        name = reg_name
        if name == "c":
            name = "ro"
        quil_reg = p.declare(name, "BIT", size)
        cregmap.update({reg_name: quil_reg})
    for sym in tkcirc.free_symbols():
        p.declare(str(sym), "REAL")
    if active_reset:
        p.reset()
    measures = []
    measured_qubits: List[Qubit] = []
    used_bits: List[Bit] = []
    for command in tkcirc:
        op = command.op
        optype = op.type
        if optype == OpType.Measure:
            qb = Qubit_(command.args[0].index[0])
            if qb in measured_qubits:
                raise NotImplementedError(
                    "Cannot apply gate on qubit " + qb.__repr__() + " after measurement"
                )
            bit = command.args[1]
            b = cregmap[bit.reg_name][bit.index[0]]
            measures.append(Measurement(qb, b))
            measured_qubits.append(qb)
            used_bits.append(bit)
            continue
        elif optype == OpType.Barrier:
            continue  # pyQuil cannot handle barriers
        qubits = [Qubit_(qb.index[0]) for qb in command.args]
        for qb in qubits:
            if qb in measured_qubits:
                raise NotImplementedError(
                    "Cannot apply gate on qubit " + qb.__repr__() + " after measurement"
                )
        try:
            gatetype = _known_quil_gate_rev[optype]
        except KeyError as error:
            raise NotImplementedError(
                "Cannot convert tket Op to pyQuil gate: " + op.get_name()
            ) from error
        params = [param_to_pyquil(p) for p in op.params]
        g = Gate(gatetype, params, qubits)
        p += g
    for m in measures:
        p += m
    if return_used_bits:
        return p, used_bits
    return p


def process_characterisation(qc: QuantumComputer) -> dict:
    """Convert a :py:class:`pyquil.api.QuantumComputer` to a dictionary containing
    Rigetti device Characteristics

    :param qc: A quantum computer to be converted
    :type qc: QuantumComputer
    :return: A dictionary containing Rigetti device characteristics
    """
    isa = qc.quantum_processor.to_compiler_isa()
    coupling_map = [[int(i) for i in e.ids] for e in isa.edges.values()]

    str_to_gate_1qb = {
        "RX": {
            "PI": OpType.X,
            "PIHALF": OpType.V,
            "-PIHALF": OpType.Vdg,
            "-PI": OpType.X,
            "ANY": OpType.Rx,
        },
        "RZ": {
            "ANY": OpType.Rz,
        },
    }
    str_to_gate_2qb = {"CZ": OpType.CZ, "XY": OpType.ISWAP}

    link_errors: Dict[Tuple[Node, Node], Dict[OpType, float]] = defaultdict(dict)
    node_errors: Dict[Node, Dict[OpType, float]] = defaultdict(dict)
    readout_errors: dict = {}
    # T1s and T2s are currently left empty
    t1_times_dict: dict = {}
    t2_times_dict: dict = {}

    for q in isa.qubits.values():
        node = Node(q.id)
        for g in q.gates:
            if g.fidelity is None:
                g.fidelity = 1.0
            if isinstance(g, GateInfo) and g.operator in str_to_gate_1qb:
                angle = _get_angle_type(g.parameters[0])
                if angle is not None:
                    try:
                        optype = str_to_gate_1qb[g.operator][angle]
                    except KeyError:
                        warning(
                            f"Ignoring unrecognised angle {g.parameters[0]} "
                            f"for gate {g.operator}. This may mean that some "
                            "hardware-supported gates won't be used."
                        )
                        continue
                    if node in node_errors and optype in node_errors[node]:
                        if abs(1.0 - g.fidelity - node_errors[node][optype]) > 1e-7:
                            # fidelities for Rx(PI) and Rx(-PI) are given, hopefully
                            # they are always identical
                            warning(
                                f"Found two differing fidelities for {optype} on node "
                                f"{node}, using error = {node_errors[node][optype]}"
                            )
                    else:
                        node_errors[node].update({optype: 1.0 - g.fidelity})
            elif isinstance(g, MeasureInfo) and g.operator == "MEASURE":
                # for some reason, there are typically two MEASURE entries,
                # one with target="_", and one with target=Node
                # in all pyquil code I have seen, both have the same value
                if node in readout_errors:
                    if abs(1.0 - g.fidelity - readout_errors[node]) > 1e-7:
                        warning(
                            f"Found two differing readout fidelities for node {node},"
                            f" using RO error = {readout_errors[node]}"
                        )
                else:
                    readout_errors[node] = 1.0 - g.fidelity
            elif g.operator == "I":
                continue
            else:
                warning(f"Ignoring fidelity for unknown operator {g.operator}")

    for e in isa.edges.values():
        n1, n2 = Node(e.ids[0]), Node(e.ids[1])
        for g in e.gates:
            if g.fidelity is None:
                g.fidelity = 1.0
            if g.operator in str_to_gate_2qb:
                optype = str_to_gate_2qb[g.operator]
                link_errors[(n1, n2)].update({optype: 1.0 - g.fidelity})
            else:
                warning(f"Ignoring fidelity for unknown operator {g.operator}")

    arc = Architecture(coupling_map)

    characterisation = dict()
    characterisation["NodeErrors"] = node_errors
    characterisation["EdgeErrors"] = link_errors
    characterisation["Architecture"] = arc
    characterisation["t1times"] = t1_times_dict
    characterisation["t2times"] = t2_times_dict

    return characterisation


def _get_angle_type(angle: Union[float, str]) -> Optional[str]:
    if angle == "theta":
        return "ANY"
    else:
        angles = {pi: "PI", pi / 2: "PIHALF", 0: None, -pi / 2: "-PIHALF", -pi: "-PI"}
        if not isinstance(angle, str):
            for val, code in angles.items():
                if abs(angle - val) < 1e-7:
                    return code
        warning(
            f"Ignoring unrecognised angle {angle}. This may mean that some "
            "hardware-supported gates won't be used."
        )
        return None


def get_avg_characterisation(
    characterisation: Dict[str, Any]
) -> Dict[str, Dict[Node, float]]:
    """
    Convert gate-specific characterisation into readout, one- and two-qubit errors

    Used to convert a typical output from `process_characterisation` into an input
    noise characterisation for NoiseAwarePlacement
    """

    K = TypeVar("K")
    V1 = TypeVar("V1")
    V2 = TypeVar("V2")
    map_values_t = Callable[[Callable[[V1], V2], Dict[K, V1]], Dict[K, V2]]
    map_values: map_values_t = lambda f, d: {k: f(v) for k, v in d.items()}

    node_errors = cast(Dict[Node, Dict[OpType, float]], characterisation["NodeErrors"])
    link_errors = cast(
        Dict[Tuple[Node, Node], Dict[OpType, float]], characterisation["EdgeErrors"]
    )

    avg: Callable[[Dict[Any, float]], float] = lambda xs: sum(xs.values()) / len(xs)
    avg_node_errors = map_values(avg, node_errors)
    avg_link_errors = map_values(avg, link_errors)

    return {
        "node_errors": avg_node_errors,
        "link_errors": avg_link_errors,
    }
