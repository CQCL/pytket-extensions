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

"""Methods to allow conversion between pyzx and tket data types
"""

from typing import Dict, Tuple
from fractions import Fraction
import pyzx as zx  # type: ignore
from pyzx.circuit import Circuit as pyzxCircuit  # type: ignore
from pyzx.routing.architecture import Architecture as PyzxArc  # type: ignore
from pyzx.graph.graph import Graph as pyzxgraph  # type: ignore
from pytket.circuit import OpType, Circuit, Op, Qubit, UnitID  # type: ignore
from pytket.architecture import Architecture  # type: ignore

_tk_to_pyzx_gates = {
    OpType.Rz: zx.gates.ZPhase,
    OpType.Rx: zx.gates.XPhase,
    OpType.X: zx.gates.NOT,
    OpType.Z: zx.gates.Z,
    OpType.S: zx.gates.S,  # gate.adjoint == False
    OpType.Sdg: zx.gates.S,  # gate.adjoint == True
    OpType.T: zx.gates.T,  # gate.adjoint == False
    OpType.Tdg: zx.gates.T,  # gate.adjoint == True
    OpType.CX: zx.gates.CNOT,
    OpType.CZ: zx.gates.CZ,
    OpType.H: zx.gates.HAD,
    OpType.SWAP: zx.gates.SWAP,
}

_pyzx_to_tk_gates: Dict = dict(
    ((item[1], item[0]) for item in _tk_to_pyzx_gates.items())
)

_parameterised_gates = {OpType.Rz, OpType.Rx}


def tk_to_pyzx(tkcircuit: Circuit, denominator_limit: int = 1000000) -> pyzxCircuit:
    """
    Convert a tket :py:class:`Circuit` to a
    :py:class:`pyzx.Circuit`.

    :param tkcircuit: A circuit to be converted
    :param denominator_limit: The limit for denominator size when converting
        floats to fractions. Smaller limits allow for correct representation of simple
        fractions with non-exact floating-point representations, while larger limits
        allow for more precise angles.

    :return: The converted circuit
    """
    if not tkcircuit.is_simple:
        raise Exception("Cannot convert a non-simple tket Circuit to PyZX")
    c = pyzxCircuit(tkcircuit.n_qubits)
    if tkcircuit.name:
        c.name = tkcircuit.name
    for command in tkcircuit:
        op = command.op
        if op.type not in _tk_to_pyzx_gates:
            raise Exception(
                "Cannot convert tket gate: "
                + str(op)
                + ", as the gate type is unrecognised."
            )
        gate_class = _tk_to_pyzx_gates[op.type]
        adjoint = op.type == OpType.Sdg or op.type == OpType.Tdg
        qbs = [q.index[0] for q in command.args]
        gate = 0  # assignment
        n_params = len(op.params)
        if n_params == 1:
            phase = op.params[0]
            if type(phase) != float:
                raise Exception(
                    "Cannot convert tket gate: "
                    + str(op)
                    + ", as it contains symbolic parameters."
                )
            phase = Fraction(phase).limit_denominator(denominator_limit)
            gate = gate_class(*qbs, phase=phase)
        elif n_params > 1:
            raise Exception(
                "Cannot convert tket gate: "
                + str(op)
                + ", as it contains multiple parameters."
            )
        elif adjoint:
            gate = gate_class(*qbs, adjoint=True)
        else:
            gate = gate_class(*qbs)
        c.add_gate(gate)
    return c


def pyzx_to_tk(pyzx_circ: pyzxCircuit) -> Circuit:
    """
    Convert a :py:class:`pyzx.Circuit` to a tket :py:class:`Circuit` .
    All PyZX basic gate operations are currently supported by pytket. Run
    `pyzx_circuit_name.to_basic_gates()` before conversion.

    :param pyzx_circ: A circuit to be converted

    :return: The converted circuit
    """
    c = Circuit(pyzx_circ.qubits, name=pyzx_circ.name)
    for g in pyzx_circ.gates:
        if not type(g) in _pyzx_to_tk_gates:
            raise Exception(
                "Cannot parse PyZX gate of type " + g.name + "into tket Circuit"
            )
        op_type = _pyzx_to_tk_gates[type(g)]
        if hasattr(g, "control"):
            qbs = [getattr(g, "control"), getattr(g, "target")]
        else:
            qbs = [getattr(g, "target")]

        if op_type == OpType.Sdg and not getattr(g, "adjoint"):
            op_type = OpType.S
        elif op_type == OpType.Tdg and not getattr(g, "adjoint"):
            op_type = OpType.T

        if hasattr(g, "printphase") and op_type in _parameterised_gates:
            op = Op.create(op_type, g.phase)
        else:
            op = Op.create(op_type)

        c.add_gate(Op=op, args=qbs)
    return c


def tk_to_pyzx_arc(pytket_arc: Architecture) -> PyzxArc:
    """
    Convert a pytket :py:class:`Architecture` to a pyzx
    :py:class:`pyzx.routing.architecture.Architecture` .
    The conversion will remove all the node names and will
    keep them only integer named in the order they are given
    in the node set of the pytket.

    :param pytket_arc: A Architecture to be converted

    :return: The converted pyzx Architecture
    """

    arcgraph = pyzxgraph()
    vertices = arcgraph.add_vertices(len(pytket_arc.nodes))
    arc_dict = dict()

    for i, x in enumerate(pytket_arc.nodes):
        arc_dict[x] = i

    edges = [
        (vertices[arc_dict[v1]], vertices[arc_dict[v2]])
        for v1, v2 in pytket_arc.coupling
    ]

    arcgraph.add_edges(edges)

    pyzx_arc = PyzxArc("pyzx_arc", coupling_graph=arcgraph)

    return pyzx_arc


def pyzx_to_tk_arc(pyzx_arc: PyzxArc) -> Architecture:
    """
    Convert a pyzx :py:class:`pyzx.routing.architecture.Architecture`
    to a pytket :py:class:`Architecture` .

    :param pytket_arc: A Architecture to be converted

    :return: The converted pyzx Architecture
    """

    pytket_arc = Architecture([tuple(s) for s in pyzx_arc.graph.edges()])

    return pytket_arc


def tk_to_pyzx_placed_circ(
    pytket_circ: Circuit, pytket_arc: Architecture, denominator_limit: int = 1000000
) -> Tuple[PyzxArc, pyzxCircuit, Dict[UnitID, UnitID]]:
    """
    Convert a (placed) tket :py:class:`Circuit` with
    a given :py:class:`Architecture` to a
    :py:class:`pyzx.Circuit` and the
    :py:class:`pyzx.routing.architecture.Architecture`
    and a map to give the information for converting
    this back to pytket using `pyzx_to_tk_placed_circ`.

    :param pytket_circ: A circuit to be converted
    :param pytket_arc: Corresponding Architecture
    :param denominator_limit: The limit for denominator size when converting
        floats to fractions. Smaller limits allow for correct representation of simple
        fractions with non-exact floating-point representations, while larger limits
        allow for more precise angles.

    :return: Tuple containing generated :py:class:`pyzx.Circuit` ,
        :py:class:`pyzx.routing.architecture.Architecture` and
        map for conversion

    """

    simple_circ = pytket_circ.copy()

    q_map = simple_circ.flatten_registers()

    inv_q_map = {v: k for k, v in q_map.items()}

    arcgraph = pyzxgraph()
    vertices = arcgraph.add_vertices(len(pytket_arc.nodes))
    arc_dict = dict()
    qubit_dict = dict()

    for i, x in enumerate(pytket_arc.nodes):
        q = Qubit(i)
        qubit_dict[q] = i

    for i, x in enumerate(pytket_arc.nodes):
        arc_dict[x] = qubit_dict[q_map[x]]

    edges = [
        (vertices[arc_dict[v1]], vertices[arc_dict[v2]])
        for v1, v2 in pytket_arc.coupling
    ]

    arcgraph.add_edges(edges)

    pyzx_arc = PyzxArc("pyzx_arc", coupling_graph=arcgraph)

    pyzx_circ = tk_to_pyzx(simple_circ)

    return (pyzx_arc, pyzx_circ, inv_q_map)


def pyzx_to_tk_placed_circ(
    pyzx_circ: pyzxCircuit, q_map: Dict[UnitID, UnitID]
) -> Circuit:
    """
    Convert a :py:class:`pyzx.Circuit` and a placment map
    to a placed tket :py:class:`Circuit` .
    All PyZX basic gate operations are currently supported by pytket. Run
    `pyzx_circuit_name.to_basic_gates()` before conversion.

    :param pyzx_circ: A circuit to be converted
    :param q_map: placment map

    :return: The converted pytket circuit
    """

    pytket_circ = pyzx_to_tk(pyzx_circ)

    pytket_circ.rename_units(q_map)

    return pytket_circ
