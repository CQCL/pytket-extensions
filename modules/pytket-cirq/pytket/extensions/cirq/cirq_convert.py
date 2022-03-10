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

"""
Methods to allow conversion between Cirq and tket data types, including Circuits and
Devices
"""

from typing import List, Dict, FrozenSet, cast, Any, Union
import cmath
from logging import warning
from cirq.devices import LineQubit, GridQubit
import cirq.ops
import cirq_google
from pytket.circuit import Circuit, OpType, Qubit, Bit, Node  # type: ignore
from pytket.architecture import Architecture  # type: ignore
from sympy import pi, Basic, Symbol

# For translating cirq circuits to tket circuits
cirq_common = cirq.ops.common_gates
cirq_pauli = cirq.ops.pauli_gates

cirq_CH = cirq_common.H.controlled(1)

# map cirq common gates to pytket gates
_cirq2ops_mapping = {
    cirq_common.CNOT: OpType.CX,
    cirq_common.H: OpType.H,
    cirq_common.MeasurementGate: OpType.Measure,
    cirq_common.XPowGate: OpType.Rx,
    cirq_common.YPowGate: OpType.Ry,
    cirq_common.ZPowGate: OpType.Rz,
    cirq_common.XPowGate(exponent=0.5): OpType.V,
    cirq_common.XPowGate(exponent=-0.5): OpType.Vdg,
    cirq_common.S: OpType.S,
    cirq_common.SWAP: OpType.SWAP,
    cirq_common.T: OpType.T,
    cirq_pauli.X: OpType.X,
    cirq_pauli.Y: OpType.Y,
    cirq_pauli.Z: OpType.Z,
    cirq.ops.I: OpType.noop,
    cirq_common.CZPowGate: OpType.CU1,
    cirq_common.CZ: OpType.CZ,
    cirq_CH: OpType.CH,
    cirq.ops.CSwapGate: OpType.CSWAP,
    cirq_common.ISwapPowGate: OpType.ISWAP,
    cirq_common.ISWAP: OpType.ISWAPMax,
    cirq.ops.FSimGate: OpType.FSim,
    cirq_google.SYC: OpType.Sycamore,
    cirq.ops.parity_gates.ZZPowGate: OpType.ZZPhase,
    cirq.ops.parity_gates.XXPowGate: OpType.XXPhase,
    cirq.ops.parity_gates.YYPowGate: OpType.YYPhase,
    cirq.ops.PhasedXPowGate: OpType.PhasedX,
    cirq.ops.PhasedISwapPowGate: OpType.PhasedISWAP,
}
# reverse mapping for convenience
_ops2cirq_mapping: Dict = dict((item[1], item[0]) for item in _cirq2ops_mapping.items())
# spot special rotation gates
_constant_gates = (
    cirq_common.CNOT,
    cirq_common.H,
    cirq_common.S,
    cirq_common.SWAP,
    cirq_common.T,
    cirq_pauli.X,
    cirq_pauli.Y,
    cirq_pauli.Z,
    cirq_common.CZ,
    cirq_CH,
    cirq_common.ISWAP,
    cirq_google.SYC,
    cirq.ops.I,
)
_rotation_types = (
    cirq_common.XPowGate,
    cirq_common.YPowGate,
    cirq_common.ZPowGate,
    cirq_common.CZPowGate,
    cirq_common.ISwapPowGate,
    cirq.ops.parity_gates.ZZPowGate,
    cirq.ops.parity_gates.XXPowGate,
    cirq.ops.parity_gates.YYPowGate,
)


def cirq_to_tk(circuit: cirq.circuits.Circuit) -> Circuit:
    """Converts a Cirq :py:class:`Circuit` to a tket :py:class:`Circuit` object.

    :param circuit: The input Cirq :py:class:`Circuit`

    :raises NotImplementedError: If the input contains a Cirq :py:class:`Circuit`
        operation which is not yet supported by pytket

    :return: The tket :py:class:`Circuit` corresponding to the input circuit
    """
    tkcirc = Circuit()
    qmap = {}
    for qb in circuit.all_qubits():
        if isinstance(qb, LineQubit):
            uid = Qubit("q", qb.x)
        elif isinstance(qb, GridQubit):
            uid = Qubit("g", qb.row, qb.col)
        elif isinstance(qb, cirq.ops.NamedQubit):
            uid = Qubit(qb.name)
        else:
            raise NotImplementedError("Cannot convert qubits of type " + str(type(qb)))
        tkcirc.add_qubit(uid)
        qmap.update({qb: uid})
    for moment in circuit:
        for op in moment.operations:
            if isinstance(op, cirq.ops.GlobalPhaseOperation):
                tkcirc.add_phase(cmath.phase(op.coefficient) / pi)
                continue
            gate = op.gate
            gatetype = type(gate)
            qb_lst = [qmap[q] for q in op.qubits]

            if isinstance(gate, cirq_common.HPowGate) and gate.exponent == 1:
                gate = cirq_common.H
            elif (
                gatetype == cirq_common.CNotPowGate
                and cast(cirq_common.CNotPowGate, gate).exponent == 1
            ):
                gate = cirq_common.CNOT
            elif (
                gatetype == cirq_pauli._PauliX
                and cast(cirq_pauli._PauliX, gate).exponent == 1
            ):
                gate = cirq_pauli.X
            elif (
                gatetype == cirq_pauli._PauliY
                and cast(cirq_pauli._PauliY, gate).exponent == 1
            ):
                gate = cirq_pauli.Y
            elif (
                gatetype == cirq_pauli._PauliZ
                and cast(cirq_pauli._PauliZ, gate).exponent == 1
            ):
                gate = cirq_pauli.Z

            apply_in_parallel = False
            if isinstance(gate, cirq.ops.ParallelGate):
                if gate.num_copies != len(qb_lst):
                    raise NotImplementedError(
                        "ParallelGate parameters defined incorrectly."
                    )
                gate = gate.sub_gate
                gatetype = type(gate)
                apply_in_parallel = True

            if gate in _constant_gates:
                try:
                    optype = _cirq2ops_mapping[gate]
                except KeyError as error:
                    raise NotImplementedError(
                        "Operation not supported by tket: " + str(op.gate)
                    ) from error
                params: List[Union[float, Basic, Symbol]] = []
            elif isinstance(gate, cirq_common.MeasurementGate):
                uid = Bit(gate.key)
                tkcirc.add_bit(uid)
                tkcirc.Measure(*qb_lst, uid)
                continue
            elif isinstance(gate, cirq.ops.PhasedXPowGate):
                optype = OpType.PhasedX
                pe = gate.phase_exponent
                params = [gate.exponent, pe]
            elif isinstance(gate, cirq.ops.FSimGate):
                optype = OpType.FSim
                params = [gate.theta / pi, gate.phi / pi]
            elif isinstance(gate, cirq.ops.PhasedISwapPowGate):
                optype = OpType.PhasedISWAP
                params = [gate.phase_exponent, gate.exponent]
            else:
                try:
                    optype = _cirq2ops_mapping[gatetype]
                    params = [cast(Any, gate).exponent]
                except (KeyError, AttributeError) as error:
                    raise NotImplementedError(
                        "Operation not supported by tket: " + str(op.gate)
                    ) from error
            if apply_in_parallel:
                for qb in qb_lst:
                    tkcirc.add_gate(optype, params, [qb])
            else:
                tkcirc.add_gate(optype, params, qb_lst)
    return tkcirc


def tk_to_cirq(tkcirc: Circuit, copy_all_qubits: bool = False) -> cirq.circuits.Circuit:
    """Converts a tket :py:class:`Circuit` object to a Cirq :py:class:`Circuit`.

    :param tkcirc: The input tket :py:class:`Circuit`

    :return: The Cirq :py:class:`Circuit` corresponding to the input circuit
    """
    if copy_all_qubits:
        tkcirc = tkcirc.copy()
        for q in tkcirc.qubits:
            tkcirc.add_gate(OpType.noop, [q])

    qmap = {}
    line_name = None
    grid_name = None
    # Since Cirq can only support registers of up to 2 dimensions, we explicitly
    # check for 3-dimensional registers whose third dimension is trivial.
    # SquareGrid architectures are of this form.
    indices = [qb.index for qb in tkcirc.qubits]
    is_flat_3d = all(idx[2] == 0 for idx in indices if len(idx) == 3)
    for qb in tkcirc.qubits:
        if len(qb.index) == 0:
            qmap.update({qb: cirq.ops.NamedQubit(qb.reg_name)})
        elif len(qb.index) == 1:
            if line_name != None and line_name != qb.reg_name:
                raise NotImplementedError(
                    "Cirq can only support a single linear register"
                )
            line_name = qb.reg_name
            qmap.update({qb: LineQubit(qb.index[0])})
        elif len(qb.index) == 2 or (len(qb.index) == 3 and is_flat_3d):
            if grid_name != None and grid_name != qb.reg_name:
                raise NotImplementedError(
                    "Cirq can only support a single grid register"
                )
            grid_name = qb.reg_name
            qmap.update({qb: GridQubit(qb.index[0], qb.index[1])})
        else:
            raise NotImplementedError(
                "Cirq can only support registers of dimension <=2"
            )
    oplst = []
    for command in tkcirc:
        op = command.op
        optype = op.type
        try:
            gatetype = _ops2cirq_mapping[optype]
        except KeyError as error:
            raise NotImplementedError(
                "Cannot convert tket Op to Cirq gate: " + op.get_name()
            ) from error
        if optype == OpType.Measure:
            qid = qmap[command.args[0]]
            bit = command.args[1]
            cirqop = cirq.ops.measure(qid, key=bit.__repr__())
        else:
            qids = [qmap[qbit] for qbit in command.args]
            params = op.params
            if len(params) == 0:
                cirqop = gatetype(*qids)
            elif optype == OpType.PhasedX:
                cirqop = gatetype(phase_exponent=params[1], exponent=params[0])(*qids)
            elif optype == OpType.FSim:
                cirqop = gatetype(
                    theta=float(params[0] * pi), phi=float(params[1] * pi)
                )(*qids)
            elif optype == OpType.PhasedISWAP:
                cirqop = gatetype(phase_exponent=params[0], exponent=params[1])(*qids)
            else:
                cirqop = gatetype(exponent=params[0])(*qids)
        oplst.append(cirqop)
    try:

        coeff = cmath.exp(float(tkcirc.phase) * cmath.pi * 1j)
        if coeff.real < 1e-8:  # tolerance permitted by cirq for GlobalPhaseOperation
            coeff = coeff.imag * 1j
        if coeff.imag < 1e-8:
            coeff = coeff.real
        if coeff != 1.0:
            oplst.append(cirq.ops.GlobalPhaseOperation(coeff))
    except ValueError:
        warning(
            "Global phase is dependent on a symbolic parameter, so cannot adjust for "
            "phase"
        )
    return cirq.circuits.Circuit(*oplst)


# For converting cirq devices to tket devices


def _sort_row_col(qubits: FrozenSet[GridQubit]) -> List[GridQubit]:
    """Sort grid qubits first by row then by column"""

    return sorted(qubits, key=lambda x: (x.row, x.col))


def process_characterisation(xmon: cirq_google.XmonDevice) -> dict:
    """Generates a tket dictionary containing device characteristics for a Cirq
    :py:class:`XmonDevice`.

    :param xmon: The device to convert

    :return: A dictionary containing device characteristics
    """
    qb_map = {q: Node("q", q.row, q.col) for q in xmon.qubits}

    indexed_qubits = _sort_row_col(xmon.qubits)
    coupling_map = []
    for qb in indexed_qubits:
        neighbours = xmon.neighbors_of(qb)
        for x in neighbours:
            coupling_map.append((qb_map[qb], qb_map[x]))
    arc = Architecture(coupling_map)

    characterisation = dict()
    characterisation["Architecture"] = arc

    return characterisation
