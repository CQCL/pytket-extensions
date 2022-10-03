# Copyright 2019-2022 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
##
#     http://www.apache.org/licenses/LICENSE-2.0
##
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conversion from to tket circuits to Qulacs circuits
"""
from typing import List, Tuple
import numpy as np
from qulacs import QuantumCircuit, gate  # type: ignore
from pytket.circuit import Circuit, OpType, Qubit  # type: ignore

_ONE_QUBIT_GATES = {
    OpType.X: gate.X,
    OpType.Y: gate.Y,
    OpType.Z: gate.Z,
    OpType.H: gate.H,
    OpType.S: gate.S,
    OpType.Sdg: gate.Sdag,
    OpType.T: gate.T,
    OpType.Tdg: gate.Tdag,
}

_ONE_QUBIT_ROTATIONS = {OpType.Rx: gate.RX, OpType.Ry: gate.RY, OpType.Rz: gate.RZ}

_MEASURE_GATES = {OpType.Measure: gate.Measurement}

_TWO_QUBIT_GATES = {OpType.CX: gate.CNOT, OpType.CZ: gate.CZ, OpType.SWAP: gate.SWAP}

_IBM_GATES = {OpType.U1: gate.U1, OpType.U2: gate.U2, OpType.U3: gate.U3}


def _get_implicit_swaps(circuit: Circuit) -> List[Tuple[Qubit, Qubit]]:
    # We implement the implicit qubit permutation using SWAPs
    qubits = circuit.qubits
    perm = circuit.implicit_qubit_permutation()
    # output wire -> qubit
    qubit_2_wire = wire_2_qubit = {q: q for q in qubits}
    # qubit -> output wire
    swaps = []
    for q in qubits:
        q_wire = qubit_2_wire[q]
        target_wire = perm[q]
        if q_wire == target_wire:
            continue
        # find which qubit is on target_wire
        p = wire_2_qubit[target_wire]
        # swap p and q so q is on the target wire
        swaps.append((q_wire, target_wire))
        # update dicts
        qubit_2_wire[q] = target_wire
        qubit_2_wire[p] = q_wire
        wire_2_qubit[q_wire] = p
        wire_2_qubit[target_wire] = q
    return swaps


def tk_to_qulacs(
    circuit: Circuit, reverse_index: bool = False, replace_implicit_swaps: bool = False
) -> QuantumCircuit:
    """Convert a pytket circuit to a qulacs circuit object."""
    n_qubits = circuit.n_qubits
    qulacs_circ = QuantumCircuit(circuit.n_qubits)
    index_map = {
        i: (i if not reverse_index else n_qubits - 1 - i) for i in range(n_qubits)
    }
    for com in circuit:
        optype = com.op.type
        if optype in _IBM_GATES:
            qulacs_gate = _IBM_GATES[optype]
            index = index_map[com.qubits[0].index[0]]

            if optype == OpType.U1:
                param = com.op.params[0]
                add_gate = qulacs_gate(index, param * np.pi)
            elif optype == OpType.U2:
                param0, param1 = com.op.params
                add_gate = qulacs_gate(index, param0 * np.pi, param1 * np.pi)
            elif optype == OpType.U3:
                param0, param1, param2 = com.op.params
                add_gate = qulacs_gate(
                    index, param0 * np.pi, param1 * np.pi, param2 * np.pi
                )

        elif optype in _ONE_QUBIT_GATES:
            qulacs_gate = _ONE_QUBIT_GATES[optype]  # type: ignore
            index = index_map[com.qubits[0].index[0]]
            add_gate = qulacs_gate(index)

        elif optype in _ONE_QUBIT_ROTATIONS:
            qulacs_gate = _ONE_QUBIT_ROTATIONS[optype]  # type: ignore
            index = index_map[com.qubits[0].index[0]]
            param = com.op.params[0] * np.pi
            add_gate = qulacs_gate(index, -param)  # parameter negated for qulacs

        elif optype in _TWO_QUBIT_GATES:
            qulacs_gate = _TWO_QUBIT_GATES[optype]  # type: ignore
            id1 = index_map[com.qubits[0].index[0]]
            id2 = index_map[com.qubits[1].index[0]]
            add_gate = qulacs_gate(id1, id2)

        elif optype in _MEASURE_GATES:
            continue
            # gate = _MEASURE_GATES[optype]
            # qubit = com.qubits[0].index[0]
            # bit = com.bits[0].index[0]
            # add_gate = (gate(qubit, bit))

        elif optype == OpType.Barrier:
            continue

        else:
            raise NotImplementedError(
                "Gate: {} Not Implemented in Qulacs!".format(optype)
            )
        qulacs_circ.add_gate(add_gate)

    if replace_implicit_swaps:
        swaps = _get_implicit_swaps(circuit)
        for p, q in swaps:
            qulacs_circ.add_gate(
                gate.SWAP(index_map[p.index[0]], index_map[q.index[0]])
            )

    return qulacs_circ
