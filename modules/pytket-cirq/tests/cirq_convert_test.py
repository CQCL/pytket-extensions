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

from typing import Union, List
import pytest
from pytket import OpType  # type: ignore
from pytket.extensions.cirq import cirq_to_tk, tk_to_cirq, process_characterisation
from pytket.architecture import Architecture  # type: ignore

import cirq
import cirq_google
from cirq.circuits import InsertStrategy
from cirq.devices import LineQubit, GridQubit
from cirq.ops import NamedQubit


CirqQubitType = Union[LineQubit, GridQubit, NamedQubit]


def get_match_circuit(cirq_qubit_type: str = "LineQubit") -> cirq.Circuit:
    qubits = List[CirqQubitType]
    if cirq_qubit_type == "LineQubit":
        qubits = [LineQubit(i) for i in range(9)]
    if cirq_qubit_type == "GridQubit":
        qubits = GridQubit.square(3)
    if cirq_qubit_type == "NamedQubit":
        qubits = NamedQubit.range(9, prefix="cirq")

    g = cirq.CZPowGate(exponent=0.1)
    zz = cirq.ZZPowGate(exponent=0.3)
    px = cirq.PhasedXPowGate(phase_exponent=0.6, exponent=0.2)
    circ = cirq.Circuit(
        [
            cirq.H(qubits[0]),
            cirq.X(qubits[1]),
            cirq.Y(qubits[2]),
            cirq.Z(qubits[3]),
            cirq.S(qubits[4]),
            cirq.CNOT(qubits[1], qubits[4]),
            cirq.T(qubits[3]),
            cirq.CNOT(qubits[6], qubits[8]),
            cirq.I(qubits[5]),
            cirq.XPowGate(exponent=0.1)(qubits[5]),
            cirq.YPowGate(exponent=0.1)(qubits[6]),
            cirq.ZPowGate(exponent=0.1)(qubits[7]),
            g(qubits[2], qubits[3]),
            zz(qubits[3], qubits[4]),
            px(qubits[6]),
            cirq.CZ(qubits[2], qubits[3]),
            cirq.H.controlled(1)(qubits[0], qubits[1]),
            cirq.ISWAP(qubits[4], qubits[5]),
            cirq.FSimGate(1.4, 0.7)(qubits[6], qubits[7]),
            cirq_google.SYC(qubits[3], qubits[0]),
            cirq.PhasedISwapPowGate(phase_exponent=0.7, exponent=0.8)(
                qubits[3], qubits[4]
            ),
            cirq.global_phase_operation(1j),
            cirq.measure_each(*qubits[3:-2]),
        ],
        strategy=InsertStrategy.EARLIEST,
    )
    return circ


@pytest.mark.parametrize("cirq_qubit_type", ["LineQubit", "GridQubit", "NamedQubit"])
def test_conversions(cirq_qubit_type: str) -> None:
    circ = get_match_circuit(cirq_qubit_type=cirq_qubit_type)
    coms = cirq_to_tk(circ)

    cirq_false = tk_to_cirq(coms, copy_all_qubits=False)
    cirq_true = tk_to_cirq(coms, copy_all_qubits=True)
    assert str(circ) == str(cirq_false)
    assert str(circ) != str(cirq_true)

    tket_false = cirq_to_tk(cirq_false)
    tket_true = cirq_to_tk(cirq_true)
    assert len(tket_false.get_commands()) + len(tket_false.qubits) == len(
        tket_true.get_commands()
    )
    assert tket_true != coms
    assert tket_false == coms


def test_device() -> None:
    syc = cirq_google.devices.Sycamore
    char = process_characterisation(syc)
    arc = char.get("Architecture", Architecture([]))
    assert str(arc) == "<tket::Architecture, nodes=54>"


@pytest.mark.parametrize("cirq_qubit_type", ["LineQubit", "GridQubit", "NamedQubit"])
def test_parallel_ops(cirq_qubit_type: str) -> None:
    if cirq_qubit_type == "LineQubit":
        q0, q1, q2 = [LineQubit(i) for i in range(3)]
    if cirq_qubit_type == "GridQubit":
        q0, q1, q2 = GridQubit.rect(rows=1, cols=3)
    if cirq_qubit_type == "NamedQubit":
        q0, q1, q2 = NamedQubit.range(3, prefix="cirq")
    circ = cirq.Circuit([cirq.ops.ParallelGate(cirq.Y**0.3, 3).on(q0, q1, q2)])
    c_tk = cirq_to_tk(circ)
    assert c_tk.n_gates_of_type(OpType.Ry) == 3
    assert c_tk.n_gates == 3


def test_unsupported_qubit_type() -> None:
    qdit = cirq.LineQid(1, dimension=2)
    circ = cirq.Circuit(
        [
            cirq.H(qdit),
        ]
    )
    with pytest.raises(NotImplementedError) as error:
        cirq_to_tk(circ)
        assert "Cannot convert qubits of type" in str(error.value)
