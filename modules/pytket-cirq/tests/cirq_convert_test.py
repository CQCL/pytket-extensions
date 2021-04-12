# Copyright 2019-2021 Cambridge Quantum Computing
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

from pytket.extensions.cirq import cirq_to_tk, tk_to_cirq, process_characterisation
from pytket.routing import Architecture  # type: ignore
from pytket.device import Device  # type: ignore

import cirq

from cirq.google import Foxtail
from cirq.circuits import InsertStrategy


def get_match_circuit() -> cirq.Circuit:
    qubits = [cirq.LineQubit(i) for i in range(9)]

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
            cirq.ISWAP(qubits[4], qubits[5]),
            cirq.FSimGate(1.4, 0.7)(qubits[6], qubits[7]),
            cirq.google.SYC(qubits[3], qubits[0]),
            cirq.PhasedISwapPowGate(phase_exponent=0.7, exponent=0.8)(
                qubits[3], qubits[4]
            ),
            cirq.GlobalPhaseOperation(1j),
            cirq.measure_each(*qubits[3:-2]),
        ],
        strategy=InsertStrategy.EARLIEST,
    )
    return circ


def test_conversions() -> None:
    circ = get_match_circuit()
    coms = cirq_to_tk(circ)

    cirq_false = tk_to_cirq(coms, copy_all_qubits=False)
    cirq_true = tk_to_cirq(coms, copy_all_qubits=True)
    assert str(circ) == str(cirq_false)
    assert str(cirq) != str(cirq_true)

    tket_false = cirq_to_tk(cirq_false)
    tket_true = cirq_to_tk(cirq_true)
    assert len(tket_false.get_commands()) + len(tket_false.qubits) == len(
        tket_true.get_commands()
    )
    assert tket_true != coms


def test_device() -> None:
    fox = Foxtail
    char = process_characterisation(fox)
    dev = Device(
        char.get("NodeErrors", {}),
        char.get("EdgeErrors", {}),
        char.get("Architecture", Architecture([])),
    )
    arc = dev.architecture
    assert str(arc) == "<tket::Architecture, nodes=22>"
