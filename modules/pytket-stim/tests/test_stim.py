# Copyright 2020-2021 Cambridge Quantum Computing
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

from pytket.circuit import Bit, Circuit, OpType, Qubit  # type: ignore
from pytket.extensions.stim import StimBackend
import pytest


def test_bell_circuit() -> None:
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    b = StimBackend()
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10
    assert all(len(shot) == 2 and shot[0] == shot[1] for shot in shots)


def test_compilation() -> None:
    c = Circuit(3).Rx(0.5, 0).Ry(0.5, 1).Rz(0.5, 2).CX(0, 1).H(1).CZ(1, 2).measure_all()
    b = StimBackend()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


def test_non_clifford() -> None:
    c = Circuit(1).Rx(0.499, 0).measure_all()
    b = StimBackend()
    with pytest.raises(ValueError) as e:
        b.get_compiled_circuit(c)
        assert "Non-Clifford" in str(e.value)


def test_many_qubits() -> None:
    c = Circuit(30)
    c.H(0)
    for i in range(1, 30):
        c.CX(0, i)
    c.measure_all()
    b = StimBackend()
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10
    assert all(
        len(shot) == 30 and all(shot[0] == shot[i] for i in range(30)) for shot in shots
    )


def test_qubit_readout() -> None:
    c = Circuit(3, 2).X(1).X(2)
    c.add_gate(OpType.Measure, [Qubit(0), Bit(1)])
    c.add_gate(OpType.Measure, [Qubit(2), Bit(0)])
    b = StimBackend()
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10
    c0 = c.qubit_readout[Qubit(0)]
    c2 = c.qubit_readout[Qubit(2)]
    assert all(shot[c0] == 0 and shot[c2] == 1 for shot in shots)


def test_counts() -> None:
    c = Circuit(3).H(0).CX(0, 1).CX(0, 2).measure_all()
    b = StimBackend()
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    counts = r.get_counts()
    assert set(counts.keys()) <= set([(0, 0, 0), (1, 1, 1)])
    assert sum(counts.values()) == 10


def test_reset() -> None:
    c = Circuit(2).H(0).CX(0, 1)
    c.add_gate(OpType.Reset, [0])
    c.X(0)
    c.measure_all()
    b = StimBackend()
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    shots = r.get_shots()
    assert all(shot[0] == 1 for shot in shots)


def test_mid_circuit_measurement() -> None:
    c = Circuit(2, 2).X(0)
    c.add_gate(OpType.Measure, [Qubit(0), Bit(0)])
    c.H(1).CX(1, 0)
    c.add_gate(OpType.Measure, [Qubit(1), Bit(1)])
    b = StimBackend()
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    shots = r.get_shots()
    assert all(shot[0] == 1 for shot in shots)
