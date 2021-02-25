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

from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.extensions.qsharp import QsharpSimulatorBackend


def test_bell() -> None:
    """
    Simulate a circuit that generates a Bell state, and check that the results
    are all (0,0) or (1,1).
    """
    b = QsharpSimulatorBackend()
    c = Circuit(2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    b.compile_circuit(c)
    n_shots = 10
    counts = b.get_counts(c, n_shots)
    assert all(m[0] == m[1] for m in counts.keys())
    assert sum(counts.values()) == n_shots


def test_rotations() -> None:
    """
    Check that Rz(0.5) acts as the identity.
    """
    b = QsharpSimulatorBackend()
    c = Circuit(1)
    c.Rz(0.5, 0)
    c.measure_all()
    b.compile_circuit(c)
    n_shots = 10
    shots = b.get_shots(c, n_shots)
    assert all(shots[i, 0] == 0 for i in range(n_shots))


def test_rebase() -> None:
    """
    Check that we can compile from a circuit containing non-Q# gates.
    """
    b = QsharpSimulatorBackend()
    c = Circuit(2)
    c.CY(0, 1)
    b.compile_circuit(c)


def test_cnx() -> None:
    """
    Simulate a CnX gate.
    """
    b = QsharpSimulatorBackend()
    c = Circuit(4)
    c.X(0).X(1).X(2)
    c.add_gate(OpType.CnX, [0, 1, 2, 3])
    c.measure_all()
    b.compile_circuit(c)
    n_shots = 3
    shots = b.get_shots(c, n_shots)
    assert all(shots[i, 3] == 1 for i in range(n_shots))


def test_handles() -> None:
    b = QsharpSimulatorBackend()
    c = Circuit(4)
    c.X(0).X(1).X(2)
    c.add_gate(OpType.CnX, [0, 1, 2, 3])
    c.measure_all()
    b.compile_circuit(c)
    n_shots = 3
    shots = b.get_shots(c, n_shots=n_shots)
    assert all(shots[i, 3] == 1 for i in range(n_shots))


def test_default_pass() -> None:
    b = QsharpSimulatorBackend()
    for ol in range(3):
        comp_pass = b.default_compilation_pass(ol)
        c = Circuit(4, 4)
        c.H(0)
        c.CX(0, 1)
        c.CSWAP(1, 0, 2)
        c.ZZPhase(0.84, 2, 0)
        c.measure_all()
        comp_pass.apply(c)
        for pred in b.required_predicates:
            assert pred.verify(c)


def test_nondefault_registers() -> None:
    c = Circuit()

    qreg = c.add_q_register("g", 3)
    creg1 = c.add_c_register("a", 3)
    creg2 = c.add_c_register("b", 3)

    c.X(qreg[1])
    c.X(qreg[0])
    c.Measure(qreg[1], creg1[1])
    c.Measure(qreg[0], creg2[0])

    b = QsharpSimulatorBackend()
    b.compile_circuit(c)
    counts = b.get_result(b.process_circuit(c, 10)).get_counts()

    assert counts == {(0, 1, 0, 1, 0, 0): 10}
