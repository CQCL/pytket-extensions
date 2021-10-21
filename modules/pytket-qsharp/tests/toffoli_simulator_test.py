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
from pytket.extensions.qsharp import QsharpToffoliSimulatorBackend


def test_incrementer() -> None:
    """
    Simulate an 8-bit incrementer
    """
    b = QsharpToffoliSimulatorBackend()
    c = Circuit(8)
    c.add_gate(OpType.CnX, [0, 1, 2, 3, 4, 5, 6, 7])
    c.add_gate(OpType.CnX, [0, 1, 2, 3, 4, 5, 6])
    c.add_gate(OpType.CnX, [0, 1, 2, 3, 4, 5])
    c.add_gate(OpType.CnX, [0, 1, 2, 3, 4])
    c.add_gate(OpType.CnX, [0, 1, 2, 3])
    c.CCX(0, 1, 2)
    c.CX(0, 1)
    c.X(0)

    for x in [0, 23, 79, 198, 255]:  # some arbitrary 8-bit numbers
        circ = Circuit(8)
        # prepare the state corresponding to x
        for i in range(8):
            if (x >> i) % 2 == 1:
                circ.X(i)
        # append the incrementer
        circ.add_circuit(c, list(range(8)))
        circ.measure_all()
        # run the simulator
        circ = b.get_compiled_circuit(circ)
        bits = b.run_circuit(circ, n_shots=1).get_shots()[0]
        # check the result
        for i in range(8):
            assert bits[i] == ((x + 1) >> i) % 2


def test_compile() -> None:
    """
    Compile a circuit containing SWAPs and noops down to CnX's
    """
    b = QsharpToffoliSimulatorBackend()
    c = Circuit(4)
    c.CX(0, 1)
    c.CCX(0, 1, 2)
    c.add_gate(OpType.CnX, [0, 1, 2, 3])
    c.add_gate(OpType.noop, [2])
    c.X(3)
    c.SWAP(1, 2)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    shots = b.run_circuit(c, n_shots=2).get_shots()
    assert all(shots[0] == shots[1])


def test_handles() -> None:
    b = QsharpToffoliSimulatorBackend()
    c = Circuit(4)
    c.CX(0, 1)
    c.CCX(0, 1, 2)
    c.add_gate(OpType.CnX, [0, 1, 2, 3])
    c.add_gate(OpType.noop, [2])
    c.X(3)
    c.SWAP(1, 2)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    shots = b.run_circuit(c, n_shots=2).get_shots()
    assert all(shots[0] == shots[1])


def test_default_pass() -> None:
    b = QsharpToffoliSimulatorBackend()
    for ol in range(3):
        comp_pass = b.default_compilation_pass(ol)
        c = Circuit(4, 4)
        c.CX(0, 1)
        c.CCX(0, 1, 2)
        c.add_gate(OpType.CnX, [0, 1, 2, 3])
        c.add_gate(OpType.noop, [2])
        c.X(3)
        c.SWAP(1, 2)
        c.measure_all()
        comp_pass.apply(c)
        for pred in b.required_predicates:
            assert pred.verify(c)
