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

from math import isclose
from projectq import MainEngine  # type: ignore
from projectq.ops import All, Measure, H, NOT, CNOT, Rz, X  # type: ignore
from projectq.cengines._basics import ForwarderEngine  # type: ignore
from pytket.extensions.projectq import tketBackendEngine, tk_to_projectq, tketOptimiser
from pytket.circuit import Circuit, OpType  # type: ignore
import pytest

eps = 1e-7


@pytest.mark.filterwarnings("ignore: the matrix subclass ")
def test_H() -> None:
    circ = Circuit(1)
    circ.H(0)
    eng = MainEngine()
    qureg = eng.allocate_qureg(circ.n_qubits)
    tk_to_projectq(eng, qureg, circ)
    eng.flush()
    p1 = eng.backend.get_probability([0], qureg)
    p2 = eng.backend.get_probability([1], qureg)
    assert isclose(p1, 0.5, abs_tol=eps)
    assert isclose(p2, 0.5, abs_tol=eps)
    All(Measure) | qureg  # pylint: disable=expression-not-assigned


@pytest.mark.filterwarnings("ignore: the matrix subclass ")
def test_bellpair() -> None:
    circ = Circuit(2)
    circ.H(0)
    circ.CX(0, 1)
    eng = MainEngine()
    qureg = eng.allocate_qureg(circ.n_qubits)
    tk_to_projectq(eng, qureg, circ)
    eng.flush()
    p1 = eng.backend.get_probability([0, 0], qureg)
    p2 = eng.backend.get_probability([0, 1], qureg)
    p3 = eng.backend.get_probability([1, 0], qureg)
    p4 = eng.backend.get_probability([1, 1], qureg)
    assert p2 == 0
    assert p3 == 0
    assert isclose(p1, 0.5, abs_tol=eps)
    assert isclose(p4, 0.5, abs_tol=eps)
    All(Measure) | qureg  # pylint: disable=expression-not-assigned


@pytest.mark.filterwarnings("ignore: the matrix subclass ")
def test_backend_engine() -> None:
    # pylint: disable=pointless-statement
    # pylint: disable=expression-not-assigned
    backend_engine = tketBackendEngine()
    fwd = ForwarderEngine(backend_engine)
    engines = [fwd]
    eng = MainEngine(backend=backend_engine, engine_list=engines, verbose=True)
    qureg = eng.allocate_qureg(5)
    H | qureg[0]
    NOT | qureg[1]
    CNOT | (qureg[3], qureg[4])
    Rz(0.3) | qureg[2]
    All(Measure) | qureg
    eng.flush()
    c = backend_engine.circuit
    assert c.depth() == 2
    assert c.n_qubits == 5
    assert c._n_vertices() == 29
    assert c.n_gates == 9
    assert c.n_gates_of_type(OpType.CX) == 1
    assert c.n_gates_of_type(OpType.H) == 1
    assert c.n_gates_of_type(OpType.Measure) == 5
    assert c.n_gates_of_type(OpType.X) == 1
    assert c.n_gates_of_type(OpType.Rz) == 1


@pytest.mark.filterwarnings("ignore: the matrix subclass ")
def test_middle_engine() -> None:
    # pylint: disable=pointless-statement
    # pylint: disable=expression-not-assigned
    opti = tketOptimiser()
    engines = [opti]
    eng = MainEngine(engine_list=engines)

    circ = Circuit(2)
    circ.H(0)
    circ.CX(0, 1)
    circ.CX(0, 1)
    circ.CX(0, 1)
    circ.Rx(0.2, 1)
    circ.Rx(-0.2, 1)
    qureg = eng.allocate_qureg(circ.n_qubits)
    tk_to_projectq(eng, qureg, circ)
    Rz(0.3) | qureg[0]
    Rz(-0.3) | qureg[0]
    H | qureg[1]
    H | qureg[1]
    X | qureg[1]

    eng.flush()
    p1 = eng.backend.get_probability([0, 0], qureg)
    p2 = eng.backend.get_probability([0, 1], qureg)
    p3 = eng.backend.get_probability([1, 0], qureg)
    p4 = eng.backend.get_probability([1, 1], qureg)
    assert isclose(p1, 0.0, abs_tol=eps)
    assert isclose(p2, 0.5, abs_tol=eps)
    assert isclose(p3, 0.5, abs_tol=eps)
    assert isclose(p4, 0.0, abs_tol=eps)
