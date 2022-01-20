# Copyright 2020-2022 Cambridge Quantum Computing
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

import warnings

import numpy as np
from qulacs import QuantumCircuit, QuantumState  # type: ignore
from qulacs.state import inner_product  # type: ignore
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.extensions.qulacs import tk_to_qulacs

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.warn("this will not show", DeprecationWarning)


def test_H() -> None:
    state = QuantumState(1)
    state.set_zero_state()
    circ = Circuit(1).H(0)
    qulacs_circ = tk_to_qulacs(circ)
    qulacs_circ.update_quantum_state(state)

    state0 = QuantumState(1)
    state0.set_computational_basis(0)
    probability = inner_product(state0, state) ** 2
    assert np.isclose(probability, 0.5)


def test_bellpair() -> None:
    state = QuantumState(2)
    circ = Circuit(2).H(0).CX(0, 1)
    qulacs_circ = tk_to_qulacs(circ)
    qulacs_circ.update_quantum_state(state)

    state0 = QuantumState(2)
    state0.set_computational_basis(0)
    probability = inner_product(state0, state) ** 2
    assert np.isclose(probability, 0.5)

    state1 = QuantumState(2)
    state1.set_computational_basis(1)
    probability = inner_product(state1, state) ** 2
    assert np.isclose(probability, 0)

    state2 = QuantumState(2)
    state2.set_computational_basis(2)
    probability = inner_product(state2, state) ** 2
    assert np.isclose(probability, 0)

    state3 = QuantumState(2)
    state3.set_computational_basis(3)
    probability = inner_product(state3, state) ** 2
    assert np.isclose(probability, 0.5)


def test_ibm_gateset() -> None:
    state = QuantumState(3)
    state.set_zero_state()
    circ = Circuit(3)
    circ.add_gate(OpType.U1, 0.19, [0])
    circ.add_gate(OpType.U2, [0.19, 0.24], [1])
    circ.add_gate(OpType.U3, [0.19, 0.24, 0.32], [2])
    qulacs_circ = tk_to_qulacs(circ)
    qulacs_circ.update_quantum_state(state)

    state1 = QuantumState(3)
    state1.set_zero_state()
    qulacs_circ1 = QuantumCircuit(3)
    qulacs_circ1.add_U1_gate(0, 0.19 * np.pi)
    qulacs_circ1.add_U2_gate(1, 0.19 * np.pi, 0.24 * np.pi)
    qulacs_circ1.add_U3_gate(2, 0.19 * np.pi, 0.24 * np.pi, 0.32 * np.pi)
    qulacs_circ1.update_quantum_state(state1)
    overlap = inner_product(state1, state)
    assert np.isclose(1, overlap)


def test_rotations() -> None:
    # https://github.com/CQCL/pytket/issues/35
    circ = Circuit(1).Rx(0.5, 0)
    qulacs_circ = tk_to_qulacs(circ)
    state = QuantumState(1)
    qulacs_circ.update_quantum_state(state)
    v = state.get_vector()
    assert np.isclose(v[0], np.sqrt(0.5))
    assert np.isclose(v[1], -1j * np.sqrt(0.5))
