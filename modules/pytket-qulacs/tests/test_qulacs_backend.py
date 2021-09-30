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

from collections import Counter
import warnings
import math
from hypothesis import given, strategies
import numpy as np
import pytest
from openfermion.ops import QubitOperator  # type: ignore
from openfermion.linalg import eigenspectrum  # type: ignore
from pytket.circuit import Circuit, BasisOrder, OpType, Qubit  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.passes import CliffordSimp  # type: ignore
from pytket.utils.operators import QubitPauliOperator
from pytket.extensions.qulacs import QulacsBackend

backends = [QulacsBackend()]

try:
    from pytket.extensions.qulacs import QulacsGPUBackend

    backends.append(QulacsGPUBackend())
except ImportError:
    warnings.warn("local settings failed to import QulacsGPUBackend", ImportWarning)

PARAM = -0.11176849


def h2_1q_circ(theta: float) -> Circuit:
    circ = Circuit(1)
    circ.Ry(-2 / np.pi * -theta, 0)
    return circ


def h2_2q_circ(theta: float) -> Circuit:
    circ = Circuit(2).X(0)
    circ.Rx(0.5, 0).H(1)
    circ.CX(0, 1)
    circ.Rz((-2 / np.pi) * theta, 1)
    circ.CX(0, 1)
    circ.Rx(-0.5, 0).H(1)
    return circ


def h2_3q_circ(theta: float) -> Circuit:
    circ = Circuit(3).X(0).X(1)
    circ.Rx(0.5, 0).H(1).H(2)
    circ.CX(0, 1).CX(1, 2)
    circ.Rz((-2 / np.pi) * theta, 2)
    circ.CX(1, 2).CX(0, 1)
    circ.Rx(-0.5, 0).H(1).H(2)
    return circ


def h2_4q_circ(theta: float) -> Circuit:
    circ = Circuit(4).X(0).X(1)
    circ.Rx(0.5, 0).H(1).H(2).H(3)
    circ.CX(0, 1).CX(1, 2).CX(2, 3)
    circ.Rz((-2 / np.pi) * theta, 3)
    circ.CX(2, 3).CX(1, 2).CX(0, 1)
    circ.Rx(-0.5, 0).H(1).H(2).H(3)
    return circ


def test_get_state() -> None:
    qulacs_circ = h2_4q_circ(PARAM)
    correct_state = np.array(
        [
            -4.97881051e-19 + 3.95546482e-17j,
            -2.04691245e-17 + 4.26119488e-18j,
            -2.05107665e-17 - 1.16628720e-17j,
            -1.11535930e-01 - 2.20309881e-16j,
            1.14532773e-16 + 1.84639112e-16j,
            -2.35945152e-18 + 1.00839027e-17j,
            -3.27177146e-18 - 1.35977120e-17j,
            1.68171141e-17 - 3.67997979e-17j,
            6.96542384e-18 + 6.20603820e-17j,
            2.94777720e-17 + 1.82756571e-19j,
            1.43716480e-17 + 3.62382653e-18j,
            3.41937038e-17 - 8.77511869e-18j,
            9.93760402e-01 + 1.59594560e-15j,
            -2.73151084e-18 + 6.31487294e-17j,
            2.09501038e-17 + 6.22364095e-17j,
            -8.59510231e-18 + 5.90202794e-18j,
        ]
    )
    for b in backends:
        qulacs_circ = b.get_compiled_circuit(qulacs_circ)
        qulacs_state = b.run_circuit(qulacs_circ).get_state()
        assert np.allclose(qulacs_state, correct_state)


def test_statevector_phase() -> None:
    for b in backends:
        circ = Circuit(2)
        circ.H(0).CX(0, 1)
        circ = b.get_compiled_circuit(circ)
        state = b.run_circuit(circ).get_state()
        assert np.allclose(state, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)], atol=1e-10)
        circ.add_phase(0.5)
        state1 = b.run_circuit(circ).get_state()
        assert np.allclose(state1, state * 1j, atol=1e-10)


def test_swaps_basisorder() -> None:
    # Check that implicit swaps can be corrected irrespective of BasisOrder
    for b in backends:
        c = Circuit(4)
        c.X(0)
        c.CX(0, 1)
        c.CX(1, 0)
        CliffordSimp(True).apply(c)
        assert c.n_gates_of_type(OpType.CX) == 1
        c = b.get_compiled_circuit(c)
        res = b.run_circuit(c)
        s_ilo = res.get_state(basis=BasisOrder.ilo)
        s_dlo = res.get_state(basis=BasisOrder.dlo)
        correct_ilo = np.zeros((16,))
        correct_ilo[4] = 1.0
        assert np.allclose(s_ilo, correct_ilo)
        correct_dlo = np.zeros((16,))
        correct_dlo[2] = 1.0
        assert np.allclose(s_dlo, correct_dlo)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_statevector_expectation() -> None:
    hamiltonian = QubitOperator()
    hamiltonian.terms = {
        (): 0.08406444459465776,
        ((0, "Z"),): 0.17218393261915543,
        ((1, "Z"),): 0.17218393261915546,
        ((0, "Z"), (1, "Z")): 0.16892753870087912,
        ((0, "Y"), (1, "X"), (2, "Y")): 0.04523279994605785,
        ((0, "X"), (1, "X"), (2, "X")): (0.04523279994605785),
        ((0, "Y"), (1, "Y"), (2, "X")): -0.04523279994605785,
        ((0, "X"), (1, "Y"), (2, "Y")): (0.04523279994605785),
        ((2, "Z"),): -0.45150698444804915,
        ((0, "Z"), (2, "Z")): 0.2870580651815905,
        ((1, "Z"), (2, "Z")): 0.2870580651815905,
    }
    target = eigenspectrum(hamiltonian)[0]
    circ = h2_3q_circ(PARAM)
    for b in backends:
        circ = b.get_compiled_circuit(circ)
        energy = b.get_operator_expectation_value(
            circ, QubitPauliOperator.from_OpenFermion(hamiltonian)
        )
        assert np.isclose(energy, target)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_basisorder() -> None:
    for b in backends:
        c = Circuit(2)
        c.X(1)
        b.process_circuit(c)
        assert (b.run_circuit(c).get_state() == np.asarray([0, 1, 0, 0])).all()
        assert (
            b.run_circuit(c).get_state(basis=BasisOrder.dlo) == np.asarray([0, 0, 1, 0])
        ).all()
        c.measure_all()
        assert b.run_circuit(c, n_shots=4, seed=4).get_shots().shape == (4, 2)
        assert b.run_circuit(c, n_shots=4, seed=4).get_counts() == {(0, 1): 4}


pauli_sym = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}


def qps_from_openfermion(paulis: QubitOperator) -> QubitPauliString:
    """Utility function to translate from openfermion format to a QubitPauliString"""
    qlist = []
    plist = []
    for q, p in paulis:
        qlist.append(Qubit(q))
        plist.append(pauli_sym[p])
    return QubitPauliString(qlist, plist)


def test_measurement_mask() -> None:
    for b in backends:
        n_shots = 10
        circ1 = Circuit(2, 2).X(0).X(1).measure_all()
        circ2 = Circuit(2, 2).X(0).measure_all()
        circ3 = Circuit(2, 1).X(1).Measure(0, 0)
        circ4 = Circuit(3, 2).X(0).Measure(0, 0).Measure(2, 1)
        circ_list = [circ1, circ2, circ3, circ4]
        target_shots = [[1, 1], [1, 0], [0], [1, 0]]

        for i, circ in enumerate(circ_list):
            shots = b.run_circuit(circ, n_shots=n_shots).get_shots()
            for sh in shots:
                assert len(sh) == len(target_shots[i])
                assert np.array_equal(sh, target_shots[i])


def test_default_pass() -> None:
    for b in backends:
        for ol in range(3):
            comp_pass = b.default_compilation_pass(ol)
            c = Circuit(3, 3)
            c.H(0)
            c.CX(0, 1)
            c.CSWAP(1, 0, 2)
            c.ZZPhase(0.84, 2, 0)
            c.measure_all()
            comp_pass.apply(c)
            for pred in b.required_predicates:
                assert pred.verify(c)


def test_no_measure_shots() -> None:
    for b in backends:
        c = Circuit(2, 2)
        c.H(0).CX(0, 1)
        # Note, no measurements
        c = b.get_compiled_circuit(c)
        handle = b.process_circuit(c, n_shots=10)
        counts = b.get_result(handle).get_counts()
        assert counts == {(0, 0): 10}


@given(
    n_shots=strategies.integers(min_value=1, max_value=10),  # type: ignore
    n_bits=strategies.integers(min_value=0, max_value=10),
)
def test_shots_bits_edgecases(n_shots, n_bits) -> None:
    c = Circuit(n_bits, n_bits)

    for qulacs_backend in backends:

        # TODO TKET-813 add more shot based backends and move to integration tests
        h = qulacs_backend.process_circuit(c, n_shots)
        res = qulacs_backend.get_result(h)

        correct_shots = np.zeros((n_shots, n_bits), dtype=int)
        correct_shape = (n_shots, n_bits)
        correct_counts = Counter({(0,) * n_bits: n_shots})
        # BackendResult
        assert np.array_equal(res.get_shots(), correct_shots)
        assert res.get_shots().shape == correct_shape
        assert res.get_counts() == correct_counts

        # Direct
        res = qulacs_backend.run_circuit(c, n_shots=n_shots)
        assert np.array_equal(res.get_shots(), correct_shots)
        assert res.get_shots().shape == correct_shape
        assert res.get_counts() == correct_counts
