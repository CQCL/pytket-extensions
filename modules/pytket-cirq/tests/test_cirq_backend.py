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

from collections import Counter
from typing import List
import math
from hypothesis import given, strategies
import numpy as np
from pytket.extensions.cirq.backends.cirq import (  # type: ignore
    CirqDensityMatrixSampleBackend,
    CirqDensityMatrixSimBackend,
    CirqStateSampleBackend,
    CirqStateSimBackend,
    CirqCliffordSampleBackend,
    CirqCliffordSimBackend,
    _CirqSimBackend,
    _CirqBaseBackend,
)
from pytket.circuit import Circuit, Qubit, Bit  # type: ignore
from cirq.contrib.noise_models import DepolarizingNoiseModel  # type: ignore


def test_blank_wires() -> None:
    backends: List[_CirqBaseBackend] = [
        CirqDensityMatrixSimBackend(),
        CirqStateSimBackend(),
    ]
    for b in backends:
        assert b.get_result(b.process_circuit(Circuit(2).X(0))).q_bits == {
            Qubit(0): 0,
            Qubit(1): 1,
        }
        assert b.get_result(b.process_circuit(Circuit(2).X(1))).q_bits == {
            Qubit(0): 0,
            Qubit(1): 1,
        }
        for r in b.get_result(b.process_circuit_moments(Circuit(2).X(0).X(0).X(0))):  # type: ignore
            assert r.q_bits == {Qubit(0): 0, Qubit(1): 1}
        for r in b.get_result(b.process_circuit_moments(Circuit(2).X(1).X(1).X(1))):  # type: ignore
            assert r.q_bits == {Qubit(0): 0, Qubit(1): 1}

    for b in [CirqDensityMatrixSampleBackend(), CirqStateSampleBackend()]:
        assert b.get_result(
            b.process_circuit(Circuit(2, 2).X(0).Measure(0, 0), 100)
        ).c_bits == {Bit(0): 0}
        assert b.get_result(
            b.process_circuit(Circuit(2, 2).X(1).Measure(1, 1), 100)
        ).c_bits == {Bit(1): 0}
        assert b.get_result(
            b.process_circuit(Circuit(2, 2).X(1).Measure(1, 0), 100)
        ).c_bits == {Bit(0): 0}
        assert b.get_result(
            b.process_circuit(Circuit(2, 2).X(1).Measure(0, 1), 100)
        ).c_bits == {Bit(1): 0}


def test_moment_backends() -> None:
    b: _CirqSimBackend = CirqDensityMatrixSimBackend()
    all_res = b.get_result(
        b.process_circuit_moments(Circuit(1, 1).X(0).X(0).X(0).X(0).X(0))
    )

    assert (
        all_res[0].get_density_matrix()[1][1]  # type: ignore
        == all_res[2].get_density_matrix()[1][1]  # type: ignore
        == all_res[4].get_density_matrix()[1][1]  # type: ignore
        == 1
    )
    assert (
        all_res[1].get_density_matrix()[0][0]  # type: ignore
        == all_res[3].get_density_matrix()[0][0]  # type: ignore
        == 1
    )
    b = CirqStateSimBackend()
    all_res = b.get_result(
        b.process_circuit_moments(Circuit(1, 1).X(0).X(0).X(0).X(0).X(0))
    )
    assert (
        all_res[0].get_state()[1]  # type: ignore
        == all_res[2].get_state()[1]  # type: ignore
        == all_res[4].get_state()[1]  # type: ignore
        == 1
    )
    assert (
        all_res[1].get_state()[0]  # type: ignore
        == all_res[3].get_state()[0]  # type: ignore
        == 1
    )
    b = CirqCliffordSimBackend()
    all_res = b.get_result(
        b.process_circuit_moments(Circuit(1, 1).X(0).X(0).X(0).X(0).X(0))
    )
    assert (
        all_res[0].get_state()[1]  # type: ignore
        == all_res[2].get_state()[1]  # type: ignore
        == all_res[4].get_state()[1]  # type: ignore
        == 1
    )
    assert (
        all_res[1].get_state()[0]  # type: ignore
        == all_res[3].get_state()[0]  # type: ignore
        == 1
    )


def test_state() -> None:
    c0 = Circuit(2).H(0).CX(0, 1)
    b = CirqStateSimBackend()
    state0 = b.get_state(c0)
    assert np.allclose(state0, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)], atol=1e-10)
    c0.add_phase(0.5)
    state1 = b.get_state(c0)
    assert np.allclose(state1, state0 * 1j, atol=1e-10)
    assert np.allclose(b.get_state(Circuit(2).X(1)), [0, 1.0, 0, 0], atol=1e-10)


def test_density_matrix() -> None:
    b = CirqDensityMatrixSimBackend()
    res_h = b.get_result(b.process_circuit(Circuit(1).H(0)))
    density_matrix_h = res_h.get_density_matrix()
    assert np.allclose(density_matrix_h, np.array([[0.5, 0.5], [0.5, 0.5]]), atol=1e-10)
    res_xh = b.get_result(b.process_circuit(Circuit(2).X(0).H(1)))
    density_matrix_xh = res_xh.get_density_matrix()
    assert np.allclose(
        density_matrix_xh,
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]]),
        atol=1e-10,
    )


def test_shots_counts_cirq_state_sample_simulator() -> None:
    b = CirqStateSampleBackend()
    assert b.supports_shots
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 100
    h0, h1 = b.process_circuits([c, c], n_shots)
    res0 = b.get_result(h0)
    readouts = res0.get_shots()
    assert all(readout[0] == readout[1] for readout in readouts)
    res1 = b.get_result(h1)
    counts = res1.get_counts()
    assert len(counts) <= 2
    assert sum(counts.values()) == n_shots

    # Circuit with unused qubits
    c = Circuit(3, 2).H(1).CX(1, 2).measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, 1)
    res = b.get_result(h)
    readout = res.get_shots()[0]
    assert readout[1] == readout[2]


def test_shots_counts_cirq_dm_sample_simulator() -> None:
    b = CirqDensityMatrixSampleBackend()

    assert b.supports_shots
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 100
    h0, h1 = b.process_circuits([c, c], n_shots)
    res0 = b.get_result(h0)
    readouts = res0.get_shots()
    assert all(readout[0] == readout[1] for readout in readouts)
    res1 = b.get_result(h1)
    counts = res1.get_counts()
    assert len(counts) <= 2
    assert sum(counts.values()) == n_shots

    # Circuit with unused qubits
    c = Circuit(3, 2).H(1).CX(1, 2).measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, 1)
    res = b.get_result(h)
    readout = res.get_shots()[0]
    assert readout[1] == readout[2]


def test_shots_counts_cirq_clifford_sample_simulator() -> None:
    b = CirqCliffordSampleBackend()

    assert b.supports_shots
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    c = b.get_compiled_circuit(c)
    assert b.valid_circuit(c)
    n_shots = 100
    h0, h1 = b.process_circuits([c, c], n_shots)
    res0 = b.get_result(h0)
    readouts = res0.get_shots()
    assert all(readout[0] == readout[1] for readout in readouts)
    res1 = b.get_result(h1)
    counts = res1.get_counts()
    assert len(counts) <= 2
    assert sum(counts.values()) == n_shots

    # Circuit with unused qubits
    c = Circuit(3, 2).H(1).CX(1, 2).measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, 1)
    res = b.get_result(h)
    readout = res.get_shots()[0]
    assert readout[1] == readout[2]


def test_clifford_compilation() -> None:
    b = CirqCliffordSimBackend()
    c = Circuit(3).H(0).X(1).Y(2).CX(0, 1).CX(2, 1).H(0).X(0).H(1).Z(2)
    c = b.get_compiled_circuit(c)
    assert b.valid_circuit(c)
    c.Rz(0.3, 0)
    c = b.get_compiled_circuit(c)
    assert not b.valid_circuit(c)


def test_noisy_simulator_backends() -> None:
    nm = DepolarizingNoiseModel(depol_prob=0.01)
    sim_backend = CirqDensityMatrixSimBackend(noise=nm)
    sample_backend = CirqDensityMatrixSampleBackend(noise=nm)

    assert sim_backend._simulator.noise == nm
    assert sample_backend._simulator.noise == nm


@given(
    n_shots=strategies.integers(min_value=1, max_value=10),  # type: ignore
    n_bits=strategies.integers(min_value=0, max_value=10),
)
def test_shots_bits_edgecases(n_shots, n_bits) -> None:
    cirq_backend = CirqStateSampleBackend()
    c = Circuit(n_bits, n_bits)

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = cirq_backend.process_circuit(c, n_shots)
    res = cirq_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    assert np.array_equal(cirq_backend.get_shots(c, n_shots), correct_shots)
    assert cirq_backend.get_shots(c, n_shots).shape == correct_shape
    assert cirq_backend.get_counts(c, n_shots) == correct_counts
