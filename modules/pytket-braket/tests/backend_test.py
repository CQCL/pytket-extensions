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

import json
from collections import Counter
from typing import cast
import os
from hypothesis import given, strategies
import numpy as np
import pytest
from pytket.extensions.braket import BraketBackend
from pytket.circuit import Circuit, OpType, Qubit  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.utils.expectations import (
    get_pauli_expectation_value,
    get_operator_expectation_value,
)
from pytket.utils.operators import QubitPauliOperator

# To test on AWS backends, first set up auth using boto3, then set the S3 bucket and
# folder in pytket config. See:
# https://github.com/aws/amazon-braket-sdk-python
# Otherwise, all tests are run on a local simulator.
skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of AWS storage)"


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_simulator() -> None:
    b = BraketBackend(
        device_type="quantum-simulator",
        provider="amazon",
        device="sv1",
    )
    assert b.supports_shots
    c = Circuit(2).H(0).CX(0, 1)
    c = b.get_compiled_circuit(c)
    n_shots = 100
    h0, h1 = b.process_circuits([c, c], n_shots)
    res0 = b.get_result(h0)
    readouts = res0.get_shots()
    assert all(readouts[i][0] == readouts[i][1] for i in range(n_shots))
    res1 = b.get_result(h1)
    counts = res1.get_counts()
    assert len(counts) <= 2
    assert sum(counts.values()) == n_shots
    zi = QubitPauliString(Qubit(0), Pauli.Z)
    assert b.get_pauli_expectation_value(
        c, zi, poll_timeout_seconds=60, poll_interval_seconds=1
    ) == pytest.approx(0)

    # Circuit with unused qubits
    c = Circuit(3).H(1).CX(1, 2)
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, 1)
    res = b.get_result(h)
    readout = res.get_shots()[0]
    assert readout[1] == readout[2]


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_dm_simulator() -> None:
    b = BraketBackend(device_type="quantum-simulator", provider="amazon", device="dm1")
    assert b.supports_density_matrix
    c = Circuit(2).H(0).SWAP(0, 1)
    cc = b.get_compiled_circuit(c)
    h = b.process_circuit(cc)
    r = b.get_result(h)
    m = r.get_density_matrix()
    m0 = np.zeros((4, 4), dtype=complex)
    m0[0, 0] = m0[1, 0] = m0[0, 1] = m0[1, 1] = 0.5
    assert np.allclose(m, m0)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_ionq() -> None:
    b = BraketBackend(
        device_type="qpu",
        provider="ionq",
        device="ionQdevice",
    )
    assert b.persistent_handles
    assert b.supports_shots
    assert not b.supports_state

    # Device is fully connected
    arch = b.backend_info.architecture
    n = len(arch.nodes)
    assert len(arch.coupling) == n * (n - 1)

    chars = b.characterisation
    assert chars is not None
    assert chars is not None
    assert all(s in chars for s in ["NodeErrors", "EdgeErrors", "ReadoutErrors"])
    assert b._characteristics is not None
    fid = b._characteristics["fidelity"]
    assert "1Q" in fid
    assert "2Q" in fid
    assert "spam" in fid
    tim = b._characteristics["timing"]
    assert "T1" in tim
    assert "T2" in tim

    c = (
        Circuit(3)
        .add_gate(OpType.XXPhase, 0.5, [0, 1])
        .add_gate(OpType.YYPhase, 0.5, [1, 2])
        .add_gate(OpType.SWAP, [0, 2])
        .add_gate(OpType.CCX, [0, 1, 2])
    )
    assert not b.valid_circuit(c)
    c = b.get_compiled_circuit(c)
    assert b.valid_circuit(c)
    h = b.process_circuit(c, 1)
    _ = b.circuit_status(h)
    b.cancel(h)

    # Circuit with unused qubits
    c = Circuit(11).H(9).CX(9, 10)
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, 1)
    b.cancel(h)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_rigetti() -> None:
    b = BraketBackend(
        device_type="qpu", provider="rigetti", device="Aspen-M-1", region="us-west-1"
    )
    assert b.persistent_handles
    assert b.supports_shots
    assert not b.supports_state

    chars = b.characterisation
    assert chars is not None
    assert all(s in chars for s in ["NodeErrors", "EdgeErrors", "ReadoutErrors"])

    c = (
        Circuit(3)
        .add_gate(OpType.CCX, [0, 1, 2])
        .add_gate(OpType.U1, 0.5, [1])
        .add_gate(OpType.ISWAP, 0.5, [0, 2])
        .add_gate(OpType.XXPhase, 0.5, [1, 2])
    )
    assert not b.valid_circuit(c)
    c = b.get_compiled_circuit(c)
    assert b.valid_circuit(c)
    h = b.process_circuit(c, 10)  # min shots = 10 for Rigetti
    _ = b.circuit_status(h)
    b.cancel(h)

    # Circuit with unused qubits
    c = Circuit(11).H(9).CX(9, 10)
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, 10)
    b.cancel(h)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_rigetti_with_rerouting() -> None:
    # A circuit that requires rerouting to a non-fully-connected architecture
    b = BraketBackend(
        device_type="qpu", provider="rigetti", device="Aspen-M-1", region="us-west-1"
    )
    c = Circuit(4).CX(0, 1).CX(0, 2).CX(0, 3).CX(1, 2).CX(1, 3).CX(2, 3)
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, 10)
    b.cancel(h)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_oqc() -> None:
    b = BraketBackend(
        device_type="qpu", provider="oqc", device="Lucy", region="eu-west-2"
    )
    assert b.persistent_handles
    assert b.supports_shots
    assert not b.supports_state

    chars = b.characterisation
    assert chars is not None
    assert all(s in chars for s in ["NodeErrors", "EdgeErrors", "ReadoutErrors"])

    c = (
        Circuit(3)
        .add_gate(OpType.CCX, [0, 1, 2])
        .add_gate(OpType.U1, 0.5, [1])
        .add_gate(OpType.ISWAP, 0.5, [0, 2])
        .add_gate(OpType.XXPhase, 0.5, [1, 2])
    )
    assert not b.valid_circuit(c)
    c = b.get_compiled_circuit(c)
    assert b.valid_circuit(c)
    h = b.process_circuit(c, 10)
    _ = b.circuit_status(h)
    b.cancel(h)

    # Circuit with unused qubits
    c = Circuit(7).H(5).CX(5, 6)
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, 10)
    b.cancel(h)


def test_local_simulator() -> None:
    b = BraketBackend(local=True)
    assert b.supports_shots
    assert b.supports_counts
    c = Circuit(2).H(0).CX(0, 1)
    c = b.get_compiled_circuit(c)
    n_shots = 100
    h = b.process_circuit(c, n_shots)
    res = b.get_result(h)
    readouts = res.get_shots()
    assert all(readouts[i][0] == readouts[i][1] for i in range(n_shots))
    counts = res.get_counts()
    assert len(counts) <= 2
    assert sum(counts.values()) == n_shots


def test_expectation() -> None:
    b = BraketBackend(local=True)
    assert b.supports_expectation
    c = Circuit(2, 2)
    c.Rz(0.5, 0)
    zi = QubitPauliString(Qubit(0), Pauli.Z)
    iz = QubitPauliString(Qubit(1), Pauli.Z)
    op = QubitPauliOperator({zi: 0.3, iz: -0.1})
    assert get_pauli_expectation_value(c, zi, b) == pytest.approx(1.0)
    assert get_operator_expectation_value(c, op, b) == pytest.approx(0.2)
    c.X(0)
    assert get_pauli_expectation_value(c, zi, b) == pytest.approx(-1.0)
    assert get_operator_expectation_value(c, op, b) == pytest.approx(-0.4)


def test_variance() -> None:
    b = BraketBackend(local=True)
    assert b.supports_variance
    # - Prepare a state (1/sqrt(2), 1/sqrt(2)).
    # - Measure w.r.t. the operator Z which has evcs (1,0) (evl=+1) and (0,1) (evl=-1).
    # - Get +1 with prob. 1/2 and -1 with prob. 1/2.
    c = Circuit(1).H(0)
    z = QubitPauliString(Qubit(0), Pauli.Z)
    assert b.get_pauli_expectation_value(c, z) == pytest.approx(0)
    assert b.get_pauli_variance(c, z) == pytest.approx(1)
    op = QubitPauliOperator({z: 3})
    assert b.get_operator_expectation_value(c, op) == pytest.approx(0)
    assert b.get_operator_variance(c, op) == pytest.approx(9)


def test_moments_with_shots() -> None:
    b = BraketBackend(local=True)
    c = Circuit(1).H(0)
    z = QubitPauliString(Qubit(0), Pauli.Z)
    e = b.get_pauli_expectation_value(c, z, n_shots=10)
    assert abs(e) <= 1
    v = b.get_pauli_variance(c, z, n_shots=10)
    assert v <= 1
    op = QubitPauliOperator({z: 3})
    e = b.get_operator_expectation_value(c, op, n_shots=10)
    assert abs(e) <= 3
    v = b.get_operator_variance(c, op, n_shots=10)
    assert v <= 9


def test_probabilities() -> None:
    b = BraketBackend(local=True)
    c = (
        Circuit(2)
        .H(0)
        .Rx(0.8, 1)
        .Rz(0.5, 0)
        .CX(0, 1)
        .Ry(0.3, 1)
        .CX(1, 0)
        .T(0)
        .S(1)
        .CX(0, 1)
        .Ry(1.8, 0)
    )
    probs01 = b.get_probabilities(c)
    probs10 = b.get_probabilities(c, qubits=[1, 0])
    probs0 = b.get_probabilities(c, qubits=[0])
    probs1 = b.get_probabilities(c, qubits=[1])
    assert probs01[0] == pytest.approx(probs10[0])
    assert probs01[1] == pytest.approx(probs10[2])
    assert probs01[2] == pytest.approx(probs10[1])
    assert probs01[3] == pytest.approx(probs10[3])
    assert probs0[0] == pytest.approx(probs01[0] + probs01[1])
    assert probs1[0] == pytest.approx(probs01[0] + probs01[2])
    h = b.process_circuit(c)
    res = b.get_result(h)
    dist = res.get_distribution()
    for (a0, a1), p in dist.items():
        assert probs01[2 * a0 + a1] == pytest.approx(p)


def test_probabilities_with_shots() -> None:
    b = BraketBackend(local=True)
    c = Circuit(2).V(1).CX(1, 0).S(1)
    probs_all = b.get_probabilities(c, n_shots=10)
    assert len(probs_all) == 4
    assert sum(probs_all) == pytest.approx(1)
    assert probs_all[1] == 0
    assert probs_all[2] == 0
    probs1 = b.get_probabilities(c, n_shots=10, qubits=[1])
    assert len(probs1) == 2
    assert sum(probs1) == pytest.approx(1)
    h = b.process_circuit(c, n_shots=10)
    res = b.get_result(h)
    dist = res.get_distribution()
    assert (1, 0) not in dist
    assert (0, 1) not in dist


def test_amplitudes() -> None:
    b = BraketBackend(local=True)
    c = Circuit(2).V(0).V(1).CX(1, 0).S(1)
    amps = b.get_amplitudes(c, states=["00", "01", "10", "11"])
    assert amps["00"] == pytest.approx(amps["11"])
    assert amps["01"] == pytest.approx(amps["10"])


def test_state() -> None:
    b = BraketBackend(local=True)
    c = Circuit(3).V(0).V(1).CX(1, 0).S(1).CCX(0, 1, 2)
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c)
    res = b.get_result(h)
    v = res.get_state()
    assert np.vdot(v, v) == pytest.approx(1)  # type: ignore


def test_default_pass() -> None:
    b = BraketBackend(local=True)
    for ol in range(3):
        comp_pass = b.default_compilation_pass(ol)
        c = Circuit(3, 3)
        c.H(0)
        c.CX(0, 1)
        c.CSWAP(1, 0, 2)
        c.ZZPhase(0.84, 2, 0)
        comp_pass.apply(c)
        for pred in b.required_predicates:
            assert pred.verify(c)


@given(
    n_shots=strategies.integers(min_value=1, max_value=10),  # type: ignore
    n_bits=strategies.integers(min_value=0, max_value=10),
)
def test_shots_bits_edgecases(n_shots, n_bits) -> None:
    braket_backend = BraketBackend(local=True)
    c = Circuit(n_bits, n_bits)

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = braket_backend.process_circuit(c, n_shots)
    res = braket_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult/
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    res = braket_backend.run_circuit(c, n_shots=n_shots)
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_postprocess_ionq() -> None:
    b = BraketBackend(
        device_type="qpu",
        provider="ionq",
        device="ionQdevice",
    )
    assert b.supports_contextual_optimisation
    c = Circuit(2).H(0).CX(0, 1).Y(0)
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(json.loads(cast(str, h[3])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    b.cancel(h)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_retrieve_available_devices() -> None:
    backend_infos = BraketBackend.available_devices()
    assert len(backend_infos) > 0
    # Test annealers are filtered out.
    backend_infos = BraketBackend.available_devices(region="us-west-2")
    assert len(backend_infos) > 0
