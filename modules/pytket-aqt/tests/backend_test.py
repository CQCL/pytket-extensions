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
from typing import cast
import os
from hypothesis import given, strategies
import numpy as np
import pytest
from pytket.circuit import Circuit  # type: ignore
from pytket.backends import StatusEnum
from pytket.extensions.aqt import AQTBackend

skip_remote_tests: bool = (
    os.getenv("PYTKET_RUN_REMOTE_TESTS") is None or os.getenv("AQT_AUTH") is None
)


@pytest.mark.skipif(
    skip_remote_tests,
    reason="requires environment variable AQT_AUTH to be a valid AQT credential",
)
def test_aqt() -> None:
    # Run a circuit on the noisy simulator.
    token = cast(str, os.getenv("AQT_AUTH"))
    b = AQTBackend(device_name="sim/noise-model-1", access_token=token, label="test 1")
    c = Circuit(4, 4)
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 2)
    c.CSWAP(0, 1, 2)
    c.CRz(0.4, 2, 3)
    c.CY(1, 3)
    c.add_barrier([0, 1])
    c.ZZPhase(0.1, 2, 0)
    c.Tdg(3)
    c.measure_all()
    b.compile_circuit(c)
    n_shots = 10
    shots = b.get_shots(c, n_shots, seed=1, timeout=30)
    counts = b.get_counts(c, n_shots)
    assert len(shots) == n_shots
    assert sum(counts.values()) == n_shots


@pytest.mark.skipif(
    skip_remote_tests,
    reason="requires environment variable AQT_AUTH to be a valid AQT credential",
)
def test_bell() -> None:
    # On the noiseless simulator, we should always get Bell states here.
    token = cast(str, os.getenv("AQT_AUTH"))
    b = AQTBackend(device_name="sim", access_token=token, label="test 2")
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    b.compile_circuit(c)
    n_shots = 10
    counts = b.get_counts(c, n_shots, timeout=30)
    assert all(q[0] == q[1] for q in counts)


def test_invalid_cred() -> None:
    token = "invalid"
    b = AQTBackend(device_name="sim", access_token=token, label="test 3")
    c = Circuit(2, 2).H(0).CX(0, 1)
    c.measure_all()
    b.compile_circuit(c)
    with pytest.raises(RuntimeError) as excinfo:
        b.process_circuits([c], 1)
        assert "Access denied" in str(excinfo.value)


@pytest.mark.skipif(
    skip_remote_tests,
    reason="requires environment variable AQT_AUTH to be a valid AQT credential",
)
def test_invalid_request() -> None:
    token = cast(str, os.getenv("AQT_AUTH"))
    b = AQTBackend(device_name="sim", access_token=token, label="test 4")
    c = Circuit(2, 2).H(0).CX(0, 1)
    c.measure_all()
    b.compile_circuit(c)
    with pytest.raises(RuntimeError) as excinfo:
        b.process_circuits([c], 1000000)
        assert "1000000" in str(excinfo.value)


@pytest.mark.skipif(
    skip_remote_tests,
    reason="requires environment variable AQT_AUTH to be a valid AQT credential",
)
def test_handles() -> None:
    token = cast(str, os.getenv("AQT_AUTH"))
    b = AQTBackend(device_name="sim/noise-model-1", access_token=token, label="test 5")
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    b.compile_circuit(c)
    n_shots = 5
    shots = b.get_shots(c, n_shots=n_shots, timeout=30)
    assert len(shots) == n_shots
    counts = b.get_counts(c, n_shots=n_shots)
    assert sum(counts.values()) == n_shots
    handles = b.process_circuits([c, c], n_shots=n_shots)
    assert len(handles) == 2
    for handle in handles:
        assert b.circuit_status(handle).status is StatusEnum.COMPLETED


def test_machine_debug() -> None:
    b = AQTBackend(device_name="sim", access_token="invalid", label="test 6")
    b._MACHINE_DEBUG = True
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    b.compile_circuit(c)
    n_shots = 10
    counts = b.get_counts(c, n_shots=n_shots, timeout=30)
    assert counts == {(0, 0): n_shots}


def test_default_pass() -> None:
    b = AQTBackend(device_name="sim/noise-model-1", access_token="invalid")
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


@given(
    n_shots=strategies.integers(min_value=1, max_value=10),  # type: ignore
    n_bits=strategies.integers(min_value=0, max_value=10),
)
def test_shots_bits_edgecases(n_shots, n_bits) -> None:
    aqt_backend = AQTBackend(device_name="sim", access_token="invalid", label="test 6")
    aqt_backend._MACHINE_DEBUG = True
    c = Circuit(n_bits, n_bits)

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = aqt_backend.process_circuit(c, n_shots)
    res = aqt_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    assert np.array_equal(aqt_backend.get_shots(c, n_shots), correct_shots)
    assert aqt_backend.get_shots(c, n_shots).shape == correct_shape
    assert aqt_backend.get_counts(c, n_shots) == correct_counts
