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

from collections import Counter
import os
from hypothesis import given, strategies
import numpy as np
import pytest
from pytket.circuit import Circuit  # type: ignore
from pytket.backends import StatusEnum
from pytket.extensions.aqt import AQTBackend
from pytket.extensions.aqt.backends.aqt import _DEVICE_INFO

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of AQT access token)"


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_aqt() -> None:
    # Run a circuit on the noisy simulator.
    b = AQTBackend(device_name="sim/noise-model-1", label="test 1")
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
    c = b.get_compiled_circuit(c)
    n_shots = 10
    res = b.run_circuit(c, n_shots=n_shots, seed=1, timeout=30)
    shots = res.get_shots()
    counts = res.get_counts()
    assert len(shots) == n_shots
    assert sum(counts.values()) == n_shots


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_bell() -> None:
    # On the noiseless simulator, we should always get Bell states here.
    b = AQTBackend(device_name="sim", label="test 2")
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    counts = b.run_circuit(c, n_shots=n_shots, timeout=30).get_counts()
    assert all(q[0] == q[1] for q in counts)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_invalid_cred() -> None:
    token = "invalid"
    b = AQTBackend(device_name="sim", access_token=token, label="test 3")
    c = Circuit(2, 2).H(0).CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    with pytest.raises(RuntimeError) as excinfo:
        b.process_circuits([c], 1)
        assert "Access denied" in str(excinfo.value)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_invalid_request() -> None:
    b = AQTBackend(device_name="sim", label="test 4")
    c = Circuit(2, 2).H(0).CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    with pytest.raises(RuntimeError) as excinfo:
        b.process_circuits([c], 1000000)
        assert "1000000" in str(excinfo.value)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_handles() -> None:
    b = AQTBackend(device_name="sim/noise-model-1", label="test 5")
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 5
    res = b.run_circuit(c, n_shots=n_shots, timeout=30)
    shots = res.get_shots()
    assert len(shots) == n_shots
    counts = res.get_counts()
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
    c = b.get_compiled_circuit(c)
    n_shots = 10
    counts = b.run_circuit(c, n_shots=n_shots, timeout=30).get_counts()
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


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_postprocess() -> None:
    b = AQTBackend(device_name="sim", label="test 7")
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=100, postprocess=True)
    r = b.get_result(h)
    shots = r.get_shots()
    assert all(shot[0] == shot[1] for shot in shots)


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
    res = aqt_backend.run_circuit(c, n_shots=n_shots)
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts


def test_retrieve_available_devices() -> None:
    backend_infos = AQTBackend.available_devices()
    for machine, v in _DEVICE_INFO.items():
        assert (
            next(
                backend_info
                for backend_info in backend_infos
                if backend_info.device_name == machine
            ).n_nodes
            == v["max_n_qubits"]
        )
