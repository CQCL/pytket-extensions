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

import os
import pytest
from pytket.circuit import Circuit  # type: ignore
from pytket.backends import StatusEnum
from pytket.extensions.iqm import IQMBackend
from requests import HTTPError

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of IQM credentials)"


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_iqm() -> None:
    # Run a circuit on the demo device.
    b = IQMBackend()
    c = Circuit(4, 4)
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 2)
    c.CSWAP(0, 1, 2)
    c.CRz(0.4, 2, 3)
    c.CY(1, 3)
    c.ZZPhase(0.1, 2, 0)
    c.Tdg(3)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    res = b.run_circuit(c, n_shots=n_shots, timeout=30)
    shots = res.get_shots()
    counts = res.get_counts()
    assert len(shots) == n_shots
    assert sum(counts.values()) == n_shots


def test_invalid_cred() -> None:
    b = IQMBackend(username="invalid", api_key="invalid")
    c = Circuit(2, 2).H(0).CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    with pytest.raises(HTTPError):
        b.process_circuit(c, 1)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_handles() -> None:
    b = IQMBackend()
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
        assert b.circuit_status(handle).status in [
            StatusEnum.SUBMITTED,
            StatusEnum.COMPLETED,
        ]
    results = b.get_results(handles)
    for handle in handles:
        assert b.circuit_status(handle).status == StatusEnum.COMPLETED
    for result in results:
        assert result.get_shots().shape == (n_shots, 2)


def test_none_nshots() -> None:
    b = IQMBackend()
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    with pytest.raises(ValueError) as errorinfo:
        _ = b.process_circuits([c])
    assert "Parameter n_shots is required" in str(errorinfo.value)


def test_default_pass() -> None:
    b = IQMBackend(username="invalid", api_key="invalid")
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
    b = IQMBackend()
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.Y(0)
    c.Z(1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10
    assert all(len(shot) == 2 for shot in shots)


def test_backendinfo() -> None:
    b = IQMBackend(username="invalid", api_key="invalid")
    info = b.backend_info
    assert info.name == type(b).__name__
    assert len(info.gate_set) >= 3
