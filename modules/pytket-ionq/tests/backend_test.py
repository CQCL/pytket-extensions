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

import json
from collections import Counter
from typing import cast, TYPE_CHECKING
import os
from hypothesis import given, strategies
import numpy as np
import pytest
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.backends.backend_exceptions import CircuitNotValidError
from pytket.extensions.ionq import IonQBackend

if TYPE_CHECKING:
    from pytket.backends import ResultHandle

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of IoNQ API key)"


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_small_circuit_ionq() -> None:
    backend = IonQBackend(device_name="simulator", label="test 1")
    # backend = IonQBackend("invalid", device_name="simulator", label="test 5")
    # backend._MACHINE_DEBUG = True

    qc = Circuit(3, 3)
    qc.H(0)
    qc.CX(0, 2)
    qc.Measure(0, 1)
    qc.Measure(1, 0)
    qc.Measure(2, 2)
    qc = backend.get_compiled_circuit(qc)
    counts = backend.run_circuit(qc, n_shots=1000).get_counts()
    # note that we are rebuilding counts from probabilities, which are
    # floats, and therefore count number is not always preserved!
    assert counts[(0, 0, 0)] > 498 and counts[(0, 0, 0)] < 502
    assert counts[(0, 1, 1)] > 498 and counts[(0, 1, 1)] < 502
    assert sum(counts.values()) == 1000


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_big_circuit_ionq() -> None:
    backend = IonQBackend(device_name="simulator", label="test 2")
    circ = Circuit(4)
    circ.X(0).Y(0).Z(0).H(1).S(1).Sdg(1).H(1).T(2).Tdg(2).V(3).Vdg(3)
    circ.SWAP(0, 1)
    circ.CX(3, 2)
    circ.ZZPhase(1.2, 0, 1)
    circ.measure_all()
    counts = backend.run_circuit(circ, n_shots=100).get_counts()
    assert counts[(0, 0, 0, 0)] == 100


def test_invalid_token() -> None:
    token = "invalid"
    b = IonQBackend(api_key=token, device_name="simulator", label="test 3")
    c = Circuit(2, 2).H(0).CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    with pytest.raises(RuntimeError) as excinfo:
        b.process_circuits([c], 1)
        assert "Invalid key provided" in str(excinfo.value)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_invalid_request() -> None:
    b = IonQBackend(device_name="simulator", label="test 4")
    c = Circuit(2, 2).H(0).CZ(0, 1)
    c.measure_all()
    with pytest.raises(CircuitNotValidError) as excinfo:
        b.process_circuits([c], 100)
        assert "does not satisfy GateSetPredicate" in str(excinfo.value)


def test_machine_debug() -> None:
    b = IonQBackend(api_key="invalid", device_name="simulator", label="test 5")
    b._MACHINE_DEBUG = True
    c = Circuit(2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    n_shots = 100
    counts = b.run_circuit(c, n_shots=n_shots, timeout=30).get_counts()
    assert counts[(0, 0)] == n_shots


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_cancellation() -> None:
    b = IonQBackend(device_name="simulator", label="test 6")

    qc = Circuit(3, 3)
    qc.H(0)
    qc.CX(0, 2)
    qc.Measure(0, 1)
    qc.Measure(1, 0)
    qc.Measure(2, 2)
    qc = b.get_compiled_circuit(qc)
    h = b.process_circuit(qc, n_shots=100)
    b.cancel(h)


def test_default_pass() -> None:
    b = IonQBackend(api_key="invalid", device_name="simulator")
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
    b = IonQBackend(device_name="simulator", label="test 7")
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.Rx(0.1, 0)
    c.Ry(0.1, 1)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(json.loads(cast(str, h[3])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    r = b.get_result(h)
    counts = r.get_counts()
    assert sum(counts.values()) == 10


@given(
    n_shots=strategies.integers(min_value=1, max_value=10),  # type: ignore
    n_bits=strategies.integers(min_value=0, max_value=10),
)
def test_shots_bits_edgecases(n_shots, n_bits) -> None:
    ionq_backend = IonQBackend(
        api_key="invalid", device_name="simulator", label="test 5"
    )
    ionq_backend._MACHINE_DEBUG = True
    c = Circuit(n_bits, n_bits)
    # TODO TKET-813 add more shot based backends and move to integration tests
    h = ionq_backend.process_circuit(c, n_shots)
    res = ionq_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    res = ionq_backend.run_circuit(c, n_shots=n_shots)
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts


def test_nshots_ionq() -> None:
    b = IonQBackend(api_key="invalid", device_name="simulator", label="test 8")
    b._MACHINE_DEBUG = True
    circuit = Circuit(1, 1).Measure(0, 0)
    n_shots = [1, 2, 3]
    handles = b.process_circuits([circuit] * 3, n_shots=n_shots)

    def get_nshots(h: "ResultHandle") -> int:
        return cast(int, h._identifiers[1])

    assert [get_nshots(h) for h in handles] == n_shots
