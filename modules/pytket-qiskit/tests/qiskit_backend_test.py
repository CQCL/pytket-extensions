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

import os
from typing import Optional
from pytket.extensions.qiskit import (
    AerBackend,
    AerStateBackend,
    AerUnitaryBackend,
    IBMQEmulatorBackend,
)
import numpy as np
import pytest
from pytket.extensions.qiskit.tket_backend import TketBackend
from qiskit import IBMQ, QuantumCircuit, execute  # type: ignore
from qiskit.providers.ibmq import AccountProvider  # type: ignore
from qiskit.opflow import CircuitStateFn, CircuitSampler  # type: ignore
from qiskit.providers import JobStatus  # type: ignore
from qiskit.providers.aer import Aer  # type: ignore
from qiskit.utils import QuantumInstance  # type: ignore

skip_remote_tests: bool = (
    not IBMQ.stored_account() or os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
)
REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires IBMQ configuration)"


@pytest.fixture
def provider() -> Optional["AccountProvider"]:
    if skip_remote_tests:
        return None
    else:
        if not IBMQ.active_account():
            IBMQ.load_account()
        return IBMQ.providers(hub="ibm-q")[0]


def circuit_gen(measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)
    if measure:
        qc.measure_all()
    return qc


def test_samples() -> None:
    qc = circuit_gen(True)
    b = AerBackend()
    for comp in (None, b.default_compilation_pass()):
        tb = TketBackend(b, comp)
        job = execute(qc, tb, shots=100, memory=True)
        shots = job.result().get_memory()
        assert all(((r[0] == "1" and r[1] == r[2]) for r in shots))
        counts = job.result().get_counts()
        assert all(((r[0] == "1" and r[1] == r[2]) for r in counts.keys()))


def test_state() -> None:
    qc = circuit_gen()
    b = AerStateBackend()
    for comp in (None, b.default_compilation_pass()):
        tb = TketBackend(b, comp)
        assert QuantumInstance(tb).is_statevector
        job = execute(qc, tb)
        state = job.result().get_statevector()
        qb = Aer.get_backend("aer_simulator_statevector")
        qc1 = qc.copy()
        qc1.save_state()
        job2 = execute(qc1, qb)
        state2 = job2.result().get_statevector()
        assert np.allclose(state, state2)


def test_unitary() -> None:
    qc = circuit_gen()
    b = AerUnitaryBackend()
    for comp in (None, b.default_compilation_pass()):
        tb = TketBackend(b, comp)
        job = execute(qc, tb)
        u = job.result().get_unitary()
        qb = Aer.get_backend("aer_simulator_unitary")
        qc1 = qc.copy()
        qc1.save_unitary()
        job2 = execute(qc1, qb)
        u2 = job2.result().get_unitary()
        assert np.allclose(u, u2)


def test_cancel() -> None:
    b = AerBackend()
    tb = TketBackend(b)
    qc = circuit_gen()
    job = execute(qc, tb, shots=1024)
    job.cancel()
    assert job.status() in [JobStatus.CANCELLED, JobStatus.DONE]


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_qiskit_counts(provider: Optional[AccountProvider]) -> None:
    num_qubits = 2
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    circfn = CircuitStateFn(qc)

    b = IBMQEmulatorBackend("ibmq_belem", account_provider=provider)

    s = CircuitSampler(TketBackend(b))

    res = s.sample_circuits([circfn])

    res_dictstatefn = res[id(circfn)][0]

    assert res_dictstatefn.num_qubits == num_qubits
