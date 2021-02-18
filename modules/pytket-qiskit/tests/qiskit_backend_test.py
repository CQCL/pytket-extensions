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

from typing import List
import platform
import numpy as np  # type: ignore
from pytket.backends import Backend
from pytket.extensions.backends.ibm import (
    AerBackend,
    AerStateBackend,
    AerUnitaryBackend,
)
from pytket.extensions.qiskit.tket_backend import TketBackend
from qiskit import QuantumCircuit, execute  # type: ignore
from qiskit.providers import JobStatus  # type: ignore
from qiskit.providers.aer import Aer  # type: ignore
from qiskit.aqua import QuantumInstance  # type: ignore
from qiskit.aqua.algorithms import BernsteinVazirani, DeutschJozsa  # type: ignore
from qiskit.aqua.components.oracles import TruthTableOracle  # type: ignore

# Memory corruption on Windows with qulacs 0.2.0 (TKET-1056)
use_qulacs = platform.system() != "Windows"
if use_qulacs:
    from pytket.extensions.backends.qulacs import QulacsBackend


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
        qb = Aer.get_backend("statevector_simulator")
        job2 = execute(qc, qb)
        state2 = job2.result().get_statevector()
        assert np.allclose(state, state2)


def test_unitary() -> None:
    qc = circuit_gen()
    b = AerUnitaryBackend()
    for comp in (None, b.default_compilation_pass()):
        tb = TketBackend(b, comp)
        job = execute(qc, tb)
        u = job.result().get_unitary()
        qb = Aer.get_backend("unitary_simulator")
        job2 = execute(qc, qb)
        u2 = job2.result().get_unitary()
        assert np.allclose(u, u2)


def test_aqua_algorithm() -> None:
    backends: List[Backend] = [AerBackend(), AerStateBackend()]
    if use_qulacs:
        backends.append(QulacsBackend())
    for b in backends:
        for comp in (None, b.default_compilation_pass()):
            if use_qulacs and type(b) == QulacsBackend and comp is None:
                continue
            tb = TketBackend(b, comp)
            ora = TruthTableOracle(bitmaps="01100110")
            alg = BernsteinVazirani(oracle=ora, quantum_instance=tb)
            result = alg.run()
            assert result["result"] == "011"
            alg = DeutschJozsa(oracle=ora, quantum_instance=tb)
            result = alg.run()
            assert result["result"] == "balanced"
            ora = TruthTableOracle(bitmaps="11111111")
            alg = DeutschJozsa(oracle=ora, quantum_instance=tb)
            result = alg.run()
            assert result["result"] == "constant"


def test_cancel() -> None:
    b = AerBackend()
    tb = TketBackend(b)
    qc = circuit_gen()
    job = execute(qc, tb)
    job.cancel()
    assert job.status() == JobStatus.CANCELLED
