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

import json
import math
from collections import Counter
from shutil import which
from time import sleep
import platform

from typing import cast, Dict
import docker  # type: ignore
import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from pytket.extensions.pyquil import (
    ForestBackend,
    ForestStateBackend,
    process_characterisation,
)
from pytket.circuit import BasisOrder, Circuit, OpType, Qubit, Node  # type: ignore
from pytket.passes import CliffordSimp  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.utils.expectations import (
    get_operator_expectation_value,
    get_pauli_expectation_value,
)
from pytket.utils import QubitPauliOperator

skip_qvm_tests = (which("docker") is None) or (platform.system() == "Windows")


@pytest.fixture(scope="module")
def qvm(request: FixtureRequest) -> None:
    dock = docker.from_env()
    container = dock.containers.run(
        image="rigetti/qvm", command="-S", detach=True, ports={5000: 5000}, remove=True
    )
    # container = dock.containers.run(image="rigetti/qvm", command="-S", detach=True,
    # publish_all_ports=True, remove=True)
    sleep(0.1)
    request.addfinalizer(container.stop)
    return None


@pytest.fixture(scope="module")
def quilc(request: FixtureRequest) -> None:
    dock = docker.from_env()
    container = dock.containers.run(
        image="rigetti/quilc",
        command="-S",
        detach=True,
        ports={5555: 5555},
        remove=True,
    )
    sleep(0.1)
    request.addfinalizer(container.stop)
    return None


def circuit_gen(measure: bool = False) -> Circuit:
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    if measure:
        c.measure_all()
    return c


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_statevector(qvm: None, quilc: None) -> None:
    c = circuit_gen()
    b = ForestStateBackend()
    state = b.run_circuit(c).get_state()
    assert np.allclose(
        state, np.asarray([math.sqrt(0.5), 0, 0, math.sqrt(0.5)]), atol=1e-10
    )
    c.add_phase(0.5)
    state1 = b.run_circuit(c).get_state()
    assert np.allclose(state1, state * 1j, atol=1e-10)


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
@pytest.mark.filterwarnings("ignore:strict=False")
def test_sim(qvm: None, quilc: None) -> None:
    c = circuit_gen(True)
    b = ForestBackend("9q-square")
    c = b.get_compiled_circuit(c)
    _ = b.run_circuit(c, n_shots=1024).get_shots()


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_measures(qvm: None, quilc: None) -> None:
    n_qbs = 9
    c = Circuit(n_qbs, n_qbs)
    x_qbs = [2, 5, 7, 8]
    for i in x_qbs:
        c.X(i)
    c.measure_all()
    b = ForestBackend("9q-square")
    c = b.get_compiled_circuit(c)
    shots = b.run_circuit(c, n_shots=10).get_shots()
    all_ones = True
    all_zeros = True
    for i in x_qbs:
        all_ones = all_ones and cast(bool, np.all(shots[:, i]))
    for i in range(n_qbs):
        if i not in x_qbs:
            all_zeros = all_zeros and (not np.any(shots[:, i]))
    assert all_ones
    assert all_zeros


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_pauli_statevector(qvm: None, quilc: None) -> None:
    c = Circuit(2, 2)
    c.Rz(0.5, 0)
    b = ForestStateBackend()
    zi = QubitPauliString(Qubit(0), Pauli.Z)
    assert get_pauli_expectation_value(c, zi, b) == 1
    c.X(0)
    assert get_pauli_expectation_value(c, zi, b) == -1


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_backendinfo(qvm: None, quilc: None) -> None:
    b = ForestBackend("9q-square")
    bi = b.backend_info
    node_gate_errors = cast(Dict, bi.all_node_gate_errors)
    edge_gate_errors = cast(Dict, bi.all_edge_gate_errors)

    assert bi
    assert len(node_gate_errors) == 9
    assert len(edge_gate_errors) == 12
    assert b.backend_info.architecture


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_pauli_sim(qvm: None, quilc: None) -> None:
    c = Circuit(2, 2)
    c.Rz(0.5, 0)
    b = ForestBackend("9q-square")
    zi = QubitPauliString(Qubit(0), Pauli.Z)
    energy = get_pauli_expectation_value(c, zi, b, 10)
    assert abs(energy - 1) < 0.001
    c.X(0)
    energy = get_pauli_expectation_value(c, zi, b, 10)
    assert abs(energy + 1) < 0.001


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_operator_statevector(qvm: None, quilc: None) -> None:
    c = Circuit(2, 2)
    c.Rz(0.5, 0)
    b = ForestStateBackend()
    zi = QubitPauliString(Qubit(0), Pauli.Z)
    iz = QubitPauliString(Qubit(1), Pauli.Z)
    op = QubitPauliOperator({zi: 0.3, iz: -0.1})
    assert get_operator_expectation_value(c, op, b) == pytest.approx(0.2)
    c.X(0)
    assert get_operator_expectation_value(c, op, b) == pytest.approx(-0.4)


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_operator_sim(qvm: None, quilc: None) -> None:
    c = Circuit(2, 2)
    c.Rz(0.5, 0)
    b = ForestBackend("9q-square")
    zi = QubitPauliString(Qubit(0), Pauli.Z)
    iz = QubitPauliString(Qubit(1), Pauli.Z)
    op = QubitPauliOperator({zi: 0.3, iz: -0.1})
    assert get_operator_expectation_value(c, op, b, 10) == pytest.approx(0.2, rel=0.001)
    c.X(0)
    assert get_operator_expectation_value(c, op, b, 10) == pytest.approx(
        -0.4, rel=0.001
    )


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_counts(qvm: None, quilc: None) -> None:
    c = circuit_gen(True)
    b = ForestBackend("9q-square")
    c = b.get_compiled_circuit(c)
    counts = b.run_circuit(c, n_shots=10).get_counts()
    assert all(x[0] == x[1] for x in counts.keys())


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_default_pass() -> None:
    b = ForestBackend("9q-square")
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


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_default_pass_2() -> None:
    b = ForestStateBackend()
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


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_ilo(qvm: None, quilc: None) -> None:
    b = ForestBackend("9q-square")
    bs = ForestStateBackend()
    c = Circuit(2)
    c.CZ(0, 1)
    c.Rx(1.0, 1)
    res = bs.run_circuit(c)
    assert np.allclose(res.get_state(), np.asarray([0, -1j, 0, 0]))
    assert np.allclose(res.get_state(basis=BasisOrder.dlo), np.asarray([0, 0, -1j, 0]))
    c.rename_units({Qubit(0): Node(0), Qubit(1): Node(1)})
    c.measure_all()

    res = b.run_circuit(c, n_shots=2)
    assert (res.get_shots() == np.asarray([[0, 1], [0, 1]])).all()
    assert (res.get_shots(basis=BasisOrder.dlo) == np.asarray([[1, 0], [1, 0]])).all()
    assert res.get_counts() == {(0, 1): 2}
    assert res.get_counts(basis=BasisOrder.dlo) == {(1, 0): 2}


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_swaps_basisorder() -> None:
    # Check that implicit swaps can be corrected irrespective of BasisOrder
    b = ForestStateBackend()
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


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_handle() -> None:
    b = ForestBackend("9q-square")
    c0 = Circuit(1)
    c0.measure_all()
    c1 = Circuit(1)
    c1.X(0)
    c1.measure_all()
    c0 = b.get_compiled_circuit(c0)
    c1 = b.get_compiled_circuit(c1)
    counts0 = b.run_circuit(c0, n_shots=10).get_counts()
    counts1 = b.run_circuit(c1, n_shots=10).get_counts()
    assert counts0 == {(0,): 10}
    assert counts1 == {(1,): 10}


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_state_handle() -> None:
    b = ForestStateBackend()
    c0 = Circuit(1)
    c1 = Circuit(1)
    c1.X(0)
    c0 = b.get_compiled_circuit(c0)
    c1 = b.get_compiled_circuit(c1)
    state0 = b.run_circuit(c0).get_state()
    state1 = b.run_circuit(c1).get_state()
    assert np.allclose(state0, np.asarray([1.0, 0.0]))
    assert np.allclose(state1, np.asarray([0.0, 1.0]))


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_delay_measures() -> None:
    b = ForestBackend("9q-square")
    # No triangles in architecture, so third CX will need a bridge
    # This will happen after the measurement on qubit 1
    c = Circuit(3, 3)
    c.CX(0, 1)
    c.CX(1, 2)
    c.CX(0, 2)
    c.Measure(0, 0)
    c.Measure(1, 1)
    c.Measure(2, 2)
    c = b.get_compiled_circuit(c)
    assert b.valid_circuit(c)


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_shots_bits_edgecases(qvm: None, quilc: None) -> None:
    forest_backend = ForestBackend("9q-square")

    for n_bits in range(1, 9):  # Getting runtime error if n_qubit > 9.
        for n_shots in range(1, 11):
            c = Circuit(n_bits, n_bits)

            # TODO TKET-813 add more shot based backends and move to integration tests
            c = forest_backend.get_compiled_circuit(c)
            h = forest_backend.process_circuit(c, n_shots)
            res = forest_backend.get_result(h)

            correct_shots = np.zeros((n_shots, n_bits), dtype=int)
            correct_shape = (n_shots, n_bits)
            correct_counts = Counter({(0,) * n_bits: n_shots})
            # BackendResult
            assert np.array_equal(res.get_shots(), correct_shots)
            assert res.get_shots().shape == correct_shape
            assert res.get_counts() == correct_counts
            # Direct
            res = forest_backend.run_circuit(c, n_shots=n_shots)
            assert np.array_equal(res.get_shots(), correct_shots)
            assert res.get_shots().shape == correct_shape
            assert res.get_counts() == correct_counts


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_postprocess() -> None:
    b = ForestBackend("9q-square")
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.Rx(0.5, 0).Rx(0.5, 1).CZ(0, 1).X(0).X(1).measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(json.loads(cast(str, h[1])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_process_characterisation(qvm: None, quilc: None) -> None:
    backend = ForestBackend("9q-square")
    char = process_characterisation(backend._qc)

    assert "NodeErrors" in char
    assert "EdgeErrors" in char
    assert "Architecture" in char

    node_errors = char["NodeErrors"]
    assert len(node_errors) == 9
    for i in range(9):
        assert Node(i) in node_errors
        errs = node_errors[Node(i)]
        for op, err in errs.items():
            assert isinstance(op, OpType)
            assert isinstance(err, float)

        assert OpType.X in errs
        assert OpType.V in errs
        assert OpType.Vdg in errs
        assert OpType.Rz in errs

    edge_errors = char["EdgeErrors"]
    assert len(edge_errors) == 12
    for (n1, n2), errs in edge_errors.items():
        assert n1 in node_errors
        assert n2 in node_errors
        for op, err in errs.items():
            assert isinstance(op, OpType)
            assert isinstance(err, float)

        assert OpType.CZ in errs
        assert OpType.ISWAP in errs

    arch = char["Architecture"]
    assert len(arch.nodes) == 9
    assert len(arch.coupling) == 12
