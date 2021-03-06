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
import platform

from typing import cast
import docker  # type: ignore
import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from pytket.extensions.pyquil import ForestBackend, ForestStateBackend
from pytket.circuit import BasisOrder, Circuit, OpType, Qubit, Node  # type: ignore
from pytket.device import Device  # type: ignore
from pytket.passes import CliffordSimp  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.routing import Architecture  # type: ignore
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
    state = b.get_state(c)
    assert np.allclose(
        state, np.asarray([math.sqrt(0.5), 0, 0, math.sqrt(0.5)]), atol=1e-10
    )
    c.add_phase(0.5)
    state1 = b.get_state(c)
    assert np.allclose(state1, state * 1j, atol=1e-10)


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
@pytest.mark.filterwarnings("ignore:strict=False")
def test_sim(qvm: None, quilc: None) -> None:
    c = circuit_gen(True)
    b = ForestBackend("9q-square")
    b.compile_circuit(c)
    _ = b.get_shots(c, 1024)


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
    b.compile_circuit(c)
    shots = b.get_shots(c, 10)
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
def test_characterisation(qvm: None, quilc: None) -> None:
    b = ForestBackend("9q-square")
    char = b.characterisation
    assert char
    dev = Device(
        char.get("NodeErrors", {}),
        char.get("EdgeErrors", {}),
        char.get("Architecture", Architecture([])),
    )
    assert dev


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
    b.compile_circuit(c)
    counts = b.get_counts(c, 10)
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
    assert np.allclose(bs.get_state(c), np.asarray([0, -1j, 0, 0]))
    assert np.allclose(
        bs.get_state(c, basis=BasisOrder.dlo), np.asarray([0, 0, -1j, 0])
    )
    c.rename_units({Qubit(0): Node(0), Qubit(1): Node(1)})
    c.measure_all()
    assert (b.get_shots(c, 2) == np.asarray([[0, 1], [0, 1]])).all()
    assert (
        b.get_shots(c, 2, basis=BasisOrder.dlo) == np.asarray([[1, 0], [1, 0]])
    ).all()
    assert b.get_counts(c, 2) == {(0, 1): 2}
    assert b.get_counts(c, 2, basis=BasisOrder.dlo) == {(1, 0): 2}


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
    b.compile_circuit(c)
    s_ilo = b.get_state(c, basis=BasisOrder.ilo)
    s_dlo = b.get_state(c, basis=BasisOrder.dlo)
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
    b.compile_circuit(c0)
    b.compile_circuit(c1)
    counts0 = b.get_counts(c0, n_shots=10)
    counts1 = b.get_counts(c1, n_shots=10)
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
    b.compile_circuit(c0)
    b.compile_circuit(c1)
    state0 = b.get_state(c0)
    state1 = b.get_state(c1)
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
    b.compile_circuit(c)
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
            forest_backend.compile_circuit(c)
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
            assert np.array_equal(forest_backend.get_shots(c, n_shots), correct_shots)
            assert forest_backend.get_shots(c, n_shots).shape == correct_shape
            assert forest_backend.get_counts(c, n_shots) == correct_counts


@pytest.mark.skipif(
    skip_qvm_tests, reason="Can only run Rigetti QVM if docker is installed"
)
def test_postprocess() -> None:
    b = ForestBackend("9q-square")
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.Rx(0.5, 0).Rx(0.5, 1).CZ(0, 1).X(0).X(1).measure_all()
    b.compile_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(json.loads(cast(str, h[1])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10
