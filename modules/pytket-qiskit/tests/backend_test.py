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
import os
import sys
from collections import Counter
from typing import Dict, Any, Tuple, cast
import math
import cmath
import pickle
from hypothesis import given, strategies
import numpy as np
from pytket.circuit import Circuit, OpType, BasisOrder, Node, Qubit, reg_eq  # type: ignore
from pytket.passes import CliffordSimp  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.predicates import CompilationUnit, NoMidMeasurePredicate  # type: ignore
from pytket.routing import Architecture, route  # type: ignore
from pytket.transform import Transform  # type: ignore
from pytket.backends import (
    ResultHandle,
    CircuitNotRunError,
    CircuitNotValidError,
    CircuitStatus,
    StatusEnum,
)
from pytket.extensions.qiskit import (
    IBMQBackend,
    AerBackend,
    AerStateBackend,
    AerUnitaryBackend,
    IBMQEmulatorBackend,
)
from pytket.extensions.qiskit.backends.ibm import _rebase_pass
from pytket.extensions.qiskit import qiskit_to_tk, process_characterisation
from pytket.utils.expectations import (
    get_pauli_expectation_value,
    get_operator_expectation_value,
)
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.results import compare_unitaries
from qiskit import IBMQ  # type: ignore
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter  # type: ignore
from qiskit.providers.aer.noise.noise_model import NoiseModel  # type: ignore
from qiskit.providers.aer.noise.errors import depolarizing_error, pauli_error  # type: ignore
import pytest

# TODO add tests for `get_operator_expectation_value`

skip_remote_tests: bool = (
    not IBMQ.stored_account() or os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
)
REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of IBMQ account)"


@pytest.fixture(scope="module")
def santiago_backend() -> IBMQBackend:
    return IBMQBackend("ibmq_santiago", hub="ibm-q", group="open", project="main")


def circuit_gen(measure: bool = False) -> Circuit:
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    if measure:
        c.measure_all()
    return c


def get_test_circuit(measure: bool) -> QuantumRegister:
    qr = QuantumRegister(5)
    cr = ClassicalRegister(5)
    qc = QuantumCircuit(qr, cr)
    # qc.h(qr[0])
    qc.x(qr[0])
    qc.x(qr[2])
    qc.cx(qr[1], qr[0])
    # qc.h(qr[1])
    qc.cx(qr[0], qr[3])
    qc.cz(qr[2], qr[0])
    qc.cx(qr[1], qr[3])
    # qc.rx(PI/2,qr[3])
    qc.z(qr[2])
    if measure:
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        qc.measure(qr[3], cr[3])
    return qc


def test_statevector() -> None:
    c = circuit_gen()
    b = AerStateBackend()
    state = b.get_state(c)
    assert np.allclose(state, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)], atol=1e-10)
    c.add_phase(0.5)
    state1 = b.get_state(c)
    assert np.allclose(state1, state * 1j, atol=1e-10)


def test_sim() -> None:
    c = circuit_gen(True)
    b = AerBackend()
    shots = b.get_shots(c, 1024)
    print(shots)


def test_measures() -> None:
    n_qbs = 12
    c = Circuit(n_qbs, n_qbs)
    x_qbs = [2, 5, 7, 11]
    for i in x_qbs:
        c.X(i)
    c.measure_all()
    b = AerBackend()
    shots = b.get_shots(c, 10)
    all_ones = True
    all_zeros = True
    for i in x_qbs:
        all_ones = all_ones and bool(np.all(shots[:, i]))
    for i in range(n_qbs):
        if i not in x_qbs:
            all_zeros = all_zeros and (not np.any(shots[:, i]))
    assert all_ones
    assert all_zeros


def test_noise() -> None:
    with open(os.path.join(sys.path[0], "ibmqx2_properties.pickle"), "rb") as f:
        properties = pickle.load(f)

    noise_model = NoiseModel.from_backend(properties)
    n_qbs = 5
    c = Circuit(n_qbs, n_qbs)
    x_qbs = [2, 0, 4]
    for i in x_qbs:
        c.X(i)
    c.measure_all()
    b = AerBackend(noise_model)
    n_shots = 50
    b.compile_circuit(c)
    shots = b.get_shots(c, n_shots, seed=4)
    zer_exp = []
    one_exp = []
    for i in range(n_qbs):
        expectation = np.sum(shots[:, i]) / n_shots
        if i in x_qbs:
            one_exp.append(expectation)
        else:
            zer_exp.append(expectation)

    assert min(one_exp) > max(zer_exp)

    c2 = (
        Circuit(4, 4)
        .H(0)
        .CX(0, 2)
        .CX(3, 1)
        .T(2)
        .CX(0, 1)
        .CX(0, 3)
        .CX(2, 1)
        .measure_all()
    )

    b.compile_circuit(c2)
    shots = b.get_shots(c2, 10, seed=5)
    assert shots.shape == (10, 4)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_process_characterisation() -> None:
    if not IBMQ.active_account():
        IBMQ.load_account()

    provider = IBMQ.providers(hub="ibm-q", group="open")[0]
    back = provider.get_backend("ibmq_santiago")

    char = process_characterisation(back)
    arch: Architecture = char.get("Architecture", Architecture([]))
    node_errors: dict = char.get("NodeErrors", {})
    link_errors: dict = char.get("EdgeErrors", {})

    assert len(arch.nodes) == 5
    assert len(arch.coupling) == 8
    assert len(node_errors) == 5
    assert len(link_errors) == 8


def test_process_characterisation_no_noise_model() -> None:
    my_noise_model = NoiseModel()
    back = AerBackend(my_noise_model)
    assert back.characterisation is None

    c = Circuit(4).CX(0, 1).H(2).CX(2, 1).H(3).CX(0, 3).H(1).X(0)
    back.compile_circuit(c)
    assert back.valid_circuit(c)


def test_process_characterisation_incomplete_noise_model() -> None:

    my_noise_model = NoiseModel()

    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 1])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [1])
    my_noise_model.add_quantum_error(depolarizing_error(0.1, 1), ["u3"], [3])
    my_noise_model.add_quantum_error(
        pauli_error([("X", 0.35), ("Z", 0.65)]), ["u2"], [0]
    )
    my_noise_model.add_quantum_error(
        pauli_error([("X", 0.35), ("Y", 0.65)]), ["u1"], [2]
    )

    back = AerBackend(my_noise_model)

    c = Circuit(4).CX(0, 1).H(2).CX(2, 1).H(3).CX(0, 3).H(1).X(0).measure_all()
    back.compile_circuit(c)
    assert back.valid_circuit(c)

    arch = back.backend_info.architecture
    nodes = arch.nodes
    assert set(arch.coupling) == set(
        [
            (nodes[0], nodes[1]),
            (nodes[0], nodes[2]),
            (nodes[0], nodes[3]),
            (nodes[1], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[2], nodes[0]),
            (nodes[2], nodes[1]),
            (nodes[2], nodes[3]),
            (nodes[3], nodes[0]),
            (nodes[3], nodes[1]),
            (nodes[3], nodes[2]),
        ]
    )


def test_circuit_compilation_complete_noise_model() -> None:
    my_noise_model = NoiseModel()
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 1])
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 2])
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 3])
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [1, 2])
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [1, 3])
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [2, 3])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [0])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [1])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [2])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [3])

    back = AerBackend(my_noise_model)

    c = Circuit(4).CX(0, 1).H(2).CX(2, 1).H(3).CX(0, 3).H(1).X(0).measure_all()
    back.compile_circuit(c)
    assert back.valid_circuit(c)


def test_process_characterisation_complete_noise_model() -> None:
    my_noise_model = NoiseModel()

    readout_error_0 = 0.2
    readout_error_1 = 0.3
    my_noise_model.add_readout_error(
        [
            [1 - readout_error_0, readout_error_0],
            [readout_error_0, 1 - readout_error_0],
        ],
        [0],
    )
    my_noise_model.add_readout_error(
        [
            [1 - readout_error_1, readout_error_1],
            [readout_error_1, 1 - readout_error_1],
        ],
        [1],
    )

    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 1])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [0])
    my_noise_model.add_quantum_error(
        pauli_error([("X", 0.35), ("Z", 0.65)]), ["u2"], [0]
    )
    my_noise_model.add_quantum_error(
        pauli_error([("X", 0.35), ("Y", 0.65)]), ["u1"], [0]
    )

    back = AerBackend(my_noise_model)
    char = cast(Dict[str, Any], back.characterisation)

    # node_errors = cast(Dict[Node, dict], char.get("NodeErrors", {}))
    # link_errors = cast(Dict[Tuple[Node, Node], dict], char.get("EdgeErrors", {}))
    node_errors = back.backend_info.all_node_gate_errors
    link_errors = back.backend_info.all_edge_gate_errors
    arch = back.backend_info.architecture

    assert char["GenericTwoQubitQErrors"][(0, 1)][0][1][0] == 0.0375
    assert char["GenericTwoQubitQErrors"][(0, 1)][0][1][15] == 0.4375
    assert char["GenericOneQubitQErrors"][0][0][1][0] == 0.125
    assert char["GenericOneQubitQErrors"][0][0][1][3] == 0.625
    assert char["GenericOneQubitQErrors"][0][1][1][0] == 0.35
    assert char["GenericOneQubitQErrors"][0][1][1][1] == 0.65
    assert char["GenericOneQubitQErrors"][0][2][1][0] == 0.35
    assert char["GenericOneQubitQErrors"][0][2][1][1] == 0.65
    assert node_errors[arch.nodes[0]][OpType.U3] == 0.375
    assert link_errors[(arch.nodes[0], arch.nodes[1])][OpType.CX] == 0.5625
    assert link_errors[(arch.nodes[1], arch.nodes[0])][OpType.CX] == 0.80859375
    assert back.backend_info.all_readout_errors[arch.nodes[0]] == [
        [0.8, 0.2],
        [0.2, 0.8],
    ]
    assert back.backend_info.all_readout_errors[arch.nodes[1]] == [
        [0.7, 0.3],
        [0.3, 0.7],
    ]


def test_cancellation_aer() -> None:
    b = AerBackend()
    c = circuit_gen(True)
    b.compile_circuit(c)
    h = b.process_circuit(c, 10)
    b.cancel(h)
    print(b.circuit_status(h))


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_cancellation_ibmq(santiago_backend: IBMQBackend) -> None:
    b = santiago_backend
    c = circuit_gen(True)
    b.compile_circuit(c)
    h = b.process_circuit(c, 10)
    b.cancel(h)
    print(b.circuit_status(h))


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_machine_debug(santiago_backend: IBMQBackend) -> None:
    backend = santiago_backend
    backend._MACHINE_DEBUG = True
    try:
        c = Circuit(2, 2).H(0).CX(0, 1).measure_all()
        with pytest.raises(CircuitNotValidError) as errorinfo:
            handles = backend.process_circuits([c, c.copy()], n_shots=2)
        assert "in submitted does not satisfy GateSetPredicate" in str(errorinfo.value)
        backend.compile_circuit(c)
        handles = backend.process_circuits([c, c.copy()], n_shots=4)
        from pytket.extensions.qiskit.backends.ibm import _DEBUG_HANDLE_PREFIX

        assert all(
            cast(str, hand[0]).startswith(_DEBUG_HANDLE_PREFIX) for hand in handles
        )

        correct_shots = np.zeros((4, 2))
        correct_counts = {(0, 0): 4}

        shots = backend.get_shots(c, n_shots=4)
        assert np.all(shots == correct_shots)
        counts = backend.get_counts(c, n_shots=4)
        assert counts == correct_counts

        newshots = backend.get_shots(c, 4)
        assert np.all(newshots == correct_shots)
        newcounts = backend.get_counts(c, 4)
        assert newcounts == correct_counts
    finally:
        # ensure shared backend is reset for other tests
        backend._MACHINE_DEBUG = False


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_nshots_batching(santiago_backend: IBMQBackend) -> None:
    backend = santiago_backend
    backend._MACHINE_DEBUG = True
    try:
        c1 = Circuit(2, 2).H(0).CX(0, 1).measure_all()
        c2 = Circuit(2, 2).Rx(0.5, 0).CX(0, 1).measure_all()
        c3 = Circuit(2, 2).H(1).CX(0, 1).measure_all()
        c4 = Circuit(2, 2).Rx(0.5, 0).CX(0, 1).CX(1, 0).measure_all()
        cs = [c1, c2, c3, c4]
        n_shots = [10, 12, 10, 13]
        for c in cs:
            backend.compile_circuit(c)
        handles = backend.process_circuits(cs, n_shots=n_shots)

        from pytket.extensions.qiskit.backends.ibm import _DEBUG_HANDLE_PREFIX

        assert all(
            cast(str, hand[0]) == _DEBUG_HANDLE_PREFIX + suffix
            for hand, suffix in zip(
                handles,
                [f"{(2, 10, 0)}", f"{(2, 12, 1)}", f"{(2, 10, 0)}", f"{(2, 13, 2)}"],
            )
        )
    finally:
        # ensure shared backend is reset for other tests
        backend._MACHINE_DEBUG = False


def test_nshots() -> None:
    backends = [AerBackend()]
    if not skip_remote_tests:
        backends.append(
            IBMQEmulatorBackend(
                "ibmq_santiago", hub="ibm-q", group="open", project="main"
            )
        )
    for b in backends:
        circuit = Circuit(1)
        n_shots = [1, 2, 3]
        results = b.get_results(b.process_circuits([circuit] * 3, n_shots=n_shots))
        assert [len(r.get_shots()) for r in results] == n_shots


def test_pauli_statevector() -> None:
    c = Circuit(2)
    c.Rz(0.5, 0)
    Transform.OptimisePostRouting().apply(c)
    b = AerStateBackend()
    zi = QubitPauliString(Qubit(0), Pauli.Z)
    assert get_pauli_expectation_value(c, zi, b) == 1
    c.X(0)
    assert get_pauli_expectation_value(c, zi, b) == -1


def test_pauli_sim() -> None:
    c = Circuit(2, 2)
    c.Rz(0.5, 0)
    Transform.OptimisePostRouting().apply(c)
    b = AerBackend()
    zi = QubitPauliString(Qubit(0), Pauli.Z)
    energy = get_pauli_expectation_value(c, zi, b, 8000)
    assert abs(energy - 1) < 0.001
    c.X(0)
    energy = get_pauli_expectation_value(c, zi, b, 8000)
    assert abs(energy + 1) < 0.001


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_default_pass(santiago_backend: IBMQBackend) -> None:
    b = santiago_backend
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


def test_aer_default_pass() -> None:
    with open(os.path.join(sys.path[0], "ibmqx2_properties.pickle"), "rb") as f:
        properties = pickle.load(f)

    noise_model = NoiseModel.from_backend(properties)
    for nm in [None, noise_model]:
        b = AerBackend(nm)
        for ol in range(3):
            comp_pass = b.default_compilation_pass(ol)
            c = Circuit(3, 3)
            c.H(0)
            c.CX(0, 1)
            c.CSWAP(1, 0, 2)
            c.ZZPhase(0.84, 2, 0)
            c.add_gate(OpType.TK1, [0.2, 0.3, 0.4], [0])
            comp_pass.apply(c)
            c.measure_all()
            for pred in b.required_predicates:
                assert pred.verify(c)


def test_routing_measurements() -> None:
    qc = get_test_circuit(True)
    circ = qiskit_to_tk(qc)
    sim = AerBackend()
    original_results = sim.get_shots(circ, 10, seed=4)
    coupling = [[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]]
    arc = Architecture(coupling)
    physical_c = route(circ, arc)
    Transform.DecomposeSWAPtoCX().apply(physical_c)
    Transform.DecomposeCXDirected(arc).apply(physical_c)
    Transform.OptimisePostRouting().apply(physical_c)
    assert (sim.get_shots(physical_c, 10) == original_results).all()


def test_routing_no_cx() -> None:
    circ = Circuit(2, 2)
    circ.H(1)
    # c.CX(1, 2)
    circ.Rx(0.2, 0)
    circ.measure_all()
    coupling = [[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]]
    arc = Architecture(coupling)
    physical_c = route(circ, arc)

    assert len(physical_c.get_commands()) == 4


def test_counts() -> None:
    qc = get_test_circuit(True)
    circ = qiskit_to_tk(qc)
    sim = AerBackend()
    counts = sim.get_counts(circ, 10, seed=4)
    assert counts == {(1, 0, 1, 1, 0): 10}


def test_ilo() -> None:
    b = AerBackend()
    bs = AerStateBackend()
    bu = AerUnitaryBackend()
    c = Circuit(2)
    c.X(1)
    assert (bs.get_state(c) == np.asarray([0, 1, 0, 0])).all()
    assert (bs.get_state(c, basis=BasisOrder.dlo) == np.asarray([0, 0, 1, 0])).all()
    assert (
        bu.get_unitary(c)
        == np.asarray([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    ).all()
    assert (
        bu.get_unitary(c, basis=BasisOrder.dlo)
        == np.asarray([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    ).all()
    c.measure_all()
    assert (b.get_shots(c, 2) == np.asarray([[0, 1], [0, 1]])).all()
    assert (
        b.get_shots(c, 2, basis=BasisOrder.dlo) == np.asarray([[1, 0], [1, 0]])
    ).all()
    assert b.get_counts(c, 2) == {(0, 1): 2}
    assert b.get_counts(c, 2, basis=BasisOrder.dlo) == {(1, 0): 2}


def test_swaps_basisorder() -> None:
    # Check that implicit swaps can be corrected irrespective of BasisOrder
    b = AerStateBackend()
    c = Circuit(4)
    c.X(0)
    c.CX(0, 1)
    c.CX(1, 0)
    c.CX(1, 3)
    c.CX(3, 1)
    c.X(2)
    cu = CompilationUnit(c)
    CliffordSimp(True).apply(cu)
    c1 = cu.circuit
    assert c1.n_gates_of_type(OpType.CX) == 2

    b.compile_circuit(c)
    b.compile_circuit(c1)

    handles = b.process_circuits([c, c1])
    s_ilo = b.get_state(c1, basis=BasisOrder.ilo)
    correct_ilo = b.get_state(c, basis=BasisOrder.ilo)

    assert np.allclose(s_ilo, correct_ilo)
    s_dlo = b.get_state(c1, basis=BasisOrder.dlo)
    correct_dlo = b.get_state(c, basis=BasisOrder.dlo)
    assert np.allclose(s_dlo, correct_dlo)

    qbs = c.qubits
    for result in b.get_results(handles):
        assert (
            result.get_state([qbs[1], qbs[2], qbs[3], qbs[0]]).real.tolist().index(1.0)
            == 6
        )
        assert (
            result.get_state([qbs[2], qbs[1], qbs[0], qbs[3]]).real.tolist().index(1.0)
            == 9
        )
        assert (
            result.get_state([qbs[2], qbs[3], qbs[0], qbs[1]]).real.tolist().index(1.0)
            == 12
        )

    bu = AerUnitaryBackend()
    u_ilo = bu.get_unitary(c1, basis=BasisOrder.ilo)
    correct_ilo = bu.get_unitary(c, basis=BasisOrder.ilo)
    assert np.allclose(u_ilo, correct_ilo)
    u_dlo = bu.get_unitary(c1, basis=BasisOrder.dlo)
    correct_dlo = bu.get_unitary(c, basis=BasisOrder.dlo)
    assert np.allclose(u_dlo, correct_dlo)


def test_pauli() -> None:
    for b in [AerBackend(), AerStateBackend()]:
        c = Circuit(2)
        c.Rz(0.5, 0)
        b.compile_circuit(c)
        zi = QubitPauliString(Qubit(0), Pauli.Z)
        assert cmath.isclose(get_pauli_expectation_value(c, zi, b), 1)
        c.X(0)
        assert cmath.isclose(get_pauli_expectation_value(c, zi, b), -1)


def test_operator() -> None:
    for b in [AerBackend(), AerStateBackend()]:
        c = circuit_gen()
        zz = QubitPauliOperator(
            {QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z]): 1.0}
        )
        assert cmath.isclose(get_operator_expectation_value(c, zz, b), 1.0)
        c.X(0)
        assert cmath.isclose(get_operator_expectation_value(c, zz, b), -1.0)


def test_aer_result_handle() -> None:
    c = Circuit(2, 2).H(0).CX(0, 1).measure_all()

    b = AerBackend()

    handles = b.process_circuits([c, c.copy()], n_shots=2)

    ids, indices = zip(*(han for han in handles))

    assert all(isinstance(idval, str) for idval in ids)
    assert indices == (0, 1)

    assert len(b.get_result(handles[0]).get_shots()) == 2

    with pytest.raises(TypeError) as errorinfo:
        _ = b.get_result(ResultHandle("43"))
    assert "ResultHandle('43',) does not match expected identifier types" in str(
        errorinfo.value
    )

    wronghandle = ResultHandle("asdf", 3)

    with pytest.raises(CircuitNotRunError) as errorinfoCirc:
        _ = b.get_result(wronghandle)
    assert "Circuit corresponding to {0!r} ".format(
        wronghandle
    ) + "has not been run by this backend instance." in str(errorinfoCirc.value)


def test_aerstate_result_handle() -> None:
    c = circuit_gen()
    b1 = AerStateBackend()
    h1 = b1.process_circuits([c])[0]
    state = b1.get_result(h1).get_state()
    status = b1.circuit_status(h1)
    assert status == CircuitStatus(StatusEnum.COMPLETED, "job has successfully run")
    assert np.allclose(state, [np.sqrt(0.5), 0, 0, math.sqrt(0.5)], atol=1e-10)
    b2 = AerUnitaryBackend()
    unitary = b2.get_unitary(c)
    assert np.allclose(
        unitary,
        np.sqrt(0.5)
        * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [1, 0, -1, 0]]),
    )


def test_cache() -> None:
    b = AerBackend()
    c = circuit_gen()
    b.compile_circuit(c)
    h = b.process_circuits([c], 2)[0]
    b.get_result(h).get_shots()
    assert h in b._cache
    b.pop_result(h)
    assert h not in b._cache
    assert not b._cache

    b.get_counts(c, n_shots=2)
    b.get_counts(c.copy(), n_shots=2)
    b.empty_cache()
    assert not b._cache


def test_mixed_circuit() -> None:
    c = Circuit()
    qr = c.add_q_register("q", 2)
    ar = c.add_c_register("a", 1)
    br = c.add_c_register("b", 1)
    c.H(qr[0])
    c.Measure(qr[0], ar[0])
    c.X(qr[1], condition=reg_eq(ar, 0))
    c.Measure(qr[1], br[0])
    backend = AerBackend()
    backend.compile_circuit(c)
    counts = backend.get_counts(c, 1024)
    for key in counts.keys():
        assert key in {(0, 1), (1, 0)}


def test_aer_placed_expectation() -> None:
    # bug TKET-695
    n_qbs = 3
    c = Circuit(n_qbs, n_qbs)
    c.X(0)
    c.CX(0, 2)
    c.CX(1, 2)
    c.H(1)
    # c.measure_all()
    b = AerBackend()
    operator = QubitPauliOperator(
        {
            QubitPauliString(Qubit(0), Pauli.Z): 1.0,
            QubitPauliString(Qubit(1), Pauli.X): 0.5,
        }
    )
    assert b.get_operator_expectation_value(c, operator) == (-0.5 + 0j)
    with open(os.path.join(sys.path[0], "ibmqx2_properties.pickle"), "rb") as f:
        properties = pickle.load(f)

    noise_model = NoiseModel.from_backend(properties)

    noise_b = AerBackend(noise_model)

    with pytest.raises(RuntimeError) as errorinfo:
        noise_b.get_operator_expectation_value(c, operator)
        assert "not supported with noise model" in str(errorinfo.value)

    c.rename_units({Qubit(1): Qubit("node", 1)})
    with pytest.raises(ValueError) as errorinfoCirc:
        b.get_operator_expectation_value(c, operator)
        assert "default register Qubits" in str(errorinfoCirc.value)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_ibmq_emulator() -> None:
    b_emu = IBMQEmulatorBackend(
        "ibmq_santiago", hub="ibm-q", group="open", project="main"
    )
    assert b_emu._noise_model is not None
    b_ibm = b_emu._ibmq
    b_aer = AerBackend()
    for ol in range(3):
        comp_pass = b_emu.default_compilation_pass(ol)
        c = Circuit(3, 3)
        c.H(0)
        c.CX(0, 1)
        c.CSWAP(1, 0, 2)
        c.ZZPhase(0.84, 2, 0)
        c_cop = c.copy()
        comp_pass.apply(c_cop)
        c.measure_all()
        for bac in (b_emu, b_ibm):
            assert all(pred.verify(c_cop) for pred in bac.required_predicates)

        c_cop_2 = c.copy()
        b_aer.compile_circuit(c_cop_2, ol)
        if ol == 0:
            assert not all(pred.verify(c_cop_2) for pred in b_emu.required_predicates)

    circ = Circuit(2, 2).H(0).CX(0, 1).measure_all()
    b_emu.compile_circuit(circ)
    b_noi = AerBackend(noise_model=b_emu._noise_model)
    emu_shots = b_emu.get_shots(circ, 10, seed=10)
    aer_shots = b_noi.get_shots(circ, 10, seed=10)
    assert np.array_equal(emu_shots, aer_shots)


@given(
    n_shots=strategies.integers(min_value=1, max_value=10),
    n_bits=strategies.integers(min_value=0, max_value=10),
)
def test_shots_bits_edgecases(n_shots: int, n_bits: int) -> None:
    c = Circuit(n_bits, n_bits)
    aer_backend = AerBackend()

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = aer_backend.process_circuit(c, n_shots)
    res = aer_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    assert np.array_equal(aer_backend.get_shots(c, n_shots), correct_shots)
    assert aer_backend.get_shots(c, n_shots).shape == correct_shape
    assert aer_backend.get_counts(c, n_shots) == correct_counts


def test_simulation_method() -> None:
    state_backends = [AerBackend(), AerBackend(simulation_method="statevector")]
    stabilizer_backend = AerBackend(simulation_method="stabilizer")

    clifford_circ = Circuit(2).H(0).CX(0, 1).measure_all()
    clifford_T_circ = Circuit(2).H(0).T(1).CX(0, 1).measure_all()

    for b in state_backends + [stabilizer_backend]:
        counts = b.get_counts(clifford_circ, 4)
        assert sum(val for _, val in counts.items()) == 4

    for b in state_backends:
        counts = b.get_counts(clifford_T_circ, 4)
        assert sum(val for _, val in counts.items()) == 4

    with pytest.raises(AttributeError) as warninfo:
        # check for the error thrown when non-clifford circuit used with
        # stabilizer backend
        stabilizer_backend.get_counts(clifford_T_circ, 4)
        assert "Attribute header is not defined" in str(warninfo.value)


def test_aer_expanded_gates() -> None:
    c = Circuit(3).CX(0, 1)
    c.add_gate(OpType.ZZPhase, 0.1, [0, 1])
    c.add_gate(OpType.CY, [0, 1])
    c.add_gate(OpType.CCX, [0, 1, 2])

    backend = AerBackend()
    assert backend.valid_circuit(c)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_remote_simulator() -> None:
    remote_qasm = IBMQBackend(
        "ibmq_qasm_simulator", hub="ibm-q", group="open", project="main"
    )
    c = Circuit(3).CX(0, 1)
    c.add_gate(OpType.ZZPhase, 0.1, [0, 1])
    c.add_gate(OpType.CY, [0, 1])
    c.add_gate(OpType.CCX, [0, 1, 2])
    c.measure_all()

    assert remote_qasm.valid_circuit(c)

    assert sum(remote_qasm.get_counts(c, 10).values()) == 10


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_ibmq_mid_measure(santiago_backend: IBMQBackend) -> None:
    c = Circuit(3, 3).H(1).CX(1, 2).Measure(0, 0).Measure(1, 1)
    c.add_barrier([0, 1, 2])

    c.CX(1, 0).H(0).Measure(2, 2)

    b = santiago_backend
    ps = b.default_compilation_pass(0)
    ps.apply(c)
    # b.compile_circuit(c)
    assert not NoMidMeasurePredicate().verify(c)
    assert b.valid_circuit(c)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_compile_x(santiago_backend: IBMQBackend) -> None:
    # TKET-1028
    b = santiago_backend
    c = Circuit(1).X(0)
    for ol in range(3):
        c1 = c.copy()
        b.compile_circuit(c1, optimisation_level=ol)
        assert c1.n_gates == 1


def lift_perm(p: Dict[int, int]) -> np.ndarray:
    """
    Given a permutation of {0,1,...,n-1} return the 2^n by 2^n permuation matrix
    representing the permutation of qubits (big-endian convention).
    """
    n = len(p)
    pm = np.zeros((1 << n, 1 << n), dtype=complex)
    for i in range(1 << n):
        j = 0
        mask = 1 << n
        for q in range(n):
            mask >>= 1
            if (i & mask) != 0:
                j |= 1 << (n - 1 - p[q])
        pm[j][i] = 1
    return pm


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_compilation_correctness(santiago_backend: IBMQBackend) -> None:
    c = Circuit(5)
    c.H(0).H(1).H(2)
    c.CX(0, 1).CX(1, 2)
    c.Rx(0.25, 1).Ry(0.75, 1).Rz(0.5, 2)
    c.CCX(2, 1, 0)
    c.CY(1, 0).CY(2, 1)
    c.H(0).H(1).H(2)
    c.Rz(0.125, 0)
    c.X(1)
    c.Rz(0.125, 2).X(2).Rz(0.25, 2)
    c.SX(3).Rz(0.125, 3).SX(3)
    c.CX(0, 3).CX(0, 4)
    u_backend = AerUnitaryBackend()
    u = u_backend.get_unitary(c)
    ibm_backend = santiago_backend
    for ol in range(3):
        p = ibm_backend.default_compilation_pass(optimisation_level=ol)
        cu = CompilationUnit(c)
        p.apply(cu)
        c1 = cu.circuit
        compiled_u = u_backend.get_unitary(c1)

        # Adjust for placement
        imap = cu.initial_map
        fmap = cu.final_map
        c_idx = {c.qubits[i]: i for i in range(5)}
        c1_idx = {c1.qubits[i]: i for i in range(5)}
        ini = {c_idx[qb]: c1_idx[node] for qb, node in imap.items()}
        inv_fin = {c1_idx[node]: c_idx[qb] for qb, node in fmap.items()}
        m_ini = lift_perm(ini)
        m_inv_fin = lift_perm(inv_fin)

        assert compare_unitaries(u, m_inv_fin @ compiled_u @ m_ini)


# pytket-extensions issue #69
def test_symbolic_rebase() -> None:
    circ = QuantumCircuit(2)
    circ.rx(Parameter("a"), 0)
    circ.ry(Parameter("b"), 1)
    circ.cx(0, 1)

    pytket_circ = qiskit_to_tk(circ)

    # rebase pass could not handle symbolic parameters originally and would fail here:
    _rebase_pass.apply(pytket_circ)

    assert len(pytket_circ.free_symbols()) == 2


def _tk1_to_rotations(a: float, b: float, c: float) -> Circuit:
    """Translate tk1 to a RzRxRz so AerUnitaryBackend can simulate"""
    circ = Circuit(1)
    circ.Rz(c, 0).Rx(b, 0).Rz(a, 0)
    return circ


def _verify_single_q_rebase(
    backend: AerUnitaryBackend, a: float, b: float, c: float
) -> bool:
    """Compare the unitary of a tk1 gate to the unitary of the translated circuit"""
    rotation_circ = _tk1_to_rotations(a, b, c)
    u_before = backend.get_unitary(rotation_circ)
    circ = Circuit(1)
    circ.add_gate(OpType.TK1, [a, b, c], [0])
    _rebase_pass.apply(circ)
    u_after = backend.get_unitary(circ)
    return np.allclose(u_before, u_after)


def test_rebase_phase() -> None:
    backend = AerUnitaryBackend()
    for a in [0.6, 0, 1, 2, 3]:
        for b in [0.7, 0, 0.5, 1, 1.5]:
            for c in [0.8, 0, 1, 2, 3]:
                assert _verify_single_q_rebase(backend, a, b, c)
                assert _verify_single_q_rebase(backend, -a, -b, -c)
                assert _verify_single_q_rebase(backend, 2 * a, 3 * b, 4 * c)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_postprocess(santiago_backend: IBMQBackend) -> None:
    b = santiago_backend
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.SX(0).SX(1).CX(0, 1).measure_all()
    b.compile_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(json.loads(cast(str, h[2])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    b.cancel(h)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_postprocess_emu() -> None:
    b = IBMQEmulatorBackend("ibmq_santiago", hub="ibm-q", group="open", project="main")
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.SX(0).SX(1).CX(0, 1).measure_all()
    b.compile_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(json.loads(cast(str, h[2])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10
