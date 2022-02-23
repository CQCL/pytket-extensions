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

from collections import Counter
from typing import cast, Callable, Any  # pylint: disable=unused-import
from ast import literal_eval
import json
import os
from hypothesis import given, settings, strategies
import numpy as np
import pytest
import hypothesis.strategies as st
from hypothesis.strategies._internal import SearchStrategy

from pytket.circuit import (  # type: ignore
    Circuit,
    Qubit,
    Bit,
    OpType,
    reg_eq,
    reg_neq,
    reg_lt,
    reg_gt,
    reg_leq,
    reg_geq,
    if_not_bit,
)
from pytket.extensions.honeywell import HoneywellBackend
from pytket.extensions.honeywell.backends.honeywell import (
    GetResultFailed,
    _GATE_SET,
)
from pytket.extensions.honeywell import split_utf8
from pytket.extensions.honeywell.backends.api_wrappers import HQSAPIError, HoneywellQAPI
from pytket.backends.status import StatusEnum

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
REASON = (
    "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of Honeywell username)"
)


def test_honeywell() -> None:
    backend = HoneywellBackend(
        device_name="HQS-LT-S1-APIVAL", machine_debug=skip_remote_tests
    )
    c = Circuit(4, 4, "test 1")
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 2)
    c.CSWAP(0, 1, 2)
    c.CRz(0.4, 2, 3)
    c.CY(1, 3)
    c.ZZPhase(0.1, 2, 0)
    c.Tdg(3)
    c.measure_all()
    c = backend.get_compiled_circuit(c)
    n_shots = 4
    handle = backend.process_circuits([c], n_shots)[0]
    correct_shots = np.zeros((4, 4))
    correct_counts = {(0, 0, 0, 0): 4}
    res = backend.get_result(handle, timeout=49)
    shots = res.get_shots()
    counts = res.get_counts()
    assert backend.circuit_status(handle).status is StatusEnum.COMPLETED
    assert np.all(shots == correct_shots)
    assert counts == correct_counts
    res = backend.run_circuit(c, n_shots=4, timeout=49)
    newshots = res.get_shots()
    assert np.all(newshots == correct_shots)
    newcounts = res.get_counts()
    assert newcounts == correct_counts
    if skip_remote_tests:
        assert backend.backend_info is None


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_bell() -> None:
    b = HoneywellBackend(device_name="HQS-LT-S1-APIVAL")
    c = Circuit(2, 2, "test 2")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    shots = b.run_circuit(c, n_shots=n_shots).get_shots()
    print(shots)
    assert all(q[0] == q[1] for q in shots)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_multireg() -> None:
    b = HoneywellBackend(device_name="HQS-LT-S1-APIVAL", label="test 3")
    c = Circuit()
    q1 = Qubit("q1", 0)
    q2 = Qubit("q2", 0)
    c1 = Bit("c1", 0)
    c2 = Bit("c2", 0)
    for q in (q1, q2):
        c.add_qubit(q)
    for cb in (c1, c2):
        c.add_bit(cb)
    c.H(q1)
    c.CX(q1, q2)
    c.Measure(q1, c1)
    c.Measure(q2, c2)
    c = b.get_compiled_circuit(c)

    n_shots = 10
    shots = b.run_circuit(c, n_shots=n_shots).get_shots()
    assert np.array_equal(shots, np.zeros((10, 2)))


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_default_pass() -> None:
    b = HoneywellBackend(device_name="HQS-LT-S1-APIVAL")
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
def test_cancel() -> None:
    b = HoneywellBackend(device_name="HQS-LT-S1-APIVAL", label="test cancel")
    c = Circuit(2, 2).H(0).CX(0, 1).measure_all()
    c = b.get_compiled_circuit(c)
    handle = b.process_circuit(c, 10)
    try:
        # will raise HTTP error if job is already completed
        b.cancel(handle)
    except HQSAPIError as err:
        check_completed = "job has completed already" in str(err)
        assert check_completed
        if not check_completed:
            raise err

    print(b.circuit_status(handle))


@st.composite
def circuits(
    draw: Callable[[SearchStrategy[Any]], Any],
    n_qubits: SearchStrategy[int] = st.integers(min_value=2, max_value=6),
    depth: SearchStrategy[int] = st.integers(min_value=1, max_value=100),
) -> Circuit:
    total_qubits = draw(n_qubits)
    circuit = Circuit(total_qubits, total_qubits)
    for _ in range(draw(depth)):
        gate = draw(st.sampled_from(list(_GATE_SET)))
        control = draw(st.integers(min_value=0, max_value=total_qubits - 1))
        if gate == OpType.ZZMax:
            target = draw(
                st.integers(min_value=0, max_value=total_qubits - 1).filter(
                    lambda x: x != control
                )
            )
            circuit.add_gate(gate, [control, target])
        elif gate == OpType.Measure:
            circuit.add_gate(gate, [control, control])
            circuit.add_gate(OpType.Reset, [control])
        elif gate == OpType.Rz:
            param = draw(st.floats(min_value=0, max_value=2))
            circuit.add_gate(gate, [param], [control])
        elif gate == OpType.PhasedX:
            param1 = draw(st.floats(min_value=0, max_value=2))
            param2 = draw(st.floats(min_value=0, max_value=2))
            circuit.add_gate(gate, [param1, param2], [control])
    circuit.measure_all()

    return circuit


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@given(
    circuits(),  # pylint: disable=no-value-for-parameter
    st.integers(min_value=1, max_value=10000),
)
@settings(max_examples=5, deadline=None)
def test_cost_estimate(c: Circuit, n_shots: int) -> None:

    b = HoneywellBackend("HQS-LT-S1-APIVAL")
    c = b.get_compiled_circuit(c)
    estimate = b.cost_estimate(c, n_shots)
    status = b.circuit_status(b.process_circuit(c, n_shots))
    status_msg = status.message
    message_dict = literal_eval(status_msg)
    if message_dict["cost"] is not None:
        api_cost = float(message_dict["cost"])
        assert estimate == api_cost


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_classical() -> None:
    # circuit to cover capabilities covered in HQS example notebook
    c = Circuit(1)
    a = c.add_c_register("a", 8)
    b = c.add_c_register("b", 10)
    c.add_c_register("c", 10)

    c.add_c_setbits([True], [a[0]])
    c.add_c_setbits([False, True] + [False] * 6, list(a))
    c.add_c_setbits([True, True] + [False] * 8, list(b))

    c.X(0, condition=reg_eq(a ^ b, 1))
    c.X(0, condition=(a[0] ^ b[0]))
    c.X(0, condition=reg_eq(a & b, 1))
    c.X(0, condition=reg_eq(a | b, 1))

    c.X(0, condition=a[0])
    c.X(0, condition=reg_neq(a, 1))
    c.X(0, condition=if_not_bit(a[0]))
    c.X(0, condition=reg_gt(a, 1))
    c.X(0, condition=reg_lt(a, 1))
    c.X(0, condition=reg_geq(a, 1))
    c.X(0, condition=reg_leq(a, 1))

    b = HoneywellBackend("HQS-LT-S1-APIVAL")

    c = b.get_compiled_circuit(c)
    assert b.run_circuit(c, n_shots=10).get_counts()


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_postprocess() -> None:
    b = HoneywellBackend("HQS-LT-S1-APIVAL")
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.add_gate(OpType.PhasedX, [1, 1], [0])
    c.add_gate(OpType.PhasedX, [1, 1], [1])
    c.add_gate(OpType.ZZMax, [0, 1])
    c.measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(json.loads(cast(str, h[1])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


@given(
    n_shots=strategies.integers(min_value=1, max_value=10),  # type: ignore
    n_bits=strategies.integers(min_value=0, max_value=10),
)
def test_shots_bits_edgecases(n_shots, n_bits) -> None:

    honeywell_backend = HoneywellBackend("HQS-LT-S1-APIVAL", machine_debug=True)
    c = Circuit(n_bits, n_bits)

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = honeywell_backend.process_circuit(c, n_shots)
    res = honeywell_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    res = honeywell_backend.run_circuit(c, n_shots=n_shots)
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts


@given(
    utf_str=st.text(min_size=0, max_size=3000),
    chunksize=st.integers(min_value=4, max_value=1030),
)
def test_split_utf8(utf_str: str, chunksize: int) -> None:
    split = list(split_utf8(utf_str, chunksize))
    assert max(len(i) for i in split) <= chunksize
    assert "".join(split) == utf_str


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_simulator() -> None:
    circ = Circuit(2, name="sim_test").H(0).CX(0, 1).measure_all()
    n_shots = 1000
    state_backend = HoneywellBackend("HQS-LT-S1-SIM")
    stabilizer_backend = HoneywellBackend("HQS-LT-S1-SIM", simulator="stabilizer")

    circ = state_backend.get_compiled_circuit(circ)

    noisy_handle = state_backend.process_circuit(circ, n_shots)
    pure_handle = state_backend.process_circuit(circ, n_shots, noisy_simulation=False)
    stab_handle = stabilizer_backend.process_circuit(
        circ, n_shots, noisy_simulation=False
    )

    noisy_counts = state_backend.get_result(noisy_handle).get_counts()
    assert sum(noisy_counts.values()) == n_shots
    assert len(noisy_counts) > 2  # some noisy results likely

    pure_counts = state_backend.get_result(pure_handle).get_counts()
    assert sum(pure_counts.values()) == n_shots
    assert len(pure_counts) == 2

    stab_counts = stabilizer_backend.get_result(stab_handle).get_counts()
    assert sum(stab_counts.values()) == n_shots
    assert len(stab_counts) == 2

    # test non-clifford circuit fails on stabilizer backend
    # unfortunately the job is accepted, then fails, so have to check get_result
    non_stab_circ = (
        Circuit(2, name="non_stab_circ").H(0).Rz(0.1, 0).CX(0, 1).measure_all()
    )
    non_stab_circ = stabilizer_backend.get_compiled_circuit(non_stab_circ)
    broken_handle = stabilizer_backend.process_circuit(non_stab_circ, n_shots)

    with pytest.raises(GetResultFailed) as _:
        _ = stabilizer_backend.get_result(broken_handle)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_retrieve_available_devices() -> None:
    backend_infos = HoneywellBackend.available_devices()
    assert len(backend_infos) > 0
    api_handler = HoneywellQAPI()
    backend_infos = HoneywellBackend.available_devices(api_handler=api_handler)
    assert len(backend_infos) > 0


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_batching() -> None:
    circ = Circuit(2, name="batching_test").H(0).CX(0, 1).measure_all()
    state_backend = HoneywellBackend("HQS-LT-S1-SIM")
    circ = state_backend.get_compiled_circuit(circ)

    handles = state_backend.process_circuits([circ, circ], 10)
    assert state_backend.get_results(handles)


# hard to run as it involves removing credentials
# def test_delete_authentication():
#     print("first login")
#     b = HoneywellBackend()
#     print("delete login")


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_submission_with_group() -> None:
    b = HoneywellBackend(device_name="HQS-LT-S1-APIVAL")
    c = Circuit(2, 2, "test 2")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    shots = b.run_circuit(c, n_shots=n_shots, group="DEFAULT").get_shots()  # type: ignore
    print(shots)
    assert all(q[0] == q[1] for q in shots)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_zzphase() -> None:
    backend = HoneywellBackend(
        device_name="HQS-LT-S1-APIVAL", machine_debug=skip_remote_tests
    )
    c = Circuit(2, 2, "test rzz")
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 0)
    c.CY(0, 1)
    c.ZZPhase(0.1, 1, 0)
    c.measure_all()
    c0 = backend.get_compiled_circuit(c, 0)

    assert c0.n_gates_of_type(OpType.ZZPhase) > 0

    # simulator does not yet support ZZPhase
    backsim = HoneywellBackend(
        device_name="HQS-LT-S1-SIM", machine_debug=skip_remote_tests
    )
    assert backsim.get_compiled_circuit(c, 0).n_gates_of_type(OpType.ZZPhase) == 0

    n_shots = 4
    handle = backend.process_circuits([c0], n_shots)[0]
    correct_counts = {(0, 0): 4}
    res = backend.get_result(handle, timeout=49)
    counts = res.get_counts()
    assert counts == correct_counts
