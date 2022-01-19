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

from typing import Tuple, List
import json
import os
import numpy as np
import pytest
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.extensions.aqt.backends.aqt import _translate_aqt, AQTBackend, _aqt_rebase

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of AQT access token)"


def tk_to_aqt(circ: Circuit) -> Tuple[List[List], str]:
    """Convert a circuit to AQT list representation"""
    c = circ.copy()
    AQTBackend(device_name="sim/noise-model-1").default_compilation_pass().apply(c)
    return _translate_aqt(c)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_convert() -> None:
    circ = Circuit(4, 4)
    circ.H(0).CX(0, 1)
    circ.add_gate(OpType.noop, [1])
    circ.CRz(0.5, 1, 2)
    circ.add_barrier([2])
    circ.ZZPhase(0.3, 2, 3).CX(3, 0).Tdg(1)
    circ.Measure(0, 0)
    circ.Measure(1, 2)
    circ.Measure(2, 3)
    circ.Measure(3, 1)

    circ_aqt = tk_to_aqt(circ)
    assert json.loads(circ_aqt[1]) == [0, 3, 1, 2]
    assert all(gate[0] in ["X", "Y", "MS"] for gate in circ_aqt[0])


def test_rebase_CX() -> None:
    circ = Circuit(2)
    circ.CX(0, 1)
    orig_circ = circ.copy()

    _aqt_rebase().apply(circ)

    u1 = orig_circ.get_unitary()
    u2 = circ.get_unitary()

    assert np.allclose(u1, u2)


def test_rebase_singleq() -> None:
    circ = Circuit(1)
    # some arbitrary unitary
    circ.add_gate(OpType.U3, [0.01231, 0.848, 38.200], [0])
    orig_circ = circ.copy()

    _aqt_rebase().apply(circ)

    u1 = orig_circ.get_unitary()
    u2 = circ.get_unitary()

    assert np.allclose(u1, u2)


def test_rebase_large() -> None:
    circ = Circuit(3)
    # some arbitrary unitary
    circ.Rx(0.21, 0).Rz(0.12, 1).Rz(8.2, 2).X(2).CX(0, 1).CX(1, 2).Rz(0.44, 1).Rx(
        0.43, 0
    )
    orig_circ = circ.copy()

    _aqt_rebase().apply(circ)

    u1 = orig_circ.get_unitary()
    u2 = circ.get_unitary()

    assert np.allclose(u1, u2)
