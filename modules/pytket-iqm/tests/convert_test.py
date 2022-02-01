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

from typing import List
import os
import numpy as np
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.extensions.iqm.backends.iqm import _translate_iqm, IQMBackend, _iqm_rebase

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of IQM credentials)"


def tk_to_iqm(circ: Circuit) -> List:
    """Convert a circuit to IQM list representation"""
    c = circ.copy()
    IQMBackend().default_compilation_pass().apply(c)
    return _translate_iqm(c)


def test_rebase_CX() -> None:
    circ = Circuit(2)
    circ.CX(0, 1)
    orig_circ = circ.copy()

    _iqm_rebase().apply(circ)

    u1 = orig_circ.get_unitary()
    u2 = circ.get_unitary()

    assert np.allclose(u1, u2)


def test_rebase_singleq() -> None:
    circ = Circuit(1)
    # some arbitrary unitary
    circ.add_gate(OpType.U3, [0.2, 0.3, 0.45], [0])
    orig_circ = circ.copy()

    _iqm_rebase().apply(circ)

    u1 = orig_circ.get_unitary()
    u2 = circ.get_unitary()

    assert np.allclose(u1, u2)


def test_rebase_large() -> None:
    circ = Circuit(3)
    # some arbitrary unitary
    circ.Rx(0.2, 0).Rz(0.1, 1).Rz(8.2, 2).X(2).CX(0, 1).CX(1, 2).Rz(0.4, 1).Rx(0.7, 0)
    orig_circ = circ.copy()

    _iqm_rebase().apply(circ)

    u1 = orig_circ.get_unitary()
    u2 = circ.get_unitary()

    assert np.allclose(u1, u2)
