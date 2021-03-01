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

from pytket.extensions.pyzx import tk_to_pyzx, pyzx_to_tk
from pytket.circuit import Circuit, fresh_symbol  # type: ignore

# Temporary fix before pytket_qiskit 0.7.2 release
try:
    from pytket.extensions.qiskit import AerStateBackend  # type: ignore
except ImportError:
    from pytket.extensions.backends.qiskit import AerStateBackend

import numpy as np  # type: ignore
import pytest


@pytest.mark.filterwarnings("ignore:strict=False")
def test_statevector() -> None:
    b = AerStateBackend()
    circ = Circuit(3, name="test")
    circ.H(2)
    circ.X(0)
    circ.H(0)
    circ.CX(0, 1)
    circ.CZ(1, 2)
    circ.Sdg(0)
    circ.Tdg(1)
    circ.Z(1)
    circ.T(2)
    circ.Rx(0.3333, 1)
    circ.Rz(0.3333, 1)
    zxcirc = tk_to_pyzx(circ)
    assert zxcirc.name == circ.name
    b.compile_circuit(circ)
    state = b.get_state(circ)
    circ2 = pyzx_to_tk(zxcirc)
    assert circ2.name == circ.name
    b.compile_circuit(circ2)
    state2 = b.get_state(circ2)
    assert np.allclose(state, state2, atol=1e-10)


@pytest.mark.filterwarnings("ignore:strict=False")
def test_sym_parameterised() -> None:
    circ = Circuit(3, name="test")
    circ.Z(1)
    alpha = fresh_symbol("alpha")
    circ.Rx(alpha, 0)
    with pytest.raises(Exception) as excinfo:
        _ = tk_to_pyzx(circ)
        assert "as it contains symbolic parameters." in str(excinfo.value)


@pytest.mark.filterwarnings("ignore:strict=False")
def test_invalid_gate() -> None:
    circ = Circuit(1, name="test")
    circ.measure_all()
    with pytest.raises(Exception) as excinfo:
        _ = tk_to_pyzx(circ)
        assert "as the gate type is unrecognised." in str(excinfo.value)
