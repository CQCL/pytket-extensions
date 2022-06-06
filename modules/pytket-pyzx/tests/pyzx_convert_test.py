# Copyright 2019-2022 Cambridge Quantum Computing
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

from pytket.extensions.pyzx import (
    tk_to_pyzx,
    pyzx_to_tk,
    tk_to_pyzx_arc,
    pyzx_to_tk_arc,
    tk_to_pyzx_placed_circ,
    pyzx_to_tk_placed_circ,
)
from pytket.circuit import Circuit, fresh_symbol  # type: ignore
from pytket.architecture import Architecture  # type: ignore
from pytket.passes import AASRouting, CXMappingPass  # type: ignore
from pytket.placement import GraphPlacement  # type: ignore
import numpy as np
import pytest


@pytest.mark.filterwarnings("ignore:strict=False")
def test_statevector() -> None:
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
    state = circ.get_statevector()
    circ2 = pyzx_to_tk(zxcirc)
    assert circ2.name == circ.name
    state2 = circ2.get_statevector()
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


@pytest.mark.filterwarnings("ignore:strict=False")
def test_arc_conversion() -> None:
    arc = Architecture([[0, 1], [1, 2], [2, 3], [3, 4]])
    arc_pyzx = tk_to_pyzx_arc(arc)
    arc_2 = pyzx_to_tk_arc(arc_pyzx)
    assert arc == arc_2
    arc_pyzx_2 = tk_to_pyzx_arc(arc_2)
    assert list(arc_pyzx.graph.edges()) == list(arc_pyzx_2.graph.edges())
    assert list(arc_pyzx.graph.vertices()) == list(arc_pyzx_2.graph.vertices())


@pytest.mark.filterwarnings("ignore:strict=False")
def test_placed_circ_tests() -> None:
    arc = Architecture([[0, 2], [1, 2], [2, 3], [3, 4]])
    initial_circ = Circuit(5)
    initial_circ.H(0)
    initial_circ.H(1)
    initial_circ.H(2)
    initial_circ.H(3)
    initial_circ.H(4)
    initial_circ.CX(3, 4)
    c = initial_circ.copy()

    aas_pass = AASRouting(arc)

    aas_pass.apply(c)

    _, pyzx_circ, inv_map = tk_to_pyzx_placed_circ(c, arc)

    pytket_circ_2 = pyzx_to_tk_placed_circ(pyzx_circ, inv_map)

    assert pytket_circ_2.qubits == c.qubits

    state = c.get_statevector()
    state2 = pytket_circ_2.get_statevector()
    assert np.allclose(state, state2, atol=1e-10)


@pytest.mark.filterwarnings("ignore:strict=False")
def test_placed_circ_tests_2() -> None:
    arc = Architecture([[0, 2], [1, 2], [2, 3], [3, 4]])
    initial_circ = Circuit(5)
    initial_circ.H(0)
    initial_circ.H(1)
    initial_circ.H(2)
    initial_circ.H(3)
    initial_circ.H(4)
    initial_circ.CX(1, 4)
    initial_circ.CX(4, 2)
    initial_circ.CX(0, 4)
    initial_circ.CX(1, 4)

    c = initial_circ.copy()

    g_place = GraphPlacement(arc)

    cx_pass = CXMappingPass(arc, g_place)

    cx_pass.apply(c)

    _, pyzx_circ, inv_map = tk_to_pyzx_placed_circ(c, arc)

    pytket_circ_2 = pyzx_to_tk_placed_circ(pyzx_circ, inv_map)

    assert pytket_circ_2.qubits == c.qubits

    state = c.get_statevector()
    state2 = pytket_circ_2.get_statevector()
    assert np.allclose(state, state2, atol=1e-10)
