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

import pytest
from pytket.circuit import Circuit, PauliExpBox, fresh_symbol  # type: ignore
from pytket.pauli import Pauli  # type: ignore
from pytket.extensions.qsharp import tk_to_qsharp
from sympy import Symbol  # type: ignore


def test_convert() -> None:
    c = Circuit(3)
    c.H(0)
    c.H(1)
    c.CX(1, 0)
    c.X(1)
    pbox = PauliExpBox([Pauli.X, Pauli.Z, Pauli.X], 0.25)
    c.add_pauliexpbox(pbox, [2, 0, 1])
    qs = tk_to_qsharp(c)
    assert "H(q[1]);" in qs


def test_convert_symbolic() -> None:
    c = Circuit(2)
    alpha = Symbol("alpha")
    c.Rx(alpha, 0)
    beta = fresh_symbol("alpha")
    c.Rz(beta * 2, 1)
    with pytest.raises(RuntimeError):
        qs = tk_to_qsharp(c)
    s_map = {alpha: 0.5, beta: 3.2}
    c.symbol_substitution(s_map)
    qs = tk_to_qsharp(c)
    assert "Rx" in qs
