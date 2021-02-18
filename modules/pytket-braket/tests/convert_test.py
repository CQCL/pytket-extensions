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

from pytket.extensions.braket import braket_to_tk, tk_to_braket
from pytket.circuit import Circuit, OpType  # type: ignore


def test_convert() -> None:
    c = Circuit(3)
    c.add_gate(OpType.CCX, [0, 1, 2])
    c.add_gate(OpType.CX, [0, 1])
    c.add_gate(OpType.CU1, 0.1, [0, 1])
    c.add_gate(OpType.CSWAP, [0, 1, 2])
    c.add_gate(OpType.CY, [0, 1])
    c.add_gate(OpType.CZ, [0, 1])
    c.add_gate(OpType.H, [0])
    c.add_gate(OpType.ISWAPMax, [0, 1])
    c.add_gate(OpType.U1, 0.2, [0])
    c.add_gate(OpType.Rx, 0.3, [0])
    c.add_gate(OpType.Ry, 0.4, [0])
    c.add_gate(OpType.Rz, 0.5, [0])
    c.add_gate(OpType.S, [0])
    c.add_gate(OpType.Sdg, [0])
    c.add_gate(OpType.SWAP, [0, 1])
    c.add_gate(OpType.T, [0])
    c.add_gate(OpType.Tdg, [0])
    c.add_gate(OpType.V, [0])
    c.add_gate(OpType.Vdg, [0])
    c.add_gate(OpType.X, [0])
    c.add_gate(OpType.XXPhase, 0.6, [0, 1])
    c.add_gate(OpType.ISWAP, 0.7, [0, 1])
    c.add_gate(OpType.Y, [0])
    c.add_gate(OpType.YYPhase, 0.8, [0, 1])
    c.add_gate(OpType.Z, [0])
    c.add_gate(OpType.ZZPhase, 0.9, [0, 1])
    bkc = tk_to_braket(c)
    c1 = braket_to_tk(bkc)
    assert c.get_commands() == c1.get_commands()


def test_convert_qubit_order() -> None:
    c = Circuit(3).H(2).CX(2, 1)
    bkc = tk_to_braket(c)
    c1 = braket_to_tk(bkc)
    assert c.get_commands() == c1.get_commands()
