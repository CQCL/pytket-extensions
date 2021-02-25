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

from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.extensions.ionq.ionq_convert import tk_to_ionq


def test_convert() -> None:
    circ = Circuit(3, 3)
    circ.H(0).CX(0, 1)
    circ.add_gate(OpType.XXPhase, 0.12, [1, 2])
    circ.add_gate(OpType.noop, [1])
    circ.add_barrier([2])
    circ.Measure(0, 1)
    circ.Measure(1, 0)
    circ.Measure(2, 2)

    (circ_ionq, measures) = tk_to_ionq(circ)
    assert measures[0] == 1
    assert measures[1] == 0
    assert measures[2] == 2

    assert circ_ionq["qubits"] == 3
    assert circ_ionq["circuit"][0]["gate"] == "h"
    assert circ_ionq["circuit"][1]["gate"] == "cnot"
    assert circ_ionq["circuit"][2]["gate"] == "xx"
