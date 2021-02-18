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
from pytket.passes import RebaseHQS  # type: ignore
from pytket.qasm import circuit_to_qasm_str


def test_convert() -> None:
    circ = Circuit(4)
    circ.H(0).CX(0, 1)
    circ.add_gate(OpType.noop, [1])
    circ.CRz(0.5, 1, 2)
    circ.add_barrier([2])
    circ.ZZPhase(0.3, 2, 3).CX(3, 0).Tdg(1)
    circ.measure_all()

    RebaseHQS().apply(circ)
    circ_hqs = circuit_to_qasm_str(circ, header="hqslib1")
    qasm_str = circ_hqs.split("\n")[6:-1]
    test = True
    for com in qasm_str:
        test &= any(
            com.startswith(gate) for gate in ("rz", "U1q", "ZZ", "measure", "barrier")
        )
    assert test
