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

"""Shared utility methods for cirq backends.
"""

from typing import Tuple, List, cast
from cirq.circuits import Circuit as CirqCircuit
from cirq.ops import QubitOrder, MeasurementGate
from cirq.protocols import is_measurement
from pytket.circuit import Circuit, Qubit, Bit  # type: ignore


def _get_default_uids(
    cirq_circuit: CirqCircuit, tket_circuit: Circuit
) -> Tuple[List[Bit], List[Qubit]]:

    if len(tket_circuit.qubit_readout) == 0:
        return [], tket_circuit.qubits
    else:
        ordered_cirq_qubits = QubitOrder.as_qubit_order(QubitOrder.DEFAULT).order_for(
            cirq_circuit.all_qubits()
        )

        cirq_measures = [c[1] for c in cirq_circuit.findall_operations(is_measurement)]

        tket_bit_to_qubit_map = {b: q for q, b in tket_circuit.qubit_to_bit_map.items()}

        ordered_tket_bits = []
        ordered_tket_qubits = []
        for cirq_qubit in ordered_cirq_qubits:
            for cirq_measure in cirq_measures:
                if len(cirq_measure.qubits) > 1:
                    raise ValueError(
                        "Cirq Qubit measurement assigned to multiple classical bits."
                    )
                if cirq_measure.qubits[0] == cirq_qubit:
                    for tket_bit, tket_qubit in tket_bit_to_qubit_map.items():
                        if cast(MeasurementGate, cirq_measure.gate).key == str(
                            tket_bit
                        ):
                            ordered_tket_bits.append(tket_bit)
                            ordered_tket_qubits.append(tket_qubit)

        return (ordered_tket_bits, ordered_tket_qubits)
