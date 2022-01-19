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

from typing import Union
from qiskit.dagcircuit import DAGCircuit  # type: ignore
from qiskit.providers import BaseBackend, BackendV1  # type: ignore
from qiskit.transpiler.basepasses import TransformationPass, BasePass as qBasePass  # type: ignore
from qiskit.converters import circuit_to_dag, dag_to_circuit  # type: ignore
from qiskit.providers.aer.aerprovider import AerProvider  # type: ignore
from qiskit.providers.ibmq.accountprovider import AccountProvider  # type: ignore
from pytket.passes import BasePass  # type: ignore
from pytket.extensions.qiskit import (
    IBMQBackend,
    AerBackend,
    AerStateBackend,
    AerUnitaryBackend,
)
from .qiskit_convert import qiskit_to_tk, tk_to_qiskit


class TketPass(TransformationPass):
    """The tket compiler to be plugged in to the Qiskit compilation sequence"""

    def __init__(self, tket_pass: BasePass):
        """Wraps a pytket compiler pass as a
        :py:class:`qiskit.transpiler.TransformationPass`. A
        :py:class:`qiskit.dagcircuit.DAGCircuit` is converted to a pytket
        :py:class:`Circuit`. `tket_pass` will be run and the result is converted back.

        :param tket_pass: The pytket compiler pass to run
        :type tket_pass: BasePass
        """
        qBasePass.__init__(self)
        self._pass = tket_pass

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run a preconfigured optimisation pass on the circuit and route for the given
        backend.

        :param dag: The circuit to optimise and route

        :return: The modified circuit
        """
        qc = dag_to_circuit(dag)
        old_parameters = qc.parameters
        circ = qiskit_to_tk(qc)
        self._pass.apply(circ)
        qc = tk_to_qiskit(circ)
        new_param_lookup = {p._symbol_expr: p for p in qc.parameters}
        subs_map = {new_param_lookup[p._symbol_expr]: p for p in old_parameters}
        qc.assign_parameters(subs_map, inplace=True)
        newdag = circuit_to_dag(qc)
        newdag.name = dag.name
        return newdag


class TketAutoPass(TketPass):
    """The tket compiler to be plugged in to the Qiskit compilation sequence"""

    _aer_backend_map = {
        "aer_simulator": AerBackend,
        "aer_simulator_statevector": AerStateBackend,
        "aer_simulator_unitary": AerUnitaryBackend,
    }

    def __init__(
        self, backend: Union[BaseBackend, BackendV1], optimisation_level: int = 1
    ):
        """Identifies a Qiskit backend and provides the corresponding default
        compilation pass from pytket as a
        :py:class:`qiskit.transpiler.TransformationPass`.

        :param backend: The Qiskit backend to target. Accepts Aer or IBMQ backends.
        :param optimisation_level: The level of optimisation to perform during
            compilation. Level 0 just solves the device constraints without
            optimising. Level 1 additionally performs some light optimisations.
            Level 2 adds more intensive optimisations that can increase compilation
            time for large circuits. Defaults to 1.
        :type optimisation_level: int, optional
        """
        if isinstance(backend._provider, AerProvider):
            tk_backend = self._aer_backend_map[backend.name()]()
        elif isinstance(backend._provider, AccountProvider):
            tk_backend = IBMQBackend(backend.name())
        else:
            raise NotImplementedError("This backend provider is not supported.")
        super().__init__(tk_backend.default_compilation_pass(optimisation_level))
