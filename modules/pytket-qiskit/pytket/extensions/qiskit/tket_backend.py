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

from typing import Optional, List, Union, Any
from qiskit.circuit.quantumcircuit import QuantumCircuit  # type: ignore
from qiskit.providers.backend import BackendV1 as QiskitBackend  # type: ignore
from qiskit.providers.models import QasmBackendConfiguration  # type: ignore
from qiskit.providers import Options  # type: ignore
from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk, _gate_str_2_optype_rev
from pytket.extensions.qiskit.tket_job import TketJob, JobInfo
from pytket.backends import Backend
from pytket.passes import BasePass  # type: ignore
from pytket.predicates import (  # type: ignore
    NoClassicalControlPredicate,
    GateSetPredicate,
    CompilationUnit,
)


def _extract_basis_gates(backend: Backend) -> List[str]:
    for pred in backend.required_predicates:
        if type(pred) == GateSetPredicate:
            return [
                _gate_str_2_optype_rev[optype]
                for optype in pred.gate_set
                if optype in _gate_str_2_optype_rev.keys()
            ]
    return []


class TketBackend(QiskitBackend):
    """Wraps a :py:class:`Backend` as a :py:class:`qiskit.providers.BaseBackend` for use
    within the Qiskit software stack.

    Each :py:class:`qiskit.circuit.quantumcircuit.QuantumCircuit` passed in will be
    converted to a :py:class:`Circuit` object. If a :py:class:`BasePass` is provided for
    ``comp_pass``, this is applied to the :py:class:`Circuit`. Then it is processed by
    the :py:class:`Backend`, wrapping the :py:class:`ResultHandle` s in a
    :py:class:`TketJob`, retrieving the results when called on the job object. The
    required predicates of the :py:class:`Backend` are presented to the Qiskit
    transpiler to enable it to perform the compilation in many cases. This may not
    always be possible due to unsupported gatesets or additional constraints that cannot
    be captured in Qiskit's transpiler, in which case a custom
    :py:class:`qiskit.transpiler.TranspilationPass` should be used to map into a tket-
    compatible gateset and set ``comp_pass`` to compile for the backend. To compile with
    tket only, set ``comp_pass`` and just use Qiskit to map into a tket-compatible
    gateset. In Qiskit Aqua, you should wrap the :py:class:`TketBackend` in a
    :py:class:`qiskit.aqua.QuantumInstance`, providing a custom
    :py:class:`qiskit.transpiler.PassManager` with a
    :py:class:`qiskit.transpiler.passes.Unroller`. For examples, see the `user manual
    <https://cqcl.github.io/pytket/manual/manual_backend.html#embedding-into-
    qiskit>`_ or the `Qiskit integration example <ht
    tps://github.com/CQCL/pytket/blob/main/examples/qiskit_integration. ipynb>`_.
    """

    def __init__(self, backend: Backend, comp_pass: Optional[BasePass] = None):
        """Create a new :py:class:`TketBackend` from a :py:class:`Backend`.

        :param backend: The device or simulator to wrap up
        :type backend: Backend
        :param comp_pass: The (optional) tket compilation pass to apply to each circuit
         before submitting to the :py:class:`Backend`, defaults to None
        :type comp_pass: Optional[BasePass], optional
        """
        arch = backend.backend_info.architecture if backend.backend_info else None
        config = QasmBackendConfiguration(
            backend_name=("statevector_" if backend.supports_state else "")
            + "pytket/"
            + str(type(backend)),
            backend_version="0.0.1",
            n_qubits=len(arch.nodes) if arch and arch.nodes else 40,
            basis_gates=_extract_basis_gates(backend),
            gates=[],
            local=False,
            simulator=False,
            conditional=not any(
                (
                    type(pred) == NoClassicalControlPredicate
                    for pred in backend.required_predicates
                )
            ),
            open_pulse=False,
            memory=backend.supports_shots,
            max_shots=10000,
            coupling_map=[[n.index[0], m.index[0]] for n, m in arch.coupling]
            if arch
            else None,
            max_experiments=10000,
        )
        super().__init__(configuration=config, provider=None)
        self._backend = backend
        self._comp_pass = comp_pass

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=None, memory=False)

    def run(
        self, run_input: Union[QuantumCircuit, List[QuantumCircuit]], **options: Any
    ) -> TketJob:
        if isinstance(run_input, QuantumCircuit):
            run_input = [run_input]
        circ_list = [qiskit_to_tk(qc) for qc in run_input]
        n_shots = options.get("shots", None)
        jobinfos = [JobInfo(circ.qubits, circ.bits, n_shots) for circ in circ_list]
        if self._comp_pass:
            final_maps = []
            compiled_list = []
            for c in circ_list:
                cu = CompilationUnit(c)
                self._comp_pass.apply(cu)
                compiled_list.append(cu.circuit)
                final_maps.append(cu.final_map)
            circ_list = compiled_list
        else:
            final_maps = [None] * len(circ_list)
        handles = self._backend.process_circuits(circ_list, n_shots=n_shots)

        return TketJob(self, handles, jobinfos, final_maps)
