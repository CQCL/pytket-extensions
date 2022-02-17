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

import json
from typing import (
    cast,
    Dict,
    Optional,
    List,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from pytket.backends import Backend, CircuitNotRunError, ResultHandle
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit  # type: ignore
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit
from pytket.extensions.qiskit.result_convert import (
    qiskit_experimentresult_to_backendresult,
)
from pytket.utils import prepare_circuit
from pytket.utils.results import KwargTypes

from qiskit.providers.aer import AerSimulator  # type: ignore
from qiskit.providers.aer.noise.noise_model import NoiseModel  # type: ignore

from .aer import AerBackend
from .ibm import IBMQBackend
from .ibm_utils import _batch_circuits

if TYPE_CHECKING:
    from pytket.predicates import Predicate  # type: ignore
    from pytket.passes import BasePass  # type: ignore
    from qiskit.providers.aer import AerJob  # type: ignore
    from qiskit.providers.ibmq import AccountProvider  # type: ignore
    from qiskit.result.models import ExperimentResult  # type: ignore


class IBMQEmulatorBackend(AerBackend):
    """A backend which uses the AerBackend to emulate the behaviour of IBMQBackend.
    Attempts to perform the same compilation and predicate checks as IBMQBackend.
    Requires a valid IBMQ account.

    """

    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = False
    _supports_expectation = False

    def __init__(
        self,
        backend_name: str,
        hub: Optional[str] = None,
        group: Optional[str] = None,
        project: Optional[str] = None,
        account_provider: Optional["AccountProvider"] = None,
    ):
        """Construct an IBMQEmulatorBackend. Identical to :py:class:`IBMQBackend`
        constructor, except there is no `monitor` parameter. See :py:class:`IBMQBackend`
        docs for more details.
        """

        self._ibmq = IBMQBackend(
            backend_name=backend_name,
            hub=hub,
            group=group,
            project=project,
            account_provider=account_provider,
        )
        aer_sim = AerSimulator.from_backend(self._ibmq._backend)
        super().__init__(noise_model=NoiseModel.from_backend(aer_sim))
        self._backend = aer_sim

        # cache of results keyed by job id and circuit index
        self._ibm_res_cache: Dict[Tuple[str, int], ExperimentResult] = dict()

    @property
    def backend_info(self) -> BackendInfo:
        return self._ibmq.backend_info

    def rebase_pass(self) -> "BasePass":
        return self._ibmq.rebase_pass()

    @property
    def required_predicates(self) -> List["Predicate"]:
        return list(self._ibmq.required_predicates)

    def default_compilation_pass(self, optimisation_level: int = 1) -> "BasePass":
        return self._ibmq.default_compilation_pass(optimisation_level)

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, int, str)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: `seed`, `postprocess`.
        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=True,
        )

        if valid_check:
            self._check_all_circuits(circuits)

        postprocess = kwargs.get("postprocess", False)
        seed = cast(Optional[int], kwargs.get("seed"))

        handle_list: List[Optional[ResultHandle]] = [None] * len(circuits)
        circuit_batches, batch_order = _batch_circuits(circuits, n_shots_list)

        for (n_shots, batch), indices in zip(circuit_batches, batch_order):
            qcs, ppcirc_strs = [], []
            for tkc in batch:
                if postprocess:
                    c0, ppcirc = prepare_circuit(tkc, allow_classical=False)
                    ppcirc_rep = ppcirc.to_dict()
                else:
                    c0, ppcirc_rep = tkc, None
                qcs.append(tk_to_qiskit(c0))
                ppcirc_strs.append(json.dumps(ppcirc_rep))
            job = self._backend.run(
                qcs,
                shots=n_shots,
                memory=self._memory,
                seed_simulator=seed,
                noise_model=self._noise_model,
            )
            jobid = job.job_id()
            for i, ind in enumerate(indices):
                handle = ResultHandle(jobid, i, ppcirc_strs[i])
                handle_list[ind] = handle
                self._cache[handle] = {"job": job}
        return cast(List[ResultHandle], handle_list)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: none.
        """
        self._check_handle_type(handle)
        if handle in self._cache and "result" in self._cache[handle]:
            return cast(BackendResult, self._cache[handle]["result"])
        jobid, index, ppcirc_str = handle
        ppcirc_rep = json.loads(ppcirc_str)
        ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
        cache_key = (jobid, index)
        if cache_key not in self._ibm_res_cache:
            try:
                job: "AerJob" = self._cache[handle]["job"]
            except KeyError:
                raise CircuitNotRunError(handle)
            res = job.result()
            for circ_index, r in enumerate(res.results):
                self._ibm_res_cache[(jobid, circ_index)] = r
        result = qiskit_experimentresult_to_backendresult(
            self._ibm_res_cache[cache_key], ppcirc
        )
        self._cache[handle] = {"result": result}
        return result
