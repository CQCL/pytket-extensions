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

import itertools
import logging
from ast import literal_eval
import json
from typing import (
    cast,
    List,
    Optional,
    Dict,
    Any,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
)
from warnings import warn

import qiskit  # type: ignore
from qiskit import IBMQ
from qiskit.qobj import QobjExperimentHeader  # type: ignore
from qiskit.providers.ibmq.exceptions import IBMQBackendApiError  # type: ignore
from qiskit.providers.ibmq.job import IBMQJob  # type: ignore
from qiskit.result import Result, models  # type: ignore
from qiskit.tools.monitor import job_monitor  # type: ignore
from qiskit.providers.ibmq.utils.utils import api_status_to_job_status  # type: ignore

from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.backends import Backend, CircuitNotRunError, CircuitStatus, ResultHandle
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.extensions.qiskit.qiskit_convert import (
    process_characterisation,
    get_avg_characterisation,
)
from pytket.extensions.qiskit._metadata import __extension_version__
from pytket.passes import (  # type: ignore
    BasePass,
    auto_rebase_pass,
    RemoveRedundancies,
    SequencePass,
    SynthesiseTket,
    CXMappingPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    CliffordSimp,
    SimplifyInitial,
    NaivePlacementPass,
)
from pytket.predicates import (  # type: ignore
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    Predicate,
)
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit, _tk_gate_set
from pytket.extensions.qiskit.result_convert import (
    qiskit_experimentresult_to_backendresult,
)
from pytket.architecture import FullyConnected  # type: ignore
from pytket.placement import NoiseAwarePlacement  # type: ignore
from pytket.utils import prepare_circuit
from pytket.utils.results import KwargTypes
from .ibm_utils import _STATUS_MAP, _batch_circuits
from .config import QiskitConfig

if TYPE_CHECKING:
    from qiskit.providers.ibmq import (  # type: ignore
        IBMQBackend as _QiskIBMQBackend,
        AccountProvider,
    )
    from qiskit.providers.models import QasmBackendConfiguration  # type: ignore

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"


def _gen_debug_results(n_qubits: int, shots: int, index: int) -> Result:
    raw_counts = {"0x0": shots}
    raw_memory = ["0x0"] * shots
    base_result_args = dict(
        backend_name="test_backend",
        backend_version="1.0.0",
        qobj_id="id-123",
        job_id="job-123",
        success=True,
    )
    data = models.ExperimentResultData(counts=raw_counts, memory=raw_memory)
    exp_result_header = QobjExperimentHeader(
        creg_sizes=[["c", n_qubits]], memory_slots=n_qubits
    )
    exp_result = models.ExperimentResult(
        shots=shots,
        success=True,
        meas_level=2,
        data=data,
        header=exp_result_header,
        memory=True,
    )
    results = [exp_result] * (index + 1)
    return Result(results=results, **base_result_args)


class NoIBMQAccountError(Exception):
    """Raised when there is no IBMQ account available for the backend"""

    def __init__(self) -> None:
        super().__init__(
            "No IBMQ credentials found on disk, store your account using qiskit,"
            " or using :py:meth:`pytket.extensions.qiskit.set_ibmq_config` first."
        )


class IBMQBackend(Backend):
    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        backend_name: str,
        hub: Optional[str] = None,
        group: Optional[str] = None,
        project: Optional[str] = None,
        monitor: bool = True,
        account_provider: Optional["AccountProvider"] = None,
    ):
        """A backend for running circuits on remote IBMQ devices.
        The provider arguments of `hub`, `group` and `project` can
        be specified here as parameters or set in the config file
        using :py:meth:`pytket.extensions.qiskit.set_ibmq_config`.
        This function can also be used to set the IBMQ API token.

        :param backend_name: Name of the IBMQ device, e.g. `ibmqx4`,
         `ibmq_16_melbourne`.
        :type backend_name: str
        :param hub: Name of the IBMQ hub to use for the provider.
         If None, just uses the first hub found. Defaults to None.
        :type hub: Optional[str], optional
        :param group: Name of the IBMQ group to use for the provider. Defaults to None.
        :type group: Optional[str], optional
        :param project: Name of the IBMQ project to use for the provider.
         Defaults to None.
        :type project: Optional[str], optional
        :param monitor: Use the IBM job monitor. Defaults to True.
        :type monitor: bool, optional
        :raises ValueError: If no IBMQ account is loaded and none exists on the disk.
        :param account_provider: An AccountProvider returned from IBMQ.enable_account.
         Used to pass credentials in if not configured on local machine (as well as hub,
         group and project). Defaults to None.
        :type account_provider: Optional[AccountProvider]
        """
        super().__init__()
        self._pytket_config = QiskitConfig.from_default_config_file()
        self._provider = (
            self._get_provider(hub, group, project, self._pytket_config)
            if account_provider is None
            else account_provider
        )
        self._backend: "_QiskIBMQBackend" = self._provider.get_backend(backend_name)
        self._config = self._backend.configuration()
        self._max_per_job = getattr(self._config, "max_experiments", 1)

        gate_set = _tk_gate_set(self._backend)
        self._backend_info = self._get_backend_info(self._backend)

        self._standard_gateset = gate_set >= {OpType.X, OpType.SX, OpType.Rz, OpType.CX}

        self._monitor = monitor

        # cache of results keyed by job id and circuit index
        self._ibm_res_cache: Dict[Tuple[str, int], models.ExperimentResult] = dict()

        self._MACHINE_DEBUG = False

    @staticmethod
    def _get_provider(
        hub: Optional[str],
        group: Optional[str],
        project: Optional[str],
        qiskit_config: Optional[QiskitConfig],
    ) -> "AccountProvider":
        if not IBMQ.active_account():
            if IBMQ.stored_account():
                IBMQ.load_account()
            else:
                if (
                    qiskit_config is not None
                    and qiskit_config.ibmq_api_token is not None
                ):
                    IBMQ.save_account(qiskit_config.ibmq_api_token)
                else:
                    raise NoIBMQAccountError()
        provider_kwargs: Dict[str, Optional[str]] = {}
        if hub:
            provider_kwargs["hub"] = hub
        else:
            provider_kwargs["hub"] = qiskit_config.hub if qiskit_config else None
        if group:
            provider_kwargs["group"] = group
        else:
            provider_kwargs["group"] = qiskit_config.group if qiskit_config else None
        if project:
            provider_kwargs["project"] = project
        else:
            provider_kwargs["project"] = (
                qiskit_config.project if qiskit_config else None
            )
        try:
            if any(x is not None for x in provider_kwargs.values()):
                provider = IBMQ.get_provider(**provider_kwargs)
            else:
                provider = IBMQ.providers()[0]
        except qiskit.providers.ibmq.exceptions.IBMQProviderError as err:
            logging.warn(
                (
                    "Provider was not specified enough, specify hub,"
                    "group and project correctly (check your IBMQ account)."
                )
            )
            raise err

        return provider

    @property
    def backend_info(self) -> BackendInfo:
        return self._backend_info

    @classmethod
    def _get_backend_info(cls, backend: "_QiskIBMQBackend") -> BackendInfo:
        config = backend.configuration()
        characterisation = process_characterisation(backend)
        averaged_errors = get_avg_characterisation(characterisation)
        characterisation_keys = [
            "t1times",
            "t2times",
            "Frequencies",
            "GateTimes",
        ]
        arch = characterisation["Architecture"]
        # filter entries to keep
        filtered_characterisation = {
            k: v for k, v in characterisation.items() if k in characterisation_keys
        }
        supports_mid_measure = config.simulator or config.multi_meas_enabled
        supports_fast_feedforward = supports_mid_measure
        # simulator i.e. "ibmq_qasm_simulator" does not have `supported_instructions`
        # attribute
        gate_set = _tk_gate_set(backend)
        backend_info = BackendInfo(
            cls.__name__,
            backend.name(),
            __extension_version__,
            arch,
            gate_set,
            supports_midcircuit_measurement=supports_mid_measure,
            supports_fast_feedforward=supports_fast_feedforward,
            all_node_gate_errors=characterisation["NodeErrors"],
            all_edge_gate_errors=characterisation["EdgeErrors"],
            all_readout_errors=characterisation["ReadoutErrors"],
            averaged_node_gate_errors=averaged_errors["node_errors"],
            averaged_edge_gate_errors=averaged_errors["edge_errors"],
            averaged_readout_errors=averaged_errors["readout_errors"],
            misc={"characterisation": filtered_characterisation},
        )
        return backend_info

    @classmethod
    def available_devices(cls, **kwargs: Any) -> List[BackendInfo]:
        provider: Optional["AccountProvider"] = kwargs.get("account_provider")
        if provider is None:
            provider = cls._get_provider(
                kwargs.get("hub"), kwargs.get("group"), kwargs.get("project"), None
            )
        backend_info_list = [
            cls._get_backend_info(backend) for backend in provider.backends()
        ]
        return backend_info_list

    @property
    def required_predicates(self) -> List[Predicate]:
        predicates = [
            NoSymbolsPredicate(),
            GateSetPredicate(
                self._backend_info.gate_set.union(
                    {
                        OpType.Barrier,
                    }
                )
            ),
        ]
        mid_measure = self._backend_info.supports_midcircuit_measurement
        fast_feedforward = self._backend_info.supports_fast_feedforward
        if not mid_measure:
            predicates = [
                NoClassicalControlPredicate(),
                NoMidMeasurePredicate(),
            ] + predicates
        if not fast_feedforward:
            predicates = [
                NoFastFeedforwardPredicate(),
            ] + predicates
        return predicates

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passlist = [DecomposeBoxes()]
        if optimisation_level == 0:
            if self._standard_gateset:
                passlist.append(self.rebase_pass())
        elif optimisation_level == 1:
            passlist.append(SynthesiseTket())
        elif optimisation_level == 2:
            passlist.append(FullPeepholeOptimise())
        mid_measure = self._backend_info.supports_midcircuit_measurement
        arch = self._backend_info.architecture
        if not isinstance(arch, FullyConnected):
            passlist.append(
                CXMappingPass(
                    arch,
                    NoiseAwarePlacement(
                        arch,
                        self._backend_info.averaged_node_gate_errors,
                        self._backend_info.averaged_edge_gate_errors,
                        self._backend_info.averaged_readout_errors,
                    ),
                    directed_cx=False,
                    delay_measures=(not mid_measure),
                )
            )
            passlist.append(NaivePlacementPass(arch))
        if optimisation_level == 1:
            passlist.append(SynthesiseTket())
        if optimisation_level == 2:
            passlist.extend([CliffordSimp(False), SynthesiseTket()])
        if self._standard_gateset:
            passlist.extend([self.rebase_pass(), RemoveRedundancies()])
        if optimisation_level > 0:
            passlist.append(
                SimplifyInitial(allow_classical=False, create_all_qubits=True)
            )
        return SequencePass(passlist)

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, int, str)

    def rebase_pass(self) -> BasePass:
        return auto_rebase_pass(
            {OpType.CX, OpType.X, OpType.SX, OpType.Rz},
        )

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: `postprocess`.
        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
        )

        handle_list: List[Optional[ResultHandle]] = [None] * len(circuits)
        circuit_batches, batch_order = _batch_circuits(circuits, n_shots_list)

        batch_id = 0  # identify batches for debug purposes only
        for (n_shots, batch), indices in zip(circuit_batches, batch_order):
            for chunk in itertools.zip_longest(
                *([iter(zip(batch, indices))] * self._max_per_job)
            ):
                filtchunk = list(filter(lambda x: x is not None, chunk))
                batch_chunk, indices_chunk = zip(*filtchunk)

                if valid_check:
                    self._check_all_circuits(batch_chunk)

                postprocess = kwargs.get("postprocess", False)

                qcs, ppcirc_strs = [], []
                for tkc in batch_chunk:
                    if postprocess:
                        c0, ppcirc = prepare_circuit(tkc, allow_classical=False)
                        ppcirc_rep = ppcirc.to_dict()
                    else:
                        c0, ppcirc_rep = tkc, None
                    qcs.append(tk_to_qiskit(c0))
                    ppcirc_strs.append(json.dumps(ppcirc_rep))
                if self._MACHINE_DEBUG:
                    for i, ind in enumerate(indices_chunk):
                        handle_list[ind] = ResultHandle(
                            _DEBUG_HANDLE_PREFIX
                            + str((batch_chunk[i].n_qubits, n_shots, batch_id)),
                            i,
                            ppcirc_strs[i],
                        )
                else:
                    job = self._backend.run(
                        qcs, shots=n_shots, memory=self._config.memory
                    )
                    jobid = job.job_id()
                    for i, ind in enumerate(indices_chunk):
                        handle_list[ind] = ResultHandle(jobid, i, ppcirc_strs[i])
            batch_id += 1
        for handle in handle_list:
            assert handle is not None
            self._cache[handle] = dict()
        return cast(List[ResultHandle], handle_list)

    def _retrieve_job(self, jobid: str) -> IBMQJob:
        return self._backend.retrieve_job(jobid)

    def cancel(self, handle: ResultHandle) -> None:
        if not self._MACHINE_DEBUG:
            jobid = cast(str, handle[0])
            job = self._retrieve_job(jobid)
            cancelled = job.cancel()
            if not cancelled:
                warn(f"Unable to cancel job {jobid}")

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        self._check_handle_type(handle)
        jobid = cast(str, handle[0])
        apistatus = self._provider._api_client.job_status(jobid)["status"]
        ibmstatus = api_status_to_job_status(apistatus)
        return CircuitStatus(_STATUS_MAP[ibmstatus], ibmstatus.value)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`.
        """
        self._check_handle_type(handle)
        if handle in self._cache:
            cached_result = self._cache[handle]
            if "result" in cached_result:
                return cast(BackendResult, cached_result["result"])
        jobid, index, ppcirc_str = handle
        ppcirc_rep = json.loads(ppcirc_str)
        ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
        cache_key = (jobid, index)
        if cache_key not in self._ibm_res_cache:
            if self._MACHINE_DEBUG or jobid.startswith(_DEBUG_HANDLE_PREFIX):
                shots: int
                n_qubits: int
                n_qubits, shots, _ = literal_eval(jobid[len(_DEBUG_HANDLE_PREFIX) :])
                res = _gen_debug_results(n_qubits, shots, index)
            else:
                try:
                    job = self._retrieve_job(jobid)
                except IBMQBackendApiError:
                    raise CircuitNotRunError(handle)

                if self._monitor and job:
                    job_monitor(job)
                newkwargs = {
                    key: kwargs[key] for key in ("wait", "timeout") if key in kwargs
                }
                res = job.result(**newkwargs)

            for circ_index, r in enumerate(res.results):
                self._ibm_res_cache[(jobid, circ_index)] = r
        result = qiskit_experimentresult_to_backendresult(
            self._ibm_res_cache[cache_key], ppcirc
        )
        self._cache[handle] = {"result": result}
        return result
