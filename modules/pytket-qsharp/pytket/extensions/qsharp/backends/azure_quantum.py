# Copyright 2021-2022 Cambridge Quantum Computing
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
import time
from ast import literal_eval
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union, cast

import qsharp  # type: ignore
import qsharp.azure  # type: ignore
from qsharp import compile as qscompile  # type: ignore

from pytket.config import get_config_file_path
from pytket.backends.backend import Backend, BackendResult, CircuitStatus, KwargTypes
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.resulthandle import ResultHandle, _ResultIdTuple
from pytket.backends.status import StatusEnum
from pytket.circuit import Circuit  # type: ignore
from pytket.passes import BasePass, SequencePass, SimplifyInitial  # type: ignore
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray

from pytket.extensions.qsharp.backends.config import QSharpConfig
from pytket.extensions.qsharp.backends.common import _QsharpBaseBackend
from pytket.extensions.qsharp.qsharp_convert import tk_to_qsharp


if TYPE_CHECKING:
    from qsharp.loader import QSharpCallable  # type: ignore

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"


_STATUS_MAP = {
    "succeeded": StatusEnum.COMPLETED,
    "cancelled": StatusEnum.CANCELLED,
    "failed": StatusEnum.ERROR,
    "waiting": StatusEnum.SUBMITTED,
    "executing": StatusEnum.RUNNING,
    "finishing": StatusEnum.RUNNING,
}


class AzureBackend(_QsharpBaseBackend):
    """Backend for running circuits remotely using Azure Quantum
    devices and simulators.
    Requires an Azure Quantum workspace set up, as well as the Azure CLI and
    quantum extension.
    Requires ``resourceId`` and ``location`` for Quantum Resource (found when you click
    on the resource in Azure).
    This can be provided as a parameter at initialisation or stored in
    config using :py:meth:`pytket.extensions.qsharp.set_azure_config`

    Optional parameters can be provided in the same way:
    ``storage``: The connection string to the Azure storage account.
    Required if the specified Azure Quantum workspace was not linked to a storage
    account at workspace creation time.
    """

    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        target_name: str,
        resourceId: Optional[str] = None,
        location: Optional[str] = None,
        storage: Optional[str] = None,
        machine_debug: bool = False,
    ):
        """Intialise a new AzureBackend

        :param target_name: name of azure target, e.g. "ionq.simulator"
        :type target_name: str
        :param resourceId: ID for Quantum Resource, can be used to override ID in
            config file, defaults to None
        :type resourceId: Optional[str], optional
        :param location: The Azure region where the Azure Quantum workspace is
            provisioned.
            This may be specified as a region name such as "East US" or a location name
            such as "eastus". If no valid value is specified, defaults to "westus".
            Can be used to override ID in config file, defaults to None
        :type location: Optional[str], optional
        :param storage: The connection string to the Azure storage account,
            can be used to override ID in config file, defaults to None
        :type storage: Optional[str], optional
        :raises ValueError: No resourceId found
        :raises RuntimeError: Azure authentication error
        :raises ValueError: Target not available
        """
        super().__init__(backend_name=target_name)
        self._MACHINE_DEBUG = machine_debug
        if not self._MACHINE_DEBUG:
            try:
                config = QSharpConfig.from_default_config_file()
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Error loading config file at {get_config_file_path()}. "
                    "Try reinstalling a compatible version of pytket."
                ) from e
            if resourceId:
                config.resourceId = resourceId
            if location:
                config.location = location
            if storage:
                config.storage = storage
            # check required parameters
            if any(val is None for val in (config.resourceId, config.location)):
                raise ValueError(
                    "Azure Quantum resourceId and location must be provided as"
                    f" parameter or stored in the config file {get_config_file_path()}"
                    r" as:{'extensions':{'qsharp':"
                    r"{'resourceId':<val>,'location':<val>}}}"
                )

            try:
                target_list = qsharp.azure.connect(**config.__dict__)
            except qsharp.azure.AzureError as e:
                if e.error_name == "AuthenticationFailed":
                    raise RuntimeError(
                        "Could not authenticate with Azure Quantum. "
                        "Ensure you have logged in in your environment"
                        " and installed the required packages."
                    )
                elif e.error_name == "WorkspaceNotFound":
                    raise RuntimeError(
                        "No suitable Azure Quantum workspace found"
                        f" check the resourceId is correct: {config.resourceId}"
                    )
                else:
                    raise e
            try:
                self.target = next(
                    targ for targ in target_list if targ.id == target_name
                )
                qsharp.azure.target(self.target.id)
            except StopIteration:
                raise ValueError(
                    f"Target with id {target_name}"
                    f" not available at resource {config.resourceId}."
                )

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        comp_pass = super().default_compilation_pass(optimisation_level)
        if optimisation_level == 0:
            return comp_pass
        else:
            return SequencePass(
                [
                    comp_pass,
                    SimplifyInitial(allow_classical=False, create_all_qubits=True),
                ]
            )

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
        Supported kwargs: `postprocess`.
        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
        )

        if valid_check:
            self._check_all_circuits(circuits, nomeasure_warn=True)

        postprocess = kwargs.get("postprocess", False)

        handles = []
        for c, n_shots in zip(circuits, n_shots_list):
            if postprocess:
                c0, ppcirc = prepare_circuit(c, allow_classical=False)
                ppcirc_rep = ppcirc.to_dict()
            else:
                c0, ppcirc_rep = c, None
            ppcirc_str = json.dumps(ppcirc_rep)
            if self._MACHINE_DEBUG:
                handles.append(
                    ResultHandle(
                        _DEBUG_HANDLE_PREFIX + str(len(c.bits)), n_shots, ppcirc_str
                    )
                )
            else:
                qs = tk_to_qsharp(c0, sim=False)
                qc = cast("QSharpCallable", qscompile(qs))
                qsharp.azure.target(self.target.id)
                job = qsharp.azure.submit(qc, jobName=c.name, shots=n_shots)
                handle = ResultHandle(job.id, n_shots, ppcirc_str)
                handles.append(handle)
        for handle in handles:
            self._cache[handle] = dict()
        return handles

    def _update_cache_result(
        self, handle: ResultHandle, result_dict: Dict[str, BackendResult]
    ) -> None:
        if handle in self._cache:
            self._cache[handle].update(result_dict)
        else:
            self._cache[handle] = result_dict

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        self._check_handle_type(handle)
        jobid = cast(str, handle[0])
        message = ""
        n_shots = cast(int, handle[1])
        ppcirc_rep = json.loads(cast(str, handle[2]))
        ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
        if self._MACHINE_DEBUG:
            n_bits = literal_eval(jobid[len(_DEBUG_HANDLE_PREFIX) :])
            empty_ar = OutcomeArray.from_ints([0] * n_shots, n_bits, big_endian=True)
            self._update_cache_result(handle, {"result": BackendResult(shots=empty_ar)})
            statenum = StatusEnum.COMPLETED
        else:
            job = qsharp.azure.status(jobid)
            status = job.status.lower()
            statenum = _STATUS_MAP.get(status, StatusEnum.ERROR)
            message = repr(job)
            if statenum is StatusEnum.COMPLETED:
                output = qsharp.azure.output(jobid)
                self._update_cache_result(
                    handle, {"result": _convert_result(output, n_shots, ppcirc=ppcirc)}
                )
        return CircuitStatus(statenum, message)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`.
        """
        try:
            return Backend.get_result(self, handle)
        except CircuitNotRunError:
            timeout = cast(float, kwargs.get("timeout"))
            wait = kwargs.get("wait", 1.0)
            # Wait for job to finish; result will then be in the cache.
            end_time = (time.time() + timeout) if (timeout is not None) else None
            while (end_time is None) or (time.time() < end_time):
                circuit_status = self.circuit_status(handle)
                if circuit_status.status is StatusEnum.COMPLETED:
                    return cast(BackendResult, self._cache[handle]["result"])
                if circuit_status.status is StatusEnum.ERROR:
                    raise RuntimeError(circuit_status.message)
                time.sleep(cast(float, wait))
            raise RuntimeError(f"Timed out: no results after {timeout} seconds.")


def _convert_result(
    output: Dict[str, float], n_shots: int, ppcirc: Optional[Circuit] = None
) -> BackendResult:
    counts = Counter(
        {
            OutcomeArray.from_readouts([json.loads(state)]): int(prob * n_shots)
            for state, prob in output.items()
        }
    )

    return BackendResult(counts=counts, ppcirc=ppcirc)
