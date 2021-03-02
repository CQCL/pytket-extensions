# Copyright 2021 Cambridge Quantum Computing
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
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, cast

import qsharp  # type: ignore
import qsharp.azure  # type: ignore
from qsharp import compile as qscompile  # type: ignore

from pytket.config import load_ext_config, get_config_file_path
from pytket.backends.backend import Backend, BackendResult, CircuitStatus, KwargTypes
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.resulthandle import ResultHandle, _ResultIdTuple
from pytket.backends.status import StatusEnum
from pytket.circuit import Circuit  # type: ignore
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
    This can be provided as a parameter at initialisation or stored in the
    `~/.config/pytket/config.json` file as
    `{'extensions':{'qsharp':{'resourceId':<val>, 'location':<val>}}}`

    Optional parameters can be provided in the same way:
    ``storage``: The connection string to the Azure storage account.
    Required if the specified Azure Quantum workspace was not linked to a storage
    account at workspace creation time.
    """

    _supports_counts = True

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
        super().__init__()
        self._MACHINE_DEBUG = machine_debug
        if not self._MACHINE_DEBUG:
            try:
                config = load_ext_config(QSharpConfig)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Error loading config file at {get_config_file_path()}. "
                    "Try reinstalling a compatible version of pytket."
                )
            if resourceId:
                config.resourceId = resourceId
            if location:
                config.location = location
            if storage:
                config.storage = storage
            print(config.__dict__)
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

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, int)

    def process_circuits(
        self,
        circuits: Iterable[Circuit],
        n_shots: Optional[int] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: none.
        """
        if n_shots is None or n_shots < 1:
            raise ValueError("Parameter n_shots is required for this backend")

        if valid_check:
            self._check_all_circuits(circuits, nomeasure_warn=True)
        handles = []
        for c in circuits:
            if self._MACHINE_DEBUG:
                handles.append(
                    ResultHandle(_DEBUG_HANDLE_PREFIX + str(len(c.bits)), n_shots)
                )
            else:
                qs = tk_to_qsharp(c, sim=False)
                qc = cast("QSharpCallable", qscompile(qs))
                qsharp.azure.target(self.target.id)
                job = qsharp.azure.submit(qc, jobName=c.name, shots=n_shots)
                handle = ResultHandle(job.id, n_shots)
                handles.append(handle)
        for handle in handles:
            self._cache[handle] = dict()
        return handles

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        self._check_handle_type(handle)
        jobid = cast(str, handle[0])
        message = ""
        n_shots = cast(int, handle[1])
        if self._MACHINE_DEBUG:
            n_bits = literal_eval(jobid[len(_DEBUG_HANDLE_PREFIX) :])
            empty_ar = OutcomeArray.from_ints([0] * n_shots, n_bits, big_endian=True)
            self._cache[handle].update({"result": BackendResult(shots=empty_ar)})
            statenum = StatusEnum.COMPLETED
        else:
            job = qsharp.azure.status(jobid)
            status = job.status.lower()
            statenum = _STATUS_MAP.get(status, StatusEnum.ERROR)
            message = repr(job)
            if statenum is StatusEnum.COMPLETED:
                output = qsharp.azure.output(jobid)
                self._cache[handle].update({"result": _convert_result(output, n_shots)})
        return CircuitStatus(statenum, message)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`.
        """
        try:
            return Backend.get_result(self, handle)
        except CircuitNotRunError:
            timeout = kwargs.get("timeout")
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


def _convert_result(output: Dict[str, float], n_shots: int) -> BackendResult:
    counts = Counter(
        {
            OutcomeArray.from_readouts([json.loads(state)]): int(prob * n_shots)
            for state, prob in output.items()
        }
    )

    return BackendResult(counts=counts)
