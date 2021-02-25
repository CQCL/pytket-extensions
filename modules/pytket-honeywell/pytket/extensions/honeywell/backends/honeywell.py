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
"""Pytket Backend for Honeywell devices."""

from ast import literal_eval
import json
from http import HTTPStatus
from typing import Dict, Iterable, List, Optional, Sequence, Any, cast

import numpy as np  # type: ignore
import requests
import keyring  # type: ignore

from pytket.backends import Backend, ResultHandle, CircuitStatus, StatusEnum
from requests.models import Response
from pytket.backends.backend import KwargTypes
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.backends.backendresult import BackendResult
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.circuit import Circuit, OpType, Bit  # type: ignore
from pytket.device import Device  # type: ignore
from pytket.qasm import circuit_to_qasm_str
from pytket.passes import (  # type: ignore
    BasePass,
    SequencePass,
    SynthesiseIBM,
    RemoveRedundancies,
    RebaseHQS,
    SquashHQS,
    FullPeepholeOptimise,
    DecomposeBoxes,
    DecomposeClassicalExp,
)
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    MaxNQubitsPredicate,
    Predicate,
    NoSymbolsPredicate,
)
from pytket.routing import FullyConnected  # type: ignore
from pytket.utils.outcomearray import OutcomeArray
from pytket.config import PytketConfig, get_config_file_path, load_ext_config

from .config import HoneywellConfig
from .api_wrappers import HQSAPIError, HoneywellQAPI

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"


HONEYWELL_URL_PREFIX = "https://qapi.honeywell.com/v1/"

HONEYWELL_DEVICE_APIVAL = "HQS-LT-1.0-APIVAL"

_STATUS_MAP = {
    "queued": StatusEnum.QUEUED,
    "running": StatusEnum.RUNNING,
    "completed": StatusEnum.COMPLETED,
    "failed": StatusEnum.ERROR,
    "canceling": StatusEnum.CANCELLED,
    "canceled": StatusEnum.CANCELLED,
}


_GATE_SET = {
    OpType.Rz,
    OpType.PhasedX,
    OpType.ZZMax,
    OpType.Reset,
    OpType.Measure,
    OpType.Barrier,
    OpType.RangePredicate,
    OpType.MultiBit,
    OpType.ExplicitPredicate,
    OpType.ExplicitModifier,
    OpType.SetBits,
}


class HoneywellBackend(Backend):
    """
    Interface to a Honeywell device.
    """

    _supports_shots = True
    _supports_counts = True
    _persistent_handles = True

    def __init__(
        self,
        device_name: str = HONEYWELL_DEVICE_APIVAL,
        label: Optional[str] = "job",
        machine_debug: bool = False,
        login: bool = True,
    ):
        """
        Construct a new Honeywell backend.

        :param      device_name:  device name  e.g. "HQS-LT-1.0"
        :type       device_name:  string
        :param      label:        label to apply to submitted jobs
        :type       label:        string
        """
        super().__init__()
        self._device_name = device_name
        self._label = label

        self._device = None
        if machine_debug:
            self._api_handler = None
        else:
            self._api_handler = HoneywellQAPI(machine=device_name, login=login)

    @property
    def _MACHINE_DEBUG(self) -> bool:
        return self._api_handler is None

    @_MACHINE_DEBUG.setter
    def _MACHINE_DEBUG(self, val: bool) -> None:
        if val:
            self._api_handler = None
        elif self._api_handler is None:
            raise RuntimeError("_MACHINE_DEBUG cannot be False with no _api_handler.")

    @classmethod
    def available_devices(
        cls, _api_handler: Optional[HoneywellQAPI] = None
    ) -> List[Dict[str, Any]]:
        """List devices available from Honeywell.

        >>> HoneywellBackend.available_devices()
        e.g. [{'name': 'HQS-LT-1.0-APIVAL', 'n_qubits': 6}]

        :param _api_handler: Instance of API handler, defaults to None
        :type _api_handler: Optional[HoneywellQAPI], optional
        :return: Dictionaries of machine name and number of qubits.
        :rtype: List[Dict[str, Any]]
        """
        if _api_handler is None:
            _api_handler = HoneywellQAPI()
        id_token = _api_handler.login()
        res = requests.get(
            f"{_api_handler.url}machine/?config=true",
            headers={"Authorization": id_token},
        )
        _api_handler._response_check(res, "get machine list")
        jr = res.json()
        return jr  # type: ignore

    def _retrieve_device(self, machine: str) -> Device:
        jr = self.available_devices(self._api_handler)
        try:
            self._machine_info = next(entry for entry in jr if entry["name"] == machine)
        except StopIteration:
            raise RuntimeError(f"Device {machine} is not available.")
        return Device(FullyConnected(self._machine_info["n_qubits"]))

    @classmethod
    def device_state(
        cls, device_name: str, _api_handler: Optional[HoneywellQAPI] = None
    ) -> str:
        """Check the status of a device.

        >>> HoneywellBackend.device_state('HQS-LT-1.0-APIVAL') # e.g. "online"


        :param device_name: Name of the device.
        :type device_name: str
        :param _api_handler: Instance of API handler, defaults to None
        :type _api_handler: Optional[HoneywellQAPI], optional
        :return: String of state, e.g. "online"
        :rtype: str
        """
        if _api_handler is None:
            _api_handler = HoneywellQAPI()

        res = requests.get(
            f"{_api_handler.url}machine/{device_name}",
            headers={"Authorization": _api_handler.login()},
        )
        _api_handler._response_check(res, "get machine status")
        jr = res.json()
        return str(jr["state"])

    @property
    def device(self) -> Optional[Device]:
        if self._device is None and not self._MACHINE_DEBUG:
            self._device = self._retrieve_device(self._device_name)
        return self._device

    @property
    def required_predicates(self) -> List[Predicate]:
        preds = [
            NoSymbolsPredicate(),
            GateSetPredicate(_GATE_SET),
        ]
        if not self._MACHINE_DEBUG:
            assert self.device is not None
            preds.append(MaxNQubitsPredicate(len(self.device.nodes)))
        return preds

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        if optimisation_level == 0:
            return SequencePass(
                [DecomposeClassicalExp(), DecomposeBoxes(), RebaseHQS()]
            )
        elif optimisation_level == 1:
            return SequencePass(
                [
                    DecomposeClassicalExp(),
                    DecomposeBoxes(),
                    SynthesiseIBM(),
                    RebaseHQS(),
                    RemoveRedundancies(),
                    SquashHQS(),
                ]
            )
        else:
            return SequencePass(
                [
                    DecomposeClassicalExp(),
                    DecomposeBoxes(),
                    FullPeepholeOptimise(),
                    RebaseHQS(),
                    RemoveRedundancies(),
                    SquashHQS(),
                ]
            )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return tuple((str,))

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
            self._check_all_circuits(circuits)
        basebody = {
            "machine": self._device_name,
            "language": "OPENQASM 2.0",
            "priority": "normal",
            "count": n_shots,
            "options": None,
        }
        handle_list = []
        for i, circ in enumerate(circuits):
            honeywell_circ = circuit_to_qasm_str(circ, header="hqslib1")
            body = basebody.copy()
            body["name"] = circ.name if circ.name else f"{self._label}_{i}"
            body["program"] = honeywell_circ
            if self._api_handler is None:
                handle_list.append(
                    ResultHandle(_DEBUG_HANDLE_PREFIX + str((circ.n_qubits, n_shots)))
                )
            else:
                try:
                    res = _submit_job(self._api_handler, body)
                    if res.status_code != HTTPStatus.OK:
                        self.relogin()
                        res = _submit_job(self._api_handler, body)

                    jobdict = res.json()
                    if res.status_code != HTTPStatus.OK:
                        raise HQSAPIError(
                            f'HTTP error submitting job, {jobdict["error"]["text"]}'
                        )
                except ConnectionError:
                    raise ConnectionError(
                        f"{self._label} Connection Error: Error during submit..."
                    )

                # extract job ID from response
                handle = ResultHandle(jobdict["job"])
                handle_list.append(handle)
                self._cache[handle] = dict()

        return handle_list

    def _retrieve_job(
        self, jobid: str, timeout: Optional[int] = None, wait: Optional[int] = None
    ) -> Dict:
        if not self._api_handler:
            raise RuntimeError("API handler not set")
        with self._api_handler.override_timeouts(timeout=timeout, retry_timeout=wait):
            # set and unset optional timeout parameters

            try:
                job_dict = self._api_handler.retrieve_job(jobid, use_websocket=True)
            except HQSAPIError:
                self.relogin()
                job_dict = self._api_handler.retrieve_job(jobid, use_websocket=True)

        if job_dict is None:
            raise RuntimeError(f"Unable to retrieve job {jobid}")
        return job_dict

    def cancel(self, handle: ResultHandle) -> None:
        if self._api_handler is not None:
            jobid = str(handle[0])
            self._api_handler.cancel(jobid)

    def _update_cache_result(self, handle: ResultHandle, res: BackendResult) -> None:
        rescache = {"result": res}

        if handle in self._cache:
            self._cache[handle].update(rescache)
        else:
            self._cache[handle] = rescache

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        self._check_handle_type(handle)
        jobid = str(handle[0])
        if self._api_handler is None or jobid.startswith(_DEBUG_HANDLE_PREFIX):
            return CircuitStatus(StatusEnum.COMPLETED)
        # TODO check queue position and add to message
        try:
            response = self._api_handler.retrieve_job_status(jobid, use_websocket=True)
        except HQSAPIError:
            self.relogin()
            response = self._api_handler.retrieve_job_status(jobid, use_websocket=True)

        if response is None:
            raise RuntimeError(f"Unable to retrieve circuit status for handle {handle}")
        circ_status = _parse_status(response)
        if circ_status.status is StatusEnum.COMPLETED:
            if "results" in response:
                self._update_cache_result(handle, _convert_result(response["results"]))
        return circ_status

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`.
        """
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            jobid = str(handle[0])

            if self._MACHINE_DEBUG or jobid.startswith(_DEBUG_HANDLE_PREFIX):
                debug_handle_info = jobid[len(_DEBUG_HANDLE_PREFIX) :]
                n_qubits, shots = literal_eval(debug_handle_info)
                return _convert_result({"c": (["0" * n_qubits] * shots)})
            # TODO exception handling when jobid not found on backend
            timeout = kwargs.get("timeout")
            if timeout is not None:
                timeout = int(timeout)
            wait = kwargs.get("wait")
            if wait is not None:
                wait = int(wait)

            job_retrieve = self._retrieve_job(jobid, timeout, wait)
            circ_status = _parse_status(job_retrieve)
            if circ_status.status not in (StatusEnum.COMPLETED, StatusEnum.CANCELLED):
                raise RuntimeError(
                    f"Cannot retrieve results, job status is {circ_status.message}"
                )
            try:
                res = job_retrieve["results"]
            except KeyError:
                raise RuntimeError("Results missing.")

            backres = _convert_result(res)
            self._update_cache_result(handle, backres)
            return backres

    def cost_estimate(self, circuit: Circuit, n_shots: int) -> float:
        """
        Estimate the cost in Honeywell Quantum Credits (HQC) to complete this `circuit`
        with `n_shots` repeats. The estimate is based on hard-coded constants, which may
        be out of date, invalidating the estimate. Use with caution.

        With 𝑁1𝑞 PhasedX gates, 𝑁2𝑞 ZZMax gates, 𝑁𝑚 state preparations and measurements,
        and 𝐶 shots:

        𝐻𝑄𝐶= 5 + (𝑁1𝑞 + 10𝑁2𝑞 + 5𝑁𝑚)*𝐶/5000

        :param circuit: Circuit to calculate runtime estimate for. Must be valid for
            backend.
        :type circuit: Circuit
        :param n_shots: Number of shots.
        :type n_shots: int
        :raises ValueError: Circuit is not valid, needs to be compiled.
        :return: Cost in HQC to execute the shots.
        :rtype: float
        """
        if not self.valid_circuit(circuit):
            raise ValueError(
                "Circuit does not satisfy predicates of backend."
                + " Try running `backend.compile_circuit` first"
            )
        gate_counts: Dict[OpType, int] = {
            g_type: circuit.n_gates_of_type(g_type) for g_type in _GATE_SET
        }
        n_1q = gate_counts[OpType.PhasedX]
        n_m = circuit.n_qubits + gate_counts[OpType.Measure] + gate_counts[OpType.Reset]
        n_2q = gate_counts[OpType.ZZMax]
        cost: float = 5 + (n_1q + 10 * n_2q + 5 * n_m) * n_shots / 5000
        return cost

    def delete_authentication(self) -> None:
        """Remove stored Honeywell Credentials, you will need to log in again."""

        if self._api_handler is not None:
            self._api_handler.delete_authentication()

    def relogin(self) -> None:
        """Remove stored Honeywell Credentials, and ask for credentials again."""

        if self._api_handler is not None:
            self.delete_authentication()
            self._api_handler.login()

    @staticmethod
    def save_login(user_name: str, pwd: str) -> None:
        """NOT RECOMMENDED

        Use this method to (somewhat) securely store your password on your system so you
        don't have to login frequently. This uses the keyring package.

        This is not advised, providing your credentials when prompted is more secure.
        """
        _set_config_username(user_name)

        keyring.set_password(HoneywellQAPI.KEYRING_NAME, user_name, pwd)  # type: ignore

    @staticmethod
    def clear_saved_login(user_name: str) -> None:
        """Delete saved password for user_name if it exists"""
        _set_config_username(None)
        try:
            keyring.delete_password(HoneywellQAPI.KEYRING_NAME, user_name)  # type: ignore
        except keyring.errors.PasswordDeleteError:
            raise ValueError(f"No credentials stored for {user_name}")


def _set_config_username(username: Optional[str]) -> None:
    config_path = get_config_file_path()
    config = PytketConfig.read_file(config_path)
    hconfig = load_ext_config(HoneywellConfig)
    hconfig.username = username
    # TODO fix the type of config.extensions to not need cast
    if cast(Optional[Dict], config.extensions) is None:
        config.extensions = dict()
    config.extensions["honeywell"] = hconfig.to_dict()

    config.write_file(config_path)


def _convert_result(resultdict: Dict[str, List[str]]) -> BackendResult:
    array_dict = {
        creg: np.array([list(a) for a in reslist]).astype(np.uint8)
        for creg, reslist in resultdict.items()
    }
    reversed_creg_names = sorted(array_dict.keys(), reverse=True)
    c_bits = [
        Bit(name, ind)
        for name in reversed_creg_names
        for ind in range(array_dict[name].shape[-1] - 1, -1, -1)
    ]

    stacked_array = np.hstack([array_dict[name] for name in reversed_creg_names])
    return BackendResult(
        c_bits=c_bits,
        shots=OutcomeArray.from_readouts(cast(Sequence[Sequence[int]], stacked_array)),
    )


def _parse_status(response: Dict) -> CircuitStatus:
    h_status = response["status"]
    msgdict = {
        k: response.get(k, None)
        for k in (
            "name",
            "submit-date",
            "result-date",
            "queue-position",
            "cost",
            "error",
        )
    }
    message = str(msgdict)
    return CircuitStatus(_STATUS_MAP[h_status], message)


def _submit_job(api_handler: HoneywellQAPI, body: Dict) -> Response:
    id_token = api_handler.login()
    # send job request
    return requests.post(
        f"{api_handler.url}job",
        json.dumps(body),
        headers={"Authorization": id_token},
    )
