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
"""Pytket Backend for Honeywell devices."""

from ast import literal_eval
import json
from http import HTTPStatus
from typing import Dict, List, Set, Optional, Sequence, Union, Any, cast

import numpy as np
import requests
import keyring  # type: ignore

from pytket.backends import Backend, ResultHandle, CircuitStatus, StatusEnum
from requests.models import Response
from pytket.backends.backend import KwargTypes
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.backends.backendinfo import BackendInfo, fully_connected_backendinfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.circuit import Circuit, OpType, Bit  # type: ignore
from pytket.extensions.honeywell._metadata import __extension_version__
from pytket.qasm import circuit_to_qasm_str
from pytket.passes import (  # type: ignore
    BasePass,
    SequencePass,
    SynthesiseTket,
    RemoveRedundancies,
    FullPeepholeOptimise,
    DecomposeBoxes,
    DecomposeClassicalExp,
    SimplifyInitial,
    auto_rebase_pass,
    auto_squash_pass,
)
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    MaxNQubitsPredicate,
    Predicate,
    NoSymbolsPredicate,
)
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray

from .config import set_honeywell_config
from .api_wrappers import HQSAPIError, HoneywellQAPI
from .credential_storage import PersistentStorage

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"
HONEYWELL_URL_PREFIX = "https://qapi.honeywell.com/"

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
    OpType.ZZPhase,
    OpType.Reset,
    OpType.Measure,
    OpType.Barrier,
    OpType.RangePredicate,
    OpType.MultiBit,
    OpType.ExplicitPredicate,
    OpType.ExplicitModifier,
    OpType.SetBits,
}


def _get_gateset(machine_name: str) -> Set[OpType]:
    gs = _GATE_SET.copy()
    if "sim" in machine_name.lower():
        gs.remove(OpType.ZZPhase)
    return gs


class GetResultFailed(Exception):
    pass


class HoneywellBackend(Backend):
    """
    Interface to a Honeywell device.
    """

    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        device_name: str,
        label: Optional[str] = "job",
        login: bool = True,
        simulator: str = "state-vector",
        machine_debug: bool = False,
    ):
        """Construct a new Honeywell backend.

        :param device_name: Name of device, e.g. "HQS-LT-S1-APIVAL"
        :type device_name: str
        :param label: Job labels used if Circuits have no name, defaults to "job"
        :type label: Optional[str], optional
        :param login: Flag to attempt login on construction, defaults to True
        :type login: bool, optional
        :param simulator: Only applies to simulator devices, options are
            "state-vector" or "stabilizer", defaults to "state-vector"
        :type simulator: str, optional
        """

        super().__init__()
        self._device_name = device_name
        self._label = label

        self._backend_info: Optional[BackendInfo] = None
        if machine_debug:
            self._api_handler = None
        else:
            self._api_handler = HoneywellQAPI(machine=device_name, login=login)

        self.simulator_type = simulator
        self._gate_set = _get_gateset(self._device_name)

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
    def _available_devices(
        cls, _api_handler: Optional[HoneywellQAPI] = None
    ) -> List[Dict[str, Any]]:
        """List devices available from Honeywell.

        >>> HoneywellBackend._available_devices()
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

    @classmethod
    def available_devices(cls, **kwargs: Any) -> List[BackendInfo]:
        """
        See :py:meth:`pytket.backends.Backend.available_devices`.
        Supported kwargs: `api_handler` (default none).
        """
        api_handler: Optional[HoneywellQAPI] = kwargs.get("api_handler")
        if api_handler is None:
            api_handler = HoneywellQAPI(login=False)
        id_token = api_handler.login()
        res = requests.get(
            f"{api_handler.url}machine/?config=true",
            headers={"Authorization": id_token},
        )
        api_handler._response_check(res, "get machine list")
        jr = res.json()
        return [
            fully_connected_backendinfo(
                cls.__name__,
                machine["name"],
                __extension_version__,
                machine["n_qubits"],
                _get_gateset(machine["name"]),
            )
            for machine in jr
        ]

    def _retrieve_backendinfo(self, machine: str) -> BackendInfo:
        jr = self._available_devices(self._api_handler)
        try:
            self._machine_info = next(entry for entry in jr if entry["name"] == machine)
        except StopIteration:
            raise RuntimeError(f"Device {machine} is not available.")
        return fully_connected_backendinfo(
            type(self).__name__,
            machine,
            __extension_version__,
            self._machine_info["n_qubits"],
            self._gate_set,
        )

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
    def backend_info(self) -> Optional[BackendInfo]:
        if self._backend_info is None and not self._MACHINE_DEBUG:
            self._backend_info = self._retrieve_backendinfo(self._device_name)
        return self._backend_info

    @property
    def required_predicates(self) -> List[Predicate]:
        preds = [
            NoSymbolsPredicate(),
            GateSetPredicate(self._gate_set),
        ]
        if not self._MACHINE_DEBUG:
            assert self.backend_info is not None
            preds.append(MaxNQubitsPredicate(self.backend_info.n_nodes))

        return preds

    def rebase_pass(self) -> BasePass:
        return auto_rebase_pass(self._gate_set)

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passlist = [DecomposeClassicalExp(), DecomposeBoxes()]
        squash = auto_squash_pass({OpType.PhasedX, OpType.Rz})
        if optimisation_level == 0:
            return SequencePass(passlist + [self.rebase_pass()])
        elif optimisation_level == 1:
            return SequencePass(
                passlist
                + [
                    SynthesiseTket(),
                    self.rebase_pass(),
                    RemoveRedundancies(),
                    squash,
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
                ]
            )
        else:
            return SequencePass(
                passlist
                + [
                    FullPeepholeOptimise(),
                    self.rebase_pass(),
                    RemoveRedundancies(),
                    squash,
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
                ]
            )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return tuple((str, str))

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.

        Supported kwargs:

        * `postprocess`: boolean flag to allow classical postprocessing.
        * `noisy_simulation`: boolean flag to specify whether the simulator should
          perform noisy simulation with an error model (default value is `True`).
        * `group`: string identifier of a collection of jobs, can be used for usage
          tracking.
        * `max_batch_cost`: maximum HQC usable by submitted batch, default is 500.
        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
        )

        if valid_check:
            self._check_all_circuits(circuits)

        postprocess = kwargs.get("postprocess", False)
        noisy_simulation = kwargs.get("noisy_simulation", True)
        basebody: Dict[str, Any] = {
            "machine": self._device_name,
            "language": "OPENQASM 2.0",
            "priority": "normal",
            "options": {
                "simulator": self.simulator_type,
                "error-model": noisy_simulation,
            },
        }
        group = kwargs.get("group")
        if group is not None:
            basebody["group"] = group

        handle_list = []
        batch_exec: Union[int, str] = cast(int, kwargs.get("max_batch_cost", 500))
        final_index = len(circuits) - 1
        for i, (circ, n_shots) in enumerate(zip(circuits, n_shots_list)):
            if postprocess:
                c0, ppcirc = prepare_circuit(circ, allow_classical=False, xcirc=_xcirc)
                ppcirc_rep = ppcirc.to_dict()
            else:
                c0, ppcirc_rep = circ, None
            honeywell_circ = circuit_to_qasm_str(c0, header="hqslib1")
            body = basebody.copy()
            body["name"] = circ.name if circ.name else f"{self._label}_{i}"
            body["program"] = honeywell_circ
            body["count"] = n_shots

            if final_index > 0:
                # only set batch fields if more than one job submitted
                body["batch-exec"] = batch_exec
                if i == final_index:
                    # flag to signal end of batch
                    body["batch-end"] = True
            if circ.n_gates_of_type(OpType.ZZPhase) > 0:
                body["options"]["compiler-options"] = {"parametrized_zz": True}
            if self._api_handler is None:
                handle_list.append(
                    ResultHandle(
                        _DEBUG_HANDLE_PREFIX + str((circ.n_qubits, n_shots)),
                        json.dumps(ppcirc_rep),
                    )
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
                            f'HTTP error submitting job, {jobdict["error"]}'
                        )
                except ConnectionError:
                    raise ConnectionError(
                        f"{self._label} Connection Error: Error during submit..."
                    )

                # extract job ID from response
                jobid = cast(str, jobdict["job"])
                if i == 0:
                    # `batch-exec` field set to max batch cost for first job of batch
                    # and to the id of first job of batch otherwise
                    _ = self._api_handler.retrieve_job_status(jobid, use_websocket=True)
                    batch_exec = jobid
                handle = ResultHandle(jobid, json.dumps(ppcirc_rep))
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
                ppcirc_rep = json.loads(cast(str, handle[1]))
                ppcirc = (
                    Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
                )
                self._update_cache_result(
                    handle, _convert_result(response["results"], ppcirc)
                )
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
            ppcirc_rep = json.loads(cast(str, handle[1]))
            ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None

            if self._MACHINE_DEBUG or jobid.startswith(_DEBUG_HANDLE_PREFIX):
                debug_handle_info = jobid[len(_DEBUG_HANDLE_PREFIX) :]
                n_qubits, shots = literal_eval(debug_handle_info)
                return _convert_result({"c": (["0" * n_qubits] * shots)}, ppcirc)
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
                raise GetResultFailed(
                    f"Cannot retrieve result; job status is {circ_status}"
                )
            try:
                res = job_retrieve["results"]
            except KeyError:
                raise GetResultFailed("Results missing in device return data.")

            backres = _convert_result(res, ppcirc)
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
                + " Try running `backend.get_compiled_circuit` first"
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
        set_honeywell_config(user_name)

        keyring.set_password(PersistentStorage.KEYRING_SERVICE, user_name, pwd)  # type: ignore

    @staticmethod
    def clear_saved_login(user_name: str) -> None:
        """Delete saved password for user_name if it exists"""
        set_honeywell_config(None)
        try:
            keyring.delete_password(PersistentStorage.KEYRING_SERVICE, user_name)  # type: ignore
        except keyring.errors.PasswordDeleteError:
            raise ValueError(f"No credentials stored for {user_name}")


_xcirc = Circuit(1).add_gate(OpType.PhasedX, [1, 0], [0])
_xcirc.add_phase(0.5)


def _convert_result(
    resultdict: Dict[str, List[str]], ppcirc: Optional[Circuit] = None
) -> BackendResult:
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
        ppcirc=ppcirc,
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
