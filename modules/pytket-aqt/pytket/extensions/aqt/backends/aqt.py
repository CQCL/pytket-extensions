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

import json
import time
from ast import literal_eval
from typing import Iterable, List, Optional, Tuple, cast

from requests import put
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.circuit import Circuit, OpType, Qubit  # type: ignore
from pytket.device import Device  # type: ignore


from pytket.passes import (  # type: ignore
    BasePass,
    SequencePass,
    SynthesiseIBM,
    FullPeepholeOptimise,
    FlattenRegisters,
    RebaseCustom,
    EulerAngleReduction,
    DecomposeBoxes,
    RenameQubitsPass,
    SimplifyInitial,
)
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    MaxNQubitsPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.routing import FullyConnected  # type: ignore
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray
from .config import AQTConfig

AQT_URL_PREFIX = "https://gateway.aqt.eu/marmot/"

AQT_DEVICE_QC = "lint"
AQT_DEVICE_SIM = "sim"
AQT_DEVICE_NOISY_SIM = "sim/noise-model-1"

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"

# Hard-coded for now as there is no API to retrieve these.
# All devices are fully connected.
device_info = {
    AQT_DEVICE_QC: {"max_n_qubits": 4},
    AQT_DEVICE_SIM: {"max_n_qubits": 10},
    AQT_DEVICE_NOISY_SIM: {"max_n_qubits": 10},
}


AQTResult = Tuple[int, List[int]]  # (n_qubits, list of readouts)

# TODO add more
_STATUS_MAP = {
    "finished": StatusEnum.COMPLETED,
    "error": StatusEnum.ERROR,
    "queued": StatusEnum.QUEUED,
}


class AqtAuthenticationError(Exception):
    """Raised when there is no AQT access token available."""

    def __init__(self) -> None:
        super().__init__("No AQT access token provided or found in config file.")


class AQTBackend(Backend):
    """
    Interface to an AQT device or simulator.
    """

    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        device_name: str = AQT_DEVICE_SIM,
        access_token: Optional[str] = None,
        label: str = "",
    ):
        """
        Construct a new AQT backend.

        Requires a valid API key/access token, this can either be provided as a
        parameter or set in config using :py:meth:`pytket.extensions.aqt.set_aqt_config`

        :param      device_name:  device name (suffix of URL, e.g. "sim/noise-model-1")
        :type       device_name:  string
        :param      access_token: AQT access token, default None
        :type       access_token: string
        :param      label:        label to apply to submitted jobs
        :type       label:        string
        """
        super().__init__()
        self._url = AQT_URL_PREFIX + device_name
        self._label = label
        config = AQTConfig.from_default_config_file()

        if access_token is None:
            access_token = config.access_token
        if access_token is None:
            raise AqtAuthenticationError()

        self._header = {"Ocp-Apim-Subscription-Key": access_token, "SDK": "pytket"}
        if device_name in device_info:
            self._max_n_qubits: Optional[int] = device_info[device_name]["max_n_qubits"]
            self._device = FullyConnected(self._max_n_qubits)
            self._qm = {Qubit(i): node for i, node in enumerate(self._device.nodes)}
        else:
            self._max_n_qubits = None
            self._device = None
            self._qm = {}
        self._MACHINE_DEBUG = False

    @property
    def device(self) -> Optional[Device]:
        return self._device

    @property
    def required_predicates(self) -> List[Predicate]:
        preds = [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoMidMeasurePredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(
                {OpType.Rx, OpType.Ry, OpType.XXPhase, OpType.Measure, OpType.Barrier}
            ),
        ]
        if self._max_n_qubits is not None:
            preds.append(MaxNQubitsPredicate(self._max_n_qubits))
        return preds

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:

        assert optimisation_level in range(3)
        if optimisation_level == 0:
            return SequencePass(
                [
                    FlattenRegisters(),
                    RenameQubitsPass(self._qm),
                    DecomposeBoxes(),
                    _aqt_rebase(),
                ]
            )
        elif optimisation_level == 1:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    SynthesiseIBM(),
                    FlattenRegisters(),
                    RenameQubitsPass(self._qm),
                    _aqt_rebase(),
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
                    EulerAngleReduction(OpType.Ry, OpType.Rx),
                ]
            )
        else:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    FullPeepholeOptimise(),
                    FlattenRegisters(),
                    RenameQubitsPass(self._qm),
                    _aqt_rebase(),
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
                    EulerAngleReduction(OpType.Ry, OpType.Rx),
                ]
            )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, str, str)

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

        postprocess = kwargs.get("postprocess", False)

        handles = []
        for i, c in enumerate(circuits):
            if postprocess:
                c0, ppcirc = prepare_circuit(c, allow_classical=False, xcirc=_xcirc)
                ppcirc_rep = ppcirc.to_dict()
            else:
                c0, ppcirc_rep = c, None
            (aqt_circ, measures) = _translate_aqt(c0)
            if self._MACHINE_DEBUG:
                handles.append(
                    ResultHandle(
                        _DEBUG_HANDLE_PREFIX + str((c.n_qubits, n_shots)),
                        measures,
                        json.dumps(ppcirc_rep),
                    )
                )
            else:
                resp = put(
                    self._url,
                    data={
                        "data": json.dumps(aqt_circ),
                        "repetitions": n_shots,
                        "no_qubits": c.n_qubits,
                        "label": c.name if c.name else f"{self._label}_{i}",
                    },
                    headers=self._header,
                ).json()
                if "status" not in resp:
                    raise RuntimeError(resp["message"])
                if resp["status"] == "error":
                    raise RuntimeError(resp["ERROR"])
                handles.append(
                    ResultHandle(resp["id"], measures, json.dumps(ppcirc_rep))
                )
        for handle in handles:
            self._cache[handle] = dict()
        return handles

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        self._check_handle_type(handle)
        jobid = handle[0]
        message = ""
        measure_permutations = json.loads(handle[1])  # type: ignore
        ppcirc_rep = json.loads(cast(str, handle[2]))
        ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
        if self._MACHINE_DEBUG:
            n_qubits, n_shots = literal_eval(jobid[len(_DEBUG_HANDLE_PREFIX) :])  # type: ignore
            empty_ar = OutcomeArray.from_ints([0] * n_shots, n_qubits, big_endian=True)
            self._cache[handle].update(
                {"result": BackendResult(shots=empty_ar, ppcirc=ppcirc)}
            )
            statenum = StatusEnum.COMPLETED
        else:
            data = put(self._url, data={"id": jobid}, headers=self._header).json()
            status = data["status"]
            if "ERROR" in data:
                message = data["ERROR"]
            statenum = _STATUS_MAP.get(status, StatusEnum.ERROR)
            if statenum is StatusEnum.COMPLETED:
                shots = OutcomeArray.from_ints(
                    data["samples"], data["no_qubits"], big_endian=True
                )
                shots = shots.choose_indices(measure_permutations)
                self._cache[handle].update(
                    {"result": BackendResult(shots=shots, ppcirc=ppcirc)}
                )
        return CircuitStatus(statenum, message)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`.
        """
        try:
            return super().get_result(handle)
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


def _translate_aqt(circ: Circuit) -> Tuple[List[List], str]:
    """Convert a circuit in the AQT gate set to AQT list representation,
    along with a JSON string describing the measure result permutations."""
    gates: List = list()
    measures: List = list()
    for cmd in circ.get_commands():
        op = cmd.op
        optype = op.type
        # https://www.aqt.eu/aqt-gate-definitions/
        if optype == OpType.Rx:
            gates.append(["X", op.params[0], [q.index[0] for q in cmd.args]])
        elif optype == OpType.Ry:
            gates.append(["Y", op.params[0], [q.index[0] for q in cmd.args]])
        elif optype == OpType.XXPhase:
            gates.append(["MS", op.params[0], [q.index[0] for q in cmd.args]])
        elif optype == OpType.Measure:
            # predicate has already checked format is correct, so
            # errors are not handled here
            qb_id = cmd.qubits[0].index[0]
            bit_id = cmd.bits[0].index[0]
            while len(measures) <= bit_id:
                measures.append(None)
            measures[bit_id] = qb_id
        else:
            assert optype in {OpType.noop, OpType.Barrier}
    if None in measures:
        raise IndexError("Bit index not written to by a measurement.")
    return (gates, json.dumps(measures))


def _aqt_rebase() -> BasePass:
    # CX replacement
    c_cx = Circuit(2)
    c_cx.Ry(0.5, 0).Rx(0.5, 0)
    c_cx.Rx(-0.5, 1)
    c_cx.add_gate(OpType.XXPhase, 0.5, [0, 1])
    c_cx.Ry(0.5, 0).Rx(-1, 0)
    c_cx.add_phase(-0.25)

    # TK1 replacement
    c_tk1 = lambda a, b, c: Circuit(1).Rx(-0.5, 0).Ry(c, 0).Rx(b, 0).Ry(a, 0).Rx(0.5, 0)

    return RebaseCustom({OpType.XXPhase}, c_cx, {OpType.Rx, OpType.Ry}, c_tk1)


_xcirc = Circuit(1).Rx(1, 0)
_xcirc.add_phase(0.5)
