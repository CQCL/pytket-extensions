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

import json
from os import PathLike
from typing import cast, Dict, List, Optional, Sequence, Tuple, Union
from uuid import UUID
from iqm_client.iqm_client import Circuit as IQMCircuit  # type: ignore
from iqm_client.iqm_client import (  # type: ignore
    Instruction,
    IQMClient,
    RunStatus,
    SingleQubitMapping,
)
import numpy as np
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit, Node, OpType  # type: ignore
from pytket.extensions.iqm._metadata import __extension_version__
from pytket.passes import (  # type: ignore
    BasePass,
    SequencePass,
    SynthesiseTket,
    FullPeepholeOptimise,
    FlattenRegisters,
    RebaseCustom,
    DecomposeBoxes,
    RemoveRedundancies,
    DefaultMappingPass,
    DelayMeasures,
    SimplifyInitial,
)
from pytket.predicates import (  # type: ignore
    ConnectivityPredicate,
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoBarriersPredicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.architecture import Architecture  # type: ignore
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray
from .config import IQMConfig

_GATE_SET = {OpType.PhasedX, OpType.CZ, OpType.Measure}

# https://iqm-finland.github.io/cirq-on-iqm/api/cirq_iqm.devices.adonis.Adonis.html
_DEFAULT_COUPLING = [
    ("QB1", "QB3"),
    ("QB2", "QB3"),
    ("QB4", "QB3"),
    ("QB5", "QB3"),
]


class IqmAuthenticationError(Exception):
    """Raised when there is no IQM access credentials available."""

    def __init__(self) -> None:
        super().__init__("No IQM access credentials provided or found in config file.")


class IQMBackend(Backend):
    """
    Interface to an IQM device or simulator.
    """

    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        url: str,
        settings: PathLike,
        arch: Optional[List[Tuple[str, str]]] = None,
        auth_server_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Construct a new IQM backend.

        Requires a valid username and API key. These can either be provided as
        parameters or set in config using
        :py:meth:`pytket.extensions.iqm.set_iqm_config`.

        :param url: base URL for requests
        :param settings: path of JSON file containing device settings
        :param arch: list of couplings between the qubits defined in the device settings
            (default: [("QB1", "QB3"), ("QB2", "QB3"), ("QB4", "QB3"), ("QB5", "QB3")],
            i.e. a 5-qubit star topology centred on "QB3")
        :param auth_server_url: base URL of authentication server
        :param username: IQM username
        :param password: IQM password
        """
        super().__init__()
        self._url = url
        config = IQMConfig.from_default_config_file()

        if auth_server_url is None:
            auth_server_url = config.auth_server_url
        if username is None:
            username = config.username
        if username is None:
            raise IqmAuthenticationError()
        if password is None:
            password = config.password
        if password is None:
            raise IqmAuthenticationError()

        with open(settings) as f:
            settings_json = json.load(f)
        self._client = IQMClient(
            self._url,
            settings_json,
            auth_server_url=auth_server_url,
            username=username,
            password=password,
        )
        self._qubits = [
            _as_node(qb)
            for qb in settings_json["subtrees"].keys()
            if qb.startswith("QB")
        ]
        self._n_qubits = len(self._qubits)
        if arch is None:
            arch = _DEFAULT_COUPLING
        coupling = [(_as_node(a), _as_node(b)) for (a, b) in arch]
        if any(qb not in self._qubits for couple in coupling for qb in couple):
            raise ValueError("Architecture contains qubits not in device")
        self._arch = Architecture(coupling)
        self._backendinfo = BackendInfo(
            type(self).__name__,
            settings_json["name"],
            __extension_version__,
            self._arch,
            _GATE_SET,
        )

    @property
    def backend_info(self) -> BackendInfo:
        return self._backendinfo

    @property
    def required_predicates(self) -> List[Predicate]:
        return [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoBarriersPredicate(),
            NoMidMeasurePredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(_GATE_SET),
            ConnectivityPredicate(self._arch),
        ]

    def rebase_pass(self) -> BasePass:
        return _iqm_rebase()

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passes = [DecomposeBoxes(), FlattenRegisters()]
        if optimisation_level == 0:
            passes.append(self.rebase_pass())  # to satisfy MaxTwoQubitGatesPredicate
        elif optimisation_level == 1:
            passes.append(SynthesiseTket())
        elif optimisation_level == 2:
            passes.append(FullPeepholeOptimise())
        passes.append(DefaultMappingPass(self._arch))
        passes.append(DelayMeasures())
        passes.append(self.rebase_pass())
        passes.append(RemoveRedundancies())
        if optimisation_level >= 1:
            passes.append(
                SimplifyInitial(
                    allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                )
            )
        return SequencePass(passes)

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (bytes, str)

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
            self._check_all_circuits(circuits)

        postprocess = kwargs.get("postprocess", False)

        handles = []
        for i, (c, n_shots) in enumerate(zip(circuits, n_shots_list)):
            if postprocess:
                c0, ppcirc = prepare_circuit(c, allow_classical=False, xcirc=_xcirc)
                ppcirc_rep = ppcirc.to_dict()
            else:
                c0, ppcirc_rep = c, None
            instrs = _translate_iqm(c0)
            qm = [
                SingleQubitMapping(logical_name=str(qb), physical_name=_as_name(qb))
                for qb in c.qubits
            ]
            iqmc = IQMCircuit(
                name=c.name if c.name else f"circuit_{i}", instructions=instrs
            )
            run_id = self._client.submit_circuit(iqmc, qm, shots=n_shots)
            handles.append(ResultHandle(run_id.bytes, json.dumps(ppcirc_rep)))
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
        run_id = UUID(bytes=cast(bytes, handle[0]))
        run_result = self._client.get_run(run_id)
        status = run_result.status
        if status is RunStatus.PENDING:
            return CircuitStatus(StatusEnum.SUBMITTED)
        elif status is RunStatus.READY:
            measurements = run_result.measurements
            shots = OutcomeArray.from_readouts(
                np.array(
                    [[r[0] for r in rlist] for cbstr, rlist in measurements.items()],
                    dtype=int,
                )
                .transpose()
                .tolist()
            )
            ppcirc_rep = json.loads(cast(str, handle[1]))
            ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
            self._update_cache_result(
                handle, {"result": BackendResult(shots=shots, ppcirc=ppcirc)}
            )
            return CircuitStatus(StatusEnum.COMPLETED)
        else:
            assert status is RunStatus.FAILED
            return CircuitStatus(StatusEnum.ERROR, run_result.message)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout` (default 900).
        """
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            timeout = kwargs.get("timeout", 900)
            # Wait for job to finish; result will then be in the cache.
            run_id = UUID(bytes=cast(bytes, handle[0]))
            self._client.wait_for_results(run_id, timeout_secs=timeout)
            circuit_status = self.circuit_status(handle)
            if circuit_status.status is StatusEnum.COMPLETED:
                return cast(BackendResult, self._cache[handle]["result"])
            else:
                assert circuit_status.status is StatusEnum.ERROR
                raise RuntimeError(circuit_status.message)


def _as_node(qname: str) -> Node:
    assert qname.startswith("QB")
    x = int(qname[2:])
    assert x >= 1
    return Node(x - 1)


def _as_name(qnode: Node) -> str:
    assert qnode.reg_name == "node"
    return f"QB{qnode.index[0] + 1}"


def _translate_iqm(circ: Circuit) -> List[Instruction]:
    """Convert a circuit in the IQM gate set to IQM list representation."""
    instrs = []
    for cmd in circ.get_commands():
        op = cmd.op
        qbs = cmd.qubits
        cbs = cmd.bits
        optype = op.type
        params = op.params
        if optype == OpType.PhasedX:
            instr = Instruction(
                name="phased_rx",
                qubits=[str(qbs[0])],
                args={"angle_t": 0.5 * params[0], "phase_t": 0.5 * params[1]},
            )
        elif optype == OpType.CZ:
            instr = Instruction(name="cz", qubits=[str(qbs[0]), str(qbs[1])], args={})
        else:
            assert optype == OpType.Measure
            instr = Instruction(
                name="measurement", qubits=[str(qbs[0])], args={"key": str(cbs[0])}
            )
        instrs.append(instr)
    return instrs


def _iqm_rebase() -> BasePass:
    # CX replacement
    c_cx = Circuit(2)
    c_cx.add_gate(OpType.PhasedX, [-0.5, 0.5], [1])
    c_cx.CZ(0, 1)
    c_cx.add_gate(OpType.PhasedX, [0.5, 0.5], [1])

    # TK1 replacement
    c_tk1 = (
        lambda a, b, c: Circuit(1)
        .add_gate(OpType.PhasedX, [-1, (a - c) / 2], [0])
        .add_gate(OpType.PhasedX, [1 + b, a], [0])
    )

    return RebaseCustom({OpType.CZ, OpType.PhasedX}, c_cx, c_tk1)


_xcirc = Circuit(1).add_gate(OpType.PhasedX, [1, 0], [0]).add_phase(0.5)
