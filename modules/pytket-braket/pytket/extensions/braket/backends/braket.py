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
from enum import Enum
import time
from typing import (
    cast,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
    Tuple,
    Set,
    TYPE_CHECKING,
)
from uuid import uuid4
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.extensions.braket.braket_convert import (
    tk_to_braket,
    get_avg_characterisation,
)
from pytket.extensions.braket._metadata import __extension_version__
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    CXMappingPass,
    RebaseCustom,
    RemoveRedundancies,
    SequencePass,
    SynthesiseTket,
    FullPeepholeOptimise,
    CliffordSimp,
    SquashCustom,
    DecomposeBoxes,
    SimplifyInitial,
    NaivePlacementPass,
)
from pytket._tket.circuit._library import _TK1_to_RzRx  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.predicates import (  # type: ignore
    ConnectivityPredicate,
    GateSetPredicate,
    MaxNQubitsPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.architecture import Architecture  # type: ignore
from pytket.placement import NoiseAwarePlacement  # type: ignore
from pytket.utils import prepare_circuit
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.outcomearray import OutcomeArray
import braket  # type: ignore
from braket.aws import AwsDevice, AwsSession  # type: ignore
from braket.aws.aws_device import AwsDeviceType  # type: ignore
from braket.aws.aws_quantum_task import AwsQuantumTask  # type: ignore
import braket.circuits  # type: ignore
from braket.circuits.observable import Observable  # type: ignore
from braket.circuits.qubit_set import QubitSet  # type: ignore
from braket.circuits.result_type import ResultType  # type: ignore
from braket.device_schema import DeviceActionType  # type: ignore
from braket.devices import LocalSimulator  # type: ignore
from braket.tasks.local_quantum_task import LocalQuantumTask  # type: ignore
import boto3  # type: ignore
import numpy as np

from .config import BraketConfig

if TYPE_CHECKING:
    from pytket.circuit import Node  # type: ignore

# Known schemas for noise characteristics
IONQ_SCHEMA = {
    "name": "braket.device_schema.ionq.ionq_provider_properties",
    "version": "1",
}
RIGETTI_SCHEMA = {
    "name": "braket.device_schema.rigetti.rigetti_provider_properties",
    "version": "1",
}
OQC_SCHEMA = {
    "name": "braket.device_schema.oqc.oqc_provider_properties",
    "version": "1",
}

_gate_types = {
    "amplitude_damping": None,
    "bit_flip": None,
    "ccnot": OpType.CCX,
    "cnot": OpType.CX,
    "cphaseshift": OpType.CU1,
    "cphaseshift00": None,
    "cphaseshift01": None,
    "cphaseshift10": None,
    "cswap": OpType.CSWAP,
    "cv": OpType.CV,
    "cy": OpType.CY,
    "cz": OpType.CZ,
    "depolarizing": None,
    "ecr": OpType.ECR,
    "end_verbatim_box": None,
    "generalized_amplitude_damping": None,
    "h": OpType.H,
    "i": OpType.noop,
    "iswap": OpType.ISWAPMax,
    "kraus": None,
    "pauli_channel": None,
    "pswap": None,
    "phase_damping": None,
    "phase_flip": None,
    "phaseshift": OpType.U1,
    "rx": OpType.Rx,
    "ry": OpType.Ry,
    "rz": OpType.Rz,
    "s": OpType.S,
    "si": OpType.Sdg,
    "start_verbatim_box": None,
    "swap": OpType.SWAP,
    "t": OpType.T,
    "ti": OpType.Tdg,
    "two_qubit_dephasing": None,
    "two_qubit_depolarizing": None,
    "unitary": None,
    "v": OpType.V,
    "vi": OpType.Vdg,
    "x": OpType.X,
    "xx": OpType.XXPhase,
    "xy": OpType.ISWAP,
    "y": OpType.Y,
    "yy": OpType.YYPhase,
    "z": OpType.Z,
    "zz": OpType.ZZPhase,
}

_multiq_gate_types = {
    "ccnot",
    "cnot",
    "cphaseshift",
    "cphaseshift00",
    "cphaseshift01",
    "cphaseshift10",
    "cswap",
    "cv",
    "cy",
    "cz",
    "ecr",
    "iswap",
    "pswap",
    "swap",
    "two_qubit_dephasing",
    "two_qubit_depolarizing",
    "unitary",
    "xx",
    "xy",
    "yy",
    "zz",
}

_observables = {
    Pauli.I: Observable.I(),
    Pauli.X: Observable.X(),
    Pauli.Y: Observable.Y(),
    Pauli.Z: Observable.Z(),
}


def _obs_from_qps(pauli: QubitPauliString) -> Tuple[Observable, QubitSet]:
    obs, qbs = [], []
    for q, p in pauli.map.items():
        obs.append(_observables[p])
        qbs.append(q.index[0])
    return Observable.TensorProduct(obs), qbs


def _obs_from_qpo(operator: QubitPauliOperator, n_qubits: int) -> Observable:
    H = operator.to_sparse_matrix(n_qubits).toarray()
    return Observable.Hermitian(H)


def _get_result(
    completed_task: Union[AwsQuantumTask, LocalQuantumTask],
    n_qubits: int,
    want_state: bool,
    want_dm: bool,
    ppcirc: Optional[Circuit] = None,
) -> Dict[str, BackendResult]:
    result = completed_task.result()
    kwargs = {}
    if want_state or want_dm:
        assert ppcirc is None
        if want_state:
            kwargs["state"] = result.get_value_by_result_type(ResultType.StateVector())
        if want_dm:
            m = result.get_value_by_result_type(
                ResultType.DensityMatrix(target=list(range(n_qubits)))
            )
            if type(completed_task) == AwsQuantumTask:
                kwargs["density_matrix"] = np.array(
                    [[complex(x, y) for x, y in row] for row in m], dtype=complex
                )
            else:
                kwargs["density_matrix"] = m
    else:
        kwargs["shots"] = OutcomeArray.from_readouts(result.measurements)
        kwargs["ppcirc"] = ppcirc
    return {"result": BackendResult(**kwargs)}


class _DeviceType(str, Enum):
    LOCAL = "LOCAL"
    SIMULATOR = "SIMULATOR"
    QPU = "QPU"


class BraketBackend(Backend):
    """Interface to Amazon Braket service"""

    _persistent_handles = True

    def __init__(
        self,
        local: bool = False,
        device: Optional[str] = None,
        region: str = "",
        s3_bucket: Optional[str] = None,
        s3_folder: Optional[str] = None,
        device_type: Optional[str] = None,
        provider: Optional[str] = None,
        aws_session: Optional[AwsSession] = None,
    ):
        """
        Construct a new braket backend.

        If `local=True`, other parameters are ignored.

        All parameters except `device` can be set in config using
        :py:meth:`pytket.extensions.braket.set_braket_config`.
        For `device_type`, `provider` and `device` if no parameter
        is specified as a keyword argument or
        in the config file the defaults specified below are used.

        :param local: use simulator running on local machine,
            default: False
        :param device: device name from device ARN (e.g. "ionQdevice", "Aspen-8", ...),
            default: "sv1"
        :param s3_bucket: name of S3 bucket to store results
        :param s3_folder: name of folder ("key") in S3 bucket to store results in
        :param device_type: device type from device ARN (e.g. "qpu"),
            default: "quantum-simulator"
        :param provider: provider name from device ARN (e.g. "ionq", "rigetti", "oqc",
            ...),
            default: "amazon"
        :param aws_session: braket AwsSession object, to pass credentials in if not
            configured on local machine
        """
        super().__init__()
        # load config
        config = BraketConfig.from_default_config_file()
        if s3_bucket is None:
            s3_bucket = config.s3_bucket
        if s3_folder is None:
            s3_folder = config.s3_folder
        if device_type is None:
            device_type = config.device_type
        if provider is None:
            provider = config.provider

        # set defaults if not overridden
        if device_type is None:
            device_type = "quantum-simulator"
        if provider is None:
            provider = "amazon"
        if device is None:
            device = "sv1"

        # set up AwsSession to use; if it's None, braket will create sessions as needed
        self._aws_session = aws_session

        if local:
            self._device = LocalSimulator()
            self._device_type = _DeviceType.LOCAL
        else:
            self._device = AwsDevice(
                "arn:aws:braket:"
                + region
                + "::"
                + "/".join(
                    ["device", device_type, provider, device],
                ),
                aws_session=self._aws_session,
            )
            self._s3_dest = (s3_bucket, s3_folder)
            aws_device_type = self._device.type
            if aws_device_type == AwsDeviceType.SIMULATOR:
                self._device_type = _DeviceType.SIMULATOR
            elif aws_device_type == AwsDeviceType.QPU:
                self._device_type = _DeviceType.QPU
            else:
                raise ValueError(f"Unsupported device type {aws_device_type}")
        props = self._device.properties.dict()
        try:
            device_info = props["action"][DeviceActionType.JAQCD]
        except KeyError:
            # This can happen with quantum anealers (e.g. D-Wave devices)
            raise ValueError(f"Unsupported device {device}")

        supported_ops = set(op.lower() for op in device_info["supportedOperations"])
        supported_result_types = device_info["supportedResultTypes"]
        self._result_types = set()
        for rt in supported_result_types:
            rtname = rt["name"]
            rtminshots = rt["minShots"]
            rtmaxshots = rt["maxShots"]
            self._result_types.add(rtname)
            if rtname == "StateVector":
                self._supports_state = True
                # Always use n_shots = 0 for StateVector
            elif rtname == "Amplitude":
                pass  # Always use n_shots = 0 for Amplitude
            elif rtname == "Probability":
                self._probability_min_shots = rtminshots
                self._probability_max_shots = rtmaxshots
            elif rtname == "Expectation":
                self._supports_expectation = True
                self._expectation_allows_nonhermitian = False
                self._expectation_min_shots = rtminshots
                self._expectation_max_shots = rtmaxshots
            elif rtname == "Sample":
                self._supports_shots = True
                self._supports_counts = True
                self._supports_contextual_optimisation = True
                self._sample_min_shots = rtminshots
                self._sample_max_shots = rtmaxshots
            elif rtname == "Variance":
                self._variance_min_shots = rtminshots
                self._variance_max_shots = rtmaxshots
            elif rtname == "DensityMatrix":
                self._supports_density_matrix = True
                # Always use n_shots = 0 for DensityMatrix
        # Don't use contextual optimization for non-QPU backends
        if self._device_type != _DeviceType.QPU:
            self._supports_contextual_optimisation = False

        self._singleqs, self._multiqs = self._get_gate_set(
            supported_ops, self._device_type
        )

        arch, self._all_qubits = self._get_arch_info(props, self._device_type)
        self._characteristics: Optional[Dict] = None
        if self._device_type == _DeviceType.QPU:
            self._characteristics = props["provider"]
        self._backend_info = self._get_backend_info(
            arch,
            device,
            self._singleqs,
            self._multiqs,
            self._characteristics,
        )

        paradigm = props["paradigm"]
        n_qubits = paradigm["qubitCount"]

        self._req_preds = [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoMidMeasurePredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(self._multiqs | self._singleqs),
            MaxNQubitsPredicate(n_qubits),
        ]

        if (
            self._device_type == _DeviceType.QPU
            and not paradigm["connectivity"]["fullyConnected"]
        ):
            self._req_preds.append(ConnectivityPredicate(arch))

        self._rebase_pass = RebaseCustom(
            self._multiqs | self._singleqs,
            Circuit(),
            _TK1_to_RzRx,
        )
        self._squash_pass = SquashCustom(
            self._singleqs,
            _TK1_to_RzRx,
        )

    @staticmethod
    def _get_gate_set(
        supported_ops: Set[str], device_type: _DeviceType
    ) -> Tuple[Set[OpType], Set[OpType]]:
        multiqs = set()
        singleqs = set()
        if not {"cnot", "rx", "rz", "x"} <= supported_ops:
            # This is so that we can define RebaseCustom without prior knowledge of the
            # gate set, and use X as the bit-flip gate in contextual optimization. We
            # could do better than this, by defining different options depending on the
            # supported gates. But it seems all existing backends support these gates.
            raise NotImplementedError("Device must support cnot, rx, rz and x gates.")
        for t in supported_ops:
            tkt = _gate_types[t]
            if tkt is not None:
                if t in _multiq_gate_types:
                    if device_type == _DeviceType.QPU and t in ["ccnot", "cswap"]:
                        # FullMappingPass can't handle 3-qubit gates, so ignore them.
                        continue
                    multiqs.add(tkt)
                else:
                    singleqs.add(tkt)
        return singleqs, multiqs

    @staticmethod
    def _get_arch_info(
        device_properties: Dict[str, Any], device_type: _DeviceType
    ) -> Tuple[Architecture, List[int]]:
        # return the architecture, and all_qubits
        paradigm = device_properties["paradigm"]
        n_qubits = paradigm["qubitCount"]
        connectivity_graph = None  # None means "fully connected"
        if device_type == _DeviceType.QPU:
            connectivity = paradigm["connectivity"]
            if connectivity["fullyConnected"]:
                all_qubits: List = list(range(n_qubits))
            else:
                connectivity_graph = connectivity["connectivityGraph"]
                # Convert strings to ints
                connectivity_graph = dict(
                    (int(k), [int(v) for v in l]) for k, l in connectivity_graph.items()
                )
                all_qubits = sorted(connectivity_graph.keys())
                if n_qubits < len(all_qubits):
                    # This can happen, at least on rigetti devices, and causes errors.
                    # As a kludgy workaround, remove some qubits from the architecture.
                    all_qubits = all_qubits[: (n_qubits - len(all_qubits))]
                    connectivity_graph = dict(
                        (k, [v for v in l if v in all_qubits])
                        for k, l in connectivity_graph.items()
                        if k in all_qubits
                    )
        else:
            all_qubits = list(range(n_qubits))

        if connectivity_graph is None:
            connectivity_graph = dict(
                (k, [v for v in range(n_qubits) if v != k]) for k in range(n_qubits)
            )
        arch = Architecture([(k, v) for k, l in connectivity_graph.items() for v in l])
        return arch, all_qubits

    @classmethod
    def _get_backend_info(
        cls,
        arch: Architecture,
        device_name: str,
        singleqs: Set[OpType],
        multiqs: Set[OpType],
        characteristics: Optional[Dict[str, Any]],
    ) -> BackendInfo:
        if characteristics is not None:
            schema = characteristics["braketSchemaHeader"]
            if schema == IONQ_SCHEMA:
                fid = characteristics["fidelity"]
                get_node_error: Callable[["Node"], float] = lambda n: 1.0 - cast(
                    float, fid["1Q"]["mean"]
                )
                get_readout_error: Callable[["Node"], float] = lambda n: 0.0
                get_link_error: Callable[
                    ["Node", "Node"], float
                ] = lambda n0, n1: 1.0 - cast(float, fid["2Q"]["mean"])
            elif schema == RIGETTI_SCHEMA:
                specs = characteristics["specs"]
                specs1q, specs2q = specs["1Q"], specs["2Q"]
                get_node_error = lambda n: 1.0 - cast(
                    float, specs1q[f"{n.index[0]}"].get("f1QRB", 1.0)
                )
                get_readout_error = lambda n: 1.0 - cast(
                    float, specs1q[f"{n.index[0]}"].get("fRO", 1.0)
                )
                get_link_error = lambda n0, n1: 1.0 - cast(
                    float,
                    specs2q[
                        f"{min(n0.index[0],n1.index[0])}-{max(n0.index[0],n1.index[0])}"
                    ].get("fCZ", 1.0),
                )
            elif schema == OQC_SCHEMA:
                properties = characteristics["properties"]
                props1q, props2q = properties["one_qubit"], properties["two_qubit"]
                get_node_error = lambda n: 1.0 - cast(
                    float, props1q[f"{n.index[0]}"]["fRB"]
                )
                get_readout_error = lambda n: 1.0 - cast(
                    float, props1q[f"{n.index[0]}"]["fRO"]
                )
                get_link_error = lambda n0, n1: 1.0 - cast(
                    float, props2q[f"{n0.index[0]}-{n1.index[0]}"]["fCX"]
                )
            # readout error as symmetric 2x2 matrix
            to_sym_mat: Callable[[float], List[List[float]]] = lambda x: [
                [1.0 - x, x],
                [x, 1.0 - x],
            ]
            node_errors = {
                node: {optype: get_node_error(node) for optype in singleqs}
                for node in arch.nodes
            }
            readout_errors = {
                node: to_sym_mat(get_readout_error(node)) for node in arch.nodes
            }
            link_errors = {
                (n0, n1): {optype: get_link_error(n0, n1) for optype in multiqs}
                for n0, n1 in arch.coupling
            }

            backend_info = BackendInfo(
                cls.__name__,
                device_name,
                __extension_version__,
                arch,
                singleqs.union(multiqs),
                all_node_gate_errors=node_errors,
                all_edge_gate_errors=link_errors,
                all_readout_errors=readout_errors,
            )
        else:
            backend_info = BackendInfo(
                cls.__name__,
                device_name,
                __extension_version__,
                arch,
                singleqs.union(multiqs),
            )
        return backend_info

    @property
    def required_predicates(self) -> List[Predicate]:
        return self._req_preds

    def rebase_pass(self) -> BasePass:
        return self._rebase_pass

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passes = [DecomposeBoxes()]
        if optimisation_level == 1:
            passes.append(SynthesiseTket())
        elif optimisation_level == 2:
            passes.append(FullPeepholeOptimise())
        passes.append(self.rebase_pass())
        if self._device_type == _DeviceType.QPU and self.characterisation is not None:
            arch = self.backend_info.architecture
            passes.append(
                CXMappingPass(
                    arch,
                    NoiseAwarePlacement(
                        arch, **get_avg_characterisation(self.characterisation)
                    ),
                    directed_cx=False,
                    delay_measures=True,
                )
            )
            passes.append(NaivePlacementPass(arch))
            # If CX weren't supported by the device then we'd need to do another
            # rebase_pass here. But we checked above that it is.
        if optimisation_level == 1:
            passes.extend([RemoveRedundancies(), self._squash_pass])
        if optimisation_level == 2:
            passes.extend(
                [
                    CliffordSimp(False),
                    SynthesiseTket(),
                    self.rebase_pass(),
                    self._squash_pass,
                ]
            )
        if self.supports_contextual_optimisation and optimisation_level > 0:
            passes.append(
                SimplifyInitial(allow_classical=False, create_all_qubits=True)
            )
        return SequencePass(passes)

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        # (task ID, whether state vector / density matrix are wanted, serialized ppcirc
        # or "null")
        return (str, int, bool, bool, str)

    def _run(
        self, bkcirc: braket.circuits.Circuit, n_shots: int = 0, **kwargs: KwargTypes
    ) -> Union[AwsQuantumTask, LocalQuantumTask]:
        if self._device_type == _DeviceType.LOCAL:
            return self._device.run(bkcirc, shots=n_shots, **kwargs)
        else:
            return self._device.run(bkcirc, self._s3_dest, shots=n_shots, **kwargs)

    def _to_bkcirc(self, circuit: Circuit) -> braket.circuits.Circuit:
        if self._device_type == _DeviceType.QPU:
            return tk_to_braket(circuit, self._all_qubits)
        else:
            return tk_to_braket(circuit)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        Supported `kwargs`: none
        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots, len(circuits), optional=True, set_zero=True
        )

        if not self.supports_shots and not self.supports_state:
            raise RuntimeError("Backend does not support shots or state")

        if any(
            map(
                lambda n: n > 0
                and (n < self._sample_min_shots or n > self._sample_max_shots),
                n_shots_list,
            )
        ):
            raise ValueError(
                "For sampling, n_shots must be between "
                f"{self._sample_min_shots} and {self._sample_max_shots}. "
                "For statevector simulation, omit this parameter."
            )

        if valid_check:
            self._check_all_circuits(circuits, nomeasure_warn=False)

        postprocess = kwargs.get("postprocess", False)

        handles = []
        for circ, n_shots in zip(circuits, n_shots_list):
            want_state = (n_shots == 0) and self.supports_state
            want_dm = (n_shots == 0) and self.supports_density_matrix
            if postprocess:
                circ_measured = circ.copy()
                circ_measured.measure_all()
                c0, ppcirc = prepare_circuit(circ_measured, allow_classical=False)
                ppcirc_rep = ppcirc.to_dict()
            else:
                c0, ppcirc, ppcirc_rep = circ, None, None
            bkcirc = self._to_bkcirc(c0)
            if want_state:
                bkcirc.add_result_type(ResultType.StateVector())
            if want_dm:
                bkcirc.add_result_type(ResultType.DensityMatrix(target=bkcirc.qubits))
            if not bkcirc.instructions and len(circ.bits) == 0:
                task = None
            else:
                task = self._run(bkcirc, n_shots=n_shots)
            if self._device_type == _DeviceType.LOCAL:
                # Results are available now. Put them in the cache.
                if task is not None:
                    assert task.state() == "COMPLETED"
                    results = _get_result(
                        task, bkcirc.qubit_count, want_state, want_dm, ppcirc
                    )
                else:
                    results = {"result": self.empty_result(circ, n_shots=n_shots)}
            else:
                # Task is asynchronous. Must wait for results.
                results = {}
            if task is not None:
                handle = ResultHandle(
                    task.id,
                    bkcirc.qubit_count,
                    want_state,
                    want_dm,
                    json.dumps(ppcirc_rep),
                )
            else:
                handle = ResultHandle(
                    str(uuid4()), bkcirc.qubit_count, False, False, json.dumps(None)
                )
            self._cache[handle] = results
            handles.append(handle)
        return handles

    def _update_cache_result(
        self, handle: ResultHandle, result_dict: Dict[str, BackendResult]
    ) -> None:
        if handle in self._cache:
            self._cache[handle].update(result_dict)
        else:
            self._cache[handle] = result_dict

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        if self._device_type == _DeviceType.LOCAL:
            return CircuitStatus(StatusEnum.COMPLETED)
        task_id, n_qubits, want_state, want_dm, ppcirc_str = handle
        ppcirc_rep = json.loads(ppcirc_str)
        ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
        task = AwsQuantumTask(task_id, aws_session=self._aws_session)
        state = task.state()
        if state == "FAILED":
            return CircuitStatus(StatusEnum.ERROR, task.metadata()["failureReason"])
        elif state == "CANCELLED":
            return CircuitStatus(StatusEnum.CANCELLED)
        elif state == "COMPLETED":
            self._update_cache_result(
                handle, _get_result(task, n_qubits, want_state, want_dm, ppcirc)
            )
            return CircuitStatus(StatusEnum.COMPLETED)
        elif state == "QUEUED" or state == "CREATED":
            return CircuitStatus(StatusEnum.QUEUED)
        elif state == "RUNNING":
            return CircuitStatus(StatusEnum.RUNNING)
        else:
            return CircuitStatus(StatusEnum.ERROR, f"Unrecognized state '{state}'")

    @property
    def characterisation(self) -> Optional[Dict[str, Any]]:
        node_errors = self._backend_info.all_node_gate_errors
        edge_errors = self._backend_info.all_edge_gate_errors
        readout_errors = self._backend_info.all_readout_errors
        if node_errors is None and edge_errors is None and readout_errors is None:
            return None
        return {
            "NodeErrors": node_errors,
            "EdgeErrors": edge_errors,
            "ReadoutErrors": readout_errors,
        }

    @property
    def backend_info(self) -> BackendInfo:
        return self._backend_info

    @classmethod
    def available_devices(cls, **kwargs: Any) -> List[BackendInfo]:
        """
        See :py:meth:`pytket.backends.Backend.available_devices`.
        Supported kwargs: `region` (default none).
        The particular AWS region to search for devices (e.g. us-east-1).
        Default to the region configured with AWS.
        See the Braket docs for more details.
        """
        region: Optional[str] = kwargs.get("region")
        if region is not None:
            session = AwsSession(boto_session=boto3.Session(region_name=region))
        else:
            session = AwsSession()

        devices = session.search_devices(statuses=["ONLINE"])

        backend_infos = []

        for device in devices:
            aws_device = AwsDevice(device["deviceArn"], aws_session=session)
            if aws_device.type == AwsDeviceType.SIMULATOR:
                device_type = _DeviceType.SIMULATOR
            elif aws_device.type == AwsDeviceType.QPU:
                device_type = _DeviceType.QPU
            else:
                continue

            props = aws_device.properties.dict()
            try:
                device_info = props["action"][DeviceActionType.JAQCD]
                supported_ops = set(
                    op.lower() for op in device_info["supportedOperations"]
                )
                singleqs, multiqs = cls._get_gate_set(supported_ops, device_type)
            except KeyError:
                # The device has unsupported ops or it's a quantum annealer
                continue
            arch, _ = cls._get_arch_info(props, device_type)
            characteristics = None
            if device_type == _DeviceType.QPU:
                characteristics = props["provider"]
            backend_info = cls._get_backend_info(
                arch,
                device["deviceName"],
                singleqs,
                multiqs,
                characteristics,
            )
            backend_infos.append(backend_info)
        return backend_infos

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout` (default none), `wait` (default 1s).
        """
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            timeout = cast(float, kwargs.get("timeout", 60.0))
            wait = cast(float, kwargs.get("wait", 1.0))
            # Wait for job to finish; result will then be in the cache.
            end_time = (time.time() + timeout) if (timeout is not None) else None
            while (end_time is None) or (time.time() < end_time):
                circuit_status = self.circuit_status(handle)
                if circuit_status.status is StatusEnum.COMPLETED:
                    return cast(BackendResult, self._cache[handle]["result"])
                if circuit_status.status is StatusEnum.ERROR:
                    raise RuntimeError(circuit_status.message)
                time.sleep(wait)
            raise RuntimeError(f"Timed out: no results after {timeout} seconds.")

    def _get_expectation_value(
        self,
        bkcirc: braket.circuits.Circuit,
        observable: Observable,
        target: QubitSet,
        n_shots: int,
        **kwargs: KwargTypes,
    ) -> np.float64:
        if not self.supports_expectation:
            raise RuntimeError("Backend does not support expectation")
        if (
            n_shots < self._expectation_min_shots
            or n_shots > self._expectation_max_shots
        ):
            raise ValueError(
                f"n_shots must be between {self._expectation_min_shots} and "
                f"{self._expectation_max_shots}"
            )
        restype = ResultType.Expectation(observable, target=target)
        bkcirc.add_result_type(restype)
        task = self._run(bkcirc, n_shots=n_shots, **kwargs)
        res = task.result()
        return res.get_value_by_result_type(restype)  # type: ignore

    @property
    def supports_variance(self) -> bool:
        """
        Whether the backend support calculation of operator variance
        """
        return "Variance" in self._result_types

    @property
    def supports_probability(self) -> bool:
        """
        Whether the backend support calculation of outcome probabilities
        """
        return "Probability" in self._result_types

    @property
    def supports_amplitude(self) -> bool:
        """
        Whether the backend support calculation of final state amplitudes
        """
        return "Amplitude" in self._result_types

    def _get_variance(
        self,
        bkcirc: braket.circuits.Circuit,
        observable: Observable,
        target: QubitSet,
        n_shots: int,
        **kwargs: KwargTypes,
    ) -> np.float64:
        if not self.supports_variance:
            raise RuntimeError("Backend does not support variance")
        if n_shots < self._variance_min_shots or n_shots > self._variance_max_shots:
            raise ValueError(
                f"n_shots must be between {self._variance_min_shots} and "
                f"{self._variance_max_shots}"
            )
        restype = ResultType.Variance(observable, target=target)
        bkcirc.add_result_type(restype)
        task = self._run(bkcirc, n_shots=n_shots, **kwargs)
        res = task.result()
        return res.get_value_by_result_type(restype)  # type: ignore

    def get_pauli_expectation_value(
        self,
        state_circuit: Circuit,
        pauli: QubitPauliString,
        n_shots: int = 0,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> np.float64:
        """
        Compute the (exact or empirical) expectation of the observed eigenvalues.

        See `pytket.expectations.get_pauli_expectation_value`.

        If `n_shots > 0` the probabilities are calculated empirically by measurements.
        If `n_shots = 0` (if supported) they are calculated exactly by simulation.

        Supported `kwargs` (not valid for local simulator):

        - `poll_timeout_seconds` (int) : Polling timeout for synchronous retrieval of
          result, in seconds (default: 5 days).
        - `poll_interval_seconds` (int) : Polling interval for synchronous retrieval of
          result, in seconds (default: 1 second).
        """
        if valid_check:
            self._check_all_circuits([state_circuit], nomeasure_warn=False)
        bkcirc = self._to_bkcirc(state_circuit)
        observable, qbs = _obs_from_qps(pauli)
        return self._get_expectation_value(bkcirc, observable, qbs, n_shots, **kwargs)

    def get_operator_expectation_value(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        n_shots: int = 0,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> np.float64:
        """
        Compute the (exact or empirical) expectation of the observed eigenvalues.

        See `pytket.expectations.get_operator_expectation_value`.

        If `n_shots > 0` the probabilities are calculated empirically by measurements.
        If `n_shots = 0` (if supported) they are calculated exactly by simulation.

        Supported `kwargs` are as for `BraketBackend.get_pauli_expectation_value`.
        """
        if valid_check:
            self._check_all_circuits([state_circuit], nomeasure_warn=False)
        bkcirc = self._to_bkcirc(state_circuit)
        observable = _obs_from_qpo(operator, state_circuit.n_qubits)
        return self._get_expectation_value(
            bkcirc, observable, bkcirc.qubits, n_shots, **kwargs
        )

    def get_pauli_variance(
        self,
        state_circuit: Circuit,
        pauli: QubitPauliString,
        n_shots: int = 0,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> np.float64:
        """
        Compute the (exact or empirical) variance of the observed eigenvalues.

        See `pytket.expectations.get_pauli_expectation_value`.

        If `n_shots > 0` the probabilities are calculated empirically by measurements.
        If `n_shots = 0` (if supported) they are calculated exactly by simulation.

        Supported `kwargs` are as for `BraketBackend.get_pauli_expectation_value`.
        """
        if valid_check:
            self._check_all_circuits([state_circuit], nomeasure_warn=False)
        bkcirc = self._to_bkcirc(state_circuit)
        observable, qbs = _obs_from_qps(pauli)
        return self._get_variance(bkcirc, observable, qbs, n_shots, **kwargs)

    def get_operator_variance(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        n_shots: int = 0,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> np.float64:
        """
        Compute the (exact or empirical) variance of the observed eigenvalues.

        See `pytket.expectations.get_operator_expectation_value`.

        If `n_shots > 0` the probabilities are calculated empirically by measurements.
        If `n_shots = 0` (if supported) they are calculated exactly by simulation.

        Supported `kwargs` are as for `BraketBackend.get_pauli_expectation_value`.
        """
        if valid_check:
            self._check_all_circuits([state_circuit], nomeasure_warn=False)
        bkcirc = self._to_bkcirc(state_circuit)
        observable = _obs_from_qpo(operator, state_circuit.n_qubits)
        return self._get_variance(bkcirc, observable, bkcirc.qubits, n_shots, **kwargs)

    def get_probabilities(
        self,
        circuit: Circuit,
        qubits: Union[Iterable[int], None] = None,
        n_shots: int = 0,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> np.ndarray:
        """
        Compute the (exact or empirical) probability distribution of outcomes.

        If `n_shots > 0` the probabilities are calculated empirically by measurements.
        If `n_shots = 0` (if supported) they are calculated exactly by simulation.

        Supported `kwargs` are as for `BraketBackend.process_circuits`.

        The order is big-endian with respect to the order of qubits in the argument.
        For example, if qubits=[0,1] then the order of probabilities is [p(0,0), p(0,1),
        p(1,0), p(1,1)], while if qubits=[1,0] the order is [p(0,0), p(1,0), p(0,1),
        p(1,1)], where p(i,j) is the probability of qubit 0 being in state i and qubit 1
        being in state j.

        :param qubits: qubits of interest

        :returns: list of probabilities of outcomes if initial state is all-zeros
        """
        if not self.supports_probability:
            raise RuntimeError("Backend does not support probability")
        if (
            n_shots < self._probability_min_shots
            or n_shots > self._probability_max_shots
        ):
            raise ValueError(
                f"n_shots must be between {self._probability_min_shots} and "
                f"{self._probability_max_shots}"
            )
        if valid_check:
            self._check_all_circuits([circuit], nomeasure_warn=False)
        bkcirc = self._to_bkcirc(circuit)
        restype = ResultType.Probability(target=qubits)
        bkcirc.add_result_type(restype)
        task = self._run(bkcirc, n_shots=n_shots, **kwargs)
        res = task.result()
        return res.get_value_by_result_type(restype)  # type: ignore

    def get_amplitudes(
        self,
        circuit: Circuit,
        states: List[str],
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> Dict[str, complex]:
        """
        Compute the complex coefficients of the final state.

        Supported `kwargs` are as for `BraketBackend.process_circuits`.

        :param states: classical states of interest, as binary strings of '0' and '1'

        :returns: final complex amplitudes if initial state is all-zeros
        """
        if not self.supports_amplitude:
            raise RuntimeError("Backend does not support amplitude")
        if valid_check:
            self._check_all_circuits([circuit], nomeasure_warn=False)
        bkcirc = self._to_bkcirc(circuit)
        restype = ResultType.Amplitude(states)
        bkcirc.add_result_type(restype)
        task = self._run(bkcirc, n_shots=0, **kwargs)
        res = task.result()
        amplitudes = res.get_value_by_result_type(restype)
        cdict = {}
        for k, v in amplitudes.items():
            # The amazon/sv1 simulator gives us 2-element lists [re, im].
            # The local simulator gives us numpy.complex128.
            cdict[k] = complex(*v) if type(v) is list else complex(v)
        return cdict

    def cancel(self, handle: ResultHandle) -> None:
        if self._device_type == _DeviceType.LOCAL:
            raise NotImplementedError("Circuits on local device cannot be cancelled")
        task_id = handle[0]
        task = AwsQuantumTask(task_id, aws_session=self._aws_session)
        if task.state() != "COMPLETED":
            task.cancel()
