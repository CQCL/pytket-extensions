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

from enum import Enum
import time
from typing import (
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    Tuple,
)
from uuid import uuid4
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.extensions.braket.braket_convert import tk_to_braket
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.device import Device, QubitErrorContainer  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    CXMappingPass,
    RebaseCustom,
    RemoveRedundancies,
    SequencePass,
    SynthesiseIBM,
    FullPeepholeOptimise,
    CliffordSimp,
    SquashCustom,
    DecomposeBoxes,
)
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
from pytket.routing import Architecture, FullyConnected, NoiseAwarePlacement  # type: ignore
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.outcomearray import OutcomeArray
import braket  # type: ignore
from braket.aws import AwsDevice  # type: ignore
from braket.aws.aws_device import AwsDeviceType  # type: ignore
from braket.aws.aws_quantum_task import AwsQuantumTask  # type: ignore
import braket.circuits  # type: ignore
from braket.circuits.observable import Observable  # type: ignore
from braket.circuits.qubit_set import QubitSet  # type: ignore
from braket.circuits.result_type import ResultType  # type: ignore
from braket.device_schema import DeviceActionType  # type: ignore
from braket.devices import LocalSimulator  # type: ignore
from braket.tasks.local_quantum_task import LocalQuantumTask  # type: ignore
import numpy as np  # type: ignore

# Known schemas for noise characteristics
IONQ_SCHEMA = {
    "name": "braket.device_schema.ionq.ionq_provider_properties",
    "version": "1",
}
RIGETTI_SCHEMA = {
    "name": "braket.device_schema.rigetti.rigetti_provider_properties",
    "version": "1",
}

_gate_types = {
    "ccnot": OpType.CCX,
    "cnot": OpType.CX,
    "cphaseshift": OpType.CU1,
    "cphaseshift00": None,
    "cphaseshift01": None,
    "cphaseshift10": None,
    "cswap": OpType.CSWAP,
    "cy": OpType.CY,
    "cz": OpType.CZ,
    "h": OpType.H,
    "i": OpType.noop,
    "iswap": OpType.ISWAPMax,
    "pswap": None,
    "phaseshift": OpType.U1,
    "rx": OpType.Rx,
    "ry": OpType.Ry,
    "rz": OpType.Rz,
    "s": OpType.S,
    "si": OpType.Sdg,
    "swap": OpType.SWAP,
    "t": OpType.T,
    "ti": OpType.Tdg,
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
    "cy",
    "cz",
    "iswap",
    "pswap",
    "swap",
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
    for q, p in pauli.to_dict().items():
        obs.append(_observables[p])
        qbs.append(q.index[0])
    return Observable.TensorProduct(obs), qbs


def _obs_from_qpo(operator: QubitPauliOperator, n_qubits: int) -> Observable:
    H = operator.to_sparse_matrix(n_qubits).toarray()
    return Observable.Hermitian(H)


def _get_result(
    completed_task: Union[AwsQuantumTask, LocalQuantumTask], want_state: bool
) -> Dict[str, BackendResult]:
    result = completed_task.result()
    kwargs = {}
    if want_state:
        kwargs["state"] = result.get_value_by_result_type(ResultType.StateVector())
    else:
        kwargs["shots"] = OutcomeArray.from_readouts(result.measurements)
    return {"result": BackendResult(**kwargs)}


class _DeviceType(str, Enum):
    LOCAL = "LOCAL"
    SIMULATOR = "SIMULATOR"
    QPU = "QPU"


class BraketBackend(Backend):
    """ Interface to Amazon Braket service """

    _persistent_handles = True

    def __init__(
        self,
        local: bool = False,
        s3_bucket: str = "",
        s3_folder: str = "",
        device_type: str = "quantum-simulator",
        provider: str = "amazon",
        device: str = "sv1",
    ):
        """
        Construct a new braket backend.

        If `local=True`, other parameters are ignored.

        :param local: use simulator running on local machine
        :param s3_bucket: name of S3 bucket to store results
        :param s3_folder: name of folder ("key") in S3 bucket to store results in
        :param device_type: device type from device ARN (e.g. "qpu")
        :param provider: provider name from device ARN (e.g. "ionq", "rigetti", ...)
        :paran device: device name from device ARN (e.g. "ionQdevice", "Aspen-8", ...)
        """
        super().__init__()
        if local:
            self._device = LocalSimulator()
            self._device_type = _DeviceType.LOCAL
        else:
            self._device = AwsDevice(
                "arn:aws:braket:::"
                + "/".join(["device", device_type, provider, device])
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
        paradigm = props["paradigm"]
        n_qubits = paradigm["qubitCount"]
        connectivity_graph = None  # None means "fully connected"
        if self._device_type == _DeviceType.QPU:
            connectivity = paradigm["connectivity"]
            if connectivity["fullyConnected"]:
                self._all_qubits: List = list(range(n_qubits))
            else:
                connectivity_graph = connectivity["connectivityGraph"]
                # Convert strings to ints
                connectivity_graph = dict(
                    (int(k), [int(v) for v in l]) for k, l in connectivity_graph.items()
                )
                self._all_qubits = sorted(connectivity_graph.keys())
                if n_qubits < len(self._all_qubits):
                    # This can happen, at least on rigetti devices, and causes errors.
                    # As a kludgy workaround, remove some qubits from the architecture.
                    self._all_qubits = self._all_qubits[
                        : (n_qubits - len(self._all_qubits))
                    ]
                    connectivity_graph = dict(
                        (k, [v for v in l if v in self._all_qubits])
                        for k, l in connectivity_graph.items()
                        if k in self._all_qubits
                    )
            self._characteristics: Optional[Dict] = props["provider"]
        else:
            self._all_qubits = list(range(n_qubits))
            self._characteristics = None

        device_info = props["action"][DeviceActionType.JAQCD]
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
                self._sample_min_shots = rtminshots
                self._sample_max_shots = rtmaxshots
            elif rtname == "Variance":
                self._variance_min_shots = rtminshots
                self._variance_max_shots = rtmaxshots

        self._multiqs = set()
        self._singleqs = set()
        if not {"cnot", "rx", "rz"} <= supported_ops:
            # This is so that we can define RebaseCustom without prior knowledge of the
            # gate set. We could do better than this, by having a few different options
            # for the CX- and tk1-replacement circuits. But it seems all existing
            # backends support these gates.
            raise NotImplementedError("Device must support cnot, rx and rz gates.")
        for t in supported_ops:
            tkt = _gate_types[t]
            if tkt is not None:
                if t in _multiq_gate_types:
                    if self._device_type == _DeviceType.QPU and t in ["ccnot", "cswap"]:
                        # FullMappingPass can't handle 3-qubit gates, so ignore them.
                        continue
                    self._multiqs.add(tkt)
                else:
                    self._singleqs.add(tkt)
        self._req_preds = [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoMidMeasurePredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(self._multiqs | self._singleqs),
            MaxNQubitsPredicate(n_qubits),
        ]

        if connectivity_graph is None:
            arch = FullyConnected(n_qubits)
        else:
            arch = Architecture(
                [(k, v) for k, l in connectivity_graph.items() for v in l]
            )
        if self._device_type == _DeviceType.QPU:
            assert self._characteristics is not None
            node_errs = {}
            edge_errs = {}
            schema = self._characteristics["braketSchemaHeader"]
            if schema == IONQ_SCHEMA:
                fid = self._characteristics["fidelity"]
                mean_1q_err = 1 - fid["1Q"]["mean"]
                mean_2q_err = 1 - fid["2Q"]["mean"]
                err_1q_cont = QubitErrorContainer(self._singleqs)
                for optype in self._singleqs:
                    err_1q_cont.add_error((optype, mean_1q_err))
                err_2q_cont = QubitErrorContainer(self._multiqs)
                for optype in self._multiqs:
                    err_2q_cont.add_error((optype, mean_2q_err))
                for node in arch.nodes:
                    node_errs[node] = err_1q_cont
                for coupling in arch.coupling:
                    edge_errs[coupling] = err_2q_cont
            elif schema == RIGETTI_SCHEMA:
                specs = self._characteristics["specs"]
                specs1q, specs2q = specs["1Q"], specs["2Q"]
                for node in arch.nodes:
                    nodespecs = specs1q[f"{node.index[0]}"]
                    err_1q_cont = QubitErrorContainer(self._singleqs)
                    for optype in self._singleqs:
                        err_1q_cont.add_error((optype, 1 - nodespecs.get("f1QRB", 1)))
                    err_1q_cont.add_readout(nodespecs.get("fRO", 1))
                    node_errs[node] = err_1q_cont
                for coupling in arch.coupling:
                    node0, node1 = coupling
                    n0, n1 = node0.index[0], node1.index[0]
                    couplingspecs = specs2q[f"{min(n0,n1)}-{max(n0,n1)}"]
                    err_2q_cont = QubitErrorContainer({OpType.CZ})
                    err_2q_cont.add_error((OpType.CZ, 1 - couplingspecs.get("fCZ", 1)))
                    edge_errs[coupling] = err_2q_cont
            self._tket_device = Device(node_errs, edge_errs, arch)
            if connectivity_graph is not None:
                self._req_preds.append(ConnectivityPredicate(self._tket_device))
        else:
            self._tket_device = None

        self._rebase_pass = RebaseCustom(
            self._multiqs,
            Circuit(),
            self._singleqs,
            lambda a, b, c: Circuit(1).Rz(c, 0).Rx(b, 0).Rz(a, 0),
        )
        self._squash_pass = SquashCustom(
            self._singleqs,
            lambda a, b, c: Circuit(1).Rz(c, 0).Rx(b, 0).Rz(a, 0),
        )

    @property
    def required_predicates(self) -> List[Predicate]:
        return self._req_preds

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passes = [DecomposeBoxes()]
        if optimisation_level == 1:
            passes.append(SynthesiseIBM())
        elif optimisation_level == 2:
            passes.append(FullPeepholeOptimise())
        passes.append(self._rebase_pass)
        if self._device_type == _DeviceType.QPU:
            passes.append(
                CXMappingPass(
                    self._tket_device,
                    NoiseAwarePlacement(self._tket_device),
                    directed_cx=False,
                    delay_measures=True,
                )
            )
            # If CX weren't supported by the device then we'd need to do another
            # rebase_pass here. But we checked above that it is.
        if optimisation_level == 1:
            passes.extend([RemoveRedundancies(), self._squash_pass])
        if optimisation_level == 2:
            passes.extend(
                [
                    CliffordSimp(False),
                    SynthesiseIBM(),
                    self._rebase_pass,
                    self._squash_pass,
                ]
            )
        return SequencePass(passes)

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        # (task ID, whether state vector is wanted)
        return (str, bool)

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
        circuits: Iterable[Circuit],
        n_shots: Optional[int] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        Supported `kwargs`: none
        """
        if not self.supports_shots and not self.supports_state:
            raise RuntimeError("Backend does not support shots or state")
        if n_shots is None:
            n_shots = 0
        want_state = n_shots == 0
        if (not want_state) and (
            n_shots < self._sample_min_shots or n_shots > self._sample_max_shots
        ):
            raise ValueError(
                "For sampling, n_shots must be between "
                f"{self._sample_min_shots} and {self._sample_max_shots}. "
                "For statevector simulation, omit this parameter."
            )
        if valid_check:
            self._check_all_circuits(circuits, nomeasure_warn=False)
        handles = []
        for circ in circuits:
            bkcirc = self._to_bkcirc(circ)
            if want_state:
                bkcirc.add_result_type(ResultType.StateVector())
            if not bkcirc.instructions and len(circ.bits) == 0:
                task = None
            else:
                task = self._run(bkcirc, n_shots=n_shots)
            if self._device_type == _DeviceType.LOCAL:
                # Results are available now. Put them in the cache.
                if task is not None:
                    assert task.state() == "COMPLETED"
                    results = _get_result(task, want_state)
                else:
                    results = {"result": self.empty_result(circ, n_shots=n_shots)}
            else:
                # Task is asynchronous. Must wait for results.
                results = {}
            if task is not None:
                handle = ResultHandle(task.id, want_state)
            else:
                handle = ResultHandle(str(uuid4()), False)
            self._cache[handle] = results
            handles.append(handle)
        return handles

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        if self._device_type == _DeviceType.LOCAL:
            return CircuitStatus(StatusEnum.COMPLETED)
        task_id, want_state = handle
        task = AwsQuantumTask(task_id)
        state = task.state()
        if state == "FAILED":
            result = task.result()
            return CircuitStatus(StatusEnum.ERROR, result.task_metadata.failureReason)
        elif state == "CANCELLED":
            return CircuitStatus(StatusEnum.CANCELLED)
        elif state == "COMPLETED":
            self._cache[handle].update(_get_result(task, want_state))
            return CircuitStatus(StatusEnum.COMPLETED)
        elif state == "QUEUED" or state == "CREATED":
            return CircuitStatus(StatusEnum.QUEUED)
        elif state == "RUNNING":
            return CircuitStatus(StatusEnum.RUNNING)
        else:
            return CircuitStatus(StatusEnum.ERROR, f"Unrecognized state '{state}'")

    @property
    def characterisation(self) -> Optional[Dict]:
        return self._characteristics

    def device(self) -> Optional[Device]:
        return self._tket_device

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout` (default none), `wait` (default 1s).
        """
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            timeout = kwargs.get("timeout", 60.0)
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
        task = AwsQuantumTask(task_id)
        task.cancel()
