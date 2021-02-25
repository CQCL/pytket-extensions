# Copyright 2019-2021 Cambridge Quantum Computing
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
from typing import cast, Iterable, List, Optional, Dict, Any, TYPE_CHECKING, Set
from warnings import warn

import qiskit  # type: ignore
from qiskit import IBMQ
from qiskit.compiler import assemble  # type: ignore
from qiskit.qobj import QobjExperimentHeader  # type: ignore
from qiskit.providers.ibmq.exceptions import IBMQBackendApiError  # type: ignore
from qiskit.providers.ibmq.job import IBMQJob  # type: ignore
from qiskit.result import Result, models  # type: ignore
from qiskit.tools.monitor import job_monitor  # type: ignore

from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.backends import Backend, CircuitNotRunError, CircuitStatus, ResultHandle
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.extensions.qiskit.qiskit_convert import process_characterisation
from pytket.device import Device  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    RebaseCustom,
    RemoveRedundancies,
    RebaseIBM,
    SequencePass,
    SynthesiseIBM,
    CXMappingPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    CliffordSimp,
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
from pytket.extensions.qiskit.result_convert import qiskit_result_to_backendresult
from pytket.routing import NoiseAwarePlacement, Architecture  # type: ignore
from pytket.utils.results import KwargTypes
from .ibm_utils import _STATUS_MAP

if TYPE_CHECKING:
    from qiskit.providers.ibmq import IBMQBackend as _QiskIBMQBackend  # type: ignore
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
            "No IBMQ credentials found on disk, store your account using qiskit first."
        )


def _approx_0_mod_2(x: float, eps: float = 1e-10) -> bool:
    x %= 2
    return min(x, 2 - x) < eps


def _tk1_to_x_v_rz(a: float, b: float, c: float) -> Circuit:
    circ = Circuit(1)
    if _approx_0_mod_2(b):
        circ.Rz(a + c, 0)
    elif _approx_0_mod_2(b + 1):
        if _approx_0_mod_2(a - 0.5) and _approx_0_mod_2(c - 0.5):
            circ.X(0)
        else:
            circ.Rz(c, 0).X(0).Rz(a, 0)
    else:
        if _approx_0_mod_2(a - 0.5) and _approx_0_mod_2(c - 0.5):
            circ.V(0).Rz(1 - b, 0).V(0)
        else:
            circ.Rz(c + 0.5, 0).V(0).Rz(b - 1, 0).V(0).Rz(a + 0.5, 0)
    return circ


class IBMQBackend(Backend):
    _supports_shots = True
    _supports_counts = True
    _persistent_handles = True

    def __init__(
        self,
        backend_name: str,
        hub: Optional[str] = None,
        group: Optional[str] = None,
        project: Optional[str] = None,
        monitor: bool = True,
    ):
        """A backend for running circuits on remote IBMQ devices.

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
        """
        super().__init__()
        if not IBMQ.active_account():
            if IBMQ.stored_account():
                IBMQ.load_account()
            else:
                raise NoIBMQAccountError()
        provider_kwargs = {}
        if hub:
            provider_kwargs["hub"] = hub
        if group:
            provider_kwargs["group"] = group
        if project:
            provider_kwargs["project"] = project

        try:
            if provider_kwargs:
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
        self._backend: "_QiskIBMQBackend" = provider.get_backend(backend_name)
        self._config: "QasmBackendConfiguration" = self._backend.configuration()
        self._gate_set: Set[OpType]
        # simulator i.e. "ibmq_qasm_simulator" does not have `supported_instructions`
        # attribute
        self._gate_set = _tk_gate_set(self._backend)

        self._mid_measure = self._config.simulator or self._config.multi_meas_enabled

        self._legacy_gateset = OpType.V not in self._gate_set

        if self._legacy_gateset:
            if not self._gate_set >= {OpType.U1, OpType.U2, OpType.U3, OpType.CX}:
                raise NotImplementedError(f"Gate set {self._gate_set} unsupported")
            self._rebase_pass = RebaseIBM()
        else:
            if not self._gate_set >= {OpType.X, OpType.V, OpType.Rz, OpType.CX}:
                raise NotImplementedError(f"Gate set {self._gate_set} unsupported")
            self._rebase_pass = RebaseCustom(
                {OpType.CX},
                Circuit(2).CX(0, 1),
                {OpType.X, OpType.V, OpType.Rz},
                _tk1_to_x_v_rz,
            )

        if hasattr(self._config, "max_experiments"):
            self._max_per_job = self._config.max_experiments
        else:
            self._max_per_job = 1

        self._characterisation: Dict[str, Any] = process_characterisation(self._backend)
        self._device = Device(
            self._characterisation.get("NodeErrors", {}),
            self._characterisation.get("EdgeErrors", {}),
            self._characterisation.get("Architecture", Architecture([])),
        )
        self._monitor = monitor

        self._MACHINE_DEBUG = False

    @property
    def characterisation(self) -> Optional[Dict[str, Any]]:
        return self._characterisation

    @property
    def device(self) -> Optional[Device]:
        return self._device

    @property
    def required_predicates(self) -> List[Predicate]:
        predicates = [
            NoSymbolsPredicate(),
            GateSetPredicate(
                self._gate_set.union(
                    {
                        OpType.Barrier,
                    }
                )
            ),
        ]
        if not self._mid_measure:
            predicates = [
                NoClassicalControlPredicate(),
                NoFastFeedforwardPredicate(),
                NoMidMeasurePredicate(),
            ] + predicates
        return predicates

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passlist = [DecomposeBoxes()]
        if optimisation_level == 0:
            passlist.append(self._rebase_pass)
        elif optimisation_level == 1:
            passlist.append(SynthesiseIBM())
        elif optimisation_level == 2:
            passlist.append(FullPeepholeOptimise())
        passlist.append(
            CXMappingPass(
                self._device,
                NoiseAwarePlacement(self._device),
                directed_cx=False,
                delay_measures=(not self._mid_measure),
            )
        )
        if optimisation_level == 1:
            passlist.append(SynthesiseIBM())
        if optimisation_level == 2:
            passlist.extend([CliffordSimp(False), SynthesiseIBM()])
        if not self._legacy_gateset:
            passlist.extend([self._rebase_pass, RemoveRedundancies()])
        return SequencePass(passlist)

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
        handle_list = []
        for chunk in itertools.zip_longest(*([iter(circuits)] * self._max_per_job)):
            filtchunk = list(filter(lambda x: x is not None, chunk))
            if valid_check:
                self._check_all_circuits(filtchunk)
            qcs = [tk_to_qiskit(tkc) for tkc in filtchunk]
            qobj = assemble(qcs, shots=n_shots, memory=self._config.memory)
            if self._MACHINE_DEBUG:
                handle_list += [
                    ResultHandle(_DEBUG_HANDLE_PREFIX + str((c.n_qubits, n_shots)), i)
                    for i, c in enumerate(filtchunk)
                ]
            else:
                job = self._backend.run(qobj)
                jobid = job.job_id()
                handle_list += [ResultHandle(jobid, i) for i in range(len(filtchunk))]
        for handle in handle_list:
            self._cache[handle] = dict()
        return handle_list

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
        ibmstatus = self._retrieve_job(cast(str, handle[0])).status()
        return CircuitStatus(_STATUS_MAP[ibmstatus], ibmstatus.value)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`.
        """
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            jobid = cast(str, handle[0])
            index = cast(int, handle[1])
            if self._MACHINE_DEBUG or jobid.startswith(_DEBUG_HANDLE_PREFIX):
                shots: int
                n_qubits: int
                n_qubits, shots = literal_eval(jobid[len(_DEBUG_HANDLE_PREFIX) :])
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
            backresults = list(qiskit_result_to_backendresult(res))
            self._cache.update(
                (ResultHandle(jobid, circ_index), {"result": backres})
                for circ_index, backres in enumerate(backresults)
            )

            return cast(BackendResult, self._cache[handle]["result"])
