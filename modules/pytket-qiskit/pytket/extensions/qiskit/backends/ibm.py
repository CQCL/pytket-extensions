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

from sympy import Expr  # type: ignore
import qiskit  # type: ignore
from qiskit import IBMQ
from qiskit.qobj import QobjExperimentHeader  # type: ignore
from qiskit.providers.ibmq.exceptions import IBMQBackendApiError  # type: ignore
from qiskit.providers.ibmq.job import IBMQJob  # type: ignore
from qiskit.result import Result, models  # type: ignore
from qiskit.tools.monitor import job_monitor  # type: ignore

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
    RebaseCustom,
    RemoveRedundancies,
    RebaseIBM,
    SequencePass,
    SynthesiseIBM,
    CXMappingPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    CliffordSimp,
    SimplifyInitial,
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
from pytket.routing import NoiseAwarePlacement  # type: ignore
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


def _approx_0_mod_2(x: Union[float, Expr], eps: float = 1e-10) -> bool:
    if isinstance(x, Expr) and not x.is_constant():
        return False
    x = float(x)
    x %= 2
    return min(x, 2 - x) < eps


def int_half(angle: float) -> int:
    # assume float is approximately an even integer, and return the half
    two_x = round(angle)
    assert not two_x % 2
    return two_x // 2


def _tk1_to_x_sx_rz(
    a: Union[float, Expr], b: Union[float, Expr], c: Union[float, Expr]
) -> Circuit:
    circ = Circuit(1)
    correction_phase = 0.0

    # all phase identities use, for integer k,
    # Rx(2k) = Rz(2k) = (-1)^{k}I

    # _approx_0_mod_2 checks if parameters are constant
    # so they can be assumed to be constant
    if _approx_0_mod_2(b):
        circ.Rz(a + c, 0)
        # b = 2k, if k is odd, then Rx(b) = -I
        correction_phase += int_half(float(b))

    elif _approx_0_mod_2(b + 1):
        # Use Rx(2k-1) = i(-1)^{k}X
        correction_phase += -0.5 + int_half(float(b) - 1)
        if _approx_0_mod_2(a - c):
            circ.X(0)
            # a - c = 2m
            # overall operation is (-1)^{m}Rx(2k -1)
            correction_phase += int_half(float(a - c))

        else:
            circ.Rz(c, 0).X(0).Rz(a, 0)

    elif _approx_0_mod_2(b - 0.5) and _approx_0_mod_2(a) and _approx_0_mod_2(c):
        # a = 2k, b = 2m+0.5, c = 2n
        # Rz(2k)Rx(2m + 0.5)Rz(2n) = (-1)^{k+m+n}e^{-i \pi /4} SX
        circ.SX(0)
        correction_phase += (
            int_half(float(b) - 0.5) + int_half(float(a)) + int_half(float(c)) - 0.25
        )

    elif _approx_0_mod_2(b + 0.5) and _approx_0_mod_2(a) and _approx_0_mod_2(c):
        # a = 2k, b = 2m-0.5, c = 2n
        # Rz(2k)Rx(2m - 0.5)Rz(2n) = (-1)^{k+m+n}e^{i \pi /4} X.SX
        circ.X(0).SX(0)
        correction_phase += (
            int_half(float(b) + 0.5) + int_half(float(a)) + int_half(float(c)) + 0.25
        )
    elif _approx_0_mod_2(a - 0.5) and _approx_0_mod_2(c - 0.5):
        # Rz(2k + 0.5)Rx(b)Rz(2m + 0.5) = -i(-1)^{k+m}SX.Rz(1-b).SX
        circ.SX(0).Rz(1 - b, 0).SX(0)
        correction_phase += int_half(float(a) - 0.5) + int_half(float(c) - 0.5) - 0.5
    else:
        circ.Rz(c + 0.5, 0).SX(0).Rz(b - 1, 0).SX(0).Rz(a + 0.5, 0)
        correction_phase += -0.5

    circ.add_phase(correction_phase)
    return circ


_rebase_pass = RebaseCustom(
    {OpType.CX},
    Circuit(2).CX(0, 1),
    {OpType.X, OpType.SX, OpType.Rz},
    _tk1_to_x_sx_rz,
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
        if account_provider is None:
            self._pytket_config = QiskitConfig.from_default_config_file()
            if not IBMQ.active_account():
                if IBMQ.stored_account():
                    IBMQ.load_account()
                else:
                    if self._pytket_config.ibmq_api_token is not None:
                        IBMQ.save_account(self._pytket_config.ibmq_api_token)
                    else:
                        raise NoIBMQAccountError()
            provider_kwargs = {}
            provider_kwargs["hub"] = hub if hub else self._pytket_config.hub
            provider_kwargs["group"] = group if group else self._pytket_config.group
            provider_kwargs["project"] = (
                project if project else self._pytket_config.project
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
        else:
            provider = account_provider
        self._backend: "_QiskIBMQBackend" = provider.get_backend(backend_name)
        self._config = self._backend.configuration()
        self._max_per_job = getattr(self._config, "max_experiments", 1)

        # gather and store device specifics in BackendInfo
        characterisation = process_characterisation(self._backend)
        characterisation_keys = [
            "NodeErrors",
            "EdgeErrors",
            "ReadoutErrors",
            "GenericOneQubitQErrors",
            "GenericTwoQubitQErrors",
        ]
        arch = characterisation["Architecture"]
        # filter entries to keep
        characterisation = {
            k: v for k, v in characterisation.items() if k in characterisation_keys
        }
        supports_mid_measure = self._config.simulator or self._config.multi_meas_enabled
        supports_fast_feedforward = supports_mid_measure
        # simulator i.e. "ibmq_qasm_simulator" does not have `supported_instructions`
        # attribute
        gate_set = _tk_gate_set(self._backend)

        self._backend_info = BackendInfo(
            type(self).__name__,
            backend_name,
            __extension_version__,
            arch,
            gate_set,
            supports_midcircuit_measurement=supports_mid_measure,
            supports_fast_feedforward=supports_fast_feedforward,
            misc={"characterisation": characterisation},
        )

        self._legacy_gateset = OpType.SX not in gate_set

        if self._legacy_gateset:
            if not gate_set >= {OpType.U1, OpType.U2, OpType.U3, OpType.CX}:
                raise NotImplementedError(f"Gate set {gate_set} unsupported")
            self._rebase_pass = RebaseIBM()
        else:
            if not gate_set >= {OpType.X, OpType.SX, OpType.Rz, OpType.CX}:
                raise NotImplementedError(f"Gate set {gate_set} unsupported")
            self._rebase_pass = _rebase_pass

        self._monitor = monitor

        # cache of results keyed by job id and circuit index
        self._ibm_res_cache: Dict[Tuple[str, int], models.ExperimentResult] = dict()

        self._MACHINE_DEBUG = False

    @property
    def characterisation(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self._backend_info.get_misc("characterisation"))

    @property
    def backend_info(self) -> BackendInfo:
        return self._backend_info

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
            passlist.append(self._rebase_pass)
        elif optimisation_level == 1:
            passlist.append(SynthesiseIBM())
        elif optimisation_level == 2:
            passlist.append(FullPeepholeOptimise())
        arch = self._backend_info.architecture
        mid_measure = self._backend_info.supports_midcircuit_measurement
        passlist.append(
            CXMappingPass(
                arch,
                NoiseAwarePlacement(
                    arch, **get_avg_characterisation(self.characterisation)
                ),
                directed_cx=False,
                delay_measures=(not mid_measure),
            )
        )
        if optimisation_level == 1:
            passlist.append(SynthesiseIBM())
        if optimisation_level == 2:
            passlist.extend([CliffordSimp(False), SynthesiseIBM()])
        if not self._legacy_gateset:
            passlist.extend([self._rebase_pass, RemoveRedundancies()])
        if optimisation_level > 0:
            passlist.append(
                SimplifyInitial(allow_classical=False, create_all_qubits=True)
            )
        return SequencePass(passlist)

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, int, str)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Optional[Union[int, Sequence[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: `postprocess`.
        """
        circuits = list(circuits)
        n_shots_list: List[int] = []
        if hasattr(n_shots, "__iter__"):
            for n in cast(Sequence[Optional[int]], n_shots):
                if n is None or n < 1:
                    raise ValueError(
                        "n_shots values are required for all circuits for this backend"
                    )
                n_shots_list.append(n)
            if len(n_shots_list) != len(circuits):
                raise ValueError("The length of n_shots and circuits must match")
        else:
            if n_shots is None:
                raise ValueError("Parameter n_shots is required for this backend")
            # convert n_shots to a list
            n_shots_list = [cast(int, n_shots)] * len(circuits)

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
        ibmstatus = self._retrieve_job(cast(str, handle[0])).status()
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
