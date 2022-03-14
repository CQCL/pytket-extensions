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

from typing import (
    TYPE_CHECKING,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Union,
)
from uuid import uuid4

from qsharp import compile as qscompile  # type: ignore
from pytket.backends import (
    Backend,
    CircuitNotRunError,
    CircuitStatus,
    ResultHandle,
    StatusEnum,
)
from pytket.backends.backend import KwargTypes
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    RebaseCustom,
    SequencePass,
    DecomposeBoxes,
    SynthesiseTket,
    FullPeepholeOptimise,
    FlattenRegisters,
)
from pytket._tket.circuit._library import _TK1_to_RzRx  # type: ignore
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    Predicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
)
from pytket.architecture import Architecture  # type: ignore
from pytket.extensions.qsharp._metadata import __extension_version__
from pytket.extensions.qsharp.qsharp_convert import tk_to_qsharp

if TYPE_CHECKING:
    from qsharp.loader import QSharpCallable  # type: ignore


def qs_predicates(gate_set: Set[OpType]) -> List[Predicate]:
    return [
        NoMidMeasurePredicate(),
        NoSymbolsPredicate(),
        NoClassicalControlPredicate(),
        NoFastFeedforwardPredicate(),
        GateSetPredicate(gate_set),
    ]


class _QsharpBaseBackend(Backend):
    """Shared base backend for Qsharp backends."""

    _persistent_handles = False
    _GATE_SET: Set[OpType] = {
        OpType.CCX,
        OpType.CX,
        OpType.PauliExpBox,
        OpType.H,
        OpType.noop,
        OpType.Rx,
        OpType.Ry,
        OpType.Rz,
        OpType.S,
        OpType.SWAP,
        OpType.T,
        OpType.X,
        OpType.Y,
        OpType.Z,
        OpType.CnX,
        OpType.Measure,
    }

    def __init__(self, backend_name: str = "Qsharp Backend"):
        super().__init__()
        self._backend_info = BackendInfo(
            type(self).__name__,
            backend_name,
            __extension_version__,
            Architecture([]),
            self._GATE_SET,
        )

    @property
    def required_predicates(self) -> List[Predicate]:
        return qs_predicates(self._GATE_SET)

    def rebase_pass(self) -> BasePass:
        return RebaseCustom(
            {
                OpType.CX,
                OpType.CCX,
                OpType.PauliExpBox,
                OpType.SWAP,
                OpType.CnX,
                OpType.H,
                OpType.Rx,
                OpType.Ry,
                OpType.Rz,
                OpType.S,
                OpType.T,
                OpType.X,
                OpType.Y,
                OpType.Z,
            },
            Circuit(),  # cx_replacement (irrelevant)
            _TK1_to_RzRx,
        )  # tk1_replacement

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        if optimisation_level == 0:

            return SequencePass(
                [DecomposeBoxes(), FlattenRegisters(), self.rebase_pass()]
            )
        elif optimisation_level == 1:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    SynthesiseTket(),
                    FlattenRegisters(),
                    self.rebase_pass(),
                ]
            )
        else:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    FullPeepholeOptimise(),
                    FlattenRegisters(),
                    self.rebase_pass(),
                ]
            )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    @property
    def backend_info(self) -> BackendInfo:
        return self._backend_info

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        if handle in self._cache:
            return CircuitStatus(StatusEnum.COMPLETED)
        raise CircuitNotRunError(handle)

    def _calculate_results(
        self, qscall: "QSharpCallable", n_shots: Optional[int] = None
    ) -> Union[BackendResult, MutableMapping]:
        raise NotImplementedError()

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: none.
        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=True,
        )

        if valid_check:
            self._check_all_circuits(circuits, nomeasure_warn=False)
        handles = []
        for c, n_shots in zip(circuits, n_shots_list):
            qs = tk_to_qsharp(c)
            qc = qscompile(qs)
            results = self._calculate_results(qc, n_shots)
            handle = ResultHandle(str(uuid4()))
            key = "result" if isinstance(results, BackendResult) else "resource"
            self._cache.update({handle: {key: results}})
            handles.append(handle)
        return handles


class _QsharpSimBaseBackend(_QsharpBaseBackend):
    _supports_shots = True
    _supports_counts = True
