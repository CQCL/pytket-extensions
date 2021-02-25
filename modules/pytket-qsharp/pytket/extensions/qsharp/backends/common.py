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

from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    MutableMapping,
    Optional,
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
from pytket.backends.backendresult import BackendResult
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    RebaseCustom,
    SequencePass,
    DecomposeBoxes,
    SynthesiseIBM,
    FullPeepholeOptimise,
    FlattenRegisters,
)
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    Predicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
)
from pytket.extensions.qsharp.qsharp_convert import tk_to_qsharp

if TYPE_CHECKING:
    from qsharp.loader import QSharpCallable  # type: ignore
    from pytket.device import Device  # type: ignore


def _from_tk1(a: float, b: float, c: float) -> Circuit:
    circ = Circuit(1)
    circ.Rz(c, 0)
    circ.Rx(b, 0)
    circ.Rz(a, 0)
    return circ


def qs_predicates() -> List[Predicate]:
    return [
        NoMidMeasurePredicate(),
        NoSymbolsPredicate(),
        NoClassicalControlPredicate(),
        NoFastFeedforwardPredicate(),
        GateSetPredicate(
            {
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
        ),
    ]


def qs_compilation_pass() -> BasePass:
    return RebaseCustom(
        {OpType.CX, OpType.CCX, OpType.PauliExpBox, OpType.SWAP, OpType.CnX},  # multiqs
        Circuit(),  # cx_replacement (irrelevant)
        {
            OpType.H,
            OpType.Rx,
            OpType.Ry,
            OpType.Rz,
            OpType.S,
            OpType.T,
            OpType.X,
            OpType.Y,
            OpType.Z,
        },  # singleqs
        _from_tk1,
    )  # tk1_replacement


class _QsharpBaseBackend(Backend):
    """Shared base backend for Qsharp backends."""

    _persistent_handles = False

    @property
    def required_predicates(self) -> List[Predicate]:
        return qs_predicates()

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        if optimisation_level == 0:

            return SequencePass(
                [DecomposeBoxes(), FlattenRegisters(), qs_compilation_pass()]
            )
        elif optimisation_level == 1:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    SynthesiseIBM(),
                    FlattenRegisters(),
                    qs_compilation_pass(),
                ]
            )
        else:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    FullPeepholeOptimise(),
                    FlattenRegisters(),
                    qs_compilation_pass(),
                ]
            )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    @property
    def device(self) -> Optional["Device"]:
        return None

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
        circuits: Iterable[Circuit],
        n_shots: Optional[int] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: none.
        """
        if valid_check:
            self._check_all_circuits(circuits, nomeasure_warn=False)
        handles = []
        for c in circuits:
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
