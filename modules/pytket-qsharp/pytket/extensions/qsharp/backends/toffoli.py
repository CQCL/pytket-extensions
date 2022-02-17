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

from typing import TYPE_CHECKING, MutableMapping, Optional, Union
import numpy as np
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.passes import BasePass, RebaseCustom  # type: ignore
from pytket.utils.outcomearray import OutcomeArray
from pytket.passes._decompositions import approx_0_mod_2
from .common import _QsharpSimBaseBackend, BackendResult

if TYPE_CHECKING:
    from qsharp.loader import QSharpCallable  # type: ignore


def toffoli_from_tk1(a: float, b: float, c: float) -> Circuit:
    """Only accept operations equivalent to I or X."""
    circ = Circuit(1)
    if approx_0_mod_2(b) and approx_0_mod_2(a + c):
        # identity
        pass
    elif approx_0_mod_2(b + 1) and approx_0_mod_2(a - c):
        # X
        circ.X()
    else:
        raise RuntimeError(
            "Cannot compile to Toffoli gate set: TK1({}, {}, {}) ∉ {{I, X}}".format(
                a, b, c
            )
        )
    return circ


class QsharpToffoliSimulatorBackend(_QsharpSimBaseBackend):
    """Backend for simulating a Toffoli circuit using the QDK."""

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        return RebaseCustom(
            {OpType.CX, OpType.CCX, OpType.CnX, OpType.SWAP, OpType.X},
            Circuit(),  # cx_replacement (irrelevant)
            toffoli_from_tk1,
        )  # tk1_replacement

    def _calculate_results(
        self, qscall: "QSharpCallable", n_shots: Optional[int] = None
    ) -> Union[BackendResult, MutableMapping]:
        if n_shots:
            shots_ar = np.array(
                [qscall.toffoli_simulate() for _ in range(n_shots)], dtype=np.uint8
            )
            shots = OutcomeArray.from_readouts(shots_ar)  # type: ignore
            # ^ type ignore as array is ok for Sequence[Sequence[int]]
            # outputs should correspond to default register,
            # as mapped by FlattenRegisters()
            return BackendResult(shots=shots)
        raise ValueError("Parameter n_shots is required for this backend")
