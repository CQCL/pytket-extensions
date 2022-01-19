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

from typing import TYPE_CHECKING, Optional
import numpy as np
from pytket.utils.outcomearray import OutcomeArray
from .common import _QsharpSimBaseBackend, BackendResult

if TYPE_CHECKING:
    from qsharp.loader import QSharpCallable  # type: ignore


class QsharpSimulatorBackend(_QsharpSimBaseBackend):
    """Backend for simulating a circuit using the QDK."""

    _supports_shots = True
    _supports_counts = True

    def _calculate_results(
        self, qscall: "QSharpCallable", n_shots: Optional[int] = None
    ) -> BackendResult:
        if n_shots:
            shots_ar = np.array(
                [qscall.simulate() for _ in range(n_shots)], dtype=np.uint8
            )
            shots = OutcomeArray.from_readouts(shots_ar)  # type: ignore
            # ^ type ignore as array is ok for Sequence[Sequence[int]]
            # outputs should correspond to default register,
            # as mapped by FlattenRegisters()
            return BackendResult(shots=shots)
        raise ValueError("Parameter n_shots is required for this backend")
