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

from typing import TYPE_CHECKING, Optional, Union, Dict

from pytket.backends import ResultHandle, StatusEnum
from pytket.circuit import Circuit  # type: ignore

from .common import _QsharpBaseBackend, BackendResult

if TYPE_CHECKING:
    from typing import MutableMapping
    from qsharp.loader import QSharpCallable  # type: ignore

ResourcesResult = Dict[str, int]


class QsharpEstimatorBackend(_QsharpBaseBackend):
    """ Backend for estimating resources of a circuit using the QDK. """

    def _calculate_results(
        self, qscall: "QSharpCallable", n_shots: Optional[int] = None
    ) -> Union[BackendResult, "MutableMapping"]:
        results = qscall.estimate_resources()
        results["Measure"] = 0  # Measures were added by qscompile()
        return results  # type: ignore

    def get_resources(self, circuit: Union[Circuit, ResultHandle]) -> ResourcesResult:
        """Calculate resource estimates for circuit.

        :param circuit: Circuit to calculate or result handle to retrieve for
        :type circuit: Union[Circuit, ResultHandle]
        :return: Resource estimate
        :rtype: Dict[str, int]
        """
        if isinstance(circuit, Circuit):
            handle = self.process_circuits([circuit])[0]
        elif isinstance(circuit, ResultHandle):
            handle = circuit
            circ_status = self.circuit_status(handle)
            if circ_status.status is not StatusEnum.COMPLETED:
                raise ValueError(f"Handle is '{circ_status}'")
        else:
            raise TypeError(
                "Provide either a Circuit to run or a ResultHandle to a previously "
                "submitted circuit."
            )
        return self._cache[handle]["resource"]  # type: ignore
