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

from typing import List, Dict, Optional, Any
from qiskit.providers import BaseJob, JobStatus, BaseBackend  # type: ignore
from qiskit.result import Result  # type: ignore
from qiskit.result.models import ExperimentResult  # type: ignore
from qiskit.qobj import QasmQobj  # type: ignore
from pytket.backends import ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.circuit import UnitID  # type: ignore
from pytket.extensions.qiskit.result_convert import backendresult_to_qiskit_resultdata


class TketJob(BaseJob):
    """TketJob wraps a :py:class:`ResultHandle` list as a
    :py:class:`qiskit.providers.BaseJob`"""

    def __init__(
        self,
        backend: BaseBackend,
        handles: List[ResultHandle],
        qobj: QasmQobj,
        final_maps: List[Optional[Dict[UnitID, UnitID]]],
    ):
        """Initializes the asynchronous job."""

        super().__init__(backend, str(handles[0]))
        self._handles = handles
        self._qobj = qobj
        self._result: Optional[Result] = None
        self._final_maps = final_maps

    def submit(self) -> None:
        # Circuits have already been submitted before obtaining the job
        pass

    def result(self, **kwargs: KwargTypes) -> Result:
        if self._result is not None:
            return self._result
        result_list = []
        for h, ex, fm in zip(self._handles, self._qobj.experiments, self._final_maps):
            tk_result = self._backend._backend.get_result(h)
            result_list.append(
                ExperimentResult(
                    shots=self._qobj.config.shots,
                    success=True,
                    data=backendresult_to_qiskit_resultdata(tk_result, ex.header, fm),
                    header=ex.header,
                )
            )
            self._backend._backend.pop_result(h)
        self._result = Result(
            backend_name=self._backend.name(),
            backend_version=self._backend.version(),
            qobj_id=self._qobj.qobj_id,
            job_id=self.job_id(),
            success=True,
            results=result_list,
        )
        return self._result

    def cancel(self) -> None:
        for h in self._handles:
            self._backend._backend.cancel(h)

    def status(self) -> Any:
        status_list = [self._backend._backend.circuit_status(h) for h in self._handles]
        if any((s.status == StatusEnum.RUNNING for s in status_list)):
            return JobStatus.RUNNING
        elif any((s.status == StatusEnum.ERROR for s in status_list)):
            return JobStatus.ERROR
        elif any((s.status == StatusEnum.CANCELLED for s in status_list)):
            return JobStatus.CANCELLED
        elif all((s.status == StatusEnum.COMPLETED for s in status_list)):
            return JobStatus.DONE
        else:
            return JobStatus.INITIALIZING
