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

from typing import Optional, List, TYPE_CHECKING

from qiskit.providers.aer import QasmSimulator  # type: ignore
from qiskit.providers.aer.noise.noise_model import NoiseModel  # type: ignore

from .aer import AerBackend
from .ibm import IBMQBackend

if TYPE_CHECKING:
    from pytket.device import Device  # type: ignore
    from pytket.predicates import Predicate  # type: ignore
    from pytket.passes import BasePass  # type: ignore


class IBMQEmulatorBackend(AerBackend):
    """A backend which uses the AerBackend to emulate the behaviour of IBMQBackend.
    Attempts to perform the same compilation and predicate checks as IBMQBackend.
    Requires a valid IBMQ account.

    """

    _supports_shots = True
    _supports_counts = True
    _persistent_handles = False

    def __init__(
        self,
        backend_name: str,
        hub: Optional[str] = None,
        group: Optional[str] = None,
        project: Optional[str] = None,
    ):
        """Construct an IBMQEmulatorBackend. Identical to :py:class:`IBMQBackend`
        constructor, except there is no `monitor` parameter. See :py:class:`IBMQBackend`
        docs for more details.
        """

        self._ibmq = IBMQBackend(backend_name, hub, group, project)
        qasm_sim = QasmSimulator.from_backend(self._ibmq._backend)
        super().__init__(noise_model=NoiseModel.from_backend(qasm_sim))
        self._backend = qasm_sim

    @property
    def device(self) -> Optional["Device"]:
        return self._ibmq._device

    @property
    def required_predicates(self) -> List["Predicate"]:
        return list(self._ibmq.required_predicates)

    def default_compilation_pass(self, optimisation_level: int = 1) -> "BasePass":
        return self._ibmq.default_compilation_pass(optimisation_level)
