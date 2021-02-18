# Copyright 2021 Cambridge Quantum Computing
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

from collections import Counter
import os
from pytket.backends.status import StatusEnum
from pytket.circuit import Circuit  # type: ignore
from pytket.extensions.backends.qsharp import AzureBackend

skip_remote_tests: bool = (
    os.getenv("PYTKET_RUN_REMOTE_TESTS") is None or os.getenv("AZURE_AUTH") is None
)


def test_azure_backend() -> None:
    # TODO investigate bug when not all bits are measured to
    c = (
        Circuit(4, 3, "test_name")
        .H(0)
        .CX(0, 1)
        .ZZPhase(0.1, 1, 0)
        .Measure(0, 2)
        .Measure(1, 1)
        .Measure(3, 0)
    )
    if skip_remote_tests:
        b = AzureBackend("ionq.simulator", machine_debug=True)
    else:
        # assumed environment is authenticated and
        # resourceId in config file
        b = AzureBackend("ionq.simulator")
    b.compile_circuit(c)
    h = b.process_circuit(c, 10)
    assert b.circuit_status(h).status in (
        StatusEnum.SUBMITTED,
        StatusEnum.RUNNING,
        StatusEnum.COMPLETED,
    )

    assert isinstance(h[0], str)
    assert h[1] == 10

    res = b.get_result(h)
    assert b.circuit_status(h).status == StatusEnum.COMPLETED
    counts = res.get_counts()
    assert sum(counts.values()) == 10
    if skip_remote_tests:
        assert counts == Counter({(0, 0, 0): 10})
    else:
        assert counts == Counter({(0, 0, 0): 5, (0, 1, 1): 5})


test_azure_backend()
