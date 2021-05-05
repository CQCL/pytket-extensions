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

from ast import literal_eval
from collections import Counter
import os
from typing import cast
from pytket.backends.status import StatusEnum
from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.extensions.qsharp import AzureBackend

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None


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


def test_postprocess() -> None:
    b = AzureBackend("ionq.simulator", machine_debug=skip_remote_tests)
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.Rx(0.5, 0).Rx(0.5, 1).CZ(0, 1).X(0).X(1).measure_all()
    b.compile_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(literal_eval(cast(str, h[2])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10
