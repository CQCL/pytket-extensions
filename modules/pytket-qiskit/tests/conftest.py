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

import os
import pytest
from qiskit import IBMQ  # type: ignore
from pytket.extensions.qiskit import IBMQBackend


@pytest.fixture(autouse=True, scope="session")
def setup_qiskit_account() -> None:
    if os.getenv("PYTKET_RUN_REMOTE_TESTS") is not None:
        # The remote tests require an active IBMQ account
        # We check if an IBMQ account is already saved, otherwise we try
        # to enable one using the token in the env variable:
        # PYTKET_REMOTE_QISKIT_TOKEN
        # Note: The IBMQ account will only be enabled for the current session
        if not IBMQ.stored_account():
            token = os.getenv("PYTKET_REMOTE_QISKIT_TOKEN")
            if token:
                IBMQ.enable_account(token)


@pytest.fixture(scope="module")
def santiago_backend() -> IBMQBackend:
    return IBMQBackend("ibmq_santiago", hub="ibm-q", group="open", project="main")


@pytest.fixture(scope="module")
def lima_backend() -> IBMQBackend:
    return IBMQBackend("ibmq_lima", hub="ibm-q", group="open", project="main")
