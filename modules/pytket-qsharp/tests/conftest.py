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
from pytket.extensions.qsharp import AzureBackend


@pytest.fixture(name="authenticated_azure_backend")
def fixture_authenticated_azure_backend() -> AzureBackend:
    # Return an unauthenticated backend if the PYTKET_RUN_REMOTE_TESTS env
    # variable is not set
    if os.getenv("PYTKET_RUN_REMOTE_TESTS" ) is None:
        return AzureBackend("ionq.simulator", machine_debug=True)

    # Authenticated AzureBackend used for the remote tests
    # The following env variables are used to create the backend (if they exist):
    #  - PYTKET_REMOTE_QSHARP_RESOURCE_ID
    #  - PYTKET_REMOTE_QSHARP_LOCATION
    #  - PYTKET_REMOTE_QSHARP_STORAGE
    #
    # If any of the env variables does not exist, then we will try to create
    # a backend using the local config settings
    #
    # Note: by default, the target is the 'ionq_sumulator'
    env_vars = (
        "PYTKET_REMOTE_QSHARP_RESOURCE_ID",
        "PYTKET_REMOTE_QSHARP_LOCATION",
        "PYTKET_REMOTE_QSHARP_STORAGE",
    )
    required_env_vars = [os.getenv(var, default=None) for var in env_vars]
    if any(var is None for var in required_env_vars):
        backend = AzureBackend(target_name="ionq.simulator")
    else:
        backend = AzureBackend(
            target_name="ionq.simulator",
            resourceId=os.getenv("PYTKET_REMOTE_QSHARP_RESOURCE_ID"),
            location=os.getenv("PYTKET_REMOTE_QSHARP_LOCATION"),
            storage=os.getenv("PYTKET_REMOTE_QSHARP_STORAGE"),
        )
    return backend
