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
from _pytest.fixtures import SubRequest
from pytket.extensions.ionq import IonQBackend


@pytest.fixture(name="authenticated_ionq_backend")
def fixture_authenticated_ionq_backend(
    request: SubRequest,
) -> IonQBackend:
    # Authenticated IonqBackend used for the remote tests.
    # The api key is taken from the env variable PYTKET_REMOTE_IONQ_API_KEY.
    # Note: this fixture should only be used in tests where PYTKET_RUN_REMOTE_TESTS
    #       is true, by marking it with @parametrize, using the
    #       "authenticated_ionq_backend" as parameter and `indirect=True`

    # By default, the backend is created with device_name="simulator" only,
    # but other params can be specified when parametrizing the
    # authenticated_ionq_backend
    if request.param is None:
        backend = IonQBackend(
            device_name="simulator",
            api_key=os.getenv("PYTKET_REMOTE_IONQ_API_KEY"),
        )
    else:
        label = ""
        if "label" in request.param:
            label = request.param["label"]

        device_name = "simulator"
        if "device_name" in request.param:
            device_name = request.param["device_name"]

        backend = IonQBackend(
            device_name=device_name,
            api_key=os.getenv("PYTKET_REMOTE_IONQ_API_KEY"),
            label=label,
        )
    return backend
