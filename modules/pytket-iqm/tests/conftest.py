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
from pytket.extensions.iqm import IQMBackend


@pytest.fixture(name="authenticated_iqm_backend", scope="session")
def fixture_authenticated_iqm_backend() -> IQMBackend:
    # Authenticated IQMBackend used for the remote tests
    # The credentials are taken from the env variables:
    # PYTKET_REMOTE_IQM_USERNAME and PYTKET_REMOTE_IQM_APIKEY

    # By default, the backend is created with the device_settings in
    # tests/../pytket/extensions/iqm/backends/demo_settings.json
    return IQMBackend(
        device="../pytket/extensions/iqm/backends/demo_settings.json",
        username=os.getenv("PYTKET_REMOTE_IQM_USERNAME"),
        api_key=os.getenv("PYTKET_REMOTE_IQM_APIKEY"),
    )
