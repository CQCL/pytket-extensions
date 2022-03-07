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

import sys
from typing import Tuple

import pytest
from _pytest.fixtures import SubRequest
from requests_mock.mocker import Mocker
import jwt

from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumQAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    CredentialStorage,
    MemoryStorage,
    PersistentStorage,
)


@pytest.fixture()
def mock_credentials() -> Tuple[str, str]:
    username = "mark.quantinuum@mail.com"
    pwd = "1906"
    return (username, pwd)


@pytest.fixture()
def mock_token() -> str:
    # A mock token that expires in 2073
    token_payload = {"exp": 3278815149.143694}
    mock_token = jwt.encode(token_payload, key="").decode("utf-8")
    return mock_token


@pytest.fixture(name="mock_quum_api_handler", params=[True, False])
def fixture_mock_quum_api_handler(
    request: SubRequest,
    requests_mock: Mocker,
    mock_credentials: Tuple[str, str],
    mock_token: str,
) -> QuantinuumQAPI:
    """A logged-in QuantinuumQAPI fixture.
    After using this fixture in a test, call:
        mock_quum_api_handler.delete_authentication()
    To remove mock tokens from the keyring.
    """

    username, pwd = mock_credentials

    mock_url = "https://qapi.quantinuum.com/v1/login"

    requests_mock.register_uri(
        "POST",
        mock_url,
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
    )

    cred_store: CredentialStorage
    # Skip testing keyring service if running on linux
    if request.param and sys.platform != "linux":
        cred_store = PersistentStorage()
        cred_store.KEYRING_SERVICE = "HQS_API_MOCK"
    else:
        cred_store = MemoryStorage()

    cred_store.save_login_credential(
        user_name=username,
        password=pwd,
    )

    # Construct QuantinuumQAPI and login
    api_handler = QuantinuumQAPI(
        persistent_credential=False,
        login=False,
    )

    # Add the credential storage seperately in line with fixture parameters
    api_handler.user_name = username
    api_handler._cred_store = cred_store
    api_handler.login()

    return api_handler
