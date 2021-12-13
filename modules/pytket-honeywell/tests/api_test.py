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

import datetime
import sys

import pytest
from requests_mock.mocker import Mocker
import jwt

from pytket.extensions.honeywell.backends.api_wrappers import HoneywellQAPI
from pytket.extensions.honeywell.backends.credential_storage import (
    CredentialStorage,
    MemoryStorage,
    PersistentStorage,
)


@pytest.mark.parametrize("test_keyring_storage", [True, False])
def test_hqs_login(test_keyring_storage: bool, requests_mock: Mocker) -> None:
    """Test that credentials are storable and deletable using
    the HoneywellQAPI handler."""

    username = "mark.honeywell@mail.com"
    pwd = "1906"

    id_payload = {"exp": (datetime.datetime.now().timestamp() * 2)}

    mock_id_token = jwt.encode(id_payload, key="").decode("utf-8")

    # Mock /login endpoint
    mock_url = "https://qapi.honeywell.com/v1/login"

    requests_mock.register_uri(
        "POST",
        mock_url,
        json={
            "id-token": mock_id_token,
            "refresh-token": mock_id_token,
        },
        headers={"Content-Type": "application/json"},
    )

    cred_store: CredentialStorage
    # Skip testing keyring service if running on linux
    if test_keyring_storage and sys.platform != "linux":
        cred_store = PersistentStorage()
        cred_store.KEYRING_SERVICE = "HQS_API_MOCK"
    else:
        cred_store = MemoryStorage()

    cred_store.save_login_credential(
        user_name=username,
        password=pwd,
    )

    api_handler = HoneywellQAPI(
        user_name=username,
        token="",
        persistent_credential=False,
        login=False,
    )

    api_handler._cred_store = cred_store
    api_handler.login()

    # Check credentials are retrievable
    assert api_handler._cred_store.login_credential(username) == pwd
    assert api_handler._cred_store.refresh_token == mock_id_token
    assert api_handler._cred_store.id_token == mock_id_token

    # Delete authentication and verify
    api_handler.delete_authentication()
    assert api_handler._cred_store.id_token == None
    assert api_handler._cred_store.login_credential(username) == None
    assert api_handler._cred_store.refresh_token == None


def test_machine_status(
    requests_mock: Mocker,
    mock_hqs_api_handler: HoneywellQAPI,
) -> None:
    """Test that we can retrieve the machine state via  Honeywell endpoint."""

    machine_name = "HQS-LT-S1-APIVAL"
    mock_machine_state = "online"

    # Mock /login endpoint
    mock_url = f"https://qapi.honeywell.com/v1/machine/{machine_name}"

    requests_mock.register_uri(
        "GET",
        mock_url,
        json={"state": mock_machine_state},
        headers={"Content-Type": "application/json"},
    )

    assert mock_hqs_api_handler.status(machine_name) == mock_machine_state
