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

from typing import Tuple

from requests_mock.mocker import Mocker

from pytket.extensions.honeywell.backends.api_wrappers import HoneywellQAPI


def test_hqs_login(
    mock_hqs_api_handler: HoneywellQAPI,
    mock_credentials: Tuple[str, str],
    mock_token: str,
) -> None:
    """Test that credentials are storable and deletable using
    the HoneywellQAPI handler."""

    username, pwd = mock_credentials

    # Check credentials are retrievable
    assert mock_hqs_api_handler._cred_store.login_credential(username) == pwd
    assert mock_hqs_api_handler._cred_store.refresh_token == mock_token
    assert mock_hqs_api_handler._cred_store.id_token == mock_token

    # Delete authentication and verify
    mock_hqs_api_handler.delete_authentication()
    assert mock_hqs_api_handler._cred_store.id_token == None
    assert mock_hqs_api_handler._cred_store.login_credential(username) == None
    assert mock_hqs_api_handler._cred_store.refresh_token == None


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

    # Delete authentication tokens to clean them from the keyring
    mock_hqs_api_handler.delete_authentication()
