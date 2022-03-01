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

from typing import Tuple, Optional
import pytest

from requests_mock.mocker import Mocker

from pytket.extensions.honeywell.backends.api_wrappers import HoneywellQAPI
from pytket.extensions.honeywell.backends import HoneywellBackend
from pytket.circuit import Circuit  # type: ignore


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


@pytest.mark.parametrize(
    "chosen_device,max_batch_cost",
    [("HQS-LT", 300), ("HQS-LT", None), ("HQS-LT-S1", 300), ("HQS-LT-S1", None)],
)
def test_device_family(
    requests_mock: Mocker,
    mock_hqs_api_handler: HoneywellQAPI,
    chosen_device: str,
    max_batch_cost: Optional[int],
) -> None:
    """Test that batch params are NOT supplied by default
    if we are submitting to a device family.
    Doing so will get an error response from the HQS API."""

    fake_job_id = "abc-123"

    requests_mock.register_uri(
        "POST",
        "https://qapi.honeywell.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    requests_mock.register_uri(
        "GET",
        f"https://qapi.honeywell.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    family_backend = HoneywellBackend(
        device_name=chosen_device,
        machine_debug=True,
    )
    family_backend._api_handler = mock_hqs_api_handler

    circ = Circuit(2, name="batching_test").H(0).CX(0, 1).measure_all()
    circ = family_backend.get_compiled_circuit(circ)

    kwargs = {}
    if max_batch_cost is not None:
        kwargs["max_batch_cost"] = max_batch_cost
    family_backend.process_circuits(
        circuits=[circ, circ], n_shots=10, valid_check=False, **kwargs
    )

    submitted_json = {}
    if requests_mock.last_request:
        submitted_json = requests_mock.last_request.json()

    if chosen_device == "HQS-LT" and max_batch_cost is None:
        assert "batch-exec" not in submitted_json.keys()
        assert "batch-end" not in submitted_json.keys()
    else:
        assert "batch-exec" in submitted_json.keys()
        assert "batch-end" in submitted_json.keys()
