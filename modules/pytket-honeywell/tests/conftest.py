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

import pytest
from requests_mock.mocker import Mocker
import jwt

from pytket.extensions.honeywell.backends.api_wrappers import HoneywellQAPI

# pylint: disable=R0801
@pytest.fixture(name="mock_hqs_api_handler")
def fixture_mock_hqs_api_handler(
    requests_mock: Mocker,
) -> HoneywellQAPI:
    """A logged-in HoneywellQAPI fixture."""

    username = "mark.honeywell@mail.com"
    pwd = "1906"

    id_payload = {"exp": (datetime.datetime.now().timestamp() * 2)}

    mock_id_token = jwt.encode(id_payload, key="").decode("utf-8")

    # Mock /login endpoint (_request_tokens)
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

    # Construct HoneywellQAPI and login
    # pylint: disable=E1123
    api_handler = HoneywellQAPI(
        user_name=username,
        token="",
        persistent_credential=False,
        login=True,
        _HoneywellQAPI__pwd=pwd,  # type: ignore
    )

    return api_handler
