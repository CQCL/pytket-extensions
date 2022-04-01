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

# NB: This test has been placed in a separate file from api_test.py to work around an
# issue on the MacOS CI, whereby pytest would hang indefinitely after the collection
# phase.

from typing import Optional
import pytest

from requests_mock.mocker import Mocker

from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends import QuantinuumBackend
from pytket.circuit import Circuit  # type: ignore


@pytest.mark.parametrize(
    "chosen_device,max_batch_cost",
    [("H1", 300), ("H1", None), ("H1-1", 300), ("H1-1", None)],
)
def test_device_family(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
    chosen_device: str,
    max_batch_cost: Optional[int],
) -> None:
    """Test that batch params are NOT supplied by default
    if we are submitting to a device family.
    Doing so will get an error response from the Quantinuum API."""

    fake_job_id = "abc-123"

    requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    family_backend = QuantinuumBackend(
        device_name=chosen_device,
    )
    family_backend._api_handler = mock_quum_api_handler

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

    if chosen_device == "H1" and max_batch_cost is None:
        assert "batch-exec" not in submitted_json.keys()
        assert "batch-end" not in submitted_json.keys()
    else:
        assert "batch-exec" in submitted_json.keys()
        assert "batch-end" in submitted_json.keys()


def test_resumed_batching(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
) -> None:
    """Test that you can resume using a batch."""

    fake_job_id = "abc-123"

    requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    backend = QuantinuumBackend(
        device_name="H1-1E",
    )
    backend._api_handler = mock_quum_api_handler

    circ = Circuit(2, name="batching_test").H(0).CX(0, 1).measure_all()
    circ = backend.get_compiled_circuit(circ)

    [h1, _] = backend.process_circuits(
        circuits=[circ, circ], n_shots=10, valid_check=False, close_batch=False
    )

    submitted_json = {}
    if requests_mock.last_request:
        print(requests_mock.last_request)
        submitted_json = requests_mock.last_request.json()

    assert "batch-exec" in submitted_json
    assert submitted_json["batch-exec"] == backend.get_jobid(h1)
    assert "batch-end" not in submitted_json

    _ = backend.process_circuit(
        circ, n_shots=10, valid_check=False, batch_id=backend.get_jobid(h1)
    )

    if requests_mock.last_request:
        submitted_json = requests_mock.last_request.json()
    assert submitted_json["batch-exec"] == backend.get_jobid(h1)
    assert "batch-end" in submitted_json
