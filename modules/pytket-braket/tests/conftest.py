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

from typing import Optional
import os
import pytest
import boto3
from braket.aws.aws_session import AwsSession # type: ignore
from pytket.extensions.braket import BraketBackend
from _pytest.fixtures import SubRequest


def get_authenticated_aws_session(region: Optional[str] = None) -> Optional[AwsSession]:
    if os.getenv("PYTKET_RUN_REMOTE_TESTS") is not None:
        # Authenticated AwsSession used in the authenticated_braket_backend
        # The values for the access key id and secret access key are taken from the
        # following env variables:
        #  - PYTKET_REMOTE_BRAKET_ACCESS_KEY_ID
        #  - PYTKET_REMOTE_BRAKET_ACCESS_KEY_SECRET
        # If no region is specified, the value is taken from the env variable:
        #  - PYTKET_REMOTE_BRAKET_REGION
        # Note: this session fixture should be used when creating backends for tests
        #       where PYTKET_RUN_REMOTE_TESTS is true
        region_name = region or os.getenv("PYTKET_REMOTE_BRAKET_REGION")
        print(f"region_name: {region_name}")
        boto_session = boto3.Session(
            aws_access_key_id=os.getenv("PYTKET_REMOTE_BRAKET_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("PYTKET_REMOTE_BRAKET_ACCESS_KEY_SECRET"),
            region_name=region_name,
        )

        return AwsSession(boto_session=boto_session)
    return None


@pytest.fixture(name="authenticated_braket_backend")
def fixture_authenticated_braket_backend(
    request: SubRequest,
) -> BraketBackend:
    # Authenticated BraketBackend used for the remote tests
    # The S3 bucket and folder values are taken from the env variables:
    # PYTKET_REMOTE_BRAKET_BUCKET and PYTKET_REMOTE_BRAKET_FOLDER
    # Note: this fixture should only be used in tests where PYTKET_RUN_REMOTE_TESTS
    #       is true, by marking it with @parametrize, using the
    #       "authenticated_braket_backend" as parameter and `indirect=True`.
    if request.param is None:
        authenticated_aws_session = get_authenticated_aws_session()
        backend = BraketBackend(
            s3_bucket=os.getenv("PYTKET_REMOTE_BRAKET_BUCKET"),
            s3_folder=os.getenv("PYTKET_REMOTE_BRAKET_FOLDER"),
            aws_session=authenticated_aws_session,
        )
    else:
        authenticated_aws_session = get_authenticated_aws_session(
            region=request.param.get("region")
        )
        if not "s3_bucket" in request.param or not "s3_folder" in request.param:
            backend = BraketBackend(
                **request.param,
                s3_bucket=os.getenv("PYTKET_REMOTE_BRAKET_BUCKET"),
                s3_folder=os.getenv("PYTKET_REMOTE_BRAKET_FOLDER"),
                aws_session=authenticated_aws_session,
            )
        else:
            backend = BraketBackend(
                **request.param, aws_session=authenticated_aws_session
            )

    return backend
