# Copyright 2021-2022 Cambridge Quantum Computing
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

from typing import Any, ClassVar, Dict, Optional, Type
from dataclasses import dataclass
from pytket.config import PytketExtConfig


@dataclass
class BraketConfig(PytketExtConfig):
    """Holds config parameters for pytket-braket."""

    ext_dict_key: ClassVar[str] = "braket"

    s3_bucket: Optional[str]
    s3_folder: Optional[str]
    device_type: Optional[str]
    provider: Optional[str]

    @classmethod
    def from_extension_dict(
        cls: Type["BraketConfig"], ext_dict: Dict[str, Any]
    ) -> "BraketConfig":
        return cls(
            ext_dict.get("s3_bucket", None),
            ext_dict.get("s3_folder", None),
            ext_dict.get("device_type", None),
            ext_dict.get("provider", None),
        )


def set_braket_config(
    s3_bucket: Optional[str] = None,
    s3_folder: Optional[str] = None,
    device_type: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    """Set default values for any of s3_bucket, s3_folder, device_type or provider
    for AWS Braket. Can be overridden in backend construction."""
    config = BraketConfig.from_default_config_file()
    if s3_bucket is not None:
        config.s3_bucket = s3_bucket
    if s3_folder is not None:
        config.s3_folder = s3_folder
    if device_type is not None:
        config.device_type = device_type
    if provider is not None:
        config.provider = provider
    config.update_default_config_file()
