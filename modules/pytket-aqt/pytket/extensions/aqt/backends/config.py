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

"""AQT config."""

from typing import Any, Dict, Optional, Type, ClassVar
from dataclasses import dataclass
from pytket.config import PytketExtConfig


@dataclass
class AQTConfig(PytketExtConfig):
    """Holds config parameters for pytket-aqt."""

    ext_dict_key: ClassVar[str] = "aqt"

    access_token: Optional[str]

    @classmethod
    def from_extension_dict(
        cls: Type["AQTConfig"], ext_dict: Dict[str, Any]
    ) -> "AQTConfig":
        return cls(ext_dict.get("access_token", None))


def set_aqt_config(
    access_token: Optional[str] = None,
) -> None:
    """Set default value for AQT API token."""
    config = AQTConfig.from_default_config_file()
    if access_token is not None:
        config.access_token = access_token
    config.update_default_config_file()
