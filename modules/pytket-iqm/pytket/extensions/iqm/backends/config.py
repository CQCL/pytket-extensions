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

"""IQM config."""

from typing import Any, Dict, Optional, Type, ClassVar
from dataclasses import dataclass
from pytket.config import PytketExtConfig


@dataclass
class IQMConfig(PytketExtConfig):
    """Holds config parameters for pytket-iqm."""

    ext_dict_key: ClassVar[str] = "iqm"

    auth_server_url: Optional[str]
    username: Optional[str]
    password: Optional[str]

    @classmethod
    def from_extension_dict(
        cls: Type["IQMConfig"], ext_dict: Dict[str, Any]
    ) -> "IQMConfig":
        return cls(
            ext_dict.get("auth_server_url"),
            ext_dict.get("username"),
            ext_dict.get("password"),
        )


def set_iqm_config(
    auth_server_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """Set default value for IQM API token."""
    config = IQMConfig.from_default_config_file()
    if auth_server_url is not None:
        config.auth_server_url = auth_server_url
    if username is not None:
        config.username = username
    if password is not None:
        config.password = password
    config.update_default_config_file()
