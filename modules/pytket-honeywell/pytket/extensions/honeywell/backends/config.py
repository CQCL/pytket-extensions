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

"""Honeywell config."""

from typing import Any, Dict, Optional, Type, ClassVar
from dataclasses import dataclass
from pytket.config import PytketExtConfig


@dataclass
class HoneywellConfig(PytketExtConfig):
    """Holds config parameters for pytket-honeywell."""

    ext_dict_key: ClassVar[str] = "honeywell"

    username: Optional[str]

    @classmethod
    def from_extension_dict(
        cls: Type["HoneywellConfig"], ext_dict: Dict[str, Any]
    ) -> "HoneywellConfig":
        return cls(ext_dict.get("username", None))


def set_honeywell_config(username: Optional[str]) -> None:
    """Set default value for HQS username.
    Can be overriden in backend construction."""
    hconfig = HoneywellConfig.from_default_config_file()
    hconfig.username = username
    hconfig.update_default_config_file()
