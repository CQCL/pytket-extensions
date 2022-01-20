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

"""IonQ config."""

from typing import Any, Dict, Optional, Type, ClassVar
from dataclasses import dataclass
from pytket.config import PytketExtConfig


@dataclass
class IonQConfig(PytketExtConfig):
    """Holds config parameters for pytket-ionq."""

    ext_dict_key: ClassVar[str] = "ionq"

    api_key: Optional[str]

    @classmethod
    def from_extension_dict(
        cls: Type["IonQConfig"], ext_dict: Dict[str, Any]
    ) -> "IonQConfig":
        return cls(ext_dict.get("api_key", None))


def set_ionq_config(
    api_key: Optional[str] = None,
) -> None:
    """Set default value for IonQ API key."""
    config = IonQConfig.from_default_config_file()
    if api_key is not None:
        config.api_key = api_key
    config.update_default_config_file()
