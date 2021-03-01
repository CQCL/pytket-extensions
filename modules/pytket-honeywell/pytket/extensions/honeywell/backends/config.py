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

from typing import Dict, Optional, cast, Type, TypeVar
from dataclasses import dataclass
from pytket.config import PytketConfig, PytketExtConfig


T = TypeVar("T", bound="HoneywellConfig")


@dataclass
class HoneywellConfig(PytketExtConfig):
    username: Optional[str]

    @classmethod
    def from_pytketconfig(cls: Type[T], config: PytketConfig) -> T:
        if config.extensions and "honeywell" in config.extensions:
            config_dict = cast(Dict[str, str], config.extensions["honeywell"])
            return cls(config_dict.get("username"))

        return cls(None)
