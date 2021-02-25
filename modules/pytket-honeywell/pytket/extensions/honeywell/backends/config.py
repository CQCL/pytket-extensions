# Copyright 2020-2021 Cambridge Quantum Computing
#
# You may not use this file except in compliance with the Licence.
# You may obtain a copy of the Licence in the LICENCE file accompanying
# these documents or at:
#
#     https://cqcl.github.io/pytket/build/html/licence.html
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
