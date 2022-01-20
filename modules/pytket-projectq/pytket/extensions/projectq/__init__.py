# Copyright 2019-2022 Cambridge Quantum Computing
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
"""Module for conversion between ProjectQ and tket primitives."""

# _metadata.py is copied to the folder after installation.
from ._metadata import __extension_version__, __extension_name__  # type: ignore
from .backends import ProjectQBackend
from .projectq_convert import tk_to_projectq, tketBackendEngine, tketOptimiser
