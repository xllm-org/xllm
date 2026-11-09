# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def _has_registered_op_api(probe: Callable[[str], bool], op_name: str) -> bool:
    return bool(probe(op_name))


def has_op_api(op_name: str) -> bool:
    """Require both ACLNN entry points; a missing Torch binding may appear after native registration."""
    probe = getattr(torch.ops.xllm_ops, "has_npu_op_api", None)
    return probe is not None and _has_registered_op_api(probe, op_name)
