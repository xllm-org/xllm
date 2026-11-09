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

"""NPU hyper-connection kernels for multi-stream residual (mHC) layers.

``hc_pre`` fuses rsqrt RMS + linear projection + sigmoid gating + Sinkhorn
normalisation + weighted-sum reduction into a single call. ``hc_post``
applies the post-attention/FFN recombination (post * x + comb^T @ residual).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from xllm.python.kernels_npu.op_api import has_op_api
from xllm.python.platform import current_platform

try:
    hc_pre = torch.ops.xllm_ops.hc_pre
except (AttributeError, RuntimeError):
    hc_pre = None

try:
    hc_post = torch.ops.xllm_ops.hc_post
except (AttributeError, RuntimeError):
    hc_post = None


# Resolve lazily because Torch operators may be registered after this module loads.
_hc_pre_va_: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None


def _resolve_hc_pre_va() -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None:
    global _hc_pre_va_
    if _hc_pre_va_ is not None:
        return _hc_pre_va_
    va_op = getattr(torch.ops.xllm_ops, "hc_pre_va", None)
    if va_op is not None and current_platform.get_ascend_soc_generation() == "a3" and has_op_api("aclnnHcPreVa"):
        _hc_pre_va_ = va_op
        return _hc_pre_va_
    return None


def __getattr__(
    name: str,
) -> Callable[..., torch.Tensor] | Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None:
    if name == "hc_pre_va":
        return _resolve_hc_pre_va()
    if name == "hc_post_va":
        if has_op_api("aclnnHcPostVa"):
            return getattr(torch.ops.xllm_ops, "hc_post_va", None)
        return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "hc_pre",
    "hc_pre_va",  # noqa: F822
    "hc_post",
    "hc_post_va",  # noqa: F822
]
