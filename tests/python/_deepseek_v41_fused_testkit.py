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

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import xllm.python.models.deepseek_v41 as v41
from xllm.python.model_executor.forward_context import ForwardContext

_MODE_EAGER = v41._HC_PRE_MODE_EAGER
_MODE_FUSED = v41._HC_PRE_MODE_FUSED


def _npu_ready() -> bool:
    npu = getattr(torch, "npu", None)
    return npu is not None and npu.is_available()


def _load_fused_ops_lib() -> bool:
    return all(
        getattr(torch.ops.xllm_ops, name, None) is not None
        for name in ("hc_post", "hc_pre", "rms_norm", "npu_inplace_partial_rotary_mul")
    )


def _dp_forward_context(counts: list[int]) -> ForwardContext:
    metadata = SimpleNamespace(dp_execution_token_counts=list(counts))
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
    )


def _reference_indexer_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.float().abs().amax(dim=-1) / 127.0).half().float().clamp_min(2.0**-24)
    codes = torch.round(x.float() / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return codes, scale.half()


def _rope_cache(max_pos: int, rotary_dim: int, device: str = "cpu") -> torch.Tensor:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device).contiguous()
