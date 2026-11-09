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

from dataclasses import dataclass
from typing import Sequence

import torch

from xllm.python import distributed
from xllm.python.model_executor.forward_context import get_forward_context, get_forward_context_or_none


@dataclass(frozen=True)
class DsaCpContext:
    world_size: int
    rank: int
    num_tokens: int
    tokens_per_rank: int
    start: int
    end: int
    global_q_lens: tuple[int, ...]
    local_q_lens: tuple[int, ...]
    query_suffix: tuple[int, ...]

    def shard(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self.start : self.end]

    def local_context_lengths(self, global_lengths: torch.Tensor) -> torch.Tensor:
        if all(length == 1 for length in self.global_q_lens):
            # Decode graph extents are fixed; derive changing owner masks from device-side lengths.
            rows = torch.arange(len(self.global_q_lens), device=global_lengths.device)
            active = (rows >= self.start) & (rows < self.end)
            return torch.where(active, global_lengths.reshape(-1), 0)
        suffix = global_lengths.new_tensor(self.query_suffix)
        lengths = (global_lengths.reshape(-1) - suffix).clamp_min(0)
        active = global_lengths.new_tensor(self.local_q_lens) > 0
        return torch.where(active, lengths, 0)

    def restore_tp_heads(self, local_output: torch.Tensor) -> torch.Tensor:
        tokens, heads, width = local_output.shape
        if heads % self.world_size or tokens != self.end - self.start:
            raise ValueError("DSA CP output does not match its token/head plan")
        padded = local_output
        if tokens < self.tokens_per_rank:
            padded = local_output.new_zeros((self.tokens_per_rank, heads, width))
            padded[:tokens] = local_output
        local_heads = heads // self.world_size
        send = (
            padded.view(self.tokens_per_rank, self.world_size, local_heads, width)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(-1, local_heads, width)
        )
        received = distributed.all_to_all(send, self.world_size, "tp")
        return received[: self.num_tokens]


@dataclass(frozen=True)
class DsaCpRuntime:
    plan: DsaCpContext
    local_context_lengths: torch.Tensor


def get_current_dsa_cp_runtime() -> DsaCpRuntime | None:
    context = get_forward_context_or_none()
    runtime = context.layer_shared_cache.get(DsaCpRuntime) if context is not None else None
    if runtime is not None and not isinstance(runtime, DsaCpRuntime):
        raise TypeError("Invalid DSA CP forward context")
    return runtime


def get_dsa_cp_runtime(num_tokens: int, world_size: int, rank: int) -> DsaCpRuntime:
    context = get_forward_context()
    runtime = get_current_dsa_cp_runtime()
    if runtime is None:
        dsa = context.metadata.dsa_metadata
        q_lens = [1] * dsa.seq_lens_q.numel() if getattr(dsa, "is_acl_graph", False) else dsa.seq_lens_q_host
        plan = build_dsa_cp_context(num_tokens, q_lens, world_size, rank)
        global_lengths = torch.as_tensor(dsa.seq_lens, device=context.device)
        runtime = DsaCpRuntime(plan, plan.local_context_lengths(global_lengths))
        # Cache per ForwardContext, not on device buffers reused by later graph entries.
        context.layer_shared_cache[DsaCpRuntime] = runtime
    return runtime


def build_dsa_cp_context(
    num_tokens: int,
    q_lens: Sequence[int],
    world_size: int,
    rank: int,
) -> DsaCpContext:
    if world_size <= 0 or not 0 <= rank < world_size or num_tokens < 0:
        raise ValueError("Invalid DSA CP topology or token count")
    lengths = tuple(int(value) for value in q_lens)
    if any(value < 0 for value in lengths) or sum(lengths) > num_tokens:
        raise ValueError("DSA CP query lengths exceed the packed token buffer")
    per_rank = (num_tokens + world_size - 1) // world_size
    actual_tokens = sum(lengths)
    start = min(rank * per_rank, actual_tokens)
    end = min((rank + 1) * per_rank, actual_tokens)
    local_lengths, suffix = [], []
    offset = 0
    for length in lengths:
        query_end = offset + length
        local_start = min(max(offset, start), end)
        local_end = min(max(query_end, start), end)
        local_lengths.append(local_end - local_start)
        suffix.append(max(0, query_end - local_end))
        offset = query_end
    return DsaCpContext(
        world_size, rank, num_tokens, per_rank, start, end, lengths, tuple(local_lengths), tuple(suffix)
    )
