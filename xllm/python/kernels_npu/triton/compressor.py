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


import torch
import triton
import triton.language as tl


@triton.jit
def _pooled_blocked(
    group_idx: tl.tensor,
    start_pos: tl.tensor,
    seg_row_base: tl.tensor,
    cache_row: tl.tensor,
    kv_ptr: tl.tensor,
    score_ptr: tl.tensor,
    kv_cache_ptr: tl.tensor,
    score_cache_ptr: tl.tensor,
    offs_h: tl.tensor,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
) -> tl.tensor:
    # Ascend can form masked DMA addresses before applying the mask; keep them within the allocation.
    score_max = tl.full((HEAD_DIM,), -float("inf"), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        slot = token_pos % CACHE_SIZE
        score_cache = tl.load(
            score_cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & (seg_off < 0))[:, None],
            other=0.0,
        )
        score = tl.where(in_seg[:, None], score_seg, score_cache)
        score = tl.where(row_valid[:, None], score, -float("inf"))
        score_max = tl.maximum(score_max, tl.max(score, axis=0))
    exp_sum = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        slot = token_pos % CACHE_SIZE
        score_cache = tl.load(
            score_cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & (seg_off < 0))[:, None],
            other=0.0,
        )
        score = tl.where(in_seg[:, None], score_seg, score_cache)
        score = tl.where(row_valid[:, None], score, -float("inf"))
        exp_sum += tl.sum(tl.exp(score - score_max[None, :]), axis=0)
    pooled = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        slot = token_pos % CACHE_SIZE
        score_cache = tl.load(
            score_cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & (seg_off < 0))[:, None],
            other=0.0,
        )
        score = tl.where(in_seg[:, None], score_seg, score_cache)
        score = tl.where(row_valid[:, None], score, -float("inf"))
        prob = tl.exp(score - score_max[None, :]) / exp_sum[None, :]
        kv_seg = tl.load(
            kv_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        slot = token_pos % CACHE_SIZE
        kv_cache = tl.load(
            kv_cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & (seg_off < 0))[:, None],
            other=0.0,
        )
        kv_vals = tl.where(in_seg[:, None], kv_seg, kv_cache)
        pooled += tl.sum(prob * kv_vals, axis=0)
    return pooled


@triton.jit
def _pool_kernel(
    out_ptr: tl.tensor,
    kv_ptr: tl.tensor,
    score_ptr: tl.tensor,
    kv_cache_ptr: tl.tensor,
    score_cache_ptr: tl.tensor,
    meta_ptr: tl.tensor,  # Packed metadata: [start_pos | used_len | out_row_offset | seg_row_base].
    block_table_ptr: tl.tensor,
    total_group_slots: tl.tensor,
    GROUP_SLOTS_PER_BATCH: tl.tensor,
    NUM_BATCH: tl.tensor,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    NUM_CACHE_BLOCKS: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_h = tl.arange(0, HEAD_DIM)
    for task_id in range(pid, total_group_slots, tl.num_programs(0)):
        batch = task_id // GROUP_SLOTS_PER_BATCH
        local_group = task_id - batch * GROUP_SLOTS_PER_BATCH
        start_pos = tl.load(meta_ptr + batch)
        used_len = tl.load(meta_ptr + NUM_BATCH + batch)
        groups_in_batch = (start_pos + used_len) // RATIO - start_pos // RATIO
        cache_row = tl.load(block_table_ptr + batch)
        valid = local_group < groups_in_batch
        valid = valid & (cache_row > 0) & (cache_row < NUM_CACHE_BLOCKS)
        if valid:
            group_idx = start_pos // RATIO + local_group
            seg_row_base = tl.load(meta_ptr + 3 * NUM_BATCH + batch)
            rows = tl.arange(0, CHUNK_ROWS)
            token_pos = group_idx * RATIO + rows
            seg_off = token_pos - start_pos
            residual = start_pos - group_idx * RATIO
            if residual > 0:
                pooled = _pooled_blocked(
                    group_idx,
                    start_pos,
                    seg_row_base,
                    cache_row,
                    kv_ptr,
                    score_ptr,
                    kv_cache_ptr,
                    score_cache_ptr,
                    offs_h,
                    CACHE_SIZE,
                    HEAD_DIM,
                    RATIO,
                    RATIO_PAD,
                    CHUNK_ROWS,
                )
            else:
                score = tl.load(score_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :])
                score_max = tl.max(score, axis=0)
                e = tl.exp(score - score_max[None, :])
                exp_sum = tl.sum(e, axis=0)
                prob = e / exp_sum[None, :]
                kv_vals = tl.load(kv_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :])
                pooled = tl.sum(prob * kv_vals, axis=0)
            out_row = seg_row_base + (group_idx + 1) * RATIO - 1 - start_pos
            tl.store(out_ptr + out_row * HEAD_DIM + offs_h, pooled)


@triton.jit
def _cache_update_kernel(
    kv_ptr: tl.tensor,
    score_ptr: tl.tensor,
    kv_cache_ptr: tl.tensor,
    score_cache_ptr: tl.tensor,
    meta_ptr: tl.tensor,
    block_table_ptr: tl.tensor,
    total_write_slots: tl.tensor,
    WRITE_SLOTS_PER_BATCH: tl.tensor,
    NUM_BATCH: tl.tensor,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_CACHE_BLOCKS: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_h = tl.arange(0, HEAD_DIM)
    for task_id in range(pid, total_write_slots, tl.num_programs(0)):
        batch = task_id // WRITE_SLOTS_PER_BATCH
        write_idx = task_id - batch * WRITE_SLOTS_PER_BATCH
        used_len = tl.load(meta_ptr + NUM_BATCH + batch)
        tail_count = used_len if used_len < CACHE_SIZE else CACHE_SIZE
        cache_row = tl.load(block_table_ptr + batch)
        valid = write_idx < tail_count
        valid = valid & (cache_row > 0) & (cache_row < NUM_CACHE_BLOCKS)
        if valid:
            token_in_seg = used_len - tail_count + write_idx
            proj_row = tl.load(meta_ptr + 3 * NUM_BATCH + batch) + token_in_seg
            token_pos = tl.load(meta_ptr + batch) + token_in_seg
            slot = token_pos % CACHE_SIZE
            kv_vals = tl.load(kv_ptr + proj_row * HEAD_DIM + offs_h)
            score_vals = tl.load(score_ptr + proj_row * HEAD_DIM + offs_h)
            tl.store(kv_cache_ptr + (cache_row * CACHE_SIZE + slot) * HEAD_DIM + offs_h, kv_vals)
            tl.store(score_cache_ptr + (cache_row * CACHE_SIZE + slot) * HEAD_DIM + offs_h, score_vals)


_CUBE_CORE_NUM: int | None = None


def _cube_core_num() -> int:
    global _CUBE_CORE_NUM
    if _CUBE_CORE_NUM is None:
        driver = triton.runtime.driver

        _CUBE_CORE_NUM = driver.active.utils.get_device_properties(torch.npu.current_device())["num_aicore"]
    return _CUBE_CORE_NUM


def compressor_from_projected(
    kv: torch.Tensor,
    scores: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    metadata: torch.Tensor,
    out: torch.Tensor,
    *,
    max_query_len: int,
    num_cores: int,
) -> torch.Tensor:
    if kv.dtype != torch.float32 or scores.dtype != torch.float32:
        raise ValueError("Aurora projections must be FP32")
    if kv_state.dtype != torch.float32 or score_state.dtype != torch.float32:
        raise ValueError("Aurora ring state must be FP32")
    if kv.ndim != 2 or scores.shape != kv.shape:
        raise ValueError("Expected matching [tokens, width] projections")
    tokens, width = kv.shape
    if width < 1 or width & (width - 1):
        raise ValueError("Compressor width must be a positive power of two")
    if kv_state.ndim != 3 or kv_state.shape[1:] != (32, width):
        raise ValueError("Expected [blocks, 32, width] kv ring state")
    if score_state.ndim != 3 or score_state.shape[1:] != (32, width):
        raise ValueError("Expected [blocks, 32, width] score ring state")
    if kv_state.shape[0] != score_state.shape[0]:
        raise ValueError("Compressor kv/score ring state must have the same block count")
    if out.dtype not in (torch.bfloat16, torch.float32) or out.shape != kv.shape:
        raise ValueError("Expected token-aligned BF16 or FP32 pooled output")
    if metadata.dtype != torch.int32 or metadata.ndim != 2 or metadata.shape[0] != 5:
        raise ValueError("Expected INT32 [5, requests] device metadata")
    tensors = (kv, scores, kv_state, score_state, metadata, out)
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("Compressor requires contiguous projections, ring pages and controls")
    if not all(t.device == kv.device for t in tensors):
        raise ValueError("Compressor tensors must share one device")
    if num_cores <= 0 or not 0 <= max_query_len <= tokens:
        raise ValueError("Invalid compressor launch bounds")
    out.zero_()
    batches = metadata.shape[1]
    if tokens == 0 or batches == 0 or max_query_len == 0:
        return out
    if kv.device.type != "npu":
        raise ValueError("Projected Triton compressor requires an NPU")
    groups = (max_query_len + 1) // 2
    writes = min(max_query_len, 32)
    # Residual reads must precede ring writes on the same stream.
    _pool_kernel[(min(num_cores, batches * groups),)](
        out,
        kv,
        scores,
        kv_state,
        score_state,
        metadata,
        metadata[4],
        batches * groups,
        groups,
        batches,
        CACHE_SIZE=32,
        HEAD_DIM=width,
        RATIO=2,
        RATIO_PAD=2,
        CHUNK_ROWS=2,
        NUM_CACHE_BLOCKS=kv_state.shape[0],
    )
    _cache_update_kernel[(min(num_cores, batches * writes),)](
        kv,
        scores,
        kv_state,
        score_state,
        metadata,
        metadata[4],
        batches * writes,
        writes,
        batches,
        CACHE_SIZE=32,
        HEAD_DIM=width,
        NUM_CACHE_BLOCKS=kv_state.shape[0],
    )
    return out
