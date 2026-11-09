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

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS"])
def _quantize_rows(
    x: tl.tensor,
    codes: tl.tensor,
    scales: tl.tensor,
    ROWS: tl.tensor,
    WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
) -> None:
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    column = tl.arange(0, WIDTH)
    value = tl.load(x + row[:, None] * WIDTH + column[None, :], row[:, None] < ROWS, 0).to(tl.float32)
    maximum = tl.max(tl.abs(value), 1)
    # Preserve the FP16 scale rounding before normalization.
    scale = tl.maximum((maximum / 127.0).to(tl.float16).to(tl.float32), 2.0**-24)
    rounded = tl.extra.cann.libdevice.nearbyint(value / scale[:, None])
    result = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    tl.store(codes + row[:, None] * WIDTH + column[None, :], result, row[:, None] < ROWS)
    tl.store(scales + row, scale.to(tl.float16), row < ROWS)


def quantize_indexer_rows_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.device.type not in ("npu", "privateuseone") or x.dtype != torch.bfloat16 or x.ndim < 2:
        raise ValueError("Expected an NPU BF16 matrix or batch of matrices")
    if not x.is_contiguous() or x.size(-1) != 128:
        raise ValueError("Expected contiguous indexer rows with width 128")
    codes = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1], dtype=torch.float16, device=x.device)
    rows = x.numel() // 128
    if rows:
        block_rows = 4 if rows <= 256 else 32
        _quantize_rows[(triton.cdiv(rows, block_rows),)](
            x, codes, scales, ROWS=rows, WIDTH=128, BLOCK_ROWS=block_rows, enable_fp_fusion=False
        )
    return codes, scales
