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

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
from tests.python._deepseek_v41_fused_testkit import (
    _load_fused_ops_lib,
    _npu_ready,
    _rope_cache,
)


@pytest.mark.skipif(not _npu_ready() or not _load_fused_ops_lib(), reason="needs NPU + fused ops lib")
def test_interleaved_rope_fused_matches_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    device = "npu"
    cache = _rope_cache(256, 64, device)
    cases = [
        (8, 4, 512),
        (8, 1, 512),
        (16, 32, 128),
        (8, 512),
        (5, 128),
    ]
    for shape in cases:
        torch.manual_seed(sum(shape) + 7)
        x = (torch.randn(*shape) * 0.9).to(torch.bfloat16).to(device)
        positions = torch.randperm(200)[: shape[0]].to(torch.int32).to(device)

        with monkeypatch.context() as context:
            context.setattr(csa2, "_get_partial_rotary_op", lambda: None)
            eager = csa2.apply_interleaved_rope(x.clone(), positions, cache)
        fused = csa2.apply_interleaved_rope(x.clone(), positions, cache)

        assert fused.shape == eager.shape
        diff = (fused.float() - eager.float()).abs()
        assert (diff <= 2**-6 + 2**-6 * eager.float().abs()).all(), f"shape={shape} max_abs={diff.max().item():.3e}"
        rope_width = cache.size(-1)
        if shape[-1] > rope_width:
            assert torch.equal(fused[..., : shape[-1] - rope_width], eager[..., : shape[-1] - rope_width])

    tiny_cache = _rope_cache(64, 8, device)
    x = torch.randn(4, 2, 32, dtype=torch.bfloat16, device=device)
    positions = torch.arange(4, dtype=torch.int32, device=device)
    tiny = csa2.apply_interleaved_rope(x.clone(), positions, tiny_cache)
    with monkeypatch.context() as context:
        context.setattr(csa2, "_get_partial_rotary_op", lambda: None)
        tiny_ref = csa2.apply_interleaved_rope(x.clone(), positions, tiny_cache)
    assert torch.equal(tiny, tiny_ref)


@pytest.mark.skipif(not _npu_ready() or not _load_fused_ops_lib(), reason="needs NPU + fused ops lib")
def test_interleaved_rope_fused_inverse_matches_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    device = "npu"
    cache = _rope_cache(256, 64, device)
    cases = [
        (8, 4, 512),
        (5, 128),
    ]
    for shape in cases:
        torch.manual_seed(sum(shape) + 13)
        x = (torch.randn(*shape) * 0.9).to(torch.bfloat16).to(device)
        positions = torch.randperm(200)[: shape[0]].to(torch.int32).to(device)

        fused = csa2.apply_interleaved_rope(x.clone(), positions, cache, inverse=True)
        with monkeypatch.context() as context:
            context.setattr(csa2, "_get_partial_rotary_op", lambda: None)
            eager = csa2.apply_interleaved_rope(x.clone(), positions, cache, inverse=True)

        assert fused.shape == eager.shape
        diff = (fused.float() - eager.float()).abs()
        assert (diff <= 2**-6 + 2**-6 * eager.float().abs()).all(), (
            f"inverse shape={shape} max_abs={diff.max().item():.3e}"
        )
        rope_width = cache.size(-1)
        if shape[-1] > rope_width:
            assert torch.equal(fused[..., : shape[-1] - rope_width], eager[..., : shape[-1] - rope_width])


def _per_call_expansion(
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = cos_sin_cache.size(-1) // 2
    rows = cos_sin_cache.index_select(0, positions.reshape(-1).to(torch.int64))
    cos = rows[..., :half].repeat_interleave(2, dim=-1).to(dtype).contiguous()
    sin = rows[..., half:].repeat_interleave(2, dim=-1).to(dtype).contiguous()
    return cos, sin


def test_rope_preexpand_bitwise_matches_per_call() -> None:
    device = "npu" if _npu_ready() else "cpu"
    for max_pos, rotary_dim in [(256, 64), (4096, 64)]:
        cache = _rope_cache(max_pos, rotary_dim, device)
        for dtype in (torch.bfloat16, torch.float32):
            expanded = csa2._expanded_rope_tables(cache, dtype)
            assert expanded is not None
            cos_t, sin_t = expanded
            assert cos_t.shape == (max_pos, rotary_dim)
            assert cos_t.dtype == dtype and sin_t.dtype == dtype
            for seed in (1, 2, 3):
                gen = torch.Generator().manual_seed(seed)
                positions = torch.cat(
                    [
                        torch.randperm(max_pos, generator=gen)[:9],
                        torch.tensor([0, 0, max_pos - 1, max_pos - 1]),
                    ]
                ).to(device)
                cos_ref, sin_ref = _per_call_expansion(cache, positions, dtype)
                assert torch.equal(cos_t.index_select(0, positions), cos_ref)
                assert torch.equal(sin_t.index_select(0, positions), sin_ref)


def test_rope_preexpand_cache_memoization(monkeypatch: pytest.MonkeyPatch) -> None:
    device = "npu" if _npu_ready() else "cpu"
    cache = _rope_cache(128, 64, device)

    monkeypatch.setattr(csa2, "_is_stream_capturing", lambda: True)
    assert csa2._expanded_rope_tables(cache, torch.bfloat16) is None
    monkeypatch.setattr(csa2, "_is_stream_capturing", lambda: False)

    first = csa2._expanded_rope_tables(cache, torch.bfloat16)
    assert first is not None
    again = csa2._expanded_rope_tables(cache, torch.bfloat16)
    assert again[0] is first[0] and again[1] is first[1]
    other = csa2._expanded_rope_tables(cache, torch.float32)
    assert other is not None and other[0] is not first[0]
    cache2 = _rope_cache(128, 64, device)
    fresh = csa2._expanded_rope_tables(cache2, torch.bfloat16)
    assert fresh is not None and fresh[0] is not first[0]


@pytest.mark.skipif(not _npu_ready() or not _load_fused_ops_lib(), reason="needs NPU + fused ops lib")
def test_interleaved_rope_preexpand_fused_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    device = "npu"
    cache = _rope_cache(4096, 64, device)
    cases = [
        (1, 8, 512),
        (16, 8, 512),
        (16, 1, 512),
        (16, 32, 128),
        (16, 1, 128),
        (5, 128),
    ]
    for shape in cases:
        for seed in (1, 2):
            torch.manual_seed(sum(shape) * 10 + seed)
            x = (torch.randn(*shape) * 0.9).to(torch.bfloat16).to(device)
            positions = torch.randperm(cache.size(0) - shape[0] - 1)[: shape[0]].to(torch.int32).to(device)

            with monkeypatch.context() as context:
                context.setattr(csa2, "_expanded_rope_tables", lambda *a, **k: None)
                per_call = csa2.apply_interleaved_rope(x.clone(), positions, cache)
            cached = csa2.apply_interleaved_rope(x.clone(), positions, cache)
            assert torch.equal(cached, per_call), f"shape={shape} seed={seed}"


@pytest.mark.skipif(not _npu_ready() or not _load_fused_ops_lib(), reason="needs NPU + fused ops lib")
def test_interleaved_rope_preexpand_graph_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    device = "npu"
    cache = _rope_cache(4096, 64, device)
    shape = (16, 8, 512)
    torch.manual_seed(13)
    x = (torch.randn(*shape) * 0.9).to(torch.bfloat16).to(device)
    positions = (torch.arange(16, dtype=torch.int32, device=device) + 100).contiguous()

    for _ in range(3):
        csa2.apply_interleaved_rope(x.clone(), positions, cache)
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):
            csa2.apply_interleaved_rope(x, positions, cache)
    torch.npu.synchronize()
    with torch.npu.graph(graph, stream=stream):
        csa2.apply_interleaved_rope(x, positions, cache)

    torch.manual_seed(14)
    x2 = (torch.randn(*shape) * 0.9).to(torch.bfloat16).to(device)
    pos2 = (torch.arange(16, dtype=torch.int32, device=device) + 700).contiguous()
    x.copy_(x2)
    positions.copy_(pos2)
    graph.replay()
    torch.npu.synchronize()

    with monkeypatch.context() as context:
        context.setattr(csa2, "_expanded_rope_tables", lambda *a, **k: None)
        expected = csa2.apply_interleaved_rope(x2.clone(), pos2, cache)
    assert torch.equal(x, expected)
