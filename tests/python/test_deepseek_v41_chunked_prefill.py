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

from dataclasses import replace
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
from tests.python._deepseek_v41_fused_testkit import _reference_indexer_quant
from tests.python.test_deepseek_v41_model import (
    _DSV41_FORWARD_CONFIG,
    _active_device,
    _allocate_v41_layer_caches,
    _make_csa2_backend,
    _make_forward_lm,
    _random_init_v41,
    _V41ForwardHarness,
    npu_only,
)
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_attention import (
    Csa2AttentionBackend,
    Csa2LayerContext,
    CSA2SharedRuntime,
    _quantize_indexer_rows_int8,
    _window_visible_idxs,
    apply_interleaved_rope,
)
from xllm.python.attention.csa_attention import _scatter_by_slot
from xllm.python.attention.dsa_metadata import DsaMetadata, last_ring_writes
from xllm.python.model_executor.dsa_cp import DsaCpContext, DsaCpRuntime, build_dsa_cp_context
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models.deepseek_v41 import (
    CSA2_MODE_FULL,
    CSA2_MODE_SWA,
    CSA2LayerMode,
    DeepseekV41Attention,
    DeepseekV41Config,
    DeepseekV41ForCausalLM,
    DeepseekV41HyperConnection,
)

_STATE_ROWS = 32


def _assert_model_logits_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        return
    assert torch.isfinite(actual).all()
    error = (actual - expected).abs()
    assert error.max() <= 0.0625
    relative_rms = error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-6)
    assert relative_rms <= 0.005
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
    assert cosine.min() >= 0.999


def _poison_v41_caches(caches: list[LayerCache]) -> None:
    for cache in caches:
        for name in ("swa", "key", "index", "indexer_scale", "compress_kv_state", "compress_score_state"):
            tensor = getattr(cache, name)
            if tensor is not None:
                tensor.fill_(-128 if tensor.dtype == torch.int8 else float("nan"))


def _make_swa_backend(window: int, width: int, heads: int) -> Csa2AttentionBackend:
    return Csa2AttentionBackend(
        compress_ratios=[0],
        window_size=window,
        n_layers=1,
        num_heads=heads,
        attn_head_dim=width,
        index_topk=8,
        index_n_heads=2,
        index_head_dim=width,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=8,
        candidate_block_size=2,
    )


def _swa_layer(sink: torch.Tensor | None = None, *, cp: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        mode=CSA2LayerMode(layer_id=0, mode=CSA2_MODE_SWA, compress_ratio=0, main_kv_source_layer=-1),
        layer_id=0,
        attn_sink=torch.zeros(1) if sink is None else sink,
        attn_sink_loaded=sink is not None,
        enable_dsa_cp=cp,
    )


def _reference_swa(
    q: torch.Tensor,
    kv: torch.Tensor,
    prefix: int,
    window: int,
    sink: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = torch.einsum("shd,nd->shn", q, kv) * kv.size(-1) ** -0.5
    positions = prefix + torch.arange(q.size(0))
    columns = torch.arange(kv.size(0))
    visible = (columns <= positions[:, None]) & (columns >= (positions[:, None] - window + 1).clamp_min(0))
    scores.masked_fill_(~visible[:, None, :], float("-inf"))
    maximum = scores.amax(-1, keepdim=True)
    probs = torch.exp(scores - maximum)
    denominator = probs.sum(-1, keepdim=True)
    if sink is not None:
        denominator = denominator + torch.exp(sink[None, :, None] - maximum)
    return torch.einsum("shn,nd->shd", probs, kv) / denominator


def _ring_slots(table_row: torch.Tensor, positions: torch.Tensor, block: int) -> torch.Tensor:
    return table_row[positions // block % table_row.numel()].long() * block + positions % block


def _state_slots(table_row: torch.Tensor, positions: torch.Tensor, window: int) -> torch.Tensor:
    blocks = table_row[positions // window % table_row.numel()].long()
    return blocks * _STATE_ROWS + positions % _STATE_ROWS


@pytest.mark.parametrize(
    "prefixes,chunks",
    [([0], [1]), ([7], [2]), ([23], [33]), ([0, 8], [5, 17]), ([15, 1, 32], [1, 35, 3])],
)
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("unwritten_value", [0.0, float("nan")], ids=["zero-cache", "uninitialized-cache"])
def test_swa_chunk_reads_history_before_ring_wrap(
    prefixes: list[int], chunks: list[int], cp_size: int, unwritten_value: float
) -> None:
    window, width, heads = 8, 4, 4
    batch = len(prefixes)
    torch.manual_seed(817)
    values = [torch.randn(p + c + 3, width) for p, c in zip(prefixes, chunks)]
    queries = [torch.randn(c + 3, heads, width) for c in chunks]
    sink = torch.randn(heads)
    backend = _make_swa_backend(window, width, heads)
    cache = torch.full((1 + batch, window, width), unwritten_value)
    raw_table = torch.arange(1, 1 + batch, dtype=torch.int32).view(batch, 1)
    backend.bind_kv_caches([LayerCache(key=None, value=None, swa=cache)])
    for seq, prefix in enumerate(prefixes):
        pos = torch.arange(max(0, prefix - window), prefix)
        _scatter_by_slot(cache, _ring_slots(raw_table[seq], pos, window), values[seq][pos])
    layer = _swa_layer(sink, cp=cp_size > 1)
    contexts = list(prefixes)
    for step in range(4):
        lengths = chunks if step == 0 else [1] * batch
        q_parts = [
            queries[s][: chunks[s]] if step == 0 else queries[s][chunks[s] + step - 1 : chunks[s] + step]
            for s in range(batch)
        ]
        q = torch.cat(q_parts)
        kv = torch.cat([values[s][contexts[s] : contexts[s] + lengths[s]] for s in range(batch)])
        positions = torch.cat([torch.arange(contexts[s], contexts[s] + lengths[s]) for s in range(batch)])
        next_contexts = [contexts[s] + lengths[s] for s in range(batch)]
        layer._csa2_ctx = Csa2LayerContext(kv, kv, positions, torch.empty(0))
        dsa = backend._builder.build(
            [raw_table],
            next_contexts,
            lengths,
            positions,
            step == 0 and not any(prefixes),
            step == 0,
        )
        metadata = SimpleNamespace(dsa_metadata=dsa)
        backend.prepare(metadata)
        expected = torch.cat(
            [_reference_swa(q_parts[s], values[s][: next_contexts[s]], contexts[s], window, sink) for s in range(batch)]
        )
        cache_before = cache.clone()
        for rank in range(cp_size):
            cache.copy_(cache_before)
            plan = build_dsa_cp_context(sum(lengths), lengths, cp_size, rank) if cp_size > 1 else None
            context = ForwardContext(backend, torch.device("cpu"), metadata, [])
            if plan is not None:
                context.layer_shared_cache[DsaCpRuntime] = DsaCpRuntime(plan, plan.local_context_lengths(dsa.seq_lens))
            with torch.inference_mode(), forward_context(context):
                actual = backend.execute_csa2_layer(plan.shard(q) if plan is not None else q, kv, layer)
            torch.testing.assert_close(
                actual, plan.shard(expected) if plan is not None else expected, atol=2e-6, rtol=2e-5
            )
        for seq, end in enumerate(next_contexts):
            pos = torch.arange(max(0, end - window), end)
            torch.testing.assert_close(
                cache.view(-1, width)[_ring_slots(raw_table[seq], pos, window)], values[seq][pos], atol=0, rtol=0
            )
        contexts = next_contexts


@pytest.mark.parametrize("prefix", [15, 23])
@pytest.mark.parametrize("q_len", [1, 3])
@pytest.mark.parametrize("unwritten_value", [float("nan"), float("inf")])
def test_swa_retired_page_cannot_contaminate_visible_history(prefix: int, q_len: int, unwritten_value: float) -> None:
    window, width, heads = 8, 4, 2
    backend = _make_swa_backend(window, width, heads)
    torch.manual_seed(311)
    values = torch.randn(prefix + q_len, width)
    q = torch.randn(q_len, heads, width)
    cache = torch.full((3, window, width), unwritten_value)
    table = torch.full((1, 2), -1, dtype=torch.int32)
    active_column = (prefix // window) % 2
    table[0, active_column] = 1
    retired_block = 0
    if q_len > 1:
        table[0, 1 - active_column] = 2
        retired_block = 2
    history = torch.arange(prefix - window + 1, prefix)
    _scatter_by_slot(cache, _ring_slots(table[0], history, window), values[history])
    backend.bind_kv_caches([LayerCache(None, None, swa=cache)])
    layer = _swa_layer(torch.randn(heads))
    positions = torch.arange(prefix, prefix + q_len)
    layer._csa2_ctx = Csa2LayerContext(values[prefix:], values[prefix:], positions, torch.empty(0))
    dsa = backend._builder.build([table], [prefix + q_len], [q_len], positions, False, q_len > 1)
    metadata = SimpleNamespace(dsa_metadata=dsa)
    backend.prepare(metadata)
    with torch.inference_mode(), forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        actual = backend.execute_csa2_layer(q, values[prefix:], layer)
    expected = _reference_swa(q, values, prefix, window, layer.attn_sink)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        cache[retired_block, -1],
        torch.full((width,), unwritten_value),
        atol=0,
        rtol=0,
        equal_nan=True,
    )


def test_history_masking_is_exact_not_magnitude_clipped() -> None:
    window, width, heads, prefix, q_len = 8, 4, 2, 8, 3
    backend = _make_swa_backend(window, width, heads)
    torch.manual_seed(311)
    kv_chunk = torch.full((q_len, width), 2e-4)
    values = torch.cat((torch.randn(prefix, width) * 50.0, kv_chunk))
    q = torch.randn(q_len, heads, width)
    cache = torch.full((2, window, width), float("nan"))
    table = torch.tensor([[1]], dtype=torch.int32)
    pos = torch.arange(0, prefix)
    _scatter_by_slot(cache, _ring_slots(table[0], pos, window), values[pos])
    backend.bind_kv_caches([LayerCache(None, None, swa=cache)])
    layer = _swa_layer(torch.randn(heads))
    layer._csa2_ctx = Csa2LayerContext(
        values[prefix:], values[prefix:], torch.arange(prefix, prefix + q_len), torch.empty(0)
    )
    dsa = backend._builder.build([table], [prefix + q_len], [q_len], torch.arange(prefix, prefix + q_len), False, False)
    metadata = SimpleNamespace(dsa_metadata=dsa)
    backend.prepare(metadata)
    with torch.inference_mode(), forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        actual = backend.execute_csa2_layer(q, values[prefix:], layer)
    expected = _reference_swa(q, values, prefix, window, layer.attn_sink)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_dense_scores_are_bounded_per_attention_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    window, width, heads, topk, length = 8, 8, 2, 32, 3 * csa2._ATTENTION_QUERY_TILE + 7
    backend = Csa2AttentionBackend(
        compress_ratios=[2],
        window_size=window,
        n_layers=1,
        num_heads=heads,
        attn_head_dim=width,
        index_topk=topk,
        index_n_heads=1,
        index_head_dim=8,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
    )
    torch.manual_seed(817)
    cache = torch.randn(2, 128, width)
    swa = torch.full((2, window, width), float("nan"))
    backend.bind_kv_caches([LayerCache(key=cache, value=None, swa=swa)])
    q = torch.randn(length, heads, width)
    kv = torch.randn(length, width)
    positions = torch.arange(length)
    tables = [torch.tensor([[1]], dtype=torch.int32)] * 2
    dsa = backend._builder.build(tables, [length], [length], positions, True, True)
    layer = SimpleNamespace(
        layer_id=0,
        mode=CSA2LayerMode(0, CSA2_MODE_FULL, 2, 0),
        attn_sink=torch.randn(heads),
        attn_sink_loaded=True,
    )
    shared = CSA2SharedRuntime(key_cache=cache, key_block_table=tables[1], compress_ratio=2)
    columns = torch.arange(min(topk, length // 2))
    picks = (
        torch.where(columns[None, :] < (positions[:, None] + 1) // 2, window + length + columns[None, :], -1)
        .unsqueeze(0)
        .to(torch.int32)
    )
    observed: list[int] = []
    original = csa2._picked_row_mask

    def observe(idxs: torch.Tensor, num_rows: int) -> torch.Tensor:
        observed.append(int(idxs.shape[0]))
        return original(idxs, num_rows)

    with torch.inference_mode():
        monkeypatch.setattr(csa2, "_picked_row_mask", observe)
        actual = backend._csa2_attention(q, kv, layer, dsa, [length], dsa.seq_lens, picks, shared)
        tiled_observed = list(observed)
        saved = csa2._ATTENTION_QUERY_TILE
        monkeypatch.setattr(csa2, "_ATTENTION_QUERY_TILE", length)
        expected = backend._csa2_attention(q, kv, layer, dsa, [length], dsa.seq_lens, picks, shared)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert sum(tiled_observed) == length
    assert len(tiled_observed) > 1, "attention scores were still materialized for the whole chunk"
    assert max(tiled_observed) <= max(saved, 1)


def _assert_compressor_index_rows(cache: LayerCache, block: int, first: int, rows: torch.Tensor) -> None:
    assert cache.index.dtype == torch.int8
    assert cache.indexer_scale.dtype == torch.float16
    codes, scales = _reference_indexer_quant(rows)
    end = first + rows.size(0)
    torch.testing.assert_close(cache.index[block, first:end], codes, atol=0, rtol=0)
    torch.testing.assert_close(cache.indexer_scale[block, first:end, 0], scales, atol=0, rtol=0)


class _CompressorHarness:
    def __init__(
        self,
        lm: DeepseekV41ForCausalLM,
        *,
        batch: int = 1,
        layer_id: int = 2,
        ratio: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = lm.cfg if ratio is None else replace(lm.cfg, compress_ratios=[0, 0, ratio, ratio, 1, 1])
        self.backend = _make_csa2_backend(self.cfg)
        caches = _allocate_v41_layer_caches(self.cfg, batch, dtype)
        self.cache = caches[layer_id]
        self.backend.bind_kv_caches(caches)
        self.layer = lm.model.layers[layer_id].self_attn
        if ratio is not None:
            self.layer.mode = replace(self.layer.mode, compress_ratio=ratio)
        self.mapping = self.backend._resolve_cache_mapping(layer_id, self.layer.mode.compress_ratio)
        self.cos_sin = lm.model.compress_rotary.cos_sin_cache
        self.shared = CSA2SharedRuntime()
        self.rmsnorm_calls: list[torch.Tensor] = []
        original = self.layer.compress_rmsnorm

        def counted(value: torch.Tensor) -> torch.Tensor:
            self.rmsnorm_calls.append(value)
            return original(value)

        self.layer.compress_rmsnorm = counted

    def run(
        self,
        ctx: Csa2LayerContext,
        tables: list[torch.Tensor],
        seq_lens: list[int],
        q_lens: list[int],
        is_prefill: bool = False,
    ) -> None:
        dsa = self.backend._builder.build(tables, seq_lens, q_lens, ctx.positions, is_prefill, True)
        self.backend._run_compressor(self.layer, ctx, dsa, self.cache, self.mapping, q_lens, dsa.seq_lens, self.shared)


@pytest.mark.parametrize("prefix,chunk", [(1, 17), (7, 33), (15, 34), (31, 33), (33, 31), (47, 17)])
@pytest.mark.parametrize("reverse_blocks", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ratio2_compressor_reads_parked_prefix_before_wrapping(
    prefix: int, chunk: int, reverse_blocks: bool, dtype: torch.dtype
) -> None:
    lm = _make_forward_lm(dtype, seed=11)
    harness = _CompressorHarness(lm, dtype=dtype)
    cfg, cache, layer, cos_sin = harness.cfg, harness.cache, harness.layer, harness.cos_sin
    torch.manual_seed(53)
    x = torch.randn(chunk + 1, cfg.hidden_size).to(dtype)
    kv, score = layer.compress_kv_score(x)
    assert kv.dtype == score.dtype == torch.float32
    table = torch.arange(2, dtype=torch.int32).view(1, 2)
    if reverse_blocks:
        table = table.flip(-1)
    assert cache.compress_kv_state.shape == (2, _STATE_ROWS, cfg.head_dim)
    assert cache.compress_kv_state.dtype == cache.compress_score_state.dtype == torch.float32
    cache.compress_kv_state.fill_(float("nan"))
    cache.compress_score_state.fill_(float("nan"))
    cache.key.fill_(float("nan"))
    cache.index.fill_(-128)
    cache.indexer_scale.fill_(float("nan"))
    old_slot = _state_slots(table[0], torch.tensor([prefix - 1]), cfg.window_size)[0]
    cache.compress_kv_state.view(-1, cfg.head_dim)[old_slot] = kv[0]
    cache.compress_score_state.view(-1, cfg.head_dim)[old_slot] = score[0]
    assert prefix + chunk <= cos_sin.size(0), "test must fit its configured RoPE horizon"
    positions = torch.arange(prefix, prefix + chunk)
    token_table = torch.tensor([[0]], dtype=torch.int32)
    ctx = Csa2LayerContext(x[1:], torch.randn(chunk, cfg.q_lora_rank).to(dtype), positions, cos_sin)
    harness.run(ctx, [table, token_table, token_table], [prefix + chunk], [chunk])
    committed = (prefix + chunk) // 2 - prefix // 2
    members = kv[: committed * 2].view(committed, 2, cfg.head_dim)
    gates = score[: committed * 2].view(committed, 2, cfg.head_dim).softmax(1)
    latent = layer.compress_rmsnorm((members * gates).sum(1).to(dtype))
    starts = prefix - 1 + torch.arange(committed) * 2
    key = apply_interleaved_rope(latent, starts, cos_sin).to(cache.key.dtype)
    index = apply_interleaved_rope(layer.indexer_key(latent), starts, cos_sin)
    first = prefix // 2
    torch.testing.assert_close(cache.key[0, first : first + committed], key, atol=1e-6, rtol=1e-6)
    _assert_compressor_index_rows(cache, 0, first, index)
    for untouched in (slice(0, first), slice(first + committed, None)):
        assert torch.isnan(cache.key[0, untouched]).all()
        assert (cache.index[0, untouched] == -128).all()
        assert torch.isnan(cache.indexer_scale[0, untouched]).all()
    assert harness.shared.index_cache is cache.index
    assert harness.shared.index_scale is cache.indexer_scale
    pending = torch.arange((prefix + chunk) // 2 * 2, prefix + chunk)
    if pending.numel():
        slots = _state_slots(table[0], pending, cfg.window_size)
        torch.testing.assert_close(cache.compress_kv_state.view(-1, cfg.head_dim)[slots], kv[-1:])
        torch.testing.assert_close(cache.compress_score_state.view(-1, cfg.head_dim)[slots], score[-1:])


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("chunk", [1, 3, 17, 33])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compressor_preserves_every_unfinished_tail_member(ratio: int, chunk: int, dtype: torch.dtype) -> None:
    lm = _make_forward_lm(dtype, seed=37)
    harness = _CompressorHarness(lm, ratio=ratio, dtype=dtype)
    cfg, cache, layer = harness.cfg, harness.cache, harness.layer
    torch.manual_seed(79)
    values = torch.randn(48, cfg.hidden_size).to(dtype)
    kv, scores = layer.compress_kv_score(values)
    assert kv.dtype == scores.dtype == torch.float32
    count = values.size(0) // ratio
    grouped_kv = kv[: count * ratio].view(count, ratio, cfg.head_dim)
    grouped_scores = scores[: count * ratio].view(count, ratio, cfg.head_dim)
    latent = layer.compress_rmsnorm((grouped_kv * grouped_scores.softmax(1)).sum(1).to(dtype))
    starts = torch.arange(count) * ratio
    expected = apply_interleaved_rope(latent, starts, harness.cos_sin).to(cache.key.dtype)
    expected_index = apply_interleaved_rope(layer.indexer_key(latent), starts, harness.cos_sin)
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    token_table = torch.tensor([[0]], dtype=torch.int32)
    end = 0
    while end < values.size(0):
        start = end
        end = min(end + chunk, values.size(0))
        positions = torch.arange(start, end)
        ctx = Csa2LayerContext(values[start:end], values[start:end, : cfg.q_lora_rank], positions, harness.cos_sin)
        harness.run(ctx, [table, token_table, token_table], [end], [end - start], is_prefill=start == 0)
        pending = torch.arange(end // ratio * ratio, end)
        if pending.numel():
            slots = _state_slots(table[0], pending, cfg.window_size)
            torch.testing.assert_close(cache.compress_kv_state.view(-1, cfg.head_dim)[slots], kv[pending])
            torch.testing.assert_close(cache.compress_score_state.view(-1, cfg.head_dim)[slots], scores[pending])
        committed = end // ratio
        torch.testing.assert_close(cache.key[0, :committed], expected[:committed], atol=2e-6, rtol=2e-5)
        _assert_compressor_index_rows(cache, 0, 0, expected_index[:committed])
    torch.testing.assert_close(cache.key[0, :count], expected, atol=2e-6, rtol=2e-5)
    _assert_compressor_index_rows(cache, 0, 0, expected_index)
    assert harness.rmsnorm_calls, "compressor must run its single post-pool RMSNorm"


@pytest.mark.parametrize("layer_id", [2, 4])
@pytest.mark.parametrize("empty_first", [False, True])
def test_compressor_skips_zero_query_sequences(layer_id: int, empty_first: bool) -> None:
    lm = _make_forward_lm(torch.float32, seed=37)
    harness = _CompressorHarness(lm, batch=2, layer_id=layer_id)
    reference = _CompressorHarness(lm, layer_id=layer_id)
    cfg, cache = harness.cfg, harness.cache
    active_seq = 1 if empty_first else 0
    q_lens = [0, 3] if empty_first else [3, 0]
    positions = torch.arange(3)
    values = torch.randn(3, cfg.hidden_size, generator=torch.Generator().manual_seed(119))
    swa_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    token_table = torch.tensor([[0], [1]], dtype=torch.int32)
    ctx = Csa2LayerContext(values, values[:, : cfg.q_lora_rank], positions, harness.cos_sin)
    with torch.no_grad():
        harness.run(ctx, [swa_table, token_table, token_table], q_lens, q_lens, is_prefill=True)
        reference.run(ctx, [swa_table[:1], token_table[:1], token_table[:1]], [3], [3], is_prefill=True)
    for name in ("key", "index", "indexer_scale", "compress_kv_state", "compress_score_state"):
        actual = getattr(cache, name)
        expected = getattr(reference.cache, name)
        if actual is None:
            assert expected is None
            continue
        blocks = expected.size(0)
        torch.testing.assert_close(actual[active_seq * blocks : (active_seq + 1) * blocks], expected, atol=0, rtol=0)
        inactive = 1 - active_seq
        assert torch.count_nonzero(actual[inactive * blocks : (inactive + 1) * blocks]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compressor_keeps_fp32_projection_and_single_rmsnorm(dtype: torch.dtype) -> None:
    lm = _make_forward_lm(dtype, seed=41)
    harness = _CompressorHarness(lm, dtype=dtype)
    cfg, layer, cache = harness.cfg, harness.layer, harness.cache
    torch.manual_seed(211)
    x = torch.randn(17, cfg.hidden_size).to(dtype)
    kv, score = layer.compress_kv_score(x)
    assert kv.dtype == torch.float32 and score.dtype == torch.float32
    assert cache.compress_kv_state.dtype == cache.compress_score_state.dtype == torch.float32
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    token_table = torch.tensor([[0]], dtype=torch.int32)
    ctx = Csa2LayerContext(x, torch.randn(17, cfg.q_lora_rank).to(dtype), torch.arange(17), harness.cos_sin)
    harness.run(ctx, [table, token_table, token_table], [17], [17], is_prefill=True)
    assert len(harness.rmsnorm_calls) == 1, "RMSNorm must run once per chunk, not per path"
    assert harness.rmsnorm_calls[0].size(0) == 9
    slots = _state_slots(table[0], torch.tensor([16]), cfg.window_size)
    torch.testing.assert_close(cache.compress_kv_state.view(-1, cfg.head_dim)[slots], kv[-1:])


@pytest.mark.parametrize("fail", [False, True])
def test_attention_releases_request_sized_context_after_handoff(fail: bool) -> None:
    lm = _make_forward_lm(torch.float32, seed=19)
    attention = lm.model.layers[0].self_attn

    def execute(q: torch.Tensor, kv: torch.Tensor, layer: object) -> torch.Tensor:
        assert layer._csa2_ctx is not None
        if fail:
            raise RuntimeError("attention failed")
        return torch.zeros_like(q)

    metadata = SimpleNamespace()
    backend = SimpleNamespace(execute_csa2_layer=execute)
    with torch.inference_mode(), forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        args = (torch.ones(9, lm.cfg.hidden_size), torch.arange(9), lm.model._layer_rope_cache(attention.mode))
        if fail:
            with pytest.raises(RuntimeError, match="attention failed"):
                attention(*args)
        else:
            attention(*args)
    assert attention._csa2_ctx is None


def test_empty_dp_metadata_has_no_fabricated_csa2_history() -> None:
    backend = _make_swa_backend(8, 4, 4)
    cache = torch.full((2, 8, 4), float("nan"))
    backend.bind_kv_caches([LayerCache(None, None, swa=cache)])
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        dsa_positions=torch.tensor([0]),
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_graph_mode=False,
    )
    dsa = backend._build_dsa_metadata_for_forward(metadata)
    assert dsa.seq_lens.tolist() == [0]
    assert dsa.actual_seq_lengths_kv.tolist() == [0]
    assert dsa.seq_lens_q.tolist() == [1]
    assert dsa.kv_cu_seq_lens.tolist() == [0, 0]
    assert dsa.start_pos.tolist() == [-1]
    assert all((slots == -1).all() for layer in dsa.slot_mappings for slots in layer)
    metadata.dsa_metadata = dsa
    backend.prepare(metadata)
    layer = _swa_layer(torch.zeros(4))
    layer._csa2_ctx = Csa2LayerContext(torch.ones(1, 4), torch.ones(1, 4), torch.tensor([0]), torch.empty(0))
    with torch.inference_mode(), forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        output = backend.execute_csa2_layer(torch.ones(1, 4, 4), torch.ones(1, 4), layer)
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isnan(cache).all()


@pytest.mark.parametrize("graph_metadata", [False, True])
def test_dummy_model_forward_keeps_positions_safe_and_caches_inert(
    graph_metadata: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _active_device()
    lm = _make_forward_lm(torch.float32, seed=101, device=device)
    backend = _make_csa2_backend(lm.cfg, device=device, max_model_len=64)
    caches = _allocate_v41_layer_caches(
        lm.cfg, 1, torch.float32, device=device, reserve_token_sink=True, reserve_swa_sink=True
    )
    _poison_v41_caches(caches)
    backend.bind_kv_caches(caches)
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_graph_mode=graph_metadata,
        dsa_graph_token_capacity=64 if graph_metadata else 0,
        dsa_graph_block_table_cols=4 if graph_metadata else 0,
        is_spec_verify=False,
    )
    used_topk: list[torch.Tensor] = []
    attention = backend._csa2_attention

    def record_topk(
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: DeepseekV41Attention,
        dsa: DsaMetadata,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        topk: torch.Tensor | None,
        shared: CSA2SharedRuntime,
        cp_context: DsaCpContext | None = None,
    ) -> torch.Tensor:
        if topk is not None:
            used_topk.append(topk.detach().clone())
        return attention(q, kv, layer, dsa, q_lens, ctx_lens, topk, shared, cp_context)

    monkeypatch.setattr(backend, "_csa2_attention", record_topk)
    rope_calls: list[torch.Tensor] = []
    original_rope = csa2.apply_interleaved_rope

    def checked_rope(
        value: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, *, inverse: bool = False
    ) -> torch.Tensor:
        assert (positions >= 0).all()
        assert (positions < cos_sin_cache.size(0)).all()
        rope_calls.append(positions.detach().cpu())
        return original_rope(value, positions, cos_sin_cache, inverse=inverse)

    monkeypatch.setattr(csa2, "apply_interleaved_rope", checked_rope)
    with torch.inference_mode(), forward_context(ForwardContext(backend, device, metadata, caches)):
        output = lm.model(torch.tensor([1], device=device), torch.tensor([0], device=device))
    assert rope_calls
    assert torch.isfinite(output).all()
    dsa = metadata.dsa_metadata
    torch.testing.assert_close(dsa.start_pos, dsa.seq_lens - dsa.seq_lens_q, atol=0, rtol=0)
    assert dsa.start_pos.tolist() == [-1]
    assert all((slots == -1).all() for layer_slots in dsa.slot_mappings for slots in layer_slots)
    assert used_topk
    assert all((topk == -1).all() for topk in used_topk)
    for cache in caches:
        for name in ("swa", "key", "index", "indexer_scale", "compress_kv_state", "compress_score_state"):
            tensor = getattr(cache, name)
            if tensor is not None:
                assert (tensor == -128).all() if tensor.dtype == torch.int8 else torch.isnan(tensor).all()


@npu_only
@pytest.mark.parametrize("graph_metadata", [False, True])
def test_dummy_model_triton_compressor_keeps_ring_states_inert(
    graph_metadata: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _active_device()
    lm = _make_forward_lm(torch.float32, seed=101, device=device)
    backend = _make_csa2_backend(lm.cfg, device=device, max_model_len=64)
    caches = _allocate_v41_layer_caches(
        lm.cfg, 1, torch.float32, device=device, reserve_token_sink=True, reserve_swa_sink=True
    )
    before = []
    for cache in caches:
        if cache.compress_kv_state is not None:
            cache.compress_kv_state.copy_(torch.randn_like(cache.compress_kv_state))
            cache.compress_score_state.copy_(torch.randn_like(cache.compress_score_state))
            before.append((cache.compress_kv_state.clone(), cache.compress_score_state.clone()))
    backend.bind_kv_caches(caches)
    assert backend._swa_ring_blocks == 1
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_graph_mode=graph_metadata,
        dsa_graph_token_capacity=64 if graph_metadata else 0,
        dsa_graph_block_table_cols=4 if graph_metadata else 0,
        is_spec_verify=False,
    )
    module = csa2._get_v41_compressor_module()
    assert module is not None, "NPU regression requires the production Triton compressor"
    original = module.compressor_from_projected
    launches: list[tuple[torch.Tensor, torch.Tensor]] = []

    def observe_launch(*args: torch.Tensor, **kwargs: int) -> torch.Tensor:
        result = original(*args, **kwargs)
        launches.append((args[4].detach().clone(), result.detach().clone()))
        return result

    monkeypatch.setattr(module, "compressor_from_projected", observe_launch)
    with torch.inference_mode(), forward_context(ForwardContext(backend, device, metadata, caches)):
        output = lm.model(torch.tensor([1], device=device), torch.tensor([0], device=device))
    torch.npu.synchronize()
    dsa = metadata.dsa_metadata
    assert dsa.start_pos.tolist() == [-1]
    assert dsa.seq_lens.tolist() == [0]
    assert dsa.seq_lens_q.tolist() == [1]
    expected_launches = sum(layer.mode.is_full and layer.mode.compress_ratio == 2 for layer in lm.model.layers)
    assert expected_launches > 0
    assert len(launches) == expected_launches
    for controls, pooled in launches:
        assert controls[:, 0].tolist() == [-1, 1, 0, 0, 0]
        torch.testing.assert_close(pooled, torch.zeros_like(pooled), atol=0, rtol=0)
    assert torch.isfinite(output).all()
    for cache, (kv, scores) in zip((c for c in caches if c.compress_kv_state is not None), before, strict=True):
        torch.testing.assert_close(cache.compress_kv_state, kv, atol=0, rtol=0)
        torch.testing.assert_close(cache.compress_score_state, scores, atol=0, rtol=0)


def test_window_gather_is_chronological_with_unordered_physical_blocks() -> None:
    window, query_len, previous = 8, 17, 65
    backend = _make_swa_backend(window, 4, 4)
    table = torch.tensor([[3, 1, 4]], dtype=torch.int32)
    cache = torch.full((5, window, 1), float("nan"))
    history = torch.arange(previous - table.size(1) * window, previous)
    cache.view(-1, 1)[_ring_slots(table[0], history, window)] = history.float().unsqueeze(-1)
    ring_positions = torch.arange(previous - window, previous)
    ring = backend._gather_swa_positions(cache.view(-1, 1), table, 0, ring_positions, window)
    torch.testing.assert_close(ring.flatten(), ring_positions.float(), atol=0, rtol=0)
    positions = torch.arange(previous, previous + query_len)
    window_kv = torch.cat((ring, positions.float().unsqueeze(-1)))
    picks = _window_visible_idxs(positions, torch.tensor(previous), window)
    picks = picks[picks >= 0].reshape(query_len, window)
    expected_positions = positions.unsqueeze(-1) - window + 1 + torch.arange(window)
    torch.testing.assert_close(window_kv[picks.long()].squeeze(-1), expected_positions.float(), atol=0, rtol=0)


@pytest.mark.parametrize("slots", [[], [-1, -1], [3], [4, 1, 4, -1, 1, 4, 2], [0] * 257, list(range(128)) * 17])
def test_last_cache_writes_are_unique_latest_and_ignore_padding(slots: list[int]) -> None:
    values = torch.arange(len(slots) * 3, dtype=torch.float32).view(-1, 3)
    physical = torch.tensor(slots, dtype=torch.int32)
    latest = {slot: i for i, slot in enumerate(slots) if slot >= 0}
    actual_slots, actual_values = last_ring_writes(physical, values)
    assert actual_slots.tolist() == sorted(latest)
    expected_rows = torch.tensor([latest[slot] for slot in sorted(latest)], dtype=torch.long)
    torch.testing.assert_close(actual_values, values[expected_rows], atol=0, rtol=0)


def test_last_cache_writes_rejects_misaligned_values() -> None:
    with pytest.raises(ValueError, match="row counts"):
        last_ring_writes(torch.tensor([1, 2]), torch.ones(1, 4))


@pytest.mark.parametrize("exponents", [range(-24, -14), range(-14, 1), range(1, 15)])
def test_indexer_fp16_scale_binades_preserve_nextafter_boundaries(exponents: range) -> None:
    scales = torch.exp2(torch.tensor(list(exponents), dtype=torch.float64)).half()
    next_scales = torch.nextafter(scales, torch.full_like(scales, float("inf")))
    midpoints = (scales.float() + next_scales.float()) * 0.5
    maxima = midpoints * 127
    boundaries = torch.stack(
        [
            torch.nextafter(maxima, torch.zeros_like(maxima)),
            maxima,
            torch.nextafter(maxima, torch.full_like(maxima, float("inf"))),
        ],
        dim=-1,
    ).flatten()
    values = torch.stack([boundaries, -boundaries, boundaries / 3, torch.zeros_like(boundaries)], dim=-1)
    codes, actual_scales = _quantize_indexer_rows_int8(values)
    expected_codes, expected_scales = _reference_indexer_quant(values)
    torch.testing.assert_close(actual_scales, expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(codes, expected_codes, atol=0, rtol=0)
    assert (actual_scales > 0).all() and torch.isfinite(actual_scales).all()
    partitions = [_quantize_indexer_rows_int8(part) for part in values.tensor_split(3)]
    torch.testing.assert_close(torch.cat([part[0] for part in partitions]), codes, atol=0, rtol=0)
    torch.testing.assert_close(torch.cat([part[1] for part in partitions]), actual_scales, atol=0, rtol=0)


def test_indexer_int8_rounding_all_grid_midpoints() -> None:
    grid = torch.arange(-127, 128, dtype=torch.float32)
    midpoints = (grid[:-1] + grid[1:]) * 0.5
    values = torch.cat(
        [
            grid,
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, float("-inf"))),
            torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
        ]
    )
    rows = torch.stack([torch.full_like(values, 127), values], dim=-1)
    codes, scales = _quantize_indexer_rows_int8(rows)
    torch.testing.assert_close(scales, torch.ones_like(scales), atol=0, rtol=0)
    torch.testing.assert_close(codes[:, 0], torch.full_like(codes[:, 0], 127), atol=0, rtol=0)
    torch.testing.assert_close(codes[:, 1], values.double().round().to(torch.int8), atol=0, rtol=0)
    assert (codes != -128).all()
    edge_rows = torch.tensor([[0, 0], [2**-25, -(2**-25)], [2**-24, -(2**-24)], [127.04, -127.04]])
    edge_codes, edge_scales = _quantize_indexer_rows_int8(edge_rows)
    torch.testing.assert_close(
        edge_codes, torch.tensor([[0, 0], [0, 0], [1, -1], [127, -127]], dtype=torch.int8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        edge_scales, torch.tensor([2**-24, 2**-24, 2**-24, 1], dtype=torch.float16), atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hc_mult", [2, 4, 8])
def test_hc_pre_stream_reduction_is_partition_invariant(dtype: torch.dtype, hc_mult: int) -> None:
    device = _active_device()
    torch.manual_seed(101)
    hidden = torch.randn(129, hc_mult, 128, dtype=dtype, device=device)
    pre_mix = torch.rand(129, hc_mult, dtype=torch.float32, device=device)
    if hc_mult == 4:
        hidden[60, :, 108] = torch.tensor(
            [-0.193359375, -0.193359375, -0.1923828125, -0.1953125], dtype=dtype, device=device
        )
        pre_mix[60] = torch.tensor(
            [0.5022825598716736, 0.49681442975997925, 0.5081950426101685, 0.49532201886177063],
            dtype=torch.float32,
            device=device,
        )
    hc = object.__new__(DeepseekV41HyperConnection)
    expected = (pre_mix.cpu().double().unsqueeze(-1) * hidden.cpu().double()).sum(-2).to(device=device, dtype=dtype)
    actual = hc.hc_pre(hidden, pre_mix)
    partitioned = torch.cat([hc.hc_pre(hidden[s : s + 5], pre_mix[s : s + 5]) for s in range(0, 129, 5)])
    replay = torch.cat([hc.hc_pre(hidden[i : i + 1], pre_mix[i : i + 1]) for i in range(129)])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.008 if dtype == torch.bfloat16 else 1e-6)
    torch.testing.assert_close(partitioned, actual, atol=0, rtol=0)
    torch.testing.assert_close(replay, actual, atol=0, rtol=0)


@pytest.mark.parametrize("required_mode", ["selected", "fused"])
@pytest.mark.parametrize("chunk_size", [1, 8, 31])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_hc_mode_is_partition_invariant(
    required_mode: str,
    chunk_size: int,
    dtype: torch.dtype,
    record_property: Callable[[str, object], None],
) -> None:
    device = _active_device()
    cfg = DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)
    hc = DeepseekV41HyperConnection(cfg, device, dtype=dtype)
    record_property("hc_pre_mode", hc._hc_pre_mode)
    record_property("device", str(device))
    if required_mode == "fused" and hc._hc_pre_mode != "fused":
        pytest.skip(f"native HC not selected: mode={hc._hc_pre_mode}, device={device}, dtype={dtype}")
    torch.manual_seed(101)
    with torch.no_grad():
        for name, parameter in hc.named_parameters():
            parameter.normal_(std=0.004 if name.endswith("_fn") else 0.5)
    hidden = torch.randn(129, cfg.hc_mult, cfg.hidden_size, device=device).to(dtype)
    pre_mix = torch.rand(129, cfg.hc_mult, device=device) + 0.1
    hidden[60, :, 108] = hidden.new_tensor([-0.193359375, -0.193359375, -0.1923828125, -0.1953125])
    pre_mix[60] = pre_mix.new_tensor([0.5022825598716736, 0.49681442975997925, 0.5081950426101685, 0.49532201886177063])

    def run(x: torch.Tensor, previous_pre: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if hc._hc_pre_mode == "fused":
            return hc.hc_pre_fused(x, hc.hc_attn_fn, hc.hc_attn_scale, hc.hc_attn_base, previous_pre)
        pre, post, comb = hc.hc_mixes(x, hc.hc_attn_fn, hc.hc_attn_scale, hc.hc_attn_base)
        return hc.hc_pre(x, previous_pre), pre, post, comb

    with torch.inference_mode():
        full = run(hidden, pre_mix)
        pieces = [
            run(hidden[s : s + chunk_size], pre_mix[s : s + chunk_size]) for s in range(0, hidden.size(0), chunk_size)
        ]
        for index, expected in enumerate(full):
            actual = torch.cat([piece[index] for piece in pieces])
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(
                actual,
                expected,
                atol=0 if dtype == torch.bfloat16 and index == 0 else 2e-5,
                rtol=0 if dtype == torch.bfloat16 and index == 0 else 2e-5,
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [0, 1, 127, 129])
def test_hc_mixes_does_not_pad_small_chunks(dtype: torch.dtype, rows: int) -> None:
    device = _active_device()
    torch.manual_seed(917)
    cfg = DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)
    hc = object.__new__(DeepseekV41HyperConnection)
    hc.hc_mult = cfg.hc_mult
    hc.norm_eps = cfg.rms_norm_eps
    hc.hc_eps = cfg.hc_eps
    hc.sinkhorn_iters = cfg.hc_sinkhorn_iters
    hidden = torch.randn(rows, cfg.hc_mult, 32, dtype=dtype, device=device)
    weight = torch.randn(24, cfg.hc_mult * 32, device=device)
    seen: list[int] = []
    linear = torch.nn.functional.linear

    def observe(value: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        seen.append(int(value.shape[0]))
        return linear(value, projection)

    with torch.inference_mode(), pytest.MonkeyPatch.context() as context:
        context.setattr(torch.nn.functional, "linear", observe)
        pre, post, comb = hc.hc_mixes(hidden, weight, torch.ones(3, device=device), torch.zeros(24, device=device))
    assert seen == [rows]
    assert pre.shape[0] == post.shape[0] == comb.shape[0] == rows


@npu_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prompt_len,chunk_size", [(33, 8), (65, 16), (65, 17), (129, 32), (257, 128)])
def test_chunked_model_logits_and_following_decode_match_token_replay(
    prompt_len: int, chunk_size: int, dtype: torch.dtype
) -> None:
    device = _active_device()
    config = {**_DSV41_FORWARD_CONFIG, "max_position_embeddings": 512, "dtype": dtype, "device": str(device)}
    torch.manual_seed(101)
    lm = DeepseekV41ForCausalLM(config)
    _random_init_v41(lm)
    with torch.no_grad():
        for decoder in lm.model.layers:
            indexer = decoder.self_attn.indexer
            if dtype == torch.bfloat16 and indexer is not None:
                assert indexer.weights_proj.bias is None
                indexer.weights_proj.weight.zero_()
    reference = _V41ForwardHarness(lm, config, 2, dtype, device=device, max_tokens=prompt_len + 4)
    chunked = _V41ForwardHarness(lm, config, 2, dtype, device=device, max_tokens=prompt_len + 4)
    full = _V41ForwardHarness(lm, config, 2, dtype, device=device, max_tokens=prompt_len + 4)
    for harness in (reference, chunked, full):
        _poison_v41_caches(harness.layer_caches)
    ids = (torch.arange(2 * prompt_len) % lm.cfg.vocab_size).reshape(2, prompt_len).to(device)
    with torch.inference_mode():

        def record_selections(harness: _V41ForwardHarness, records: dict[int, list[torch.Tensor]]) -> None:
            for layer_id, topk in harness.used_topk.items():
                if topk is not None:
                    selected = topk.new_full((*topk.shape[:-1], harness.cfg.index_topk), -1)
                    if topk.shape[-1]:
                        rows = topk - harness.cfg.window_size - topk.shape[1]
                        selected[..., : topk.shape[-1]].copy_(torch.where(topk >= 0, rows, -1))
                    records.setdefault(layer_id, []).append(selected.sort(dim=-1).values)

        reference_selections: dict[int, list[torch.Tensor]] = {}
        chunked_selections: dict[int, list[torch.Tensor]] = {}
        expected_rows = []
        for p in range(prompt_len):
            expected_rows.append(
                reference.step(ids[:, p], torch.full((2,), p, device=device), [p + 1] * 2, [1] * 2, is_prefill=False)
            )
            record_selections(reference, reference_selections)
        expected = torch.stack(expected_rows, dim=1)
        full_hidden = full.step(
            ids.reshape(-1),
            torch.arange(prompt_len, device=device).repeat(2),
            [prompt_len] * 2,
            [prompt_len] * 2,
            is_prefill=True,
        ).view(2, prompt_len, -1)
        full_selections: dict[int, list[torch.Tensor]] = {}
        record_selections(full, full_selections)
        chunks = []
        for start in range(0, prompt_len, chunk_size):
            end = min(start + chunk_size, prompt_len)
            hidden = chunked.step(
                ids[:, start:end].reshape(-1),
                torch.arange(start, end, device=device).repeat(2),
                [end] * 2,
                [end - start] * 2,
                is_prefill=start == 0,
                is_chunked_prefill=True,
            )
            chunks.append(hidden.view(2, end - start, -1))
            record_selections(chunked, chunked_selections)
        actual = torch.cat(chunks, dim=1)
        assert full_selections.keys() == {i for i, ratio in enumerate(lm.cfg.compress_ratios) if ratio > 0}
        assert full_selections.keys() == reference_selections.keys() == chunked_selections.keys()
        for layer_id, selections in full_selections.items():
            expected_selection = torch.cat(reference_selections[layer_id], dim=1)
            valid = expected_selection >= 0
            assert valid.any(), f"layer {layer_id}: no compressed rows selected"
            if layer_id == min(full_selections):
                expected_count = min(lm.cfg.index_topk, prompt_len // lm.cfg.compress_ratios[layer_id])
                assert torch.all(valid[:, -1].sum(dim=-1) == expected_count)
            torch.testing.assert_close(
                torch.cat(selections, dim=1),
                expected_selection,
                atol=0,
                rtol=0,
                msg=f"layer {layer_id}: full-prefill selected rows differ from token replay",
            )
            torch.testing.assert_close(
                torch.cat(chunked_selections[layer_id], dim=1),
                expected_selection,
                atol=0,
                rtol=0,
                msg=f"layer {layer_id}: chunked-prefill selected rows differ from token replay",
            )
        reference_logits = reference.logits(expected.reshape(-1, lm.cfg.hidden_size))
        full_logits = full.logits(full_hidden.reshape(-1, lm.cfg.hidden_size))
        chunked_logits = chunked.logits(actual.reshape(-1, lm.cfg.hidden_size))
        _assert_model_logits_close(full_logits, reference_logits, dtype)
        _assert_model_logits_close(chunked_logits, full_logits, dtype)
        _assert_model_logits_close(chunked_logits, reference_logits, dtype)
        for p in range(prompt_len, prompt_len + 4):
            args = (torch.tensor([7, 11], device=device), torch.full((2,), p, device=device), [p + 1] * 2, [1] * 2)
            reference_logits = reference.logits(reference.step(*args, is_prefill=False))
            full_logits = full.logits(full.step(*args, is_prefill=False))
            chunked_logits = chunked.logits(chunked.step(*args, is_prefill=False))
            _assert_model_logits_close(full_logits, reference_logits, dtype)
            _assert_model_logits_close(chunked_logits, full_logits, dtype)
            _assert_model_logits_close(chunked_logits, reference_logits, dtype)
