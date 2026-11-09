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
from typing import Any

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
from tests.python._deepseek_v41_fused_testkit import _reference_indexer_quant
from xllm.python.attention.csa2_attention import (
    Csa2AttentionBackend,
    Csa2LayerContext,
    CSA2SharedRuntime,
    _QliGraphState,
    apply_interleaved_rope,
)

_N_LAYERS = 3
_COMPRESS_RATIOS = [0, 2, 1]
_KV_SOURCE_LAYERS = [1]
_INDEX_SOURCE_LAYERS = [1, 2]
_WINDOW = 128
_HEAD_DIM = 512
_NUM_HEADS = 8
_INDEX_TOPK = 4
_ROPE_HEAD_DIM = 8
_INDEX_HEAD_DIM = 128
_INDEX_N_HEADS = 2
_BLOCK_SIZE = 128
_CAPACITY = 256
_ROWS = 4
_CTX_LENS = [300, 500, 200, 129]
_SHADOW_BLOCKS = 8


class _QliRecorder:
    def __init__(self) -> None:
        self.indexer_calls: list[dict[str, Any]] = []
        self.metadata_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def indexer(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        topk: int,
        quant_mode: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.indexer_calls.append(
            {
                "query": query,
                "key": key,
                "weights": weights,
                "query_scale": query_scale,
                "key_scale": key_scale,
                "topk": topk,
                "quant_mode": quant_mode,
                **kwargs,
            }
        )
        rows = query.size(0)
        picks = torch.full((rows, 1, topk), -1, dtype=torch.int32)
        if rows:
            picks[0, 0, 0] = 0
            if topk > 1:
                picks[0, 0, 1] = 2
        if int(kwargs.get("candidate_mode", csa2._QLI_V3_CAND_MODE_OFF)) == csa2._QLI_V3_CAND_MODE_SOURCE:
            blocks = int(kwargs.get("candidate_topk_blocks", csa2._QLI_V3_CAND_TOPK_BLOCKS_MAX))
            candidate_out = torch.full((rows, 1, blocks), -1, dtype=torch.int32)
            if rows:
                candidate_out[0, 0, 0] = 0
        else:
            candidate_out = torch.empty(0, dtype=torch.int32)
        return picks, torch.empty(0, dtype=torch.bfloat16), candidate_out

    def metadata(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.metadata_calls.append((args, kwargs))
        return torch.zeros(csa2._QLI_V3_METADATA_SIZE, dtype=torch.int32)


def _install_ops(monkeypatch: pytest.MonkeyPatch, recorder: _QliRecorder) -> None:
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: (recorder.indexer, recorder.metadata))


def _stub_qli(monkeypatch: pytest.MonkeyPatch) -> _QliRecorder:
    recorder = _QliRecorder()
    _install_ops(monkeypatch, recorder)
    return recorder


def _make_backend(
    *,
    index_head_dim: int = _INDEX_HEAD_DIM,
    candidate_source_layer_id: int = -1,
    n_layers: int = _N_LAYERS,
    compress_ratios: list[int] | None = None,
    kv_source_layer_ids: list[int] | None = None,
    index_source_layer_ids: list[int] | None = None,
    candidate_topk_blocks: int = 2048,
    candidate_block_size: int = 8,
) -> Csa2AttentionBackend:
    return Csa2AttentionBackend(
        compress_ratios=list(_COMPRESS_RATIOS if compress_ratios is None else compress_ratios),
        window_size=_WINDOW,
        n_layers=n_layers,
        num_heads=_NUM_HEADS,
        attn_head_dim=_HEAD_DIM,
        index_topk=_INDEX_TOPK,
        index_n_heads=_INDEX_N_HEADS,
        index_head_dim=index_head_dim,
        rope_head_dim=_ROPE_HEAD_DIM,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_source_layer_ids=list(_KV_SOURCE_LAYERS if kv_source_layer_ids is None else kv_source_layer_ids),
        index_source_layer_ids=list(_INDEX_SOURCE_LAYERS if index_source_layer_ids is None else index_source_layer_ids),
        candidate_source_layer_id=candidate_source_layer_id,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


class _StubIndexerLayer:
    def __init__(self, layer_id: int, ratio: int) -> None:
        self.layer_id = layer_id
        self.mode = SimpleNamespace(compress_ratio=ratio)

    def indexer_query(self, qr: torch.Tensor) -> torch.Tensor:
        return torch.zeros(qr.size(0), _INDEX_N_HEADS, _INDEX_HEAD_DIM, dtype=torch.bfloat16)

    def indexer_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.size(0), _INDEX_N_HEADS), 0.5, dtype=torch.bfloat16)


class _StubWriterLayer:
    def __init__(self, layer_id: int, ratio: int = 1) -> None:
        self.layer_id = layer_id
        self.mode = SimpleNamespace(compress_ratio=ratio)
        self.indexer = SimpleNamespace(owns_k=True)

    def compress_kv_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        rows = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(-1)
        cols = torch.arange(1, _INDEX_HEAD_DIM + 1, dtype=torch.float32).unsqueeze(0)
        kv = (rows * cols).to(torch.bfloat16)
        return kv, kv

    def compress_rmsnorm(self, v: torch.Tensor) -> torch.Tensor:
        return v

    def indexer_key(self, latent: torch.Tensor) -> torch.Tensor:
        return latent[:, :_INDEX_HEAD_DIM]


class _StubLayerCache:
    def __init__(
        self,
        *,
        blocks: int = _SHADOW_BLOCKS,
        block_size: int = _BLOCK_SIZE,
        head_dim: int = _HEAD_DIM,
        index_head_dim: int = _INDEX_HEAD_DIM,
    ) -> None:
        self.key = torch.zeros(blocks, block_size, 1, head_dim, dtype=torch.bfloat16)
        self.index = torch.zeros(blocks, block_size, 1, index_head_dim, dtype=torch.int8)
        self.indexer_scale = torch.zeros(blocks, block_size, 1, dtype=torch.float16)
        self.swa = torch.zeros(blocks, block_size, 1, head_dim, dtype=torch.bfloat16)
        self.compress_kv_state = None
        self.compress_score_state = None


def _make_ctx(rows: int = _ROWS) -> Csa2LayerContext:
    half = _ROPE_HEAD_DIM // 2
    cache = torch.zeros(1024, _ROPE_HEAD_DIM, dtype=torch.float32)
    cache[:, :half] = 1.0
    return Csa2LayerContext(
        x=torch.zeros(rows, 128, dtype=torch.bfloat16),
        qr=torch.zeros(rows, _INDEX_N_HEADS * _INDEX_HEAD_DIM, dtype=torch.bfloat16),
        positions=torch.arange(rows, dtype=torch.int64),
        cos_sin_cache=cache,
    )


def _make_shared(ratio: int) -> CSA2SharedRuntime:
    return CSA2SharedRuntime(
        index_cache=torch.zeros(
            _SHADOW_BLOCKS, _BLOCK_SIZE, csa2._QLI_V3_KEY_NUM_HEADS, _INDEX_HEAD_DIM, dtype=torch.int8
        ),
        index_block_table=torch.arange(_ROWS * 2, dtype=torch.int32).view(_ROWS, 2),
        index_scale=torch.zeros(_SHADOW_BLOCKS, _BLOCK_SIZE, csa2._QLI_V3_KEY_NUM_HEADS, dtype=torch.float16),
        compress_ratio=ratio,
    )


def _seed_index_plane(shared: CSA2SharedRuntime, seed: CSA2SharedRuntime) -> None:
    shared.index_cache = seed.index_cache
    shared.index_block_table = seed.index_block_table
    shared.index_scale = seed.index_scale
    shared.compress_ratio = seed.compress_ratio


def _make_dsa(
    *,
    is_acl_graph: bool = True,
    capacity: int = _CAPACITY,
    seq_lens: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        is_acl_graph=is_acl_graph,
        graph_token_capacity=capacity,
        seq_lens=torch.tensor(list(_CTX_LENS if seq_lens is None else seq_lens), dtype=torch.int32),
    )


def _drive(
    backend: Csa2AttentionBackend,
    layer: _StubIndexerLayer,
    dsa: SimpleNamespace,
    shared: CSA2SharedRuntime,
    ctx: Csa2LayerContext | None = None,
) -> torch.Tensor:
    return backend._run_indexer(layer, ctx or _make_ctx(), dsa, [1] * _ROWS, dsa.seq_lens, shared)


def test_qli_v3_op_missing_stays_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    backend = _make_backend()

    topk = _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))

    assert recorder.indexer_calls == []
    assert recorder.metadata_calls == []
    assert backend._qli_state == {}
    assert topk.dtype == torch.int32
    assert topk.shape == (_ROWS, 1, _INDEX_TOPK)


def test_qli_v3_dispatch_on_routes_through_the_op(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()
    shared = _make_shared(1)

    topk = _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)

    assert len(recorder.indexer_calls) == 1
    call = recorder.indexer_calls[0]
    assert call["query"].dtype == torch.int8
    assert call["query"].shape == (_ROWS, _INDEX_N_HEADS, _INDEX_HEAD_DIM)
    assert call["query_scale"].dtype == torch.float16
    assert call["query_scale"].shape == (_ROWS, _INDEX_N_HEADS)
    num_k_heads = csa2._QLI_V3_KEY_NUM_HEADS
    assert call["key"] is shared.index_cache
    assert call["key"].dtype == torch.int8
    assert call["key"].shape == (_SHADOW_BLOCKS, _BLOCK_SIZE, num_k_heads, _INDEX_HEAD_DIM)
    assert call["key_scale"] is shared.index_scale
    assert call["key_scale"].dtype == torch.float16
    assert call["key_scale"].shape == (_SHADOW_BLOCKS, _BLOCK_SIZE, num_k_heads)
    assert call["weights"].dtype == torch.float16
    assert call["weights"].shape == (_ROWS, _INDEX_N_HEADS)
    assert call["topk"] == _INDEX_TOPK
    assert call["quant_mode"] == 2
    assert call["block_table"] is shared.index_block_table
    assert call["block_table"].dtype == torch.int32
    assert call["cu_seqlens_q"].dtype == torch.int32
    assert call["cu_seqlens_q"].tolist() == [0, 1, 2, 3, 4]
    assert call["cu_seqlens_k"] is None
    assert call["seqused_k"].tolist() == _CTX_LENS
    assert call["cmp_residual_k"] is None
    assert call["metadata"].shape == (csa2._QLI_V3_METADATA_SIZE,) and call["metadata"].dtype == torch.int32
    assert call["layout_q"] == "TND"
    assert call["layout_k"] == "PA_BBND"
    assert call["mask_mode"] == 3
    assert call["cmp_ratio"] == 1
    assert call["candidate_mode"] == csa2._QLI_V3_CAND_MODE_OFF
    assert "candidate_topk_index" not in call
    assert "candidate_topk_blocks" not in call
    assert "candidate_block_size" not in call
    assert call["max_seqlen_q"] == 1
    assert topk.shape == (_ROWS, 1, _INDEX_TOPK)
    assert topk[0, 0].tolist() == [_WINDOW + 1, _WINDOW + 3, -1, -1]
    assert bool((topk[1:] == -1).all())


def test_qli_rebase_filters_invisible_and_sorts() -> None:
    picks = torch.tensor([[3, 0, -1, 7, 1]], dtype=torch.int32)
    visible = torch.tensor([[4]], dtype=torch.int32)
    out = csa2._rebase_qli_picks(picks, visible, offset=100)
    assert out.tolist() == [[100, 101, 103, -1, -1]]
    assert out.dtype == torch.int32


def test_qli_rebase_boundary_and_all_invalid() -> None:
    picks = torch.tensor([[2, 1], [-1, -1]], dtype=torch.int32)
    visible = torch.tensor([[2], [3]], dtype=torch.int32)
    out = csa2._rebase_qli_picks(picks, visible, offset=10)
    assert out.tolist() == [[11, -1], [-1, -1]]


def test_qli_rebase_matches_eager_topk_contract() -> None:
    score = torch.tensor([[0.1, 0.9, 0.5, 0.3]], dtype=torch.float32)
    visible = torch.tensor([[3]], dtype=torch.int32)
    offset = 7
    eager = csa2._indexer_topk(score, visible.reshape(-1), index_topk=4, offset=offset)
    picks = torch.tensor([[1, 2, 0, 3]], dtype=torch.int32)
    qli = csa2._rebase_qli_picks(picks, visible, offset=offset)
    assert qli.tolist() == eager.tolist() == [[7, 8, 9, -1]]


def test_qli_rebase_fp32_sort_is_bitwise_equal_to_int32_sort() -> None:
    generator = torch.Generator().manual_seed(11)
    for fraction in (0.0, 0.25, 0.6):
        rows = 64
        picks = torch.randint(-1, 40, (rows, 16), generator=generator, dtype=torch.int32)
        visible = torch.randint(0, 41, (rows, 1), generator=generator, dtype=torch.int32)
        if fraction == 0.0:
            picks = torch.full((rows, 16), -1, dtype=torch.int32)
        int32_out = csa2._rebase_qli_picks(picks, visible, offset=1000, fp32_sort=False)
        fp32_out = csa2._rebase_qli_picks(picks, visible, offset=1000, fp32_sort=True)
        assert torch.equal(fp32_out, int32_out)


def test_qli_rebase_fp32_sort_handles_the_topk_width_boundary() -> None:
    picks = torch.tensor([[5, 2, 9, 0, 3, 1, 8, 7]], dtype=torch.int32)
    visible = torch.tensor([[16]], dtype=torch.int32)
    int32_out = csa2._rebase_qli_picks(picks, visible, offset=3, fp32_sort=False)
    fp32_out = csa2._rebase_qli_picks(picks, visible, offset=3, fp32_sort=True)
    assert int32_out.tolist() == fp32_out.tolist() == [[3, 4, 5, 6, 8, 10, 11, 12]]


def test_qli_v3_filters_out_of_range_picks_and_sorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _QliRecorder()
    real_indexer = recorder.indexer

    def indexer(
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        topk: int,
        quant_mode: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        picks, extra_a, extra_b = real_indexer(query, key, weights, query_scale, key_scale, topk, quant_mode, **kwargs)
        rows = picks.size(0)
        picks = torch.full((rows, 1, topk), -1, dtype=torch.int32)
        if rows:
            picks[0, 0, 0] = 150
            picks[0, 0, 1] = 3
            picks[0, 0, 2] = 3
            picks[0, 0, 3] = 1
        return picks, extra_a, extra_b

    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: (indexer, recorder.metadata))
    backend = _make_backend()

    topk = _drive(backend, _StubIndexerLayer(1, 2), _make_dsa(), _make_shared(2))

    assert topk[0, 0].tolist() == [_WINDOW + 2, _WINDOW + 4, _WINDOW + 4, -1]
    assert bool((topk[1:] == -1).all())


def test_qli_v3_ratio2_uses_the_compressed_capacity_and_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()

    _drive(backend, _StubIndexerLayer(1, 2), _make_dsa(), _make_shared(2))

    call = recorder.indexer_calls[0]
    assert call["key"].shape == (_SHADOW_BLOCKS, _BLOCK_SIZE, 1, _INDEX_HEAD_DIM)
    assert call["seqused_k"].tolist() == [length // 2 for length in _CTX_LENS]
    assert call["cmp_residual_k"].tolist() == [length % 2 for length in _CTX_LENS]
    assert call["cmp_ratio"] == 2
    assert call["layout_k"] == "PA_BBND"


def test_qli_v3_metadata_kwargs_per_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()

    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))

    assert len(recorder.metadata_calls) == 2
    by_ratio = {kwargs["cmp_ratio"]: (args, kwargs) for args, kwargs in recorder.metadata_calls}
    assert set(by_ratio) == {1, 2}
    for ratio, (args, kwargs) in by_ratio.items():
        assert args[:3] == (_INDEX_N_HEADS, 1, _INDEX_HEAD_DIM)
        assert args[3] == min(_INDEX_TOPK, max(_CAPACITY // ratio, 1))
        assert args[4] == 2
        assert kwargs["batch_size"] == _ROWS
        assert kwargs["max_seqlen_q"] == 1
        assert kwargs["max_seqlen_k"] == max(_CAPACITY // ratio, 1)
        assert kwargs["layout_q"] == "TND"
        assert kwargs["layout_k"] == "PA_BBND"
        assert kwargs["mask_mode"] == csa2._QLI_V3_MASK_MODE
        assert kwargs["cu_seqlens_k"] is None
        assert kwargs["cu_seqlens_q"].tolist() == [0, 1, 2, 3, 4]
        assert kwargs["seqused_k"].tolist() == [length // ratio for length in _CTX_LENS]
        if ratio > 1:
            assert kwargs["cmp_residual_k"].tolist() == [length % ratio for length in _CTX_LENS]
        else:
            assert kwargs["cmp_residual_k"] is None


def test_qli_v3_state_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_qli(monkeypatch)
    backend = _make_backend()
    dsa = _make_dsa()

    _drive(backend, _StubIndexerLayer(2, 1), dsa, _make_shared(1))

    assert set(backend._qli_state) == {(_ROWS, _CAPACITY)}
    state = backend._qli_state[(_ROWS, _CAPACITY)]
    assert isinstance(state, _QliGraphState)
    assert set(state.metadata) == {1, 2}
    assert state.cu_seqlens_q.dtype == torch.int32
    assert not hasattr(state, "cu_seqlens_k")
    assert torch.equal(state.seqused_k[2], torch.div(dsa.seq_lens, 2, rounding_mode="floor"))
    assert torch.equal(state.cmp_residual[2], torch.remainder(dsa.seq_lens, 2))

    metadata_before = state.metadata[2]
    seqused_before = state.seqused_k[2]
    dsa.seq_lens = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    backend._refresh_qli_state(dsa, state)
    assert state.metadata[2] is metadata_before
    assert state.seqused_k[2] is seqused_before
    assert state.seqused_k[1].tolist() == [10, 11, 12, 13]
    assert state.cmp_residual[2].tolist() == [0, 1, 0, 1]


def test_qli_v3_state_refuses_to_build_under_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_ops(monkeypatch, _QliRecorder())
    backend = _make_backend()
    monkeypatch.setattr(csa2, "_is_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="eager warmup"):
        _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))


def test_quantize_indexer_rows_int8_matches_the_reference() -> None:
    gen = torch.Generator().manual_seed(7)
    x = (torch.randn(6, 128, generator=gen) * 3.0).to(torch.bfloat16)
    codes, scale = csa2._quantize_indexer_rows_int8(x)

    ref_codes, ref_scale = _reference_indexer_quant(x)
    assert codes.dtype == torch.int8 and codes.shape == x.shape
    assert scale.dtype == torch.float16 and scale.shape == (6,)
    assert torch.equal(codes, ref_codes)
    assert torch.equal(scale, ref_scale)
    assert bool((codes >= -127).all()) and bool((codes <= 127).all())
    assert bool((scale > 0).all())

    zeros = torch.zeros(2, 128, dtype=torch.bfloat16)
    zero_codes, zero_scale = csa2._quantize_indexer_rows_int8(zeros)
    assert bool((zero_codes == 0).all())
    assert torch.equal(zero_scale, torch.full((2,), 2.0**-24, dtype=torch.float16))


def test_qli_v3_quantize_indexer_rows_boundaries() -> None:
    full = torch.tensor([[127.0, -127.0, 0.0, 63.5]], dtype=torch.bfloat16)
    codes, scale = csa2._quantize_indexer_rows_int8(full)
    ref_codes, ref_scale = _reference_indexer_quant(full)
    assert torch.equal(codes, ref_codes) and torch.equal(scale, ref_scale)
    assert scale[0].item() == pytest.approx(1.0)
    assert codes[0].tolist() == [127, -127, 0, 64]

    over = torch.tensor([[254.0, -254.0, 0.0, 0.0]], dtype=torch.bfloat16)
    over_codes, over_scale = csa2._quantize_indexer_rows_int8(over)
    assert over_scale[0].item() == pytest.approx(2.0)
    assert over_codes[0, 0] == 127 and over_codes[0, 1] == -127

    special = torch.tensor(
        [[float("nan")] * 4, [float("inf"), -float("inf"), 0.0, 1.0]],
        dtype=torch.float32,
    ).to(torch.bfloat16)
    special_codes, special_scale = csa2._quantize_indexer_rows_int8(special)
    ref_special_codes, ref_special_scale = _reference_indexer_quant(special)
    assert torch.equal(special_codes, ref_special_codes)
    assert torch.allclose(special_scale.float(), ref_special_scale.float(), equal_nan=True)
    assert torch.isnan(special_scale[0])


def _make_writer_backend(*, candidate_source_layer_id: int = -1) -> Csa2AttentionBackend:
    return _make_backend(
        n_layers=2,
        compress_ratios=[0, 1],
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1],
        candidate_source_layer_id=candidate_source_layer_id,
    )


def _writer_dsa(rows: int) -> SimpleNamespace:
    block_zero = torch.zeros(1, 1, dtype=torch.int32)
    return SimpleNamespace(
        slot_mappings=[],
        block_tables=[[], [block_zero.clone(), block_zero, block_zero.clone()]],
        seq_lens=torch.tensor([rows], dtype=torch.int32),
        is_acl_graph=False,
        graph_token_capacity=0,
    )


def _run_writer(
    backend: Csa2AttentionBackend,
    layer: _StubWriterLayer,
    cache: _StubLayerCache,
    shared: CSA2SharedRuntime,
    *,
    rows: int,
) -> None:
    ctx = _make_ctx(rows)
    mapping = backend._resolve_cache_mapping(layer.layer_id, layer.mode.compress_ratio)
    backend._run_compressor(
        layer,
        ctx,
        _writer_dsa(rows),
        cache,
        mapping,
        [rows],
        torch.tensor([rows], dtype=torch.int32),
        shared,
    )


def test_qli_v3_index_plane_quantizes_and_scatters(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_qli(monkeypatch)
    backend = _make_writer_backend()
    layer = _StubWriterLayer(1)
    cache = _StubLayerCache()
    shared = CSA2SharedRuntime()

    _run_writer(backend, layer, cache, shared, rows=5)

    assert cache.index.dtype == torch.int8
    assert cache.index.shape == (_SHADOW_BLOCKS, _BLOCK_SIZE, csa2._QLI_V3_KEY_NUM_HEADS, _INDEX_HEAD_DIM)
    assert cache.indexer_scale.dtype == torch.float16
    assert cache.indexer_scale.shape == (_SHADOW_BLOCKS, _BLOCK_SIZE, csa2._QLI_V3_KEY_NUM_HEADS)
    assert shared.index_cache is cache.index
    assert shared.index_scale is cache.indexer_scale

    latent = layer.compress_rmsnorm(layer.compress_kv_score(_make_ctx(5).x)[0])
    index_k = apply_interleaved_rope(layer.indexer_key(latent), torch.arange(5), _make_ctx(5).cos_sin_cache)
    expected_codes, expected_scales = csa2._quantize_indexer_rows_int8(index_k)
    assert torch.equal(cache.index.view(-1, _INDEX_HEAD_DIM)[:5], expected_codes)
    assert torch.equal(cache.indexer_scale.view(-1)[:5], expected_scales)

    _run_writer(backend, layer, cache, shared, rows=5)
    assert shared.index_cache is cache.index
    assert shared.index_scale is cache.indexer_scale


def test_qli_v3_index_plane_written_for_candidate_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_ops(monkeypatch, _QliRecorder())
    backend = _make_writer_backend(candidate_source_layer_id=1)
    layer = _StubWriterLayer(1)
    shared = CSA2SharedRuntime()

    _run_writer(backend, layer, _StubLayerCache(), shared, rows=5)

    assert shared.index_cache is not None
    assert shared.index_scale is not None


def test_qli_v3_falls_back_and_keeps_the_eager_result(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _QliRecorder()
    _install_ops(monkeypatch, recorder)
    backend = _make_backend()

    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    eager = _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))
    _install_ops(monkeypatch, recorder)

    narrow = _make_backend(index_head_dim=16)
    assert torch.equal(_drive(narrow, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1)), eager)

    off_graph = _make_dsa(is_acl_graph=False)
    assert torch.equal(_drive(backend, _StubIndexerLayer(2, 1), off_graph, _make_shared(1)), eager)

    assert recorder.indexer_calls == []


def test_qli_v3_candidate_layers_stay_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    backend = _make_backend(candidate_source_layer_id=1)
    dsa = _make_dsa()

    shared = _make_shared(2)
    _drive(backend, _StubIndexerLayer(1, 2), dsa, shared)
    assert shared.candidates is not None
    consumer = _make_shared(1)
    consumer_reason = backend._qli_v3_fallback_reason(
        _StubIndexerLayer(2, 1), dsa, [1] * _ROWS, consumer, consumer.index_cache
    )
    assert consumer_reason is not None and "not registered" in consumer_reason

    assert shared.candidate_blocks is None


def _candidate_plan_backend(**kwargs: Any) -> Csa2AttentionBackend:
    params: dict[str, Any] = {"compress_ratios": [0, 1, 1], "candidate_source_layer_id": 1}
    params.update(kwargs)
    return _make_backend(**params)


def test_qli_v3_candidate_mode_is_chosen_per_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()
    plain = _make_backend(compress_ratios=[0, 1, 1])
    shared = _make_shared(1)

    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)
    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)
    _drive(plain, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))

    assert [call["candidate_mode"] for call in recorder.indexer_calls] == [
        csa2._QLI_V3_CAND_MODE_SOURCE,
        csa2._QLI_V3_CAND_MODE_CONSUMER,
        csa2._QLI_V3_CAND_MODE_OFF,
    ]


def test_qli_v3_candidate_source_publishes_the_block_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()
    shared = _make_shared(1)

    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)

    call = recorder.indexer_calls[0]
    assert call["candidate_mode"] == csa2._QLI_V3_CAND_MODE_SOURCE == 1
    assert "candidate_topk_index" not in call
    assert call["candidate_topk_blocks"] == 2048
    assert call["candidate_block_size"] == 8
    assert shared.candidate_blocks is not None
    assert shared.candidate_blocks.dtype == torch.int32
    assert shared.candidate_blocks.shape == (_ROWS, 1, 2048)
    assert shared.candidate_blocks_layer == 1


def test_qli_v3_candidate_consumer_uses_the_published_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()
    shared = _make_shared(1)

    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)
    published = shared.candidate_blocks
    assert published is not None
    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)

    call = recorder.indexer_calls[-1]
    assert call["candidate_mode"] == csa2._QLI_V3_CAND_MODE_CONSUMER == 2
    assert call["candidate_topk_index"] is published
    assert call["candidate_topk_blocks"] == 2048
    assert call["candidate_block_size"] == 8


def test_qli_v3_candidate_consumer_requires_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()

    with pytest.raises(RuntimeError, match="requires the candidate pool"):
        _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))


def test_qli_v3_candidate_layers_fall_back_when_the_op_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    backend = _candidate_plan_backend()
    shared = _make_shared(1)

    for layer_id in (1, 2):
        reason = backend._qli_v3_fallback_reason(
            _StubIndexerLayer(layer_id, 1), _make_dsa(), [1] * _ROWS, shared, shared.index_cache
        )
        assert reason is not None and "not registered" in reason

    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)
    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)
    assert shared.candidates is not None
    assert shared.candidate_blocks is None


def test_qli_v3_candidate_invalid_config_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _stub_qli(monkeypatch)

    bad_block = _candidate_plan_backend(candidate_block_size=16)
    shared = _make_shared(1)
    reason = bad_block._qli_v3_fallback_reason(
        _StubIndexerLayer(1, 1), _make_dsa(), [1] * _ROWS, shared, shared.index_cache
    )
    assert reason is not None and "candidate block size 16" in reason
    _drive(bad_block, _StubIndexerLayer(1, 1), _make_dsa(), _make_shared(1))
    assert recorder.indexer_calls == []

    bad_topk = _candidate_plan_backend(candidate_topk_blocks=1000)
    shared = _make_shared(1)
    reason = bad_topk._qli_v3_fallback_reason(
        _StubIndexerLayer(1, 1), _make_dsa(), [1] * _ROWS, shared, shared.index_cache
    )
    assert reason is not None and "candidate topk blocks 1000" in reason


def test_qli_v3_candidate_consumer_rejects_a_pool_whose_marker_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()
    shared = _make_shared(1)
    shared.candidate_blocks = torch.zeros(1, 1, 8, dtype=torch.int32)
    shared.candidate_blocks_layer = 99

    with pytest.raises(RuntimeError, match="requires the candidate pool"):
        _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)
    assert recorder.indexer_calls == []


def test_qli_v3_candidate_pool_does_not_leak_across_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()

    first = backend._shared_runtime()
    _seed_index_plane(first, _make_shared(1))
    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), first)
    assert first.candidate_blocks is not None
    assert first.candidate_blocks_layer == 1

    backend.reset_forward()
    second = backend._shared_runtime()
    assert second is not first
    assert second.candidate_blocks is None
    assert second.candidate_blocks_layer == -1

    _seed_index_plane(second, _make_shared(1))
    with pytest.raises(RuntimeError, match="requires the candidate pool"):
        _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), second)


@pytest.mark.parametrize(
    "present",
    [
        {"quant_lightning_indexer_v2_va": object()},
        {"quant_lightning_indexer_v2_metadata_va": object()},
        {},
    ],
)
def test_qli_v3_ops_requires_both_native_handles(
    monkeypatch: pytest.MonkeyPatch,
    present: dict[str, Any],
) -> None:
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    monkeypatch.setattr(torch.ops, "xllm_ops", SimpleNamespace(**present), raising=False)
    assert csa2._get_qli_v3_ops() is None


def test_qli_v3_ops_returns_none_without_the_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    monkeypatch.setattr(torch.ops, "xllm_ops", None, raising=False)
    assert csa2._get_qli_v3_ops() is None


def test_qli_v3_ops_ignores_the_temporary_wheel_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    monkeypatch.setattr(
        torch.ops,
        "cann_ops_transformer",
        SimpleNamespace(quant_lightning_indexer_v3=object()),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops,
        "xllm_ops",
        SimpleNamespace(quant_lightning_indexer_v2_metadata=object()),
        raising=False,
    )
    assert csa2._get_qli_v3_ops() is None

    monkeypatch.setattr(
        torch.ops,
        "xllm_ops",
        SimpleNamespace(quant_lightning_indexer_v2_metadata_va=object()),
        raising=False,
    )
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    assert csa2._get_qli_v3_ops() is None


def test_qli_v3_ops_caches_a_hit_and_not_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(csa2, "has_op_api", lambda _: True)
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    indexer = object()
    metadata = object()
    monkeypatch.setattr(
        torch.ops,
        "xllm_ops",
        SimpleNamespace(
            quant_lightning_indexer_v2_va=indexer,
            quant_lightning_indexer_v2_metadata_va=metadata,
        ),
        raising=False,
    )

    resolved = csa2._get_qli_v3_ops()
    assert resolved == (indexer, metadata)
    monkeypatch.setattr(torch.ops, "xllm_ops", None, raising=False)
    assert csa2._get_qli_v3_ops() is resolved

    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    monkeypatch.setattr(torch.ops, "xllm_ops", SimpleNamespace(), raising=False)
    assert csa2._get_qli_v3_ops() is None
    monkeypatch.setattr(
        torch.ops,
        "xllm_ops",
        SimpleNamespace(
            quant_lightning_indexer_v2_va=indexer,
            quant_lightning_indexer_v2_metadata_va=metadata,
        ),
        raising=False,
    )
    assert csa2._get_qli_v3_ops() == (indexer, metadata)


def test_qli_v3_eager_read_dequantizes_the_int8_index_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    backend = _make_backend()

    codes = torch.randint(-127, 128, (_SHADOW_BLOCKS, _BLOCK_SIZE, 1, _INDEX_HEAD_DIM), dtype=torch.int8)
    scales = (torch.rand(_SHADOW_BLOCKS, _BLOCK_SIZE, 1) + 0.1).to(torch.float16)
    shared = CSA2SharedRuntime(
        index_cache=codes,
        index_scale=scales,
        index_block_table=torch.arange(_ROWS * 2, dtype=torch.int32).view(_ROWS, 2),
        compress_ratio=1,
    )

    seen: dict[str, torch.Tensor] = {}
    real_score = csa2._indexer_score

    def spy_score(q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if "k" not in seen:
            seen["k"] = index_k.clone()
        return real_score(q, index_k, weights)

    monkeypatch.setattr(csa2, "_indexer_score", spy_score)
    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)

    expected = codes.view(-1, _INDEX_HEAD_DIM)[:_CAPACITY].float() * scales.view(-1)[:_CAPACITY].float().unsqueeze(-1)
    assert torch.allclose(seen["k"], expected)
    assert not torch.equal(seen["k"], codes.view(-1, _INDEX_HEAD_DIM)[:_CAPACITY].float())


def _reference_rebase(
    picks: torch.Tensor,
    visible: torch.Tensor,
    offset: int,
    *,
    sort: bool,
) -> torch.Tensor:
    sentinel = torch.iinfo(torch.int32).max
    sortable = torch.where(
        (picks >= 0) & (picks < visible),
        picks,
        torch.full_like(picks, sentinel),
    )
    if sort:
        sortable = sortable.sort(dim=-1).values
    picks = torch.where(sortable >= sentinel, torch.full_like(sortable, -1), sortable)
    return torch.where(picks >= 0, picks + offset, torch.full_like(picks, -1)).to(torch.int32)


def test_qli_rebase_always_matches_the_sorted_reference() -> None:
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        picks = torch.randint(-1, 12, (5, 6), generator=generator, dtype=torch.int32)
        visible = torch.randint(0, 6, (5, 1), generator=generator, dtype=torch.int32)
        assert torch.equal(
            csa2._rebase_qli_picks(picks, visible, offset=128),
            _reference_rebase(picks, visible, 128, sort=True),
        )


def test_qli_picks_order_does_not_change_pure_torch_contract() -> None:
    picks = torch.tensor([[3, 0, -1, 7, 1], [2, 1, -1, -1, -1]], dtype=torch.int32)
    visible = torch.tensor([[4], [2]], dtype=torch.int32)
    offset = 16

    sorted_picks = csa2._rebase_qli_picks(picks, visible, offset=offset)
    unsorted_picks = _reference_rebase(picks, visible, offset, sort=False)
    assert not torch.equal(sorted_picks, unsorted_picks)

    num_rows = offset + int(visible.max())
    assert torch.equal(
        csa2._picked_row_mask(sorted_picks, num_rows),
        csa2._picked_row_mask(unsorted_picks, num_rows),
    )


_PREFILL_INDEX_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8]
_PREFILL_N_LAYERS = 9
_PREFILL_RATIOS = [0] + [1] * 8


def _make_prefill_backend(**kwargs: Any) -> Csa2AttentionBackend:
    return _make_backend(
        n_layers=_PREFILL_N_LAYERS,
        compress_ratios=list(_PREFILL_RATIOS),
        kv_source_layer_ids=[1],
        index_source_layer_ids=list(_PREFILL_INDEX_LAYERS),
        **kwargs,
    )


def _make_prefill_ctx(q_lens: list[int]) -> Csa2LayerContext:
    return _make_ctx(sum(int(length) for length in q_lens))


def _drive_prefill(
    backend: Csa2AttentionBackend,
    layer: _StubIndexerLayer,
    dsa: SimpleNamespace,
    shared: CSA2SharedRuntime,
    q_lens: list[int],
) -> torch.Tensor:
    return backend._run_indexer(layer, _make_prefill_ctx(q_lens), dsa, list(q_lens), dsa.seq_lens, shared)


def _install_topk_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counter = {"calls": 0}
    real = csa2._indexer_topk

    def counted(*args: Any, **kwargs: Any) -> torch.Tensor:
        counter["calls"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(csa2, "_indexer_topk", counted)
    return counter


class _PerSeqPicksRecorder(_QliRecorder):
    def indexer(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        topk: int,
        quant_mode: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _picks, _values, candidate_out = super().indexer(
            query, key, weights, query_scale, key_scale, topk, quant_mode, **kwargs
        )
        cu_seqlens = kwargs["cu_seqlens_q"].tolist()
        picks = torch.full((query.size(0), 1, topk), -1, dtype=torch.int32)
        for seq in range(len(cu_seqlens) - 1):
            picks[cu_seqlens[seq], 0, 0] = 0
            if topk > 1:
                picks[cu_seqlens[seq], 0, 1] = 2
        return picks, torch.empty(0, dtype=torch.bfloat16), candidate_out


def test_qli_v3_prefill_op_missing_restores_eager_indexer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    topk_counter = _install_topk_counter(monkeypatch)
    backend = _make_backend()
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10])

    topk = _drive_prefill(backend, _StubIndexerLayer(2, 1), dsa, _make_shared(1), [4])

    assert topk_counter["calls"] == 1
    assert topk.shape == (1, 4, 4)
    assert backend._qli_state == {}


def test_qli_v3_prefill_fuses_every_index_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    topk_counter = _install_topk_counter(monkeypatch)
    recorder = _QliRecorder()
    _install_ops(monkeypatch, recorder)
    backend = _make_prefill_backend()
    assert backend.candidate_source_layer_id == -1
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10, 5, 8, 12])
    q_lens = [3, 1, 2, 4]
    shared = _make_shared(1)
    ctx = _make_prefill_ctx(q_lens)

    topk = None
    for layer_id in _PREFILL_INDEX_LAYERS:
        topk = backend._run_indexer(_StubIndexerLayer(layer_id, 1), ctx, dsa, q_lens, dsa.seq_lens, shared)

    assert len(recorder.indexer_calls) == len(_PREFILL_INDEX_LAYERS) == 8
    assert len(recorder.metadata_calls) == 8
    assert topk_counter["calls"] == 0
    assert topk is not None and topk.shape == (4, 4, 4)
    assert shared.topk_idxs is topk


def test_qli_v3_prefill_dispatch_relaxes_for_an_eager_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ops(monkeypatch, _QliRecorder())
    backend = _make_backend()
    layer = _StubIndexerLayer(2, 1)
    graph_dsa = _make_dsa()
    shared = _make_shared(1)
    eager_dsa = _make_dsa(is_acl_graph=False)
    prefill_lens = [3, 1, 1, 1]

    assert backend._qli_v3_fallback_reason(layer, eager_dsa, prefill_lens, shared, shared.index_cache) is None
    assert (
        backend._qli_v3_fallback_reason(layer, eager_dsa, [1] * _ROWS, shared, shared.index_cache)
        == "not an ACL-graph decode step"
    )
    assert (
        backend._qli_v3_fallback_reason(layer, graph_dsa, prefill_lens, shared, shared.index_cache)
        == "query length != 1"
    )


def test_qli_v3_prefill_multiseq_descriptor_and_rectangular_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10, 5, 8, 12])
    q_lens = [3, 1, 2, 4]
    shared = _make_shared(1)

    topk = _drive_prefill(backend, _StubIndexerLayer(2, 1), dsa, shared, q_lens)

    call = recorder.indexer_calls[0]
    assert call["cu_seqlens_q"].tolist() == [0, 3, 4, 6, 10]
    assert call["seqused_k"].tolist() == [10, 5, 8, 12]
    assert call["cu_seqlens_k"] is None
    assert call["cmp_residual_k"] is None
    assert call["max_seqlen_q"] == 4
    assert call["layout_q"] == "TND"
    assert call["layout_k"] == "PA_BBND"
    assert call["mask_mode"] == 3
    assert call["cmp_ratio"] == 1
    assert call["candidate_mode"] == csa2._QLI_V3_CAND_MODE_OFF
    _args, kwargs = recorder.metadata_calls[0]
    assert kwargs["cu_seqlens_q"].tolist() == [0, 3, 4, 6, 10]
    assert kwargs["seqused_k"].tolist() == [10, 5, 8, 12]
    assert kwargs["max_seqlen_q"] == 4
    assert kwargs["batch_size"] == 4
    assert kwargs["max_seqlen_k"] == 12
    assert topk.shape == (4, 4, 4)
    assert shared.topk_idxs is topk
    assert shared.topk_layer == 2


def test_qli_v3_prefill_rebases_with_the_per_sequence_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _PerSeqPicksRecorder()
    _install_ops(monkeypatch, recorder)
    backend = _make_backend()
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10, 12])
    q_lens = [2, 3]
    shared = _make_shared(1)

    topk = _drive_prefill(backend, _StubIndexerLayer(2, 1), dsa, shared, q_lens)

    assert topk[0, 0].tolist() == [130, -1, -1, -1]
    assert topk[1, 0].tolist() == [131, 133, -1, -1]
    assert int(topk[0, 1].sum()) == -4
    assert int(topk[1, 1:].sum()) == -8


def test_qli_v3_prefill_covers_all_eight_layers_with_a_candidate_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topk_counter = _install_topk_counter(monkeypatch)
    recorder = _QliRecorder()
    _install_ops(monkeypatch, recorder)
    backend = _make_prefill_backend(candidate_source_layer_id=5)
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10, 5])
    q_lens = [3, 2]
    shared = _make_shared(1)
    ctx = _make_prefill_ctx(q_lens)

    for layer_id in _PREFILL_INDEX_LAYERS:
        backend._run_indexer(_StubIndexerLayer(layer_id, 1), ctx, dsa, q_lens, dsa.seq_lens, shared)

    assert len(recorder.indexer_calls) == 8
    assert topk_counter["calls"] == 0
    assert [call["candidate_mode"] for call in recorder.indexer_calls] == [
        csa2._QLI_V3_CAND_MODE_OFF,
        csa2._QLI_V3_CAND_MODE_OFF,
        csa2._QLI_V3_CAND_MODE_OFF,
        csa2._QLI_V3_CAND_MODE_OFF,
        csa2._QLI_V3_CAND_MODE_SOURCE,
        csa2._QLI_V3_CAND_MODE_CONSUMER,
        csa2._QLI_V3_CAND_MODE_CONSUMER,
        csa2._QLI_V3_CAND_MODE_CONSUMER,
    ]
    assert "candidate_topk_index" not in recorder.indexer_calls[4]
    assert "candidate_topk_index" in recorder.indexer_calls[-1]


def test_qli_v3_prefill_ratio2_uses_the_compressed_capacity_and_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[11, 7])
    q_lens = [2, 2]
    shared = _make_shared(2)

    _drive_prefill(backend, _StubIndexerLayer(1, 2), dsa, shared, q_lens)

    call = recorder.indexer_calls[0]
    assert call["cmp_ratio"] == 2
    assert call["seqused_k"].tolist() == [5, 3]
    assert call["cmp_residual_k"].tolist() == [1, 1]
    assert call["max_seqlen_q"] == 2
    _args, kwargs = recorder.metadata_calls[0]
    assert kwargs["max_seqlen_k"] == 5
    assert kwargs["seqused_k"].tolist() == [5, 3]


def test_qli_v3_prefill_leaves_decode_graph_steps_on_the_decode_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _StubIndexerLayer(2, 1)

    def _boom(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("an ACL-graph decode step must not enter the prefill branch")

    monkeypatch.setattr(Csa2AttentionBackend, "_run_indexer_qli_v3_prefill", _boom)
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()
    topk = _drive(backend, layer, _make_dsa(), _make_shared(1))

    assert backend._qli_state
    assert topk.shape[1] == 1
    assert recorder.metadata_calls[0][1]["max_seqlen_q"] == 1
    assert recorder.metadata_calls[0][1]["max_seqlen_k"] == _CAPACITY
    assert recorder.indexer_calls[0]["cu_seqlens_q"].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("missing_api", ("aclnnQuantLightningIndexerV2Va", "aclnnQuantLightningIndexerV2MetadataVa"))
def test_qli_v3_requires_loaded_native_entry_points(monkeypatch: pytest.MonkeyPatch, missing_api: str) -> None:
    indexer, metadata = object(), object()
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    monkeypatch.setattr(
        csa2.torch.ops,
        "xllm_ops",
        SimpleNamespace(
            quant_lightning_indexer_v2_va=indexer,
            quant_lightning_indexer_v2_metadata_va=metadata,
        ),
    )
    monkeypatch.setattr(csa2, "has_op_api", lambda name: name != missing_api)
    assert csa2._get_qli_v3_ops() is None
    monkeypatch.setattr(csa2, "has_op_api", lambda _: True)
    assert csa2._get_qli_v3_ops() == (indexer, metadata)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires the native QLI operators on an Ascend NPU",
)
@pytest.mark.parametrize("ratio", [1, 2])
def test_native_qli_capacity_bound_and_graph_length_refresh(ratio: int) -> None:
    torch.npu.set_device(0)
    backend = Csa2AttentionBackend(
        compress_ratios=[2, 1],
        window_size=128,
        n_layers=2,
        num_heads=16,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=32,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("npu"),
        dtype=torch.bfloat16,
        kv_source_layer_ids=[0, 1],
        index_source_layer_ids=[0, 1],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=128,
    )
    rows, capacity, block = 2, 2048, 128
    key = torch.ones((rows * capacity // block, block, 1, 128), device="npu", dtype=torch.int8)
    scales = torch.arange(1, capacity + 1, device="npu", dtype=torch.float32).half()
    scales = scales.repeat(rows).view(rows * capacity // block, block, 1)
    table = torch.arange(rows * capacity // block, device="npu", dtype=torch.int32).view(rows, -1)
    query = torch.ones((rows, 32, 128), device="npu", dtype=torch.bfloat16)
    ctx = SimpleNamespace(x=torch.zeros(rows, 7168, device="npu", dtype=torch.bfloat16))
    layer = SimpleNamespace(
        layer_id=0,
        indexer_weights=lambda x: torch.ones((rows, 32), device=x.device, dtype=x.dtype),
    )
    dsa = SimpleNamespace(
        seq_lens=torch.tensor([700 * ratio, 900 * ratio], device="npu", dtype=torch.int32),
        graph_token_capacity=capacity,
        is_acl_graph=True,
    )
    shared = CSA2SharedRuntime(
        index_cache=key,
        index_scale=scales,
        index_block_table=table,
        compress_ratio=ratio,
    )
    state = backend._qli_state_for(dsa)

    def assert_picks(actual: torch.Tensor, counts: tuple[int, int]) -> None:
        for row, count in enumerate(counts):
            expected = torch.arange(count - 512, count, device="npu", dtype=torch.int32) + 129
            torch.testing.assert_close(actual[row, 0], expected, rtol=0, atol=0)

    args = (layer, query, ctx, dsa, [1, 1], shared, table, capacity // ratio)
    assert_picks(backend._run_indexer_qli_v3(*args), (700, 900))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = backend._run_indexer_qli_v3(*args)
    dsa.seq_lens.copy_(torch.tensor([800 * ratio, 1000 * ratio], device="npu", dtype=torch.int32))
    backend._refresh_qli_state(dsa, state)
    graph.replay()
    assert_picks(captured, (800, 1000))
