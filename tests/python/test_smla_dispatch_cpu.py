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

from xllm.python.attention import csa2_attention as csa2
from xllm.python.attention.csa2_attention import (
    Csa2AttentionBackend,
    CSA2SharedRuntime,
)

HEADS = 16
WIDTH = 512
WINDOW = 128
CAPACITY = 128


def _make_ops(events: list[tuple[str, dict[str, Any]]]) -> tuple[Any, Any]:
    def fused(q: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        ori = kwargs.get("ori_kv")
        events.append(
            (
                "smla_va",
                {
                    "kwargs": dict(kwargs),
                    "ori_snapshot": None if ori is None else ori.clone(),
                },
            )
        )
        return torch.empty_like(q), torch.empty(0, dtype=torch.float32, device=q.device)

    def metadata(num_heads_q: int, num_heads_kv: int, head_dim: int, **kwargs: Any) -> torch.Tensor:
        del num_heads_q, num_heads_kv, head_dim
        events.append(("smla_va_metadata", dict(kwargs)))
        return torch.zeros(1024, dtype=torch.int32, device=torch.device("cpu"))

    return fused, metadata


def _backend(ratio: int, num_heads: int = HEADS, window: int = WINDOW) -> Csa2AttentionBackend:
    kv_sources = [0] if ratio > 0 else []
    return Csa2AttentionBackend(
        compress_ratios=[ratio],
        window_size=window,
        n_layers=1,
        num_heads=num_heads,
        attn_head_dim=WIDTH,
        index_topk=8,
        index_n_heads=32,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_source_layer_ids=kv_sources,
        index_source_layer_ids=kv_sources,
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=128,
    )


def _fixture(
    ratio: int = 0,
    rows: int = 2,
    num_heads: int = HEADS,
    window: int = WINDOW,
    capacity: int = CAPACITY,
) -> tuple[Any, ...]:
    backend = _backend(ratio, num_heads=num_heads, window=window)
    q = torch.randn(rows, num_heads, WIDTH, dtype=torch.bfloat16)
    kv = torch.randn(rows, WIDTH, dtype=torch.bfloat16)
    swa = torch.randn(rows, WINDOW, WIDTH, dtype=torch.bfloat16)
    slots = torch.tensor([row * WINDOW + 5 for row in range(rows)], dtype=torch.int32)
    table = torch.arange(rows, dtype=torch.int32).view(-1, 1)
    picks = torch.full((rows, 1, 8), -1, dtype=torch.int32)
    picks[..., 0] = WINDOW + 2
    layer = SimpleNamespace(
        mode=SimpleNamespace(compress_ratio=ratio, is_full=False, is_reindex=False, is_reuse=True),
        layer_id=0,
        attn_sink=torch.zeros(num_heads),
        attn_sink_loaded=False,
        _csa2_ctx=SimpleNamespace(x=torch.zeros(1), qr=torch.zeros(1, 1), positions=torch.zeros(1)),
    )
    dsa = SimpleNamespace(
        seq_lens=torch.full((rows,), 32, dtype=torch.int32),
        seq_lens_q=torch.ones(rows, dtype=torch.int32),
        slot_mappings=[[slots]],
        block_tables=[[table]],
        is_acl_graph=True,
        graph_token_capacity=capacity,
    )
    shared = CSA2SharedRuntime(
        key_cache=torch.randn(rows, CAPACITY, WIDTH, dtype=torch.bfloat16),
        key_block_table=table,
        topk_idxs=picks,
    )
    backend._kv_caches = [SimpleNamespace(swa=swa)]
    backend._resolve_cache_mapping = lambda *_: SimpleNamespace(ori_cache_idx=0)  # type: ignore[method-assign]
    backend._metadata = SimpleNamespace(dsa_metadata=dsa)
    backend._v41_shared = shared
    return backend, layer, dsa, shared, q, kv, swa, slots, picks


def _enable_smla(monkeypatch: pytest.MonkeyPatch, events: list[tuple[str, dict[str, Any]]]) -> None:
    fused, metadata = _make_ops(events)
    monkeypatch.setattr(csa2, "_smla_ops", (fused, metadata))
    monkeypatch.setattr(csa2, "_smla_device_supported", lambda q: True)


def _reason(backend: Csa2AttentionBackend, case: tuple[Any, ...] | None = None, **overrides: Any) -> str | None:
    case = _fixture() if case is None else case
    _, layer, dsa, shared, q, _, swa, slots, picks = case
    values: dict[str, Any] = {
        "q": q,
        "layer": layer,
        "dsa": dsa,
        "q_lens": [1] * int(q.size(0)),
        "cp_context": None,
        "topk": picks,
        "shared": shared,
        "layer_cache": SimpleNamespace(swa=swa),
        "swa_slot": slots,
    }
    values.update(overrides)
    return backend._smla_fallback_reason(**values)


def test_capability_predicate_accepts_supported_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    _enable_smla(monkeypatch, events)
    assert _reason(_backend(0), case=_fixture(ratio=0)) is None
    assert _reason(_backend(1), case=_fixture(ratio=1)) is None


def test_capability_predicate_requires_compressed_operands(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    _enable_smla(monkeypatch, events)
    case = _fixture(ratio=1)
    case[3].key_cache = None
    assert "compressed cache unavailable" in (_reason(case[0], case=case) or "")
    case = _fixture(ratio=1)
    case[3].key_cache = torch.zeros(2, CAPACITY, dtype=torch.bfloat16)
    assert "layout" in (_reason(case[0], case=case) or "")
    case = _fixture(ratio=1)
    assert "no indexer top-k" in (_reason(case[0], case=case, topk=None) or "")
    case = _fixture(ratio=1)
    empty = torch.empty((2, 1, 0), dtype=torch.int32)
    assert _reason(case[0], case=case, topk=empty) is None


def test_smla_graph_state_caches_the_ring_column_map(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    _enable_smla(monkeypatch, events)
    backend, layer, dsa, shared, q, kv, swa, slots, picks = _fixture(ratio=1)
    state = backend._get_smla_state(dsa)
    assert state.ring_cols.dtype == torch.int64
    assert state.ring_cols.tolist() == [0] * state.ring_cols.numel()
    assert set(state.metadata) == set(backend._smla_used_ratio_ids) == {0, 1}
    assert state.cu_seqlens_q.tolist() == [0, 1, 2]
    assert set(state.seqused_cmp) == {1}
    assert state.cmp_residual == {}
    first_cols = state.ring_cols
    metadata_calls = len([event for event in events if event[0] == "smla_va_metadata"])
    dsa.seq_lens.fill_(7)
    backend._refresh_smla_state(dsa, state)
    assert state.ring_cols is first_cols
    assert state.seqused_cmp[1].tolist() == [7, 7]
    assert len([event for event in events if event[0] == "smla_va_metadata"]) == metadata_calls + len(state.metadata)


def test_smla_graph_state_residual_only_for_ratio_above_one(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    _enable_smla(monkeypatch, events)
    backend, layer, dsa, shared, q, kv, swa, slots, picks = _fixture(ratio=2)
    state = backend._get_smla_state(dsa)
    assert set(state.seqused_cmp) == {2}
    assert set(state.cmp_residual) == {2}
    assert state.cmp_residual[2].dtype == torch.int32
    dsa.seq_lens.fill_(7)
    backend._refresh_smla_state(dsa, state)
    assert state.seqused_cmp[2].tolist() == [3, 3]
    assert state.cmp_residual[2].tolist() == [1, 1]


@pytest.mark.parametrize("shape", [(), (512,), (2, 512)])
def test_query_rank_falls_back_without_index_error(monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]) -> None:
    _enable_smla(monkeypatch, [])
    case = _fixture()
    assert "query is not" in _reason(case[0], case=case, q=torch.empty(shape, dtype=torch.bfloat16))


@pytest.mark.parametrize("shape", [(0, 128, 512), (2, 0, 512)])
def test_empty_cache_falls_back_before_native_dispatch(monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]) -> None:
    _enable_smla(monkeypatch, [])
    case = _fixture()
    cache = SimpleNamespace(swa=torch.empty(shape, dtype=torch.bfloat16))
    assert "cache layout" in _reason(case[0], case=case, layer_cache=cache)


def _record_scatter(monkeypatch: pytest.MonkeyPatch, events: list[tuple[str, dict[str, Any]]]) -> None:
    real_scatter = csa2._scatter_by_slot

    def scatter(cache: torch.Tensor, slot: torch.Tensor, value: torch.Tensor) -> Any:
        events.append(("scatter", {"slot": slot.clone(), "value": value.clone()}))
        return real_scatter(cache, slot, value)

    monkeypatch.setattr(csa2, "_scatter_by_slot", scatter)


def _contains_rows(bank: torch.Tensor, rows: torch.Tensor) -> bool:
    flat = bank.reshape(-1, bank.size(-1))
    return all(bool((flat == row).all(dim=-1).any()) for row in rows)


def test_execute_csa2_layer_fused_scatters_current_kv_once_before_smla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    _enable_smla(monkeypatch, events)
    _record_scatter(monkeypatch, events)
    backend, layer, dsa, shared, q, kv, swa, slots, picks = _fixture(ratio=0)
    old_ring = swa.clone()

    out = backend.execute_csa2_layer(q, kv, layer)

    assert out.shape == q.shape
    names = [name for name, _ in events]
    assert names.count("scatter") == 1
    assert names.count("smla_va") == 1
    assert names.index("scatter") < names.index("smla_va")
    snapshot = next(payload["ori_snapshot"] for name, payload in events if name == "smla_va")
    assert snapshot is not None
    assert _contains_rows(snapshot, kv)
    assert not _contains_rows(snapshot, old_ring[:, 5])
    for row in range(swa.size(0)):
        assert torch.equal(swa[row, 5], kv[row])


@pytest.mark.parametrize("ratio", [1, 2])
def test_execute_csa2_layer_fused_compressed_operands(
    monkeypatch: pytest.MonkeyPatch,
    ratio: int,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    _enable_smla(monkeypatch, events)
    _record_scatter(monkeypatch, events)
    backend, layer, dsa, shared, q, kv, swa, slots, picks = _fixture(ratio=ratio)
    dsa.seq_lens = torch.full((2,), 33, dtype=torch.int32)

    out = backend.execute_csa2_layer(q, kv, layer)

    assert out.shape == q.shape
    names = [name for name, _ in events]
    assert names.count("scatter") == 1
    assert names.count("smla_va") == 1
    assert names.index("scatter") < names.index("smla_va")
    for row in range(swa.size(0)):
        assert torch.equal(swa[row, 5], kv[row])
    fused_kwargs = next(payload["kwargs"] for name, payload in events if name == "smla_va")
    indices = fused_kwargs["cmp_sparse_indices"]
    expected_width = csa2._round_cmp_width(min(backend.index_topk, CAPACITY // ratio))
    assert indices.dtype == torch.int32
    assert indices.shape == (2, 1, expected_width)
    assert indices[:, 0, 0].tolist() == [WINDOW + 2 - WINDOW - 1] * 2
    assert (indices[:, 0, 1:] == -1).all()
    snapshot = next(payload["ori_snapshot"] for name, payload in events if name == "smla_va")
    for row in range(swa.size(0)):
        assert torch.equal(snapshot[row, 5, 0], kv[row])
    assert fused_kwargs["cmp_ratio"] == ratio
    assert fused_kwargs["cmp_kv"].shape == (2, CAPACITY, 1, WIDTH)
    assert fused_kwargs["cmp_block_table"] is shared.key_block_table
    assert fused_kwargs["seqused_ori_kv"] is dsa.seq_lens
    assert fused_kwargs["seqused_cmp_kv"].tolist() == [33 // ratio] * 2
    if ratio > 1:
        assert fused_kwargs["cmp_residual_kv"].tolist() == [33 % ratio] * 2
    else:
        assert fused_kwargs["cmp_residual_kv"] is None
    assert fused_kwargs["cmp_mask_mode"] == csa2._SMLA_CMP_MASK_MODE
    assert fused_kwargs["ori_mask_mode"] == csa2._SMLA_ORI_MASK_MODE
    assert fused_kwargs["sinks"] is None
    meta = next(payload for name, payload in events if name == "smla_va_metadata" and payload["cmp_ratio"] == ratio)
    assert meta["cmp_topk"] == indices.size(-1)
    assert meta["batch_size"] == 2
    assert meta["max_seqlen_q"] == 1
    assert meta["max_seqlen_cmp_kv"] == max(CAPACITY // ratio, 1)
    assert meta["has_cmp_kv"] is True
    assert meta["seqused_cmp_kv"] is fused_kwargs["seqused_cmp_kv"]
    assert meta["cmp_residual_kv"] is fused_kwargs["cmp_residual_kv"]
    assert meta["seqused_ori_kv"] is dsa.seq_lens


def test_execute_csa2_layer_reference_sees_old_ring_and_scatters_once_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    _enable_smla(monkeypatch, events)
    monkeypatch.setattr(csa2, "_get_smla_ops", lambda: None)
    _record_scatter(monkeypatch, events)
    backend, layer, dsa, shared, q, kv, swa, slots, picks = _fixture(ratio=0)
    old_ring = swa.clone()
    captured: dict[str, torch.Tensor] = {}

    def reference_attention(
        q_: torch.Tensor,
        kv_: torch.Tensor,
        layer_: Any,
        dsa_: Any,
        q_lens_: list[int],
        ctx_lens_: Any,
        topk_: torch.Tensor | None,
        shared_: Any,
        cp_context_: Any,
    ) -> torch.Tensor:
        captured["swa"] = swa.clone()
        captured["kv"] = kv_.clone()
        events.append(("reference_attention", {}))
        return torch.zeros_like(q_)

    monkeypatch.setattr(backend, "_csa2_attention", reference_attention)

    out = backend.execute_csa2_layer(q, kv, layer)

    assert out.shape == q.shape
    names = [name for name, _ in events]
    assert names.count("scatter") == 1
    assert names.index("reference_attention") < names.index("scatter")
    assert torch.equal(captured["kv"], kv)
    for row in range(swa.size(0)):
        assert torch.equal(captured["swa"][row, 5], old_ring[row, 5])
        assert not torch.equal(captured["swa"][row, 5], kv[row])
        assert torch.equal(swa[row, 5], kv[row])
