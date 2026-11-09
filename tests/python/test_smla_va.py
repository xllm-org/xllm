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


from types import SimpleNamespace
from typing import Any

import pytest
import torch

from xllm.python.attention import csa2_attention as csa2
from xllm.python.attention.csa2_attention import (
    Csa2AttentionBackend,
    CSA2SharedRuntime,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="SMLA requires an Ascend NPU",
)


def _require_ops() -> None:
    if not hasattr(torch.ops.xllm_ops, "smla_va") or not hasattr(torch.ops.xllm_ops, "smla_va_metadata"):
        pytest.skip("smla_va native registration is required")


def _case(
    q_lens: list[int],
    ratio: int = 2,
    history: int = 12,
    topk: int = 8,
) -> tuple[Any, ...]:
    device = "npu"
    torch.manual_seed(42)
    heads, width = 16, 512
    capacity = max(128, (history + max(q_lens) + 127) // 128 * 128)
    lengths = torch.tensor(q_lens, dtype=torch.int32) + history
    q = (torch.randn(sum(q_lens), heads, width) * 0.3).bfloat16().to(device)
    swa = torch.randn(len(q_lens), 128, 1, width).bfloat16().to(device)
    compressed = torch.randn(len(q_lens), capacity, 1, width).bfloat16().to(device)
    table = torch.arange(len(q_lens), dtype=torch.int32).view(-1, 1).to(device)
    picks = torch.full((len(q_lens), max(q_lens), topk), -1, dtype=torch.int32)
    for seq, length in enumerate(q_lens):
        for row in range(length):
            count = min(topk, (history + row + 1) // max(ratio, 1))
            picks[seq, row, :count] = torch.arange(count) + 128 + length
    layer = SimpleNamespace(
        mode=SimpleNamespace(compress_ratio=ratio),
        layer_id=0,
        attn_sink=torch.linspace(-1, 1, heads).to(device),
        attn_sink_loaded=True,
    )
    backend = Csa2AttentionBackend(
        compress_ratios=[ratio],
        window_size=128,
        n_layers=1,
        num_heads=heads,
        attn_head_dim=width,
        index_topk=picks.size(-1),
        index_n_heads=32,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device(device),
        dtype=torch.bfloat16,
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=128,
    )
    backend._kv_caches = [SimpleNamespace(swa=swa)]
    backend._resolve_cache_mapping = lambda *_: SimpleNamespace(ori_cache_idx=0)
    dsa = SimpleNamespace(
        seq_lens=lengths.to(device),
        block_tables=[[table]],
        is_acl_graph=True,
        graph_token_capacity=capacity,
    )
    shared = CSA2SharedRuntime(key_cache=compressed, key_block_table=table)
    return backend, layer, dsa, shared, q, picks.to(device)


def test_smla_metadata_bucket_is_built_before_capture() -> None:
    _require_ops()
    backend, layer, dsa, shared, q, picks = _case([1, 1], ratio=2)
    state = backend._get_smla_state(dsa)
    assert (state.rows, state.capacity) == (2, dsa.graph_token_capacity)
    assert state.cu_seqlens_q.tolist() == [0, 1, 2]
    assert state.expanded_ori_bt.dtype == torch.int32
    assert set(state.metadata) == {0, 2}
    assert backend._get_smla_state(dsa) is state


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("history", [12, 140])
def test_smla_decode_matches_reference_tail(ratio: int, history: int) -> None:
    _require_ops()
    backend, layer, dsa, shared, q, picks = _case([1, 1], ratio=ratio, history=history)
    kv = q[:, 0, :].contiguous()
    for seq in range(q.size(0)):
        backend._kv_caches[0].swa[seq, history % 128, 0].copy_(kv[seq])
    fused = backend._csa2_attention_smla(q, layer, dsa, picks, shared)
    reference = backend._csa2_attention(q, kv, layer, dsa, [1, 1], dsa.seq_lens, picks, shared)
    assert fused.shape == reference.shape == q.shape
    assert fused.dtype == torch.bfloat16
    assert torch.isfinite(fused.float()).all()
    scale = reference.float().abs().amax().clamp_min(1.0)
    assert (fused.float() - reference.float()).abs().amax() <= 2e-2 * scale


def test_smla_no_candidates_falls_back_to_window_only_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_ops()
    backend, layer, dsa, shared, q, _ = _case([1, 1], ratio=2)
    empty = torch.empty((2, 1, 0), dtype=torch.int32, device=q.device)
    captured: dict[str, Any] = {}
    fused_op, metadata_op = csa2._get_smla_ops()

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return fused_op(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(csa2, "_smla_ops", (spy, metadata_op))
        backend._csa2_attention_smla(q, layer, dsa, empty, shared)
    assert captured["cmp_ratio"] == 0
    assert captured["cmp_kv"] is None
    assert captured["cmp_sparse_indices"] is None
    assert captured["cmp_mask_mode"] == csa2._SMLA_NO_MASK_MODE


def test_smla_missing_state_under_capture_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, layer, dsa, shared, q, picks = _case([1, 1], ratio=2)
    backend._smla_state.clear()
    with monkeypatch.context() as context:
        context.setattr(csa2, "_is_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="during capture"):
            backend._csa2_attention_smla(q, layer, dsa, picks, shared)


def test_smla_dispatch_predicate_keeps_reference_tail() -> None:
    backend, layer, dsa, shared, q, picks = _case([1, 1], ratio=2)
    slot = torch.tensor([0, 1], dtype=torch.int32)
    dsa.is_acl_graph = False
    assert backend._smla_fallback_reason(q, layer, dsa, [1, 1], None, picks, shared, backend._kv_caches[0], slot)
    dsa.is_acl_graph = True
    assert backend._smla_fallback_reason(q, layer, dsa, [2, 0], None, picks, shared, backend._kv_caches[0], slot)
    assert (
        backend._smla_fallback_reason(q, layer, dsa, [1, 1], None, picks, shared, backend._kv_caches[0], slot) is None
    )
