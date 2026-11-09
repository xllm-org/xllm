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

import pytest
import torch

from xllm.python import distributed
from xllm.python.attention.csa2_attention import Csa2AttentionBackend, Csa2LayerContext, CSA2SharedRuntime
from xllm.python.model_executor import dsa_cp
from xllm.python.model_executor.dsa_cp import build_dsa_cp_context, get_dsa_cp_runtime
from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context


@pytest.mark.parametrize("lengths", ([1], [1] * 5, [3, 5, 1], [0, 1, 0, 2], [129, 257], [4, 8, 12]))
@pytest.mark.parametrize("size", (2, 4))
def test_query_intersections_preserve_causal_positions(lengths: list[int], size: int) -> None:
    lengths = list(lengths)
    prefix = torch.tensor([17 + i for i in range(len(lengths))])
    context_lengths = prefix + torch.tensor(lengths)
    total = sum(lengths)
    restored = []
    for rank in range(size):
        plan = build_dsa_cp_context(total, lengths, size, rank)
        assert all(suffix >= 0 for suffix in plan.query_suffix)
        rows = plan.shard(torch.arange(total))
        restored.extend(rows.tolist())
        local_lengths = plan.local_context_lengths(context_lengths)
        offset = 0
        for seq, length in enumerate(lengths):
            local_rows = rows[(rows >= offset) & (rows < offset + length)]
            assert local_rows.numel() == plan.local_q_lens[seq]
            if local_rows.numel():
                assert local_lengths[seq] == prefix[seq] + local_rows[-1] - offset + 1
            else:
                assert local_lengths[seq] == 0
            offset += length
    assert restored == list(range(total))


@pytest.mark.parametrize("tokens", (1, 3, 5, 9))
def test_head_exchange_restores_tokens_and_discards_padding(monkeypatch: pytest.MonkeyPatch, tokens: int) -> None:
    size, heads, width = 4, 8, 3
    reference = torch.arange(tokens * heads * width).reshape(tokens, heads, width)
    per_rank = (tokens + size - 1) // size
    global_padded = torch.zeros(per_rank * size, heads, width, dtype=reference.dtype)
    global_padded[:tokens] = reference
    for rank in range(size):
        plan = build_dsa_cp_context(tokens, [tokens], size, rank)

        def _exchange(send: torch.Tensor, world_size: int, group_name: str, rank: int = rank) -> torch.Tensor:
            assert world_size == size and group_name == "tp"
            local = global_padded[rank * per_rank : (rank + 1) * per_rank]
            expected_send = local.reshape(per_rank, size, heads // size, width).permute(1, 0, 2, 3)
            torch.testing.assert_close(send, expected_send.reshape(-1, heads // size, width))
            return global_padded[:, rank * (heads // size) : (rank + 1) * (heads // size)].contiguous()

        monkeypatch.setattr(distributed, "all_to_all", _exchange, raising=False)
        actual = plan.restore_tp_heads(plan.shard(reference))
        torch.testing.assert_close(actual, reference[:, rank * (heads // size) : (rank + 1) * (heads // size)])


@pytest.mark.parametrize("rank", range(4))
def test_sparse_tail_matches_global_query_reference(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    torch.manual_seed(17)
    q_lens, heads, width = [5, 7], 8, 4
    global_lengths = torch.tensor([8, 11])
    q = torch.randn(sum(q_lens), heads, width)
    kv = torch.randn(sum(q_lens), width)
    global_dsa = SimpleNamespace(seq_lens=global_lengths, block_tables=[])
    backend = object.__new__(Csa2AttentionBackend)
    backend.window_size = 2
    backend.scale = width**-0.5
    backend._kv_caches = [SimpleNamespace(swa=None)]
    backend._resolve_cache_mapping = lambda *_: SimpleNamespace(ori_cache_idx=0)
    layer = SimpleNamespace(
        mode=SimpleNamespace(compress_ratio=0), layer_id=0, attn_sink=torch.randn(heads), _csa2_ctx=None
    )
    reference = backend._csa2_attention(q, kv, layer, global_dsa, q_lens, global_lengths, None, CSA2SharedRuntime())
    plan = build_dsa_cp_context(sum(q_lens), q_lens, 4, rank)
    local = backend._csa2_attention(
        plan.shard(q),
        kv,
        layer,
        global_dsa,
        list(plan.local_q_lens),
        plan.local_context_lengths(global_lengths),
        None,
        CSA2SharedRuntime(),
        plan,
    )
    torch.testing.assert_close(local, plan.shard(reference))


def test_rejects_invalid_topology_and_lengths() -> None:
    with pytest.raises(ValueError, match="query lengths exceed the packed token buffer"):
        build_dsa_cp_context(3, [4], 4, 0)
    with pytest.raises(ValueError, match="Invalid DSA CP topology or token count"):
        build_dsa_cp_context(3, [3], 4, 4)


def test_cp_execution_keeps_full_layer_context_and_passes_query_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(17)
    q_lens, heads, width = [5, 7], 8, 4
    lengths = torch.tensor([8, 11])
    q = torch.randn(sum(q_lens), heads, width)
    kv = torch.randn(sum(q_lens), width)
    ctx = Csa2LayerContext(torch.randn(12, 8), torch.randn(12, 4), torch.arange(12), torch.empty(12, 4))
    layer = SimpleNamespace(
        mode=SimpleNamespace(compress_ratio=1, is_full=True, is_reindex=False, is_reuse=False),
        layer_id=0,
        attn_sink=torch.randn(heads),
        _csa2_ctx=ctx,
        enable_dsa_cp=True,
    )
    dsa = SimpleNamespace(
        seq_lens=lengths,
        seq_lens_q=torch.tensor(q_lens),
        seq_lens_q_host=q_lens,
        block_tables=[],
        slot_mappings=[],
        is_acl_graph=False,
    )
    metadata = SimpleNamespace(dsa_metadata=dsa)
    backend = object.__new__(Csa2AttentionBackend)
    backend.window_size, backend.scale = 2, width**-0.5
    backend.index_topk = 3
    backend._kv_caches = [SimpleNamespace(swa=None)]
    backend._resolve_cache_mapping = lambda *_: SimpleNamespace(ori_cache_idx=0)
    backend._current_forward_metadata = lambda: metadata
    backend._shared_runtime = CSA2SharedRuntime
    reference = backend._csa2_attention(q, kv, layer, dsa, q_lens, lengths, None, CSA2SharedRuntime())
    calls = []

    def _compressor(layer: object, received_ctx: Csa2LayerContext, *args: object) -> None:
        assert received_ctx is ctx
        calls.append("compressor")

    def _indexer(layer: object, received_ctx: Csa2LayerContext, *args: object) -> None:
        assert received_ctx is not ctx
        torch.testing.assert_close(received_ctx.x, runtime.plan.shard(ctx.x))
        torch.testing.assert_close(received_ctx.qr, runtime.plan.shard(ctx.qr))
        torch.testing.assert_close(received_ctx.positions, runtime.plan.shard(ctx.positions))
        calls.append("indexer")

    monkeypatch.setattr(backend, "_run_compressor", _compressor)
    monkeypatch.setattr(backend, "_run_indexer", _indexer)
    context = ForwardContext(backend, torch.device("cpu"), metadata, [])
    with forward_context(context):
        runtime = get_dsa_cp_runtime(12, 4, 1)
        output = backend.execute_csa2_layer(runtime.plan.shard(q), kv, layer)
    assert calls == ["compressor", "indexer"]
    assert layer._csa2_ctx is ctx
    torch.testing.assert_close(output, runtime.plan.shard(reference))


def test_single_query_lengths_retain_live_device_values_without_host_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = build_dsa_cp_context(5, [1] * 5, 4, 1)

    def _forbid_host_copy(*args: object, **kwargs: object) -> None:
        raise AssertionError("Decode graph lengths must not copy host constants")

    monkeypatch.setattr(torch.Tensor, "new_tensor", _forbid_host_copy)
    lengths = torch.tensor([100, 200, 300, 400, 500])
    torch.testing.assert_close(plan.local_context_lengths(lengths), torch.tensor([0, 0, 300, 400, 0]))
    lengths.add_(7)
    torch.testing.assert_close(plan.local_context_lengths(lengths), torch.tensor([0, 0, 307, 407, 0]))


def test_plan_and_device_lengths_are_shared_only_within_one_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"build": 0, "tolist": 0, "lengths": 0}
    original_build = build_dsa_cp_context
    original_lengths = dsa_cp.DsaCpContext.local_context_lengths

    class _QueryLengths:
        def tolist(self) -> list[int]:
            calls["tolist"] += 1
            return [3, 5, 1]

    def _build(*args: object) -> dsa_cp.DsaCpContext:
        calls["build"] += 1
        return original_build(*args)

    def _lengths(plan: dsa_cp.DsaCpContext, lengths: torch.Tensor) -> torch.Tensor:
        calls["lengths"] += 1
        return original_lengths(plan, lengths)

    monkeypatch.setattr(dsa_cp, "build_dsa_cp_context", _build)
    monkeypatch.setattr(dsa_cp.DsaCpContext, "local_context_lengths", _lengths)
    previous = None
    for prefix in (10, 100):
        metadata = SimpleNamespace(
            dsa_metadata=SimpleNamespace(
                seq_lens_q=_QueryLengths(), seq_lens_q_host=[3, 5, 1], seq_lens=torch.tensor([3, 5, 1]) + prefix
            )
        )
        context = ForwardContext(None, torch.device("cpu"), metadata, [])
        with forward_context(context):
            runtime = get_dsa_cp_runtime(9, 4, 1)
            for _ in range(40):
                assert get_dsa_cp_runtime(9, 4, 1) is runtime
            assert runtime is not previous
            torch.testing.assert_close(runtime.local_context_lengths, torch.tensor([0, prefix + 3, 0]))
            previous = runtime
    assert calls == {"build": 2, "tolist": 0, "lengths": 2}


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("tokens", [1, 5, 8])
def test_decode_capture_rebuilds_live_lengths_after_warmup(
    monkeypatch: pytest.MonkeyPatch, rank: int, tokens: int
) -> None:
    def _forbid_host_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("Decode capture must not read device query lengths")

    monkeypatch.setattr(torch.Tensor, "tolist", _forbid_host_read)
    lengths = torch.arange(100, 100 + tokens)
    metadata = SimpleNamespace(
        dsa_metadata=SimpleNamespace(seq_lens_q=torch.ones(tokens), seq_lens=lengths, is_acl_graph=True)
    )
    execution_state = AclGraphExecutionState(persistent_buffers={})
    warmup = ForwardContext(None, torch.device("cpu"), metadata, [], execution_state=execution_state)
    with forward_context(warmup):
        warmup_runtime = get_dsa_cp_runtime(tokens, 4, rank)
    lengths.add_(7)
    capture = ForwardContext(None, torch.device("cpu"), metadata, [], execution_state=execution_state)
    with forward_context(capture):
        runtime = get_dsa_cp_runtime(tokens, 4, rank)
        assert get_dsa_cp_runtime(tokens, 4, rank) is runtime
        assert runtime is not warmup_runtime
        rows = torch.arange(tokens)
        active = (rows >= runtime.plan.start) & (rows < runtime.plan.end)
        torch.testing.assert_close(runtime.local_context_lengths, torch.where(active, lengths, 0))
    assert not execution_state.persistent_buffers


@pytest.mark.parametrize("ratio", (1, 2))
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("graph_mode", (False, True))
@pytest.mark.parametrize("query_lengths", ([3, 8], [1, 3, 1, 2], [1, 1, 1, 1], [1]))
def test_compressed_picks_use_global_chunk_offset(
    ratio: int, rank: int, graph_mode: bool, query_lengths: list[int]
) -> None:
    torch.manual_seed(31)
    q_lens, heads, width, capacity = query_lengths, 8, 4, 4
    lengths = torch.tensor(q_lens) + 8
    q, kv = torch.randn(sum(q_lens), heads, width), torch.randn(sum(q_lens), width)
    compressed = torch.randn(len(q_lens), capacity, width)
    global_dsa = SimpleNamespace(
        seq_lens=lengths, block_tables=[], is_acl_graph=graph_mode, graph_token_capacity=capacity * ratio
    )
    backend = object.__new__(Csa2AttentionBackend)
    backend.window_size, backend.scale = 2, width**-0.5
    backend.index_topk = 3
    backend._kv_caches = [SimpleNamespace(swa=None)]
    backend._resolve_cache_mapping = lambda *_: SimpleNamespace(ori_cache_idx=0)
    backend._current_forward_metadata = lambda: SimpleNamespace(dsa_metadata=global_dsa)
    if not graph_mode:
        backend._committed_capacity = lambda *_: capacity
    backend._gather_cache_rows = lambda cache, table, seq, count, size: compressed[seq]
    shared = CSA2SharedRuntime(
        key_cache=torch.empty(len(q_lens), capacity, 1, width), key_block_table=torch.zeros(len(q_lens), 1)
    )
    layer = SimpleNamespace(
        mode=SimpleNamespace(compress_ratio=ratio), layer_id=0, attn_sink=torch.randn(heads), _csa2_ctx=None
    )
    global_picks = torch.full((len(q_lens), max(q_lens), 3), -1, dtype=torch.int32)
    for seq, q_len in enumerate(q_lens):
        global_picks[seq, :q_len, :2] = torch.tensor([0, 2]) + backend.window_size + q_len
    reference = backend._csa2_attention(q, kv, layer, global_dsa, q_lens, lengths, global_picks, shared)
    plan = build_dsa_cp_context(sum(q_lens), q_lens, 4, rank)
    local_picks = torch.full((len(q_lens), max(plan.local_q_lens), 3), -1, dtype=torch.int32)
    for seq, q_len in enumerate(plan.local_q_lens):
        local_picks[seq, :q_len, :2] = torch.tensor([0, 2]) + backend.window_size + q_len
    local = backend._csa2_attention(
        plan.shard(q),
        kv,
        layer,
        global_dsa,
        list(plan.local_q_lens),
        plan.local_context_lengths(lengths),
        local_picks,
        shared,
        plan,
    )
    torch.testing.assert_close(local, plan.shard(reference))


def test_current_dsa_cp_runtime_is_absent_outside_forward() -> None:
    assert dsa_cp.get_current_dsa_cp_runtime() is None
