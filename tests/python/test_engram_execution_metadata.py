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

from tests.python._engram_reference import _reference_engram_lookup
from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.models import deepseek_v41_engram as e


@pytest.mark.parametrize("layer_ids", [[0, 1], [1, 0]])
def test_execution_metadata_preserves_graph_storage_across_eager_growth(
    monkeypatch: pytest.MonkeyPatch, layer_ids: list[int]
) -> None:
    monkeypatch.setattr(e, "_host_buffer", lambda shape, dtype: (torch.empty(shape, dtype=dtype), True, None))
    lookup_calls = 0

    def lookup(weight: torch.Tensor, scale: torch.Tensor, ids: torch.Tensor, output: torch.Tensor) -> None:
        nonlocal lookup_calls
        lookup_calls += 1
        _reference_engram_lookup(weight, scale, ids, output)

    monkeypatch.setattr(e, "_get_engram_lookup_op", lambda: lookup)
    layout = e.EngramLayout(layer_ids, [512, 512], 3, 2, 32, 64, 16, 0)

    def hash_state() -> e.EngramHashState:
        return e.EngramHashState(layout, torch.arange(64) % 16, e.compute_hash_multipliers(layer_ids, 3, 16))

    state, oracle = hash_state(), hash_state()
    engrams = [
        e.Engram(
            layout,
            layer_ids.index(layer_id),
            32,
            1,
            1e-5,
            1,
            0,
            torch.bfloat16,
            torch.device("cpu"),
            storage=e.ENGRAM_STORAGE_INT8,
            offload=True,
        )
        for layer_id in sorted(layer_ids)
    ]
    rng = torch.Generator().manual_seed(571)
    for engram in engrams:
        engram.embed_tokens.weight.data.copy_(
            torch.randint(-120, 120, engram.embed_tokens.weight.shape, dtype=torch.int8, generator=rng)
        )
        engram.embed_tokens.weight_scale.data.copy_(
            torch.rand(engram.embed_tokens.weight_scale.shape, generator=rng) * 0.02
        )
    cache = torch.zeros(6, 8, 1, 32, dtype=torch.bfloat16)
    oracle.ensure_cache(cache)
    builder = e.EngramExecutionMetadataBuilder(state, engrams, 8)
    builder.bind_layer_caches([SimpleNamespace(swa=cache)])
    model = torch.nn.Identity()
    model.execution_metadata_builders = (builder,)
    runner = EagerRunner(model, SimpleNamespace(), torch.device("cpu"))

    def batch(
        ids: list[int], positions: list[int], counts: list[int], computed: list[int], capacity: int | None = None
    ) -> InputBatch:
        ids = torch.tensor(ids, dtype=torch.int64)
        ends = [0]
        for n in counts:
            ends.append(ends[-1] + n)
        capacity = len(ids) if capacity is None else capacity
        padding = torch.arange(capacity) >= sum(counts)
        return InputBatch(
            ids,
            torch.tensor(positions),
            len(counts),
            len(ids),
            capacity,
            tuple(counts),
            tuple(computed),
            tuple(ends),
            (False,) * len(counts),
            padding,
            False,
        )

    def check(result: e.EngramExecutionMetadata, b: InputBatch, table: torch.Tensor) -> None:
        slots = []
        seq = []
        for i, (count, computed) in enumerate(zip(b.num_scheduled_tokens, b.num_computed_tokens)):
            slots.extend([int(table[i, 0]) * 8 + p % 8 for p in range(computed, computed + count)])
            seq.append(count)
        slots.extend([-1] * (b.num_tokens_after_padding - len(slots)))
        if sum(seq) < b.num_tokens_after_padding:
            seq.append(b.num_tokens_after_padding - sum(seq))
        cu = torch.tensor([0] + list(torch.tensor(seq).cumsum(0)), dtype=torch.int64)
        dead = e.image_sentinel_mask(b.input_ids) | b.is_padding
        hashes = oracle(b.input_ids, b.positions, cu, dead, None, torch.tensor(slots), table, 8)
        for engram in engrams:
            ids = hashes[:, engram.layer_hash_index]
            emb = engram.embed_tokens
            expected = (emb.weight[ids].float() * emb.weight_scale[ids].repeat_interleave(32, -1)).bfloat16()
            torch.testing.assert_close(result.local_rows[engram.layer_hash_index], expected, rtol=0, atol=0)
        torch.testing.assert_close(result.gate_mask, ~dead)

    table = torch.tensor([[1], [3]], dtype=torch.int32)
    b = batch([1, 2, 3, 4], [0, 1, 7, 8], [2, 2], [0, 7])
    meta = SimpleNamespace(multi_block_tables=[table], new_cache_slots_host_values=None)
    graph = builder.allocate_persistent(b, meta)
    pointer = graph.local_rows[0].data_ptr()
    builder.update_persistent(graph, b, meta)
    check(graph, b, table)
    large = batch(list(range(10, 19)), list(range(9)), [9], [0])
    large_meta = SimpleNamespace(
        multi_block_tables=[torch.tensor([[2]], dtype=torch.int32)],
        new_cache_slots_host_values=None,
        slot_mapping=torch.empty(0, dtype=torch.int32),
    )
    eager = runner._build_execution_contexts(large_meta, large)[e.EngramExecutionMetadata]
    check(eager, large, large_meta.multi_block_tables[0])
    assert eager.local_rows[0].data_ptr() != pointer
    b2 = batch([5, 6, 7, 0], [2, 3, 9, 0], [2, 1], [2, 9])
    builder.update_persistent(graph, b2, meta)
    check(graph, b2, table)
    assert graph.local_rows[0].data_ptr() == pointer

    missing_table = SimpleNamespace(multi_block_tables=[], slot_mapping=torch.empty(0, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="requires the SWA block table"):
        builder.update_persistent(graph, b2, missing_table)

    dummy = InputBatch.from_runtime(
        torch.tensor([63, 62, 61, 60]),
        torch.zeros(4, dtype=torch.int64),
        SimpleNamespace(
            num_reqs=0,
            num_tokens=4,
            num_scheduled_tokens=[],
            num_computed_tokens=[],
            query_start_loc=[0],
            is_prefilling=[],
        ),
    )
    lookup_count = lookup_calls
    builder.update_persistent(graph, dummy, missing_table)
    assert not graph.gate_mask.any()
    assert not graph.local_rows[0].any()
    assert graph.local_rows[0].data_ptr() == pointer
    idle_eager = runner._build_execution_contexts(missing_table, dummy)[e.EngramExecutionMetadata]
    assert not idle_eager.gate_mask.any()
    assert not idle_eager.local_rows[0].any()
    assert lookup_calls == lookup_count

    b3 = batch([8, 9, 10, 0], [4, 5, 10, 0], [2, 1], [4, 10])
    builder.update_persistent(graph, b3, meta)
    check(graph, b3, table)
    assert graph.local_rows[0].data_ptr() == pointer


@pytest.mark.parametrize("with_engram", [False, True])
@pytest.mark.parametrize("slot_count", [0, 1])
def test_flat_slot_builders_still_reject_incomplete_slots(with_engram: bool, slot_count: int) -> None:
    class FlatSlotsBuilder:
        metadata_type = object

        def build(self, input_batch: InputBatch, metadata: AttentionMetadata) -> None:
            raise AssertionError("invalid slots must be rejected before building metadata")

    builders = (FlatSlotsBuilder(),)
    if with_engram:
        builders += (e.EngramExecutionMetadataBuilder(None, [], 8),)
    model = torch.nn.Identity()
    model.execution_metadata_builders = builders
    runner = EagerRunner(model, SimpleNamespace(), torch.device("cpu"))
    metadata = SimpleNamespace(slot_mapping=torch.arange(slot_count, dtype=torch.int32))
    with pytest.raises(ValueError, match="one slot per input token"):
        runner._build_execution_contexts(metadata, SimpleNamespace(num_tokens=2))


def test_staging_wait_uses_current_stream_and_upload_records_its_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    streams = [SimpleNamespace(synchronize=lambda: calls.append("d2h-current-stream"))]

    class Event:
        def record(self, stream: object) -> None:
            self.owner = stream
            calls.append("record-upload")

        def synchronize(self) -> None:
            calls.append(self.owner)

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda device: streams[-1],
            synchronize=lambda *args: (_ for _ in ()).throw(AssertionError("global wait")),
            Event=Event,
        ),
        raising=False,
    )
    obj = SimpleNamespace(setup_device=torch.device("privateuseone:0"))
    e._sync_accelerator(obj.setup_device)
    e.EngramEmbedding._record_upload(obj)
    streams.append(SimpleNamespace())
    e.EngramEmbedding._wait_for_upload(obj)
    assert calls == ["d2h-current-stream", "record-upload", streams[0]]
    event = obj._upload_event
    e.EngramEmbedding._record_upload(obj)
    assert obj._upload_event is event
    assert event.owner is streams[1]
    e.EngramEmbedding._wait_for_upload(obj)
    assert calls[-2:] == ["record-upload", streams[1]]


def test_fallback_rejects_table_demotion_after_persistent_metadata_allocation() -> None:
    layout = e.EngramLayout([0], [290], 3, 2, 32, 64, 16, 0)
    engram = e.Engram(
        layout, 0, 32, 1, 1e-5, 1, 0, torch.bfloat16, torch.device("cpu"), storage=e.ENGRAM_STORAGE_INT8, offload=True
    )
    state = e.EngramHashState(layout, torch.arange(64) % 16, e.compute_hash_multipliers([0], 3, 16))
    builder = e.EngramExecutionMetadataBuilder(state, [engram], 8)
    builder.bind_layer_caches([SimpleNamespace(swa=torch.zeros(4, 8, 1, 32))])
    batch = SimpleNamespace(num_tokens_after_padding=4, input_ids=torch.arange(4))
    rows = builder.allocate_persistent(batch, None).local_rows[0]
    weight = engram.embed_tokens.weight
    with pytest.raises(RuntimeError, match="persistent execution metadata"):
        engram.embed_tokens.fallback_to_device()
    assert engram.offload
    assert engram.embed_tokens.weight is weight
    assert rows.shape == (4, 4, 32)


def test_unbound_swa_preserves_eager_profiling_but_refuses_capture() -> None:
    builder = e.EngramExecutionMetadataBuilder(None, [], 8)
    batch = SimpleNamespace(num_tokens_after_padding=4, input_ids=torch.arange(4))
    context = builder.build(batch, None)
    assert context.gate_mask is None
    assert context.local_rows == ()
    with pytest.raises(RuntimeError, match="bound SWA cache"):
        builder.allocate_persistent(batch, None)
