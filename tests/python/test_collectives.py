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

"""Tests for Python process-group rendezvous ownership."""

from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from xllm.python import distributed
from xllm.python.distributed import collectives
from xllm.python.distributed import cuda as cuda_collectives
from xllm.python.models import glm5_2


def test_tp_all_to_all_uses_native_group_without_python_rendezvous(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _exchange(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        assert world_size == 2
        calls.append(tensor.clone())
        return tensor.flip(0)

    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(tp_all_to_all=_exchange))
    value = torch.arange(12).reshape(4, 3)
    result = collectives.all_to_all(value, 2, "tp")
    torch.testing.assert_close(result, value.flip(0))
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], value)


@pytest.mark.parametrize("native_present", (False, True))
def test_all_to_all_fallback_accepts_noncontiguous_input(monkeypatch: pytest.MonkeyPatch, native_present: bool) -> None:
    group = SimpleNamespace(size=lambda: 2)
    calls = []

    def _no_native_runtime(tensor: torch.Tensor, world_size: int) -> None:
        calls.append(world_size)

    monkeypatch.setattr(collectives, "_native_runtime_op", lambda *_: _no_native_runtime if native_present else None)
    monkeypatch.setattr(collectives, "get_process_group", lambda *_: group)
    value = torch.arange(24).reshape(3, 8).t()
    assert not value.is_contiguous()

    def _exchange(output: torch.Tensor, input_tensor: torch.Tensor, group: object) -> None:
        assert input_tensor.is_contiguous() and output.is_contiguous()
        output.copy_(input_tensor.flip(0))

    monkeypatch.setattr(dist, "all_to_all_single", _exchange)
    result = collectives.all_to_all(value, 2, "tp")
    torch.testing.assert_close(result, value.flip(0))
    assert calls == ([2] if native_present else [])


def test_all_to_all_fallback_rejects_world_size_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collectives, "_native_runtime_op", lambda *_: None)
    monkeypatch.setattr(collectives, "get_process_group", lambda *_: SimpleNamespace(size=lambda: 4))
    with pytest.raises(RuntimeError, match="world-size mismatch"):
        collectives.all_to_all(torch.zeros(8, 3), 2, "tp")


@pytest.mark.parametrize("world_size", (1, 2))
def test_all_to_all_fake_matches_real_noncontiguous_strides(monkeypatch: pytest.MonkeyPatch, world_size: int) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    monkeypatch.setattr(collectives, "_native_runtime_op", lambda *_: None)
    monkeypatch.setattr(collectives, "get_process_group", lambda *_: SimpleNamespace(size=lambda: world_size))
    monkeypatch.setattr(dist, "all_to_all_single", lambda output, input_tensor, group: output.copy_(input_tensor))
    value = torch.arange(24).reshape(3, 8).t()
    actual = collectives.all_to_all(value, world_size, "tp")
    with FakeTensorMode() as mode:
        fake = collectives.all_to_all(mode.from_tensor(value), world_size, "tp")
    assert fake.shape == actual.shape
    assert fake.dtype == actual.dtype
    assert fake.device == actual.device
    assert fake.stride() == actual.stride()
    assert actual.is_contiguous() == (world_size > 1)


class _FakeGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.fixture(autouse=True)
def _clear_collective_state():
    def reset():
        collectives._groups.clear()
        collectives._group_ranks.clear()
        collectives._stores.clear()
        cuda_collectives._symm_eligible.clear()
        cuda_collectives._symm_buffers.clear()
        collectives._world_topology = None
        collectives._world_rank = None
        collectives._world_endpoint = None
        collectives._world_initialized = False

    reset()
    yield
    reset()


class _FakeStore:
    def __init__(self, topology: list[dict[str, object]] | None = None) -> None:
        self.values: dict[str, bytes] = {}
        if topology is not None:
            for rank, entry in enumerate(topology):
                self.values[f"xllm/python_collectives/topology/v1/{rank}"] = json.dumps(entry).encode("utf-8")

    def set(self, key: str, value: str) -> None:
        self.values[key] = value.encode("utf-8")

    def get(self, key: str) -> bytes:
        return self.values[key]


def _mock_process_groups(
    monkeypatch: pytest.MonkeyPatch,
    global_rank: int,
    topology: list[dict[str, object]] | None = None,
):
    """Stand in for c10d so the rendezvous can be inspected without a world.

    ``new_group`` reports this rank's position inside the membership it is
    handed, which is what the module checks its caller's rank against.
    """
    if topology is None:
        topology = [{"hostname": "node-0", "device_index": rank} for rank in range(16)]
    base_store = _FakeStore(topology)
    tcp_store = MagicMock(return_value=base_store)
    init_world = MagicMock()
    new_group = MagicMock(
        side_effect=lambda ranks, timeout, backend: _FakeGroup(
            ranks.index(global_rank) if global_rank in ranks else -1, len(ranks)
        )
    )
    monkeypatch.setattr(dist, "TCPStore", tcp_store)
    monkeypatch.setattr(dist, "init_process_group", init_world)
    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(collectives.socket, "gethostname", lambda: "node-0")
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda _a, _b: True)
    return base_store, tcp_store, init_world, new_group


def _run_glm_ep1_tp_collective(global_rank: int, rendezvous_path: str) -> None:
    world_size = 4
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous_path}",
            rank=global_rank,
            world_size=world_size,
            timeout=timedelta(seconds=20),
        )
        tp_groups = [
            dist.new_group(
                ranks=[0, 1],
                backend="gloo",
                timeout=timedelta(seconds=20),
            ),
            dist.new_group(
                ranks=[2, 3],
                backend="gloo",
                timeout=timedelta(seconds=20),
            ),
        ]
        collectives._groups[("tp", "cpu")] = tp_groups[global_rank // 2]
        # CP uses the world-sized moe_tp group, so every rank contributes to
        # the same expert output reduction.
        collectives._groups[("moe_tp", "cpu")] = dist.group.WORLD

        cp_rank = global_rank // 2
        tp_rank = global_rank % 2
        local_value = float(cp_rank * 10 + tp_rank + 1)
        routed = torch.tensor([[local_value]])
        shared = torch.tensor([[local_value * 10]])
        moe = SimpleNamespace(
            ep_size=1,
            moe_tp_size=world_size,
            cfg=SimpleNamespace(
                ep_size=1,
                tp_size=2,
                tp_rank=tp_rank,
                moe_tp_size=world_size,
                moe_tp_rank=global_rank,
                dp_size=1,
                cp_size=2,
                enable_attn_dp_weight_sharding=False,
            ),
        )

        # This numerical fixture deliberately uses CPU/Gloo even on an NPU
        # host; production dispatch is fixed by the platform, not the tensor.
        with (
            patch.object(collectives, "_all_reduce", dist.all_reduce),
            patch.object(
                glm5_2.distributed,
                "all_reduce_",
                collectives.all_reduce_,
                create=True,
            ),
        ):
            output = glm5_2.Glm52MoE._combine_expert_outputs(moe, routed, shared, False)

        expected = torch.tensor([[286.0]])
        torch.testing.assert_close(output, expected)
    finally:
        collectives._groups.clear()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_parallel_groups_share_one_multitenant_tcp_store(monkeypatch):
    base_store, tcp_store, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=0)

    collectives.init_process_group("tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 2, 0)
    collectives.init_process_group("moe_tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 2, 0)

    tcp_store.assert_called_once()
    assert tcp_store.call_args.args[:4] == ("127.0.0.1", 46001, 2, True)
    assert tcp_store.call_args.kwargs["wait_for_workers"] is False
    assert tcp_store.call_args.kwargs["multi_tenant"] is True

    # Every parallel group is a subgroup of one world, so the world rendezvous
    # happens once no matter how many groups the caller asks for.
    init_world.assert_called_once()
    assert init_world.call_args.kwargs["store"] is base_store
    assert init_world.call_args.kwargs["rank"] == 0
    assert init_world.call_args.kwargs["world_size"] == 2
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [
        [0, 1],
        [0, 1],
    ]


@pytest.mark.parametrize("group_name", ["tp", "dp", "moe_tp", "moe_ep", "cp", "layerwise", "dcp"])
def test_npu_all_reduce_binding_applies_to_every_group(monkeypatch: pytest.MonkeyPatch, group_name: str) -> None:
    from xllm.python.distributed.npu import all_reduce_on_current_stream

    def unexpected_native_call(x: torch.Tensor, comm: int) -> None:
        raise AssertionError("CPU input must be rejected before native execution")

    # The Python helper resolves the native symbol before validating arguments.
    monkeypatch.setattr(torch.ops.xllm_ops, "npu_all_reduce", unexpected_native_call, raising=False)
    assert collectives._all_reduce is all_reduce_on_current_stream
    collectives._groups[(group_name, "cpu")] = _FakeGroup(0, 2)
    # The selected NPU implementation must reject a CPU tensor for every group,
    # rather than silently routing non-TP groups back through c10d.
    with pytest.raises(RuntimeError, match="HCCL requires an NPU tensor, got cpu"):
        collectives.all_reduce_(torch.ones(1), group_name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("extra_elements", [0, 1])
def test_npu_all_reduce_selects_managed_stream_above_inline_limit(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, extra_elements: int
) -> None:
    from xllm.python.distributed.npu import hccl

    item_size = torch.empty((), dtype=dtype).element_size()
    value = torch.ones(hccl._CURRENT_STREAM_ALL_REDUCE_MAX_BYTES // item_size + extra_elements, dtype=dtype)
    group = _FakeGroup(0, 2)
    communicator = MagicMock(return_value=17)
    inline = MagicMock(side_effect=lambda tensor, comm: tensor.mul_(2))
    managed = MagicMock(side_effect=lambda tensor, group: tensor.mul_(2))
    # Test the size-based dispatch without constructing an NPU communicator.
    # Real HCCL graph replay is covered by the opt-in multi-rank probe.
    monkeypatch.setattr(hccl, "get_hccl_comm", communicator)
    monkeypatch.setattr(torch.ops.xllm_ops, "npu_all_reduce", inline)
    monkeypatch.setattr(hccl.dist, "all_reduce", managed)

    hccl.all_reduce_on_current_stream(value, group)

    communicator.assert_called_once_with(group, value.device)
    if extra_elements:
        managed.assert_called_once_with(value, group=group)
        inline.assert_not_called()
    else:
        inline.assert_called_once_with(value, 17)
        managed.assert_not_called()
    torch.testing.assert_close(value, torch.full_like(value, 2))


def test_native_runtime_bridge_bypasses_python_process_groups(monkeypatch):
    calls: list[str] = []

    runtime = SimpleNamespace(
        tp_all_reduce=lambda tensor: (calls.append("tp_reduce"), tensor.add_(1)),
        tp_all_gather=lambda tensor, dim: (
            calls.append(f"tp_gather:{dim}"),
            torch.cat((tensor, tensor), dim=dim),
        )[1],
        dp_all_gather=lambda tensor, counts: (
            calls.append(f"dp_gather:{counts}"),
            torch.cat((tensor, tensor), dim=0),
        )[1],
        moe_tp_all_reduce=lambda tensor: (
            calls.append("moe_tp_reduce"),
            tensor.add_(2),
        ),
        moe_ep_all_reduce=lambda tensor: (
            calls.append("moe_ep_reduce"),
            tensor.add_(4),
        ),
    )
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    python_reduce = MagicMock(side_effect=AssertionError("c10d fallback used"))
    python_gather = MagicMock(side_effect=AssertionError("c10d fallback used"))
    monkeypatch.setattr(collectives, "all_reduce_", python_reduce)
    monkeypatch.setattr(collectives, "all_gather", python_gather)

    value = torch.tensor([[1.0]])
    collectives.tp_all_reduce(value)
    gathered = collectives.tp_all_gather(value, 1, 2)
    dp_gathered, dp_offset = collectives.gather_dp_execution_tokens(
        value,
        (1, 1),
        rank=1,
    )
    collectives.moe_tp_all_reduce(value)
    collectives.moe_ep_all_reduce(value)

    assert calls == [
        "tp_reduce",
        "tp_gather:1",
        "dp_gather:[1, 1]",
        "moe_tp_reduce",
        "moe_ep_reduce",
    ]
    assert gathered.tolist() == [[2.0, 2.0]]
    assert dp_gathered.tolist() == [[2.0], [2.0]]
    assert dp_offset == 1
    assert value.tolist() == [[8.0]]
    python_reduce.assert_not_called()
    python_gather.assert_not_called()


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_npu_gather_preserves_rank_order_with_embedded_runtime(monkeypatch: pytest.MonkeyPatch, dim: int) -> None:
    group = _FakeGroup(1, 3)
    collectives._groups[("tp", "cpu")] = group
    native_gather = MagicMock(side_effect=AssertionError("Python group was bypassed"))
    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(tp_all_gather=native_gather))
    # A transposed local shard also checks the high-level layout contract.
    peers = [torch.arange(6).reshape(3, 2).T + 100 * rank for rank in range(3)]

    def gather(input: torch.Tensor, output: torch.Tensor, group: _FakeGroup) -> None:
        assert group.rank() == 1 and group.size() == 3
        assert input.is_contiguous()
        torch.testing.assert_close(input, peers[1])
        output.copy_(torch.stack(peers))

    monkeypatch.setattr(collectives, "_all_gather", gather)
    actual = collectives.tp_all_gather(peers[1], dim, 3)
    torch.testing.assert_close(actual, torch.cat(peers, dim=dim), rtol=0, atol=0)
    native_gather.assert_not_called()


def test_npu_variable_gather_preserves_empty_rank_and_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = [2, 0, 3]
    group = _FakeGroup(0, 3)
    collectives._groups[("dp", "cpu")] = group
    peers = [torch.arange(6).reshape(3, 2) + 100 * rank for rank in range(3)]

    def gather(input: torch.Tensor, output: torch.Tensor, group: _FakeGroup) -> None:
        torch.testing.assert_close(input[:2], peers[0][:2])
        assert input[2].count_nonzero() == 0
        output.copy_(torch.stack(peers))

    monkeypatch.setattr(collectives, "_all_gather", gather)
    actual = collectives.all_gather_variable(peers[0][:2], counts, 0, "dp")
    torch.testing.assert_close(actual, torch.cat((peers[0][:2], peers[2])), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("name", "group_name"),
    [("tp_all_reduce", "tp"), ("moe_tp_all_reduce", "moe_tp"), ("moe_ep_all_reduce", "moe_ep")],
)
def test_npu_reduce_uses_owned_python_group(monkeypatch: pytest.MonkeyPatch, name: str, group_name: str) -> None:
    group = _FakeGroup(0, 2)
    collectives._groups[(group_name, "cpu")] = group
    native_reduce = MagicMock(side_effect=AssertionError("Python group was bypassed"))
    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(**{name: native_reduce}))
    selected_groups = []

    def reduce(input: torch.Tensor, group: _FakeGroup) -> None:
        selected_groups.append(group)
        input.mul_(2)

    monkeypatch.setattr(collectives, "_all_reduce", reduce)
    actual = torch.tensor([3.0])
    getattr(collectives, name)(actual)
    assert selected_groups == [group]
    torch.testing.assert_close(actual, torch.tensor([6.0]), rtol=0, atol=0)
    native_reduce.assert_not_called()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
def test_glm_ep1_tp_reduce_does_not_mix_cp_cohorts(tmp_path: Path) -> None:
    rendezvous_path = tmp_path / "glm-ep1-tp-reduce"

    process_context = torch.multiprocessing.start_processes(
        _run_glm_ep1_tp_collective,
        args=(str(rendezvous_path),),
        nprocs=4,
        join=False,
        start_method="fork",
    )
    deadline = time.monotonic() + 30.0
    try:
        while not process_context.join(
            timeout=max(0.0, deadline - time.monotonic()),
            grace_period=5.0,
        ):
            if time.monotonic() >= deadline:
                pytest.fail("Gloo CP2 x TP2 collective test timed out")
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
        cleanup_deadline = time.monotonic() + 5.0
        for process in process_context.processes:
            process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        for process in process_context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)


def test_dcp_group_is_strided_like_kv_split_rank(monkeypatch):
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=2)

    # world=8, dcp=2 → group_count=4; membership matches rank/(world/dcp).
    collectives.init_process_group("dcp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 8, 2)

    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]


@pytest.mark.parametrize(
    ("group", "size", "world", "stride", "global_rank", "local_rank", "index", "own_members"),
    [
        ("cp", 2, 8, 2, 4, 0, 2, [4, 6]),
        ("dcp", 2, 8, 1, 4, 0, 2, [4, 5]),
        ("dcp", 2, 8, 4, 4, 1, 0, [0, 4]),
        ("dcp", 2, 16, 4, 12, 1, 4, [8, 12]),
    ],
)
def test_group_stride_membership_contracts(
    monkeypatch, group, size, world, stride, global_rank, local_rank, index, own_members
):
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=global_rank)

    group_obj = collectives.init_process_group(
        group, "127.0.0.1", 46001, local_rank, size, "cuda:0", global_rank, world, index, group_stride=stride
    )

    assert group_obj.rank() == local_rank and group_obj.size() == size
    assert list(collectives._group_ranks[(group, "cuda:0")]) == own_members
    assert new_group.call_args_list[index].kwargs["ranks"] == own_members


def test_dp_execution_gather_uses_fixed_collective_for_equal_counts(monkeypatch):
    value = torch.zeros(4, 8)
    gathered = torch.zeros(8, 8)
    fixed_gather = MagicMock(return_value=gathered)
    variable_gather = MagicMock()
    monkeypatch.setattr(collectives, "all_gather", fixed_gather)
    monkeypatch.setattr(collectives, "all_gather_variable", variable_gather)

    output, offset = collectives.gather_dp_execution_tokens(
        value,
        (4, 4),
        rank=1,
    )

    assert output is gathered
    assert offset == 4
    fixed_gather.assert_called_once_with(
        value,
        dim=0,
        world_size=2,
        group_name="dp",
    )
    variable_gather.assert_not_called()


def test_dp_execution_gather_uses_variable_collective_for_uneven_counts(monkeypatch):
    value = torch.zeros(1, 8)
    gathered = torch.zeros(4, 8)
    fixed_gather = MagicMock()
    variable_gather = MagicMock(return_value=gathered)
    monkeypatch.setattr(collectives, "all_gather", fixed_gather)
    monkeypatch.setattr(collectives, "all_gather_variable", variable_gather)

    output, offset = collectives.gather_dp_execution_tokens(
        value,
        (3, 1),
        rank=1,
    )

    assert output is gathered
    assert offset == 3
    variable_gather.assert_called_once_with(value, [3, 1], 1, "dp")
    fixed_gather.assert_not_called()


def test_dp_execution_gather_rejects_local_shape_mismatch() -> None:
    with pytest.raises(RuntimeError, match="does not match the local tensor"):
        collectives.gather_dp_execution_tokens(
            torch.zeros(1, 8),
            (3, 2),
            rank=1,
        )


def test_dp_execution_gather_rejects_zero_execution_count() -> None:
    with pytest.raises(RuntimeError, match="must be positive"):
        collectives.gather_dp_execution_tokens(
            torch.zeros(1, 8),
            (3, 0),
            rank=1,
        )


def test_tcp_store_master_is_global_rank_zero_not_group_rank_zero(monkeypatch):
    _, tcp_store, _, _ = _mock_process_groups(monkeypatch, global_rank=2)

    collectives.init_process_group("tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 4, 1)

    assert tcp_store.call_args.args[:4] == ("127.0.0.1", 46001, 4, False)


def test_symmetric_memory_rejects_cross_host_group(monkeypatch: pytest.MonkeyPatch) -> None:
    topology = [
        {"hostname": "node-0", "device_index": 0},
        {"hostname": "node-1", "device_index": 0},
    ]
    can_access_peer = MagicMock(return_value=True)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", can_access_peer)

    assert not cuda_collectives._supports_symmetric_memory(torch.device("cuda:0"), [0, 1], topology)
    can_access_peer.assert_not_called()


def test_symmetric_memory_rejects_incomplete_peer_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    topology = [
        {"hostname": "node-0", "device_index": 0},
        {"hostname": "node-0", "device_index": 1},
    ]
    monkeypatch.setattr(
        torch.cuda,
        "can_device_access_peer",
        lambda source, destination: (source, destination) != (1, 0),
    )

    assert not cuda_collectives._supports_symmetric_memory(torch.device("cuda:0"), [0, 1], topology)


def _symmetric_tensor(dtype: torch.dtype = torch.float32, numel: int = 8) -> MagicMock:
    tensor = MagicMock()
    tensor.device = torch.device("cuda:0")
    tensor.dtype = dtype
    tensor.is_contiguous.return_value = True
    tensor.numel.return_value = numel
    tensor.element_size.return_value = torch.empty((), dtype=dtype).element_size()
    return tensor


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_symmetric_buffer_rejects_unsupported_dtype(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype) -> None:
    group = _FakeGroup(0, 2)
    tensor = _symmetric_tensor(dtype)
    cuda_collectives._symm_eligible[(group, str(tensor.device))] = True
    empty = MagicMock()
    rendezvous = MagicMock()
    monkeypatch.setattr(cuda_collectives.symm_mem, "empty", empty)
    monkeypatch.setattr(cuda_collectives.symm_mem, "rendezvous", rendezvous)

    assert cuda_collectives._symm_buffer(group, tensor) is None
    empty.assert_not_called()
    rendezvous.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_symmetric_buffer_accepts_supported_dtype(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype) -> None:
    tensor = _symmetric_tensor(dtype)
    buffer = object()
    group = _FakeGroup(0, 2)
    group.group_name = "tp-group"
    cuda_collectives._symm_eligible[(group, str(tensor.device))] = True
    empty = MagicMock(return_value=buffer)
    rendezvous = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(cuda_collectives.symm_mem, "empty", empty)
    monkeypatch.setattr(cuda_collectives.symm_mem, "rendezvous", rendezvous)

    assert cuda_collectives._symm_buffer(group, tensor) is buffer
    empty.assert_called_once_with(8, dtype=dtype, device=tensor.device)
    rendezvous.assert_called_once_with(buffer, "tp-group")

    # The cache is keyed by element count, not shape, and remains usable during
    # capture and after the allocation cap is reached.
    tensor.shape = (2, 4)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(cuda_collectives, "_SYMM_MEM_MAX_BUFFERS", 1)
    assert cuda_collectives._symm_buffer(group, tensor) is buffer
    empty.assert_called_once()


@pytest.mark.parametrize("reason", ["ineligible", "noncontiguous", "too_large", "capturing", "cache_full"])
def test_symmetric_buffer_rejects_ineligible_allocation(monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    group = _FakeGroup(0, 2)
    tensor = _symmetric_tensor()
    cuda_collectives._symm_eligible[(group, str(tensor.device))] = reason != "ineligible"
    tensor.is_contiguous.return_value = reason != "noncontiguous"
    if reason == "too_large":
        tensor.numel.return_value = cuda_collectives._SYMM_MEM_MAX_BYTES // tensor.element_size() + 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: reason == "capturing")
    if reason == "cache_full":
        monkeypatch.setattr(cuda_collectives, "_SYMM_MEM_MAX_BUFFERS", 0)
    empty = MagicMock()
    monkeypatch.setattr(cuda_collectives.symm_mem, "empty", empty)

    assert cuda_collectives._symm_buffer(group, tensor) is None
    empty.assert_not_called()


def test_attn_dp_reuses_default_world_without_an_extra_communicator(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=0)
    world = _FakeGroup(0, 4)
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=world))

    collectives.init_process_group("dp", "127.0.0.1", 46004, 0, 2, "cuda:0", 0, 4, 0)
    before = new_group.call_count
    group = collectives.init_process_group("attn_dp", "127.0.0.1", 46004, 0, 4, "cuda:0", 0, 4, 0)

    assert group is world
    assert collectives._group_ranks[("attn_dp", "cuda:0")] == (0, 1, 2, 3)
    assert new_group.call_count == before
    init_world.assert_called_once()


def test_attn_dp_rejects_a_partial_world_before_rendezvous() -> None:
    with (
        patch.object(collectives, "_ensure_world", side_effect=AssertionError("invalid rendezvous")),
        pytest.raises(ValueError, match="full process world"),
    ):
        collectives.init_process_group("attn_dp", "127.0.0.1", 46004, 0, 2, "cuda:0", 0, 4, 0)


@pytest.mark.parametrize(
    "dp_size, shard_size, world, expected",
    [
        (4, 2, 8, [[0, 2], [4, 6], [1, 3], [5, 7]]),
        (4, 4, 8, [[0, 2, 4, 6], [1, 3, 5, 7]]),
        (8, 4, 8, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (2, 2, 8, [[0, 4], [1, 5], [2, 6], [3, 7]]),
        (4, 2, 16, [[0, 4], [8, 12], [1, 5], [9, 13], [2, 6], [10, 14], [3, 7], [11, 15]]),
    ],
)
def test_engram_dp_memberships_partition_dp_cohorts(
    dp_size: int, shard_size: int, world: int, expected: list[list[int]]
) -> None:
    assert collectives._engram_dp_memberships(dp_size, shard_size, world) == expected


@pytest.mark.parametrize(
    "dp_size, shard_size, world",
    [
        (4, 2, 8),
        (8, 4, 8),
        (2, 1, 8),
        (4, 4, 16),
        (8, 2, 16),
        (6, 3, 12),
    ],
)
def test_engram_dp_memberships_are_increasing_shards_of_one_dp_cohort(
    dp_size: int, shard_size: int, world: int
) -> None:
    memberships = collectives._engram_dp_memberships(dp_size, shard_size, world)
    dp_cohorts = collectives._group_memberships("dp", dp_size, world)

    assert len(memberships) * shard_size == world
    assert sorted(rank for group in memberships for rank in group) == list(range(world))
    assert all(len(group) == shard_size for group in memberships)
    assert all(group == sorted(group) for group in memberships)
    for group in memberships:
        assert sum(set(group) <= set(cohort) for cohort in dp_cohorts) == 1


@pytest.mark.parametrize("dp_size, shard_size, world", [(4, 3, 8), (4, 0, 8), (4, -1, 8), (6, 4, 12)])
def test_engram_dp_memberships_reject_bad_shard_size(dp_size: int, shard_size: int, world: int) -> None:
    with pytest.raises(ValueError, match="must divide DP size"):
        collectives._engram_dp_memberships(dp_size, shard_size, world)


@pytest.mark.parametrize("dp_size, world", [(3, 8), (5, 12)])
def test_engram_dp_memberships_reject_dp_that_does_not_tile_the_world(dp_size: int, world: int) -> None:
    with pytest.raises(ValueError, match="does not divide the world size"):
        collectives._engram_dp_memberships(dp_size, 1, world)


@pytest.mark.parametrize(
    "world_size, dp_size, replica_size, expected_shard_size",
    [
        (8, 4, 2, 2),
        (16, 8, 2, 4),
        (8, 8, 1, 4),
        (16, 4, 4, 2),
    ],
)
def test_engram_dp_memberships_agree_with_the_topology_shard_size(
    world_size: int, dp_size: int, replica_size: int, expected_shard_size: int
) -> None:
    hosts = ["node-0"] * (world_size // 2) + ["node-1"] * (world_size // 2)
    topology = [{"hostname": host} for host in hosts]
    shard_size = collectives.get_engram_dp_shard_size(world_size, dp_size, replica_size, topology)
    assert shard_size == expected_shard_size

    memberships = collectives._engram_dp_memberships(dp_size, shard_size, world_size)
    for group in memberships:
        assert len({topology[rank]["hostname"] for rank in group}) == 1


def _engram_topology(hosts: str | None) -> list[dict[str, object]] | None:
    if hosts is None:
        return None
    return [{"hostname": host} for host in hosts]


@pytest.mark.parametrize(
    "world_size, dp_size, replica_size, hosts, expected",
    [
        (16, 4, 4, None, 1),
        (0, 4, 4, "", 1),
        (16, 0, 4, "a" * 16, 1),
        (16, -4, 4, "a" * 16, 1),
        (16, 4, 0, "a" * 16, 1),
        (16, 4, -2, "a" * 16, 1),
        (16, 4, 4, "a" * 8, 1),
        (6, 3, 2, "aaabbb", 1),
        (8, 4, 2, "aabbccdd", 1),
        (16, 4, 4, "a" * 16, 4),
        (16, 4, 4, "a" * 8 + "b" * 8, 2),
        (12, 6, 2, "a" * 6 + "b" * 6, 3),
        (8, 4, 2, "abababab", 1),
        (8, 4, 2, "aaaaaabb", 1),
    ],
)
def test_get_engram_dp_shard_size_accepts_and_declines(
    world_size: int, dp_size: int, replica_size: int, hosts: str | None, expected: int
) -> None:
    topology = _engram_topology(hosts)
    assert collectives.get_engram_dp_shard_size(world_size, dp_size, replica_size, topology) == expected


@pytest.mark.parametrize("hostname", [None, "", 7, 1.5, ["a"]])
def test_get_engram_dp_shard_size_rejects_nonstring_hostname(hostname: object) -> None:
    topology: list[dict[str, object]] = [{"hostname": "a"} for _ in range(8)]
    topology[3] = {"hostname": hostname}
    assert collectives.get_engram_dp_shard_size(8, 4, 2, topology) == 1


def test_get_engram_dp_shard_size_rejects_missing_hostname() -> None:
    topology: list[dict[str, object]] = [{"hostname": "a"} for _ in range(8)]
    topology[3] = {}
    assert collectives.get_engram_dp_shard_size(8, 4, 2, topology) == 1


def test_engram_group_reuses_framework_with_gloo_override(monkeypatch: pytest.MonkeyPatch) -> None:
    topology = [{"hostname": "node-0" if rank < 4 else "node-1", "device_index": rank} for rank in range(8)]
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=2, topology=topology)
    collectives.init_process_group("tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 8, 1)
    new_group.reset_mock()
    group = collectives.init_engram_dp_group("cuda:0", 4, 2)
    assert group.rank() == 1 and group.size() == 2
    assert collectives._group_ranks[("engram_dp", "cuda:0")] == (0, 2)
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 2], [4, 6], [1, 3], [5, 7]]
    assert all(call.kwargs["backend"] == "gloo" for call in new_group.call_args_list)
    monkeypatch.setattr(dist, "get_backend", lambda _: "gloo")
    assert collectives.init_engram_dp_group("cuda:0", 4, 2) is group


@pytest.mark.parametrize("memberships", [[[0, 1], [1, 3]], [[1, 0], [2, 3]], [[0, True], [2, 3]]])
def test_explicit_group_memberships_fail_before_world_creation(
    monkeypatch: pytest.MonkeyPatch, memberships: list[list[int]]
) -> None:
    _, init_world, _, new_group = _mock_process_groups(monkeypatch, global_rank=0)
    with pytest.raises((ValueError, TypeError)):
        collectives.init_process_group(
            "engram_dp", "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 4, 0, group_ranks=memberships, backend="gloo"
        )
    init_world.assert_not_called()
    new_group.assert_not_called()


def test_engram_collectives_keep_the_callers_group_with_multiple_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, selected = _FakeGroup(0, 2), _FakeGroup(1, 2)
    monkeypatch.setattr(collectives, "_groups", {("engram_dp", "cuda:0"): first, ("engram_dp", "cuda:1"): selected})
    global_rank = MagicMock(return_value=7)
    broadcast, gather, barrier, monitored = (MagicMock() for _ in range(4))
    monkeypatch.setattr(dist, "get_global_rank", global_rank)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(dist, "barrier", barrier)
    monkeypatch.setattr(dist, "monitored_barrier", monitored)
    payload, gathered = ["descriptor"], [None, None]
    collectives.engram_broadcast_object_list(payload, src=1, group=selected)
    collectives.engram_all_gather_object(gathered, None, group=selected)
    collectives.engram_barrier(group=selected)
    timeout = timedelta(seconds=3)
    collectives.engram_barrier(group=selected, timeout=timeout)
    global_rank.assert_called_once_with(selected, 1)
    broadcast.assert_called_once_with(payload, src=7, group=selected)
    gather.assert_called_once_with(gathered, None, group=selected)
    barrier.assert_called_once_with(group=selected)
    monitored.assert_called_once_with(group=selected, timeout=timeout, wait_all_ranks=True)
