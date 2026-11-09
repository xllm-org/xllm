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

import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence, cast

import pytest
import torch
import xllm.python.kernels as kernels

import xllm.python.distributed as distributed
import xllm.python.models.deepseek_v41 as v41
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    ForwardContext,
    forward_context,
)

_MC2_HIDDEN = 4
_TOPK = 2
_TOKENS = 3
_LOCAL_EXPERTS = 2
_TOTAL_EXPERTS = 4
_MC2_INTER = 3
_DISPATCH_ROWS = 5


def _mc2_hidden(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(_TOKENS, _MC2_HIDDEN, dtype=dtype, device=torch.device("cpu"))


def _context(
    *,
    is_prefill: bool = False,
    is_chunked_prefill: bool = False,
    is_dummy: bool = False,
    token_counts: Sequence[int] | None = None,
    dp_is_decode: Sequence[int] | None = None,
) -> ForwardContext:
    metadata = SimpleNamespace(
        is_prefill=is_prefill,
        is_chunked_prefill=is_chunked_prefill,
        is_dummy=is_dummy,
        dp_execution_token_counts=list(token_counts) if token_counts is not None else [],
        dp_is_decode=list(dp_is_decode) if dp_is_decode is not None else [],
    )
    return ForwardContext(
        attention_backend=cast(Any, None),
        device=torch.device("cpu"),
        metadata=cast(Any, metadata),
        layer_caches=[],
    )


def _capture_context() -> ForwardContext:
    return replace(_context(), acl_graph=AclGraphCaptureContext(stream=None, tasks=[]))


class _Mc2Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.base_calls: list[dict[str, Any]] = []
        self.ep_reduces: list[torch.Tensor] = []
        self.dispatch_rows = _TOKENS


@pytest.fixture
def mc2(monkeypatch: pytest.MonkeyPatch) -> _Mc2Recorder:
    rec = _Mc2Recorder()

    def fake_dispatch(**kwargs: Any) -> tuple:
        rec.calls.append(("dispatch", kwargs))
        rec.dispatch_rows = int(kwargs["x"].shape[0])
        return (
            torch.randn(_DISPATCH_ROWS, _MC2_HIDDEN, dtype=torch.bfloat16),
            None,
            torch.zeros(rec.dispatch_rows * _TOPK, dtype=torch.int32),
            torch.tensor([2, 3], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            None,
        )

    def fake_combine(**kwargs: Any) -> torch.Tensor:
        rec.calls.append(("combine", kwargs))
        rows = rec.dispatch_rows
        return torch.full((rows, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16)

    def fake_dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rec.calls.append(("dynamic_quant", {"x": x}))
        return torch.zeros(x.shape, dtype=torch.int8), torch.ones(x.shape[0], dtype=torch.float32)

    def fake_dequant_swiglu_quant(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        rec.calls.append(("dequant_swiglu_quant", kwargs))
        return (
            torch.zeros(_DISPATCH_ROWS, _MC2_INTER, dtype=torch.int8),
            torch.ones(_DISPATCH_ROWS, dtype=torch.float32),
        )

    def fake_grouped_matmul(**kwargs: Any) -> list[torch.Tensor]:
        rec.calls.append(("grouped_matmul", kwargs))
        rows = kwargs["x"][0].shape[0]
        if kwargs["output_dtype"] == torch.int32:
            return [torch.zeros(rows, 2 * _MC2_INTER, dtype=torch.int32)]
        return [torch.zeros(rows, _MC2_HIDDEN, dtype=torch.bfloat16)]

    def fake_gating(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, None]:
        rec.calls.append(("gating", kwargs))
        rows = int(kwargs["x"].shape[0])
        return (
            torch.full((rows, _TOPK), 0.5, dtype=torch.float32),
            torch.zeros(rows, _TOPK, dtype=torch.int64),
            None,
        )

    def fake_all_gather(tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        rec.calls.append(("all_gather", {"tensor": tensor, **kwargs}))
        world_size = int(kwargs["world_size"])
        return torch.cat([tensor + rank for rank in range(world_size)], dim=0)

    def base_forward(
        self: Any,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_norm: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        rec.base_calls.append({"hidden": hidden, "input_ids": input_ids, "post_norm": post_norm})
        return hidden + 1.0

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_moe_distribute_dispatch_v2=fake_dispatch,
            npu_moe_distribute_combine_v2=fake_combine,
        ),
    )
    monkeypatch.setattr(torch.ops.npu, "npu_grouped_matmul", fake_grouped_matmul, raising=False)
    monkeypatch.setattr(kernels, "dynamic_quant", fake_dynamic_quant, raising=False)
    monkeypatch.setattr(kernels, "dequant_swiglu_quant", fake_dequant_swiglu_quant, raising=False)
    monkeypatch.setattr(kernels, "moe_gating_top_k_hash", fake_gating, raising=False)
    monkeypatch.setattr(distributed, "moe_ep_all_reduce", lambda x: rec.ep_reduces.append(x), raising=False)
    monkeypatch.setattr(distributed, "moe_tp_all_reduce", lambda x: rec.ep_reduces.append(x), raising=False)
    monkeypatch.setattr(distributed, "all_gather", fake_all_gather, raising=False)
    monkeypatch.setattr(v41, "_moe_mc2_ops_available", lambda: True)
    monkeypatch.setattr(v41, "_moe_mc2_device_supported", lambda x: True)
    monkeypatch.setattr(v41.DeepseekV4MoE, "forward", base_forward)
    return rec


def _make_moe(**overrides: Any) -> v41.DeepseekV41W8A8MoE:
    moe = object.__new__(v41.DeepseekV41W8A8MoE)
    torch.nn.Module.__init__(moe)
    moe.cfg = SimpleNamespace(ep_size=2, tp_size=1, n_shared_experts=1, swiglu_limit=10.0)
    moe.hash_layer = False
    moe.w4a8_dynamic = False
    moe.moe_tp_size = 1
    moe.dp_size = 1
    moe.dp_rank = 0
    moe.num_total_experts = _TOTAL_EXPERTS
    moe.num_experts_per_rank = _LOCAL_EXPERTS
    moe.topk = _TOPK
    moe.scoring_func = "sqrtsoftplus"
    moe.routed_scaling = 1.5
    moe.shared_expert_gate_is_loaded = False
    moe.experts_w13 = torch.zeros(_LOCAL_EXPERTS, _MC2_HIDDEN, 2 * _MC2_INTER, dtype=torch.int8)
    moe.experts_w2 = torch.zeros(_LOCAL_EXPERTS, _MC2_INTER, _MC2_HIDDEN, dtype=torch.int8)
    moe.experts_w13_scale = torch.ones(_LOCAL_EXPERTS, 2 * _MC2_INTER, dtype=torch.float32)
    moe.experts_w2_scale = torch.ones(_LOCAL_EXPERTS, _MC2_HIDDEN, dtype=torch.float32)
    moe.gate = torch.nn.Linear(_MC2_HIDDEN, _TOTAL_EXPERTS, bias=False)
    with torch.no_grad():
        moe.gate.weight.zero_()
    moe._mc2_ep_group_info = ("hccl-mc2", 1, 2)
    moe.shared_experts = lambda hidden: torch.full((hidden.shape[0], _MC2_HIDDEN), 0.25, dtype=hidden.dtype)
    moe.enable_fused_mc2 = True
    moe._mc2_capacity = 512
    for key, value in overrides.items():
        setattr(moe, key, value)
    return moe


def _names(rec: _Mc2Recorder) -> list[str]:
    return [name for name, _ in rec.calls]


def test_off_delegates_bitwise(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe(enable_fused_mc2=False)
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)
    assert torch.equal(out, hidden + 1.0)
    assert len(mc2.base_calls) == 1
    assert mc2.calls == []


def test_on_runs_mc2_chain_in_order(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe()
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)

    assert mc2.base_calls == []
    assert _names(mc2) == [
        "gating",
        "dispatch",
        "dynamic_quant",
        "grouped_matmul",
        "dequant_swiglu_quant",
        "grouped_matmul",
        "combine",
    ]

    dispatch = mc2.calls[1][1]
    assert dispatch["group_ep"] == "hccl-mc2"
    assert dispatch["ep_world_size"] == 2
    assert dispatch["ep_rank_id"] == 1
    assert dispatch["moe_expert_num"] == _TOTAL_EXPERTS
    assert dispatch["quant_mode"] == 0
    assert dispatch["global_bs"] == _TOKENS * 2
    assert dispatch["expert_token_nums_type"] == 1
    assert dispatch["group_tp"] == "" and dispatch["tp_world_size"] == 1
    assert dispatch["x"] is hidden
    assert dispatch["expert_ids"].dtype == torch.int32
    assert dispatch["expert_scales"].dtype == torch.float32

    gemm1 = mc2.calls[3][1]
    assert gemm1["output_dtype"] == torch.int32
    assert gemm1["split_item"] == 2 and gemm1["group_list_type"] == 1
    assert gemm1["group_list"].tolist() == [2, 3]
    assert gemm1["weight"][0] is moe.experts_w13

    dequant = mc2.calls[4][1]
    assert dequant["clamp_limit"] == 10.0
    assert dequant["quant_mode"] == 1 and dequant["swiglu_mode"] == 1
    assert dequant["activate_left"] is True
    assert dequant["group_index"].tolist() == [2, 3]
    assert dequant["weight_scale"] is moe.experts_w13_scale

    gemm2 = mc2.calls[5][1]
    assert gemm2["output_dtype"] == torch.bfloat16
    assert gemm2["scale"][0] is moe.experts_w2_scale
    assert gemm2["per_token_scale"] is not None

    combine = mc2.calls[6][1]
    assert combine["group_ep"] == "hccl-mc2"
    assert combine["ep_send_counts"] is not None
    assert combine["tp_send_counts"] is not None
    assert combine["expand_scales"] is None
    assert combine["expert_ids"].dtype == torch.int32
    assert combine["global_bs"] == _TOKENS * 2

    assert mc2.ep_reduces == []
    expected = torch.full((_TOKENS, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16) + 0.25
    assert torch.equal(out, expected)


@pytest.mark.parametrize(
    "flags",
    [
        {"is_prefill": True},
        {"is_chunked_prefill": True},
        {"is_dummy": True},
    ],
)
def test_mc2_only_on_decode(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder, flags: dict[str, bool]) -> None:
    moe = _make_moe()
    hidden = _mc2_hidden()
    with forward_context(_context(**flags)):
        out = moe.forward(hidden)
    assert torch.equal(out, hidden + 1.0)
    assert len(mc2.base_calls) == 1
    assert mc2.calls == []


def test_mc2_needs_a_forward_context(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe()
    hidden = _mc2_hidden()
    out = moe.forward(hidden)
    assert torch.equal(out, hidden + 1.0)
    assert len(mc2.base_calls) == 1
    assert mc2.calls == []


@pytest.mark.parametrize(
    ("override", "expected_reason"),
    [
        ({"hash_layer": True}, "hash"),
        ({"w4a8_dynamic": True}, "W4A8"),
        ({"moe_tp_size": 2}, "moe_tp_size"),
        ({"dp_size": 2}, "DP token counts"),
        (
            {"cfg": SimpleNamespace(ep_size=1, tp_size=1, n_shared_experts=1, swiglu_limit=10.0)},
            "ep_size",
        ),
    ],
)
def test_guard_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    mc2: _Mc2Recorder,
    override: dict[str, Any],
    expected_reason: str,
) -> None:
    moe = _make_moe(**override)
    hidden = _mc2_hidden()
    with forward_context(_context()):
        moe.forward(hidden)
        reason = moe._mc2_guard_reason(hidden)
    assert reason is not None
    assert expected_reason in reason


def test_mc2_dp1_keeps_the_single_replica_chain(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe(dp_size=1)
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)

    assert "all_gather" not in _names(mc2)
    dispatch = next(kwargs for name, kwargs in mc2.calls if name == "dispatch")
    combine = next(kwargs for name, kwargs in mc2.calls if name == "combine")
    assert dispatch["global_bs"] == _TOKENS * 2
    assert dispatch["x_active_mask"] is None
    assert combine["global_bs"] == _TOKENS * 2
    assert combine["x_active_mask"] is None
    assert out.shape == (_TOKENS, _MC2_HIDDEN)


def test_mc2_dp_local_rows_no_gather_no_slice(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe(dp_size=3, dp_rank=1)
    token_counts = [2, 2, 2]
    local_rows = token_counts[moe.dp_rank]
    hidden = torch.randn(local_rows, _MC2_HIDDEN, dtype=torch.bfloat16)

    post_norm_rows: list[int] = []

    def fake_post_norm(hidden: torch.Tensor, post_norm: Any) -> tuple[torch.Tensor, torch.Tensor]:
        post_norm_rows.append(int(hidden.shape[0]))
        return hidden, hidden.to(torch.float32)

    monkeypatch.setattr(moe, "_apply_post_norm", fake_post_norm)

    with forward_context(_context(token_counts=token_counts, dp_is_decode=[1, 1, 1])):
        out = moe.forward(hidden, post_norm=object())

    assert "all_gather" not in _names(mc2)
    assert _names(mc2) == [
        "gating",
        "dispatch",
        "dynamic_quant",
        "grouped_matmul",
        "dequant_swiglu_quant",
        "grouped_matmul",
        "combine",
    ]
    assert post_norm_rows == [local_rows]

    dispatch = next(kwargs for name, kwargs in mc2.calls if name == "dispatch")
    assert dispatch["x"].shape == (local_rows, _MC2_HIDDEN)
    assert dispatch["expert_ids"].shape[0] == local_rows
    assert dispatch["x_active_mask"] is None
    assert dispatch["global_bs"] == 0

    combine = next(kwargs for name, kwargs in mc2.calls if name == "combine")
    assert combine["global_bs"] == 0
    assert combine["x_active_mask"] is None

    assert out.shape == (local_rows, _MC2_HIDDEN)
    expected = torch.full((local_rows, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16) + 0.25
    assert torch.equal(out, expected)
    assert mc2.ep_reduces == []
    assert mc2.base_calls == []


def test_mc2_dp_empty_rank_falls_back_on_all_ranks(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe(dp_size=4, dp_rank=0)
    token_counts = [0, 3, 1, 2]
    hidden = torch.randn(0, _MC2_HIDDEN, dtype=torch.bfloat16)

    with forward_context(_context(is_dummy=True, token_counts=token_counts, dp_is_decode=[1, 1, 1, 1])):
        reason = moe._mc2_guard_reason(hidden)

    assert reason == "an empty DP rank is present"


def test_mc2_dp_empty_rank_decides_like_a_nonempty_peer(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    token_counts = [0, 3, 1, 2]
    layout = dict(token_counts=token_counts, dp_is_decode=[1, 1, 1, 1])

    empty = _make_moe(dp_size=4, dp_rank=0)
    peer = _make_moe(dp_size=4, dp_rank=1)

    with forward_context(_context(is_dummy=True, **layout)):
        empty_reason = empty._mc2_guard_reason(torch.randn(0, _MC2_HIDDEN, dtype=torch.bfloat16))
    with forward_context(_context(is_dummy=False, **layout)):
        peer_reason = peer._mc2_guard_reason(torch.randn(3, _MC2_HIDDEN, dtype=torch.bfloat16))

    assert empty_reason == "an empty DP rank is present"
    assert peer_reason == empty_reason


@pytest.mark.parametrize("dp_size", [1, 2])
def test_mc2_rows_respect_limit(mc2: _Mc2Recorder, dp_size: int) -> None:
    limit = v41.Eplv2CommPolicy.from_geometry(5120, 384, 16, 6, 200).mc2_capacity
    moe = _make_moe(dp_size=dp_size, _mc2_capacity=limit)
    hidden = torch.randn(limit, _MC2_HIDDEN, dtype=torch.bfloat16)
    with forward_context(_context(token_counts=[limit] * dp_size, dp_is_decode=[1] * dp_size)):
        moe.forward(hidden)
    assert mc2.base_calls == []

    hidden = torch.randn(limit + 1, _MC2_HIDDEN, dtype=torch.bfloat16)
    with forward_context(_context(token_counts=[limit + 1] * dp_size, dp_is_decode=[1] * dp_size)):
        moe.forward(hidden)
    assert len(mc2.base_calls) == 1


def test_mc2_guard_declines_on_uneven_dp_rows(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe(dp_size=3, dp_rank=1)
    hidden = torch.randn(2, _MC2_HIDDEN, dtype=torch.bfloat16)

    def layout(counts: list[int]) -> ForwardContext:
        return _context(token_counts=counts, dp_is_decode=[1, 1, 1])

    with forward_context(layout([3, 2, 4])):
        reason = moe._mc2_guard_reason(hidden)
    assert reason == "uneven DP rows (4 vs 2) break global_bs=0"

    with forward_context(layout([2, 2, 2])):
        assert moe._mc2_guard_reason(hidden) is None

    with forward_context(layout([0, 2, 2])):
        assert moe._mc2_guard_reason(hidden) == "an empty DP rank is present"


def test_mc2_dp_empty_rank_still_declines_when_a_peer_is_not_decode(
    monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder
) -> None:
    empty = _make_moe(dp_size=2, dp_rank=0)
    with forward_context(_context(is_dummy=True, token_counts=[0, 2], dp_is_decode=[1, 0])):
        reason = empty._mc2_guard_reason(torch.randn(0, _MC2_HIDDEN, dtype=torch.bfloat16))
    assert reason is not None and "all DP ranks" in reason


def test_mc2_guard_requires_all_dp_ranks_decode(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe(dp_size=2, dp_rank=0)
    hidden = torch.randn(2, _MC2_HIDDEN, dtype=torch.bfloat16)

    with forward_context(_context(token_counts=[2, 2], dp_is_decode=[1, 0])):
        reason = moe._mc2_guard_reason(hidden)
    assert reason is not None and "all DP ranks" in reason

    with forward_context(_context(token_counts=[2, 2], dp_is_decode=[1, 1])):
        assert moe._mc2_guard_reason(hidden) is None


def test_moe_mc2_all_dp_ranks_decode_mirrors_cpp() -> None:
    assert v41._moe_mc2_all_dp_ranks_decode(
        SimpleNamespace(dp_is_decode=[1, 1, 1], dp_execution_token_counts=[1, 1, 1])
    )
    assert not v41._moe_mc2_all_dp_ranks_decode(
        SimpleNamespace(dp_is_decode=[1, 1, 0], dp_execution_token_counts=[1, 1, 0])
    )
    assert v41._moe_mc2_all_dp_ranks_decode(SimpleNamespace(dp_is_decode=[], dp_execution_token_counts=[]))
    assert v41._moe_mc2_all_dp_ranks_decode(SimpleNamespace(dp_is_decode=[], dp_execution_token_counts=[7]))
    assert not v41._moe_mc2_all_dp_ranks_decode(SimpleNamespace(dp_is_decode=[], dp_execution_token_counts=[7, 7]))


_DRAFT_REASON = "DSpark draft MoE (uneven per-rank rows break global_bs=0)"


def test_mc2_draft_flag_declines_on_every_rank(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    busy = _make_moe(dp_size=4, dp_rank=1, _moe_is_dspark_draft=True)
    empty = _make_moe(dp_size=4, dp_rank=0, _moe_is_dspark_draft=True)
    layout = dict(token_counts=[1, 5, 1, 1], dp_is_decode=[1, 1, 1, 1])

    with forward_context(_context(**layout)):
        busy_reason = busy._mc2_guard_reason(torch.randn(5, _MC2_HIDDEN, dtype=torch.bfloat16))
    with forward_context(_context(**layout)):
        empty_reason = empty._mc2_guard_reason(torch.randn(1, _MC2_HIDDEN, dtype=torch.bfloat16))

    assert busy_reason == _DRAFT_REASON
    assert empty_reason == _DRAFT_REASON

    assert busy._mc2_guard_reason(torch.randn(5, _MC2_HIDDEN, dtype=torch.bfloat16)) == _DRAFT_REASON


def test_mc2_target_without_flag_keeps_original_guard(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    target = _make_moe()
    assert getattr(target, "_moe_is_dspark_draft", False) is False
    hidden = _mc2_hidden()

    with forward_context(_context()):
        assert target._mc2_guard_reason(hidden) is None
        out = target.forward(hidden)

    assert mc2.base_calls == []
    assert _names(mc2)[0] == "gating"
    assert torch.equal(out, torch.full((_TOKENS, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16) + 0.25)

    hashed = _make_moe(hash_layer=True)
    with forward_context(_context()):
        reason = hashed._mc2_guard_reason(hidden)
    assert reason is not None and "hash" in reason
    assert _DRAFT_REASON not in reason


@pytest.mark.parametrize("capture", [False, True])
def test_mc2_does_not_copy_device_counts_to_host(
    monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder, capture: bool
) -> None:
    moe = _make_moe()
    hidden = _mc2_hidden()

    def forbid_host_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("MC2 decode synchronizes device counts with the host")

    for method in ("__int__", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, method, forbid_host_read)
    with forward_context(_capture_context() if capture else _context()):
        out = moe.forward(hidden)
    assert _names(mc2)[-1] == "combine"
    assert out.shape == (_TOKENS, _MC2_HIDDEN)


def test_mc2_uses_dedicated_communicator(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    def shared_group(_device: torch.device) -> tuple[str, int, int]:
        raise AssertionError("MC2 must not query the shared EP communicator")

    monkeypatch.setattr(distributed, "moe_ep_hccl_info", shared_group)
    moe = _make_moe(_mc2_ep_group_info=("hccl-dedicated", 1, 2))
    hidden = _mc2_hidden()
    with forward_context(_context()):
        moe.forward(hidden)
        moe.forward(hidden)
    calls = [args for name, args in mc2.calls if name in ("dispatch", "combine")]
    assert len(calls) == 4
    assert all(args["group_ep"] == "hccl-dedicated" and args["ep_rank_id"] == 1 for args in calls)


def test_mc2_missing_communicator_keeps_base_forward(mc2: _Mc2Recorder) -> None:
    moe = _make_moe(_mc2_ep_group_info=None)
    hidden = _mc2_hidden()
    with forward_context(_context()):
        actual = moe.forward(hidden)
    assert torch.equal(actual, hidden + 1)
    assert len(mc2.base_calls) == 1
    assert not mc2.calls


def test_mc2_reuses_common_linear_gate(mc2: _Mc2Recorder) -> None:
    moe = _make_moe()
    hidden = _mc2_hidden()
    observed = []
    handle = moe.gate.register_forward_pre_hook(lambda module, args: observed.append(args[0]))
    try:
        with forward_context(_context()):
            moe.forward(hidden)
    finally:
        handle.remove()
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], hidden.float(), rtol=0, atol=0)
