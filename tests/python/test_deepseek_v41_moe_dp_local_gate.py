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
import torch.nn as nn

import xllm.python.models.deepseek_v4 as v4
import xllm.python.models.deepseek_v41 as v41
from tests.python._deepseek_v41_fused_testkit import _dp_forward_context
from xllm.python import kernels
from xllm.python.layers.moe_parallel import DpScatterState
from xllm.python.model_executor.forward_context import (
    forward_context,
)


class _StopForward(Exception):
    pass


class _FixedGate(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.inputs: list[torch.Tensor] = []

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        assert gate_input.dtype == torch.float32
        self.inputs.append(gate_input.clone())
        return torch.nn.functional.linear(gate_input, self.weight)


_GATE_WEIGHT = torch.tensor([[0.25, -0.5], [0.75, 0.125]], dtype=torch.float32)
_RAW_ROWS = torch.tensor([[0.5, -0.25], [1.25, 0.5], [-0.5, 1.5]])


def _make_moe(moe_class: type, dp_size: int, dp_rank: int, gate: nn.Module) -> Any:
    moe = moe_class.__new__(moe_class)
    nn.Module.__init__(moe)
    moe.dp_size = dp_size
    moe.dp_rank = dp_rank
    moe.gate = gate
    moe.num_total_experts = _GATE_WEIGHT.shape[0]
    moe.hash_layer = False
    moe.scoring_func = "sigmoid"
    moe.topk = 1
    moe.routed_scaling = 1.0
    moe.cfg = v41.DeepseekV41Config(ep_size=1, tp_size=1, swiglu_limit=0.0)
    moe.moe_tp_size = 1
    moe.start_expert_id = 0
    moe.num_experts_per_rank = moe.num_total_experts
    moe.w4a8_dynamic = False
    moe.shared_expert_gate_is_loaded = False
    moe.e_score_correction_bias = torch.zeros(moe.num_total_experts, dtype=torch.float32)
    for name in (
        "experts_w13",
        "experts_w2",
        "experts_w13_scale",
        "experts_w2_scale",
        "experts_w13_offset",
        "experts_w2_offset",
    ):
        setattr(moe, name, None)
    return moe


def _local_rows(rank: int, count: int, dtype: torch.dtype) -> torch.Tensor:
    return _RAW_ROWS[:count].to(dtype) + rank + 2**-12


def _make_fake_gather(
    counts: list[int],
    per_call_expected: list[list[torch.Tensor]],
    dp_rank: int,
) -> tuple[Any, list[str]]:
    calls: list[str] = []

    def fake(tensor: torch.Tensor, dp_size: int, rank: int) -> tuple[torch.Tensor, DpScatterState]:
        assert dp_size == len(counts) and rank == dp_rank
        call = 0 if not calls else 1
        expected_local = per_call_expected[call][dp_rank]
        torch.testing.assert_close(tensor, expected_local, rtol=0, atol=0)
        calls.append("hidden" if call == 0 else "logits")
        assert all(count > 0 for count in counts)
        gathered = torch.cat(per_call_expected[call], dim=0)
        return gathered, DpScatterState(sum(counts[:dp_rank]), counts[dp_rank], True)

    return fake, calls


@pytest.mark.parametrize("token_counts", [(3, 1, 1, 2), (1, 1, 1, 1)])
@pytest.mark.parametrize("dp_rank", [0, 2, 3])
@pytest.mark.parametrize("with_norm", [False, True])
def test_w8a8_gates_local_rows_before_dp_gather(
    monkeypatch: pytest.MonkeyPatch,
    token_counts: tuple[int, ...],
    dp_rank: int,
    with_norm: bool,
) -> None:
    dtype = torch.bfloat16
    dp_size = len(token_counts)
    count = token_counts[dp_rank]
    gate = _FixedGate(_GATE_WEIGHT)
    moe = _make_moe(v41.DeepseekV41W8A8MoE, dp_size, dp_rank, gate)

    raw_inputs = [_local_rows(r, c, dtype) for r, c in enumerate(token_counts)]
    norm_inputs: list[torch.Tensor] = []
    norm_calls: list[torch.Tensor] = []

    def apply_norm(hidden: torch.Tensor, post_norm: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        norm_calls.append(hidden.clone())
        normalized = hidden * 0.5
        norm_inputs.append(normalized)
        return normalized, normalized.to(torch.float32)

    normalized = [h * 0.5 for h in raw_inputs] if with_norm else raw_inputs
    local_logits = [torch.nn.functional.linear(h.to(torch.float32), _GATE_WEIGHT) for h in normalized]
    fake_gather, gather_calls = _make_fake_gather(
        list(token_counts),
        [normalized, local_logits],
        dp_rank,
    )

    captured: dict[str, Any] = {}

    def fake_top_k(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured.update(kwargs)
        raise _StopForward

    monkeypatch.setattr(v4, "dp_gather_tokens", fake_gather)
    monkeypatch.setattr(moe, "_apply_post_norm", apply_norm)
    monkeypatch.setattr(kernels, "moe_gating_top_k_hash", fake_top_k)

    with forward_context(_dp_forward_context(list(token_counts))), pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe, raw_inputs[dp_rank], None, object() if with_norm else None)

    assert gather_calls == ["hidden", "logits"]
    assert len(gate.inputs) == int(count > 0)
    if count > 0:
        torch.testing.assert_close(gate.inputs[0], normalized[dp_rank].to(torch.float32), rtol=0, atol=0)
    assert len(norm_calls) == int(with_norm and count > 0)
    if with_norm and count > 0:
        torch.testing.assert_close(norm_calls[0], raw_inputs[dp_rank], rtol=0, atol=0)
    gathered_logits = torch.cat(local_logits, dim=0)
    torch.testing.assert_close(captured["x"], gathered_logits, rtol=0, atol=0)
    torch.testing.assert_close(captured["bias"], moe.e_score_correction_bias, rtol=0, atol=0)
    assert captured["input_ids"] is None and captured["tid2eid"] is None


def test_w8a8_zero_rows_skip_norm_and_gate_without_dp() -> None:
    gate = _FixedGate(_GATE_WEIGHT)
    moe = _make_moe(v41.DeepseekV41W8A8MoE, 1, 0, gate)
    hidden = torch.empty((0, _GATE_WEIGHT.size(1)), dtype=torch.bfloat16)
    normalized, logits = moe._prepare_moe_gate(hidden, None)
    assert normalized is hidden
    assert logits.shape == (0, moe.num_total_experts)
    assert logits.dtype == torch.float32
    assert gate.inputs == []


@pytest.mark.parametrize("dp_rank", [0, 1, 2, 3])
@pytest.mark.parametrize("with_norm", [False, True])
def test_w8a8_full_forward_routes_gathered_local_logits(
    monkeypatch: pytest.MonkeyPatch,
    dp_rank: int,
    with_norm: bool,
) -> None:
    dtype = torch.bfloat16
    token_counts = (3, 1, 1, 2)
    dp_size = len(token_counts)
    count = token_counts[dp_rank]
    gate = _FixedGate(_GATE_WEIGHT)
    moe = _make_moe(v41.DeepseekV41W8A8MoE, dp_size, dp_rank, gate)
    moe.shared_experts = lambda hidden: hidden * 0.5  # type: ignore[assignment]

    reduce_calls: list[tuple[torch.dtype, torch.dtype]] = []

    def fake_reduce(routed_out: torch.Tensor, shared_out: torch.Tensor) -> torch.Tensor:
        reduce_calls.append((routed_out.dtype, shared_out.dtype))
        return routed_out + shared_out

    def select_experts(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights, indices = torch.sigmoid(kwargs["x"]).topk(kwargs["k"], dim=-1)
        return weights, indices, torch.empty(0)

    def run_experts(hidden: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, *args: Any) -> torch.Tensor:
        return hidden * weights.to(hidden.dtype)

    monkeypatch.setattr(moe, "_reduce_moe_outputs", fake_reduce)
    monkeypatch.setattr(kernels, "moe_gating_top_k_hash", select_experts)
    monkeypatch.setattr(kernels, "grouped_moe_with_selected_experts", run_experts)

    for step in range(2):
        raw_inputs = [_local_rows(r, c, dtype) + (0.25 * step if step else 0.0) for r, c in enumerate(token_counts)]
        raw_inputs[2].zero_()
        normalized = [h * 0.5 for h in raw_inputs] if with_norm else raw_inputs
        if with_norm:
            monkeypatch.setattr(
                moe,
                "_apply_post_norm",
                lambda hidden, post_norm: (hidden * 0.5, (hidden * 0.5).to(torch.float32)),
            )
        local_logits = [torch.nn.functional.linear(h.to(torch.float32), _GATE_WEIGHT) for h in normalized]
        fake_gather, gather_calls = _make_fake_gather(list(token_counts), [normalized, local_logits], dp_rank)
        monkeypatch.setattr(v4, "dp_gather_tokens", fake_gather)
        gate.inputs.clear()
        reduce_calls.clear()

        with forward_context(_dp_forward_context(list(token_counts))):
            output = v4.DeepseekV4MoE.forward(moe, raw_inputs[dp_rank], None, object() if with_norm else None)

        assert gather_calls == ["hidden", "logits"]
        assert len(gate.inputs) == int(count > 0)
        gathered_hidden = torch.cat(normalized, dim=0)
        gathered_logits = torch.cat(local_logits, dim=0)
        expected_weights = torch.sigmoid(gathered_logits).topk(1, dim=-1).values.to(dtype)
        expected = gathered_hidden * expected_weights + gathered_hidden * 0.5
        start = sum(token_counts[:dp_rank])
        torch.testing.assert_close(output, expected[start : start + count], rtol=0, atol=0)
        assert output.shape == (count, raw_inputs[0].shape[1])
        assert output.dtype == dtype
        assert reduce_calls == [(dtype, dtype)]


@pytest.mark.parametrize("with_norm", [False, True])
@pytest.mark.parametrize("dp_rank", [0, 1])
def test_v4_base_gathers_before_norm_and_gate(
    monkeypatch: pytest.MonkeyPatch,
    with_norm: bool,
    dp_rank: int,
) -> None:
    dtype = torch.bfloat16
    token_counts = (3, 1)
    dp_size = len(token_counts)
    gate = _FixedGate(_GATE_WEIGHT)
    moe = _make_moe(v4.DeepseekV4MoE, dp_size, dp_rank, gate)

    raw_inputs = [_local_rows(r, c, dtype) for r, c in enumerate(token_counts)]
    norm_calls: list[torch.Tensor] = []
    fake_gather, gather_calls = _make_fake_gather(list(token_counts), [raw_inputs], dp_rank)

    def apply_norm(hidden: torch.Tensor, post_norm: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        norm_calls.append(hidden.clone())
        normalized = hidden * 0.5
        return normalized, normalized.to(torch.float32)

    captured: dict[str, Any] = {}

    def fake_top_k(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured.update(kwargs)
        raise _StopForward

    monkeypatch.setattr(v4, "dp_gather_tokens", fake_gather)
    monkeypatch.setattr(moe, "_apply_post_norm", apply_norm)
    monkeypatch.setattr(kernels, "moe_gating_top_k_hash", fake_top_k)

    with forward_context(_dp_forward_context(list(token_counts))), pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe, raw_inputs[dp_rank], None, object() if with_norm else None)

    gathered_raw = torch.cat(raw_inputs, dim=0)
    assert gather_calls == ["hidden"]
    assert len(norm_calls) == int(with_norm)
    if with_norm:
        torch.testing.assert_close(norm_calls[0], gathered_raw, rtol=0, atol=0)
    assert len(gate.inputs) == 1
    expected_gate_input = (gathered_raw * 0.5).to(torch.float32) if with_norm else gathered_raw.to(torch.float32)
    torch.testing.assert_close(gate.inputs[0], expected_gate_input, rtol=0, atol=0)
    torch.testing.assert_close(
        captured["x"], torch.nn.functional.linear(expected_gate_input, _GATE_WEIGHT), rtol=0, atol=0
    )
    assert captured["x"].shape[0] == sum(token_counts)


def test_v4_base_dp1_runs_without_any_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    dtype = torch.bfloat16
    gate = _FixedGate(_GATE_WEIGHT)
    moe = _make_moe(v4.DeepseekV4MoE, 1, 0, gate)
    hidden = _local_rows(0, 3, dtype)

    def fake_top_k(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise _StopForward

    monkeypatch.setattr(kernels, "moe_gating_top_k_hash", fake_top_k)
    with forward_context(_dp_forward_context([3])), pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe, hidden, None, None)

    assert len(gate.inputs) == 1
    torch.testing.assert_close(gate.inputs[0], hidden.to(torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("moe_class", [v4.DeepseekV4MoE, v41.DeepseekV41W8A8MoE])
@pytest.mark.parametrize("ep_size,moe_tp_size", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_shared_reduction_order_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
    moe_class: type[nn.Module],
    ep_size: int,
    moe_tp_size: int,
    dtype: torch.dtype,
) -> None:
    owner = moe_class.__new__(moe_class)
    nn.Module.__init__(owner)
    owner.cfg = v41.DeepseekV41Config(ep_size=ep_size, tp_size=1, swiglu_limit=0.0)
    owner.moe_tp_size = moe_tp_size
    calls = []

    def tp_reduce(tensor: torch.Tensor) -> None:
        calls.append(("tp", tensor.dtype))
        tensor.add_(3)

    def ep_reduce(tensor: torch.Tensor) -> None:
        calls.append(("ep", tensor.dtype))
        tensor.add_(2)

    monkeypatch.setattr(v4, "distributed", SimpleNamespace(moe_tp_all_reduce=tp_reduce, moe_ep_all_reduce=ep_reduce))
    routed = torch.full((2, 3), 0.5, dtype=dtype)
    shared = torch.full_like(routed, 0.25)
    actual = owner._reduce_moe_outputs(routed, shared)
    expected_calls = []
    value = 0.75
    if ep_size > 1:
        if moe_tp_size > 1:
            expected_calls.append(("tp", dtype))
            value += 3
        expected_calls.append(("ep", dtype))
        value += 2
        if moe_tp_size > 1:
            expected_calls.append(("tp", dtype))
            value += 3
    elif moe_tp_size > 1:
        expected_calls.append(("tp", dtype))
        value += 3
    assert calls == expected_calls
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, torch.full_like(actual, value), rtol=0, atol=0)
