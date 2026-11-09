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
from tests.python.test_deepseek_v41_hc import _make_stubbed_layer, _run_layer
from tests.python.test_deepseek_v41_model import _DSV41_TINY_CONFIG
from xllm.python.model_executor.forward_context import (
    forward_context,
)


class _StopForward(Exception):
    pass


class _RecordingForward(nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.calls: list[tuple[Any, ...]] = []

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        self.calls.append((hidden, input_ids, post_norm))
        return hidden * 0.5


class _RecordingMoeStub(_RecordingForward, v41.DeepseekV41MoE):
    pass


class _RecordingW8A8Stub(_RecordingForward, v41.DeepseekV41W8A8MoE):
    pass


def _make_norm_cast_layer(seed: int) -> tuple[Any, dict[str, Any], _RecordingMoeStub]:
    layer, inputs = _make_stubbed_layer("cpu", seed=seed)
    norm = v41.RMSNorm(
        inputs["hidden"].shape[-1],
        _DSV41_TINY_CONFIG["rms_norm_eps"],
        torch.bfloat16,
    )
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.2)
    layer.post_attention_layernorm = norm
    moe = _RecordingMoeStub()
    layer.mlp = moe
    return layer, inputs, moe


@pytest.mark.parametrize("available", [False, True])
def test_rms_norm_cast_requires_vendor_entry_points(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    from xllm.python.kernels_npu import op_api

    native_op = lambda *args: args
    monkeypatch.setattr(v41, "_rms_norm_cast_op", None)
    monkeypatch.setattr(op_api, "has_op_api", lambda name: available)
    monkeypatch.setattr(v41, "kernels", SimpleNamespace(rms_norm_cast_va=native_op))
    monkeypatch.setattr(
        v41, "torch", SimpleNamespace(ops=SimpleNamespace(xllm_ops=SimpleNamespace(rms_norm_cast_va=native_op)))
    )
    assert (v41._get_rms_norm_cast_op() is not None) is available


def test_rms_norm_cast_layer_delegates_the_un_normalised_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer, inputs, moe = _make_norm_cast_layer(seed=91)
    norm = layer.post_attention_layernorm
    original_forward = norm.forward
    norm_calls: list[torch.Tensor] = []

    def recording_forward(x: torch.Tensor) -> torch.Tensor:
        norm_calls.append(x.clone())
        return original_forward(x)

    monkeypatch.setattr(norm, "forward", recording_forward)
    _run_layer(layer, inputs)

    assert len(moe.calls) == 1
    hidden_moe, _, post_norm_moe = moe.calls[0]
    assert post_norm_moe is norm
    assert norm_calls == []
    with torch.inference_mode():
        normed = original_forward(hidden_moe)
    assert not torch.equal(normed, hidden_moe)


def test_rms_norm_cast_layer_delegates_for_the_w8a8_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer, inputs, _ = _make_norm_cast_layer(seed=95)
    moe = _RecordingW8A8Stub()
    layer.mlp = moe
    norm = layer.post_attention_layernorm
    _run_layer(layer, inputs)

    assert len(moe.calls) == 1
    assert moe.calls[0][2] is norm


def _make_w8a8_moe(
    dp_size: int,
    dp_rank: int = 0,
    gate_logits: Any = None,
) -> v41.DeepseekV41W8A8MoE:
    moe = v41.DeepseekV41W8A8MoE.__new__(v41.DeepseekV41W8A8MoE)
    nn.Module.__init__(moe)
    moe.dp_size = dp_size
    moe.dp_rank = dp_rank
    moe.hash_layer = False
    if gate_logits is not None:
        moe.gate = lambda hidden: gate_logits(hidden, None)
    return moe


def _repeat_dp_tokens(tensor: torch.Tensor, dp_size: int, dp_rank: int) -> tuple[torch.Tensor, None]:
    assert 0 <= dp_rank < dp_size
    return tensor.repeat(dp_size, *([1] * (tensor.dim() - 1))), None


@pytest.mark.parametrize("dp_size", [2, 4])
def test_v41_moe_fused_norm_cast_runs_before_the_dp_gather(
    monkeypatch: pytest.MonkeyPatch,
    dp_size: int,
) -> None:
    monkeypatch.setattr(v4, "dp_gather_tokens", _repeat_dp_tokens)

    captured: dict[str, torch.Tensor] = {}

    def capture_gate_input(gate_input: torch.Tensor, gate_input_ids: Any) -> torch.Tensor:
        captured["gate_input"] = gate_input
        raise _StopForward

    op_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []

    def fake_op(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        y = (x.float() * 0.5).to(x.dtype)
        op_calls.append((x.clone(), y.clone(), gamma, epsilon))
        return y, y.float()

    monkeypatch.setattr(v41, "_rms_norm_cast_supported", lambda x, w: True)
    monkeypatch.setattr(v41, "_get_rms_norm_cast_op", lambda: fake_op)
    monkeypatch.setattr(v41, "_rms_norm_cast_active_logged", False)

    norm = v41.RMSNorm(8, 1e-6, torch.bfloat16)
    moe = _make_w8a8_moe(dp_size, gate_logits=capture_gate_input)
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)

    with forward_context(_dp_forward_context([3] * dp_size)), pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe, hidden, post_norm=norm)

    assert len(op_calls) == 1
    op_input, op_y, op_gamma, op_epsilon = op_calls[0]
    assert op_input.shape[0] == 3
    assert torch.equal(op_input, hidden)
    assert torch.equal(op_gamma, norm.weight)
    assert op_epsilon == norm.eps
    gate_input = captured["gate_input"]
    assert gate_input.shape[0] == 3
    assert torch.equal(gate_input, op_y.float())


@pytest.mark.parametrize("dp_size", [1, 2])
def test_v4_moe_without_post_norm_is_unchanged(monkeypatch: pytest.MonkeyPatch, dp_size: int) -> None:
    monkeypatch.setattr(v4, "dp_gather_tokens", _repeat_dp_tokens)
    captured: dict[str, torch.Tensor] = {}

    def capture_gate_input(gate_input: torch.Tensor) -> torch.Tensor:
        captured["gate_input"] = gate_input
        raise _StopForward

    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    moe = SimpleNamespace(
        dp_size=dp_size,
        dp_rank=0,
        hash_layer=False,
        gate=capture_gate_input,
        _GATE_BEFORE_DP=False,
    )
    moe._prepare_moe_gate = v4.DeepseekV4MoE._prepare_moe_gate.__get__(moe)
    with forward_context(_dp_forward_context([3] * dp_size)), pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe, hidden)
    expected = hidden.repeat(dp_size, 1).float()
    torch.testing.assert_close(captured["gate_input"], expected, rtol=0, atol=0)


def test_v4_moe_fused_norm_cast_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, torch.Tensor] = {}

    def capture_gate_input(gate_input: torch.Tensor, gate_input_ids: Any) -> torch.Tensor:
        captured["gate_input"] = gate_input
        raise _StopForward

    norm = v41.RMSNorm(8, 1e-6, torch.bfloat16)
    original_forward = norm.forward
    norm_calls: list[torch.Tensor] = []

    def recording_forward(x: torch.Tensor) -> torch.Tensor:
        norm_calls.append(x.clone())
        return original_forward(x)

    monkeypatch.setattr(norm, "forward", recording_forward)
    monkeypatch.setattr(v41, "_rms_norm_cast_fallback_logged", False)
    monkeypatch.setattr(v41, "_get_rms_norm_cast_op", lambda: None)
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    expected = original_forward(hidden).to(torch.float32)

    moe = _make_w8a8_moe(1, gate_logits=capture_gate_input)
    with pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe, hidden, post_norm=norm)
    assert torch.equal(captured["gate_input"], expected)
    assert len(norm_calls) == 1


def test_v4_moe_fused_norm_cast_is_applied_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, torch.Tensor] = {}

    def capture_gate_input(gate_input: torch.Tensor, gate_input_ids: Any) -> torch.Tensor:
        captured["gate_input"] = gate_input
        raise _StopForward

    norm = v41.RMSNorm(8, 1e-6, torch.bfloat16)
    original_forward = norm.forward
    norm_calls: list[torch.Tensor] = []

    def recording_forward(x: torch.Tensor) -> torch.Tensor:
        norm_calls.append(x)
        return original_forward(x)

    monkeypatch.setattr(norm, "forward", recording_forward)
    op_calls: list[torch.Tensor] = []

    def fake_op(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        op_calls.append(x)
        return x, x.float()

    monkeypatch.setattr(v41, "_rms_norm_cast_supported", lambda x, w: True)
    monkeypatch.setattr(v41, "_get_rms_norm_cast_op", lambda: fake_op)
    monkeypatch.setattr(v41, "_rms_norm_cast_active_logged", False)

    moe = _make_w8a8_moe(1, gate_logits=capture_gate_input)
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    with pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe, hidden, post_norm=norm)

    assert len(op_calls) == 1
    assert norm_calls == []
    assert torch.equal(captured["gate_input"], op_calls[0].float())


def test_pure_torch_moe_applies_post_norm_before_routing() -> None:
    captured: dict[str, torch.Tensor] = {}

    def capture_selection(x: torch.Tensor, input_ids: Any) -> Any:
        captured["x"] = x
        raise _StopForward

    norm = v41.RMSNorm(8, 1e-6, torch.float32)
    hidden = torch.randn(4, 8, dtype=torch.float32)
    moe = SimpleNamespace(cfg=SimpleNamespace(hidden_size=8), _selection_logits=capture_selection)

    with pytest.raises(_StopForward):
        v41.DeepseekV41MoE.forward(moe, hidden, post_norm=norm)
    assert torch.equal(captured["x"], norm(hidden).reshape(-1, 8))

    captured.clear()
    with pytest.raises(_StopForward):
        v41.DeepseekV41MoE.forward(moe, hidden)
    assert torch.equal(captured["x"], hidden.reshape(-1, 8))
