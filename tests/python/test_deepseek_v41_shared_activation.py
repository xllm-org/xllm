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
from typing import Any, cast

import pytest
import torch

import xllm.python.models.deepseek_v41 as v41
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.models.deepseek_v32 import _swiglu_with_clamp
from xllm.python.models.deepseek_v41 import DeepseekV41SharedMLP

_LIMIT = 10.0
_HIDDEN = 8
_INTER = 8
_ACT_EQUIV_TOL = 2**-6
_ON_OFF_PEAK_RATIO_MAX = 2.0 / 127.0


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = x.float()
    scale = (value.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(2.0**-24)
    codes = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return scale, codes


class _CpuW8A8Kernels:
    def __init__(self) -> None:
        self.dequant_calls: list[dict[str, Any]] = []
        self.quant_matmul_dtypes: list[torch.dtype] = []
        self.activations: list[torch.Tensor] = []

    def dynamic_quant(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, codes = _quantize_rows(x)
        return codes, scale

    def quant_matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        transpose: bool,
        weight_scale: torch.Tensor,
        weight_offset: torch.Tensor | None = None,
        pertoken: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        self.quant_matmul_dtypes.append(out_dtype)
        weight_f = weight.float().t() if transpose else weight.float()
        acc = x.float() @ weight_f
        if out_dtype is torch.int32:
            return acc.to(torch.int32)
        out = acc * weight_scale.float().flatten().unsqueeze(0)
        if pertoken is not None:
            out = out * pertoken.float().reshape(-1, 1)
        if bias is not None:
            out = out + bias.float()
        return out.to(out_dtype)

    def dequant_swiglu_quant(
        self,
        x: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        quant_scale: torch.Tensor | None = None,
        quant_offset: torch.Tensor | None = None,
        group_index: torch.Tensor | None = None,
        activate_left: bool = True,
        quant_mode: int = 1,
        swiglu_mode: int = 1,
        clamp_limit: float = 0.0,
        glu_alpha: float = 1.0,
        glu_bias: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del bias, quant_scale, quant_offset, glu_alpha, glu_bias
        self.dequant_calls.append(
            dict(
                clamp_limit=clamp_limit,
                swiglu_mode=swiglu_mode,
                quant_mode=quant_mode,
                activate_left=activate_left,
                group_index=group_index,
                weight_scale_shape=tuple(weight_scale.shape),
                shape=tuple(x.shape),
            )
        )
        deq = x.float() * weight_scale.float().flatten().unsqueeze(0)
        deq = deq * activation_scale.float().reshape(-1, 1)
        gate, up = deq.chunk(2, dim=-1)
        effective_limit = clamp_limit if group_index is not None else 0.0
        if 0.0 < effective_limit < 1_000_000.0:
            gate = gate.clamp_max(effective_limit)
            up = up.clamp(-effective_limit, effective_limit)
        act = torch.nn.functional.silu(gate) * up
        self.activations.append(act.detach().clone())
        scale, codes = _quantize_rows(act)
        return codes, scale

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)


@pytest.fixture
def w8a8(monkeypatch: pytest.MonkeyPatch) -> _CpuW8A8Kernels:
    from xllm.python import kernels as kernels_stub

    stand_in = _CpuW8A8Kernels()
    for name in ("dynamic_quant", "quant_matmul", "dequant_swiglu_quant", "silu_and_mul"):
        monkeypatch.setattr(kernels_stub, name, getattr(stand_in, name), raising=False)
    monkeypatch.setattr(v41, "_fused_shared_act_device_supported", lambda x: True)
    monkeypatch.setattr(v41, "_fused_shared_act_active_logged", False)
    monkeypatch.setattr(v41, "_fused_shared_act_fallback_logged", False)
    return stand_in


def _make_mlp(limit: float = _LIMIT) -> DeepseekV41SharedMLP:
    mlp = DeepseekV41SharedMLP(
        SimpleNamespace(tp_size=1, hidden_size=_HIDDEN),
        _INTER,
        torch.bfloat16,
        "cpu",
        skip_tp_reduce=True,
        swiglu_limit=limit,
    )
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(
            torch.randint(-8, 9, mlp.gate_up_proj.weight.shape, generator=generator, dtype=torch.int8)
        )
        mlp.down_proj.weight.copy_(
            torch.randint(-8, 9, mlp.down_proj.weight.shape, generator=generator, dtype=torch.int8)
        )
        mlp.gate_up_proj.weight_scale.copy_(torch.full_like(mlp.gate_up_proj.weight_scale, 0.125))
        mlp.down_proj.weight_scale.copy_(torch.full_like(mlp.down_proj.weight_scale, 0.125))
    return mlp


def _hidden(rows: int = 4, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return (torch.randn(rows, _HIDDEN, generator=generator) * scale).to(torch.bfloat16)


def _eager_activation(mlp: DeepseekV41SharedMLP, hidden: torch.Tensor) -> torch.Tensor:
    gate_up = mlp.gate_up_proj(hidden)
    return _swiglu_with_clamp(gate_up, mlp.swiglu_limit).float()


def _boundary_hidden() -> torch.Tensor:
    return torch.tensor(
        [
            [6.0, -6.0, 2.0, -2.0, 0.5, -0.5, 0.0, 12.0],
            [-9.0, 9.0, 4.0, -4.0, 1.5, -1.5, 0.25, -12.0],
        ],
        dtype=torch.bfloat16,
    )


def _graph_context(state: AclGraphExecutionState | None) -> ForwardContext:
    return ForwardContext(
        attention_backend=cast(Any, None),
        device=torch.device("cpu"),
        metadata=cast(Any, None),
        layer_caches=[],
        execution_state=state,
    )


def test_on_uses_the_fused_op_with_the_model_clamp_limit(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    mlp = _make_mlp()

    out = mlp(_hidden())

    assert len(w8a8.dequant_calls) == 1
    call = w8a8.dequant_calls[0]
    assert call["clamp_limit"] == _LIMIT
    assert call["swiglu_mode"] == 1
    assert call["quant_mode"] == 1
    assert call["activate_left"] is True
    grouped = call["group_index"]
    assert grouped is not None
    assert grouped.dtype == torch.int64
    assert grouped.shape == (1,)
    assert int(grouped[0]) == call["shape"][0] == 4
    assert call["weight_scale_shape"] == (1, 2 * _INTER)
    assert w8a8.quant_matmul_dtypes == [torch.int32, torch.bfloat16]
    assert out.dtype == torch.bfloat16 and out.shape == (4, _HIDDEN)


def test_grouped_form_is_what_makes_the_clamp_take_effect(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    hidden = _boundary_hidden()
    mlp = _make_mlp()

    mlp(hidden)
    fused = w8a8.activations[0]

    gate_up = mlp.gate_up_proj(hidden)
    gate, up = gate_up.float().chunk(2, dim=-1)
    clamped = torch.nn.functional.silu(gate.clamp_max(_LIMIT)) * up.clamp(-_LIMIT, _LIMIT)
    unclamped = torch.nn.functional.silu(gate) * up
    assert not torch.allclose(clamped, unclamped)

    torch.testing.assert_close(fused, clamped, rtol=_ACT_EQUIV_TOL, atol=_ACT_EQUIV_TOL)
    assert not torch.allclose(fused, unclamped, rtol=_ACT_EQUIV_TOL, atol=_ACT_EQUIV_TOL), (
        "the fused activation matches the UNCLAMPED reference: the clamp was dropped"
    )

    mlp.forward_dequant_swiglu_quant(hidden, clamp_limit=_LIMIT)
    assert len(w8a8.dequant_calls) == 2
    assert w8a8.dequant_calls[1]["group_index"] is None
    torch.testing.assert_close(w8a8.activations[1], unclamped, rtol=_ACT_EQUIV_TOL, atol=_ACT_EQUIV_TOL)


def test_on_parks_group_index_in_the_graph_buffer(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    mlp = _make_mlp()
    state = AclGraphExecutionState({})

    with forward_context(_graph_context(state)):
        mlp(_hidden(rows=4))
        mlp(_hidden(rows=4))

    first, second = (call["group_index"] for call in w8a8.dequant_calls)
    assert first is second
    assert first.dtype == torch.int64 and first.shape == (1,) and int(first[0]) == 4
    assert ("SHARED_ACT_GROUP_INDEX", first.device, 4) in state.persistent_buffers

    with forward_context(_graph_context(AclGraphExecutionState({}))):
        mlp(_hidden(rows=2))
    other = w8a8.dequant_calls[2]["group_index"]
    assert other is not first
    assert other.dtype == torch.int64 and other.shape == (1,) and int(other[0]) == 2


def test_on_builds_group_index_inline_without_an_execution_state(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    mlp = _make_mlp()

    with forward_context(_graph_context(None)):
        mlp(_hidden(rows=4))
        mlp(_hidden(rows=4))
    mlp(_hidden(rows=4))
    mlp(_hidden(rows=4))

    first, second, third, fourth = (call["group_index"] for call in w8a8.dequant_calls)
    assert first is not second
    assert third is not fourth
    assert len({id(grouped) for grouped in (first, second, third, fourth)}) == 4
    for grouped in (first, second, third, fourth):
        assert grouped.dtype == torch.int64 and grouped.shape == (1,) and int(grouped[0]) == 4


def test_on_matches_off_within_the_derived_int8_bound(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    hidden = _boundary_hidden()
    mlp = _make_mlp()

    real_supported = v41._fused_shared_act_supported
    monkeypatch.setattr(v41, "_fused_shared_act_supported", lambda *a, **k: False)
    off = mlp(hidden)
    monkeypatch.setattr(v41, "_fused_shared_act_supported", real_supported)
    on = mlp(hidden)

    assert len(w8a8.activations) == 1
    fused_activation = w8a8.activations[0]
    eager_activation = _eager_activation(mlp, hidden)
    torch.testing.assert_close(fused_activation, eager_activation, rtol=_ACT_EQUIV_TOL, atol=_ACT_EQUIV_TOL)

    down = mlp.down_proj
    w = down.weight.float().t() * down.weight_scale.float().flatten().unsqueeze(0)
    s_f, q_f = _quantize_rows(fused_activation)
    s_e, q_e = _quantize_rows(eager_activation)
    differing = (q_f != q_e).to(fused_activation.dtype)
    scale_term = (s_f - s_e).abs() * (q_e.float().abs() @ w.abs())
    code_term = torch.maximum(s_f, s_e) * (differing @ w.abs())
    bound = float((scale_term + code_term).max())
    delta = float((on.float() - off.float()).abs().max())
    differing_fraction = float(differing.mean())
    assert delta <= bound, f"ON/OFF max abs delta {delta:.4e} exceeds the tight INT8 bound {bound:.4e}"

    peak_ratio = delta / float(off.float().abs().max())
    assert peak_ratio <= _ON_OFF_PEAK_RATIO_MAX, (
        f"ON/OFF max|Δ| = {delta:.4e} is {peak_ratio:.4e} of the output peak "
        f"({peak_ratio * 127:.2f}/127 vs the {_ON_OFF_PEAK_RATIO_MAX * 127:.2f}/127 gate; "
        f"codes differ on {differing_fraction:.1%} of elements)"
    )

    unclamped = _make_mlp(limit=0.0)(hidden)
    assert not torch.allclose(unclamped.float(), off.float())


def test_on_is_bitwise_off_when_the_activation_is_exactly_representable(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    mlp = _make_mlp()
    with torch.no_grad():
        mlp.gate_up_proj.weight[_INTER:, :].zero_()
    hidden = _hidden()

    real_supported = v41._fused_shared_act_supported
    monkeypatch.setattr(v41, "_fused_shared_act_supported", lambda *a, **k: False)
    off = mlp(hidden)
    monkeypatch.setattr(v41, "_fused_shared_act_supported", real_supported)
    on = mlp(hidden)

    assert torch.equal(on, off)
    assert bool((mlp(hidden) == 0).all())


def test_v4_shared_expert_retains_the_eager_mlp(w8a8: _CpuW8A8Kernels) -> None:
    shared = _make_mlp()
    baseline = v41.DeepseekV4MoE._SHARED_MLP_CLS(
        SimpleNamespace(tp_size=1, hidden_size=_HIDDEN),
        _INTER,
        torch.bfloat16,
        "cpu",
        skip_tp_reduce=True,
        swiglu_limit=_LIMIT,
    )
    baseline.load_state_dict(shared.state_dict())
    hidden = _hidden()
    expected = v41.DeepseekV3MLP.forward(shared, hidden)
    assert torch.equal(baseline(hidden), expected)
    assert w8a8.dequant_calls == []


def test_shared_forward_preserves_tp_reduce_add(w8a8: _CpuW8A8Kernels) -> None:
    mlp = _make_mlp()
    hidden = _hidden()
    residual = torch.full_like(hidden, 0.25)
    expected = mlp(hidden) + residual
    assert torch.equal(mlp(hidden, residual), expected)


def test_shared_fused_forward_reuses_output_storage(monkeypatch: pytest.MonkeyPatch, w8a8: _CpuW8A8Kernels) -> None:
    mlp = _make_mlp()
    hidden = _hidden()
    output = torch.empty_like(hidden)
    original = mlp.down_proj.forward_quantized
    seen = []

    def write_output(codes: torch.Tensor, scales: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seen.append(target)
        target.copy_(original(codes, scales))
        return target

    monkeypatch.setattr(mlp.down_proj, "forward_quantized_out", write_output)
    actual = mlp.forward_dequant_swiglu_quant(hidden, output=output, clamp_limit=_LIMIT, grouped_activation=True)
    assert len(seen) == 1 and seen[0] is output
    assert actual.data_ptr() == output.data_ptr()
    expected = mlp.forward_dequant_swiglu_quant(hidden, clamp_limit=_LIMIT, grouped_activation=True)
    assert torch.equal(actual, expected)
