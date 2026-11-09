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

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from xllm_kernel.platform import current_platform

import xllm.python.models.deepseek_v41 as v41
from tests.python._deepseek_v41_fused_testkit import _MODE_EAGER, _MODE_FUSED, _npu_ready
from tests.python.test_deepseek_v41_model import _DSV41_TINY_CONFIG


def test_va_fused_reuses_native_pre_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = Path(v41.__file__).parents[1] / "kernels_npu" / "hyper_connection.py"
    spec = importlib.util.spec_from_file_location("_hc_va_fused_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layer, inputs = _make_stubbed_layer("cpu", seed=83)
    hc = layer.hc
    hidden = inputs["hidden"]
    expected = (
        torch.randn(hidden.size(0), hidden.size(-1)),
        torch.randn(hidden.size(0), hc.hc_mult),
        torch.randn(hidden.size(0), hc.hc_mult, hc.hc_mult),
        torch.randn(hidden.size(0), hc.hc_mult),
    )
    calls: list[tuple[Any, ...]] = []

    def native_op(*args: Any) -> tuple[torch.Tensor, ...]:
        calls.append(args)
        return expected

    monkeypatch.setattr(
        module, "torch", SimpleNamespace(ops=SimpleNamespace(xllm_ops=SimpleNamespace(hc_pre_va=native_op)))
    )
    monkeypatch.setattr(module, "has_op_api", lambda name: True)
    monkeypatch.setattr(current_platform, "get_ascend_soc_generation", lambda: "a3")
    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", module._resolve_hc_pre_va)
    y, pre, post, comb = hc.hc_pre_fused(hidden, hc.hc_attn_fn, hc.hc_attn_scale, hc.hc_attn_base, inputs["pre_mix"])
    assert y is expected[0] and pre is expected[3]
    assert post is expected[1] and comb is expected[2]
    assert len(calls) == 1 and calls[0][4] is inputs["pre_mix"]


@pytest.mark.parametrize("soc_generation", ["a2", "a3", "a5", None])
@pytest.mark.parametrize("available", [False, True])
def test_hc_va_requires_vendor_entry_points(
    monkeypatch: pytest.MonkeyPatch, available: bool, soc_generation: str | None
) -> None:
    from xllm.python.kernels_npu import hyper_connection

    native_op = lambda *args: args
    monkeypatch.setattr(hyper_connection, "_hc_pre_va_", None)
    monkeypatch.setattr(hyper_connection, "has_op_api", lambda name: available)
    monkeypatch.setattr(current_platform, "get_ascend_soc_generation", lambda: soc_generation)
    monkeypatch.setattr(
        hyper_connection,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(xllm_ops=SimpleNamespace(hc_pre_va=native_op, hc_post_va=native_op))),
    )
    assert (hyper_connection.hc_pre_va is not None) is (available and soc_generation == "a3")
    assert (hyper_connection.hc_post_va is not None) is available


@pytest.mark.skipif(
    not _npu_ready() or getattr(torch.ops.xllm_ops, "hc_pre_va", None) is None,
    reason="needs NPU and the native VA binding",
)
@pytest.mark.parametrize("tokens", [1, 6, 33])
def test_va_gates_match_fp32_reference(tokens: int) -> None:
    torch.manual_seed(83)
    width, hc = 5120, 4
    hidden = torch.randn(tokens, hc, width, dtype=torch.bfloat16)
    weight = torch.randn(24, hc * width) / (hc * width) ** 0.5
    scale = torch.tensor([1.0, 1.0, 1.0])
    base = torch.randn(24) * 0.1
    external_pre = torch.rand(tokens, hc)
    flattened = hidden.float().flatten(-2)
    projection = F.linear(flattened, weight)
    projection *= torch.rsqrt(flattened.square().mean(-1, keepdim=True) + 1e-20)
    expected_pre = torch.sigmoid(projection[:, :hc] * scale[0] + base[:hc]) + 1e-6
    expected_y = (external_pre.unsqueeze(-1) * hidden.float()).sum(-2).to(hidden.dtype)
    y, post, comb, pre = torch.ops.xllm_ops.hc_pre_va(
        hidden.npu(), weight.npu(), scale.npu(), base.npu(), external_pre.npu(), hc, 20, 1e-20, 1e-6
    )
    torch.testing.assert_close(pre.cpu(), expected_pre, rtol=0, atol=1e-4)
    torch.testing.assert_close(y.cpu(), expected_y, rtol=4e-3, atol=4e-3)
    assert post.dtype == comb.dtype == pre.dtype == torch.float32


class _StubAttention(nn.Module):
    def forward(self, x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: Any) -> torch.Tensor:
        return x * 0.9


class _StubNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _StubMlp(nn.Module):
    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        return x * 0.5


def _make_stubbed_layer(device: str, seed: int = 77) -> tuple[Any, dict[str, Any]]:
    torch.manual_seed(seed)
    cfg = {**_DSV41_TINY_CONFIG, "device": device, "dtype": torch.bfloat16}
    lm = v41.DeepseekV41ForCausalLM(cfg)
    with torch.no_grad():
        for name, param in lm.named_parameters():
            if not name.startswith("model.layers.3.hc."):
                continue
            std = 0.004 if name.endswith("_fn") else 0.5
            param.normal_(mean=0.0, std=std)
    layer = lm.model.layers[3]
    layer.self_attn = _StubAttention()
    layer.input_layernorm = _StubNorm()
    layer.post_attention_layernorm = _StubNorm()
    layer.mlp = _StubMlp()
    inputs = {
        "hidden": (torch.randn(5, 4, cfg["hidden_size"]) * 1.3).to(torch.bfloat16).to(device),
        "positions": torch.arange(5, device=device),
        "cos_sin_cache": None,
        "input_ids": None,
        "pre_mix": torch.rand(5, 4, device=device) + 0.1,
    }
    return layer, inputs


def _run_layer(layer: Any, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        hidden, ffn_pre = layer(
            inputs["hidden"], inputs["positions"], inputs["cos_sin_cache"], inputs["input_ids"], inputs["pre_mix"]
        )
    return hidden.clone(), ffn_pre.clone()


def _hc_pre_va_op_available() -> bool:
    return getattr(torch.ops.xllm_ops, "hc_pre_va", None) is not None


def _recording_hc_mixes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    recorded: list[tuple[Any, ...]] = []
    original = v41.DeepseekV41HyperConnection.hc_mixes

    def wrapper(
        self: Any, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre, post, comb = original(self, x, hc_fn, hc_scale, hc_base)
        recorded.append((x, hc_fn, hc_scale, hc_base, pre, post, comb))
        return pre, post, comb

    monkeypatch.setattr(v41.DeepseekV41HyperConnection, "hc_mixes", wrapper)
    return recorded


def _normalized_subblocks(
    monkeypatch: pytest.MonkeyPatch,
    layer: Any,
    inputs: dict[str, Any],
) -> tuple[list[tuple[Any, ...]], torch.Tensor, torch.Tensor]:
    hc = layer.hc
    mixes = _recording_hc_mixes(monkeypatch)
    hidden, ffn_pre = _run_layer(layer, inputs)
    attn_pre = mixes[0][4]
    records = [
        (
            mixes[0][0],
            mixes[0][1],
            mixes[0][2],
            mixes[0][3],
            inputs["pre_mix"],
            attn_pre,
            mixes[0][5],
            mixes[0][6],
            hc.hc_pre(mixes[0][0], inputs["pre_mix"]),
        ),
        (
            mixes[1][0],
            mixes[1][1],
            mixes[1][2],
            mixes[1][3],
            attn_pre,
            mixes[1][4],
            mixes[1][5],
            mixes[1][6],
            hc.hc_pre(mixes[1][0], attn_pre),
        ),
    ]
    return records, hidden, ffn_pre


def _recording_hc_post(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    recorded: list[tuple[Any, ...]] = []
    original = v41.DeepseekV41HyperConnection.hc_post

    def wrapper(
        self: Any, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        out = original(self, x, residual, post, comb)
        recorded.append((x, residual, post, comb, out))
        return out

    monkeypatch.setattr(v41.DeepseekV41HyperConnection, "hc_post", wrapper)
    return recorded


def _hc_mixes_of(hc: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        return hc.hc_mixes(x, hc.hc_attn_fn, hc.hc_attn_scale, hc.hc_attn_base)


def test_hc_pre_switch_gates_on_hc_mult_device_and_op(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("privateuseone")
    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", lambda: None)
    assert v41._hc_pre_mode_decision(torch.bfloat16, device, 4) == _MODE_EAGER

    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", lambda: object())
    assert v41._hc_pre_mode_decision(torch.bfloat16, device, 4) == _MODE_FUSED
    assert v41._hc_pre_mode_decision(torch.bfloat16, device, 2) == _MODE_EAGER
    assert v41._hc_pre_mode_decision(torch.float32, device, 4) == _MODE_EAGER
    assert v41._hc_pre_mode_decision(torch.bfloat16, torch.device("cpu"), 4) == _MODE_EAGER


class _RecordingAttention(nn.Module):
    def __init__(self, scale: float = 0.9) -> None:
        super().__init__()
        self.scale = scale
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: Any) -> torch.Tensor:
        self.inputs.append(x.clone())
        return x * self.scale


def test_decoder_layer_attention_input_uses_external_pre_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    layer, inputs = _make_stubbed_layer("cpu", seed=88)
    hc = layer.hc
    attention = _RecordingAttention()
    layer.self_attn = attention
    layer.hc._hc_pre_mode = _MODE_EAGER

    _normalized_subblocks(monkeypatch, layer, inputs)

    hidden = inputs["hidden"]
    attn_input = attention.inputs[0]
    expected = hc.hc_pre(hidden, inputs["pre_mix"])
    assert torch.equal(attn_input, expected), "attention input does not use the external pre_mix"

    own_pre = _hc_mixes_of(hc, hidden)[0]
    wrong = hc.hc_pre(hidden, own_pre)
    gap = (expected.float() - wrong.float()).abs().max().item()
    assert gap > 1e-2, f"pre_mix and attn_pre too close to discriminate (gap={gap:.3e})"


class _RecordingMlp(nn.Module):
    def __init__(self, scale: float = 0.5) -> None:
        super().__init__()
        self.scale = scale
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        del input_ids
        self.inputs.append(x.clone())
        return x * self.scale


def test_decoder_layer_mlp_input_uses_this_block_attn_pre(monkeypatch: pytest.MonkeyPatch) -> None:
    layer, inputs = _make_stubbed_layer("cpu", seed=89)
    hc = layer.hc
    mlp = _RecordingMlp()
    layer.mlp = mlp
    layer.hc._hc_pre_mode = _MODE_EAGER
    records, _, ffn_pre_out = _normalized_subblocks(monkeypatch, layer, inputs)

    attn_rec, ffn_rec = records[0], records[1]
    attn_pre = attn_rec[5]
    post_attn_hidden = ffn_rec[0]
    mlp_input = mlp.inputs[0]
    expected = hc.hc_pre(post_attn_hidden, attn_pre)
    assert torch.equal(mlp_input, expected), "MLP input does not collapse with this block's attn_pre"
    assert torch.equal(ffn_rec[8], expected), "sub-block collapse != MLP input"
    assert mlp_input.dim() == 2 and mlp_input.shape == expected.shape, f"shape={tuple(mlp_input.shape)}"

    assert torch.equal(ffn_rec[4], attn_pre), "ffn pre_mix != this block's attn_pre"
    assert torch.equal(ffn_pre_out, ffn_rec[5]), "returned ffn_pre mismatch"

    ffn_pre = ffn_rec[5]
    wrong_self = hc.hc_pre(post_attn_hidden, ffn_pre)
    gap_self = (expected.float() - wrong_self.float()).abs().max().item()
    assert gap_self > 1e-2, f"attn_pre and ffn_pre too close (gap={gap_self:.3e})"


def test_decoder_layer_hc_post_uses_the_pre_sub_block_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    layer, inputs = _make_stubbed_layer("cpu", seed=90)
    hc = layer.hc
    attention = _RecordingAttention()
    layer.self_attn = attention
    layer.hc._hc_pre_mode = _MODE_EAGER
    hc_post_handler = v41.DeepseekV41HyperConnection.hc_post
    posts = _recording_hc_post(monkeypatch)
    records, layer_hidden, _ = _normalized_subblocks(monkeypatch, layer, inputs)
    attn_rec, ffn_rec = records[0], records[1]

    block_hidden = inputs["hidden"]
    attn_post, attn_comb = attn_rec[6], attn_rec[7]
    ffn_post, ffn_comb = ffn_rec[6], ffn_rec[7]
    attn_out, residual_attn, post_attn, comb_attn, post_attn_hidden = posts[0]
    mlp_out, residual_ffn, post_ffn, comb_ffn, final_hidden = posts[1]

    assert torch.equal(attn_out, attention.inputs[0] * attention.scale), "attn_out mismatch"

    assert torch.equal(attn_rec[0], block_hidden), "attn sub-block input != block input"
    assert torch.equal(ffn_rec[0], post_attn_hidden), "ffn sub-block input != post-attn hidden"

    assert torch.equal(residual_attn, block_hidden), "attn hc_post residual != block input"
    assert torch.equal(post_attn, attn_post) and torch.equal(comb_attn, attn_comb), (
        "attn hc_post post/comb are not the attn sub-block's coefficients"
    )
    expected_post_attn = hc_post_handler(hc, attn_out, block_hidden, attn_post, attn_comb)
    assert torch.equal(post_attn_hidden, expected_post_attn), "post-attn hidden mismatch"

    assert torch.equal(residual_ffn, post_attn_hidden), "ffn hc_post residual != post-attn hidden"
    assert torch.equal(post_ffn, ffn_post) and torch.equal(comb_ffn, ffn_comb), (
        "ffn hc_post post/comb are not the ffn sub-block's coefficients"
    )
    expected_final = hc_post_handler(hc, mlp_out, post_attn_hidden, ffn_post, ffn_comb)
    assert torch.equal(final_hidden, expected_final), "final hidden mismatch"
    assert torch.equal(layer_hidden, final_hidden), "block did not return the recombined hidden"

    wrong_attn = {
        "residual=post-attn hidden": hc_post_handler(hc, attn_out, post_attn_hidden, attn_post, attn_comb),
        "post/comb=ffn sub-block": hc_post_handler(hc, attn_out, block_hidden, ffn_post, ffn_comb),
    }
    for label, wrong in wrong_attn.items():
        gap = (post_attn_hidden.float() - wrong.float()).abs().max().item()
        assert gap > 1e-2, f"attn hc_post {label} too close (gap={gap:.3e})"
    stale = hc_post_handler(hc, mlp_out, block_hidden, ffn_post, ffn_comb)
    gap_stale = (final_hidden.float() - stale.float()).abs().max().item()
    assert gap_stale > 1e-2, f"ffn hc_post stale residual too close (gap={gap_stale:.3e})"


def test_hc_pre_falls_back_without_op(monkeypatch: pytest.MonkeyPatch) -> None:
    if _hc_pre_va_op_available():
        pytest.skip("hc_pre_va op registered; the NPU gates test covers this host")
    device = "npu" if _npu_ready() else "cpu"

    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", lambda: None)
    layer_on, inputs_on = _make_stubbed_layer(device, seed=83)
    assert layer_on.hc._hc_pre_mode == _MODE_EAGER
    pre_on, post_on, comb_on = _hc_mixes_of(layer_on.hc, inputs_on["hidden"])

    layer_off, inputs_off = _make_stubbed_layer(device, seed=83)
    assert layer_off.hc._hc_pre_mode == _MODE_EAGER
    pre_off, post_off, comb_off = _hc_mixes_of(layer_off.hc, inputs_off["hidden"])

    hc = layer_off.hc
    x = inputs_off["hidden"]
    flattened = x.flatten(-2).float()
    rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + hc.norm_eps)
    ref = v41._hc_split_sinkhorn(
        F.linear(flattened, hc.hc_attn_fn) * rsqrt,
        hc.hc_attn_scale,
        hc.hc_attn_base,
        hc.hc_mult,
        hc.sinkhorn_iters,
        hc.hc_eps,
    )
    assert torch.equal(pre_off, ref[0]) and torch.equal(post_off, ref[1]) and torch.equal(comb_off, ref[2])
    assert torch.equal(pre_on, pre_off) and torch.equal(post_on, post_off) and torch.equal(comb_on, comb_off)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden", [32, 64])
def test_hc_post_preserves_shared_kernel_fallback(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, hidden: int
) -> None:
    x = torch.randn(2, hidden, dtype=dtype)
    residual = torch.randn(2, 4, hidden, dtype=dtype)
    post = torch.randn(2, 4)
    comb = torch.randn(2, 4, 4)
    expected = post.unsqueeze(-1) * x.float().unsqueeze(-2)
    expected = (expected + torch.einsum("tjk,tkd->tjd", comb, residual.float())).to(dtype)
    original_guard = v41._hc_post_npu_supported

    class NpuMetadata:
        device = SimpleNamespace(type="npu")

        def __getattr__(self, name: str) -> Any:
            return getattr(x, name)

    monkeypatch.setattr(
        v41,
        "_hc_post_npu_supported",
        lambda _x, r, p, c, mult: original_guard(NpuMetadata(), r, p, c, mult),
    )
    monkeypatch.setattr(v41, "_get_hc_post_op", lambda: None)
    calls: list[tuple[torch.Tensor, ...]] = []

    def shared_kernel(*args: torch.Tensor) -> torch.Tensor:
        calls.append(args)
        return expected.unsqueeze(0)

    monkeypatch.setattr(v41.kernels, "hc_post", shared_kernel)
    result = v41.DeepseekV41HyperConnection.hc_post(SimpleNamespace(hc_mult=4), x, residual, post, comb)
    torch.testing.assert_close(result, expected)
    assert len(calls) == (1 if hidden >= 64 else 0)
    if calls:
        torch.testing.assert_close(calls[0][0], x.unsqueeze(0))
        torch.testing.assert_close(calls[0][1], residual.unsqueeze(0))
        torch.testing.assert_close(calls[0][2], post.unsqueeze(0))
        torch.testing.assert_close(calls[0][3], comb.transpose(-2, -1).unsqueeze(0))
        assert calls[0][3].is_contiguous()
