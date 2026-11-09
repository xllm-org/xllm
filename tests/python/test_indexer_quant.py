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
from types import ModuleType, SimpleNamespace
from typing import Callable

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
from tests.python._deepseek_v41_fused_testkit import _reference_indexer_quant

_KERNEL_PATH = "xllm/python/kernels_npu/triton/indexer_quant.py"


def _load_kernel_module() -> ModuleType:
    pytest.importorskip("triton")
    source = Path(__file__).resolve().parents[2] / _KERNEL_PATH
    spec = importlib.util.spec_from_file_location("indexer_quant_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def kernel_quantize() -> Callable:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is required for Triton indexer quantization")
    module = _load_kernel_module()
    torch.npu.set_device(0)
    torch.npu.set_deterministic_level(1)
    return module.quantize_indexer_rows_int8


@pytest.mark.parametrize(
    "shape",
    [
        (0, 128),
        (1, 128),
        (7, 128),
        (1, 32, 128),
        (6, 32, 128),
        (17, 32, 128),
        (3, 7, 128),
        (2, 1, 128),
        (1024, 32, 128),
    ],
)
def test_indexer_quant_live_graph_is_bitwise_exact(kernel_quantize: Callable, shape: tuple[int, ...]) -> None:
    torch.manual_seed(42)
    x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    for _ in range(3):
        kernel_quantize(x)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = kernel_quantize(x)
    for kind in ["random", "zero", "tiny", "halfway", "new_input"]:
        x.normal_()
        if kind == "zero":
            x.zero_()
        elif kind == "tiny":
            x.mul_(1e-9)
        elif kind == "halfway":
            pattern = torch.tensor(
                [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 127],
                device="npu",
                dtype=torch.bfloat16,
            ).repeat(16)
            x.copy_(pattern)
        graph.replay()
        expected = _reference_indexer_quant(x)
        for lhs, rhs in zip(actual, expected):
            torch.testing.assert_close(lhs.cpu(), rhs.cpu(), rtol=0, atol=0)


def test_cpu_path_matches_declared_contract() -> None:
    x = torch.randn((4, 128), dtype=torch.bfloat16)
    codes, scales = csa2._quantize_indexer_rows_int8(x)
    reference_codes, reference_scales = _reference_indexer_quant(x)
    assert codes.dtype == torch.int8 and scales.dtype == torch.float16
    assert scales.shape == x.shape[:-1]
    assert torch.equal(codes, reference_codes)
    assert torch.equal(scales, reference_scales)


def test_zero_rows_clamp_scale_to_two_pow_minus_24() -> None:
    x = torch.zeros((3, 128), dtype=torch.bfloat16)
    codes, scales = csa2._quantize_indexer_rows_int8(x)
    assert torch.equal(codes, torch.zeros_like(codes))
    assert torch.equal(scales, torch.full_like(scales, 2.0**-24, dtype=torch.float16))


def test_tiny_rows_use_clamped_scale_and_round_to_zero() -> None:
    x = torch.full((2, 128), 1e-9, dtype=torch.bfloat16)
    codes, scales = csa2._quantize_indexer_rows_int8(x)
    assert torch.equal(scales, torch.tensor([2.0**-24, 2.0**-24], dtype=torch.float16))
    assert torch.equal(codes, torch.zeros_like(codes))


def test_half_to_even_ties_are_preserved() -> None:
    values = [127.0, 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5] + [0.0] * 119
    x = torch.tensor(values, dtype=torch.bfloat16).reshape(1, 128)
    codes, scales = csa2._quantize_indexer_rows_int8(x)
    assert scales.item() == 1.0
    assert codes[0, :9].tolist() == [127, 0, 2, 2, 4, 0, -2, -2, -4]
    assert torch.equal(codes, _reference_indexer_quant(x)[0])


def test_int8_boundary_values_are_exact() -> None:
    x = torch.cat(
        [
            torch.full((32,), 127.0, dtype=torch.bfloat16),
            torch.full((32,), -127.0, dtype=torch.bfloat16),
            torch.full((64,), 63.5, dtype=torch.bfloat16),
        ]
    ).reshape(1, 128)
    codes, scales = csa2._quantize_indexer_rows_int8(x)
    assert scales.item() == 1.0
    assert codes.max().item() == 127 and codes.min().item() == -127
    assert torch.equal(codes, _reference_indexer_quant(x)[0])


class _RowStub:
    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        rank: int = 2,
        contiguous: bool = True,
        width: int = 128,
        device: str = "npu",
    ) -> None:
        self.dtype = dtype
        self._rank = rank
        self._contiguous = contiguous
        self._width = width
        self.device = SimpleNamespace(type=device)

    def dim(self) -> int:
        return self._rank

    def is_contiguous(self) -> bool:
        return self._contiguous

    def size(self, axis: int = -1) -> int:
        assert axis == -1
        return self._width


def test_reason_tracks_kernel_module_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csa2._get_indexer_quant_module.cache_clear()
    module = csa2._get_indexer_quant_module()
    reason = csa2._indexer_quant_triton_reason(_RowStub())
    if module is None:
        assert reason == "triton indexer quantization unavailable"
    else:
        assert reason is None
        assert hasattr(module, "quantize_indexer_rows_int8")
        assert module.__file__.endswith("kernels_npu/triton/indexer_quant.py")
        assert Path(module.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])


def test_noncontiguous_input_falls_back_and_matches_reference() -> None:
    base = torch.randn((4, 256), dtype=torch.bfloat16)
    strided = base[:, ::2]
    assert strided.shape == (4, 128) and not strided.is_contiguous()
    codes, scales = csa2._quantize_indexer_rows_int8(strided)
    reference_codes, reference_scales = _reference_indexer_quant(strided)
    assert torch.equal(codes, reference_codes)
    assert torch.equal(scales, reference_scales)


def test_unsupported_dtype_and_width_fall_back_without_error() -> None:
    fp32 = torch.randn((2, 128), dtype=torch.float32)
    narrow = torch.randn((2, 64), dtype=torch.bfloat16)
    vector = torch.randn((128,), dtype=torch.bfloat16)
    for tensor in (fp32, narrow, vector):
        codes, scales = csa2._quantize_indexer_rows_int8(tensor)
        reference_codes, reference_scales = _reference_indexer_quant(tensor)
        assert torch.equal(codes, reference_codes)
        assert torch.equal(scales, reference_scales)


def test_cpu_device_reason_is_static_and_reported() -> None:
    x = torch.zeros((1, 128), dtype=torch.bfloat16)
    assert csa2._indexer_quant_triton_reason(x) == "not an NPU device"


def test_batched_and_empty_shapes_are_preserved() -> None:
    batched = torch.randn((3, 7, 128), dtype=torch.bfloat16)
    codes, scales = csa2._quantize_indexer_rows_int8(batched)
    assert codes.shape == (3, 7, 128) and scales.shape == (3, 7)
    empty = torch.empty((0, 128), dtype=torch.bfloat16)
    codes, scales = csa2._quantize_indexer_rows_int8(empty)
    assert codes.shape == (0, 128) and scales.shape == (0,)


def test_guarded_dispatch_prefers_fused_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = (torch.zeros((1, 128), dtype=torch.int8), torch.ones((1,), dtype=torch.float16))
    calls: list[torch.Tensor] = []

    def fake_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(x)
        return sentinel

    monkeypatch.setattr(
        csa2, "_get_indexer_quant_module", lambda: SimpleNamespace(quantize_indexer_rows_int8=fake_quantize)
    )
    monkeypatch.setattr(csa2, "_indexer_quant_triton_reason", lambda _x: None)
    x = torch.zeros((1, 128), dtype=torch.bfloat16)
    assert csa2._quantize_indexer_rows_int8(x) is sentinel
    assert len(calls) == 1 and calls[0] is x


def test_kernel_error_is_not_hidden_by_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_quantize(_x: torch.Tensor) -> None:
        raise RuntimeError("simulated kernel failure")

    monkeypatch.setattr(
        csa2, "_get_indexer_quant_module", lambda: SimpleNamespace(quantize_indexer_rows_int8=raising_quantize)
    )
    monkeypatch.setattr(csa2, "_indexer_quant_triton_reason", lambda _x: None)
    x = torch.randn((2, 128), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="simulated kernel failure"):
        csa2._quantize_indexer_rows_int8(x)
