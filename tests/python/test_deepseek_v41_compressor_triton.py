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

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
from tests.python._deepseek_v41_fused_testkit import _npu_ready

RING_ROWS = 32
WIDTH = 64


def _fake_compressor_module(calls: list) -> SimpleNamespace:
    def fake_from_projected(
        kv: torch.Tensor,
        scores: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        meta: torch.Tensor,
        out: torch.Tensor,
        **launch: int,
    ) -> torch.Tensor:
        calls.append(
            {
                "kv": kv,
                "scores": scores,
                "kv_state": kv_state,
                "score_state": score_state,
                "metadata": meta,
                "out": out,
                **launch,
            }
        )
        out.fill_(1.5)
        return out

    module = SimpleNamespace(compressor_from_projected=fake_from_projected, _cube_core_num=lambda: 8)
    return module


def _pool(
    kv: torch.Tensor,
    scores: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    start_pos: int,
    used_len: int,
    block_id: int,
) -> torch.Tensor | None:
    return csa2._v41_compressor_pool_triton(
        kv,
        scores,
        kv_state,
        score_state,
        start_pos=torch.tensor(start_pos, dtype=torch.int64),
        used_len=used_len,
        block_id=torch.tensor(block_id, dtype=torch.int64),
    )


def test_pool_wiring_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    head_dim, q_len = WIDTH, 5
    kv = torch.randn(q_len, head_dim, dtype=torch.float32)
    scores = torch.randn(q_len, head_dim, dtype=torch.float32)
    kv_state = torch.zeros(4, RING_ROWS, head_dim, dtype=torch.float32)
    score_state = torch.zeros_like(kv_state)
    calls: list = []
    fake = _fake_compressor_module(calls)
    monkeypatch.setattr(csa2, "_V41_COMPRESSOR_MODULE", fake)

    out = _pool(kv, scores, kv_state, score_state, 7, q_len, 3)

    assert len(calls) == 1
    got = calls[0]
    assert got["kv"].dtype == torch.float32
    assert got["kv"].shape == (q_len, head_dim)
    assert got["scores"].dtype == torch.float32
    assert got["scores"].shape == (q_len, head_dim)
    assert got["kv"].is_contiguous() and got["scores"].is_contiguous()
    assert got["kv_state"] is kv_state and got["score_state"] is score_state
    meta = got["metadata"]
    assert meta.dtype == torch.int32 and meta.shape == (5, 1)
    assert meta.is_contiguous()
    assert meta[:, 0].tolist() == [7, q_len, 0, 0, 3]
    assert got["out"].dtype == torch.bfloat16
    assert got["out"].shape == (q_len, head_dim)
    assert got["max_query_len"] == q_len and got["num_cores"] == 8
    assert out is not None and out is got["out"]
    assert torch.all(out == torch.tensor(1.5, dtype=torch.bfloat16))


def test_pool_propagates_failures_and_declines_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    kv = torch.randn(4, WIDTH, dtype=torch.float32)
    scores = torch.randn_like(kv)
    kv_state = torch.zeros(2, RING_ROWS, WIDTH, dtype=torch.float32)
    score_state = torch.zeros_like(kv_state)
    calls: list = []

    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    raising = SimpleNamespace(compressor_from_projected=_raise, _cube_core_num=lambda: 8)
    monkeypatch.setattr(csa2, "_V41_COMPRESSOR_MODULE", raising)
    with pytest.raises(RuntimeError, match="boom"):
        _pool(kv, scores, kv_state, score_state, 0, 4, 1)

    zero_cores = SimpleNamespace(compressor_from_projected=lambda *a, **k: None, _cube_core_num=lambda: 0)
    monkeypatch.setattr(csa2, "_V41_COMPRESSOR_MODULE", zero_cores)
    assert _pool(kv, scores, kv_state, score_state, 0, 4, 1) is None

    fake = _fake_compressor_module(calls)
    monkeypatch.setattr(csa2, "_V41_COMPRESSOR_MODULE", fake)
    assert _pool(kv[:0], scores[:0], kv_state, score_state, 0, 0, 1) is None
    assert calls == []


def test_swa_gate_static() -> None:
    table = torch.zeros(1, 4, dtype=torch.int64)
    dsa = SimpleNamespace(block_tables=[[table]])
    gate = csa2._v41_compressor_swa_gate
    resolved, why = gate(dsa, 0, 0, ring_blocks=1)
    assert resolved is table and why is None
    why2 = gate(dsa, 0, 0, ring_blocks=2)[1]
    assert why2 == "swa ring blocks per sequence 2 != 1"
    assert gate(dsa, 0, 0)[1] == "swa ring blocks per sequence 0 != 1"
    assert gate(dsa, 9, 0, ring_blocks=1)[1] == "no swa block table"
    empty_dsa = SimpleNamespace(block_tables=[[torch.empty(0)]])
    assert gate(empty_dsa, 0, 0, ring_blocks=1)[1] == "no swa block table"


def _reference_from_projected(
    kv: torch.Tensor,
    scores: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    metadata: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    out = torch.zeros_like(kv, dtype=output_dtype)
    starts = metadata[0].tolist()
    used = metadata[1].tolist()
    in_bases = metadata[3].tolist()
    blocks = metadata[4].tolist()
    for b in range(metadata.shape[1]):
        s, u, base, row = starts[b], used[b], in_bases[b], blocks[b]
        if u <= 0 or not 0 < row < kv_state.shape[0]:
            continue
        groups_in_batch = (s + u) // 2 - s // 2
        for lg in range(groups_in_batch):
            gidx = s // 2 + lg
            member_kv, member_score = [], []
            for r in range(2):
                pos = gidx * 2 + r
                off = pos - s
                if off >= 0:
                    member_kv.append(kv[base + off])
                    member_score.append(scores[base + off])
                else:
                    slot = pos % RING_ROWS
                    member_kv.append(kv_state[row, slot])
                    member_score.append(score_state[row, slot])
            kk = torch.stack(member_kv).float()
            ss = torch.stack(member_score).float()
            pooled = (kk * ss.softmax(dim=0)).sum(dim=0)
            out_row = base + (gidx + 1) * 2 - 1 - s
            out[out_row] = pooled.to(output_dtype)
        tail = min(u, RING_ROWS)
        for w in range(tail):
            seg_row = u - tail + w
            slot = (s + seg_row) % RING_ROWS
            kv_state[row, slot] = kv[base + seg_row]
            score_state[row, slot] = scores[base + seg_row]
    return out


def _run_parity_case(
    starts: list[int],
    used: list[int],
    blocks: list[int],
    *,
    seed: int,
    width: int = WIDTH,
    output_dtype: torch.dtype = torch.bfloat16,
) -> None:
    torch.manual_seed(seed)
    requests = len(starts)
    tokens = sum(used)
    dev = torch.npu.current_device()
    kv = torch.randn(max(tokens, 1), width, dtype=torch.float32, device=dev)
    scores = torch.randn(max(tokens, 1), width, dtype=torch.float32, device=dev)
    kv, scores = kv[:tokens], scores[:tokens]
    in_bases = []
    base = 0
    for u in used:
        in_bases.append(base)
        base += u
    metadata = torch.tensor(
        [
            starts,
            used,
            [0] * requests,
            in_bases,
            blocks,
        ],
        dtype=torch.int32,
        device=dev,
    )
    num_state_blocks = max(max(blocks) + 1, 1)
    kv_state = torch.randn(num_state_blocks, RING_ROWS, width, device=dev)
    score_state = torch.randn(num_state_blocks, RING_ROWS, width, device=dev)
    ref_kv_state = kv_state.clone()
    ref_score_state = score_state.clone()

    ref_out = _reference_from_projected(kv, scores, ref_kv_state, ref_score_state, metadata, output_dtype)

    from xllm.python.kernels_npu.triton import compressor

    out = torch.zeros(tokens, width, dtype=output_dtype, device=dev)
    got = compressor.compressor_from_projected(
        kv.contiguous(),
        scores.contiguous(),
        kv_state,
        score_state,
        metadata.contiguous(),
        out,
        max_query_len=max(used),
        num_cores=8,
    )
    assert got is out
    tolerance = 2e-2 if output_dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(out, ref_out, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(kv_state, ref_kv_state)
    torch.testing.assert_close(score_state, ref_score_state)


@pytest.mark.skipif(not _npu_ready(), reason="NPU is not available")
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_parity_multi_request_random(output_dtype: torch.dtype) -> None:
    for seed, starts, used in (
        (0, [0, 3, 29, 64, 97], [32, 5, 8, 17, 1]),
        (1, [11, 40, 7, 100, 61], [9, 48, 3, 2, 33]),
        (2, [5, 16, 71, 33, 90], [40, 6, 12, 31, 4]),
    ):
        _run_parity_case(starts, used, [1, 2, 3, 4, 5], seed=seed, output_dtype=output_dtype)


@pytest.mark.skipif(not _npu_ready(), reason="NPU is not available")
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_parity_ring_boundaries_and_padding(output_dtype: torch.dtype) -> None:
    starts = [0, 29, 3, 65, 0, 12]
    used = [32, 8, 4, 7, 0, 5]
    blocks = [1, 2, 3, 4, 5, 0]
    _run_parity_case(starts, used, blocks, seed=3, output_dtype=output_dtype)
