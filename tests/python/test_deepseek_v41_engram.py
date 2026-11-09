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

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.python._engram_reference import _engram_test_token_map, _reference_engram_lookup, _tiny_engram_hash_state
from xllm.python.models import deepseek_v41_engram as engram
from xllm.python.models.deepseek_v41_engram import (
    ENGRAM_IMAGE_SENTINEL_BASE_ID,
    ENGRAM_STORAGE_FP8,
    ENGRAM_STORAGE_INT8,
    EngramEmbedding,
    EngramHashState,
    EngramLayout,
    _load_engram_hash_state_sidecar,
    build_compressed_token_map_from_texts,
    build_engram_hash_state,
    normalize_token_text,
)


def _engram_lookup_op_registered() -> bool:
    return hasattr(torch.ops.xllm_ops, "engram_int8_lookup_cpu")


def _host_pinning_available() -> bool:
    npu = getattr(torch, "npu", None)
    return torch.cuda.is_available() or (npu is not None and npu.is_available())


def _make_offload_engram(seed: int, dim: int = 64, heads: int = 4, head_size: int = 128) -> EngramEmbedding:
    torch.manual_seed(seed)
    embed = engram.EngramEmbedding(
        num_embeddings=heads * head_size,
        dim=dim,
        head_sizes=[head_size] * heads,
        tp_size=2,
        tp_rank=0,
        quant_block=32,
        storage=engram.ENGRAM_STORAGE_INT8,
        device=torch.device("cpu"),
        offload=True,
    )
    with torch.no_grad():
        embed.weight.copy_(torch.randint(-127, 127, embed.weight.shape, dtype=torch.int8))
        embed.weight_scale.copy_(torch.rand(embed.weight_scale.shape) * 0.05)
    return embed


def _assert_bf16_within_one_ulp(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if torch.equal(actual, expected):
        return
    scale = expected.float().abs().clamp_min(2.0**-30)
    ulp = torch.pow(2.0, torch.floor(torch.log2(scale)) - 7)
    diff = (actual.float() - expected.float()).abs()
    max_err = diff.max().item()
    assert not bool((diff > ulp).any()), f"{label}: exceeds 1 bf16 ulp, max_err={max_err:.3e}"


_ENGRAM_KERNEL_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "xllm"
    / "core"
    / "kernels"
    / "npu"
    / "xllm_ops"
    / "engram_int8_lookup_cpu.cpp"
)

_ENGRAM_PYBIND_SHIM = """
#include <torch/extension.h>

namespace xllm { namespace kernel { namespace npu {
void engram_int8_lookup_cpu(const torch::Tensor& weight, const torch::Tensor& scale,
                            const torch::Tensor& ids, torch::Tensor& output);
}}}

namespace {
void engram_int8_lookup_cpu_entry(const torch::Tensor& weight, const torch::Tensor& scale,
                                  const torch::Tensor& ids, torch::Tensor& output) {
  xllm::kernel::npu::engram_int8_lookup_cpu(weight, scale, ids, output);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("engram_int8_lookup_cpu", &engram_int8_lookup_cpu_entry,
        "In-repo Engram CPU int8 lookup dequant");
}
"""


@pytest.fixture(scope="module")
def engram_lookup_kernel(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from torch.utils import cpp_extension

    if shutil.which(cpp_extension.get_cxx_compiler()) is None:
        pytest.skip("requires a C++ compiler")
    build_dir = tmp_path_factory.mktemp("engram_kernel")
    shim = build_dir / "engram_pybind_shim.cpp"
    shim.write_text(_ENGRAM_PYBIND_SHIM, encoding="utf-8")
    return cpp_extension.load(
        name="engram_kernel_under_test",
        sources=[str(_ENGRAM_KERNEL_SOURCE), str(shim)],
        extra_cflags=["-O2"],
        build_directory=str(build_dir),
    ).engram_int8_lookup_cpu


def test_engram_lookup_op_kernel_matches_golden(engram_lookup_kernel: Any) -> None:
    op = engram_lookup_kernel

    width = 32
    weight = torch.empty(2, width, dtype=torch.int8)
    weight[0] = torch.arange(1, width + 1, dtype=torch.int8)
    weight[1] = -torch.arange(1, width + 1, dtype=torch.int8)
    scale = torch.tensor([[2.0**-8], [2.0**-7]], dtype=torch.float32)
    ids = torch.tensor([0, 1], dtype=torch.int64)
    out = torch.empty(ids.numel(), width, dtype=torch.bfloat16)
    op(weight, scale, ids, out)

    expected = torch.tensor(
        [
            [value / 256.0 for value in range(1, width + 1)],
            [-value / 128.0 for value in range(1, width + 1)],
        ],
        dtype=torch.bfloat16,
    )
    assert torch.equal(out, expected)
    bits = out.view(torch.int16).to(torch.int32) & 0xFFFF
    assert int(bits[0, 0]) == 0x3B80
    assert int(bits[1, 0]) == 0xBC00

    generator = torch.Generator().manual_seed(77)
    rows, dim = 37, 64
    weight = torch.randint(-127, 128, (rows, dim), dtype=torch.int8, generator=generator)
    scale = (torch.rand(rows, dim // 32, generator=generator) * 0.05 + 1e-3).float()
    ids = torch.randint(0, rows, (11,), dtype=torch.int64)
    out = torch.empty(ids.numel(), dim, dtype=torch.bfloat16)
    op(weight, scale, ids, out)
    reference = torch.empty_like(out)
    _reference_engram_lookup(weight, scale, ids, reference)
    _assert_bf16_within_one_ulp(out, reference, "engram CPU kernel vs device reference")


def test_engram_lookup_op_kernel_boundary_goldens(engram_lookup_kernel: Any) -> None:
    op = engram_lookup_kernel

    width = 64
    rows = 259
    generator = torch.Generator().manual_seed(31337)
    weight = torch.randint(-127, 128, (rows, width), dtype=torch.int8, generator=generator)
    groups = width // 32
    scale = torch.empty(rows, groups, dtype=torch.float32)
    for row in range(rows):
        for group in range(groups):
            scale[row, group] = 2.0 ** -((row + group) % 8 + 1)
    weight_rows = weight.tolist()
    scale_rows = scale.tolist()

    for numel in (0, 1, 255, 256, 257):
        ids = torch.zeros(numel, dtype=torch.int64)
        if numel:
            ids = torch.randint(0, rows, (numel,), dtype=torch.int64, generator=torch.Generator().manual_seed(numel))
        out = torch.empty(numel, width, dtype=torch.bfloat16)
        op(weight, scale, ids, out)

        expected = torch.tensor(
            [
                [weight_rows[id_row][column] * scale_rows[id_row][column // 32] for column in range(width)]
                for id_row in ids.tolist()
            ],
            dtype=torch.bfloat16,
        ).reshape(numel, width)
        assert torch.equal(out, expected), (
            f"boundary golden mismatch (numel={numel}): {out[0, :4].tolist()} vs {expected[0, :4].tolist()}"
        )


@pytest.mark.skipif(
    not _engram_lookup_op_registered() or not _host_pinning_available(),
    reason="requires a registered lookup op and pinnable host staging",
)
def test_engram_lookup_op_matches_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    embed = _make_offload_engram(seed=201)
    ids = torch.randint(0, embed.num_embeddings, (9, embed.n_hash_cols), dtype=torch.int64)
    ids[0, 0] = -1
    ids[1, 1] = embed.num_embeddings + 7

    real_op = engram._get_engram_lookup_op()
    calls: list[int] = []

    def counting_op(weight: torch.Tensor, scale: torch.Tensor, flat: torch.Tensor, out: torch.Tensor) -> None:
        calls.append(int(flat.numel()))
        return real_op(weight, scale, flat, out)

    monkeypatch.setattr(engram, "_get_engram_lookup_op", lambda: counting_op)

    fused = embed._local_block_offload(ids).clone()
    flat = embed._ids_host[: ids.shape[0]].reshape(-1)
    assert calls == [int(flat.numel())], "the offload path must route through the op"

    rows_out = embed._rows_host[: ids.shape[0]].view(-1, embed.dim)
    _reference_engram_lookup(embed.weight, embed.weight_scale, flat, rows_out)
    rows_out.masked_fill_(~embed._owned_host[: ids.shape[0]].reshape(-1, 1), 0.0)
    expected = rows_out.view(ids.shape[0], embed.part_n_hash_cols, embed.dim).clone()

    _assert_bf16_within_one_ulp(fused, expected, "engram lookup op vs eager")

    local_ids = ids[:, embed.head_start : embed.head_start + embed.part_n_hash_cols]
    owned = (local_ids >= embed.vocab_start) & (local_ids < embed.vocab_end)
    assert not bool(owned.all())
    assert torch.equal(fused[~owned], torch.zeros_like(fused[~owned]))
    assert float(fused[owned].abs().max()) > 0.0


def test_engram_lookup_op_missing_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MissingOpNamespace:
        def __getattr__(self, name: str) -> object:
            raise AttributeError(name)

    monkeypatch.setattr(engram, "_engram_lookup_op", None)
    monkeypatch.setattr(torch.ops, "xllm_ops", _MissingOpNamespace(), raising=False)
    with pytest.raises(RuntimeError, match="not registered"):
        engram._get_engram_lookup_op()


def test_engram_lookup_op_contract_rejects_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _host_pinning_available():
        pytest.skip("host staging cannot be pinned; the op contract cannot be met")

    embed = _make_offload_engram(seed=203)
    ids = torch.randint(0, embed.num_embeddings, (5, embed.n_hash_cols), dtype=torch.int64)
    embed._ensure_host_buffers(ids.shape[0])
    flat = embed._ids_host[: ids.shape[0]].reshape(-1)
    rows_out = embed._rows_host[: ids.shape[0]].view(-1, embed.dim)

    assert embed._engram_lookup_op_contract_error(flat, rows_out) == ""

    monkeypatch.setattr(embed, "offload", False)
    assert embed._engram_lookup_op_contract_error(flat, rows_out)
    monkeypatch.setattr(embed, "offload", True)

    assert embed._engram_lookup_op_contract_error(flat, torch.empty(rows_out.shape, dtype=rows_out.dtype))
    flat_unpinned = flat.clone()
    assert not flat_unpinned.is_pinned()
    assert embed._engram_lookup_op_contract_error(flat_unpinned, rows_out)
    assert embed._engram_lookup_op_contract_error(flat, rows_out) == ""


def test_offload_construction_rejects_non_32_group() -> None:
    def make(*, quant_block: int, offload: bool) -> engram.EngramEmbedding:
        return engram.EngramEmbedding(
            12,
            64,
            [5, 7],
            1,
            0,
            quant_block=quant_block,
            storage=engram.ENGRAM_STORAGE_INT8,
            device=torch.device("cpu"),
            offload=offload,
        )

    with pytest.raises(ValueError, match="quant_block=32"):
        make(quant_block=64, offload=True)

    assert make(quant_block=64, offload=False).offload is False
    assert make(quant_block=32, offload=True).offload is True


def _copy_to_fresh_storage(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return self.detach().clone()


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_fallback_to_device_leaves_everything_untouched_when_a_copy_fails(
    monkeypatch: pytest.MonkeyPatch, fail_on_call: int
) -> None:
    embed = _make_offload_engram(seed=219)
    ids = torch.randint(0, embed.num_embeddings, (3, embed.n_hash_cols), dtype=torch.int64)
    embed._ensure_host_buffers(ids.shape[0])

    weight_before = embed.weight.detach().clone()
    scale_before = embed.weight_scale.detach().clone()
    weight_ptr = embed.weight.data.data_ptr()
    scale_ptr = embed.weight_scale.data.data_ptr()
    ids_host, owned_host, rows_host = embed._ids_host, embed._owned_host, embed._rows_host
    eager_block = embed._eager_staged_device
    state_before = (
        embed.offload,
        embed._host_capacity,
        embed._ids_host_pinned,
        embed._rows_host_pinned,
        embed._graph_metadata_bound,
        embed._eager_capacity,
    )

    calls = {"n": 0}

    def faulty_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            raise RuntimeError("injected .to() failure")
        return _copy_to_fresh_storage(self)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", faulty_to)
        with pytest.raises(RuntimeError, match="injected .to\\(\\) failure"):
            embed.fallback_to_device()

    assert calls["n"] == fail_on_call, "fallback must stop at the first failing copy"
    assert embed.offload is True
    assert embed.weight.data.data_ptr() == weight_ptr
    torch.testing.assert_close(embed.weight.detach(), weight_before, check_dtype=True)
    assert embed.weight_scale.data.data_ptr() == scale_ptr
    torch.testing.assert_close(embed.weight_scale.detach(), scale_before, check_dtype=True)
    assert embed._ids_host is ids_host
    assert embed._owned_host is owned_host
    assert embed._rows_host is rows_host
    assert embed._eager_staged_device is eager_block
    assert (
        embed.offload,
        embed._host_capacity,
        embed._ids_host_pinned,
        embed._rows_host_pinned,
        embed._graph_metadata_bound,
        embed._eager_capacity,
    ) == state_before


def test_fallback_to_device_commits_both_tables_and_clears_host_state(monkeypatch: pytest.MonkeyPatch) -> None:
    embed = _make_offload_engram(seed=220)
    ids = torch.randint(0, embed.num_embeddings, (3, embed.n_hash_cols), dtype=torch.int64)
    embed._ensure_host_buffers(ids.shape[0])
    weight_before = embed.weight.detach().clone()
    scale_before = embed.weight_scale.detach().clone()
    weight_ptr = embed.weight.data.data_ptr()
    scale_ptr = embed.weight_scale.data.data_ptr()

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", _copy_to_fresh_storage)
        assert embed.fallback_to_device() is True

    assert embed.offload is False
    assert embed.weight.data.data_ptr() != weight_ptr
    assert embed.weight_scale.data.data_ptr() != scale_ptr
    torch.testing.assert_close(embed.weight.detach(), weight_before, check_dtype=True)
    torch.testing.assert_close(embed.weight_scale.detach(), scale_before, check_dtype=True)
    assert (embed._ids_host, embed._owned_host, embed._rows_host) == (None, None, None)
    assert embed._host_capacity == 0
    assert embed._ids_host_pinned is False
    assert embed._rows_host_pinned is False
    assert not embed._graph_metadata_bound
    assert (embed._eager_staged_device, embed._eager_capacity) == (None, 0)


def test_engram_layout_prime_buckets() -> None:
    layout = EngramLayout(
        layer_ids=[1, 14],
        num_embeddings=[384006168, 384016682],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    assert layout.primes[0][0] == (67, 71)
    assert layout.primes[0][1] == (73, 79)
    assert layout.primes[1][0] == (83, 89)
    assert layout.primes[1][1] == (97, 101)
    assert layout.n_hash_cols == 4
    assert torch.equal(layout.offsets, torch.tensor([[0, 67, 138, 211], [0, 83, 172, 269]]))


def test_engram_token_map_normalization_chain() -> None:
    lookup, size = build_compressed_token_map_from_texts(
        ["The", "the", "THE", " thé ", "t\u0903\u20dd\ufe0fhe", "x", "y"]
    )
    assert size == 3
    assert lookup[0] == lookup[1] == lookup[2] == lookup[3] == lookup[4]
    assert lookup[5] != lookup[6]
    assert normalize_token_text("  A\tB\nC ") == "a b c"
    assert normalize_token_text(" ") == " "


def test_engram_hash_rolling_xor_and_dead_tokens() -> None:
    state = _tiny_engram_hash_state()
    pad_id = state.pad_id
    ids = torch.tensor([5, 6, 7, 10, 11, 12])
    positions = torch.tensor([0, 1, 2, 0, 1, 2])
    q_cu = torch.tensor([0, 3, 6])
    dead = torch.tensor([False, True, False, False, False, False])
    hashes = state(ids, positions, q_cu, dead, None, None, None, 0)
    assert hashes.shape == (6, 1, 4)

    comp = torch.tensor(_engram_test_token_map(), dtype=torch.int64)
    m = state.multipliers[0]
    r_pos0_2gram = (comp[5] * m[0]) ^ (pad_id * m[1])
    r_pos0_3gram = r_pos0_2gram ^ (pad_id * m[2])
    assert torch.equal(hashes[0, 0, 0], (r_pos0_2gram % 67).to(torch.int32))
    assert torch.equal(hashes[0, 0, 2], ((r_pos0_3gram % 73) + 138).to(torch.int32))
    r_pos2_2gram = (comp[7] * m[0]) ^ (pad_id * m[1])
    r_pos2_3gram = r_pos2_2gram ^ (pad_id * m[2])
    assert torch.equal(hashes[2, 0, 0], (r_pos2_2gram % 67).to(torch.int32))
    assert torch.equal(hashes[2, 0, 2], ((r_pos2_3gram % 73) + 138).to(torch.int32))
    r = (comp[11] * m[0]) ^ (comp[10] * m[1])
    assert torch.equal(hashes[4, 0, 0], (r % 67).to(torch.int32))


def test_engram_hash_slot_cache_roundtrip() -> None:
    state = _tiny_engram_hash_state()
    state._cache = torch.zeros(64, dtype=torch.int32)
    ids1 = torch.tensor([3, 9])
    bt = torch.tensor([[0, 0]], dtype=torch.int32)
    slot1 = torch.tensor([0, 1], dtype=torch.int32)
    q_cu = torch.tensor([0, 2])
    state(
        ids1,
        torch.tensor([0, 1]),
        q_cu,
        torch.zeros(2, dtype=torch.bool),
        None,
        slot1,
        bt,
        32,
    )
    comp = torch.tensor(_engram_test_token_map(), dtype=torch.int64)
    m = state.multipliers[0]
    ids2 = torch.tensor([20])
    q_cu2 = torch.tensor([0, 1])
    slot2 = torch.tensor([2], dtype=torch.int32)
    hashes = state(
        ids2,
        torch.tensor([2]),
        q_cu2,
        torch.zeros(1, dtype=torch.bool),
        None,
        slot2,
        bt,
        32,
    )
    r = (comp[20] * m[0]) ^ (comp[9] * m[1]) ^ (comp[3] * m[2])
    assert torch.equal(hashes[0, 0, 2], ((r % 73) + 138).to(torch.int32))


def test_engram_explicit_lookback_beats_slot_cache() -> None:
    state = _tiny_engram_hash_state()
    state._cache = torch.zeros(64, dtype=torch.int32)
    comp = torch.tensor(_engram_test_token_map(), dtype=torch.int64)
    m = state.multipliers[0]
    ids = torch.tensor([5])
    positions = torch.tensor([1])
    q_cu = torch.tensor([0, 1])
    dead = torch.zeros(1, dtype=torch.bool)
    bt = torch.tensor([[0, 0]], dtype=torch.int32)
    state._cache[0] = int(comp[9])

    def _hash_with_lookback(token_id: int) -> torch.Tensor:
        return state(
            ids,
            positions,
            q_cu,
            dead,
            torch.tensor([[token_id]], dtype=torch.int64),
            None,
            bt,
            32,
        )

    chunk_roll = comp[5] * m[0]
    explicit = _hash_with_lookback(7)
    explicit_roll = chunk_roll ^ (comp[7] * m[1])
    assert torch.equal(explicit[0, 0, 0], (explicit_roll % 67).to(torch.int32))
    assert torch.equal(explicit[0, 0, 1], ((explicit_roll % 71) + 67).to(torch.int32))
    unknown = _hash_with_lookback(-1)
    cache_roll = chunk_roll ^ (comp[9] * m[1])
    assert torch.equal(unknown[0, 0, 0], (cache_roll % 67).to(torch.int32))
    assert torch.equal(unknown[0, 0, 1], ((cache_roll % 71) + 67).to(torch.int32))
    dead_lookback = _hash_with_lookback(ENGRAM_IMAGE_SENTINEL_BASE_ID)
    assert torch.equal(dead_lookback[0, 0, 0], (chunk_roll % 67).to(torch.int32))
    assert not torch.equal(dead_lookback[0, 0, 0], (cache_roll % 67).to(torch.int32))


def test_engram_hash_state_builder_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    pytest.importorskip("numpy")
    import json

    from tokenizers import Tokenizer, models

    tok = Tokenizer(
        models.WordLevel(
            vocab={"[PAD]": 0, "Hello": 1, "hello": 2, "WORLD": 3, "x": 4},
            unk_token="[PAD]",
        )
    )
    tok.add_special_tokens(["[PAD]", "WORLD"])
    tokenizer_path = tmp_path / "tokenizer.json"
    tok.save(str(tokenizer_path))
    config = {
        "model_type": "deepseek_v41",
        "vocab_size": 5,
        "engram_layer_ids": [1],
        "engram_num_embeddings": [1000],
        "engram_max_ngram_size": 3,
        "engram_compressed_vocab_size": 4,
        "engram_pad_token_id": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    written = build_engram_hash_state(str(tmp_path))
    assert written == str(tmp_path / "engram_hash_state.bin")

    sidecar = _load_engram_hash_state_sidecar(str(tmp_path))
    assert sidecar is not None
    assert sidecar["token_map"].tolist() == [0, 1, 1, 2, 3]
    assert sidecar["multipliers"].shape == (1, 3)
    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[1000],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=5,
        compressed_vocab_size=4,
        pad_token_id=0,
    )
    state = EngramHashState.from_config(layout, {"model_path": str(tmp_path)})
    assert state.token_map.tolist() == [0, 1, 1, 2, 3]
    assert torch.equal(state.multipliers, sidecar["multipliers"])


def test_engram_embedding_padded_tp_shards_reassemble() -> None:
    head_sizes = [7, 11, 13, 5, 3, 9, 4, 6]
    dim = 32
    num_rows = sum(head_sizes)
    g = torch.Generator().manual_seed(1234)
    payload = torch.randint(0, 256, (num_rows, dim), dtype=torch.uint8, generator=g)
    payload = torch.where((payload == 0x7F) | (payload == 0xFF), payload - 1, payload)
    scales = torch.randint(126, 130, (num_rows, dim // 32), dtype=torch.uint8, generator=g)

    def make(tp_size: int, tp_rank: int) -> EngramEmbedding:
        emb = EngramEmbedding(num_rows, dim, head_sizes, tp_size, tp_rank)
        with torch.no_grad():
            emb.weight.copy_(payload[emb.vocab_start : emb.vocab_end])
            emb.weight_scale.copy_(scales[emb.vocab_start : emb.vocab_end])
        return emb

    ref = make(1, 0)
    t = 6
    ids = torch.zeros(t, len(head_sizes), dtype=torch.int64)
    for j, size in enumerate(head_sizes):
        ids[:, j] = torch.randint(0, size, (t,), generator=g) + sum(head_sizes[:j])

    for tp_size in (1, 2, 3, 4, 8, 16):
        part = (len(head_sizes) + tp_size - 1) // tp_size
        blocks = [make(tp_size, rank).local_block(ids) for rank in range(tp_size)]
        assert all(block.shape == (t, part, dim) for block in blocks)
        gathered = torch.cat(blocks, dim=1)
        assert torch.equal(gathered[:, : ref.n_hash_cols], ref.lookup(ids))
        assert not torch.count_nonzero(gathered[:, ref.n_hash_cols :])


def test_engram_embedding_int8_storage_dequant() -> None:
    head_sizes = [7, 11, 13, 5, 3]
    dim = 32
    num_rows = sum(head_sizes)
    g = torch.Generator().manual_seed(4321)
    payload = torch.randint(-127, 128, (num_rows, dim), dtype=torch.int8, generator=g)
    scales = torch.rand(num_rows, dim // 32, generator=g).float() * 0.05 + 0.01

    emb = EngramEmbedding(num_rows, dim, head_sizes, 1, 0, storage=ENGRAM_STORAGE_INT8)
    assert emb.weight.dtype == torch.int8
    assert emb.weight_scale.dtype == torch.float32
    with torch.no_grad():
        emb.weight.copy_(payload)
        emb.weight_scale.copy_(scales)

    t = 9
    ids = torch.zeros(t, len(head_sizes), dtype=torch.int64)
    for j, size in enumerate(head_sizes):
        ids[:, j] = torch.randint(0, size, (t,), generator=g) + sum(head_sizes[:j])
    out = emb.lookup(ids)
    expected = (payload[ids].float() * scales[ids].repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    assert out.shape == (t, len(head_sizes), dim)
    assert torch.equal(out, expected)


def _int8_engram_payload(head_sizes: list[int], dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = sum(head_sizes)
    g = torch.Generator().manual_seed(seed)
    payload = torch.randint(-127, 128, (num_rows, dim), dtype=torch.int8, generator=g)
    scales = torch.rand(num_rows, dim // 32, generator=g).float() * 0.05 + 0.01
    return payload, scales


def _make_int8_embedding(
    head_sizes: list[int],
    dim: int,
    payload: torch.Tensor,
    scales: torch.Tensor,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    offload: bool = False,
) -> EngramEmbedding:
    emb = EngramEmbedding(
        sum(head_sizes),
        dim,
        head_sizes,
        tp_size,
        tp_rank,
        storage=ENGRAM_STORAGE_INT8,
        device=torch.device("cpu"),
        offload=offload,
    )
    with torch.no_grad():
        emb.weight.copy_(payload[emb.vocab_start : emb.vocab_end])
        emb.weight_scale.copy_(scales[emb.vocab_start : emb.vocab_end])
    return emb


def _engram_ids(head_sizes: list[int], num_tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(num_tokens, len(head_sizes), dtype=torch.int64)
    for j, size in enumerate(head_sizes):
        ids[:, j] = torch.randint(0, size, (num_tokens,), generator=g) + sum(head_sizes[:j])
    return ids


def test_engram_offload_lookup_matches_hbm_bitwise(_engram_offload_lookup_op: None) -> None:
    head_sizes = [7, 11, 13, 5, 3]
    dim = 32
    payload, scales = _int8_engram_payload(head_sizes, dim, 2024)
    ref = _make_int8_embedding(head_sizes, dim, payload, scales)
    off = _make_int8_embedding(head_sizes, dim, payload, scales, offload=True)
    assert off.offload is True
    assert off.weight.device.type == "cpu"
    assert off.weight.dtype == torch.int8

    ids = _engram_ids(head_sizes, 13, 7)
    out_off = off.lookup(ids)
    out_ref = ref.lookup(ids)
    assert out_off.shape == (13, len(head_sizes), dim)
    assert torch.equal(out_off, out_ref)

    staged = off._eager_staged_device
    assert staged is not None
    _ = off.lookup(_engram_ids(head_sizes, 40, 8))
    assert off._eager_staged_device is staged
    assert off._eager_staged_device.shape[0] >= 256
    _ = off.lookup(_engram_ids(head_sizes, 300, 9))
    assert off._eager_staged_device.shape[0] >= 300


def test_engram_offload_empty_rank_returns_zeros(_engram_offload_lookup_op: None) -> None:
    head_sizes = [7, 11, 13, 5, 3]
    embed = EngramEmbedding(sum(head_sizes), 32, head_sizes, 8, 5, storage=ENGRAM_STORAGE_INT8, offload=True)
    ids = torch.zeros(3, len(head_sizes), dtype=torch.int64)
    assert embed.part_num_embeddings == 0
    block = embed.local_block(ids)
    assert block.shape == (3, 1, 32)
    assert not torch.count_nonzero(block)


@pytest.mark.parametrize("tp_size", [1, 2, 3, 4, 8, 16])
def test_engram_offload_tp_shards_match_hbm(tp_size: int, _engram_offload_lookup_op: None) -> None:
    head_sizes = [7, 11, 13, 5, 3, 9, 4, 6]
    dim = 32
    payload, scales = _int8_engram_payload(head_sizes, dim, 99)
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(0, sum(head_sizes), (9, len(head_sizes)), generator=g)
    for rank in range(tp_size):
        ref = _make_int8_embedding(head_sizes, dim, payload, scales, tp_size=tp_size, tp_rank=rank)
        off = _make_int8_embedding(head_sizes, dim, payload, scales, tp_size=tp_size, tp_rank=rank, offload=True)
        assert torch.equal(off.local_block(ids), ref.local_block(ids))


def test_engram_offload_unsupported_storage_falls_back() -> None:
    emb = EngramEmbedding(
        12,
        32,
        [5, 7],
        1,
        0,
        storage=ENGRAM_STORAGE_FP8,
        device=torch.device("cpu"),
        offload=True,
    )
    assert emb.offload is False
    assert emb.weight.dtype == torch.uint8
    assert emb.weight.device.type == "cpu"


def test_engram_offload_fallback_to_device() -> None:
    head_sizes = [5, 7]
    dim = 32
    payload, scales = _int8_engram_payload(head_sizes, dim, 5)
    emb = _make_int8_embedding(head_sizes, dim, payload, scales, offload=True)
    assert emb.offload is True
    assert emb.fallback_to_device() is True
    assert emb.offload is False
    assert emb.weight.dtype == torch.int8
    assert emb.weight.device.type == "cpu"
    assert torch.equal(emb.weight, payload)
    assert torch.equal(emb.weight_scale, scales)
    assert emb.fallback_to_device() is False


def test_staging_operand_pinned_accepts_nonzero_offset_view() -> None:
    staging = torch.zeros(8, 4, dtype=torch.int64)
    shifted = staging[2:]
    assert shifted.data_ptr() != staging.data_ptr()
    assert EngramEmbedding._staging_operand_pinned(shifted, staging, pinned=True) is True
    assert EngramEmbedding._staging_operand_pinned(staging, staging, pinned=True) is True

    foreign = torch.zeros(6, 4, dtype=torch.int64)
    assert EngramEmbedding._staging_operand_pinned(foreign, staging, pinned=True) is False
    assert EngramEmbedding._staging_operand_pinned(shifted, staging, pinned=False) is False
    assert EngramEmbedding._staging_operand_pinned(shifted, None, pinned=True) is False


def test_host_scratch_growth_waits_for_pending_upload() -> None:
    head_sizes = [7, 11, 13, 5, 3]
    payload, scales = _int8_engram_payload(head_sizes, 32, 4242)
    emb = _make_int8_embedding(head_sizes, 32, payload, scales, offload=True)
    waits = []
    emb._ensure_host_scratch(4)
    ids = emb._ids_host
    emb._upload_done = SimpleNamespace(synchronize=lambda: waits.append(ids.data_ptr()))
    emb._ensure_host_scratch(4)
    assert waits == []
    emb._ensure_host_scratch(emb._host_capacity + 1)
    assert waits == [ids.data_ptr()]
    assert emb._upload_done is None
    assert emb._ids_host.data_ptr() != ids.data_ptr()


def test_host_pin_failure_warns_once_and_raises_with_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.logger as shared_logger

    monkeypatch.setattr(engram, "_host_pin_warned", False)

    warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class _WarningRecorder:
        def warning(self, *args: object, **kwargs: object) -> None:
            warnings.append((args, kwargs))

    monkeypatch.setattr(shared_logger, "logger", _WarningRecorder(), raising=False)

    pin = {"fails": True}
    real_empty = torch.empty

    def _pin_aware_empty(*args: object, **kwargs: object) -> torch.Tensor:
        if kwargs.get("pin_memory") and pin["fails"]:
            raise RuntimeError("no registered accelerator")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", _pin_aware_empty)

    head_sizes = [7, 11, 13, 5, 3]
    payload, scales = _int8_engram_payload(head_sizes, 32, 4242)
    emb = _make_int8_embedding(head_sizes, 32, payload, scales, offload=True)
    assert emb.offload is True

    emb._ensure_host_scratch(4)
    assert emb._ids_host_pinned is False
    assert emb._rows_host_pinned is False
    assert len(warnings) == 1
    assert "cannot allocate pinned CPU staging" in warnings[0][0][0]
    assert emb._host_pin_failure is not None

    emb._ensure_host_scratch(emb._host_capacity + 1)
    assert len(warnings) == 1

    pin["fails"] = False
    emb._ensure_host_scratch(emb._host_capacity + 1)
    assert emb._ids_host_pinned is True
    assert emb._host_pin_failure is None
    assert len(warnings) == 1

    pin["fails"] = True
    emb._ensure_host_scratch(emb._host_capacity + 1)
    assert emb._ids_host_pinned is False
    assert len(warnings) == 2

    ids = torch.zeros(4, emb.n_hash_cols, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="no registered accelerator"):
        emb._local_block_offload(ids)

    other = _make_int8_embedding(head_sizes, 32, payload, scales, offload=True)
    pin["fails"] = False
    other._ensure_host_scratch(4)
    assert other._ids_host_pinned is True
    assert other._host_pin_failure is None
    assert emb._host_pin_failure is not None


@pytest.mark.parametrize("fail_call", [2, 3])
def test_failed_host_scratch_growth_retains_buffers_and_can_retry(
    monkeypatch: pytest.MonkeyPatch, fail_call: int
) -> None:
    def allocate(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, bool, None]:
        return torch.empty(shape, dtype=dtype), False, None

    monkeypatch.setattr(engram, "_host_buffer", allocate)
    embed = _make_offload_engram(23)
    embed._ensure_host_scratch(4)
    capacity = embed._host_capacity
    buffers = (embed._ids_host, embed._owned_host, embed._rows_host)
    calls = 0

    def fail_during_growth(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, bool, None]:
        nonlocal calls
        calls += 1
        if calls == fail_call:
            raise MemoryError("scratch allocation failed")
        return allocate(shape, dtype)

    monkeypatch.setattr(engram, "_host_buffer", fail_during_growth)
    with pytest.raises(MemoryError, match="scratch allocation failed"):
        embed._ensure_host_scratch(capacity + 1)
    assert embed._host_capacity == capacity
    assert all(
        old is new for old, new in zip(buffers, (embed._ids_host, embed._owned_host, embed._rows_host), strict=True)
    )
    monkeypatch.setattr(engram, "_host_buffer", allocate)
    embed._ensure_host_scratch(capacity + 1)
    assert embed._host_capacity > capacity
    for buffer in (embed._ids_host, embed._owned_host, embed._rows_host):
        assert buffer.shape[0] == embed._host_capacity


def test_failed_eager_staging_growth_retains_buffer_and_can_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    embed = _make_offload_engram(29)
    embed._ensure_eager_staging(4)
    capacity, buffer = embed._eager_capacity, embed._eager_staged_device
    allocate = embed._alloc_staging

    def fail_during_growth(capacity: int) -> torch.Tensor:
        raise MemoryError("staging allocation failed")

    monkeypatch.setattr(embed, "_alloc_staging", fail_during_growth)
    with pytest.raises(MemoryError, match="staging allocation failed"):
        embed._ensure_eager_staging(capacity + 1)
    assert embed._eager_capacity == capacity
    assert embed._eager_staged_device is buffer
    monkeypatch.setattr(embed, "_alloc_staging", allocate)
    embed._ensure_eager_staging(capacity + 1)
    assert embed._eager_capacity > capacity
    assert embed._eager_staged_device.shape[0] == embed._eager_capacity


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("tokens", [0, 3])
def test_gather_heads_uses_local_rows_and_trims_tp_padding(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, tokens: int
) -> None:
    from xllm.python import distributed

    heads = 5
    local_heads = (heads + tp_size - 1) // tp_size
    full = torch.arange(tokens * local_heads * tp_size * 3).reshape(tokens, local_heads * tp_size, 3)
    local = full[:, :local_heads].clone()
    calls = []

    def gather(rows: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
        assert rows is local
        assert dim == 1 and world_size == tp_size
        calls.append(rows)
        return full

    monkeypatch.setattr(distributed, "tp_all_gather", gather)
    embed = SimpleNamespace(tp_size=tp_size, n_hash_cols=heads)
    result = EngramEmbedding.gather_heads(embed, local)
    if tp_size == 1:
        assert result is local
        assert not calls
    else:
        torch.testing.assert_close(result, full[:, :heads])
        assert len(calls) == 1
