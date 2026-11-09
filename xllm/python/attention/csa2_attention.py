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

from dataclasses import dataclass, replace
from typing import Any

import torch

from scripts.logger import logger
from xllm.python import kernels
from xllm.python.attention.csa_attention import (
    DsaAttentionBackend,
    _CompressedAttentionCacheMapping,
    _get_layer_cache_tensor,
    _scatter_by_slot,
)
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    READ_TABLE_LAYOUT_RING,
    DsaMetadata,
    DsaMetadataBuilder,
    build_cache_specs_v41,
    last_ring_writes,
)
from xllm.python.kernels_npu.op_api import has_op_api
from xllm.python.model_executor.dsa_cp import DsaCpContext, get_current_dsa_cp_runtime
from xllm.python.models.deepseek_v41 import (
    V41_QUANT_ASCEND,
    V41_QUANT_NONE,
)

# Softmax-scale exponent base of the sparse kernel's running-max floor.
_NEG_INF_FLOOR = -1e30
_FP32_TINY = torch.finfo(torch.float32).tiny
# Query rows per attention tile (the reference kernel processes 64 gathered
# rows per tile; a few hundred rows keep the [s, H, n] score tile bounded).
_ATTENTION_QUERY_TILE = 256
# Query rows per indexer scoring tile. The fp32 score intermediate is
# [tile, index_heads, capacity]; a whole-prompt chunk against a long-context
# committed capacity reaches multiple GiB per layer (30k ctx x 2k chunk x 32
# heads x 4B ~= 4 GiB), which overflows the activation headroom of the
# W8A8+engram full model. Top-k selection is per query row, so tiling over
# the query axis is exact; decode/graph batches (q_len <= tile) take the
# single-tile path unchanged.
_INDEXER_QUERY_TILE = 256


@dataclass
class _SwaGroupMetadata:
    """SWA-group tensors shared by every layer (one ring layout per request)."""

    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    block_size: int
    cache: torch.Tensor


@dataclass
class Csa2LayerContext:
    x: torch.Tensor
    qr: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor


@dataclass
class CSA2SharedRuntime:
    key_cache: torch.Tensor | None = None
    key_block_table: torch.Tensor | None = None
    index_cache: torch.Tensor | None = None
    index_block_table: torch.Tensor | None = None
    index_scale: torch.Tensor | None = None
    compress_ratio: int = 0
    topk_idxs: torch.Tensor | None = None
    topk_layer: int = -1
    # Eager candidate masks are indexed per query; fused candidates below are block indices.
    candidates: list[torch.Tensor | None] | None = None
    candidates_layer: int = -1
    candidate_blocks: torch.Tensor | None = None
    candidate_blocks_layer: int = -1


_ROPE_EXPANDED_ATTR = "_xllm_v41_expanded_rope"

_partial_rotary_op: Any = None


def _get_partial_rotary_op() -> Any:
    global _partial_rotary_op
    if _partial_rotary_op is None:
        if getattr(torch.ops.xllm_ops, "npu_inplace_partial_rotary_mul", None) is None:
            return None
        _partial_rotary_op = kernels.npu_inplace_partial_rotary_mul
    return _partial_rotary_op


def _rope_fused_device_supported(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def _rope_fused_supported(x: torch.Tensor, cos_sin_cache: torch.Tensor) -> bool:
    half = cos_sin_cache.size(-1) // 2
    dim = x.size(-1)
    return (
        _rope_fused_device_supported(x)
        and x.dim() in (2, 3)
        and x.dtype == torch.bfloat16
        # The vendor partial-rotary kernel supports exactly 64 rotary dimensions.
        and 2 * half == 64
        and cos_sin_cache.size(-1) == 2 * half
        and dim <= 1024
        and dim % 2 == 0
        and 2 * half <= dim
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.dtype == torch.float32
        and x.is_contiguous()
        and _get_partial_rotary_op() is not None
    )


_fused_rope_active_logged = False
_fused_rope_fallback_logged = False
_fused_rope_inverse_active_logged = False


def _log_fused_rope_active() -> None:
    global _fused_rope_active_logged
    if not _fused_rope_active_logged:
        _fused_rope_active_logged = True
        logger.info("interleaved RoPE fused path active")


def _log_fused_rope_inverse_active() -> None:
    global _fused_rope_inverse_active_logged
    if not _fused_rope_inverse_active_logged:
        _fused_rope_inverse_active_logged = True
        logger.info("interleaved RoPE fused INVERSE path active")


def _log_fused_rope_fallback(x: torch.Tensor, cos_sin_cache: torch.Tensor, inverse: bool) -> None:
    global _fused_rope_fallback_logged
    if _fused_rope_fallback_logged:
        return
    _fused_rope_fallback_logged = True
    logger.warning(
        f"the fused partial rotary op is unavailable "
        f"(device={x.device.type} dtype={x.dtype} dim={x.dim()} inverse={inverse} "
        f"cache_dtype={cos_sin_cache.dtype} op_registered={_get_partial_rotary_op() is not None}); "
        "staying on the eager path"
    )


def _expand_rope_rows(
    cos_sin_cache: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = cos_sin_cache.size(-1) // 2
    cos = cos_sin_cache[..., :half].repeat_interleave(2, dim=-1).to(dtype).contiguous()
    sin = cos_sin_cache[..., half:].repeat_interleave(2, dim=-1).to(dtype).contiguous()
    return cos, sin


_rope_preexpand_fallback_logged = False


def _log_rope_preexpand_fallback(reason: str) -> None:
    global _rope_preexpand_fallback_logged
    if _rope_preexpand_fallback_logged:
        return
    _rope_preexpand_fallback_logged = True
    logger.warning(
        f"interleaved RoPE pre-expanded table cache unavailable ({reason}); staying on the per-call expansion path"
    )


def _expanded_rope_tables(
    cos_sin_cache: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build once outside capture; replay must reuse the expanded device tables."""
    per_dtype = getattr(cos_sin_cache, _ROPE_EXPANDED_ATTR, None)
    if per_dtype is not None and dtype in per_dtype:
        return per_dtype[dtype]
    if _is_stream_capturing():
        return None
    if per_dtype is None:
        per_dtype = {}
        try:
            setattr(cos_sin_cache, _ROPE_EXPANDED_ATTR, per_dtype)
        except Exception as exc:  # noqa: BLE001
            _log_rope_preexpand_fallback(f"cannot tag table tensor: {exc}")
            return None
    try:
        pair = _expand_rope_rows(cos_sin_cache, dtype)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        _log_rope_preexpand_fallback(f"expansion OOM: {exc}")
        return None
    per_dtype[dtype] = pair
    return pair


def _apply_interleaved_rope_fused(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    half = cos_sin_cache.size(-1) // 2
    pos = positions.reshape(-1).to(torch.int64)
    expanded = _expanded_rope_tables(cos_sin_cache, x.dtype)
    if expanded is not None:
        cos = expanded[0].index_select(0, pos)
        sin = expanded[1].index_select(0, pos)
    else:
        rows = cos_sin_cache.index_select(0, pos)
        cos = rows[..., :half].repeat_interleave(2, dim=-1).to(x.dtype).contiguous()
        sin = rows[..., half:].repeat_interleave(2, dim=-1).to(x.dtype).contiguous()
    dim = x.size(-1)
    x3d = x if x.dim() == 3 else x.unsqueeze(1)
    _get_partial_rotary_op()(x3d, cos, sin, dim - 2 * half, 2 * half, inverse=inverse)
    return x


def apply_interleaved_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate adjacent element pairs in the last 2 * half dimensions."""
    if _rope_fused_supported(x, cos_sin_cache):
        if inverse:
            _log_fused_rope_inverse_active()
        else:
            _log_fused_rope_active()
        return _apply_interleaved_rope_fused(x, positions, cos_sin_cache, inverse)
    _log_fused_rope_fallback(x, cos_sin_cache, inverse)
    half = cos_sin_cache.size(-1) // 2
    rows = cos_sin_cache.index_select(0, positions.reshape(-1).to(torch.int64))
    cos = rows[..., :half].to(torch.float32)
    sin = rows[..., half:].to(torch.float32)
    if inverse:
        sin = -sin
    for _ in range(x.dim() - 2):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    dim = x.size(-1)
    tail = x[..., dim - 2 * half :].to(torch.float32)
    even = tail[..., 0::2]
    odd = tail[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    rotated = rotated.flatten(-2).to(x.dtype)
    if 2 * half == dim:
        return rotated
    return torch.cat((x[..., : dim - 2 * half], rotated), dim=-1)


def _is_stream_capturing() -> bool:
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and npu.is_available() and npu.is_current_stream_capturing())


_COMPRESSOR_TRITON_RING = 32

_v41_compressor_fallback_logged = False


def _log_v41_compressor_fallback(reason: str) -> None:
    global _v41_compressor_fallback_logged
    if _v41_compressor_fallback_logged:
        return
    _v41_compressor_fallback_logged = True
    logger.warning(f"fused V4.1 compressor is inactive ({reason}); staying on the pure-torch pooling path")


_v41_compressor_active_logged = False


def _log_v41_compressor_active() -> None:
    global _v41_compressor_active_logged
    if not _v41_compressor_active_logged:
        _v41_compressor_active_logged = True
        logger.info("fused V4.1 compressor Triton path active (op=xllm.python.kernels_npu.triton.compressor)")


_V41_COMPRESSOR_MODULE: Any = None


def _get_v41_compressor_module() -> Any:
    global _V41_COMPRESSOR_MODULE
    if _V41_COMPRESSOR_MODULE is None:
        try:
            from xllm.python.kernels_npu.triton import compressor as _module
        except ModuleNotFoundError as exc:
            if exc.name != "triton":
                raise
            return None
        _V41_COMPRESSOR_MODULE = _module
    return _V41_COMPRESSOR_MODULE


def _v41_compressor_triton_reason(
    ratio: int,
    kv_state: torch.Tensor | None,
    score_state: torch.Tensor | None,
) -> str | None:
    if ratio != 2:
        return f"unsupported ratio {ratio}"
    if kv_state is None or score_state is None:
        return "no compressor state ring"
    if kv_state.dtype != torch.float32 or score_state.dtype != torch.float32:
        return "ring state not fp32"
    if kv_state.dim() != 3 or score_state.dim() != 3:
        return "ring state not [blocks, rows, dim]"
    if kv_state.size(1) != _COMPRESSOR_TRITON_RING or score_state.size(1) != _COMPRESSOR_TRITON_RING:
        return "ring rows != 32"
    head_dim = kv_state.size(2)
    if head_dim < 1 or head_dim & (head_dim - 1):
        return "compressor width is not a positive power of two"
    if kv_state.device.type != "npu":
        return "not an NPU device"
    if _get_v41_compressor_module() is None:
        return "triton compressor unavailable"
    return None


def _v41_compressor_swa_gate(
    dsa: Any,
    layer_id: int,
    ori_cache_idx: int,
    ring_blocks: int | None = None,
) -> tuple[torch.Tensor | None, str | None]:
    block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, ori_cache_idx)
    if block_table is None or block_table.numel() == 0:
        return block_table, "no swa block table"
    count = int(ring_blocks) if ring_blocks is not None else 0
    if count != 1:
        return block_table, f"swa ring blocks per sequence {count} != 1"
    return block_table, None


def _v41_compressor_num_cores() -> int:
    module = _get_v41_compressor_module()
    if module is None:
        return 0
    return int(module._cube_core_num())


def _v41_compressor_pool_triton(
    kv_seq: torch.Tensor,
    score_seq: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    start_pos: torch.Tensor,
    used_len: int,
    block_id: torch.Tensor,
) -> torch.Tensor | None:
    """Pool one segment's ratio-2 groups with the Triton kernel.

    Returns the token-aligned bf16 pooled rows ``[q_len, head_dim]`` (the caller
    gathers the committed group rows and applies RMSNorm), or None when the op
    declines so the caller keeps the pure-torch path. ``start_pos`` / ``block_id``
    are device scalars, so the control metadata is built with device arithmetic
    and stays ACL-graph friendly.
    """
    module = _get_v41_compressor_module()
    if module is None:
        return None
    q_len = int(kv_seq.size(0))
    if q_len <= 0:
        return None
    num_cores = _v41_compressor_num_cores()
    if num_cores <= 0:
        return None
    dev = kv_seq.device
    head_dim = int(kv_seq.size(1))
    meta = torch.stack(
        (
            start_pos.to(torch.int32).reshape(()),
            torch.full((), int(used_len), dtype=torch.int32, device=dev),
            torch.zeros((), dtype=torch.int32, device=dev),
            torch.zeros((), dtype=torch.int32, device=dev),
            block_id.to(torch.int32).reshape(()),
        )
    ).reshape(5, 1)
    out = torch.empty(q_len, head_dim, dtype=torch.bfloat16, device=dev)
    module.compressor_from_projected(
        kv_seq.contiguous(),
        score_seq.contiguous(),
        kv_state,
        score_state,
        meta.contiguous(),
        out,
        max_query_len=q_len,
        num_cores=num_cores,
    )
    return out


def _indexer_score(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    score = torch.einsum("shd,td->sht", q.float(), index_k.float())
    score = score.relu_() * weights.float().unsqueeze(-1)
    return score.sum(dim=1)


def _select_candidate_blocks(
    score: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    width = score.size(-1)
    padded = torch.nn.functional.pad(score, (0, -width % block_size), value=float("-inf"))
    blocks = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = blocks.size(-1)
    # Always retain the newest, partially filled block; full older blocks can outscore it.
    last = torch.div(compress_lens - 1, block_size, rounding_mode="floor")
    blocks = blocks.masked_fill(
        torch.arange(num_blocks, device=score.device) == last.unsqueeze(-1),
        float("inf"),
    )
    top = blocks.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    # Avoid Ascend's slow repeat_interleave path for one-byte dtypes.
    expanded = keep.unsqueeze(-1).expand(*keep.shape, block_size)
    return expanded.flatten(-2)[..., :width]


def _indexer_topk(
    score: torch.Tensor,
    compress_lens: torch.Tensor,
    index_topk: int,
    offset: int,
) -> torch.Tensor:
    topk = min(index_topk, score.size(-1))
    order = torch.argsort(score, dim=-1, descending=True, stable=True)
    idxs = order[..., :topk]
    idxs = idxs.sort(dim=-1).values
    finite = torch.isfinite(score.gather(-1, idxs))
    visible = compress_lens.unsqueeze(-1)
    return torch.where((idxs < visible) & finite, idxs + offset, -1).to(torch.int32)


# These indices and the sentinel are exact in FP32; INT32 sorting falls back to AI CPU.
_QLI_V3_REBASE_FP32_SENTINEL = float(1 << 24)


def _rebase_qli_picks(
    picks: torch.Tensor,
    visible: torch.Tensor,
    offset: int,
    *,
    fp32_sort: bool = False,
) -> torch.Tensor:
    valid = (picks >= 0) & (picks < visible)
    if fp32_sort:
        sentinel_f = _QLI_V3_REBASE_FP32_SENTINEL
        key = torch.where(
            valid,
            picks.to(torch.float32),
            torch.full(picks.shape, sentinel_f, dtype=torch.float32, device=picks.device),
        )
        key = key.sort(dim=-1).values
        picks = torch.where(
            key < sentinel_f,
            key.to(torch.int32),
            torch.full_like(picks, -1),
        )
    else:
        sentinel = torch.iinfo(torch.int32).max
        sortable = torch.where(valid, picks, torch.full_like(picks, sentinel))
        sortable = sortable.sort(dim=-1).values
        picks = torch.where(sortable >= sentinel, torch.full_like(sortable, -1), sortable)
    return torch.where(picks >= 0, picks + offset, torch.full_like(picks, -1)).to(torch.int32)


def _picked_row_mask(idxs: torch.Tensor, num_rows: int) -> torch.Tensor:
    picked = idxs >= 0
    valid = picked & (idxs < num_rows)
    safe_idxs = torch.where(valid, idxs, torch.full_like(idxs, num_rows))
    wide_mask = torch.zeros(idxs.shape[0], num_rows + 1, dtype=torch.bool, device=idxs.device)
    wide_mask.scatter_(1, safe_idxs, valid)
    return wide_mask[:, :num_rows]


def _window_visible_idxs(
    positions_seq: torch.Tensor,
    col_pos: torch.Tensor,
    cols: torch.Tensor,
    win: int,
) -> torch.Tensor:
    """Per-query SWA window index matrix over the fixed-capacity window columns.

    The window columns are ``[win ring candidates ; current chunk]``: a ring
    candidate's absolute position may be negative (before the sequence start)
    and the chunk columns follow it. Query ``i`` sits at absolute position
    ``positions_seq[i]`` and must see exactly the columns whose absolute
    position lies in ``[max(p - win + 1, 0), p]`` -- a *per-query* lower bound
    (only the chunk's first token shares the chunk-level
    ``max(prev - win + 1, 0)``). Masked columns become the ``-1`` sentinel,
    which :func:`_picked_row_mask` drops.
    """
    window_lo = torch.clamp(positions_seq - win + 1, min=0)
    valid = (col_pos.unsqueeze(0) >= window_lo.unsqueeze(-1)) & (col_pos.unsqueeze(0) <= positions_seq.unsqueeze(-1))
    return torch.where(valid, cols.unsqueeze(0).expand(positions_seq.numel(), col_pos.numel()), -1)


def _paged_cache_slots(
    block_table: torch.Tensor,
    sequence: int | torch.Tensor,
    rows: torch.Tensor,
    block_size: int,
    row_stride: int | None = None,
) -> torch.Tensor:
    columns = torch.div(rows, block_size, rounding_mode="floor") % block_table.size(1)
    blocks = block_table[sequence, columns].to(torch.int64).clamp_min(0)
    stride = block_size if row_stride is None else row_stride
    return blocks * stride + rows % stride


def _token_cache_slots(
    block_table: torch.Tensor | None,
    cache: torch.Tensor | None,
    rows: torch.Tensor,
    seqs: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if block_table is None or cache is None:
        return torch.full_like(rows, -1)
    if block_table.numel() == 0:
        return torch.full_like(rows, -1)
    block_size = cache.size(1)
    slots = _paged_cache_slots(block_table, seqs, rows, block_size)
    if mask is not None:
        slots = torch.where(mask, slots, -1)
    return slots


# Use the served context bound, rather than the larger theoretical RoPE horizon.
_GRAPH_TOKEN_CAPACITY_SERVING_CAP = 32768


def _derive_graph_token_capacity_granularity(max_model_len: int) -> int:
    if max_model_len <= 0:
        return 0
    serving_capped = min(max_model_len, _GRAPH_TOKEN_CAPACITY_SERVING_CAP)
    scaled = (serving_capped + 15) // 16
    return max(1024, ((scaled + 1023) // 1024) * 1024)


_QLI_V3_OP_NAME = "quant_lightning_indexer_v2_va"
_QLI_V3_METADATA_OP_NAME = "quant_lightning_indexer_v2_metadata_va"

_QLI_V3_QUANT_MODE = 2
_QLI_V3_HEAD_DIM = 128
_QLI_V3_METADATA_SIZE = 1024
_QLI_V3_SUPPORTED_RATIOS = (1, 2)
# Preserve FP16 scale rounding before the INT8 division.
_QLI_V3_INT8_MAX = 127.0
_QLI_V3_MIN_SCALE = 2.0**-24
_QLI_V3_KEY_NUM_HEADS = 1
# Compressed rows use right-down causality; the native default is unmasked.
_QLI_V3_MASK_MODE = 3

_QLI_V3_CAND_MODE_SOURCE = 1
_QLI_V3_CAND_MODE_CONSUMER = 2
_QLI_V3_CAND_MODE_OFF = 3
_QLI_V3_CAND_BLOCK_SIZE = 8
_QLI_V3_CAND_TOPK_BLOCKS_MAX = 2048
_QLI_V3_CAND_TOPK_BLOCKS_ALIGN = 64


_qli_v3_ops: tuple[Any, Any] | None = None


def _get_qli_v3_ops() -> tuple[Any, Any] | None:
    global _qli_v3_ops
    if _qli_v3_ops is not None:
        return _qli_v3_ops
    namespace = getattr(torch.ops, "xllm_ops", None)
    if namespace is None:
        return None
    indexer = getattr(namespace, _QLI_V3_OP_NAME, None)
    metadata = getattr(namespace, _QLI_V3_METADATA_OP_NAME, None)
    if indexer is None or metadata is None:
        return None
    if not has_op_api("aclnnQuantLightningIndexerV2Va") or not has_op_api("aclnnQuantLightningIndexerV2MetadataVa"):
        return None
    _qli_v3_ops = (indexer, metadata)
    return _qli_v3_ops


_qli_v3_active_logged = False
_qli_v3_fallback_logged = False
_qli_v3_candidate_logged = False

_DECODE_ONLY_FALLBACK_REASONS = frozenset({"not an ACL-graph decode step", "query length != 1"})


def _log_qli_v3_active() -> None:
    global _qli_v3_active_logged
    if not _qli_v3_active_logged:
        _qli_v3_active_logged = True
        logger.info("DeepSeek-V4.1 QLI v3 fused indexer active")


def _log_qli_v3_candidate_active(layer_id: int) -> None:
    global _qli_v3_candidate_logged
    if not _qli_v3_candidate_logged:
        _qli_v3_candidate_logged = True
        logger.info(f"DeepSeek-V4.1 QLI v3 candidate indexer active (layer {layer_id})")


def _log_qli_v3_fallback(reason: str) -> None:
    global _qli_v3_fallback_logged
    if _qli_v3_fallback_logged or reason in _DECODE_ONLY_FALLBACK_REASONS:
        return
    _qli_v3_fallback_logged = True
    logger.warning(f"the QLI v3 indexer path is inactive ({reason}); staying on the pure-torch indexer")


def _quantize_indexer_rows_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric INT8 quantization of the last axis.

    Matches vllm-ascend's ``quantize_indexer_query`` rounding: the FP16 scale
    is ``absmax / 127`` (floored at the smallest positive value so a zero row
    stays defined) and the codes are the half-to-even rounded normalised values
    clamped to ``[-127, 127]``. Every step is row-local/pointwise, so the helper
    is ACL-graph safe (static shapes, no host sync).

    Parity note (recorded at the definition so it governs every call site):
    this helper runs bare, exactly like vllm-ascend's upstream
    ``quantize_indexer_query`` -- no inf/NaN input defense is performed here;
    evaluating such defenses is deferred to a later round.
    """
    absmax = x.float().abs().amax(dim=-1)
    scale = (absmax / _QLI_V3_INT8_MAX).to(torch.float16).to(torch.float32).clamp_min(_QLI_V3_MIN_SCALE)
    normalized = x.float() / scale.unsqueeze(-1)
    codes = torch.round(normalized).clamp(-_QLI_V3_INT8_MAX, _QLI_V3_INT8_MAX)
    return codes.to(torch.int8), scale.to(torch.float16)


@dataclass
class _QliGraphState:
    """ACL-graph-stable operands of the QLI v3 decode path for one bucket.

    Built once during the runner's eager warmup (which also JIT-compiles the
    metadata op) and refreshed in place on every replay; the captured graph
    records these persistent addresses, so no buffer may be allocated and no
    metadata op may first run inside capture. ``capacity`` is the token-level
    committed-row bound (``graph_token_capacity``); the per-ratio compressed
    capacity is ``max(capacity // ratio, 1)``, exactly like ``_run_indexer``'s.
    The paged key is addressed by the runtime ``block_table`` plus the per-row
    ``seqused_k`` (PA_BBND), so no ``cu_seqlens_k`` operand exists.
    """

    rows: int
    capacity: int
    cu_seqlens_q: torch.Tensor
    seqused_k: dict[int, torch.Tensor]
    cmp_residual: dict[int, torch.Tensor]
    metadata: dict[int, torch.Tensor]


class Csa2AttentionBackend(DsaAttentionBackend):
    def __init__(
        self,
        compress_ratios: list[int],
        window_size: int,
        n_layers: int,
        num_heads: int,
        attn_head_dim: int,
        index_topk: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        kv_source_layer_ids: list[int],
        index_source_layer_ids: list[int],
        candidate_source_layer_id: int,
        candidate_topk_blocks: int,
        candidate_block_size: int,
        max_model_len: int = 0,
        quant_mode: str = V41_QUANT_NONE,
        acl_graph_enabled: bool = False,
    ) -> None:
        self.caches_info, self.group_infos = build_cache_specs_v41(
            compress_ratios,
            kv_source_layer_ids,
            window_size,
            n_layers,
        )
        self._builder = DsaMetadataBuilder(
            self.caches_info,
            self.group_infos,
            read_table_layout=READ_TABLE_LAYOUT_RING,
            preserve_query_padding=True,
        )
        self.window_size = window_size
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.num_heads = num_heads
        self.head_dim = attn_head_dim
        self.device = device
        self.dtype = dtype
        self.scale = attn_head_dim**-0.5

        self.index_source_layer_ids = sorted({int(source) for source in index_source_layer_ids})
        self.candidate_source_layer_id = int(candidate_source_layer_id)
        self.candidate_topk_blocks = int(candidate_topk_blocks)
        self.candidate_block_size = int(candidate_block_size)
        if self.candidate_source_layer_id >= 0 and self.candidate_source_layer_id not in self.index_source_layer_ids:
            raise ValueError(
                "DeepSeek-V4.1 candidate source layer must also be an index source layer: "
                f"{self.candidate_source_layer_id}"
            )

        self.quant_mode = quant_mode
        self.moe_graph_capturable = quant_mode == V41_QUANT_ASCEND

        self.graph_token_capacity_granularity = _derive_graph_token_capacity_granularity(max_model_len)
        granularity_source = f"derived from max_model_len={max_model_len}"
        logger.info(
            f"DeepSeek-V4.1 CSA2 graph_token_capacity_granularity="
            f"{self.graph_token_capacity_granularity} ({granularity_source})"
        )

        if acl_graph_enabled and not self.moe_graph_capturable:
            logger.warning(
                f"DeepSeek-V4.1 quant_mode={quant_mode} serves the torch MoE, which no ACL graph "
                "can capture; decode-graph admission is disabled and decode stays eager"
            )
        if acl_graph_enabled and self.moe_graph_capturable and self.graph_token_capacity_granularity <= 0:
            raise RuntimeError(
                "DeepSeek-V4.1 ACL-graph serving needs a static committed-row capacity, but "
                "max_model_len was not provided to derive one. Provide "
                "max_position_embeddings, or disable graph execution."
            )

        self._kv_caches: list[Any] = []
        # Config-derived SWA ring blocks per sequence; refreshed on bind so it is
        # a plain Python int computed before capture (never a captured tensor).
        self._swa_ring_blocks = self._compute_swa_ring_blocks()
        self._metadata: Any | None = None
        self._v41_shared: CSA2SharedRuntime | None = None
        # Persistent QLI v3 indexer operands, one entry per (batch_rows,
        # token capacity) graph bucket; built in warmup, refreshed under capture.
        self._qli_state: dict[tuple[int, int], _QliGraphState] = {}

    # -- V4 backend overrides ----------------------------------------------

    def _compute_swa_ring_blocks(self) -> int:
        """SWA ring blocks per sequence, derived from the config.

        ``get_swa_blocks_per_seq(window, block) = (window - 1) // block + 1``
        (block_utils.cpp), the value the scheduler allocates; production
        ``window == block == 128`` yields 1. The SWA page width is the cache's
        own block size; before the caches are bound fall back to
        ``window_size``, which yields the same production 1. A plain Python int
        computed before capture.
        """
        swa_block = self.window_size
        if self._kv_caches:
            swa = getattr(self._kv_caches[0], "swa", None)
            if swa is not None and swa.dim() >= 2 and int(swa.size(1)) > 0:
                swa_block = int(swa.size(1))
        if swa_block <= 0:
            return 1
        return (self.window_size - 1) // swa_block + 1

    def bind_kv_caches(self, kv_caches: list[Any]) -> None:
        super().bind_kv_caches(kv_caches)
        self._swa_ring_blocks = self._compute_swa_ring_blocks()

    def reset_forward(self, metadata: Any | None = None) -> None:
        super().reset_forward(metadata)
        self._v41_shared = None

    def _build_dsa_metadata_for_forward(self, metadata: Any) -> DsaMetadata:
        dsa = super()._build_dsa_metadata_for_forward(metadata)
        if bool(getattr(metadata, "is_dummy", False)):
            # V4's synthetic tiling context is not initialized history for tensor attention.
            dsa.seq_lens.zero_()
            dsa.actual_seq_lengths_kv.zero_()
            dsa.kv_cu_seq_lens.zero_()
            dsa.max_seqlen_kv.zero_()
            dsa.max_seq_len = 0
            dsa.start_pos.copy_(-dsa.seq_lens_q)
            for layer_slots in dsa.slot_mappings:
                for slots in layer_slots:
                    slots.fill_(-1)
        return dsa

    def refresh_dsa_metadata_for_graph_replay(self, metadata: Any | None = None) -> None:
        """Refresh DSA tensors and the QLI operand buffers outside capture.

        The inherited refresh rebuilds the DSA metadata and copies the replay
        values into the graph-captured tensors; the QLI branch rewrites its
        per-ratio sequence lengths / residuals plus the indexer's tiling
        metadata. This runs before every graph replay, so it must never
        allocate a new state, build one under capture, or read a device value
        back to the host.
        """
        super().refresh_dsa_metadata_for_graph_replay(metadata)
        resolved = self._metadata if metadata is None else metadata
        dsa = getattr(resolved, "dsa_metadata", None)
        if dsa is None:
            return
        for state in self._qli_replay_states(dsa):
            self._refresh_qli_state(dsa, state)

    def _qli_replay_states(self, dsa: DsaMetadata) -> list[_QliGraphState]:
        key = (
            int(dsa.seq_lens.numel()),
            int(getattr(dsa, "graph_token_capacity", 0) or 0),
        )
        state = self._qli_state.get(key)
        return [state] if state is not None else []

    def _build_precomputed_metadata(
        self,
        compressed_metadata: DsaMetadata,
        metadata: Any = None,
        cu_seqlens_ori_kv_override: torch.Tensor | None = None,
        max_query_len_override: int | None = None,
        max_seq_len_override: int | None = None,
    ) -> None:
        del (
            compressed_metadata,
            metadata,
            cu_seqlens_ori_kv_override,
            max_query_len_override,
            max_seq_len_override,
        )

    def _resolve_cache_mapping(self, layer_id: int, compress_ratio: int) -> _CompressedAttentionCacheMapping:
        mapping = _CompressedAttentionCacheMapping()
        if layer_id < 0 or layer_id >= len(self.caches_info):
            return mapping
        token_indices: list[int] = []
        swa_indices: list[int] = []
        for cache_idx, ci in enumerate(self.caches_info[layer_id]):
            if ci.cache_type == DSA_CACHE_TOKEN:
                token_indices.append(cache_idx)
            elif ci.cache_type == DSA_CACHE_SLIDING_WINDOW:
                swa_indices.append(cache_idx)
        if token_indices and compress_ratio > 0:
            mapping.cmp_cache_idx = token_indices[0]
        if len(token_indices) > 1:
            mapping.index_cache_idx = token_indices[1]
        if swa_indices:
            mapping.ori_cache_idx = swa_indices[0]
        if len(swa_indices) > 1:
            mapping.kv_state_cache_idx = swa_indices[1]
        if len(swa_indices) > 2:
            mapping.score_state_cache_idx = swa_indices[2]
        return mapping

    def _layer_compress_ratio(self, layer_id: int) -> int:
        caches = self.caches_info[layer_id] if 0 <= layer_id < len(self.caches_info) else []
        for ci in caches:
            if ci.cache_type == DSA_CACHE_TOKEN:
                return ci.ratio
        return 0

    def swa_group_metadata(self, dsa: DsaMetadata | None) -> _SwaGroupMetadata | None:
        """The SWA-group slot mapping / block table / cache shared by all layers.

        Every layer's first cache entry addresses the SWA group and the group
        shares one block-table/slot layout (contract section 2), so layer 0's
        view serves any consumer that needs position-stable ring slots (the
        engram hash cache).
        """
        if dsa is None or not self._kv_caches:
            return None
        mapping = self._resolve_cache_mapping(0, 0)
        layer_cache = self._kv_caches[0]
        if layer_cache.swa is None or mapping.ori_cache_idx < 0:
            return None
        slot_mapping = _get_layer_cache_tensor(dsa.slot_mappings, 0, mapping.ori_cache_idx)
        block_table = _get_layer_cache_tensor(dsa.block_tables, 0, mapping.ori_cache_idx)
        if slot_mapping is None or block_table is None:
            return None
        return _SwaGroupMetadata(
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=layer_cache.swa.size(1),
            cache=layer_cache.swa,
        )

    def execute_csa2_layer(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
    ) -> torch.Tensor:
        metadata = self._current_forward_metadata()
        dsa = getattr(metadata, "dsa_metadata", None)
        if dsa is None:
            raise RuntimeError("DeepSeek-V4.1 requires prepared DSA metadata before attention")
        mode = layer.mode
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        mapping = self._resolve_cache_mapping(layer_id, mode.compress_ratio)
        ctx = layer._csa2_ctx
        if ctx is None:
            raise RuntimeError(f"DeepSeek-V4.1 layer {layer_id} ran without its CSA2 context")
        cp_runtime = get_current_dsa_cp_runtime() if getattr(layer, "enable_dsa_cp", False) else None
        cp_context = cp_runtime.plan if cp_runtime is not None else None
        if cp_context is not None:
            q_lens = list(cp_context.global_q_lens)
        elif dsa.is_acl_graph:
            # Decode has one row per sequence; use its static extent to avoid D2H inside capture.
            q_lens = [1] * dsa.seq_lens_q.numel()
        else:
            q_lens = dsa.seq_lens_q_host
        ctx_lens = dsa.seq_lens

        shared = self._shared_runtime()
        if mode.is_full:
            self._run_compressor(layer, ctx, dsa, layer_cache, mapping, q_lens, ctx_lens, shared)
        if cp_runtime is not None:
            ctx_lens = cp_runtime.local_context_lengths
            q_lens = list(cp_context.local_q_lens)
            ctx = replace(
                ctx,
                x=cp_context.shard(ctx.x),
                qr=cp_context.shard(ctx.qr),
                positions=cp_context.shard(ctx.positions),
            )
            # DSA metadata remains global for cache layout. Query consumers
            # receive local q_lens/ctx_lens explicitly.
        # 3) Indexer: Full/Reindex layers compute a fresh top-k; Reuse layers
        #    consume the tensor object their group source published.
        if mode.is_full or mode.is_reindex:
            topk = self._run_indexer(layer, ctx, dsa, q_lens, ctx_lens, shared)
        elif mode.is_reuse:
            topk = shared.topk_idxs
            if topk is None and dsa.block_tables:
                # Real batches always publish the group source's top-k;
                # profiling batches (empty block tables) degrade to the
                # plain window path.
                raise RuntimeError(
                    f"DeepSeek-V4.1 reuse layer {layer_id} has no shared top-k; its group source must run first"
                )
        else:
            topk = None
        layer._csa2_used_topk = topk
        # Current-chunk KV is read directly; preserve the ring history until
        # all queries have consumed it, even when the chunk wraps the ring.
        output = self._csa2_attention(q, kv, layer, dsa, q_lens, ctx_lens, topk, shared, cp_context)
        swa_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.ori_cache_idx)
        if layer_cache.swa is not None and swa_slot is not None and swa_slot.numel() > 0:
            if not dsa.is_acl_graph:
                swa_slot, kv = last_ring_writes(swa_slot, kv, slot_capacity=layer_cache.swa.numel() // kv.size(-1))
            _scatter_by_slot(layer_cache.swa, swa_slot, kv)
        return output

    # -- compressor ---------------------------------------------------------

    def _run_compressor(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        layer_cache: Any,
        mapping: _CompressedAttentionCacheMapping,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        shared: CSA2SharedRuntime,
    ) -> None:
        ratio = layer.mode.compress_ratio
        layer_id = layer.layer_id
        kv_state = layer_cache.compress_kv_state
        score_state = layer_cache.compress_score_state
        kv_state_flat = kv_state.view(-1, layer_cache.swa.size(-1)) if kv_state is not None else None
        score_state_flat = score_state.view(-1, layer_cache.swa.size(-1)) if score_state is not None else None
        state_rows = kv_state.size(1) if kv_state is not None else 0

        latent_rows: list[torch.Tensor] = []
        group_starts: list[torch.Tensor] = []
        slot_rows: list[torch.Tensor] = []
        slot_seqs: list[torch.Tensor] = []
        slot_masks: list[torch.Tensor] = []
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            ctx_len = ctx_lens[seq]
            prev_ctx = ctx_len - q_len
            # Padded graph rows (ctx <= 0) stay fully inert: no parking, no
            # commit, so a garbage block-table entry can never be written.
            seq_valid = ctx_len > 0
            x_seq = ctx.x[token_offset : token_offset + q_len]
            rows = torch.arange(q_len, device=ctx.x.device)
            # Fixed commit capacity: a chunk of q_len tokens completes at most
            # q_len // ratio + 1 groups (the group straddling each boundary).
            capacity = q_len // ratio + 1
            g = torch.arange(capacity, device=ctx.x.device)
            group_idx = torch.div(prev_ctx, ratio, rounding_mode="floor") + g
            commit_mask = g < (
                torch.div(ctx_len, ratio, rounding_mode="floor") - torch.div(prev_ctx, ratio, rounding_mode="floor")
            )
            commit_mask = commit_mask & seq_valid
            if ratio == 1:
                kv_seq, _ = layer.compress_kv_score(x_seq)
                latent_rows.append(layer.compress_rmsnorm(kv_seq))
                group_starts.append(ctx.positions[token_offset : token_offset + q_len].to(torch.int64))
                rows_segment = group_idx[:q_len]
                mask_segment = commit_mask[:q_len]
            else:
                kv_seq, score_seq = layer.compress_kv_score(x_seq)
                # Fused Triton pooling: pool the committed groups at their
                # completion-token rows and
                # update the state ring in place. The op consumes host control
                # metadata and the pre-capture warmup forward JIT-compiles it, so
                # it runs in both eager and ACL-graph capture; the reference torch
                # path below (same ring-32 addressing) is the fallback.
                reason = _v41_compressor_triton_reason(ratio, kv_state, score_state)
                triton_pooled = None
                if reason is None:
                    if any(int(q) != 1 for q in q_lens):
                        # The op is validated on the single-token decode rows
                        # only; a q_len > 1 row (prefill / speculative chunk) is
                        # unverified, so it keeps the torch path.
                        reason = "non-decode row"
                    else:
                        # Resolve state from the current absolute position,
                        # just like the torch path. SWA releases old blocks;
                        # column zero can be invalid while later blocks live.
                        block_table, reason = _v41_compressor_swa_gate(
                            dsa, layer_id, mapping.ori_cache_idx, self._swa_ring_blocks
                        )
                        if block_table is not None and reason is None:
                            triton_pooled = _v41_compressor_pool_triton(
                                kv_seq,
                                score_seq,
                                kv_state,
                                score_state,
                                start_pos=prev_ctx,
                                used_len=q_len,
                                block_id=torch.div(
                                    self._v41_state_slots(
                                        dsa, layer_id, mapping.ori_cache_idx, seq, prev_ctx.reshape(1), state_rows
                                    ),
                                    state_rows,
                                    rounding_mode="floor",
                                ),
                            )
                            if triton_pooled is None:
                                reason = "op declined"
                if reason is not None:
                    _log_v41_compressor_fallback(reason)
                if triton_pooled is not None:
                    _log_v41_compressor_active()
                    offsets = group_idx * ratio + (ratio - 1) - prev_ctx
                    pooled_rows = triton_pooled[offsets.clamp(0, q_len - 1)]
                    latent_rows.append(layer.compress_rmsnorm(pooled_rows.to(ctx.x.dtype)))
                    group_starts.append((group_idx * ratio).clamp(0, ctx.cos_sin_cache.size(0) - 1))
                    rows_segment = group_idx
                    mask_segment = commit_mask
                else:
                    # Read residuals before parking this chunk: a long chunk can overwrite the same ring slot.
                    pos = prev_ctx + rows
                    park_mask = ((pos + 1) % ratio != 0) & seq_valid
                    # Only the final state_rows tokens can survive as history; keep one writer per ring slot.
                    tail_start = max(0, q_len - state_rows)
                    park_mask = park_mask & (rows >= tail_start)
                    member_pos = group_idx.unsqueeze(-1) * ratio + torch.arange(ratio, device=ctx.x.device)
                    in_chunk = (member_pos >= prev_ctx).unsqueeze(-1)
                    member_row = (member_pos - prev_ctx).clamp(0, q_len - 1)
                    if kv_state_flat is not None and score_state_flat is not None:
                        parked_slot = self._v41_state_slots(
                            dsa, layer_id, mapping.ori_cache_idx, seq, member_pos.clamp_min(0), state_rows
                        )
                        member_kv = torch.where(in_chunk, kv_seq[member_row], kv_state_flat[parked_slot])
                        member_score = torch.where(in_chunk, score_seq[member_row], score_state_flat[parked_slot])
                    else:
                        member_kv = kv_seq[member_row]
                        member_score = score_seq[member_row]
                    gate = member_score.softmax(dim=1)
                    pooled = (member_kv * gate).sum(dim=1)
                    if kv_state_flat is not None and score_state_flat is not None:
                        # State planes use the state-ring stride, not the SWA page stride.
                        park_slot = torch.where(
                            park_mask,
                            self._v41_state_slots(dsa, layer_id, mapping.ori_cache_idx, seq, pos, state_rows),
                            -1,
                        )
                        _scatter_by_slot(kv_state_flat, park_slot, kv_seq)
                        _scatter_by_slot(score_state_flat, park_slot, score_seq)
                    latent_rows.append(layer.compress_rmsnorm(pooled.to(ctx.x.dtype)))
                    # Masked padding still passes through RoPE and needs a non-negative position.
                    group_starts.append((group_idx * ratio).clamp(0, ctx.cos_sin_cache.size(0) - 1))
                    rows_segment = group_idx
                    mask_segment = commit_mask
            slot_rows.append(rows_segment)
            slot_seqs.append(torch.full_like(rows_segment, seq))
            slot_masks.append(mask_segment)
            token_offset += q_len

        index_block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.index_cache_idx)
        key_block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.cmp_cache_idx)
        if latent_rows:
            latent = torch.cat(latent_rows, dim=0)
            all_starts = torch.cat(group_starts, dim=0)
            all_slot_rows = torch.cat(slot_rows, dim=0)
            all_slot_seqs = torch.cat(slot_seqs, dim=0)
            all_slot_mask = torch.cat(slot_masks, dim=0)
            # The indexer consumes the compressor latent before RoPE.
            if layer.indexer is not None and layer.indexer.owns_k:
                index_k = layer.indexer_key(latent)
                index_k = apply_interleaved_rope(index_k, all_starts, ctx.cos_sin_cache)
                if layer_cache.index is not None:
                    index_slots = _token_cache_slots(
                        index_block_table,
                        layer_cache.index,
                        all_slot_rows,
                        all_slot_seqs,
                        all_slot_mask,
                    )
                    index_codes, index_scales = _quantize_indexer_rows_int8(index_k)
                    _scatter_by_slot(layer_cache.index, index_slots, index_codes)
                    if layer_cache.indexer_scale is not None:
                        _scatter_by_slot(
                            layer_cache.indexer_scale,
                            index_slots,
                            index_scales.unsqueeze(-1),
                        )
            rotated = apply_interleaved_rope(latent, all_starts, ctx.cos_sin_cache)
            if layer_cache.key is not None:
                key_slots = _token_cache_slots(
                    key_block_table,
                    layer_cache.key,
                    all_slot_rows,
                    all_slot_seqs,
                    all_slot_mask,
                )
                _scatter_by_slot(layer_cache.key, key_slots, rotated)
        shared.key_cache = layer_cache.key
        shared.key_block_table = key_block_table
        shared.index_cache = layer_cache.index
        shared.index_block_table = index_block_table
        shared.index_scale = layer_cache.indexer_scale
        shared.compress_ratio = ratio

    def _run_indexer(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata | None,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        shared: CSA2SharedRuntime,
    ) -> torch.Tensor | None:
        layer_id = layer.layer_id
        ratio = shared.compress_ratio
        if ratio <= 0 or shared.index_cache is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 indexer for layer {layer_id} has no shared index cache; "
                "its group kv source must run first"
            )
        if shared.index_block_table is None:
            return None
        if ctx.qr.size(0) == 0:
            topk = torch.empty((len(q_lens), 0, 0), dtype=torch.int32, device=ctx.x.device)
            shared.topk_idxs = topk
            shared.topk_layer = layer_id
            if layer_id == self.candidate_source_layer_id:
                shared.candidates = [None] * len(q_lens)
                shared.candidates_layer = layer_id
            return topk
        index_cache = shared.index_cache
        block_table = shared.index_block_table
        block_size = index_cache.size(1)
        capacity = self._committed_capacity(dsa, ratio, ctx_lens)

        q_idx = layer.indexer_query(ctx.qr)
        q_idx = apply_interleaved_rope(q_idx, ctx.positions, ctx.cos_sin_cache)
        qli_reason = self._qli_v3_fallback_reason(layer, dsa, q_lens, shared, index_cache)
        if qli_reason is None:
            _log_qli_v3_active()
            return self._run_indexer_qli_v3(
                layer,
                q_idx,
                ctx,
                dsa,
                q_lens,
                shared,
                block_table,
                capacity,
            )
        _log_qli_v3_fallback(qli_reason)
        # Match the fused indexer's INT8 query domain in the eager path.
        q_codes, q_scales = _quantize_indexer_rows_int8(q_idx)
        q_idx = q_codes.float() * q_scales.unsqueeze(-1).float()
        weights = layer.indexer_weights(ctx.x)

        candidates_per_seq: list[torch.Tensor | None] | None = None
        if layer_id == self.candidate_source_layer_id:
            candidates_per_seq = [None] * len(q_lens)
        rows_per_seq: list[torch.Tensor] = []
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            if q_len == 0:
                rows_per_seq.append(torch.empty((0, 0), dtype=torch.int32, device=q_idx.device))
                continue
            q_seq = q_idx[token_offset : token_offset + q_len]
            w_seq = weights[token_offset : token_offset + q_len]
            positions_seq = ctx.positions[token_offset : token_offset + q_len].to(torch.int64)
            ctx_len = ctx_lens[seq]
            committed = torch.div(ctx_len, ratio, rounding_mode="floor")
            if capacity <= 0:
                rows_per_seq.append(torch.full((q_len, 0), -1, dtype=torch.int32, device=q_idx.device))
                token_offset += q_len
                continue
            index_k = self._gather_cache_rows(index_cache, block_table, seq, capacity, block_size)
            index_scales = self._gather_cache_rows(shared.index_scale, block_table, seq, capacity, block_size)
            index_k = index_k.float() * index_scales.float()
            visible = torch.div(positions_seq + 1, ratio, rounding_mode="floor")
            t_range = torch.arange(capacity, device=q_idx.device)
            # Compressed indices follow the fixed [ring candidates; chunk] region.
            offset = self.window_size + q_len
            row_tiles: list[torch.Tensor] = []
            pool_tiles: list[torch.Tensor] = []
            for tile_start in range(0, q_len, _INDEXER_QUERY_TILE):
                tile_end = min(tile_start + _INDEXER_QUERY_TILE, q_len)
                score = _indexer_score(q_seq[tile_start:tile_end], index_k, w_seq[tile_start:tile_end])
                tile_visible = visible[tile_start:tile_end]
                score.masked_fill_(t_range >= tile_visible.unsqueeze(-1), float("-inf"))
                score.masked_fill_(t_range >= committed.unsqueeze(-1), float("-inf"))
                if layer_id == self.candidate_source_layer_id:
                    pool_tiles.append(
                        _select_candidate_blocks(
                            score,
                            tile_visible,
                            self.candidate_topk_blocks,
                            self.candidate_block_size,
                        )
                    )
                elif 0 <= self.candidate_source_layer_id < layer_id:
                    if shared.candidates is None or seq >= len(shared.candidates):
                        raise RuntimeError(
                            f"DeepSeek-V4.1 reindex layer {layer_id} requires the candidate pool "
                            f"from layer {self.candidate_source_layer_id}"
                        )
                    pool = shared.candidates[seq]
                    if pool is None:
                        raise RuntimeError(
                            f"DeepSeek-V4.1 reindex layer {layer_id} has no candidate pool for sequence {seq}"
                        )
                    score.masked_fill_(~pool[tile_start:tile_end], float("-inf"))
                row_tiles.append(_indexer_topk(score, tile_visible, self.index_topk, offset))
            if pool_tiles and candidates_per_seq is not None:
                candidates_per_seq[seq] = torch.cat(pool_tiles, dim=0)
            if row_tiles:
                rows_per_seq.append(row_tiles[0] if len(row_tiles) == 1 else torch.cat(row_tiles, dim=0))
            else:
                rows_per_seq.append(torch.full((q_len, 0), -1, dtype=torch.int32, device=q_idx.device))
            token_offset += q_len

        seq_count = len(q_lens)
        s_max = max(q_lens) if q_lens else 0
        k_width = min(self.index_topk, capacity) if capacity > 0 else 0
        topk = torch.full((seq_count, s_max, k_width), -1, dtype=torch.int32, device=q_idx.device)
        for seq, rows in enumerate(rows_per_seq):
            if rows.numel() > 0:
                topk[seq, : rows.size(0), : rows.size(1)] = rows
        shared.topk_idxs = topk
        shared.topk_layer = layer_id
        if candidates_per_seq is not None:
            shared.candidates = candidates_per_seq
            shared.candidates_layer = layer_id
        return topk

    def _qli_v3_fallback_reason(
        self,
        layer: Any,
        dsa: DsaMetadata,
        q_lens: list[int],
        shared: CSA2SharedRuntime,
        index_cache: torch.Tensor | None,
    ) -> str | None:
        acl_graph = bool(getattr(dsa, "is_acl_graph", False))
        prefill_step = any(int(length) != 1 for length in q_lens)
        if not acl_graph and not prefill_step:
            return "not an ACL-graph decode step"
        if acl_graph and prefill_step:
            return "query length != 1"
        if self.index_head_dim != _QLI_V3_HEAD_DIM:
            return f"unsupported index head dim {self.index_head_dim} (op requires {_QLI_V3_HEAD_DIM})"
        if int(shared.compress_ratio) not in _QLI_V3_SUPPORTED_RATIOS:
            return f"unsupported compression ratio {shared.compress_ratio}"
        bucketed_capacity = int(getattr(dsa, "graph_token_capacity", 0) or 0)
        if not prefill_step and bucketed_capacity <= 0:
            return "missing static graph token capacity"
        is_candidate_layer = layer.layer_id == self.candidate_source_layer_id or self._uses_candidates(layer.layer_id)
        if is_candidate_layer:
            config_reason = self._qli_v3_candidate_config_reason()
            if config_reason is not None:
                return config_reason
        index_reason = self._qli_index_plane_ready(shared, index_cache)
        if index_reason is not None:
            return index_reason
        if _get_qli_v3_ops() is None:
            return "quant_lightning_indexer_v2_va / quant_lightning_indexer_v2_metadata_va are not registered"
        return None

    def _qli_index_plane_ready(
        self,
        shared: CSA2SharedRuntime,
        index_cache: torch.Tensor | None,
    ) -> str | None:
        key = index_cache if index_cache is not None else shared.index_cache
        scale = shared.index_scale
        if key is None or scale is None:
            missing = "key" if key is None else "scale"
            return f"index INT8 plane not ready ({missing} is None)"
        if key.dtype != torch.int8 or scale.dtype != torch.float16:
            return f"index plane dtype mismatch (key {key.dtype}, scale {scale.dtype})"
        if key.dim() != 4 or int(key.size(2)) != _QLI_V3_KEY_NUM_HEADS:
            return f"index key must be [blocks, tokens, {_QLI_V3_KEY_NUM_HEADS}, head] (got {tuple(key.shape)})"
        if scale.dim() != 3 or int(scale.size(2)) != _QLI_V3_KEY_NUM_HEADS:
            return f"index scale must be [blocks, tokens, {_QLI_V3_KEY_NUM_HEADS}] (got {tuple(scale.shape)})"
        if int(key.size(3)) != self.index_head_dim:
            return f"index key head dim {int(key.size(3))} does not match {self.index_head_dim}"
        if int(scale.size(0)) != int(key.size(0)) or int(scale.size(1)) != int(key.size(1)):
            return f"index scale geometry {tuple(scale.shape)} does not match the index key {tuple(key.shape)}"
        if shared.index_block_table is None:
            return "index INT8 PA_BBND needs the TOKEN-group block table"
        return None

    def _qli_v3_candidate_config_reason(self) -> str | None:
        if self.candidate_block_size != _QLI_V3_CAND_BLOCK_SIZE:
            return f"candidate block size {self.candidate_block_size} (op requires {_QLI_V3_CAND_BLOCK_SIZE})"
        if (
            self.candidate_topk_blocks <= 0
            or self.candidate_topk_blocks > _QLI_V3_CAND_TOPK_BLOCKS_MAX
            or self.candidate_topk_blocks % _QLI_V3_CAND_TOPK_BLOCKS_ALIGN != 0
        ):
            return (
                f"candidate topk blocks {self.candidate_topk_blocks} (op requires a "
                f"positive multiple of {_QLI_V3_CAND_TOPK_BLOCKS_ALIGN}, "
                f"<= {_QLI_V3_CAND_TOPK_BLOCKS_MAX})"
            )
        return None

    def _qli_state_for(self, dsa: DsaMetadata) -> _QliGraphState:
        token_capacity = int(getattr(dsa, "graph_token_capacity", 0) or 0)
        if token_capacity <= 0:
            seq_lens = dsa.seq_lens
            token_capacity = int(seq_lens.max().item()) if seq_lens is not None and seq_lens.numel() > 0 else 0
        key = (
            int(dsa.seq_lens.numel()),
            token_capacity,
        )
        state = self._qli_state.get(key)
        if state is not None:
            return state
        if _is_stream_capturing():
            raise RuntimeError(
                "DeepSeek-V4.1 QLI v3 state must be built in the eager warmup, before "
                f"ACL graph capture; it is missing for bucket (rows={key[0]}, capacity={key[1]})"
            )
        state = self._build_qli_state(dsa, key)
        self._qli_state[key] = state
        return state

    def _build_qli_state(self, dsa: DsaMetadata, key: tuple[int, int]) -> _QliGraphState:
        rows, capacity = key
        device = dsa.seq_lens.device
        state = _QliGraphState(
            rows=rows,
            capacity=capacity,
            cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device=device),
            seqused_k={},
            cmp_residual={},
            metadata={},
        )
        for ratio in _QLI_V3_SUPPORTED_RATIOS:
            state.seqused_k[ratio] = torch.zeros(rows, dtype=torch.int32, device=device)
            state.cmp_residual[ratio] = torch.zeros(rows, dtype=torch.int32, device=device)
            state.metadata[ratio] = torch.zeros(_QLI_V3_METADATA_SIZE, dtype=torch.int32, device=device)
        self._refresh_qli_state(dsa, state)
        return state

    def _refresh_qli_state(self, dsa: DsaMetadata, state: _QliGraphState) -> None:
        seq_lens = dsa.seq_lens
        for ratio in state.metadata:
            state.seqused_k[ratio].copy_(torch.div(seq_lens, ratio, rounding_mode="floor"))
            state.cmp_residual[ratio].copy_(torch.remainder(seq_lens, ratio))
            self._fill_qli_metadata(state, ratio)

    def _fill_qli_metadata(self, state: _QliGraphState, ratio: int) -> None:
        ops = _get_qli_v3_ops()
        if ops is None:
            return
        _, metadata_op = ops
        cap = max(state.capacity // ratio, 1)
        value = metadata_op(
            self.index_n_heads,
            1,
            self.index_head_dim,
            min(self.index_topk, cap),
            _QLI_V3_QUANT_MODE,
            cu_seqlens_q=state.cu_seqlens_q,
            # PA_BBND derives K offsets from the block table; cu_seqlens_k must be absent.
            cu_seqlens_k=None,
            seqused_k=state.seqused_k[ratio],
            # Ratio 1 has no residual operand; the native contract requires None.
            cmp_residual_k=state.cmp_residual[ratio] if ratio > 1 else None,
            batch_size=state.rows,
            max_seqlen_q=1,
            max_seqlen_k=cap,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=_QLI_V3_MASK_MODE,
            cmp_ratio=ratio,
            device=str(state.cu_seqlens_q.device),
        )
        state.metadata[ratio].copy_(value)

    def _qli_candidate_plan(self, layer: Any, shared: CSA2SharedRuntime) -> tuple[int, dict[str, Any]]:
        candidate_mode = _QLI_V3_CAND_MODE_OFF
        candidate_kwargs: dict[str, Any] = {}
        if self.candidate_source_layer_id >= 0:
            if layer.layer_id == self.candidate_source_layer_id:
                candidate_mode = _QLI_V3_CAND_MODE_SOURCE
            elif self._uses_candidates(layer.layer_id):
                candidate_mode = _QLI_V3_CAND_MODE_CONSUMER
                pool = shared.candidate_blocks
                if (
                    pool is None
                    or int(pool.numel()) == 0
                    or shared.candidate_blocks_layer != self.candidate_source_layer_id
                ):
                    raise RuntimeError(
                        f"DeepSeek-V4.1 reindex layer {layer.layer_id} requires the candidate "
                        f"pool from layer {self.candidate_source_layer_id} (this forward "
                        f"published candidate_blocks_layer={shared.candidate_blocks_layer})"
                    )
                candidate_kwargs["candidate_topk_index"] = pool
            if candidate_mode != _QLI_V3_CAND_MODE_OFF:
                candidate_kwargs["candidate_topk_blocks"] = self.candidate_topk_blocks
                candidate_kwargs["candidate_block_size"] = self.candidate_block_size
        return candidate_mode, candidate_kwargs

    def _publish_qli_candidate(
        self, layer: Any, shared: CSA2SharedRuntime, candidate_mode: int, candidate_out: torch.Tensor
    ) -> None:
        if candidate_mode != _QLI_V3_CAND_MODE_OFF:
            _log_qli_v3_candidate_active(layer.layer_id)
        if candidate_mode == _QLI_V3_CAND_MODE_SOURCE:
            shared.candidate_blocks = candidate_out
            shared.candidate_blocks_layer = layer.layer_id

    def _run_indexer_qli_v3(
        self,
        layer: Any,
        q_idx: torch.Tensor,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        q_lens: list[int],
        shared: CSA2SharedRuntime,
        block_table: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        if any(int(length) != 1 for length in q_lens):
            return self._run_indexer_qli_v3_prefill(layer, q_idx, ctx, dsa, q_lens, shared, block_table, capacity)
        ops = _get_qli_v3_ops()
        indexer, _ = ops
        ratio = int(shared.compress_ratio)
        state = self._qli_state_for(dsa)
        topk_width = min(self.index_topk, capacity)
        key = shared.index_cache
        key_scale = shared.index_scale
        query_int8, query_scale = _quantize_indexer_rows_int8(q_idx)
        weights = layer.indexer_weights(ctx.x).to(torch.float16)
        candidate_mode, candidate_kwargs = self._qli_candidate_plan(layer, shared)
        sparse_indices, _, candidate_out = indexer(
            query_int8.contiguous(),
            key,
            weights.contiguous(),
            query_scale.contiguous(),
            key_scale,
            topk_width,
            _QLI_V3_QUANT_MODE,
            block_table=block_table,
            cu_seqlens_q=state.cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_k=state.seqused_k[ratio],
            cmp_residual_k=state.cmp_residual[ratio] if ratio > 1 else None,
            metadata=state.metadata[ratio],
            max_seqlen_q=1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=_QLI_V3_MASK_MODE,
            cmp_ratio=ratio,
            candidate_mode=candidate_mode,
            **candidate_kwargs,
        )
        self._publish_qli_candidate(layer, shared, candidate_mode, candidate_out)
        offset = self.window_size + 1
        picks = sparse_indices[:, 0, :]
        visible = state.seqused_k[ratio].unsqueeze(-1)
        rows_out = _rebase_qli_picks(picks, visible, offset, fp32_sort=capacity <= _QLI_V3_REBASE_FP32_SENTINEL)
        topk = rows_out.unsqueeze(1)
        shared.topk_idxs = topk
        shared.topk_layer = layer.layer_id
        return topk

    def _run_indexer_qli_v3_prefill(
        self,
        layer: Any,
        q_idx: torch.Tensor,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        q_lens: list[int],
        shared: CSA2SharedRuntime,
        block_table: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        ops = _get_qli_v3_ops()
        indexer, metadata_op = ops
        ratio = int(shared.compress_ratio)
        device = q_idx.device
        seq_count = len(q_lens)
        total_q = sum(int(length) for length in q_lens)
        s_max = max(int(length) for length in q_lens) if q_lens else 0
        k_width = min(self.index_topk, capacity) if capacity > 0 else 0
        topk = torch.full((seq_count, s_max, k_width), -1, dtype=torch.int32, device=device)
        if capacity <= 0 or total_q == 0 or k_width == 0:
            shared.topk_idxs = topk
            shared.topk_layer = layer.layer_id
            return topk

        key = shared.index_cache
        key_scale = shared.index_scale

        ctx_lens = dsa.seq_lens
        offsets = [0]
        for length in q_lens:
            offsets.append(offsets[-1] + int(length))
        cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
        committed = torch.div(ctx_lens, ratio, rounding_mode="floor").to(torch.int32)
        cmp_residual = torch.remainder(ctx_lens, ratio).to(torch.int32) if ratio > 1 else None
        metadata_residual = cmp_residual
        max_q = s_max
        metadata = metadata_op(
            self.index_n_heads,
            1,
            self.index_head_dim,
            k_width,
            _QLI_V3_QUANT_MODE,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_k=committed,
            cmp_residual_k=metadata_residual,
            batch_size=seq_count,
            max_seqlen_q=max_q,
            max_seqlen_k=capacity,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=_QLI_V3_MASK_MODE,
            cmp_ratio=ratio,
            device=str(device),
        )
        query_int8, query_scale = _quantize_indexer_rows_int8(q_idx)
        weights = layer.indexer_weights(ctx.x).to(torch.float16)
        candidate_mode, candidate_kwargs = self._qli_candidate_plan(layer, shared)
        sparse_indices, _, candidate_out = indexer(
            query_int8.contiguous(),
            key,
            weights.contiguous(),
            query_scale.contiguous(),
            key_scale,
            k_width,
            _QLI_V3_QUANT_MODE,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_k=committed,
            cmp_residual_k=cmp_residual,
            metadata=metadata,
            max_seqlen_q=max_q,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=_QLI_V3_MASK_MODE,
            cmp_ratio=ratio,
            candidate_mode=candidate_mode,
            **candidate_kwargs,
        )
        self._publish_qli_candidate(layer, shared, candidate_mode, candidate_out)
        picks = sparse_indices[:, 0, :]
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            q_len = int(q_len)
            if q_len <= 0:
                continue
            segment = picks[token_offset : token_offset + q_len]
            positions_seq = ctx.positions[token_offset : token_offset + q_len].to(torch.int64)
            visible = torch.div(positions_seq + 1, ratio, rounding_mode="floor").unsqueeze(-1)
            rows_out = _rebase_qli_picks(
                segment,
                visible,
                self.window_size + q_len,
                fp32_sort=True,
            )
            width = min(int(rows_out.size(1)), k_width)
            if width > 0:
                topk[seq, :q_len, :width] = rows_out[:, :width]
            token_offset += q_len
        shared.topk_idxs = topk
        shared.topk_layer = layer.layer_id
        return topk

    def _uses_candidates(self, layer_id: int) -> bool:
        return 0 <= self.candidate_source_layer_id < layer_id

    def _csa2_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
        dsa: DsaMetadata,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        topk: torch.Tensor | None,
        shared: CSA2SharedRuntime,
        cp_context: DsaCpContext | None = None,
    ) -> torch.Tensor:
        mode = layer.mode
        layer_id = layer.layer_id
        ratio = mode.compress_ratio
        win = self.window_size
        head_dim = q.size(-1)
        # A missing sink adds nothing to the normalizer; the zero parameter is a loader placeholder.
        sinks = layer.attn_sink.to(torch.float32) if getattr(layer, "attn_sink_loaded", False) else None
        layer_cache = self._kv_caches[layer_id]
        mapping = self._resolve_cache_mapping(layer_id, ratio)
        swa_bt = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.ori_cache_idx)
        swa_bs = layer_cache.swa.size(1) if layer_cache.swa is not None else win
        global_q_lens = list(cp_context.global_q_lens) if cp_context is not None else q_lens
        global_ctx_lens = dsa.seq_lens if cp_context is not None else ctx_lens

        # TODO(smla-fused): integrate the SMLA fused decode tail (sparse_flash_mla
        # + metadata via the ops-transformer mainline registration) in a follow-up
        # PR; this branch keeps the pure-torch path only.
        swa_flat = layer_cache.swa.view(-1, head_dim) if layer_cache.swa is not None else None
        capacity = 0
        if q_lens and ratio > 0 and topk is not None and shared.key_cache is not None:
            capacity = self._committed_capacity(dsa, ratio, ctx_lens)
        compressed_rows = torch.arange(capacity, device=q.device) if capacity > 0 else None

        out = torch.empty_like(q)
        token_offset = 0
        global_token_offset = 0
        for seq, q_len in enumerate(q_lens):
            global_q_len = global_q_lens[seq]
            kv_seq = kv[global_token_offset : global_token_offset + global_q_len]
            global_token_offset += global_q_len
            if q_len == 0:
                continue
            ctx_len = ctx_lens[seq]
            prev_ctx = ctx_len - q_len
            global_prev_ctx = prev_ctx if cp_context is None else global_ctx_lens[seq] - global_q_len
            q_seq = q[token_offset : token_offset + q_len]
            # Window rows at fixed capacity: [win ring candidates ; chunk].
            # Ring candidate c sits at absolute position prev - win + c; which
            # candidates a given query may read is decided per query below.
            ring_pos = (global_prev_ctx - win) + torch.arange(win, device=q.device)
            if swa_flat is not None and swa_bt is not None:
                ring_rows = self._gather_swa_positions(swa_flat, swa_bt, seq, ring_pos.clamp_min(0), swa_bs)
            else:
                # Profiling / warmup batches carry no block tables: keep the
                # traced shapes intact with zeros; the mask below drops them.
                ring_rows = q.new_zeros((win, kv_seq.size(-1)))
            # Unwritten rows can contain NaN/Inf. Score masking alone still
            # lets zero probabilities multiply those values in the reduction.
            # The gather owns this storage; masking does not change the cache.
            ring_rows.masked_fill_((ring_pos < 0).unsqueeze(-1), 0)
            kv_seq = torch.where(ctx_len > 0, kv_seq, 0)
            n_w = ring_rows.size(0) + kv_seq.size(0)
            # Window index matrix: query p (position prev+i) sees exactly the
            # window rows whose absolute position lies in
            # [max(p - win + 1, 0), p]; the lower bound tightens with the query
            # (only the chunk's first token shares the chunk-level bound).
            positions_seq = prev_ctx + torch.arange(q_len, device=q.device)
            chunk_positions = (
                positions_seq if cp_context is None else global_prev_ctx + torch.arange(global_q_len, device=q.device)
            )
            col_pos = torch.cat((ring_pos, chunk_positions), dim=0)
            cols = torch.arange(n_w, device=q.device)
            window_idxs = _window_visible_idxs(positions_seq, col_pos, cols, win)

            # Compressed rows for the top-k picks (shared group cache),
            # gathered at the committed capacity; padding rows are unreachable
            # because every top-k pick past `committed` is -1.
            compress_kv = None
            if capacity > 0:
                shared_key_cache = shared.key_cache
                shared_key_table = shared.key_block_table
                if shared_key_cache is None or shared_key_table is None:
                    raise RuntimeError(
                        f"DeepSeek-V4.1 layer {layer_id} has no shared key cache for its "
                        f"compressed rows; its group kv source must run first"
                    )
                compress_kv = self._gather_cache_rows(
                    shared_key_cache,
                    shared_key_table,
                    seq,
                    capacity,
                    shared_key_cache.size(1),
                )
                committed = torch.div(ctx_len, ratio, rounding_mode="floor")
                uncommitted_rows = compressed_rows >= committed
                compress_kv.masked_fill_(uncommitted_rows.unsqueeze(-1), 0)
            if compress_kv is not None:
                kv_cat = torch.cat((ring_rows, kv_seq, compress_kv), dim=0)
                compressed_idxs = topk[seq][:q_len]
                if cp_context is not None:
                    compressed_idxs = torch.where(
                        compressed_idxs >= 0, compressed_idxs + global_q_len - q_len, compressed_idxs
                    )
                idxs = torch.cat((window_idxs, compressed_idxs), dim=-1)
            else:
                kv_cat = torch.cat((ring_rows, kv_seq), dim=0)
                idxs = window_idxs
            kv_cat_f = kv_cat.to(torch.float32)
            # Empty/padded DP-shard rows carry a context length but no written
            # SWA history, so the gathered ring rows read uninitialized HBM and
            # come back as NaN/Inf or huge *finite* garbage. Clear the
            # non-finite entries and bound everything to the magnitude of the
            # always-written current-token rows (kv_seq): a real row's history
            # is in range already (no-op for it), a fake row becomes finite
            # instead of poisoning the softmax and the DP-gathered MoE. Derive
            # the bound from a *sanitized* kv_seq so a non-finite
            # current-token row cannot make the bound itself NaN.
            kv_seq_f = torch.nan_to_num(kv_seq.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            bound = kv_seq_f.abs().amax().clamp_min(1.0) * 32.0
            kv_cat_f = torch.nan_to_num(kv_cat_f, nan=0.0, posinf=0.0, neginf=0.0)
            kv_cat_f = kv_cat_f.clamp(-bound, bound)
            # Each query attends exactly the rows its index list selects
            # (window picks + compressed top-k picks), while the gathered
            # tensor holds every committed row: expand the per-query index
            # list into a row mask with static shapes (see _picked_row_mask)
            # so the block replays inside a captured ACL graph.
            row_mask = _picked_row_mask(idxs, kv_cat.size(0))

            for tile_start in range(0, q_len, _ATTENTION_QUERY_TILE):
                tile_end = min(tile_start + _ATTENTION_QUERY_TILE, q_len)
                scores = torch.einsum("shd,nd->shn", q_seq[tile_start:tile_end].to(torch.float32), kv_cat_f)
                scores = scores * self.scale
                scores = scores.masked_fill(~row_mask[tile_start:tile_end].unsqueeze(1), float("-inf"))
                row_max = scores.amax(dim=-1).clamp_min(_NEG_INF_FLOOR)
                probs = torch.exp(scores - row_max.unsqueeze(-1))
                # The sink adds exp(sink - row_max) to the normalizer only.
                sum_exp = probs.sum(dim=-1)
                if sinks is not None:
                    sum_exp = sum_exp + torch.exp(sinks - row_max)
                tile_out = torch.einsum("shn,nd->shd", probs, kv_cat_f)
                # Empty-DP queries have no visible values and no required sink.
                tile_out = tile_out / sum_exp.clamp_min(_FP32_TINY).unsqueeze(-1)
                out[token_offset + tile_start : token_offset + tile_end] = tile_out.to(q.dtype)
            token_offset += q_len
        return out

    # -- cache gather helpers -------------------------------------------------

    def _committed_capacity(
        self,
        dsa: DsaMetadata | None,
        ratio: int,
        ctx_lens: torch.Tensor,
    ) -> int:
        """Compressed-row gather capacity (rows) for one forward.

        Under ACL graph capture this is a Python constant baked into the
        graph: the runner's per-bucket bound ``graph_token_capacity`` when the
        captured entry carries one (the decode-graph runner sets it to
        ``bucket * granularity``), otherwise ``graph_token_capacity_granularity
        // ratio`` -- so no replay ever sync-reads the device sequence lengths.
        Eagerly it is the exact batch maximum, read on the host where the sync
        is free.
        """
        if ratio <= 0:
            return 0
        if bool(getattr(dsa, "is_acl_graph", False)):
            bucketed_capacity = int(getattr(dsa, "graph_token_capacity", 0) or 0)
            if bucketed_capacity > 0:
                return max(bucketed_capacity // ratio, 1)
            if self.graph_token_capacity_granularity > 0:
                return max(self.graph_token_capacity_granularity // ratio, 1)
            raise RuntimeError(
                "DeepSeek-V4.1 ACL graph capture needs a static committed-row "
                "capacity, but the serving entry did not derive one (no "
                "max_model_len was provided). The engine should have derived "
                "a default at construction; a missing value means the "
                "deployment entry point passed no max_model_len."
            )
        if ctx_lens.numel() == 0:
            return 0
        return max(int(torch.div(ctx_lens.max(), ratio, rounding_mode="floor").item()), 0)

    def _gather_cache_rows(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq: int,
        count: int,
        block_size: int,
    ) -> torch.Tensor:
        rows = torch.arange(count, device=cache.device)
        slots = _paged_cache_slots(block_table, seq, rows, block_size)
        return cache.view(-1, cache.size(-1)).index_select(0, slots)

    def _gather_swa_positions(
        self,
        swa_flat: torch.Tensor,
        block_table: torch.Tensor,
        seq: int,
        positions: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        slots = _paged_cache_slots(block_table, seq, positions, block_size)
        return swa_flat.index_select(0, slots)

    def _v41_state_slots(
        self,
        dsa: DsaMetadata,
        layer_id: int,
        swa_cache_idx: int,
        seq: int,
        positions: torch.Tensor,
        state_rows: int,
    ) -> torch.Tensor:
        """Compressor state wraps at 32 rows independently of the SWA page size."""
        block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, swa_cache_idx)
        if block_table is None:
            return positions % state_rows
        swa = self._kv_caches[layer_id].swa
        block_size = swa.size(1) if swa is not None else self.window_size
        return _paged_cache_slots(block_table, seq, positions, block_size, row_stride=state_rows)

    def _shared_runtime(self) -> CSA2SharedRuntime:
        if self._v41_shared is None:
            self._v41_shared = CSA2SharedRuntime()
        return self._v41_shared
