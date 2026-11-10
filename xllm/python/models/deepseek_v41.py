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

import json
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.logger import logger
from xllm.python import distributed as _distributed
from xllm.python import kernels
from xllm.python.distributed import (
    engram_all_gather_object,
    engram_barrier,
    engram_dp_shard_size,
    init_engram_dp_group,
    world_topology,
)
from xllm.python.layers.attention import Attention
from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.layernorm import RMSNorm
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.layers.moe_parallel import Eplv2CommPolicy, dp_gather_tokens
from xllm.python.layers.rotary_embedding import RotaryEmbedding
from xllm.python.model_executor.forward_context import (
    get_execution_buffer,
    get_execution_context,
    get_forward_context,
    get_forward_context_or_none,
    record_layer_event,
)
from xllm.python.model_loader import W8A8WeightLoader
from xllm.python.model_loader.module_loaders import load_w8a8_dynamic_projection
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    DeepseekV4MoE,
    DeepseekV4RotaryEmbedding,
    _pick,
    _require_checkpoint_key,
    _resolve_dsv4_moe_prefix,
)
from xllm.python.models.deepseek_v32 import DeepseekV3MLP, W8A8DynamicLinear
from xllm.python.models.deepseek_v41_engram import (
    ENGRAM_STORAGE_FP8,
    ENGRAM_STORAGE_INT8,
    Engram,
    EngramEmbedding,
    EngramExecutionMetadata,
    EngramExecutionMetadataBuilder,
    EngramHashState,
    EngramLayout,
    _e4m3_to_float,
    _validate_ue8m0_scale_bytes,
    image_sentinel_mask,
    load_quarot_gate_unrotate,
    ue8m0_to_scale,
)
from xllm.python.models.engram_config import EngramConfig
from xllm.python.models.engram_shared_memory import ensure_dev_shm_capacity, estimate_shared_bytes
from xllm.python.platform import current_platform

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict

    from xllm.python.attention.csa2_attention import Csa2LayerContext


CSA2_MODE_SWA = "swa"
CSA2_MODE_FULL = "full"
CSA2_MODE_REINDEX = "reindex"
CSA2_MODE_REUSE = "reuse"


@dataclass(frozen=True)
class CSA2LayerMode:
    layer_id: int
    mode: str
    # 0: SWA only; 1: decoder main KV; 2: pooled encoder main KV.
    compress_ratio: int
    # -1 for SWA; otherwise the Full layer that owns the shared main/index caches.
    main_kv_source_layer: int

    @property
    def is_csa2(self) -> bool:
        return self.mode != CSA2_MODE_SWA

    @property
    def is_full(self) -> bool:
        return self.mode == CSA2_MODE_FULL

    @property
    def is_reindex(self) -> bool:
        return self.mode == CSA2_MODE_REINDEX

    @property
    def is_reuse(self) -> bool:
        return self.mode == CSA2_MODE_REUSE


class CSA2LayerPlan:
    def __init__(self, modes: list[CSA2LayerMode]) -> None:
        self._modes = modes

    def mode(self, layer_id: int) -> CSA2LayerMode:
        if layer_id < 0 or layer_id >= len(self._modes):
            raise IndexError(f"layer {layer_id} outside the CSA2 plan's {len(self._modes)} layers")
        return self._modes[layer_id]

    def __len__(self) -> int:
        return len(self._modes)

    @classmethod
    def build(
        cls,
        n_layers: int,
        compress_ratios: Sequence[int],
        kv_source_layer_ids: Sequence[int],
        index_source_layer_ids: Sequence[int],
    ) -> CSA2LayerPlan:
        kv_sources = sorted({int(source) for source in kv_source_layer_ids})
        index_sources = sorted({int(source) for source in index_source_layer_ids})
        kv_source_set = set(kv_sources)
        index_source_set = set(index_sources)
        for source in kv_sources:
            if not 0 <= source < n_layers:
                raise ValueError(f"kv source layer {source} is outside the model's {n_layers} layers")
        for source in index_sources:
            if not 0 <= source < n_layers:
                raise ValueError(f"index source layer {source} is outside the model's {n_layers} layers")
        non_indexed = [source for source in kv_sources if source not in index_source_set]
        if non_indexed:
            raise ValueError(f"kv source layers must also be index sources: {non_indexed}")
        for source in index_sources:
            if not any(kv_source <= source for kv_source in kv_sources):
                raise ValueError(f"index source layer {source} has no preceding kv source layer")
        ratios = [int(ratio) for ratio in compress_ratios]
        for source in kv_sources:
            if source >= len(ratios) or ratios[source] <= 0:
                raise ValueError(f"kv source layer {source} must have a positive compress ratio")

        modes: list[CSA2LayerMode] = []
        for layer_id in range(n_layers):
            ratio = ratios[layer_id] if layer_id < len(ratios) else 0
            kv_source = max((source for source in kv_sources if source <= layer_id), default=-1)
            if ratio <= 0 or kv_source < 0:
                mode = CSA2_MODE_SWA
                main_kv_source = -1
            elif layer_id in kv_source_set:
                mode = CSA2_MODE_FULL
                main_kv_source = layer_id
            elif layer_id in index_source_set:
                mode = CSA2_MODE_REINDEX
                main_kv_source = kv_source
            else:
                mode = CSA2_MODE_REUSE
                main_kv_source = kv_source
            modes.append(
                CSA2LayerMode(
                    layer_id=layer_id,
                    mode=mode,
                    compress_ratio=max(ratio, 0),
                    main_kv_source_layer=main_kv_source,
                )
            )
        return cls(modes)


def _int_list(d: dict, key: str, default: list[int] | None = None) -> list[int]:
    value = d.get(key)
    if value is None:
        return list(default or [])
    return [int(item) for item in value]


def _parse_v41_compress_ratios(raw: Sequence[int], n_layers: int) -> list[int]:
    ratios = [int(ratio) for ratio in raw]
    for ratio in ratios:
        if ratio not in (0, 1, 2, 4):
            raise ValueError(f"unsupported DeepSeek-V4.1 compression ratio: {ratio}")
    if len(ratios) < n_layers:
        ratios.extend([0] * (n_layers - len(ratios)))
    return ratios[:n_layers]


_V41_FIELD_DEFAULTS: dict[str, Any] = {
    "model_type": "deepseek_v41",
    "hidden_size": 5120,
    "q_lora_rank": 1280,
    "n_routed_experts": 384,
    "moe_intermediate_size": 2304,
    "index_n_heads": 32,
    "rms_norm_eps": 1e-20,
}

_ENGRAM_SHARED_LOAD_BARRIER_TIMEOUT = timedelta(seconds=30)


@dataclass
class DeepseekV41Config(DeepseekV4Config):
    model_type: str = "deepseek_v41"
    hidden_size: int = 5120
    n_layers: int = 40
    q_lora_rank: int = 1280
    rms_norm_eps: float = 1e-20
    n_routed_experts: int = 384
    moe_intermediate_size: int = 2304
    dense_ffn_intermediate_size: int = 0
    index_n_heads: int = 32
    kv_source_layer_ids: list[int] = field(default_factory=list)
    index_source_layer_ids: list[int] = field(default_factory=list)
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    engram_layer_ids: list[int] = field(default_factory=list)
    engram_num_embeddings: list[int] = field(default_factory=list)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_compressed_vocab_size: int = 99092
    engram_pad_token_id: int = -1
    num_nextn_predict_layers: int = 3
    dspark_target_layer_ids: list[int] = field(default_factory=lambda: [37, 38, 39])
    dspark_block_size: int = 5
    dspark_markov_rank: int = 256
    dspark_noise_token_id: int = 128799
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    enable_dsa_cp: bool = False
    enable_fused_mc2: bool = False
    mc2_group_info: tuple[str, int, int] | None = None
    image_token_id: int = -1
    engram_config: EngramConfig = field(default_factory=EngramConfig)
    pp_size: int = 1
    dcp_size: int = 1

    @property
    def dense_intermediate_size(self) -> int:
        if self.dense_ffn_intermediate_size > 0:
            return self.dense_ffn_intermediate_size
        if self.moe_intermediate_size > 0:
            return self.moe_intermediate_size
        return max(self.hidden_size * 4, 0)

    @classmethod
    def from_dict(cls, d: dict) -> DeepseekV41Config:
        merged = dict(d)
        for key, value in _V41_FIELD_DEFAULTS.items():
            if merged.get(key) is None:
                merged[key] = value
        if merged.get("num_hidden_layers") is None and merged.get("n_layers") is None:
            merged["num_hidden_layers"] = 40
        cfg = super().from_dict(merged)
        cfg = replace(
            cfg,
            dense_ffn_intermediate_size=int(_pick(merged, "intermediate_size", default=0)),
            compress_ratios=_parse_v41_compress_ratios(list(merged.get("compress_ratios") or []), cfg.n_layers),
            # Retain the inherited field, but V4.1 does not use hash routing.
            n_hash_layers=0,
            kv_source_layer_ids=_int_list(merged, "kv_source_layer_ids"),
            index_source_layer_ids=_int_list(merged, "index_source_layer_ids"),
            candidate_source_layer_id=int(_pick(merged, "candidate_source_layer_id", default=-1)),
            candidate_topk_blocks=int(_pick(merged, "candidate_topk_blocks", default=2048)),
            candidate_block_size=int(_pick(merged, "candidate_block_size", default=8)),
            engram_layer_ids=_int_list(merged, "engram_layer_ids"),
            engram_num_embeddings=_int_list(merged, "engram_num_embeddings"),
            engram_max_ngram_size=int(_pick(merged, "engram_max_ngram_size", default=4)),
            engram_vocab_size=int(_pick(merged, "engram_vocab_size", default=16000000)),
            engram_n_heads=int(_pick(merged, "engram_n_heads", default=8)),
            engram_head_dim=int(_pick(merged, "engram_head_dim", default=256)),
            engram_compressed_vocab_size=int(_pick(merged, "engram_compressed_vocab_size", default=99092)),
            engram_pad_token_id=int(_pick(merged, "engram_pad_token_id", default=-1)),
            num_nextn_predict_layers=int(_pick(merged, "num_nextn_predict_layers", default=3)),
            dspark_target_layer_ids=_int_list(merged, "dspark_target_layer_ids", default=[37, 38, 39]),
            dspark_block_size=int(_pick(merged, "dspark_block_size", default=5)),
            dspark_markov_rank=int(_pick(merged, "dspark_markov_rank", default=256)),
            dspark_noise_token_id=int(_pick(merged, "dspark_noise_token_id", default=128799)),
            dspark_n_routed_experts=int(_pick(merged, "dspark_n_routed_experts", default=128)),
            dspark_num_experts_per_tok=int(_pick(merged, "dspark_num_experts_per_tok", default=3)),
            enable_dsa_cp=bool(_pick(merged, "enable_dsa_cp", default=False)),
            enable_fused_mc2=bool(_pick(merged, "enable_fused_mc2", default=False)),
            mc2_group_info=_pick(merged, "mc2_group_info", default=None),
            image_token_id=int(_pick(merged, "image_token_id", default=-1)),
            engram_config=EngramConfig.from_value(merged.get("engram_config", {})),
            pp_size=int(_pick(merged, "pp_size", default=1)),
            dcp_size=int(_pick(merged, "dcp_size", default=1)),
        )
        _validate_engram_config(cfg)
        return cfg


def _validate_engram_config(cfg: DeepseekV41Config) -> None:
    if not cfg.engram_layer_ids:
        return
    if not cfg.engram_config.cpu_offload:
        return
    if cfg.tp_size not in (1, 2, 4, 8) or cfg.pp_size != 1 or cfg.dcp_size != 1:
        raise ValueError(
            "Engram requires TP in {1, 2, 4, 8} with PP=1 and DCP=1; got "
            f"TP={cfg.tp_size}, PP={cfg.pp_size}, DCP={cfg.dcp_size}. "
            "dp_shared_memory does not change TP sharding."
        )


V41_QUANT_FP8 = "fp8"
V41_QUANT_ASCEND = "ascend"
V41_QUANT_NONE = "none"


def _detect_v41_quant_mode(config: dict) -> str:
    model_path = config.get("model_path")
    if not model_path:
        logger.warning("DeepSeek-V4.1 quant mode detection: model_path missing; defaulting to bf16 (V41_QUANT_NONE)")
        return V41_QUANT_NONE
    if os.path.isfile(os.path.join(model_path, "quant_model_description.json")):
        return V41_QUANT_ASCEND
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return V41_QUANT_NONE
    with open(config_path, encoding="utf-8") as handle:
        checkpoint_config = json.load(handle)
    quant_config = checkpoint_config.get("quantization_config") or {}
    method = quant_config.get("quant_method")
    if method == "fp8":
        return V41_QUANT_FP8
    if method == "ascend":
        return V41_QUANT_ASCEND
    return V41_QUANT_NONE


def _make_v41_linear(
    in_features: int,
    out_features: int,
    quant_mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    if quant_mode == V41_QUANT_ASCEND:
        return W8A8DynamicLinear(
            in_features,
            out_features,
            device,
            transpose_weight_after_loading=False,
        )
    return nn.Linear(in_features, out_features, bias=False, dtype=dtype, device=device)


_rms_norm_cast_op: Any = None


def _get_rms_norm_cast_op() -> Any:
    from xllm.python.kernels_npu.op_api import has_op_api

    global _rms_norm_cast_op
    if (
        _rms_norm_cast_op is None
        and hasattr(torch.ops.xllm_ops, "rms_norm_cast_va")
        and has_op_api("aclnnRmsNormCastVa")
    ):
        _rms_norm_cast_op = getattr(kernels, "rms_norm_cast_va", None)
    return _rms_norm_cast_op


def _rms_norm_cast_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.dim() in (2, 3)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.dim() == 1
        and weight.numel() == x.size(-1)
        and _get_rms_norm_cast_op() is not None
        and x.device.type in ("npu", "privateuseone")
    )


_rms_norm_cast_active_logged = False
_rms_norm_cast_fallback_logged = False


def _log_rms_norm_cast_active() -> None:
    global _rms_norm_cast_active_logged
    if not _rms_norm_cast_active_logged:
        _rms_norm_cast_active_logged = True
        logger.info("rms_norm_cast fused path active (op=xllm_ops.rms_norm_cast_va)")


def _log_rms_norm_cast_fallback(x: torch.Tensor, weight: torch.Tensor) -> None:
    global _rms_norm_cast_fallback_logged
    if _rms_norm_cast_fallback_logged:
        return
    _rms_norm_cast_fallback_logged = True
    logger.warning(
        "the fused RMSNorm+cast op is unavailable "
        f"(device={x.device.type} dim={x.dim()} dtype={x.dtype} "
        f"weight_dtype={weight.dtype} op_registered={_get_rms_norm_cast_op() is not None}); "
        "staying on the eager path"
    )


_fused_shared_act_active_logged = False
_fused_shared_act_fallback_logged = False


def _fused_shared_act_device_supported(x: torch.Tensor) -> bool:
    return x.device.type in ("npu", "privateuseone")


def _fused_shared_act_grouped_supported(shared: nn.Module) -> bool:
    gate_up = getattr(shared, "gate_up_proj", None)
    down = getattr(shared, "down_proj", None)
    weight_scale = getattr(gate_up, "weight_scale", None)
    gate_up_width = getattr(gate_up, "out_features", None)
    down_in = getattr(down, "in_features", None)
    if weight_scale is None or gate_up_width is None or down_in is None:
        return False
    if weight_scale.dim() > 2:
        return False
    return weight_scale.numel() == gate_up_width == 2 * down_in


def _fused_shared_act_supported(x: torch.Tensor, shared: nn.Module, limit: float) -> bool:
    return (
        isinstance(shared, DeepseekV3MLP)
        and 0.0 < limit < 1_000_000.0
        and x.dim() == 2
        and x.numel() > 0
        and x.dtype in (torch.bfloat16, torch.float16)
        and getattr(shared.gate_up_proj, "weight_scale", None) is not None
        and getattr(shared.down_proj, "weight_scale", None) is not None
        and _fused_shared_act_grouped_supported(shared)
        and _fused_shared_act_device_supported(x)
    )


def _log_fused_shared_act_active() -> None:
    global _fused_shared_act_active_logged
    if not _fused_shared_act_active_logged:
        _fused_shared_act_active_logged = True
        logger.info("shared expert fused clamped-SwiGLU active (op=xllm_ops.dequant_swiglu_quant)")


def _log_fused_shared_act_fallback(x: torch.Tensor, shared: nn.Module, limit: float) -> None:
    global _fused_shared_act_fallback_logged
    if _fused_shared_act_fallback_logged:
        return
    _fused_shared_act_fallback_logged = True
    logger.warning(
        "the fused shared-expert activation is unavailable "
        f"(device={x.device.type} dim={x.dim()} dtype={x.dtype} "
        f"max={x.numel()} limit={limit} module={type(shared).__name__} "
        f"grouped_form_ok={_fused_shared_act_grouped_supported(shared)}); "
        "staying on the eager clamped SwiGLU"
    )


_MOE_MC2_MAX_LOCAL_TOKENS = 256

_moe_mc2_active_logged = False
_moe_mc2_fallback_logged = False
_moe_mc2_draft_fallback_logged = False
_moe_mc2_collective_warned = False


def _moe_mc2_ops_available() -> bool:
    try:
        import torch_npu
    except ImportError:
        return False
    return hasattr(torch_npu, "npu_moe_distribute_dispatch_v2") and hasattr(torch_npu, "npu_moe_distribute_combine_v2")


def _moe_mc2_device_supported(x: torch.Tensor) -> bool:
    return x.device.type in ("npu", "privateuseone")


def _moe_mc2_decode_step() -> bool:
    ctx = get_forward_context_or_none()
    if ctx is None:
        return False
    metadata = ctx.metadata
    if getattr(metadata, "is_dummy", False):
        return False
    return not (metadata.is_prefill or metadata.is_chunked_prefill)


def _moe_mc2_all_dp_ranks_decode(metadata: Any) -> bool:
    dp_is_decode = getattr(metadata, "dp_is_decode", None)
    if dp_is_decode:
        return all(int(flag) == 1 for flag in dp_is_decode)
    token_counts = getattr(metadata, "dp_execution_token_counts", None)
    return not token_counts or len(token_counts) <= 1


def _log_moe_mc2_active(ep_rank: int, ep_size: int) -> None:
    global _moe_mc2_active_logged
    if not _moe_mc2_active_logged:
        _moe_mc2_active_logged = True
        logger.info(
            f"MoE MC2 decode path active (ep_rank={ep_rank} ep_size={ep_size}, "
            "chain=dispatch_v2->dynamic_quant->grouped_matmul->dequant_swiglu_quant->grouped_matmul->combine_v2)"
        )


def _log_moe_mc2_fallback(reason: str) -> None:
    global _moe_mc2_fallback_logged
    if _moe_mc2_fallback_logged:
        return
    _moe_mc2_fallback_logged = True
    logger.warning(
        f"the MC2 decode path is unavailable for the target ({reason}); using the portable grouped-expert path"
    )


def _moe_mc2_draft_veto_applies(moe: Any) -> bool:
    return bool(getattr(moe, "_moe_is_dspark_draft", False))


def _log_moe_mc2_draft_fallback(reason: str) -> None:
    global _moe_mc2_draft_fallback_logged
    if _moe_mc2_draft_fallback_logged:
        return
    _moe_mc2_draft_fallback_logged = True
    logger.warning(
        "the MC2 decode path is unavailable for the DSpark "
        f"draft (Fix-1, target keeps MC2): {reason}; using the portable "
        "grouped-expert path"
    )


def _log_moe_mc2_collective(ep_rank: int, ep_size: int) -> None:
    global _moe_mc2_collective_warned
    if not _moe_mc2_collective_warned:
        _moe_mc2_collective_warned = True
        logger.warning(
            "the MC2 decode path engages an all-to-all *collective* expert path: every rank of the "
            f"moe_ep group must run it unconditionally and in lockstep (ep_rank={ep_rank} "
            f"ep_size={ep_size}); a rank that skips a layer will hang the dispatch/combine kernels"
        )


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = mixes[:, 2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    return pre, post, comb


def _get_hc_post_op() -> Any:
    from xllm.python.kernels_npu import hyper_connection

    return hyper_connection.hc_post_va


def _hc_post_npu_supported(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_mult: int,
) -> bool:
    hidden = x.size(-1)
    return (
        x.device.type in ("npu", "privateuseone")
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.dim() == 2
        and residual.dim() == 3
        and post.dim() == 2
        and comb.dim() == 3
        and hc_mult == 4
        and residual.size(1) == hc_mult
        and residual.size(2) == hidden
        and post.size(1) == hc_mult
        and comb.size(1) == hc_mult
        and comb.size(2) == hc_mult
        and post.dtype == torch.float32
        and comb.dtype == torch.float32
        and residual.dtype == x.dtype
        and x.is_contiguous()
        and residual.is_contiguous()
        and post.is_contiguous()
        and hidden % 64 == 0
    )


def _hc_post_fused_supported(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_mult: int,
) -> bool:
    return (
        _hc_post_npu_supported(x, residual, post, comb, hc_mult)
        and x.dtype == torch.bfloat16
        and _get_hc_post_op() is not None
    )


_fused_hc_post_active_logged = False
_fused_hc_post_fallback_logged = False


def _log_fused_hc_post_active() -> None:
    global _fused_hc_post_active_logged
    if not _fused_hc_post_active_logged:
        _fused_hc_post_active_logged = True
        logger.info("hc_post fused path active")


def _log_fused_hc_post_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_mult: int,
) -> None:
    global _fused_hc_post_fallback_logged
    if _fused_hc_post_fallback_logged:
        return
    _fused_hc_post_fallback_logged = True
    logger.warning(
        "the fused hc_post op is unavailable "
        f"(device={x.device.type} dtype={x.dtype} x_dim={x.dim()} residual_dim={residual.dim()} "
        f"post_dtype={post.dtype} comb_dtype={comb.dtype} hc_mult={hc_mult} "
        f"op_registered={_get_hc_post_op() is not None}); staying on the eager path"
    )


_HC_PRE_MODE_EAGER = "eager"
_HC_PRE_MODE_FUSED = "fused"


def _get_hc_pre_fused_op() -> Any:
    from xllm.python.kernels_npu import hyper_connection

    return hyper_connection.hc_pre_va


def _hc_pre_fused_guards(dtype: torch.dtype, device: torch.device, hc_mult: int) -> bool:
    device_type = torch.device(device).type
    return hc_mult == 4 and dtype == torch.bfloat16 and device_type in ("npu", "privateuseone")


def _hc_pre_mode_decision(dtype: torch.dtype, device: torch.device, hc_mult: int) -> str:
    if _hc_pre_fused_guards(dtype, device, hc_mult) and _get_hc_pre_fused_op() is not None:
        return _HC_PRE_MODE_FUSED
    return _HC_PRE_MODE_EAGER


_fused_hc_pre_active_logged = False
_fused_hc_pre_fallback_logged = False


def _log_fused_hc_pre_active() -> None:
    global _fused_hc_pre_active_logged
    if not _fused_hc_pre_active_logged:
        _fused_hc_pre_active_logged = True
        logger.info("hc_pre fused path active (native pre vector and external pre_mix)")


def _log_fused_hc_pre_fallback(dtype: torch.dtype, device: torch.device, hc_mult: int) -> None:
    global _fused_hc_pre_fallback_logged
    if _fused_hc_pre_fallback_logged:
        return
    if not _hc_pre_fused_guards(dtype, device, hc_mult):
        return
    _fused_hc_pre_fallback_logged = True
    logger.warning(
        "the fused hc_pre op is unavailable "
        f"(device={torch.device(device).type} dtype={dtype} hc_mult={hc_mult} "
        f"op_available={_get_hc_pre_fused_op() is not None}); staying on the eager sinkhorn path"
    )


def _log_hc_pre_mode(dtype: torch.dtype, device: torch.device, hc_mult: int, mode: str) -> None:
    if mode == _HC_PRE_MODE_FUSED:
        _log_fused_hc_pre_active()
        return
    _log_fused_hc_pre_fallback(dtype, device, hc_mult)


def _merged_attn_quant_supported(x: torch.Tensor) -> bool:
    return x.device.type in ("npu", "privateuseone") and x.dim() == 2


class DeepseekV41HyperConnection(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        device: torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_eps = cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        self.sinkhorn_iters = cfg.hc_sinkhorn_iters
        self._hc_pre_mode = (
            _hc_pre_mode_decision(dtype, device, cfg.hc_mult) if dtype is not None else _HC_PRE_MODE_EAGER
        )
        if dtype is not None:
            _log_hc_pre_mode(dtype, device, cfg.hc_mult, self._hc_pre_mode)
        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.hidden_size
        for part in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{part}_fn",
                nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32, device=device)),
            )
            self.register_parameter(
                f"hc_{part}_base",
                nn.Parameter(torch.empty(mix_hc, dtype=torch.float32, device=device)),
            )
            self.register_parameter(
                f"hc_{part}_scale",
                nn.Parameter(torch.empty(3, dtype=torch.float32, device=device)),
            )

    def hc_mixes(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flattened = x.flatten(-2).float()
        rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flattened, hc_fn) * rsqrt
        return _hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.hc_eps)

    def hc_pre_fused(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        pre_mix: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if pre_mix is None:
            pre_mix = self.make_identity_pre_mix(x, self.hc_mult)
        # PrivateUse1 dispatch drops keyword arguments; all nine operands must be positional.
        y, post, comb, pre = _get_hc_pre_fused_op()(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix,
            self.hc_mult,
            self.sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )
        return y, pre, post, comb

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse the hc copies: ``sum_j pre[j] * x[j]`` (model.py:957-960)."""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=-2)
        return y.to(x.dtype)

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            # Empty DP ranks must skip this kernel while still joining surrounding collectives.
            return residual
        if _hc_post_fused_supported(x, residual, post, comb, self.hc_mult):
            # The kernel consumes comb as comb^T @ residual and expects [bs, seq, ...] operands.
            comb4d = comb.transpose(-2, -1).contiguous().unsqueeze(0)
            out = _get_hc_post_op()(
                x.unsqueeze(0),
                residual.unsqueeze(0),
                post.unsqueeze(0),
                comb4d,
            )
            _log_fused_hc_post_active()
            return out.squeeze(0)
        if _hc_post_npu_supported(x, residual, post, comb, self.hc_mult):
            return kernels.hc_post(
                x.unsqueeze(0),
                residual.unsqueeze(0),
                post.unsqueeze(0),
                comb.transpose(-2, -1).contiguous().unsqueeze(0),
            ).squeeze(0)
        _log_fused_hc_post_fallback(x, residual, post, comb, self.hc_mult)
        y = post.unsqueeze(-1) * x.float().unsqueeze(-2)
        y = y + torch.einsum("tjk,tkd->tjd", comb.float(), residual.float())
        return y.to(x.dtype)

    @staticmethod
    def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
        pre_mix = torch.zeros(x.size(0), hc_mult, dtype=torch.float32, device=x.device)
        pre_mix[:, 0] = 1.0
        return pre_mix


class DeepseekV41Indexer(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
        quant_mode: str,
        owns_k: bool,
    ) -> None:
        super().__init__()
        self.owns_k = owns_k
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.wq_b = _make_v41_linear(
            cfg.q_lora_rank,
            self.n_heads * self.head_dim,
            quant_mode,
            dtype,
            device,
        )
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_heads, bias=False, dtype=dtype, device=device)
        if owns_k:
            self.wk = nn.Linear(cfg.head_dim, self.head_dim, bias=False, dtype=dtype, device=device)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype, device)

    def process_weights_after_loading(self) -> None:
        if isinstance(self.wq_b, W8A8DynamicLinear):
            self.wq_b.process_weights_after_loading()


class DeepseekV41Attention(Attention):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        mode: CSA2LayerMode,
        quant_mode: str = V41_QUANT_NONE,
    ) -> None:
        tp = cfg.tp_size
        self.enable_dsa_cp = cfg.enable_dsa_cp and tp > 1
        if self.enable_dsa_cp and cfg.n_heads % tp:
            raise ValueError("enable_dsa_cp requires n_heads divisible by tp_size")
        num_heads = cfg.n_heads if self.enable_dsa_cp else cfg.n_heads // tp
        head_dim = cfg.head_dim
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            sliding_window=cfg.window_size,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.layer_id = layer_id
        self.mode = mode
        self.quant_mode = quant_mode
        self.num_heads_local = num_heads
        self.head_dim = head_dim
        self.register_parameter(
            "attn_sink",
            nn.Parameter(torch.zeros(num_heads, dtype=torch.float32, device=device)),
        )
        self.attn_sink_loaded = False
        self.q_a_proj = _make_v41_linear(cfg.hidden_size, cfg.q_lora_rank, quant_mode, dtype, device)
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps, dtype, device)
        self.q_b_proj = _make_v41_linear(cfg.q_lora_rank, num_heads * head_dim, quant_mode, dtype, device)
        self.kv_proj = _make_v41_linear(cfg.hidden_size, head_dim, quant_mode, dtype, device)
        self.kv_a_layernorm = RMSNorm(head_dim, cfg.rms_norm_eps, dtype, device)
        if cfg.o_groups % tp != 0:
            raise ValueError(
                f"DeepSeek-V4.1 o_groups ({cfg.o_groups}) must be divisible by tensor-parallel "
                f"size {tp}; the grouped o_a/o_b output projection cannot be sharded evenly"
            )
        self.n_local_groups = cfg.o_groups // tp
        self.o_lora_rank = cfg.o_lora_rank
        o_a_in = (cfg.n_heads * head_dim) // cfg.o_groups
        if quant_mode == V41_QUANT_ASCEND:
            self.o_a_proj = ColumnParallelLinear(
                o_a_in,
                (cfg.o_groups * cfg.o_lora_rank) // tp,
                tp,
                dtype=dtype,
                device=device,
            )
            self.o_b_proj = RowParallelLinear(
                (cfg.o_groups * cfg.o_lora_rank) // tp,
                cfg.hidden_size,
                tp,
                dtype=dtype,
                device=device,
                use_checkpoint_layout=True,
            )
        else:
            self.o_a_proj = nn.Linear(
                o_a_in,
                self.n_local_groups * cfg.o_lora_rank,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.o_b_proj = nn.Linear(
                self.n_local_groups * cfg.o_lora_rank,
                cfg.hidden_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
        # Ratio-2 pooling stays FP32, including its projection parameters.
        if mode.is_full:
            ratio = mode.compress_ratio
            cmp_dtype = torch.float32 if ratio > 1 else dtype
            self.cmp_wkv = nn.Linear(cfg.hidden_size, head_dim, bias=False, dtype=cmp_dtype, device=device)
            if ratio > 1:
                self.cmp_wgate = nn.Linear(cfg.hidden_size, head_dim, bias=False, dtype=torch.float32, device=device)
            self.cmp_norm = RMSNorm(head_dim, cfg.rms_norm_eps, dtype, device)
        # Indexer: Full layers write the shared K and select; Reindex layers
        # re-select against it; Reuse/SWA layers own no indexer weights.
        self.indexer: DeepseekV41Indexer | None = (
            DeepseekV41Indexer(cfg, dtype, device, quant_mode, owns_k=mode.is_full)
            if mode.is_full or mode.is_reindex
            else None
        )
        # Per-forward hand-off to the backend.
        self._csa2_ctx: Csa2LayerContext | None = None
        self._csa2_used_topk: torch.Tensor | None = None

    def process_weights_after_loading(self) -> None:
        if self.quant_mode == V41_QUANT_ASCEND:
            for module in (self.q_a_proj, self.q_b_proj, self.kv_proj):
                module.process_weights_after_loading()
            if self.indexer is not None:
                self.indexer.process_weights_after_loading()

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        from xllm.python.attention.csa2_attention import (
            Csa2LayerContext,
            apply_interleaved_rope,
        )

        backend = get_forward_context().attention_backend
        execute_csa2 = getattr(backend, "execute_csa2_layer", None)
        if not callable(execute_csa2):
            raise RuntimeError("DeepSeek-V4.1 requires the CSA2 attention backend")
        num_tokens = hidden.size(0)
        cp_context = None
        query_positions = positions
        if self.enable_dsa_cp:
            from xllm.python.model_executor.dsa_cp import get_dsa_cp_runtime

            runtime = get_dsa_cp_runtime(num_tokens, self.cfg.tp_size, self.cfg.tp_rank)
            cp_context = runtime.plan
            query_positions = cp_context.shard(positions)
        if self.quant_mode == V41_QUANT_ASCEND and _merged_attn_quant_supported(hidden):
            hidden_int8, hidden_pertoken = kernels.dynamic_quant(hidden)
            q_a = self.q_a_proj.forward_quantized(hidden_int8, hidden_pertoken)
            kv = self.kv_a_layernorm(self.kv_proj.forward_quantized(hidden_int8, hidden_pertoken))
        else:
            q_a = self.q_a_proj(hidden)
            kv = self.kv_a_layernorm(self.kv_proj(hidden))
        qr = self.q_a_layernorm(q_a)
        query_qr = cp_context.shard(qr) if cp_context is not None else qr
        if query_qr.size(0) == 0:
            q = hidden.new_empty((0, self.num_heads_local, self.head_dim))
        else:
            q = self.q_b_proj(query_qr).view(-1, self.num_heads_local, self.head_dim)
            q = apply_interleaved_rope(q, query_positions, cos_sin_cache)
        kv = apply_interleaved_rope(kv, positions, cos_sin_cache)
        self._csa2_ctx = Csa2LayerContext(
            x=hidden,
            qr=qr,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
        )
        attn_out = execute_csa2(q, kv, self)
        # De-rotate the rope dims by the query's own rotation (conjugate)
        # BEFORE the o path (model.py:781).
        if attn_out.size(0) > 0:
            attn_out = apply_interleaved_rope(attn_out, query_positions, cos_sin_cache, inverse=True)
        if cp_context is not None:
            attn_out = cp_context.restore_tp_heads(attn_out)
        # O path: grouped LoRA projection (model.py:783-789).
        out = attn_out.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.o_a_proj.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o_low = torch.einsum("tgd,grd->tgr", out, wo_a)
        o = self.o_b_proj(o_low.reshape(num_tokens, -1))
        if self.cfg.tp_size > 1 and self.quant_mode != V41_QUANT_ASCEND:
            # Row-parallel partial: sum the per-rank head-group contributions.
            _distributed.tp_all_reduce(o)
        return o

    # -- weight helpers used by the backend -------------------------------------

    def compress_kv_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compressor projections: fp32 (kv, score) for ratio > 1, (kv, None) for ratio 1."""
        if self.mode.compress_ratio > 1:
            xf = x.float()
            return self.cmp_wkv(xf), self.cmp_wgate(xf)
        return self.cmp_wkv(x), None

    def compress_rmsnorm(self, v: torch.Tensor) -> torch.Tensor:
        """RMSNorm AFTER pooling (ratio > 1) / after the projection (ratio 1)."""
        return self.cmp_norm(v)

    def indexer_key(self, latent: torch.Tensor) -> torch.Tensor:
        """Indexer K from the pre-RoPE compressor latent: k_norm(wk(latent))."""
        k = self.indexer.wk(latent)
        return self.indexer.k_norm(k)

    def indexer_query(self, qr: torch.Tensor, qr_scale: torch.Tensor | None = None) -> torch.Tensor:
        """Indexer Q from the shared q-lora latent: wq_b(qr).

        Under the fused q norm+quant path the latent arrives already
        per-token INT8-quantized (``qr_scale`` present); the indexer then
        reuses those codes/scales through ``forward_quantized`` instead of
        re-quantizing -- the DSV4 production shape (deepseek_v4.py:826-830).
        """
        if qr_scale is not None:
            q = self.indexer.wq_b.forward_quantized(qr, qr_scale)
        else:
            q = self.indexer.wq_b(qr)
        return q.view(-1, self.indexer.n_heads, self.indexer.head_dim)

    def indexer_weights(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.indexer.head_dim**-0.5 * self.indexer.n_heads**-0.5
        return self.indexer.weights_proj(x) * scale


class DeepseekV41TorchMLP(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.swiglu_limit = cfg.swiglu_limit
        inter = cfg.dense_intermediate_size
        self.w1 = nn.Linear(cfg.hidden_size, inter, bias=False, dtype=dtype, device=device)
        self.w3 = nn.Linear(cfg.hidden_size, inter, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(inter, cfg.hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _swiglu_clamped(x, self.w1(x), self.w3(x), self.w2, self.swiglu_limit)


def _swiglu_clamped(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    w2: nn.Module,
    limit: float,
) -> torch.Tensor:
    gate = gate_proj.float()
    up = up_proj.float()
    if 0.0 < limit < 1_000_000.0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    hidden = F.silu(gate) * up
    return w2(hidden.to(x.dtype))


class DeepseekV41MoE(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.topk = cfg.n_activated_experts
        self.n_routed_experts = cfg.n_routed_experts
        self.routed_scaling = cfg.routed_scaling_factor
        self.norm_topk_prob = cfg.norm_topk_prob
        self.swiglu_limit = cfg.swiglu_limit
        self.image_token_id = cfg.image_token_id
        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False, dtype=dtype, device=device)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl = nn.Parameter(
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl_loaded = False
        inter = cfg.moe_intermediate_size
        hidden = cfg.hidden_size
        ep_size = cfg.ep_size if cfg.ep_size > 0 else 1
        ep_rank = cfg.ep_rank if cfg.ep_size > 0 else 0
        if cfg.n_routed_experts % ep_size != 0:
            raise ValueError(
                f"deepseek v4.1 torch MoE requires n_routed_experts divisible by ep_size, "
                f"got {cfg.n_routed_experts} experts and ep_size={ep_size}"
            )
        self.ep_size = ep_size
        self.num_experts_per_rank = cfg.n_routed_experts // ep_size
        self.start_expert_id = ep_rank * self.num_experts_per_rank
        self.experts_w13 = nn.Parameter(
            torch.empty(self.num_experts_per_rank, 2 * inter, hidden, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.experts_w2 = nn.Parameter(
            torch.empty(self.num_experts_per_rank, hidden, inter, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.shared_w1 = nn.Linear(hidden, inter, bias=False, dtype=dtype, device=device)
        self.shared_w2 = nn.Linear(inter, hidden, bias=False, dtype=dtype, device=device)
        self.shared_w3 = nn.Linear(hidden, inter, bias=False, dtype=dtype, device=device)

    @property
    def experts_w1(self) -> torch.Tensor:
        return self.experts_w13[:, : self.cfg.moe_intermediate_size]

    @property
    def experts_w3(self) -> torch.Tensor:
        return self.experts_w13[:, self.cfg.moe_intermediate_size :]

    def _selection_logits(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.gate.weight.float())
        scores = F.softplus(scores).sqrt()
        bias = self.e_score_correction_bias
        if input_ids is not None and self.gate_bias_vl_loaded:
            image_mask = input_ids.reshape(-1) == self.image_token_id
            bias = torch.where(image_mask.unsqueeze(-1), self.gate_bias_vl, self.e_score_correction_bias)
        return scores + bias, scores

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        if post_norm is not None:
            hidden = post_norm(hidden)
        shape = hidden.shape
        x = hidden.reshape(-1, self.cfg.hidden_size)
        logits, scores = self._selection_logits(x, input_ids)
        x, scatter_state = dp_gather_tokens(x, self.cfg.dp_size, self.cfg.dp_rank)
        if self.cfg.dp_size > 1:
            gate_values, _ = dp_gather_tokens(torch.cat((logits, scores), dim=-1), self.cfg.dp_size, self.cfg.dp_rank)
            logits, scores = gate_values.chunk(2, dim=-1)
        # Selection bias changes the chosen experts, not their routing weights.
        indices = logits.topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling
        y = kernels.grouped_moe_bf16(
            x,
            weights,
            indices.to(torch.int32),
            self.experts_w13.transpose(1, 2),
            self.experts_w2.transpose(1, 2),
            self.n_routed_experts,
            self.start_expert_id,
            self.num_experts_per_rank,
            swiglu_limit=self.swiglu_limit,
            router_weight_before_down=True,
        )
        if self.ep_size > 1:
            _distributed.moe_ep_all_reduce(y)
        y += self._shared_forward(x).float()
        return scatter_state.scatter(y.type_as(x)).view(shape)

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        return _swiglu_clamped(x, self.shared_w1(x), self.shared_w3(x), self.shared_w2, self.swiglu_limit)


class DeepseekV41SharedMLP(DeepseekV3MLP):
    def forward(self, hidden: torch.Tensor, tp_reduce_add: torch.Tensor | None = None) -> torch.Tensor:
        limit = float(self.swiglu_limit)
        if _fused_shared_act_supported(hidden, self, limit):
            _log_fused_shared_act_active()
            return self.forward_dequant_swiglu_quant(hidden, tp_reduce_add, clamp_limit=limit, grouped_activation=True)
        _log_fused_shared_act_fallback(hidden, self, limit)
        return super().forward(hidden, tp_reduce_add)

    def forward_dequant_swiglu_quant(
        self,
        x: torch.Tensor,
        tp_reduce_add: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        *,
        clamp_limit: float = 0.0,
        grouped_activation: bool = False,
    ) -> torch.Tensor:
        x_int8, pertoken = kernels.dynamic_quant(x)
        gate_up = self.gate_up_proj.forward_accumulated(x_int8)
        if grouped_activation:
            # The grouped activation takes one row count per group, rather than cumulative offsets.
            rows = gate_up.shape[0]
            if get_forward_context_or_none() is None:
                group_index = torch.tensor([rows], dtype=torch.int64, device=gate_up.device)
            else:
                group_index = get_execution_buffer(
                    ("SHARED_ACT_GROUP_INDEX", gate_up.device, rows),
                    lambda: torch.tensor([rows], dtype=torch.int64, device=gate_up.device),
                )
            act_int8, act_scale = kernels.dequant_swiglu_quant(
                gate_up,
                self.gate_up_proj.weight_scale.reshape(1, -1),
                pertoken,
                group_index=group_index,
                clamp_limit=clamp_limit,
            )
        else:
            act_int8, act_scale = kernels.dequant_swiglu_quant(
                gate_up,
                self.gate_up_proj.weight_scale,
                pertoken,
                clamp_limit=clamp_limit,
            )
        if output is None:
            out = self.down_proj.forward_quantized(act_int8, act_scale)
        else:
            out = self.down_proj.forward_quantized_out(act_int8, act_scale, output)
        return self._reduce_output(out, tp_reduce_add)


class DeepseekV41W8A8MoE(DeepseekV4MoE):
    _SHARED_MLP_CLS = DeepseekV41SharedMLP
    _GATE_BEFORE_DP = True

    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if cfg.n_hash_layers != 0:
            raise ValueError("DeepSeek-V4.1 removed hash routing; n_hash_layers must be 0")
        super().__init__(cfg, layer_id, dtype, device)
        self.enable_fused_mc2 = cfg.enable_fused_mc2
        soc = current_platform.get_ascend_soc_generation()
        self._mc2_capacity = 512 if soc == "a3" else 256
        if self.enable_fused_mc2 and soc == "a3" and self._mc2_ep_size() > 1:
            self._mc2_capacity = Eplv2CommPolicy.from_geometry(
                cfg.hidden_size,
                cfg.n_routed_experts,
                self._mc2_ep_size(),
                cfg.num_experts_per_tok,
                int(os.getenv("HCCL_BUFFSIZE", "200")),
                shared_experts=cfg.n_shared_experts,
            ).mc2_capacity
        self._mc2_ep_group_info = cfg.mc2_group_info
        self.image_token_id = cfg.image_token_id
        self.gate_bias_vl = nn.Parameter(
            torch.empty(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl_loaded = False

    def _apply_post_norm(
        self,
        hidden: torch.Tensor,
        post_norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = post_norm.weight
        if _rms_norm_cast_supported(hidden, weight):
            _log_rms_norm_cast_active()
            return _get_rms_norm_cast_op()(hidden, weight, post_norm.eps)
        _log_rms_norm_cast_fallback(hidden, weight)
        return super()._apply_post_norm(hidden, post_norm)

    def _mc2_ep_size(self) -> int:
        return self.cfg.ep_size if self.cfg.ep_size > 0 else self.cfg.tp_size

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        if self.enable_fused_mc2:
            reason = self._mc2_guard_reason(hidden)
            if reason is None:
                return self._forward_mc2_decode(hidden, post_norm)
            if _moe_mc2_draft_veto_applies(self):
                _log_moe_mc2_draft_fallback(reason)
            else:
                _log_moe_mc2_fallback(reason)
        return super().forward(hidden, input_ids, post_norm)

    def _mc2_guard_reason(self, hidden: torch.Tensor) -> str | None:
        if _moe_mc2_draft_veto_applies(self):
            return "DSpark draft MoE (uneven per-rank rows break global_bs=0)"
        ctx = get_forward_context_or_none()
        metadata = ctx.metadata if ctx is not None else None
        if self.dp_size > 1:
            if not _moe_mc2_all_dp_ranks_decode(metadata):
                return "not all DP ranks are decode"
        elif not _moe_mc2_decode_step():
            return "not a (non-dummy) decode step"
        if self.hash_layer:
            return "hash-routed layer"
        if self.w4a8_dynamic:
            return "W4A8 expert path"
        if self.moe_tp_size != 1:
            return f"moe_tp_size={self.moe_tp_size} (MC2 requires 1)"
        ep_size = self._mc2_ep_size()
        if ep_size <= 1:
            return f"ep_size={ep_size} (MC2 needs a multi-rank EP group)"
        if self.dp_size > 1:
            # Use host-planned DP metadata so all ranks dispatch or fall back together.
            token_counts = getattr(metadata, "dp_execution_token_counts", None)
            if not token_counts or len(token_counts) != self.dp_size:
                return "DP token counts unavailable"
            # An empty DP replica must make every rank fall back; the native operator rejects bs=0.
            if any(int(count) <= 0 for count in token_counts):
                return "an empty DP rank is present"
            capacity = min(self._mc2_capacity, _MOE_MC2_MAX_LOCAL_TOKENS)
            if any(int(count) > capacity for count in token_counts):
                return f"a DP rank exceeds the {capacity}-token MC2 capacity"
            # Unequal DP rows give global_bs=0 incompatible window strides and can corrupt memory.
            row_max = max(int(count) for count in token_counts)
            row_min = min(int(count) for count in token_counts)
            if row_max != row_min:
                return f"uneven DP rows ({row_max} vs {row_min}) break global_bs=0"
        if hidden.dim() != 2:
            return f"hidden shape {tuple(hidden.shape)}"
        if self.dp_size == 1 and hidden.numel() == 0:
            return f"hidden shape {tuple(hidden.shape)}"
        if hidden.dtype != torch.bfloat16:
            return f"hidden dtype {hidden.dtype}"
        if not _moe_mc2_device_supported(hidden):
            return f"device {hidden.device.type}"
        if not _moe_mc2_ops_available():
            return "npu_moe_distribute_{dispatch,combine}_v2 unavailable"
        if self.dp_size == 1 and hidden.shape[0] > self._mc2_capacity:
            return f"local rows exceed the {self._mc2_capacity}-token MC2 capacity"
        if self._mc2_ep_group_info is None:
            return "dedicated MC2 HCCL group unavailable"
        if self._mc2_ep_group_info[2] != ep_size:
            return "MC2 group size does not match the expert partition"
        return None

    def _forward_mc2_decode(
        self,
        hidden: torch.Tensor,
        post_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        import torch_npu

        ep_size = self._mc2_ep_size()
        if self._mc2_ep_group_info is None:
            raise RuntimeError("MC2 decode path entered without a dedicated HCCL group")
        group_ep, ep_rank, _ = self._mc2_ep_group_info
        _log_moe_mc2_collective(ep_rank, ep_size)
        _log_moe_mc2_active(ep_rank, ep_size)

        global_bs = hidden.shape[0] * ep_size if self.dp_size == 1 else 0
        x_active_mask: torch.Tensor | None = None

        hidden, logits = self._prepare_moe_gate(hidden, post_norm)

        norm_type = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}.get(self.scoring_func, 2)
        renorm = 0 if norm_type == 2 else 1
        bias = getattr(self, "e_score_correction_bias", None)
        topk_weights, topk_idx, _ = kernels.moe_gating_top_k_hash(
            x=logits,
            k=self.topk,
            bias=bias,
            input_ids=None,
            tid2eid=None,
            k_group=1,
            group_count=1,
            routed_scaling_factor=self.routed_scaling,
            eps=1e-20,
            group_select_mode=1,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=False,
        )
        expert_ids = topk_idx.to(torch.int32)
        expert_scales = topk_weights.to(torch.float32)

        (
            expand_x,
            dynamic_scales,
            assist_info_for_combine,
            expert_token_nums,
            ep_send_counts,
            tp_send_counts,
            expand_scales,
        ) = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden,
            expert_ids=expert_ids,
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            group_ep=group_ep,
            ep_world_size=ep_size,
            ep_rank_id=ep_rank,
            moe_expert_num=self.num_total_experts,
            group_tp="",
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=int(self.cfg.n_shared_experts),
            shared_expert_rank_num=0,
            quant_mode=0,
            global_bs=global_bs,
            expert_token_nums_type=1,
        )
        del dynamic_scales
        sorted_hidden_i8, pertoken_scale = kernels.dynamic_quant(expand_x)
        group_list = expert_token_nums[: self.num_experts_per_rank].to(torch.int64)
        # Keep dispatch counts on device; reading their sum synchronizes every eager decode step.
        gemm1_out = torch.ops.npu.npu_grouped_matmul(
            x=[sorted_hidden_i8],
            weight=[self.experts_w13],
            scale=None,
            per_token_scale=None,
            split_item=2,
            group_type=0,
            group_list_type=1,
            group_list=group_list,
            output_dtype=torch.int32,
        )[0]
        act_i8, act_pt = kernels.dequant_swiglu_quant(
            x=gemm1_out,
            weight_scale=self.experts_w13_scale,
            activation_scale=pertoken_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=1,
            clamp_limit=self.cfg.swiglu_limit,
            glu_alpha=1.0,
            glu_bias=0.0,
        )
        local_out = torch.ops.npu.npu_grouped_matmul(
            x=[act_i8],
            weight=[self.experts_w2],
            scale=[self.experts_w2_scale],
            per_token_scale=[act_pt],
            split_item=2,
            group_type=0,
            group_list_type=1,
            group_list=group_list,
            output_dtype=hidden.dtype,
        )[0]
        routed_out = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=local_out,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_send_counts,
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            group_ep=group_ep,
            ep_world_size=ep_size,
            ep_rank_id=ep_rank,
            moe_expert_num=self.num_total_experts,
            tp_send_counts=tp_send_counts,
            expand_scales=expand_scales,
            group_tp="",
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=int(self.cfg.n_shared_experts),
            shared_expert_rank_num=0,
            global_bs=global_bs,
        )

        shared_out = self.shared_experts(hidden)
        if self.shared_expert_gate_is_loaded:
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden))
            shared_out = shared_gate * shared_out
        return routed_out + shared_out


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        mode: CSA2LayerMode,
        quant_mode: str = V41_QUANT_NONE,
        engram_layout: EngramLayout | None = None,
        engram_storage: str = ENGRAM_STORAGE_FP8,
        engram_gate_unrotate: torch.Tensor | None = None,
        engram_offload: bool = False,
        engram_dp_shared_memory: bool = False,
        engram_dp_group: Any | None = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.mode = mode
        self.hc = DeepseekV41HyperConnection(cfg, device, dtype=dtype)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.self_attn = DeepseekV41Attention(cfg, layer_id, dtype, device, mode, quant_mode)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.engram: Engram | None = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(
                engram_layout,
                engram_layout.layer_ids.index(layer_id),
                cfg.hidden_size,
                cfg.hc_mult,
                cfg.rms_norm_eps,
                cfg.tp_size,
                cfg.tp_rank,
                dtype,
                device,
                storage=engram_storage,
                gate_unrotate=engram_gate_unrotate,
                offload=engram_offload,
                dp_shared_memory=engram_dp_shared_memory,
                dp_group=engram_dp_group,
            )
        is_dense = (layer_id < cfg.first_k_dense_replace) or (
            cfg.moe_layer_freq > 1 and layer_id % cfg.moe_layer_freq != 0
        )
        if is_dense:
            if quant_mode == V41_QUANT_ASCEND:
                self.mlp: nn.Module = DeepseekV3MLP(
                    cfg,
                    cfg.dense_intermediate_size,
                    dtype,
                    device,
                    swiglu_limit=cfg.swiglu_limit,
                )
            else:
                self.mlp = DeepseekV41TorchMLP(cfg, dtype, device)
        elif quant_mode == V41_QUANT_ASCEND:
            self.mlp = DeepseekV41W8A8MoE(cfg, layer_id, dtype, device)
        else:
            self.mlp = DeepseekV41MoE(cfg, layer_id, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        pre_mix: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pre_mix is None:
            pre_mix = DeepseekV41HyperConnection.make_identity_pre_mix(hidden, self.hc.hc_mult)
        if self.engram is not None and (engram_hashes is not None or engram_mask is not None):
            hidden = self.engram(
                hidden,
                None if engram_hashes is None else engram_hashes[:, self.engram.layer_hash_index],
                engram_mask,
            )
        residual = hidden
        if self.hc._hc_pre_mode == _HC_PRE_MODE_FUSED:
            x, attn_pre, attn_post, attn_comb = self.hc.hc_pre_fused(
                hidden, self.hc.hc_attn_fn, self.hc.hc_attn_scale, self.hc.hc_attn_base, pre_mix
            )
        else:
            attn_pre, attn_post, attn_comb = self.hc.hc_mixes(
                hidden, self.hc.hc_attn_fn, self.hc.hc_attn_scale, self.hc.hc_attn_base
            )
            x = self.hc.hc_pre(hidden, pre_mix)
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions, cos_sin_cache)
        hidden = self.hc.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden
        if self.hc._hc_pre_mode == _HC_PRE_MODE_FUSED:
            x, ffn_pre, ffn_post, ffn_comb = self.hc.hc_pre_fused(
                hidden, self.hc.hc_ffn_fn, self.hc.hc_ffn_scale, self.hc.hc_ffn_base, attn_pre
            )
        else:
            ffn_pre, ffn_post, ffn_comb = self.hc.hc_mixes(
                hidden, self.hc.hc_ffn_fn, self.hc.hc_ffn_scale, self.hc.hc_ffn_base
            )
            x = self.hc.hc_pre(hidden, attn_pre)
        post_norm = self.post_attention_layernorm
        if isinstance(self.mlp, (DeepseekV41MoE, DeepseekV41W8A8MoE)):
            # Let MoE own normalization so both paths use identical local rows and routing weights.
            x = self.mlp(x, input_ids, post_norm=post_norm)
        else:
            x = post_norm(x)
            x = self.mlp(x)
        hidden = self.hc.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden, ffn_pre


class DeepseekV41Model(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
        quant_mode: str = V41_QUANT_NONE,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        if cfg.cp_size > 1:
            if cfg.enable_dsa_cp:
                raise ValueError("enable_dsa_cp cannot be combined with cp_size > 1")
            raise NotImplementedError(
                "DeepSeek-V4.1 context parallelism is scheduled with the Phase-6 distributed work"
            )
        self.cfg = cfg
        self.quant_mode = quant_mode
        self.csa2_plan = CSA2LayerPlan.build(
            cfg.n_layers,
            cfg.compress_ratios,
            cfg.kv_source_layer_ids,
            cfg.index_source_layer_ids,
        )
        self.engram_layout: EngramLayout | None = None
        self.engram_hash: EngramHashState | None = None
        if cfg.engram_layer_ids:
            if cfg.engram_pad_token_id < 0:
                raise ValueError(
                    "DeepSeek-V4.1 engram is enabled (engram_layer_ids="
                    f"{cfg.engram_layer_ids}) but engram_pad_token_id is unset"
                )
            self.engram_layout = EngramLayout(
                cfg.engram_layer_ids,
                cfg.engram_num_embeddings,
                cfg.engram_max_ngram_size,
                cfg.engram_n_heads,
                cfg.engram_head_dim,
                cfg.engram_vocab_size,
                cfg.engram_compressed_vocab_size,
                cfg.engram_pad_token_id,
            )
            self.engram_hash = EngramHashState.from_config(
                self.engram_layout,
                {**(config or {}), "vocab_size": cfg.vocab_size},
            ).to(device)
        self.engram_storage = ENGRAM_STORAGE_INT8 if quant_mode == V41_QUANT_ASCEND else ENGRAM_STORAGE_FP8
        self.engram_offload_active = bool(cfg.engram_layer_ids) and cfg.engram_config.cpu_offload
        if self.engram_offload_active and self.engram_storage != ENGRAM_STORAGE_INT8:
            logger.warning(
                "engram_config.cpu_offload=true requested but engram storage is %r; keeping the table on HBM",
                self.engram_storage,
            )
            self.engram_offload_active = False
        shared_requested = bool(cfg.engram_layer_ids) and cfg.engram_config.dp_shared_memory
        if shared_requested and self.engram_storage != ENGRAM_STORAGE_INT8:
            raise ValueError(
                "engram_config.dp_shared_memory=true requires the int8 (W8A8) engram table "
                f"storage, got {self.engram_storage!r}"
            )
        self.dp_shared_memory = shared_requested and self.engram_offload_active
        self.engram_dp_group: Any | None = None
        if self.dp_shared_memory:
            # A DP replica spans TP x PP x CP; DCP reuses the TP ranks.
            replica_size = cfg.tp_size * cfg.pp_size * cfg.cp_size
            shard_size = engram_dp_shard_size(cfg.dp_size, replica_size)
            if shard_size <= 1:
                raise ValueError(
                    "engram_config.dp_shared_memory=true requires a node-local DP sharing "
                    f"group with more than one rank; got DP={cfg.dp_size}, shard_size={shard_size}."
                )
            # Check the aggregate node footprint once; per-table checks would double-count layers.
            ensure_dev_shm_capacity(self._engram_shared_node_bytes(cfg.tp_size, replica_size, shard_size))
            self.engram_dp_group = init_engram_dp_group(str(device), cfg.dp_size, shard_size)
        self.engram_gate_unrotate = None
        if cfg.engram_layer_ids and quant_mode == V41_QUANT_ASCEND:
            self.engram_gate_unrotate = load_quarot_gate_unrotate(str((config or {}).get("model_path") or ""), device)
        tp = cfg.tp_size
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(
                    cfg,
                    i,
                    dtype,
                    device,
                    self.csa2_plan.mode(i),
                    quant_mode,
                    self.engram_layout,
                    engram_storage=self.engram_storage,
                    engram_gate_unrotate=self.engram_gate_unrotate,
                    engram_offload=self.engram_offload_active,
                    engram_dp_shared_memory=self.dp_shared_memory,
                    engram_dp_group=self.engram_dp_group,
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.rotary = RotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=torch.float32,
            device="cpu",
        ).to(device)
        self.compress_rotary = DeepseekV4RotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.compress_rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.original_max_position_embeddings,
            dtype=torch.float32,
            device=device,
        )
        if self.engram_offload_active:
            assert self.engram_hash is not None
            self.execution_metadata_builders = (
                EngramExecutionMetadataBuilder(
                    self.engram_hash,
                    [layer.engram for layer in self.layers if layer.engram is not None],
                    cfg.window_size,
                ),
            )

    def _layer_rope_cache(self, mode: CSA2LayerMode) -> torch.Tensor:
        """Per-layer RoPE table: compress theta for CSA2 layers, plain otherwise."""
        return self.compress_rotary.cos_sin_cache if mode.is_csa2 else self.rotary.cos_sin_cache

    def _compute_engram_hashes(
        self,
        backend: Any,
        metadata: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.engram_hash is None:
            raise RuntimeError(
                "DeepSeek-V4.1 engram hashing requested without a built engram hash state; "
                "the model must configure engram layers to hash the token stream"
            )
        dsa = getattr(metadata, "dsa_metadata", None)
        swa_group = backend.swa_group_metadata(dsa)
        if swa_group is None or not self.engram_hash.ensure_cache(swa_group.cache):
            return None, None
        seq_lens_q = dsa.seq_lens_q.reshape(-1).to(torch.int64)
        q_cu = torch.cat((torch.zeros(1, dtype=torch.int64, device=seq_lens_q.device), seq_lens_q.cumsum(0)))
        dead = image_sentinel_mask(input_ids)
        hashes = self.engram_hash(
            input_ids,
            positions,
            q_cu,
            dead,
            None,
            swa_group.slot_mapping,
            swa_group.block_table,
            swa_group.block_size,
        )
        return hashes, ~dead

    def _engram_shared_node_bytes(self, tp_size: int, replica_size: int, shard_size: int, quant_block: int = 32) -> int:
        assert self.engram_layout is not None
        head_dim = self.engram_layout.head_dim
        per_layer = 0
        for layer_hash_index in range(len(self.engram_layout.layer_ids)):
            head_sizes = self.engram_layout.head_sizes(layer_hash_index)
            part = (len(head_sizes) + tp_size - 1) // tp_size
            for shard in range(tp_size):
                start = shard * part
                rows = sum(head_sizes[start : start + part])
                if rows:
                    per_layer += estimate_shared_bytes(rows, head_dim, quant_block)
        copies = 1
        topology = world_topology()
        if topology is not None and replica_size > 0 and shard_size > 0:
            my_host = socket.gethostname()
            ranks_on_node = sum(1 for entry in topology if entry.get("hostname") == my_host)
            replicas_per_node = ranks_on_node // replica_size
            if replicas_per_node > 0:
                copies = max(replicas_per_node // shard_size, 1)
        return per_layer * copies

    def _reconcile_engram_offload_active(self) -> bool:
        if not self.engram_offload_active or self.dp_shared_memory:
            return False
        engram_layers = [layer.engram for layer in self.layers if getattr(layer, "engram", None) is not None]
        fell_back = [engram for engram in engram_layers if not engram.embed_tokens.offload]
        if not fell_back:
            return False
        for engram in engram_layers:
            if engram.embed_tokens.offload:
                engram.embed_tokens.fallback_to_device()
        self.engram_offload_active = False
        self.execution_metadata_builders = ()
        logger.warning(
            "engram host offload disabled: %d/%d engram layer(s) failed the host-table load; "
            "falling back the whole model to the HBM table path",
            len(fell_back),
            len(engram_layers),
        )
        return True

    def close_host_offload(self) -> None:
        failures: list[str] = []
        for layer_id, layer in enumerate(self.layers):
            engram = getattr(layer, "engram", None)
            if engram is None:
                continue
            try:
                engram.close_host_offload()
            except Exception as exc:  # noqa: BLE001 - aggregated; keep releasing the rest
                failures.append(f"layer {layer_id}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError("engram host-table close failed for " + "; ".join(failures))

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        context = get_forward_context()
        backend = context.attention_backend
        metadata = context.metadata
        graph_mode = bool(getattr(metadata, "dsa_graph_mode", False))
        if not (graph_mode and getattr(metadata, "dsa_metadata", None) is not None):
            backend.reset_forward(metadata)
            metadata.dsa_graph_mode = graph_mode
            metadata.dsa_positions = positions
            prepare_dsa = getattr(backend, "prepare_dsa_metadata_for_forward", None)
            if prepare_dsa is None:
                raise RuntimeError("DeepSeek-V4.1 requires prepare_dsa_metadata_for_forward")
            prepare_dsa(metadata)

        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if self.engram_hash is not None:
            if self.engram_offload_active:
                engram_execution = get_execution_context(EngramExecutionMetadata)
                if engram_execution is None:
                    raise RuntimeError("Engram host offload requires execution metadata from the runner")
                engram_mask = engram_execution.gate_mask
            else:
                engram_hashes, engram_mask = self._compute_engram_hashes(backend, metadata, input_ids, positions)
                if engram_hashes is not None:
                    for layer in self.layers:
                        if layer.engram is not None:
                            layer.engram.prepare_embeddings(
                                engram_hashes[:, layer.engram.layer_hash_index].to(torch.int64)
                            )

        hidden = hidden.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).contiguous()
        pre_mix = DeepseekV41HyperConnection.make_identity_pre_mix(hidden, self.cfg.hc_mult)
        for layer_id, layer in enumerate(self.layers):
            layer_cos_sin_cache = self._layer_rope_cache(layer.mode)
            hidden, pre_mix = layer(
                hidden,
                positions,
                layer_cos_sin_cache,
                input_ids,
                pre_mix,
                engram_hashes,
                engram_mask,
            )
            record_layer_event(layer_id)
        merged = self.layers[-1].hc.hc_pre(hidden, pre_mix)
        return self.norm(merged)


class DeepseekV41ForCausalLM(PyModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = DeepseekV41Config.from_dict(config)
        self.quant_mode = _detect_v41_quant_mode(config)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        self.model = DeepseekV41Model(self.cfg, dtype, device, self.quant_mode, config)
        tp = self.cfg.tp_size
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        if self.quant_mode == V41_QUANT_ASCEND:
            self._load_weights_ascend(state_dicts)
        else:
            self._load_weights_torch(state_dicts)
        # A failed private host load must demote every Engram layer, not leave mixed paths.
        self.model._reconcile_engram_offload_active()

    def close_host_offload(self) -> None:
        self.model.close_host_offload()

    def _load_weights_torch(self, state_dicts: list[StateDict]) -> None:
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)
        is_fp8 = self.quant_mode == V41_QUANT_FP8

        def _linear(ckpt_prefix: str, param_prefix: str, shard_dim: int | None = None) -> None:
            if is_fp8:
                weight = loader.get_tensor(ckpt_prefix + ".weight")
                scale = loader.get_tensor(ckpt_prefix + ".scale")
                weight = dequant_fp8_block(weight, scale)
            else:
                weight = loader.get_tensor(ckpt_prefix + ".weight")
            if shard_dim is not None:
                weight = loader.shard(weight, dim=shard_dim)
            loader.copy_in(param_prefix + ".weight", weight)

        loader.copy_in(
            "model.embed_tokens.weight",
            loader.shard(loader.get_tensor("embed.weight"), dim=1),
        )
        for i in range(cfg.n_layers):
            ck = f"layers.{i}."
            pm = f"model.layers.{i}."
            attn = self.model.layers[i].self_attn
            for ckpt_name, param_name, shard_dim in (
                ("attn.wq_a", "self_attn.q_a_proj", None),
                ("attn.wq_b", "self_attn.q_b_proj", None if attn.enable_dsa_cp else 0),
                ("attn.wkv", "self_attn.kv_proj", None),
                ("attn.wo_a", "self_attn.o_a_proj", 0),
                ("attn.wo_b", "self_attn.o_b_proj", 1),
            ):
                weight_key = ck + ckpt_name + ".weight"
                if not loader.has(weight_key):
                    raise KeyError(
                        f"required DeepSeek-V4.1 attention weight '{weight_key}' "
                        f"is missing from the checkpoint (decoder layer {i})"
                    )
                _linear(ck + ckpt_name, pm + param_name, shard_dim)
            loader.copy_in(
                pm + "self_attn.q_a_layernorm.weight",
                loader.get_tensor(ck + "attn.q_norm.weight"),
            )
            loader.copy_in(
                pm + "self_attn.kv_a_layernorm.weight",
                loader.get_tensor(ck + "attn.kv_norm.weight"),
            )
            self._load_attention_extras(loader, ck, pm, attn, _linear)
            mlp = self.model.layers[i].mlp
            if isinstance(mlp, DeepseekV41MoE):
                self._load_torch_moe(loader, ck, pm, i, is_fp8)
            elif isinstance(mlp, DeepseekV41TorchMLP):
                for name in ("w1", "w2", "w3"):
                    _linear(ck + f"ffn.{name}", pm + f"mlp.{name}")
            if self.model.layers[i].engram is not None:
                self._load_engram_torch(loader, ck, pm, i, is_fp8)
        loader.copy_in("model.norm.weight", loader.get_tensor("norm.weight"))
        self._load_lm_head(loader)

    def _load_torch_moe(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        layer_id: int,
        is_fp8: bool,
    ) -> None:
        mlp = self.model.layers[layer_id].mlp
        loader.copy_in(pm + "mlp.gate.weight", loader.get_tensor(ck + "ffn.gate.weight"))
        loader.copy_in(
            pm + "mlp.e_score_correction_bias",
            loader.get_tensor(ck + "ffn.gate.bias"),
        )
        if loader.has(ck + "ffn.gate.bias_vl"):
            loader.copy_in(pm + "mlp.gate_bias_vl", loader.get_tensor(ck + "ffn.gate.bias_vl"))
            mlp.gate_bias_vl_loaded = True
        gate_and_up_slots = mlp.experts_w13.data
        inter = gate_and_up_slots.shape[1] // 2
        for local_slot in range(mlp.num_experts_per_rank):
            expert = mlp.start_expert_id + local_slot
            for name, target in (
                ("w1", gate_and_up_slots[local_slot, :inter]),
                ("w2", mlp.experts_w2.data[local_slot]),
                ("w3", gate_and_up_slots[local_slot, inter:]),
            ):
                prefix = ck + f"ffn.experts.{expert}.{name}"
                if is_fp8:
                    weight = dequant_fp4_group(
                        loader.get_tensor(prefix + ".weight"),
                        loader.get_tensor(prefix + ".scale"),
                    )
                else:
                    weight = loader.get_tensor(prefix + ".weight")
                target.copy_(weight.to(target.dtype))
        shared_prefix = ck + "ffn.shared_experts."
        for name, module_name in (
            ("w1", "shared_w1"),
            ("w2", "shared_w2"),
            ("w3", "shared_w3"),
        ):
            prefix = shared_prefix + name
            if is_fp8:
                weight = dequant_fp8_block(
                    loader.get_tensor(prefix + ".weight"),
                    loader.get_tensor(prefix + ".scale"),
                )
            else:
                weight = loader.get_tensor(prefix + ".weight")
            loader.copy_in(pm + f"mlp.{module_name}.weight", weight)

    def _load_engram_torch(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        layer_id: int,
        is_fp8: bool,
    ) -> None:
        eng = self.model.layers[layer_id].engram
        if eng is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 layer {layer_id} has no engram module to load the checkpoint engram tensors into"
            )
        table = loader.get_tensor(ck + "engram.embed.weight")
        table_u8 = table.view(torch.uint8) if table.element_size() == 1 else table
        scale = loader.get_tensor(ck + "engram.embed.scale")
        scale_u8 = scale.view(torch.uint8) if scale.element_size() == 1 else scale
        start = eng.embed_tokens.vocab_start
        rows = eng.embed_tokens.part_num_embeddings
        if table_u8.size(0) < eng.embed_tokens.num_embeddings:
            raise ValueError(
                f"engram table for layer {layer_id} has {table_u8.size(0)} rows, "
                f"config declares {eng.embed_tokens.num_embeddings}"
            )
        if rows > 0:
            local_scale = scale_u8.narrow(0, start, rows)
            _validate_ue8m0_scale_bytes(local_scale)
            self._copy_engram_rows(eng.embed_tokens, table_u8, scale_u8, start, rows)
        if is_fp8:
            wkv = dequant_fp8_block(
                loader.get_tensor(ck + "engram.wkv.weight"),
                loader.get_tensor(ck + "engram.wkv.scale"),
            )
        else:
            wkv = loader.get_tensor(ck + "engram.wkv.weight")
        loader.copy_in(pm + "engram.wkv.weight", wkv)
        loader.copy_in(pm + "engram.q_weight", loader.get_tensor(ck + "engram.q_weight"))
        loader.copy_in(pm + "engram.k_weight", loader.get_tensor(ck + "engram.k_weight"))

    def _load_weights_ascend(self, state_dicts: list[StateDict]) -> None:
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)

        def _w8a8(
            ckpt_prefix: str,
            param_prefix: str,
            shard_dims: dict | None = None,
            required: bool = False,
        ) -> None:
            load_w8a8_dynamic_projection(
                loader,
                ckpt_prefix,
                param_prefix,
                shard_dims,
                require_weight_and_scale=required,
                fill_missing_offset=True,
                description=f"DeepSeek-V4.1 W8A8 {ckpt_prefix}",
            )

        loader.copy_in(
            "model.embed_tokens.weight",
            loader.shard(loader.get_tensor("embed.weight"), dim=1),
        )
        for i in range(cfg.n_layers):
            ck = f"layers.{i}."
            pm = f"model.layers.{i}."
            attn = self.model.layers[i].self_attn
            _w8a8(ck + "attn.wq_a", pm + "self_attn.q_a_proj", required=True)
            _w8a8(
                ck + "attn.wq_b",
                pm + "self_attn.q_b_proj",
                None if attn.enable_dsa_cp else {"weight": 0, "weight_scale": 0, "weight_offset": 0},
                required=True,
            )
            _w8a8(ck + "attn.wkv", pm + "self_attn.kv_proj", required=True)
            loader.copy_in(
                pm + "self_attn.o_a_proj.weight",
                loader.shard(loader.get_tensor(ck + "attn.wo_a.weight"), dim=0),
            )
            loader.copy_in(
                pm + "self_attn.o_b_proj.weight",
                loader.shard(loader.get_tensor(ck + "attn.wo_b.weight"), dim=1),
            )
            loader.copy_in(
                pm + "self_attn.q_a_layernorm.weight",
                loader.get_tensor(ck + "attn.q_norm.weight"),
            )
            loader.copy_in(
                pm + "self_attn.kv_a_layernorm.weight",
                loader.get_tensor(ck + "attn.kv_norm.weight"),
            )
            self._load_attention_extras(
                loader, ck, pm, attn, lambda source, target: _w8a8(source, target, required=True)
            )
            attn.process_weights_after_loading()
            mlp = self.model.layers[i].mlp
            if isinstance(mlp, DeepseekV41W8A8MoE):
                self._load_ascend_moe(loader, ck, pm, i)
                mlp.process_weights_after_loading()
            elif isinstance(mlp, DeepseekV3MLP):
                DeepseekV4ForCausalLM._load_dsv4_dense_mlp(loader, ck, pm, mlp)
            if self.model.layers[i].engram is not None:
                self._load_engram_ascend(loader, ck, pm, i)
        loader.copy_in("model.norm.weight", loader.get_tensor("norm.weight"))
        self._load_lm_head(loader)

    def _load_attention_extras(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        attn: DeepseekV41Attention,
        load_indexer_projection: Callable[[str, str], None],
    ) -> None:
        cfg = self.cfg
        sink_key = ck + "attn.attn_sink"
        if not loader.has(sink_key):
            sink_key = ck + "attn.attn_sink.weight"
        if loader.has(sink_key):
            sink = loader.get_tensor(sink_key)
            if sink.dim() == 1 and sink.size(0) == cfg.n_heads and cfg.tp_size > 1 and not attn.enable_dsa_cp:
                sink = loader.shard(sink, dim=0)
            loader.copy_in(pm + "self_attn.attn_sink", sink)
            attn.attn_sink_loaded = True
        self._load_common_layer_norms(loader, ck, pm)
        if hasattr(attn, "cmp_wkv"):
            loader.copy_in(
                pm + "self_attn.cmp_wkv.weight",
                loader.get_tensor(ck + "attn.compressor.wkv.weight"),
            )
            if hasattr(attn, "cmp_wgate"):
                loader.copy_in(
                    pm + "self_attn.cmp_wgate.weight",
                    loader.get_tensor(ck + "attn.compressor.wgate.weight"),
                )
            loader.copy_in(
                pm + "self_attn.cmp_norm.weight",
                loader.get_tensor(ck + "attn.compressor.norm.weight"),
            )
        if attn.indexer is not None:
            load_indexer_projection(ck + "attn.indexer.wq_b", pm + "self_attn.indexer.wq_b")
            loader.copy_in(
                pm + "self_attn.indexer.weights_proj.weight",
                loader.get_tensor(ck + "attn.indexer.weights_proj.weight"),
            )
            if attn.indexer.owns_k:
                loader.copy_in(
                    pm + "self_attn.indexer.wk.weight",
                    loader.get_tensor(ck + "attn.indexer.wk.weight"),
                )
                loader.copy_in(
                    pm + "self_attn.indexer.k_norm.weight",
                    loader.get_tensor(ck + "attn.indexer.k_norm.weight"),
                )

    def _load_engram_ascend(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        layer_id: int,
    ) -> None:
        eng = self.model.layers[layer_id].engram
        if eng is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 layer {layer_id} has no engram module to load the checkpoint engram tensors into"
            )
        embed = eng.embed_tokens
        if embed._shared_storage is not None:
            self._load_engram_ascend_shared(loader, ck, layer_id, embed)
        else:
            self._load_engram_ascend_private(loader, ck, layer_id, embed)
        loader.copy_in(pm + "engram.wkv.weight", loader.get_tensor(ck + "engram.wkv.weight"))
        loader.copy_in(pm + "engram.q_weight", loader.get_tensor(ck + "engram.q_weight"))
        loader.copy_in(pm + "engram.k_weight", loader.get_tensor(ck + "engram.k_weight"))

    @staticmethod
    def _validate_engram_table_dtypes(layer_id: int, table: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if table.dtype == torch.uint8:
            table = table.view(torch.int8)
        if table.dtype != torch.int8:
            raise NotImplementedError(
                f"engram table for layer {layer_id} has dtype {table.dtype}; the "
                "ascend loader implements the int8_sym export only (fp32 group "
                "scales, see quant_model_description.json optional.embedding_storage)"
            )
        if scale.dtype != torch.float32:
            raise NotImplementedError(
                f"engram table scale for layer {layer_id} has dtype {scale.dtype}; "
                "expected float32 group scales of the int8_sym export"
            )
        return table

    @staticmethod
    def _copy_engram_rows(
        embed: EngramEmbedding, table: torch.Tensor, scale: torch.Tensor, start: int, rows: int
    ) -> None:
        if rows <= 0:
            return
        embed.weight.data.copy_(table.narrow(0, start, rows))
        embed.weight_scale.data.copy_(scale.narrow(0, start, rows))

    def _load_engram_ascend_table(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        layer_id: int,
        embed: EngramEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        table = loader.get_tensor(ck + "engram.embed.weight")
        scale = loader.get_tensor(ck + "engram.embed.scale")
        table = self._validate_engram_table_dtypes(layer_id, table, scale)
        start = embed.vocab_start
        rows = embed.part_num_embeddings
        if table.size(0) < embed.num_embeddings:
            raise ValueError(
                f"engram table for layer {layer_id} has {table.size(0)} rows, config declares {embed.num_embeddings}"
            )
        return table, scale, start, rows

    def _load_engram_ascend_private(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        layer_id: int,
        embed: EngramEmbedding,
    ) -> None:
        table, scale, start, rows = self._load_engram_ascend_table(loader, ck, layer_id, embed)
        # Copy directly from mmap; an intermediate device tensor would double the peak table allocation.
        try:
            self._copy_engram_rows(embed, table, scale, start, rows)
        except (RuntimeError, MemoryError) as exc:
            # Catch only host-copy failures; unrelated loading errors must propagate.
            if not embed.offload:
                raise
            logger.warning(
                "engram layer %d host-table load failed (%s); falling back to "
                "the HBM table path for this layer (the model then demotes all "
                "offload layers to HBM)",
                layer_id,
                exc,
                exc_info=True,
            )
            embed.fallback_to_device()
            try:
                self._copy_engram_rows(embed, table, scale, start, rows)
            except (RuntimeError, MemoryError) as exc:
                raise RuntimeError(
                    f"engram layer {layer_id}: HBM table copy failed even "
                    f"after the host->HBM fallback ({exc}); the engram table "
                    "is only partially populated and the model cannot serve "
                    "correctly, aborting instead of running with a "
                    "half-loaded table"
                ) from exc

    def _load_engram_ascend_shared(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        layer_id: int,
        embed: EngramEmbedding,
    ) -> None:
        storage = embed._shared_storage
        group = embed._dp_group
        assert storage is not None and group is not None
        error: str | None = None
        if group.rank() == 0:
            try:
                table, scale, start, rows = self._load_engram_ascend_table(loader, ck, layer_id, embed)
                self._copy_engram_rows(embed, table, scale, start, rows)
            except Exception as exc:  # noqa: BLE001 - propagated to every rank below
                error = f"{type(exc).__name__}: {exc}"
        errors: list[str | None] = [None] * group.size()
        engram_all_gather_object(errors, error, group=group)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        # Fence success and failure alike before any rank unwinds or enters the next collective.
        try:
            engram_barrier(group=group, timeout=_ENGRAM_SHARED_LOAD_BARRIER_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            storage.mark_failed()
            raise RuntimeError(
                f"Engram shared host-table load fence failed for layer {layer_id} "
                f"(phase=barrier, group_rank={group.rank()}): {type(exc).__name__}: {exc}"
            ) from exc
        if failures:
            storage.mark_failed()
            raise RuntimeError(f"Engram shared host-table load failed for layer {layer_id}: {failures}")
        storage.mark_loaded()

    def _load_ascend_moe(self, loader: W8A8WeightLoader, ck: str, pm: str, layer_id: int) -> None:
        source_prefix = _resolve_dsv4_moe_prefix(loader, ck, f"DeepSeek-V4.1 layer {layer_id}")
        mlp = self.model.layers[layer_id].mlp
        DeepseekV4ForCausalLM._load_dsv4_moe(mlp, loader, ck, pm, layer_id, source_prefix)
        bias_vl_key = source_prefix + "gate.bias_vl"
        if loader.has(bias_vl_key):
            loader.copy_in(pm + "mlp.gate_bias_vl", loader.get_tensor(bias_vl_key))
            mlp.gate_bias_vl_loaded = True

    def _load_common_layer_norms(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
    ) -> None:
        loader.copy_in(
            pm + "input_layernorm.weight",
            loader.get_tensor(ck + "attn_norm.weight"),
        )
        loader.copy_in(
            pm + "post_attention_layernorm.weight",
            loader.get_tensor(ck + "ffn_norm.weight"),
        )
        for part in ("attn", "ffn"):
            for suffix in ("fn", "scale", "base"):
                name = f"hc_{part}_{suffix}"
                loader.copy_in(pm + "hc." + name, loader.get_tensor(ck + name))

    def _load_lm_head(self, loader: W8A8WeightLoader) -> None:
        lm_head_key = _require_checkpoint_key(
            loader,
            ("lm_head.weight", "model.lm_head.weight", "model.head.weight", "head.weight"),
            "checkpoint output-head weight",
        )
        loader.copy_in(
            "lm_head.weight",
            loader.shard(loader.get_tensor(lm_head_key), dim=0),
        )


_FP4_NIBBLE_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def dequant_fp8_block(
    weight_e4m3: torch.Tensor,
    scale_u8: torch.Tensor,
    block: tuple[int, int] = (32, 32),
) -> torch.Tensor:
    if weight_e4m3.ndim != 2:
        raise ValueError(f"weight must be 2-D [out, in], got ndim={weight_e4m3.ndim}")
    if len(block) != 2 or block[0] <= 0 or block[1] <= 0:
        raise ValueError(f"block must be a pair of positive ints, got {block}")
    block_out, block_in = int(block[0]), int(block[1])
    out_dim, in_dim = weight_e4m3.shape
    expected_shape = (
        (out_dim + block_out - 1) // block_out,
        (in_dim + block_in - 1) // block_in,
    )
    if tuple(scale_u8.shape) != expected_shape:
        raise ValueError(
            f"scale shape {tuple(scale_u8.shape)} does not match expected "
            f"{expected_shape} for weight [{out_dim}, {in_dim}] with block {block}"
        )
    scale = ue8m0_to_scale(scale_u8)
    scale = scale.repeat_interleave(block_out, dim=0)
    scale = scale.repeat_interleave(block_in, dim=1)[:out_dim, :in_dim]
    return (_e4m3_to_float(weight_e4m3) * scale).to(torch.bfloat16)


def fp4_nibble_table(device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:
    """E2M1 nibble decoding includes both positive and negative zero."""
    return torch.tensor(_FP4_NIBBLE_VALUES, dtype=torch.float32, device=device)


def unpack_fp4(
    weight_i8: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if weight_i8.dtype == torch.uint8:
        packed = weight_i8
    elif weight_i8.element_size() == 1:
        packed = weight_i8.view(torch.uint8)
    else:
        raise ValueError(f"expected 1-byte packed fp4 storage, got {weight_i8.dtype}")
    table = fp4_nibble_table(device=packed.device)
    low = table[(packed & 0x0F).to(torch.int64)]
    high = table[(packed >> 4).to(torch.int64)]
    unpacked = torch.stack((low, high), dim=-1).flatten(-2)
    return unpacked.to(dtype)


def dequant_fp4_group(
    weight_i8: torch.Tensor,
    scale_u8: torch.Tensor,
    group: int = 32,
) -> torch.Tensor:
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")
    unpacked = unpack_fp4(weight_i8, dtype=torch.float32)
    in_dim = unpacked.size(-1)
    expected_groups = (in_dim + group - 1) // group
    if scale_u8.size(-1) != expected_groups:
        raise ValueError(
            f"scale has {scale_u8.size(-1)} groups, expected {expected_groups} for K={in_dim} with group={group}"
        )
    scale = ue8m0_to_scale(scale_u8)
    scale = scale.repeat_interleave(group, dim=-1)[..., :in_dim]
    return (unpacked * scale).to(torch.bfloat16)
