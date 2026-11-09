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

"""DeepSeek-V4.1 Flash Python model (CED + CSA2, TORCH backend).

The forward uses torch on NPU (there is no CPU/CUDA attention backend).
The reference QDQ
numerics are preserved by the inlined quantization helpers at the bottom of
this file (device-agnostic, unit-tested on CPU; ue8m0 decode, FP8 block
dequant, MXFP4 dequant, activation QDQ):

  * ``CSA2LayerPlan`` -- static Full/Reindex/Reuse/SWA assignment. Every
    kv-source layer compresses its OWN attention input (there is no
    cross-layer CED hidden capture in the reference); the group's Reindex /
    Reuse layers read that source's caches (``main_kv_source_layer``).
  * ``DeepseekV41Attention`` -- latent Q/KV projections + interleaved-pair
    RoPE, the ratio-1/ratio-2 compressor weights, the CSA2 indexer weights
    (``wk`` acts on the 512-d latent, not the hidden), and the grouped
    ``wo_a``/``wo_b`` output projection with output de-rotation.
  * ``Csa2AttentionBackend`` (``xllm/python/attention/csa2_attention.py``)
    owns the cache orchestration: SWA ring, state parking/pooling, index
    writes, scoring, candidate pool, top-k and the sink softmax.
  * ``DeepseekV41HyperConnection`` -- single-pass Mega-mHC with the
    Sinkhorn split; attention collapses with the previous block's
    ``ffn_pre``, the FFN with this block's ``attn_pre``, and the final
    merge is the learned ``hc_pre(h, last_ffn_pre)`` weighted sum.
  * ``DeepseekV41MoE`` -- fp32 sqrtsoftplus routing with bias-steered
    selection (``gate.bias`` / per-token ``gate.bias_vl``), clamped SwiGLU
    experts and one shared expert.
  * ``DeepseekV41ForCausalLM`` -- checkpoint-mode-aware construction and
    loading: ``fp8`` (official FP8-block/MXFP4 release, dequantized to bf16
    at load), ``ascend`` (W8A8 export, existing kernel path) and ``none``.

Vision and the DSpark draft model are out of scope here; their checkpoint
tensors are simply not consumed. Engram IS implemented
(:mod:`deepseek_v41_engram`): the layers listed in ``engram_layer_ids``
carry the gated n-gram lookup injected into the mHC residual stream, with
the tokenizer-derived hash state loaded from the
``engram_hash_state.bin`` sidecar.
"""

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
    get_execution_context,
    get_forward_context,
    in_acl_graph,
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
    _E4M3_SUBNORMAL_STEP,
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
    # -- Vision (Phase 4); image_token_id also drives gate.bias_vl --
    # -1 = no image tokens (text-only), matching the C++ loader default.
    image_token_id: int = -1
    # -- Engram host-offload switches (from --engram_config) --
    # Parsed strictly from the raw ``engram_config`` JSON string by
    # EngramConfig.from_value; {} (the C++ default) disables both switches.
    engram_config: EngramConfig = field(default_factory=EngramConfig)
    # -- Parallel / weight-format contract --
    # Passed in by the C++ executor (py_causal_lm.cpp). Engram currently
    # requires PP=1 and DCP=1 and is only validated against safetensors/dummy
    # weights; the values are surfaced here so config validation and the
    # Python executor read them from one place.
    pp_size: int = 1
    dcp_size: int = 1

    @property
    def dense_intermediate_size(self) -> int:
        """Match the C++ dense-width resolution for explicit and omitted values."""
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


# ---------------------------------------------------------------------------
# RoPE tables / mHC
# ---------------------------------------------------------------------------


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


class DeepseekV41HyperConnection(nn.Module):
    """Single-pass Mega-mHC hyper-connections (semantics doc section 7).

    Reference mHC mixing and pre-reduction with shared fused ``hc_post``
    expansion: ``hc_mixes`` projects the flattened hc*dim stream
    (normalized AFTER projection), ``hc_pre`` collapses the hc copies with a
    per-token coefficient vector, and ``hc_post`` expands a sublayer output
    back mixing the residual through the doubly-stochastic ``comb``.
    """

    def __init__(self, cfg: DeepseekV41Config, device: torch.device) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_eps = cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        self.sinkhorn_iters = cfg.hc_sinkhorn_iters
        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.hidden_size
        # hc_fn/scale/base per sub-block, fp32 as in the checkpoint.
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
        """Expand back: ``out[j] = post[j] * x + sum_k comb[j, k] * res[k]``."""
        if x.numel() == 0:
            # The native kernel requires non-empty dimensions; empty DP
            # ranks still participate in the surrounding collectives.
            return residual
        if self.hc_mult != 4 or x.size(-1) % 64 != 0:
            # The shared NPU kernel tiles 64 hidden elements and four streams.
            # Keep small synthetic models exact without changing its fast path.
            expanded = post.float().unsqueeze(-1) * x.float().unsqueeze(-2)
            mixed = torch.einsum("tjk,tkd->tjd", comb.float(), residual.float())
            return (expanded + mixed).to(x.dtype)
        # The shared kernel multiplies residual @ comb; V4.1 stores the
        # opposite matrix orientation. Transpose only this small hc x hc
        # matrix, retaining the fused FP32 accumulation for hidden states.
        return kernels.hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.transpose(-2, -1).contiguous().unsqueeze(0),
        ).squeeze(0)

    @staticmethod
    def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
        """Initial one-hot mix on copy 0 (model.py:1159-1163)."""
        pre_mix = torch.zeros(x.size(0), hc_mult, dtype=torch.float32, device=x.device)
        pre_mix[:, 0] = 1.0
        return pre_mix


# ---------------------------------------------------------------------------
# Indexer weights
# ---------------------------------------------------------------------------


class DeepseekV41Indexer(nn.Module):
    """DeepSeek-V4.1 CSA2 indexer weights (semantics doc section 5).

    ``wq_b`` projects the shared q-lora latent; ``weights_proj`` produces the
    per-head score weights; kv-source layers additionally own the key path
    ``wk`` + ``k_norm`` -- ``wk`` acts on the 512-d compressor latent, NOT on
    hidden states (checkpoint ``layers.N.attn.indexer.*``). There is no
    Hadamard transform and no INT8 quant in V4.1: Q/K are FP4-E2M1 QDQ and
    the scoring runs through einsum in the backend.
    """

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
        num_heads = cfg.n_heads // tp
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
        # Q path: qr = q_norm(wq_a(x)); q = wq_b(qr) (no norm after wq_b);
        # RoPE on the last rope dims at token positions (model.py:770-772).
        qr = self.q_a_layernorm(self.q_a_proj(hidden))
        q = self.q_b_proj(qr).view(num_tokens, self.num_heads_local, self.head_dim)
        q = apply_interleaved_rope(q, positions, cos_sin_cache)
        # SWA KV path: kv_norm -> RoPE -> FP8 QDQ over the whole post-RoPE
        # vector (model.py:700-720).
        kv = self.kv_a_layernorm(self.kv_proj(hidden))
        kv = apply_interleaved_rope(kv, positions, cos_sin_cache)
        kv = fp8_act_qdq(kv, 32)
        self._csa2_ctx = Csa2LayerContext(
            x=hidden,
            qr=qr,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
        )
        attn_out = execute_csa2(q, kv, self)
        # De-rotate the rope dims by the query's own rotation (conjugate)
        # BEFORE the o path (model.py:781).
        attn_out = apply_interleaved_rope(attn_out, positions, cos_sin_cache, inverse=True)
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

    def indexer_query(self, qr: torch.Tensor) -> torch.Tensor:
        """Indexer Q from the shared q-lora latent: wq_b(qr)."""
        return self.indexer.wq_b(qr).view(-1, self.indexer.n_heads, self.indexer.head_dim)

    def indexer_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head scoring weights: weights_proj(x) * (ihd^-0.5 * nh^-0.5)."""
        scale = self.indexer.head_dim**-0.5 * self.indexer.n_heads**-0.5
        return self.indexer.weights_proj(x) * scale


# ---------------------------------------------------------------------------
# MoE (torch) + dense MLP + W8A8 MoE (ascend)
# ---------------------------------------------------------------------------


class DeepseekV41TorchMLP(nn.Module):
    """Dense SwiGLU FFN with the reference clamps (fp8/none checkpoint modes)."""

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

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
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


class DeepseekV41W8A8MoE(DeepseekV4MoE):
    """Ascend W8A8-export MoE (existing V4 kernel path, NPU-only).

    Keeps the bias-based ``moe_gating_top_k_hash`` routing and the grouped
    W8A8 expert GEMMs; the visual-token ``bias_vl`` selection bias of the
    official checkpoint is loaded but only applied by the torch MoE (the
    multimodal path lands with the vision phase).
    """

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
        # Deferred until the vision phase: the W8A8 expert path never reads
        # image_token_id or gate_bias_vl; both are staged for the torch-MoE
        # multimodal path only (see the class docstring).
        self.image_token_id = cfg.image_token_id
        self.gate_bias_vl = nn.Parameter(
            torch.empty(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl_loaded = False


# ---------------------------------------------------------------------------
# Decoder layer + model
# ---------------------------------------------------------------------------


class DeepseekV41DecoderLayer(nn.Module):
    """DeepSeek-V4.1 decoder layer with the single-pass mHC shift.

    Attention collapses with the PREVIOUS block's ``ffn_pre`` and the FFN
    with THIS block's ``attn_pre``; the block returns its ``ffn_pre`` for the
    next block's attention (model.py:968-994).
    """

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
        self.hc = DeepseekV41HyperConnection(cfg, device)
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
        attn_pre, attn_post, attn_comb = self.hc.hc_mixes(
            hidden, self.hc.hc_attn_fn, self.hc.hc_attn_scale, self.hc.hc_attn_base
        )
        x = self.hc.hc_pre(hidden, pre_mix)
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions, cos_sin_cache)
        hidden = self.hc.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden
        ffn_pre, ffn_post, ffn_comb = self.hc.hc_mixes(
            hidden, self.hc.hc_ffn_fn, self.hc.hc_ffn_scale, self.hc.hc_ffn_base
        )
        x = self.hc.hc_pre(hidden, attn_pre)
        x = self.post_attention_layernorm(x)
        if isinstance(self.mlp, (DeepseekV41MoE, DeepseekV41W8A8MoE)):
            x = self.mlp(x, input_ids)
        else:
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

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
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
            # The five attention projections are constructed unconditionally by
            # DeepseekV41Attention, so a missing checkpoint tensor is a broken
            # export (not an optional module): fail instead of running with
            # uninitialized weights.
            for ckpt_name, param_name, shard_dim in (
                ("attn.wq_a", "self_attn.q_a_proj", None),
                ("attn.wq_b", "self_attn.q_b_proj", 0),
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
                {"weight": 0, "weight_scale": 0, "weight_offset": 0},
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
            if sink.dim() == 1 and sink.size(0) == cfg.n_heads and cfg.tp_size > 1:
                # Preserve already-local sinks; shard full per-head checkpoints.
                sink = loader.shard(sink, dim=0)
            loader.copy_in(pm + "self_attn.attn_sink", sink)
            attn.attn_sink_loaded = True
        self._load_common_layer_norms(loader, ck, pm)
        # Main-KV compressor (Full layers; no gate at ratio 1). Module
        # construction decides which tensors are required, exactly like the
        # five projections above: a checkpoint-presence test here would
        # silently keep the nn.Linear default init for these weights.
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


# ---------------------------------------------------------------------------
# Pure-torch quantization helpers for the official checkpoint
#
# Reference recipes from the official DeepSeek-V4.1 Flash inference/kernel.py:
# ue8m0 scale decoding, 32x32-block FP8 E4M3 weight dequant, MXFP4
# (E2M1 + per-32 ue8m0 group scale) expert dequant, and the dynamic FP8/FP4
# activation QDQ variants of the reference inference/kernel.py. Leaf section:
# depends only on torch and the stdlib (imported once at the module head),
# device-agnostic (CPU and NPU).
# ---------------------------------------------------------------------------

_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0
_FP8_AMAX_FLOOR = 1e-4
_FP4_E8M0_AMAX_FLOOR = 6.0 * (2.0**-126)
_FP4_E4M3_AMAX_FLOOR = 6.0 * (2.0**-9)

# IEEE 754 bits of the float32 subnormal 2**-127 (ue8m0 byte 0x00).

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
_E2M1_POSITIVE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Round-half-to-even index on the positive grid for each exact midpoint
# (0.75 -> 1.0, 1.75 -> 2.0, 3.5 -> 4.0, ... : ties go to the even mantissa).
_E2M1_MIDPOINT_EVEN_INDEX = (0, 2, 2, 4, 4, 6, 6)


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


def fp8_act_qdq(x: torch.Tensor, group: int = 32) -> torch.Tensor:
    orig_dtype = x.dtype
    grouped = _grouped(x.float().contiguous(), group)
    amax = grouped.abs().amax(dim=-1).clamp_min(_FP8_AMAX_FLOOR)
    scale = _pow2(_log2_ceil(amax * (1.0 / _E4M3_MAX)))
    normalized = torch.clamp(grouped / scale.unsqueeze(-1), -_E4M3_MAX, _E4M3_MAX)
    y = _round_e4m3(normalized) * scale.unsqueeze(-1)
    return y.flatten(-2).to(orig_dtype)


def fp4_act_qdq_e8m0(x: torch.Tensor, group: int = 32) -> torch.Tensor:
    orig_dtype = x.dtype
    grouped = _grouped(x.float().contiguous(), group)
    amax = grouped.abs().amax(dim=-1).clamp_min(_FP4_E8M0_AMAX_FLOOR)
    scale = _pow2(_log2_ceil(amax * (1.0 / _FP4_E2M1_MAX)))
    normalized = torch.clamp(grouped / scale.unsqueeze(-1), -_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    y = _round_e2m1(normalized) * scale.unsqueeze(-1)
    return y.flatten(-2).to(orig_dtype)


def fp4_act_qdq_e4m3_scale(x: torch.Tensor, group: int = 16) -> torch.Tensor:
    orig_dtype = x.dtype
    grouped = _grouped(x.float().contiguous(), group)
    amax = grouped.abs().amax(dim=-1).clamp_min(_FP4_E4M3_AMAX_FLOOR)
    scale = _round_e4m3(amax / _FP4_E2M1_MAX)
    normalized = torch.clamp(grouped / scale.unsqueeze(-1), -_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    y = _round_e2m1(normalized) * scale.unsqueeze(-1)
    return y.flatten(-2).to(orig_dtype)


def _log2_ceil(x: torch.Tensor) -> torch.Tensor:
    bits = x.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    has_mantissa = (bits & 0x7FFFFF) != 0
    return exponent - 127 + has_mantissa.to(torch.int32)


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    bits = (exponent + 127) << 23
    return bits.contiguous().view(torch.float32)


def _grouped(x: torch.Tensor, group: int) -> torch.Tensor:
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")
    if x.size(-1) % group != 0:
        raise ValueError(f"last dim {x.size(-1)} is not divisible by group {group}")
    return x.unflatten(-1, (-1, group))


_E2M1_TABLE_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _e2m1_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache per device: building rounding tables during ACL capture would issue synchronous H2D."""
    cached = _E2M1_TABLE_CACHE.get(device)
    if cached is None:
        cached = (
            torch.tensor(_E2M1_POSITIVE_VALUES, dtype=torch.float32, device=device),
            torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=device),
            torch.tensor(_E2M1_MIDPOINT_EVEN_INDEX, dtype=torch.int64, device=device),
        )
        _E2M1_TABLE_CACHE[device] = cached
    return cached


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    positive, midpoints, even_index = _e2m1_tables(x.device)
    magnitude = x.abs().clamp(max=_FP4_E2M1_MAX)
    index = torch.searchsorted(midpoints, magnitude, right=False)
    clipped = index.clamp(max=len(_E2M1_MIDPOINTS) - 1)
    is_tie = magnitude == midpoints[clipped]
    rounded_index = torch.where(is_tie, even_index[clipped], index)
    rounded = positive[rounded_index]
    return torch.where(x < 0, -rounded, rounded)


_E4M3_MAX = 448.0
_E4M3_SUBNORMAL_THRESHOLD = 2.0**-6


def _round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round arithmetically: native NPU float8 casts fail; overflow saturates at +/-448."""
    magnitude = x.abs().clamp(max=_E4M3_MAX)
    # NPU has no frexp kernel; bit inspection avoids a synchronous CPU fallback during capture.
    bits = magnitude.view(torch.int32)
    exp_field = (bits >> 23) & 0xFF
    exponent = torch.where(exp_field > 0, exp_field - 126, 0).to(magnitude.dtype)
    step = torch.exp2(exponent - 4.0)
    q = torch.round(magnitude / step)
    normal_val = q * step
    q_sub = torch.round(magnitude / _E4M3_SUBNORMAL_STEP)
    sub_val = q_sub * _E4M3_SUBNORMAL_STEP
    val = torch.where(magnitude < _E4M3_SUBNORMAL_THRESHOLD, sub_val, normal_val)
    # NPU lacks copysign; signed zero differences disappear on the E4M3 grid.
    return torch.where(x < 0, -val, val)
