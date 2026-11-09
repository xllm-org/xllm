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
import re
import unicodedata
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python.attention.dsa_metadata import (
    READ_TABLE_LAYOUT_RING,
    DsaMetadataBuilder,
    build_cache_specs_v41,
    last_ring_writes,
)
from xllm.python.model_executor.forward_context import get_forward_context_or_none, in_acl_graph
from xllm.python.models.engram_shared_memory import SharedHostEngramStorage

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata, LayerCache
    from xllm.python.model_executor.input_batch import InputBatch

ENGRAM_DEAD_ID = -1
ENGRAM_IMAGE_SENTINEL_BASE_ID = 129264
ENGRAM_IMAGE_PAD_ID = 129265
_ENGRAM_GATE_CLAMP = 1e-6

_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")
# A private-use character keeps a token containing one space from being stripped.
_SENTINEL = "\ue000"


_UE8M0_ZERO_BITS = 0x00400000
_E4M3_SUBNORMAL_STEP = 2.0**-9


def _validate_ue8m0_scale_bytes(raw: torch.Tensor) -> None:
    if raw.dtype != torch.uint8:
        raise ValueError(f"expected raw uint8 ue8m0 scale storage, got {raw.dtype}")
    # A reduction avoids a table-sized temporary for mmap-backed weights.
    if raw.numel() > 0 and bool(raw.amax() == 0xFF):
        raise ValueError("ue8m0 scale byte 0xFF (NaN) is invalid")


def ue8m0_to_scale(scale_u8: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
    """ue8m0: byte b means 2**(b-127); 0x00 is subnormal and 0xFF is invalid."""
    if scale_u8.dtype == torch.uint8:
        raw = scale_u8
    elif scale_u8.element_size() == 1:
        raw = scale_u8.view(torch.uint8)
    else:
        raise ValueError(f"expected 1-byte ue8m0 scale storage, got {scale_u8.dtype}")
    if validate:
        _validate_ue8m0_scale_bytes(raw)
    raw_i32 = raw.to(torch.int32)
    bits = torch.where(
        raw_i32 == 0,
        torch.full_like(raw_i32, _UE8M0_ZERO_BITS),
        raw_i32 << 23,
    )
    return bits.contiguous().view(torch.float32)


def _e4m3_to_float(x_e4m3: torch.Tensor) -> torch.Tensor:
    bits = x_e4m3.view(torch.uint8).to(torch.int32)
    sign = torch.where(bits >= 128, -1.0, 1.0)
    magnitude_bits = bits & 0x7F
    exponent = torch.div(magnitude_bits, 8, rounding_mode="floor")
    mantissa = magnitude_bits & 0x7
    mantissa_f = mantissa.to(torch.float32)
    normal = (1.0 + mantissa_f / 8.0) * torch.exp2(exponent.to(torch.float32) - 7.0)
    subnormal = mantissa_f * _E4M3_SUBNORMAL_STEP
    value = torch.where(exponent > 0, normal, subnormal)
    value = torch.where(magnitude_bits == 0x7F, torch.full_like(value, float("nan")), value)
    return value * sign


def image_sentinel_mask(input_ids: torch.Tensor) -> torch.Tensor:
    ids = input_ids.reshape(-1)
    return (ids == ENGRAM_IMAGE_SENTINEL_BASE_ID) | (ids == ENGRAM_IMAGE_PAD_ID)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


class EngramLayout:
    def __init__(
        self,
        layer_ids: Sequence[int],
        num_embeddings: Sequence[int],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        vocab_size: int,
        compressed_vocab_size: int,
        pad_token_id: int,
    ) -> None:
        self.layer_ids: tuple[int, ...] = tuple(int(i) for i in layer_ids)
        self.num_embeddings: tuple[int, ...] = tuple(int(n) for n in num_embeddings)
        self.max_ngram_size = int(max_ngram_size)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.vocab_size = int(vocab_size)
        self.compressed_vocab_size = int(compressed_vocab_size)
        self.pad_token_id = int(pad_token_id)
        if len(self.layer_ids) != len(self.num_embeddings):
            raise ValueError(
                f"engram_layer_ids {self.layer_ids} and engram_num_embeddings "
                f"{self.num_embeddings} must have the same length"
            )
        if self.max_ngram_size < 2:
            raise ValueError(f"engram_max_ngram_size must be >= 2, got {max_ngram_size}")

        primes: list[list[tuple[int, ...]]] = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes = []
                current = self.vocab_size - 1
                for _ in range(self.n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(per_ngram)
        self.primes: tuple[tuple[tuple[int, ...], ...], ...] = tuple(tuple(p) for p in primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        offsets = []
        for layer, layer_primes in enumerate(self.primes):
            flat = [size for per_ngram in layer_primes for size in per_ngram]
            total = sum(flat)
            if total > self.num_embeddings[layer]:
                raise ValueError(
                    f"engram layer {self.layer_ids[layer]} prime buckets sum to {total} rows "
                    f"but engram_num_embeddings only provides {self.num_embeddings[layer]}"
                )
            prefix = [0]
            for size in flat[:-1]:
                prefix.append(prefix[-1] + size)
            offsets.append(prefix)
        self.offsets = torch.tensor(offsets, dtype=torch.int64)

    def head_sizes(self, layer_hash_index: int) -> tuple[int, ...]:
        return tuple(size for per_ngram in self.primes[layer_hash_index] for size in per_ngram)


def compute_hash_multipliers(
    layer_ids: Sequence[int],
    max_ngram_size: int,
    compressed_vocab_size: int,
) -> torch.Tensor:
    """Checkpoint hashing fixes the PCG64 stream: seed 10007 * layer_id, then doubled-plus-one draws."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Building DeepSeek-V4.1 engram hash multipliers requires numpy (the "
            "reference RNG is numpy PCG64 and must be reproduced bit-exactly). "
            "Install numpy, or pre-generate an 'engram_hash_state.bin' sidecar "
            "with build_engram_hash_state() and place it next to "
            "the checkpoint."
        ) from exc
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1, dtype=torch.int64))
    return torch.stack(rows)


def normalize_token_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("M"))
    s = s.lower()
    s = _WHITESPACE_RE.sub(" ", s)
    if s == " ":
        s = _SENTINEL
    s = s.strip()
    return s.replace(_SENTINEL, " ")


def build_compressed_token_map_from_texts(
    token_texts: Sequence[str],
    id_to_token: Callable[[int], str] | None = None,
) -> tuple[list[int], int]:
    key_to_new: dict[str, int] = {}
    lookup: list[int] = []
    for token_id, text in enumerate(token_texts):
        if "\ufffd" in text:
            key = id_to_token(token_id) if id_to_token is not None else text
        else:
            normalized = normalize_token_text(text)
            key = normalized or text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup.append(new_id)
    return lookup, len(key_to_new)


def _load_engram_hash_state_sidecar(model_path: str) -> dict[str, torch.Tensor] | None:
    sidecar = os.path.join(model_path, "engram_hash_state.bin")
    if not os.path.isfile(sidecar):
        return None
    try:
        return torch.load(sidecar, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"failed to load engram hash-state sidecar {sidecar}: {exc}") from exc


def build_engram_hash_state(
    model_path: str,
    tokenizer_path: str | None = None,
    output_path: str | None = None,
) -> str:
    model_dir = os.path.abspath(model_path)
    if not os.path.isdir(model_dir):
        model_dir = os.path.dirname(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        top = json.load(handle)
    config = {**top, **top.get("text_config", {})}

    layer_ids = config.get("engram_layer_ids") or []
    if not layer_ids:
        raise ValueError("config.json has no engram_layer_ids; this model does not use engram")
    vocab_size = int(config["vocab_size"])
    max_ngram_size = int(config["engram_max_ngram_size"])
    compressed_vocab = int(config["engram_compressed_vocab_size"])
    pad_token_id = int(config["engram_pad_token_id"])
    if not 0 <= pad_token_id < vocab_size:
        raise ValueError(f"engram_pad_token_id {pad_token_id} is out of vocab range {vocab_size}")

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "building the engram token map requires the 'tokenizers' package (pip install tokenizers)"
        ) from exc
    tokenizer_file = tokenizer_path or os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_file}")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_vocab != vocab_size:
        raise ValueError(f"tokenizer vocab size {tokenizer_vocab} does not match config vocab_size {vocab_size}")

    token_texts = tokenizer.decode_batch([[token_id] for token_id in range(vocab_size)], skip_special_tokens=False)
    lookup, compressed = build_compressed_token_map_from_texts(token_texts, id_to_token=tokenizer.id_to_token)
    if compressed != compressed_vocab:
        raise ValueError(
            f"the tokenizer folds to {compressed} compressed ids but the config declares "
            f"engram_compressed_vocab_size={compressed_vocab}; every hash multiplier derives "
            "from this count, so the engram tables would be silently rehashed -- refusing to "
            "write the sidecar"
        )

    multipliers = compute_hash_multipliers(layer_ids, max_ngram_size, compressed_vocab)
    output = output_path or os.path.join(model_dir, "engram_hash_state.bin")
    torch.save(
        {
            "token_map": torch.tensor(lookup, dtype=torch.int32),
            "multipliers": multipliers,
        },
        output,
    )
    return output


class EngramHashState(nn.Module):
    def __init__(
        self,
        layout: EngramLayout,
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.lookback_depth = layout.max_ngram_size - 1
        self.pad_id = int(token_map[layout.pad_token_id].item())
        self.register_buffer("token_map", token_map.to(torch.int64), persistent=False)
        self.register_buffer(
            "primes",
            torch.tensor(layout.primes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("offsets", layout.offsets.to(torch.int64), persistent=False)
        self.register_buffer("multipliers", multipliers.to(torch.int64), persistent=False)
        self._cache: torch.Tensor | None = None
        self._kv_cache_ref: weakref.ReferenceType[torch.Tensor] | None = None

    @classmethod
    def from_config(cls, layout: EngramLayout, config: dict) -> EngramHashState:
        injected = config.get("engram_token_map")
        if injected is not None:
            token_map = torch.as_tensor(injected, dtype=torch.int64)
            multipliers = compute_hash_multipliers(
                layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size
            )
        else:
            sidecar = _load_engram_hash_state_sidecar(str(config.get("model_path") or ""))
            if sidecar is None:
                raise RuntimeError(
                    "DeepSeek-V4.1 engram requires the tokenizer-derived hash state. "
                    "Call build_engram_hash_state(model_path) against the "
                    "checkpoint's tokenizer to write 'engram_hash_state.bin' (or pass "
                    "'engram_token_map' in the config for tests)."
                )
            token_map = sidecar["token_map"]
            multipliers = sidecar["multipliers"]
            expected_shape = (len(layout.layer_ids), layout.max_ngram_size)
            if multipliers.shape != expected_shape:
                raise ValueError(
                    f"engram hash multipliers have shape {tuple(multipliers.shape)} but config expects {expected_shape}"
                )
        # Validate against the tokenizer vocabulary, not the separate Engram bucket count.
        expected_vocab = int(config.get("vocab_size") or token_map.numel())
        if token_map.numel() != expected_vocab:
            raise ValueError(
                f"engram token map covers {token_map.numel()} ids but model vocab_size is {expected_vocab}"
            )
        compressed = int(token_map.max().item()) + 1
        if compressed != layout.compressed_vocab_size:
            raise ValueError(
                f"engram token map compresses to {compressed} ids but config expects "
                f"{layout.compressed_vocab_size}; every hash multiplier derives from "
                "it, so the engram tables would be silently rehashed"
            )
        return cls(layout, token_map, multipliers)

    def ensure_cache(self, swa_cache: torch.Tensor | None) -> bool:
        if swa_cache is None or swa_cache.numel() == 0:
            self._cache = None
            self._kv_cache_ref = None
            return False
        if self._kv_cache_ref is not None and self._kv_cache_ref() is swa_cache:
            return True
        flat_rows = swa_cache.numel() // swa_cache.size(-1) if swa_cache.dim() > 1 else swa_cache.numel()
        self._cache = torch.zeros(flat_rows, dtype=torch.int32, device=swa_cache.device)
        self._kv_cache_ref = weakref.ref(swa_cache)
        return True

    def restore_history(
        self,
        history: torch.Tensor,
    ) -> None:
        if self._cache is None:
            raise RuntimeError("Engram history restore requires a bound cache")
        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError("Engram history must contain [slot, token] pairs")
        history = history.to(device=self._cache.device, dtype=torch.int64)
        slots, tokens = history.unbind(dim=1)
        comp = self.token_map[tokens.clamp(0, self.token_map.numel() - 1)]
        comp = torch.where(
            image_sentinel_mask(tokens).reshape(-1),
            ENGRAM_DEAD_ID,
            comp,
        ).to(torch.int64)
        self._write_hash_cache(comp, slots)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        swa_block_size: int,
    ) -> torch.Tensor:
        """Resolve history in order: current chunk, explicit lookback, then slot cache."""
        num_tokens = input_ids.shape[0]
        num_layers = len(self.layout.layer_ids)
        n_heads = self.layout.n_heads
        max_ngram = self.layout.max_ngram_size
        device = input_ids.device
        output = torch.zeros(num_tokens, num_layers, self.layout.n_hash_cols, dtype=torch.int32, device=device)
        if num_tokens == 0:
            return output
        pos = positions.reshape(-1).to(torch.int64)
        ids = input_ids.reshape(-1).to(torch.int64)
        req = self._sequence_index(pos, query_start_loc)
        chunk_start = pos[query_start_loc.to(torch.int64)[req].clamp(0, num_tokens - 1)]

        comp = self.token_map[ids.clamp(0, self.token_map.numel() - 1)]
        comp = torch.where(dead_mask.reshape(-1), ENGRAM_DEAD_ID, comp).to(torch.int64)

        rows = torch.arange(num_tokens, device=device)
        rolling = torch.zeros(num_tokens, num_layers, dtype=torch.int64, device=device)
        blocked = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        cache = self._cache
        for shift in range(max_ngram):
            lookback = pos - shift
            in_batch = lookback >= chunk_start
            source = self._chunk_source(comp, rows, shift, in_batch, num_tokens)
            known = None
            if lookback_token_ids is not None:
                source, known = self._lookback_source(source, req, lookback, in_batch, chunk_start, lookback_token_ids)
            if cache is not None and block_table is not None:
                source = self._cache_source(source, req, lookback, in_batch, known, block_table, swa_block_size)
            blocked = blocked | (lookback < 0) | (source == ENGRAM_DEAD_ID)
            value = torch.where(blocked, self.pad_id, source)
            rolling ^= value.unsqueeze(1) * self.multipliers[:, shift].unsqueeze(0)
            if shift > 0:
                lo = (shift - 1) * n_heads
                prime = self.primes[:, shift - 1, :]
                offset = self.offsets[:, lo : lo + n_heads]
                hashed = rolling.unsqueeze(-1) % prime.unsqueeze(0) + offset.unsqueeze(0)
                output[:, :, lo : lo + n_heads] = hashed.to(torch.int32)
        # A wrapping chunk must consume its old history before overwriting the ring.
        if slot_mapping is not None and self._cache is not None:
            self._write_hash_cache(comp, slot_mapping)
        return output

    def _write_hash_cache(
        self,
        comp: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self._cache is None:
            raise RuntimeError(
                "engram hash slot cache is unbound; call ensure_cache with the bound SWA "
                "KV cache before writing hash rows"
            )
        comp = comp.reshape(-1).to(torch.int32)
        slots = slot_mapping.reshape(-1).to(torch.int64)
        if not in_acl_graph():
            slots, comp = last_ring_writes(slots, comp, slot_capacity=self._cache.numel())
        valid = slots >= 0
        safe = torch.where(valid, slots, torch.zeros_like(slots))
        keep = self._cache.index_select(0, safe)
        self._cache.scatter_(0, safe, torch.where(valid, comp, keep))

    def _sequence_index(
        self,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = positions.shape[0]
        qsl = query_start_loc.to(torch.int64).reshape(-1)
        ends = qsl[1:].contiguous()
        req = torch.searchsorted(ends, torch.arange(num_tokens, device=positions.device), right=True)
        return req.clamp(0, max(qsl.numel() - 2, 0))

    def _chunk_source(
        self,
        comp: torch.Tensor,
        rows: torch.Tensor,
        shift: int,
        in_batch: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        source_row = (rows - shift).clamp_min(0).clamp_max(num_tokens - 1)
        source = comp.index_select(0, source_row)
        return torch.where(in_batch, source, torch.zeros_like(source))

    def _lookback_source(
        self,
        source: torch.Tensor,
        req: torch.Tensor,
        lookback: torch.Tensor,
        in_batch: torch.Tensor,
        chunk_start: torch.Tensor,
        lookback_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depth = lookback_token_ids.shape[-1]
        col = chunk_start - 1 - lookback
        in_window = (~in_batch) & (col >= 0) & (col < depth)
        col_safe = col.clamp(0, depth - 1)
        req_safe = req.clamp(0, lookback_token_ids.shape[0] - 1)
        token = lookback_token_ids.to(torch.int64)[req_safe, col_safe]
        known = in_window & (token >= 0)
        token_safe = token.clamp_min(0)
        mapped = self.token_map[token_safe.clamp(0, self.token_map.numel() - 1)]
        dead = (token_safe == ENGRAM_IMAGE_SENTINEL_BASE_ID) | (token_safe == ENGRAM_IMAGE_PAD_ID)
        mapped = torch.where(dead, ENGRAM_DEAD_ID, mapped)
        return torch.where(known, mapped, source), known

    def _cache_source(
        self,
        source: torch.Tensor,
        req: torch.Tensor,
        lookback: torch.Tensor,
        in_batch: torch.Tensor,
        known: torch.Tensor | None,
        block_table: torch.Tensor,
        swa_block_size: int,
    ) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError(
                "engram hash slot cache is unbound; call ensure_cache with the bound SWA "
                "KV cache before reading hash history"
            )
        table_cols = block_table.shape[1]
        clamped = lookback.clamp_min(0)
        col = torch.div(clamped, swa_block_size, rounding_mode="floor") % table_cols
        req_safe = req.clamp(0, block_table.shape[0] - 1)
        block = block_table.to(torch.int64)[req_safe, col]
        # Do not upper-clamp absolute lookbacks: it aliases later positions to the final table column.
        missing = block < 0
        safe_block = torch.where(missing, torch.zeros_like(block), block)
        slot = (safe_block * swa_block_size + clamped % swa_block_size).clamp(0, self._cache.numel() - 1)
        fallback = self._cache.index_select(0, slot).to(torch.int64)
        fallback = torch.where(
            missing,
            torch.full_like(fallback, ENGRAM_DEAD_ID),
            fallback,
        )
        resolved = in_batch if known is None else (in_batch | known)
        return torch.where(resolved, source, fallback)


ENGRAM_STORAGE_FP8 = "fp8"
ENGRAM_STORAGE_INT8 = "int8"


# Keep staging capacity monotonic so eager reuse does not churn device addresses.
_ENGRAM_HOST_MIN_CAPACITY = 256


_host_pin_warned = False


def _host_buffer(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, bool, str | None]:
    """Return the pin failure with the buffer; some hosts have no registered pinning accelerator."""
    global _host_pin_warned
    try:
        buffer = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    except (RuntimeError, ValueError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if not _host_pin_warned:
            _host_pin_warned = True
            try:
                from scripts.logger import logger

                logger.warning(
                    "engram host offload cannot allocate pinned CPU staging "
                    f"({failure}); the lookup op contract refuses "
                    "unpinned staging, so set engram_config.cpu_offload=false on "
                    "a host that cannot pin host memory"
                )
            except Exception:  # noqa: BLE001 - logging must never mask the fallback
                pass
        return torch.empty(shape, dtype=dtype, device="cpu"), False, failure
    _host_pin_warned = False
    return buffer, True, None


def _staging_stream(device: torch.device | None) -> torch.npu.Stream | torch.cuda.Stream | None:
    if device is None or device.type == "cpu":
        return None
    if device.type in ("npu", "privateuseone"):
        return torch.npu.current_stream(device)
    if device.type == "cuda":
        return torch.cuda.current_stream(device)
    raise RuntimeError(f"unsupported Engram offload device: {device}")


def _sync_accelerator(device: torch.device | None) -> None:
    stream = _staging_stream(device)
    if stream is not None:
        stream.synchronize()


_engram_lookup_op: Any = None


def _get_engram_lookup_op() -> Any:
    global _engram_lookup_op
    if _engram_lookup_op is not None:
        return _engram_lookup_op
    try:
        op = torch.ops.xllm_ops.engram_int8_lookup_cpu
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "engram host offload requires the in-repo CPU int8 lookup op "
            "torch.ops.xllm_ops.engram_int8_lookup_cpu, which is not registered; "
            "rebuild/reinstall xLLM with core/kernels/npu/xllm_ops/"
            "engram_int8_lookup_cpu.cpp compiled in (the fused op is the only "
            "offload dequant path)"
        ) from exc
    _engram_lookup_op = op
    return op


_engram_lookup_op_active_logged = False


def _log_engram_lookup_op_active() -> None:
    global _engram_lookup_op_active_logged
    if _engram_lookup_op_active_logged:
        return
    _engram_lookup_op_active_logged = True
    try:
        from scripts.logger import logger

        logger.info("engram CPU int8 lookup op active (fused offload dequant)")
    except Exception:  # noqa: BLE001 - logging must never block the lookup
        pass


def load_quarot_gate_unrotate(model_path: str, device: torch.device) -> torch.Tensor | None:
    """QuaRot rotates the residual stream; the gate needs h_rot @ Q.T in the original basis."""
    config_path = os.path.join(str(model_path), "config.json")
    try:
        with open(config_path, encoding="utf-8") as handle:
            rotation = json.load(handle).get("engram_rotation_config") or {}
    except OSError:
        return None
    if not rotation.get("value_projection_rotated"):
        return None
    quarot_path = os.path.join(str(model_path), "optional", "quarot.safetensors")
    from safetensors.torch import load_file

    q = load_file(quarot_path, device="cpu")["global_rotation"]
    return q.t().contiguous().to(device=device, dtype=torch.float32)


class EngramEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: Sequence[int],
        tp_size: int,
        tp_rank: int,
        quant_block: int = 32,
        storage: str = ENGRAM_STORAGE_FP8,
        device: torch.device | None = None,
        offload: bool = False,
        layer_id: int = -1,
        dp_shared_memory: bool = False,
        dp_group: Any = None,
    ) -> None:
        super().__init__()
        if not head_sizes or any(size <= 0 for size in head_sizes):
            raise ValueError(f"engram head sizes must be positive, got {head_sizes}")
        if sum(head_sizes) > num_embeddings:
            raise ValueError(f"engram head sizes sum to {sum(head_sizes)} rows but the table only has {num_embeddings}")
        if dim % quant_block != 0:
            raise ValueError(f"engram head dim {dim} is not divisible by {quant_block}")
        self.num_embeddings = int(num_embeddings)
        self.dim = int(dim)
        self.quant_block = int(quant_block)
        self.storage = storage
        self.n_hash_cols = len(head_sizes)
        self.part_n_hash_cols = (self.n_hash_cols + tp_size - 1) // tp_size
        self.head_start = tp_rank * self.part_n_hash_cols
        head_end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
        self.vocab_start = sum(head_sizes[: self.head_start])
        self.vocab_end = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end - self.vocab_start
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.layer_id = int(layer_id)
        self.setup_device = torch.device(device) if device is not None else torch.device("cpu")
        self.offload = bool(offload) and storage == ENGRAM_STORAGE_INT8
        if offload and not self.offload:
            self._warn_offload_unsupported(storage)
        if dp_shared_memory and not offload:
            raise ValueError("engram_config.dp_shared_memory=true requires engram_config.cpu_offload=true")
        if dp_shared_memory and storage != ENGRAM_STORAGE_INT8:
            raise ValueError(
                "engram_config.dp_shared_memory=true requires the int8 (W8A8) table "
                f"storage (the only CPU-shareable format), got storage={storage!r}"
            )
        self.dp_shared_memory = bool(dp_shared_memory) and self.offload
        if self.dp_shared_memory and dp_group is None:
            raise ValueError("engram_config.dp_shared_memory=true requires an initialized node-local DP group")
        self._dp_group = dp_group
        self._shared_storage: SharedHostEngramStorage | None = None
        if self.offload and self.quant_block != 32:
            raise ValueError(
                "engram host offload requires quant_block=32 (the fused CPU int8 "
                f"lookup op's fixed group size), got {self.quant_block}; set "
                "engram_config.cpu_offload=false to use the HBM table path"
            )
        table_device: torch.device | str | None = device
        if self.offload:
            table_device = "cpu"
            self._ids_host: torch.Tensor | None = None
            self._owned_host: torch.Tensor | None = None
            self._rows_host: torch.Tensor | None = None
            self._host_capacity = 0
            self._ids_host_pinned = False
            self._rows_host_pinned = False
            self._host_pin_failure: str | None = None
            self._graph_metadata_bound = False
            self._eager_staged_device: torch.Tensor | None = None
            self._eager_capacity = 0
        if storage == ENGRAM_STORAGE_INT8:
            if self.dp_shared_memory and self.part_num_embeddings > 0:
                self._shared_storage = SharedHostEngramStorage(
                    layer_id=self.layer_id,
                    rows=self.part_num_embeddings,
                    dim=self.dim,
                    quant_block=self.quant_block,
                    group=self._dp_group,
                )
                self._shared_storage.create_or_attach()
                self.weight = nn.Parameter(self._shared_storage.codes, requires_grad=False)
                self.weight_scale = nn.Parameter(self._shared_storage.scales, requires_grad=False)
            else:
                self.weight = nn.Parameter(
                    torch.empty(self.part_num_embeddings, self.dim, dtype=torch.int8, device=table_device),
                    requires_grad=False,
                )
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        self.part_num_embeddings,
                        self.dim // self.quant_block,
                        dtype=torch.float32,
                        device=table_device,
                    ),
                    requires_grad=False,
                )
        else:
            # Keep FP8 payload and ue8m0 scales as raw bytes to avoid unsupported native float8 gathers.
            self.weight = nn.Parameter(
                torch.empty(self.part_num_embeddings, self.dim, dtype=torch.uint8, device=device),
                requires_grad=False,
            )
            self.weight_scale = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    self.dim // self.quant_block,
                    dtype=torch.uint8,
                    device=device,
                ),
                requires_grad=False,
            )

    @staticmethod
    def _warn_offload_unsupported(storage: str) -> None:
        try:
            from scripts.logger import logger

            logger.warning(
                "engram_config.cpu_offload=true ignored for engram storage %r; only the "
                "int8 (W8A8) tables have a host decoder, falling back to the HBM path",
                storage,
            )
        except Exception:
            pass

    def _owned_local_cols(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_cols = indices[:, self.head_start : self.head_start + self.part_n_hash_cols]
        if local_cols.size(1) < self.part_n_hash_cols:
            local_cols = F.pad(local_cols, (0, self.part_n_hash_cols - local_cols.size(1)))
        owned = (local_cols >= self.vocab_start) & (local_cols < self.vocab_end)
        local = (local_cols - self.vocab_start).clamp(0, max(self.part_num_embeddings - 1, 0))
        return owned, local

    def _local_block_device(self, indices: torch.Tensor) -> torch.Tensor:
        num_tokens = indices.shape[0]
        if self.part_num_embeddings == 0:
            return torch.zeros(
                num_tokens,
                self.part_n_hash_cols,
                self.dim,
                dtype=torch.bfloat16,
                device=indices.device,
            )
        owned, local = self._owned_local_cols(indices)
        flat = local.reshape(-1).to(torch.int64)
        payload = self.weight.index_select(0, flat)
        scale = self.weight_scale.index_select(0, flat)
        if self.storage == ENGRAM_STORAGE_INT8:
            values = payload.to(torch.float32)
            scale = scale.repeat_interleave(self.quant_block, dim=-1)
        else:
            values = _e4m3_to_float(payload)
            # Persistent scales were checked at load; lookup must not introduce a D2H validation sync.
            scale = ue8m0_to_scale(scale, validate=False).repeat_interleave(self.quant_block, dim=-1)
        rows = (values * scale).to(torch.bfloat16)
        rows = rows.view(num_tokens, self.part_n_hash_cols, self.dim)
        return torch.where(owned.unsqueeze(-1), rows, torch.zeros_like(rows))

    def _ensure_host_buffers(self, num_tokens: int) -> None:
        self._ensure_host_scratch(num_tokens)
        self._ensure_eager_staging(num_tokens)

    def _wait_for_upload(self) -> None:
        event = getattr(self, "_upload_done", None)
        if event is not None:
            event.synchronize()
            self._upload_done = None

    def _record_upload(self) -> None:
        stream = _staging_stream(self.setup_device)
        if stream is None:
            return
        event = getattr(self, "_upload_event", None)
        if event is None:
            event = torch.cuda.Event() if self.setup_device.type == "cuda" else torch.npu.Event()
            self._upload_event = event
        event.record(stream)
        self._upload_done = event

    def _ensure_host_scratch(self, num_tokens: int) -> None:
        width = self.part_n_hash_cols
        if self._ids_host is not None and self._host_capacity >= num_tokens:
            return
        # Wait for the prior H2D upload before replacing pinned source buffers.
        if self._ids_host is not None:
            self._wait_for_upload()
        capacity = max(self._host_capacity, _ENGRAM_HOST_MIN_CAPACITY)
        while capacity < num_tokens:
            capacity *= 2
        ids_host, ids_pinned, ids_failure = _host_buffer((capacity, width), torch.int64)
        owned_host, _, owned_failure = _host_buffer((capacity, width), torch.bool)
        rows_host, rows_pinned, rows_failure = _host_buffer((capacity, width, self.dim), torch.bfloat16)
        # Publish scratch only after every allocation succeeds, allowing a failed grow to retry.
        self._ids_host, self._ids_host_pinned = ids_host, ids_pinned
        self._owned_host = owned_host
        self._rows_host, self._rows_host_pinned = rows_host, rows_pinned
        self._host_capacity = capacity
        self._host_pin_failure = ids_failure or owned_failure or rows_failure

    def _ensure_eager_staging(self, num_tokens: int) -> None:
        if self._eager_staged_device is not None and self._eager_capacity >= num_tokens:
            return
        capacity = max(self._eager_capacity, _ENGRAM_HOST_MIN_CAPACITY)
        while capacity < num_tokens:
            capacity *= 2
        staged = self._alloc_staging(capacity)
        self._eager_staged_device = staged
        self._eager_capacity = capacity

    def _alloc_staging(self, capacity: int) -> torch.Tensor:
        return torch.empty(
            (capacity, self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=self.setup_device,
        )

    def _engram_lookup_op_contract_error(
        self,
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> str:
        if not self.offload:
            return "table is not host-offloaded"
        if self._shared_storage is not None and not self._shared_storage.is_loaded:
            return f"shared host table is {self._shared_storage.state}, not loaded"
        if not self._staging_operand_pinned(ids, self._ids_host, self._ids_host_pinned):
            return f"ids staging is not pinned{self._pin_failure_hint(self._ids_host_pinned)}"
        if not self._staging_operand_pinned(output, self._rows_host, self._rows_host_pinned):
            return f"output staging is not pinned{self._pin_failure_hint(self._rows_host_pinned)}"
        return ""

    @staticmethod
    def _staging_operand_pinned(
        operand: torch.Tensor,
        staging: torch.Tensor | None,
        pinned: bool,
    ) -> bool:
        """Compare storage pointers so offset views of the recorded pinned buffer remain valid."""
        return (
            pinned
            and staging is not None
            and operand.untyped_storage().data_ptr() == staging.untyped_storage().data_ptr()
        )

    def _pin_failure_hint(self, pinned: bool) -> str:
        if pinned or self._host_pin_failure is None:
            return ""
        return (
            f" (this host cannot allocate pinned CPU memory: {self._host_pin_failure}; "
            "the offloaded table needs pinned staging, so set "
            "engram_config.cpu_offload=false)"
        )

    def stage_ids_offload(self, indices: torch.Tensor, *, output: torch.Tensor | None = None) -> None:
        """Queue D2H on the current stream; the caller waits once across all layers."""
        num_tokens = indices.shape[0]
        self._wait_for_upload()
        if output is None:
            self._ensure_host_buffers(num_tokens)
        else:
            expected = (num_tokens, self.part_n_hash_cols, self.dim)
            if tuple(output.shape) != expected or output.dtype != torch.bfloat16 or output.device != self.setup_device:
                raise ValueError("Engram staging output must match this shard's shape, dtype and device")
            self._ensure_host_scratch(num_tokens)

        assert self._ids_host is not None
        assert self._owned_host is not None

        owned, local = self._owned_local_cols(indices)

        self._ids_host[:num_tokens].copy_(local, non_blocking=True)
        self._owned_host[:num_tokens].copy_(owned, non_blocking=True)

    def finish_stage_offload(self, num_tokens: int, *, output: torch.Tensor | None = None) -> torch.Tensor:
        """The caller must finish D2H first; the upload event protects pinned sources until reuse."""
        assert self._ids_host is not None
        assert self._owned_host is not None
        assert self._rows_host is not None
        staging = output
        if staging is None:
            staging = self._eager_staged_device
        assert staging is not None
        flat = self._ids_host[:num_tokens].reshape(-1)
        rows_out = self._rows_host[:num_tokens].view(-1, self.dim)
        if self.part_num_embeddings == 0:
            rows_out.zero_()
        else:
            reason = self._engram_lookup_op_contract_error(flat, rows_out)
            if reason:
                raise RuntimeError(
                    f"engram host offload cannot run the CPU int8 lookup op ({reason}); "
                    "the fused op is the only dequant path for the offloaded table"
                )
            _get_engram_lookup_op()(self.weight, self.weight_scale, flat, rows_out)
        rows_out.masked_fill_(~self._owned_host[:num_tokens].reshape(-1, 1), 0.0)
        _log_engram_lookup_op_active()

        staging[:num_tokens].copy_(self._rows_host[:num_tokens], non_blocking=True)
        self._record_upload()
        return staging[:num_tokens]

    def _local_block_offload(self, indices: torch.Tensor) -> torch.Tensor:
        self.stage_ids_offload(indices)
        _sync_accelerator(self.setup_device)
        return self.finish_stage_offload(indices.shape[0])

    def fallback_to_device(self) -> bool:
        """Quiesce execution before demotion; persistent graph rows and DP-shared tables cannot be demoted."""
        if not self.offload:
            return False
        if self._shared_storage is not None:
            raise RuntimeError(
                "engram fallback_to_device() is not supported for a DP-shared host table: "
                f"layer={self.layer_id} rank={self.tp_rank} shares its table with its "
                "node-local DP group, and a per-rank HBM demotion would desynchronize the "
                "group. engram_config.dp_shared_memory requires the shared host load to succeed"
            )
        if self._graph_metadata_bound:
            raise RuntimeError(
                "engram fallback_to_device() must run before ACL-graph capture: "
                f"layer={self.layer_id} rank={self.tp_rank} already allocated a "
                "persistent execution metadata. A "
                "captured graph has baked that address, so demoting the table to HBM "
                "here would make every replay read freed memory. Reconcile the "
                "offload decision before the first capture instead."
            )
        # Complete both copies before replacing either parameter, so failure leaves a usable host table.
        device_weight = self.weight.data.to(self.setup_device)
        device_weight_scale = self.weight_scale.data.to(self.setup_device)
        self.weight.data = device_weight
        self.weight_scale.data = device_weight_scale
        self.offload = False
        self._ids_host = None
        self._owned_host = None
        self._rows_host = None
        self._host_capacity = 0
        self._ids_host_pinned = False
        self._rows_host_pinned = False
        self._eager_staged_device = None
        self._eager_capacity = 0
        return True

    def close_host_offload(self) -> None:
        storage = self._shared_storage
        if storage is None:
            return
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.int8, device="cpu"), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty(0, dtype=torch.float32, device="cpu"), requires_grad=False)
        storage.close()
        self._shared_storage = None

    def local_block(self, indices: torch.Tensor) -> torch.Tensor:
        if self.offload:
            return self._local_block_offload(indices)
        return self._local_block_device(indices)

    def gather_heads(self, local: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            from xllm.python import distributed as _distributed

            rows = _distributed.tp_all_gather(local, dim=1, world_size=self.tp_size)
            return rows[:, : self.n_hash_cols]
        return local

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return self.gather_heads(self.local_block(indices))


class Engram(nn.Module):
    def __init__(
        self,
        layout: EngramLayout,
        layer_hash_index: int,
        hidden_size: int,
        hc_mult: int,
        rms_norm_eps: float,
        tp_size: int,
        tp_rank: int,
        dtype: torch.dtype,
        device: torch.device,
        storage: str = ENGRAM_STORAGE_FP8,
        gate_unrotate: torch.Tensor | None = None,
        offload: bool = False,
        dp_shared_memory: bool = False,
        dp_group: Any = None,
    ) -> None:
        super().__init__()
        self.layer_hash_index = layer_hash_index
        self.dim = hidden_size
        self.hc_mult = hc_mult
        self.eps = rms_norm_eps
        self.embed_tokens = EngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            layout.head_sizes(layer_hash_index),
            tp_size,
            tp_rank,
            storage=storage,
            device=device,
            offload=offload,
            layer_id=layout.layer_ids[layer_hash_index],
            dp_shared_memory=dp_shared_memory,
            dp_group=dp_group,
        )
        n_hash_cols = layout.n_hash_cols
        self.wkv = nn.Linear(
            n_hash_cols * layout.head_dim,
            hidden_size * (hc_mult + 1),
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_weight = nn.Parameter(
            torch.empty(hc_mult, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty(hc_mult, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.register_buffer("gate_unrotate", gate_unrotate, persistent=False)
        self._staged_rows: torch.Tensor | None = None
        # Stage the host gather outside capture; the TP all-gather stays inside the graph.
        self._staged_local: torch.Tensor | None = None
        self._staging_num_tokens = 0

    @property
    def offload(self) -> bool:
        return self.embed_tokens.offload

    def close_host_offload(self) -> None:
        self.embed_tokens.close_host_offload()

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        if self.offload:
            self._staged_local = self.embed_tokens.local_block(hash_ids)
        else:
            self._staged_rows = self.embed_tokens.lookup(hash_ids)

    def begin_stage_offload(self, hash_ids: torch.Tensor, *, output: torch.Tensor | None = None) -> None:
        if not self.offload:
            return
        self.embed_tokens.stage_ids_offload(hash_ids, output=output)
        self._staging_num_tokens = hash_ids.shape[0]

    def finish_stage_offload(self, *, output: torch.Tensor | None = None) -> None:
        if not self.offload:
            return
        self._staged_local = self.embed_tokens.finish_stage_offload(
            self._staging_num_tokens,
            output=output,
        )

    def _reject_stale_row_count(self, staged: torch.Tensor, num_tokens: int | None) -> None:
        if num_tokens is not None and staged.shape[0] != num_tokens:
            raise RuntimeError(
                "engram mixed-mode staging row mismatch: staged "
                f"{staged.shape[0]} rows but this step has "
                f"{num_tokens} tokens; the graph would read the wrong rows"
            )

    def embed(self, hash_ids: torch.Tensor | None, num_tokens: int | None = None) -> torch.Tensor:
        if hash_ids is None:
            if self._staged_local is not None:
                self._reject_stale_row_count(self._staged_local, num_tokens)
                return self.embed_tokens.gather_heads(self._staged_local)
            if self._staged_rows is not None:
                self._reject_stale_row_count(self._staged_rows, num_tokens)
                return self._staged_rows
            raise RuntimeError(
                "engram embed called without hash ids and no staged rows; "
                "offload mode requires the pre-graph staging pass"
            )
        if self.offload:
            # A smaller invocation must gather again; a capacity check would return stale rows.
            if self._staged_local is None or self._staged_local.shape[0] != hash_ids.shape[0]:
                self._staged_local = self.embed_tokens.local_block(hash_ids)
            return self.embed_tokens.gather_heads(self._staged_local[: hash_ids.shape[0]])
        if self._staged_rows is None or self._staged_rows.shape[0] < hash_ids.shape[0]:
            self._staged_rows = self.embed_tokens.lookup(hash_ids)
        return self._staged_rows[: hash_ids.shape[0]]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor | None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = get_forward_context_or_none() if self.offload else None
        execution = context.layer_shared_cache.get(EngramExecutionMetadata) if context is not None else None
        if execution is not None:
            local_rows = execution.local_rows[self.layer_hash_index]
            self._reject_stale_row_count(local_rows, hidden_states.shape[0])
            rows = self.embed_tokens.gather_heads(local_rows)
            token_mask = execution.gate_mask
        else:
            rows = self.embed(hash_ids, num_tokens=hidden_states.shape[0])
        rows = rows.to(self.wkv.weight.dtype)
        kv = self.wkv(rows.flatten(-2))
        num_tokens, hc_mult, dim = hidden_states.shape
        keys = kv[:, : hc_mult * dim].view(num_tokens, hc_mult, dim).float()
        value = kv[:, hc_mult * dim :].float()
        hidden = hidden_states.float()
        # The gate needs the original basis; wkv already produces rotated values.
        gate_hidden = hidden
        if self.gate_unrotate is not None:
            gate_hidden = hidden @ self.gate_unrotate
        hidden_rms = torch.rsqrt(gate_hidden.square().mean(-1) + self.eps)
        key_rms = torch.rsqrt(keys.square().mean(-1) + self.eps)
        dot = (gate_hidden * self.q_weight.float() * self.k_weight.float() * keys).sum(-1)
        dot = dot * hidden_rms * key_rms * float(dim) ** -0.5
        gate_input = dot.abs().clamp_min(_ENGRAM_GATE_CLAMP).sqrt()
        gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
        gate = torch.sigmoid(gate_input)
        if token_mask is not None:
            gate = torch.where(token_mask.reshape(-1, 1), gate, torch.zeros_like(gate))
        output = hidden + gate.unsqueeze(-1) * value.unsqueeze(1)
        return output.to(hidden_states.dtype)


def stage_offload_rows(
    engrams: Sequence[Engram],
    hash_ids: Sequence[torch.Tensor],
    *,
    outputs: Sequence[torch.Tensor] | None = None,
) -> None:
    if not engrams:
        return
    destinations = [None] * len(engrams) if outputs is None else outputs
    for engram, ids, output in zip(engrams, hash_ids, destinations, strict=True):
        engram.begin_stage_offload(ids, output=output)
    _sync_accelerator(engrams[0].embed_tokens.setup_device)
    for engram, output in zip(engrams, destinations, strict=True):
        engram.finish_stage_offload(output=output)


@dataclass(slots=True)
class EngramExecutionMetadata:
    """Rows and gate mask owned by one eager invocation or captured graph."""

    local_rows: tuple[torch.Tensor, ...]
    gate_mask: torch.Tensor | None


class EngramExecutionMetadataBuilder:
    metadata_type = EngramExecutionMetadata
    requires_flat_slot_mapping = False

    def __init__(self, hash_state: EngramHashState, engrams: Sequence[Engram], window_size: int) -> None:
        self._hash_state = hash_state
        self._engrams = tuple(sorted(engrams, key=lambda engram: engram.layer_hash_index))
        self._window_size = window_size
        caches, groups = build_cache_specs_v41([0], [], window_size, 1)
        self._swa_builder = DsaMetadataBuilder(
            caches, groups, read_table_layout=READ_TABLE_LAYOUT_RING, preserve_query_padding=True
        )
        self._cache: torch.Tensor | None = None
        self._eager: EngramExecutionMetadata | None = None

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        self._cache = layer_caches[0].swa if layer_caches else None
        if self._cache is not None:
            self._hash_state.ensure_cache(self._cache)

    def _allocate(self, num_tokens: int, device: torch.device) -> EngramExecutionMetadata:
        return EngramExecutionMetadata(
            local_rows=tuple(
                torch.empty(
                    (num_tokens, engram.embed_tokens.part_n_hash_cols, engram.embed_tokens.dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                for engram in self._engrams
            ),
            gate_mask=torch.empty(num_tokens, dtype=torch.bool, device=device),
        )

    def allocate_persistent(self, input_batch: InputBatch, metadata: AttentionMetadata) -> EngramExecutionMetadata:
        if self._cache is None or self._cache.numel() == 0:
            raise RuntimeError("Engram graph capture requires a bound SWA cache")
        result = self._allocate(input_batch.num_tokens_after_padding, input_batch.input_ids.device)
        for engram in self._engrams:
            engram.embed_tokens._graph_metadata_bound = True
        return result

    def build(self, input_batch: InputBatch, metadata: AttentionMetadata) -> EngramExecutionMetadata:
        if self._cache is None or self._cache.numel() == 0:
            return EngramExecutionMetadata((), None)
        size = input_batch.num_tokens_after_padding
        if self._eager is None or self._eager.gate_mask.numel() < size:
            capacity = max(size, 2 * self._eager.gate_mask.numel() if self._eager else 1)
            self._eager = self._allocate(capacity, input_batch.input_ids.device)
        result = EngramExecutionMetadata(
            tuple(rows[:size] for rows in self._eager.local_rows), self._eager.gate_mask[:size]
        )
        self.update_persistent(result, input_batch, metadata)
        return result

    def update_persistent(
        self, persistent_metadata: object, input_batch: InputBatch, metadata: AttentionMetadata
    ) -> None:
        if not isinstance(persistent_metadata, EngramExecutionMetadata):
            raise TypeError("Engram received invalid execution metadata")
        size = input_batch.num_tokens_after_padding
        if persistent_metadata.gate_mask is None or persistent_metadata.gate_mask.shape != (size,):
            raise RuntimeError("Engram metadata capacity does not match the execution batch")
        if input_batch.is_dummy:
            for rows in persistent_metadata.local_rows:
                rows.zero_()
            persistent_metadata.gate_mask.zero_()
            return
        if self._cache is None or not self._hash_state.ensure_cache(self._cache):
            raise RuntimeError("Engram host offload requires a bound SWA cache before execution")
        if not metadata.multi_block_tables or metadata.multi_block_tables[0] is None:
            raise RuntimeError("Engram host offload requires the SWA block table")
        query_lengths = list(input_batch.num_scheduled_tokens)
        context_lengths = [
            computed + scheduled
            for computed, scheduled in zip(
                input_batch.num_computed_tokens, input_batch.num_scheduled_tokens, strict=True
            )
        ]
        padding = size - sum(query_lengths)
        if padding < 0:
            raise ValueError("Engram query lengths exceed the execution batch")
        if padding:
            query_lengths.append(padding)
            context_lengths.append(0)
        block_table = metadata.multi_block_tables[0]
        slots = self._swa_builder.build_swa_slot_mapping(
            block_table,
            self._window_size,
            context_lengths,
            query_lengths,
            len(query_lengths),
            size,
            getattr(metadata, "new_cache_slots_host_values", None),
        )
        device = input_batch.input_ids.device
        query_ends = [0]
        for count in query_lengths:
            query_ends.append(query_ends[-1] + count)
        dead = image_sentinel_mask(input_batch.input_ids)
        if input_batch.is_padding is not None:
            dead = dead | input_batch.is_padding
        elif padding:
            dead = dead | (torch.arange(size, device=device) >= size - padding)
        hashes = self._hash_state(
            input_batch.input_ids,
            input_batch.positions,
            torch.tensor(query_ends, dtype=torch.int64, device=device),
            dead,
            None,
            slots.to(device=device, dtype=torch.int64),
            block_table.to(device=device),
            self._window_size,
        )
        persistent_metadata.gate_mask.copy_(~dead)
        stage_offload_rows(
            self._engrams,
            [hashes[:, engram.layer_hash_index].to(torch.int64) for engram in self._engrams],
            outputs=persistent_metadata.local_rows,
        )
