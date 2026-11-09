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

import mmap
import os
import secrets
import stat
from multiprocessing import shared_memory
from typing import Any, cast

import torch
from torch.distributed import ProcessGroup

from xllm.python.distributed import (
    engram_all_gather_object,
    engram_barrier,
    engram_broadcast_object_list,
)

STATE_INIT = "init"
STATE_ATTACHED = "attached"
STATE_LOADED = "loaded"
STATE_FAILED = "failed"
STATE_CLOSED = "closed"

# Capacity is a precheck; attaching with the matching token proves shared-memory locality.
_SHM_DIR = "/dev/shm"

_TOKEN_BYTES = 32


def estimate_shared_bytes(rows: int, dim: int, quant_block: int) -> int:
    scale_cols = dim // quant_block
    return int(rows) * int(dim) + int(rows) * scale_cols * 4


def ensure_dev_shm_capacity(required_bytes: int) -> None:
    try:
        stat = os.statvfs(_SHM_DIR)
    except OSError:
        return
    free = stat.f_bavail * stat.f_frsize
    if required_bytes > free:
        raise RuntimeError(
            f"engram DP-shared host table needs {required_bytes} bytes in {_SHM_DIR} but only "
            f"{free} bytes are free; reduce the shared engram footprint on this node or "
            "disable engram_config.dp_shared_memory"
        )


class SharedHostEngramStorage:
    def __init__(
        self,
        *,
        layer_id: int,
        rows: int,
        dim: int,
        quant_block: int,
        group: ProcessGroup,
    ) -> None:
        if rows < 0 or dim <= 0 or dim % quant_block != 0:
            raise ValueError(f"invalid shared Engram geometry rows={rows}, dim={dim}, quant_block={quant_block}")
        self.layer_id = int(layer_id)
        self.rows = int(rows)
        self.dim = int(dim)
        self.quant_block = int(quant_block)
        self.scale_cols = self.dim // self.quant_block
        self.codes_bytes = self.rows * self.dim
        self.scales_bytes = self.rows * self.scale_cols * 4
        self.group = group
        self.rank = int(group.rank())
        self.world_size = int(group.size())
        self.owner = self.rank == 0
        self.token: str | None = None
        self._codes_handle: shared_memory.SharedMemory | mmap.mmap | None = None
        self._scales_handle: shared_memory.SharedMemory | mmap.mmap | None = None
        self._state = STATE_INIT
        self.codes: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None

    def create_or_attach(self) -> None:
        if self._state != STATE_INIT:
            raise RuntimeError(f"engram shared storage for layer {self.layer_id} is already {self._state}")
        self._attach_or_create()
        self._state = STATE_ATTACHED

    def mark_loaded(self) -> None:
        if self._state == STATE_ATTACHED:
            self._state = STATE_LOADED

    def mark_failed(self) -> None:
        if self._state in (STATE_ATTACHED, STATE_LOADED):
            self._state = STATE_FAILED

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_loaded(self) -> bool:
        return self._state == STATE_LOADED

    def _attach_or_create(self) -> None:
        payload: list[dict[str, Any] | None] = [None]
        if self.owner:
            try:
                self._check_dev_shm_capacity()
                codes = shared_memory.SharedMemory(create=True, size=self.codes_bytes + _TOKEN_BYTES)
                self._codes_handle = codes
                scales = shared_memory.SharedMemory(create=True, size=self.scales_bytes)
                self._scales_handle = scales
                token = secrets.token_hex(16)
                self.token = token
                with self._buffer(codes) as codes_buf:
                    codes_buf[self.codes_bytes : self.codes_bytes + _TOKEN_BYTES] = token.encode("ascii")
                payload = [
                    {
                        "codes": codes.name,
                        "scales": scales.name,
                        "codes_bytes": self.codes_bytes,
                        "scales_bytes": self.scales_bytes,
                        "token": self.token,
                        "layer_id": self.layer_id,
                        "rows": self.rows,
                        "dim": self.dim,
                        "quant_block": self.quant_block,
                    }
                ]
            except Exception as exc:  # noqa: BLE001 - surfaced to every rank below
                self._discard_partial_creation()
                payload = [{"error": f"{type(exc).__name__}: {exc}"}]
        engram_broadcast_object_list(payload, src=0, group=self.group)
        descriptor = payload[0]
        if not isinstance(descriptor, dict):
            raise RuntimeError(
                f"engram shared host-table for layer {self.layer_id} got a malformed descriptor {descriptor!r}"
            )
        if "error" in descriptor:
            raise RuntimeError(
                f"engram shared host-table creation failed for layer {self.layer_id}: {descriptor['error']}"
            )
        error: str | None = None
        try:
            # Aggregate descriptor errors before raising so peers do not hang in a missing collective.
            self._validate_descriptor(descriptor)
            if not self.owner:
                self._codes_handle = self._attach(descriptor["codes"], self.codes_bytes + _TOKEN_BYTES)
                self._scales_handle = self._attach(descriptor["scales"], self.scales_bytes)
                self._verify_segment_token(self._codes_handle, descriptor["token"])
            assert self._codes_handle is not None and self._scales_handle is not None
            # Bound frombuffer by count: shared-memory allocations can be rounded up to a page.
            self.codes = torch.frombuffer(
                self._buffer(self._codes_handle), dtype=torch.int8, count=self.rows * self.dim
            ).reshape(self.rows, self.dim)
            self.scales = torch.frombuffer(
                self._buffer(self._scales_handle), dtype=torch.float32, count=self.rows * self.scale_cols
            ).reshape(self.rows, self.scale_cols)
            self.token = descriptor["token"]
        except Exception as exc:  # noqa: BLE001 - aggregated below
            error = f"{type(exc).__name__}: {exc}"

        errors: list[str | None] = [None] * self.world_size
        engram_all_gather_object(errors, error, group=self.group)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        engram_barrier(group=self.group)
        if failures:
            # Unlink before close: mappings remain valid, but close drops the handles needed to unlink.
            if self.owner:
                self._unlink_created()
            try:
                self.close()
            except RuntimeError as exc:
                failures = f"{failures}; release failed: {exc}"
            raise RuntimeError(f"engram shared host-table initialization failed for layer {self.layer_id}: {failures}")
        if self.owner:
            self._unlink_created()

    def _validate_descriptor(self, descriptor: dict[str, Any]) -> None:
        expected = {
            "layer_id": self.layer_id,
            "rows": self.rows,
            "dim": self.dim,
            "quant_block": self.quant_block,
            "codes_bytes": self.codes_bytes,
            "scales_bytes": self.scales_bytes,
        }
        for key, value in expected.items():
            if descriptor.get(key) != value:
                raise RuntimeError(
                    f"engram shared host-table descriptor mismatch for layer {self.layer_id}: "
                    f"{key}={descriptor.get(key)!r}, expected {value!r}"
                )
        token = descriptor.get("token")
        if not isinstance(token, str) or len(token) != _TOKEN_BYTES:
            raise RuntimeError(
                f"engram shared host-table descriptor for layer {self.layer_id} has a missing/malformed token {token!r}"
            )

    def _verify_segment_token(self, codes: shared_memory.SharedMemory | mmap.mmap, token: str) -> None:
        with self._buffer(codes) as buf:
            attached = bytes(buf[self.codes_bytes : self.codes_bytes + _TOKEN_BYTES])
        if attached != token.encode("ascii"):
            raise RuntimeError(
                f"engram shared segment token mismatch for layer {self.layer_id}: "
                "the attached codes segment does not carry this leader's token"
            )

    @staticmethod
    def _buffer(handle: shared_memory.SharedMemory | mmap.mmap) -> memoryview:
        # The exported view keeps its mapping alive while a tensor uses it.
        return memoryview(handle if isinstance(handle, mmap.mmap) else handle.buf)

    @staticmethod
    def _attach(name: str, expected_size: int) -> mmap.mmap:
        """Attach without registering another resource-tracker owner."""
        if not isinstance(name, str) or not name or "/" in name or "\0" in name or name in {".", ".."}:
            raise ValueError("invalid shared Engram segment name")
        fd = os.open(os.path.join(_SHM_DIR, name), os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < expected_size:
                raise RuntimeError(
                    f"engram shared segment {name} is {metadata.st_size} bytes, expected at least {expected_size}"
                )
            return mmap.mmap(fd, expected_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)

    def _check_dev_shm_capacity(self) -> None:
        ensure_dev_shm_capacity(self.codes_bytes + self.scales_bytes + _TOKEN_BYTES)

    def _discard_partial_creation(self) -> None:
        self._unlink_created(warn=False)
        for handle in (self._codes_handle, self._scales_handle):
            if handle is None:
                continue
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup of a failed create
                pass
        self._codes_handle = None
        self._scales_handle = None

    def _unlink_created(self, *, warn: bool = True) -> None:
        for handle in (self._codes_handle, self._scales_handle):
            if handle is None:
                continue
            try:
                cast(shared_memory.SharedMemory, handle).unlink()
            except OSError as exc:
                if warn:
                    self._warn(f"engram shared backing unlink failed for layer {self.layer_id}: {exc}")

    def close(self) -> None:
        """Callers must drop all parameter aliases before closing mappings; failed handles remain retryable."""
        if self._state == STATE_CLOSED:
            return
        self.codes = None
        self.scales = None
        failures: list[str] = []
        for attr in ("_codes_handle", "_scales_handle"):
            handle = getattr(self, attr)
            if handle is None:
                continue
            try:
                handle.close()
                setattr(self, attr, None)
            except Exception as exc:  # noqa: BLE001 - aggregated; state stays retryable
                failures.append(f"{attr}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError(
                f"engram shared host-table close failed for layer {self.layer_id}: " + "; ".join(failures)
            )
        self._state = STATE_CLOSED

    def __del__(self) -> None:
        # Never enter a distributed collective from garbage collection.
        try:
            self.close()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass

    @staticmethod
    def _warn(message: str) -> None:
        try:
            from scripts.logger import logger

            logger.warning(message)
        except Exception:  # noqa: BLE001 - logging must never block teardown
            pass


__all__ = [
    "STATE_ATTACHED",
    "STATE_CLOSED",
    "STATE_FAILED",
    "STATE_INIT",
    "STATE_LOADED",
    "SharedHostEngramStorage",
    "ensure_dev_shm_capacity",
    "estimate_shared_bytes",
]
