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
from typing import Any

import pytest
import torch

from tests.python._engram_reference import _tiny_engram_hash_state
from xllm.python.models.deepseek_v41_engram import (
    ENGRAM_DEAD_ID,
    ENGRAM_IMAGE_SENTINEL_BASE_ID,
    EngramHashState,
    build_compressed_token_map_from_texts,
)

_TOKEN_MAP, _ = build_compressed_token_map_from_texts(["The", "the", "THE", " thé ", "x", "y"])


def _bind_cpu_cache(state: EngramHashState, rows: int = 64) -> torch.Tensor:
    swa_cache = torch.arange(rows, dtype=torch.float32).reshape(rows, 1, 1)
    assert state.ensure_cache(swa_cache)
    return swa_cache


def _decode_token_hashes(state: EngramHashState, token_id: int, position: int, slot: int) -> torch.Tensor:
    ids = torch.tensor([token_id])
    block_table = torch.tensor([[0, 0]], dtype=torch.int32)
    return state(
        ids,
        torch.tensor([position]),
        torch.tensor([0, 1]),
        torch.zeros(1, dtype=torch.bool),
        None,
        torch.tensor([slot], dtype=torch.int32),
        block_table,
        32,
    )


class TestRestoreHistorySemantics:
    def test_restore_then_hash_reads_restored_history(self) -> None:
        state = _tiny_engram_hash_state(_TOKEN_MAP)
        _bind_cpu_cache(state)
        comp = state.token_map.to(torch.int64)
        m = state.multipliers[0]

        state.restore_history(torch.tensor([[0, 4], [1, 5]], dtype=torch.int64))
        cache = state._cache
        assert int(cache[0]) == int(comp[4])
        assert int(cache[1]) == int(comp[5])

        hashes = _decode_token_hashes(state, token_id=3, position=2, slot=2)
        expected = (comp[3] * m[0]) ^ (comp[5] * m[1]) ^ (comp[4] * m[2])
        assert torch.equal(hashes[0, 0, 2], ((expected % 73) + 138).to(torch.int32))
        assert int(state._cache[2]) == int(comp[3])

    def test_restore_requires_bound_cache(self) -> None:
        state = _tiny_engram_hash_state(_TOKEN_MAP)
        with pytest.raises(RuntimeError, match="bound cache"):
            state.restore_history(torch.tensor([[0, 5]], dtype=torch.int64))

    def test_restore_rejects_non_pair_shapes(self) -> None:
        state = _tiny_engram_hash_state(_TOKEN_MAP)
        _bind_cpu_cache(state)
        with pytest.raises(ValueError, match="pairs"):
            state.restore_history(torch.tensor([[0, 5, 7]], dtype=torch.int64))
        with pytest.raises(ValueError, match="pairs"):
            state.restore_history(torch.tensor([0, 5], dtype=torch.int64))

    def test_restore_maps_image_sentinel_to_dead_id(self) -> None:
        state = _tiny_engram_hash_state(_TOKEN_MAP)
        _bind_cpu_cache(state)
        state.restore_history(torch.tensor([[0, ENGRAM_IMAGE_SENTINEL_BASE_ID]], dtype=torch.int64))
        assert int(state._cache[0]) == ENGRAM_DEAD_ID

        comp = state.token_map.to(torch.int64)
        m = state.multipliers[0]
        pad_id = state.pad_id
        hashes = _decode_token_hashes(state, token_id=3, position=2, slot=2)
        r = (comp[3] * m[0]) ^ (pad_id * m[1])
        assert torch.equal(hashes[0, 0, 0], (r % 67).to(torch.int32))

    def test_restore_duplicate_slots_last_write_wins(self) -> None:
        state = _tiny_engram_hash_state(_TOKEN_MAP)
        _bind_cpu_cache(state)
        comp = state.token_map.to(torch.int64)
        state.restore_history(torch.tensor([[0, 4], [1, 5], [0, 2]], dtype=torch.int64))
        assert int(state._cache[0]) == int(comp[2])
        assert int(state._cache[1]) == int(comp[5])

    def test_restore_drops_negative_slots(self) -> None:
        state = _tiny_engram_hash_state(_TOKEN_MAP)
        _bind_cpu_cache(state)
        comp = state.token_map.to(torch.int64)
        state.restore_history(torch.tensor([[-1, 4], [3, 5]], dtype=torch.int64))
        assert int(state._cache[3]) == int(comp[5])
        assert int(state._cache[0]) == 0


class TestModelRestoreHook:
    def _make_model(self) -> Any:
        from xllm.python.models.deepseek_v41 import DeepseekV41Model

        class _Carrier:
            restore_engram_history = DeepseekV41Model.restore_engram_history

        carrier = _Carrier()
        carrier.engram_hash = _tiny_engram_hash_state(_TOKEN_MAP)
        return carrier

    def test_restore_noop_on_empty_history(self) -> None:
        model = self._make_model()

        class _Backend:
            def swa_history_cache(self) -> torch.Tensor:
                raise AssertionError("empty history must not resolve the cache")

        model.restore_engram_history(_Backend(), SimpleNamespace(engram_history=[]))
        assert model.engram_hash._cache is None

    def test_restore_requires_sw_a_history_cache(self) -> None:
        model = self._make_model()
        _bind_cpu_cache(model.engram_hash)

        class _Backend:
            def swa_history_cache(self) -> torch.Tensor:
                return torch.arange(64, dtype=torch.float32).reshape(64, 1, 1)

        model.restore_engram_history(_Backend(), SimpleNamespace(engram_history=[0, 4, 1, 5]))
        comp = model.engram_hash.token_map.to(torch.int64)
        assert int(model.engram_hash._cache[0]) == int(comp[4])
        assert int(model.engram_hash._cache[1]) == int(comp[5])

    def test_restore_rejects_odd_history(self) -> None:
        model = self._make_model()
        _bind_cpu_cache(model.engram_hash)

        class _Backend:
            def swa_history_cache(self) -> torch.Tensor:
                return torch.arange(64, dtype=torch.float32).reshape(64, 1, 1)

        with pytest.raises(ValueError, match="pairs"):
            model.restore_engram_history(_Backend(), SimpleNamespace(engram_history=[0, 4, 1]))

    def test_restore_unbound_engram_hash_is_noop(self) -> None:
        model = self._make_model()
        model.engram_hash = None

        class _Backend:
            def swa_history_cache(self) -> torch.Tensor:
                raise AssertionError("engram_hash=None must not resolve the cache")

        model.restore_engram_history(_Backend(), SimpleNamespace(engram_history=[0, 4]))

    def test_restore_unbound_cache_raises(self) -> None:
        model = self._make_model()

        class _Backend:
            def swa_history_cache(self) -> torch.Tensor:
                return torch.zeros(0)

        with pytest.raises(RuntimeError, match="cache binding"):
            model.restore_engram_history(_Backend(), SimpleNamespace(engram_history=[0, 4]))


class _Metadata:
    def __init__(self, history: tuple = ()) -> None:
        self.is_prefill = False
        self.is_chunked_prefill = False
        self.is_spec_verify = False
        self.engram_history = history


class _RecordingRunner:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def execute(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append("eager_execute")
        return "output"


class TestExecutorRestoreHookWiring:
    @staticmethod
    def _executor(calls: list[str], model: Any) -> Any:
        from xllm.python.model_executor.executor import ModelExecutor

        executor = ModelExecutor.__new__(ModelExecutor)
        executor.requires_input_batch_metadata = False
        executor._kv_bound = True
        executor.context_kv_graph_runner = None
        executor.layerwise_split_size = 1
        executor.attention_backend = "backend"
        executor.decode_graph_runner = None
        executor.inductor_runner = None
        executor._restore_engram_history = getattr(model, "restore_engram_history", None)
        executor.eager_runner = _RecordingRunner(calls)
        return executor

    def test_execute_invokes_hook_after_kv_bound_before_runner(self) -> None:
        calls: list[str] = []

        class _Model:
            def restore_engram_history(self, backend: Any, metadata: Any) -> None:
                assert backend == "backend"
                calls.append(f"restore:{len(metadata.engram_history)}")

        executor = self._executor(calls, _Model())
        assert executor.execute(None, None, _Metadata((0, 4, 1, 5))) == "output"
        assert calls == ["restore:4", "eager_execute"]

        calls.clear()
        assert executor.execute(None, None, _Metadata(())) == "output"
        assert calls == ["restore:0", "eager_execute"]

    def test_execute_checks_kv_bound_before_hook(self) -> None:
        calls: list[str] = []

        class _Model:
            def restore_engram_history(self, backend: Any, metadata: Any) -> None:
                calls.append("restore")

        executor = self._executor(calls, _Model())
        executor._kv_bound = False
        with pytest.raises(RuntimeError, match="not bound"):
            executor.execute(None, None, _Metadata((0, 4)))
        assert calls == []
