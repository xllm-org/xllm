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
from dataclasses import dataclass
from typing import Any

_ALLOWED_KEYS = ("cpu_offload", "dp_shared_memory")


@dataclass(frozen=True)
class EngramConfig:
    cpu_offload: bool = False
    dp_shared_memory: bool = False

    @classmethod
    def from_value(cls, value: Any) -> EngramConfig:
        if value is None:
            return cls()
        if isinstance(value, str):
            value = cls._decode_json(value)
        if isinstance(value, bool):
            return cls(cpu_offload=value)
        if not isinstance(value, dict):
            raise ValueError(f"engram_config must be a JSON object or boolean, got {type(value).__name__}")
        return cls._from_object(value)

    @staticmethod
    def _decode_json(text: str) -> Any:
        duplicates: list[str] = []

        def _collect_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            seen: set[str] = set()
            for key, _ in pairs:
                if key in seen:
                    duplicates.append(key)
                seen.add(key)
            return dict(pairs)

        try:
            decoded = json.loads(text, object_pairs_hook=_collect_duplicates)
        except json.JSONDecodeError as exc:
            raise ValueError(f"engram_config is not valid JSON: {exc}") from exc
        if duplicates:
            raise ValueError(f"engram_config JSON has duplicate key(s): {', '.join(duplicates)}")
        return decoded

    @classmethod
    def _from_object(cls, obj: dict[str, Any]) -> EngramConfig:
        unknown = [key for key in obj if key not in _ALLOWED_KEYS]
        if unknown:
            raise ValueError(f"engram_config has unknown key(s) {unknown}; allowed keys are {list(_ALLOWED_KEYS)}")
        cpu_offload = cls._bool_field(obj, "cpu_offload")
        dp_shared_memory = cls._bool_field(obj, "dp_shared_memory")
        if dp_shared_memory and not cpu_offload:
            raise ValueError("engram_config.dp_shared_memory=true requires engram_config.cpu_offload=true")
        return cls(cpu_offload=cpu_offload, dp_shared_memory=dp_shared_memory)

    @staticmethod
    def _bool_field(obj: dict[str, Any], key: str) -> bool:
        value = obj.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"engram_config.{key} must be a boolean, got {type(value).__name__}")
        return value
