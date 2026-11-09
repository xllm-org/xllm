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

from xllm.python.models.engram_config import EngramConfig


def test_from_value_none_and_defaults_disable_both_switches() -> None:
    assert EngramConfig.from_value(None) == EngramConfig(cpu_offload=False, dp_shared_memory=False)
    assert EngramConfig.from_value({}) == EngramConfig()
    assert EngramConfig.from_value("{}") == EngramConfig()


def test_from_value_bool_sets_cpu_offload_only() -> None:
    assert EngramConfig.from_value(True) == EngramConfig(cpu_offload=True)
    assert EngramConfig.from_value(False) == EngramConfig(cpu_offload=False)
    assert EngramConfig.from_value("true") == EngramConfig(cpu_offload=True)
    assert EngramConfig.from_value("false") == EngramConfig(cpu_offload=False)


def test_from_value_object_reads_real_booleans() -> None:
    assert EngramConfig.from_value({"cpu_offload": True}) == EngramConfig(cpu_offload=True)
    assert EngramConfig.from_value({"cpu_offload": True, "dp_shared_memory": True}) == EngramConfig(
        cpu_offload=True, dp_shared_memory=True
    )
    assert EngramConfig.from_value('{"cpu_offload": true, "dp_shared_memory": false}') == EngramConfig(cpu_offload=True)


@pytest.mark.parametrize("value", [1, 0, [], [True], "1", "0", 1.0])
def test_from_value_rejects_non_object_non_bool(value: object) -> None:
    with pytest.raises(ValueError, match="must be a JSON object or boolean"):
        EngramConfig.from_value(value)


@pytest.mark.parametrize(
    "obj",
    [
        {"cpu_offload": "true"},
        {"cpu_offload": 1},
        {"cpu_offload": 0},
        {"dp_shared_memory": "yes"},
        {"cpu_offload": True, "dp_shared_memory": 0},
    ],
)
def test_from_value_rejects_stringly_typed_fields(obj: dict) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        EngramConfig.from_value(obj)


def test_from_value_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        EngramConfig.from_value({"cpu_offload": True, "bogus": True})


def test_from_value_rejects_duplicate_json_keys() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        EngramConfig.from_value('{"cpu_offload": true, "cpu_offload": false}')


def test_from_value_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        EngramConfig.from_value('{"cpu_offload": tru}')


@pytest.mark.parametrize(
    "obj",
    [
        {"dp_shared_memory": True},
        {"cpu_offload": False, "dp_shared_memory": True},
    ],
)
def test_from_value_rejects_shared_without_offload(obj: dict) -> None:
    with pytest.raises(ValueError, match=r"dp_shared_memory=true requires engram_config.cpu_offload=true"):
        EngramConfig.from_value(obj)


def _validate(cfg: SimpleNamespace) -> None:
    from xllm.python.models.deepseek_v41 import _validate_engram_config

    _validate_engram_config(cfg)


def _engram_cfg(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "engram_layer_ids": [1],
        "tp_size": 1,
        "pp_size": 1,
        "dcp_size": 1,
        "engram_config": EngramConfig(cpu_offload=True),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_validate_is_skipped_without_engram_layers() -> None:
    _validate(_engram_cfg(engram_layer_ids=[], tp_size=16, pp_size=2, dcp_size=2))


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_validate_accepts_supported_tp(tp_size: int) -> None:
    _validate(_engram_cfg(tp_size=tp_size))


@pytest.mark.parametrize("tp_size", [3, 6, 12, 16])
def test_validate_rejects_unsupported_tp(tp_size: int) -> None:
    with pytest.raises(ValueError, match=r"TP in \{1, 2, 4, 8\}"):
        _validate(_engram_cfg(tp_size=tp_size))


@pytest.mark.parametrize("field", ["pp_size", "dcp_size"])
def test_validate_rejects_non_singleton_pp_or_dcp(field: str) -> None:
    with pytest.raises(ValueError, match="PP=1 and DCP=1"):
        _validate(_engram_cfg(**{field: 2}))


@pytest.mark.parametrize("field,value", [("tp_size", 16), ("pp_size", 2), ("dcp_size", 2)])
def test_resident_engram_keeps_model_topology_constraints(field: str, value: int) -> None:
    _validate(_engram_cfg(engram_config=EngramConfig(), **{field: value}))
