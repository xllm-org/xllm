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


from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from xllm.python.kernels_npu import op_api


@pytest.fixture
def native_ops(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    namespace = SimpleNamespace()
    monkeypatch.setattr(op_api, "torch", SimpleNamespace(ops=SimpleNamespace(xllm_ops=namespace)))
    op_api._has_registered_op_api.cache_clear()
    yield namespace
    op_api._has_registered_op_api.cache_clear()


def test_missing_registration_can_be_loaded_later(native_ops: SimpleNamespace) -> None:
    assert not op_api.has_op_api("aclnnHcPreVa")
    native_ops.has_npu_op_api = lambda name: name == "aclnnHcPreVa"
    assert op_api.has_op_api("aclnnHcPreVa")


@pytest.mark.parametrize("available", [False, True])
def test_vendor_result_is_cached_outside_decode_loop(native_ops: SimpleNamespace, available: bool) -> None:
    queries = []

    def probe(name: str) -> bool:
        queries.append(name)
        return available

    native_ops.has_npu_op_api = probe
    assert op_api.has_op_api("aclnnHcPreVa") is available
    assert op_api.has_op_api("aclnnHcPreVa") is available
    assert queries == ["aclnnHcPreVa"]


def test_operations_and_replaced_registrations_are_independent(native_ops: SimpleNamespace) -> None:
    native_ops.has_npu_op_api = lambda name: name == "aclnnHcPreVa"
    assert op_api.has_op_api("aclnnHcPreVa")
    assert not op_api.has_op_api("aclnnHcPostVa")
    native_ops.has_npu_op_api = lambda name: False
    assert not op_api.has_op_api("aclnnHcPreVa")
