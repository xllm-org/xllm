/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "layers/mlu/dcp_batch_metadata.h"

#include <glog/logging.h>

#include <tuple>
#include <utility>

#include "layers/common/attention_metadata.h"

namespace xllm::layer {

std::pair<torch::Tensor, torch::Tensor> build_prefill_slot_order(
    const torch::Tensor& source_slots) {
  CHECK_EQ(source_slots.dim(), 1);
  CHECK(source_slots.scalar_type() == torch::kInt32 ||
        source_slots.scalar_type() == torch::kInt64);
  auto [slots, rows] = torch::sort(source_slots);
  const int64_t token_count = slots.numel();
  if (token_count > 1) {
    torch::Tensor repeated =
        (slots.slice(0, 1) == slots.slice(0, 0, -1)) & (slots.slice(0, 1) >= 0);
    repeated = torch::cat({torch::zeros({1}, repeated.options()), repeated});
    torch::Tensor indices = torch::arange(token_count, rows.options());
    // index_select checks bounds on MLU without copying a predicate to CPU.
    // A duplicate source has no unique K row and must fail, not pick a winner.
    rows = rows.index_select(0, torch::where(repeated, token_count, indices));
  }
  return {std::move(slots), std::move(rows)};
}

std::shared_ptr<const KVShardBatchMetadata> build_mlu_shard_metadata(
    const AttentionMetadata& attention_metadata,
    const KVShardLayout& layout) {
  auto metadata = build_kv_shard_batch_metadata(attention_metadata, layout);
  if (attention_metadata.is_dummy) {
    return metadata;
  }
  if (attention_metadata.kv_seq_lens.defined()) {
    metadata->local_indexer_context_lens =
        localize_kv_shard_context_lens(attention_metadata.kv_seq_lens, layout);
  }
  if (attention_metadata.is_prefill) {
    std::tie(metadata->prefill_sorted_slots, metadata->prefill_sorted_rows) =
        build_prefill_slot_order(attention_metadata.slot_mapping);
    metadata->prefill_sorted_slots =
        metadata->prefill_sorted_slots.to(torch::kInt64);
  }
  // The causal selector is consumed only by DCP prefill paths. Decode graph
  // buckets pad sequence metadata independently of token rows, so building it
  // there is both unnecessary and can produce a non-token-shaped selector.
  if ((attention_metadata.is_prefill ||
       attention_metadata.is_chunked_prefill) &&
      attention_metadata.q_cu_seq_lens.defined() &&
      attention_metadata.kv_cu_seq_lens.defined() &&
      attention_metadata.block_table.defined() &&
      attention_metadata.slot_mapping.defined()) {
    metadata->causal_selector =
        build_kv_shard_causal_selector_metadata(attention_metadata, layout);
  }
  return metadata;
}

AttentionMetadata build_dcp_prefill_selector_metadata(
    AttentionMetadata selector_metadata,
    const KVShardLayout& layout,
    int64_t index_topk) {
  const auto* shard = selector_metadata.kv_shard_batch_metadata.get();
  const KVShardCausalSelectorMetadata causal_selector =
      shard != nullptr && shard->causal_selector.block_table.defined()
          ? shard->causal_selector
          : build_kv_shard_causal_selector_metadata(selector_metadata, layout);
  selector_metadata.q_cu_seq_lens = causal_selector.q_cu_seq_lens;
  selector_metadata.kv_cu_seq_lens = causal_selector.q_cu_seq_lens;
  selector_metadata.kv_seq_lens = causal_selector.local_context_lens;
  selector_metadata.block_table = causal_selector.block_table;
  selector_metadata.max_query_len = 1;
  selector_metadata.max_seq_len = index_topk;
  // The decode-view selector does not consume total_kv_len. Keeping this
  // host metadata scalar avoids a device-to-host synchronization here.
  selector_metadata.total_kv_len = 0;
  selector_metadata.is_prefill = false;
  selector_metadata.is_chunked_prefill = false;
  return selector_metadata;
}

}  // namespace xllm::layer
