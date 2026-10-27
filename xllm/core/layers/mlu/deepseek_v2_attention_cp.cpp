/* Copyright 2025-2026 The xLLM Authors.

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

#include <tuple>

#include "core/layers/mlu/dcp_batch_metadata.h"
#include "layers/mlu/deepseek_v2_attention.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

torch::Tensor DeepseekV2AttentionImpl::forward_cp(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    const v32_cp::DeepseekV32CPContext& cp_ctx,
    KVCache& kv_cache,
    bool is_prefill_or_chunked_prefill,
    DsaTopkTransfer* topk_transfer) {
  CHECK(can_use_cp(topk_transfer))
      << "deepseek_v32 context parallel requires either a lighting indexer "
         "or reused top-k state.";
  CHECK(is_prefill_or_chunked_prefill)
      << "deepseek_v32 context parallel only supports prefill batches.";
  const bool use_dcp = enable_mla_cache_sharding_ && !attn_metadata.is_dummy;
  std::optional<torch::Tensor> k_cache_scale = kv_cache.get_k_cache_scale();
  QueryPrep query_prep = prep_query(hidden_states, tp_heads());

  std::optional<DsaTopkState> topk_state;
  std::optional<DsaTopkState> local_topk;
  v32_cp::PaddedGatherHandle mla_handle;
  IndexerCPPreOut index_pre;
  v32_cp::PaddedGatherHandle index_handle;
  std::optional<DcpIndexerGatherAsyncCtx> candidate_gather;
  int64_t candidate_topk = 0;
  const DsaTopkState* reused_topk =
      topk_transfer != nullptr ? topk_transfer->input() : nullptr;
  const bool compute_topk = !attn_metadata.is_dummy && reused_topk == nullptr;

  Device device(hidden_states.device());
  CHECK(cp_comm_stream_ != nullptr)
      << "context-parallel attention requires a model-scoped communication "
         "stream";
  if (compute_topk) {
    index_pre = indexer_->cp_pre(hidden_states,
                                 query_prep.q_norm,
                                 positions,
                                 cp_ctx.local_attn_metadata,
                                 /*quantize_output=*/false);
    auto compute_stream = device.current_stream();
    cp_comm_stream_->wait_stream(*compute_stream);
    {
      torch::StreamGuard stream_guard = cp_comm_stream_->set_stream_guard();
      index_handle =
          v32_cp::launch_all_gather_across_ranks(index_pre.k_local, cp_ctx);
    }
  }

  auto mla_inputs =
      build_cp_mla_inputs(hidden_states, positions, query_prep, cp_ctx);

  torch::Tensor k_gathered;
  if (compute_topk) {
    k_gathered = indexer_->cp_wait_k(index_pre.k_local, index_handle);
  }
  auto compute_stream = device.current_stream();
  cp_comm_stream_->wait_stream(*compute_stream);
  {
    torch::StreamGuard stream_guard = cp_comm_stream_->set_stream_guard();
    mla_handle =
        v32_cp::launch_all_gather_across_ranks(mla_inputs.k_input, cp_ctx);
  }
  if (compute_topk) {
    torch::Tensor index_cache = kv_cache.get_index_cache();
    std::optional<torch::Tensor> index_cache_scale =
        kv_cache.get_indexer_cache_scale();
    if (use_dcp) {
      CHECK(dcp_decode_context_ != nullptr);
      IndexerCPPreOut global_pre;
      global_pre.q = v32_cp::gather_and_restore_global(index_pre.q, cp_ctx);
      global_pre.k_local =
          v32_cp::restore_gathered_to_global_order(k_gathered, cp_ctx);
      global_pre.weights =
          v32_cp::gather_and_restore_global(index_pre.weights, cp_ctx);
      if (index_pre.q_scale.has_value()) {
        global_pre.q_scale =
            v32_cp::gather_and_restore_global(*index_pre.q_scale, cp_ctx);
      }

      AttentionMetadata local_indexer_metadata = attn_metadata;
      const std::shared_ptr<const KVShardBatchMetadata>& shard_metadata =
          attn_metadata.kv_shard_batch_metadata;
      if (shard_metadata != nullptr) {
        CHECK(shard_metadata->local_slot_mapping.defined())
            << "cache-sharded CP requires localized indexer slots";
        local_indexer_metadata.slot_mapping =
            shard_metadata->local_slot_mapping;
      } else {
        local_indexer_metadata.slot_mapping =
            dcp_decode_context_->localize_slots(attn_metadata.slot_mapping);
      }
      AttentionMetadata selector_metadata = build_dcp_prefill_selector_metadata(
          local_indexer_metadata, dcp_decode_context_->layout(), index_topk_);
      DcpIndexerLocalCandidates candidates =
          indexer_->forward_dcp_local_prefill_from_cp(global_pre,
                                                      index_cache,
                                                      local_indexer_metadata,
                                                      selector_metadata,
                                                      index_cache_scale);
      candidate_gather = dcp_decode_context_->launch_indexer_candidate_gather(
          candidates.scores, candidates.global_slots);
      candidate_topk = candidates.global_slots.size(1);
    } else {
      auto index_out = indexer_->cp_post(index_pre,
                                         k_gathered,
                                         index_cache,
                                         attn_metadata,
                                         cp_ctx.gathered_slot_mapping,
                                         cp_ctx,
                                         index_cache_scale);
      topk_state.emplace(std::get<0>(index_out), std::get<1>(index_out));
    }
  }
  finish_cp_k_gather(mla_inputs, mla_handle);

  torch::Tensor mla_slot_mapping = cp_ctx.gathered_slot_mapping;
  if (use_dcp) {
    CHECK(cp_ctx.local_dcp_gathered_slot_mapping.defined())
        << "cache-sharded CP requires localized MLA slots";
    mla_slot_mapping = cp_ctx.local_dcp_gathered_slot_mapping;
  }
  update_mla_k_cache(mla_inputs.k_input,
                     attn_metadata,
                     kv_cache,
                     k_cache_scale,
                     is_prefill_or_chunked_prefill,
                     mla_slot_mapping);

  if (candidate_gather.has_value()) {
    DsaTopkState global_topk =
        dcp_decode_context_->finish_indexer_candidate_merge(
            std::move(candidate_gather.value()),
            candidate_topk,
            attn_metadata.slot_mapping);
    topk_state.emplace(
        v32_cp::reorder_to_local_shard(global_topk.block_tables(), cp_ctx),
        v32_cp::reorder_to_local_shard(global_topk.context_lens(), cp_ctx));
  }

  if (attn_metadata.is_dummy) {
    topk_state.reset();
  } else {
    if (reused_topk != nullptr) {
      topk_state = *reused_topk;
    }
    CHECK(topk_state.has_value())
        << "DSA context-parallel attention requires top-k state.";
  }
  if (use_dcp && topk_state.has_value() && attn_metadata.is_chunked_prefill) {
    local_topk = resolve_dcp_local_topk(topk_state.value(), topk_transfer);
  }
  if (topk_transfer != nullptr) {
    topk_transfer->complete(topk_state, local_topk);
  }

  AttentionMetadata kernel_metadata =
      build_mla_attention_metadata(attn_metadata, topk_state);
  kernel_metadata.q_cu_seq_lens = cp_ctx.local_attn_metadata.q_cu_seq_lens;
  kernel_metadata.max_query_len = cp_ctx.local_attn_metadata.max_query_len;

  torch::Tensor attn_output_local;
  if (use_dcp && attn_metadata.is_chunked_prefill) {
    torch::Tensor packed_q =
        v32_cp::all_gather_across_ranks(mla_inputs.q_input, cp_ctx);
    DsaTopkState gathered_topk(
        v32_cp::all_gather_across_ranks(topk_state->block_tables(), cp_ctx),
        v32_cp::all_gather_across_ranks(topk_state->context_lens(), cp_ctx));
    DsaTopkState gathered_local_topk =
        dcp_decode_context_->localize_topk(gathered_topk);
    auto query_gather = launch_dcp_query(packed_q);
    // Query rows, top-k rows and merge padding slots share the CP rank order.
    torch::Tensor packed_output =
        run_dcp_chunked_prefill_attention(packed_q,
                                          gathered_local_topk,
                                          query_gather,
                                          kv_cache,
                                          attn_metadata,
                                          cp_ctx.gathered_slot_mapping)
            .view({cp_ctx.total_tokens, tp_heads().attn * kv_lora_rank_});
    attn_output_local = v32_cp::slice_local_packed(packed_output, cp_ctx);
  } else if (use_dcp) {
    torch::Tensor k_cache = kv_cache.get_k_cache();
    auto [attention_cache, compact_metadata] =
        build_prefill_cache(mla_inputs.k_input,
                            k_cache,
                            cp_ctx.sorted_gathered_slot_mapping_int64,
                            cp_ctx.sorted_gathered_slot_rows,
                            kernel_metadata);
    std::tie(attn_output_local, std::ignore) = attn_(compact_metadata,
                                                     mla_inputs.q_input,
                                                     mla_inputs.k_input,
                                                     mla_inputs.v_input,
                                                     attention_cache);
  } else {
    std::tie(attn_output_local, std::ignore) = attn_(kernel_metadata,
                                                     mla_inputs.q_input,
                                                     mla_inputs.k_input,
                                                     mla_inputs.v_input,
                                                     kv_cache);
  }
  torch::Tensor output = project_output(attn_output_local, tp_heads());
  output = parallel_state::reduce(output, tp_group_);
  return output;
}

DeepseekV2AttentionImpl::MlaInputs DeepseekV2AttentionImpl::build_cp_mla_inputs(
    const torch::Tensor& hidden_states,
    const torch::Tensor& positions,
    const QueryPrep& query_prep,
    const v32_cp::DeepseekV32CPContext& cp_ctx) {
  MlaInputs out;
  out.q_input = torch::empty({hidden_states.size(0),
                              tp_heads().attn,
                              kv_lora_rank_ + qk_rope_head_dim_},
                             hidden_states.options());
  torch::Tensor latent_cache = kv_a_proj_with_mqa_(hidden_states);
  fill_q_input(out.q_input,
               query_prep.q,
               positions,
               cp_ctx.local_attn_metadata,
               /*use_prompt_rope=*/false);
  decode_kv_pre_base(latent_cache,
                     positions,
                     cp_ctx.local_attn_metadata,
                     /*use_prompt_rope=*/false);
  out.k_input = latent_cache;
  out.q_input = out.q_input.view({out.q_input.size(0), -1});
  out.k_input = out.k_input.view({out.k_input.size(0), -1});
  return out;
}

void DeepseekV2AttentionImpl::finish_cp_k_gather(
    MlaInputs& mla_inputs,
    const v32_cp::PaddedGatherHandle& k_handle) const {
  mla_inputs.k_input = parallel_state::finish_gather(k_handle);
  mla_inputs.v_input = mla_inputs.k_input.slice(-1, 0, kv_lora_rank_);
}

}  // namespace layer
}  // namespace xllm
