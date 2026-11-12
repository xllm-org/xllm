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

// DFlash2 draft selection: walks the candidate tree step by step, argmaxing
// each row over the predecessor's edge distribution with Gumbel noise.
// Model-specific knowledge lives here; the sampling directory holds only
// the model-agnostic gumbel primitives it consumes.

#include "runtime/dflash2_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/causal_lm.h"
#if defined(USE_NPU)
#include "core/kernels/npu/utils.h"
#include "core/platform/npu/device_capture_lock.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"
#endif
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/dflash_async_input_builder.h"
#include "core/framework/speculative/draft_proposal.h"
#include "core/framework/speculative/spec_input_builder.h"
#include "core/util/tensor_helper.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/attention_metadata_builder.h"
#include "runtime/acl_graph_bucket_policy.h"
#include "util/timer.h"

namespace xllm {

bool DFlash2WorkerImpl::init_model(const std::string& model_weights_path,
                                   int32_t random_seed,
                                   MasterStatus master_status) {
  const bool result = DFlashWorkerImpl::init_model(
      model_weights_path, random_seed, master_status);
  if (result && draft_impl_->get_status() == WorkerImpl::Status::LOADED &&
      is_probabilistic_draft_) {
    const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
    const ModelArgs& target_args = impl_->context_.get_model_args();
    CHECK_EQ(draft_args.draft_vocab_size() > 0 ? draft_args.draft_vocab_size()
                                               : draft_args.vocab_size(),
             target_args.vocab_size())
        << "DFlash2 probabilistic drafts require draft and target vocabulary "
           "sizes to match.";
  }
  return result;
}

// The selector takes only temperature — truncation, penalties, and grammar
// constraints belong to verification.
DraftProposal sample_dflash2_path(const DFlash2CandidateOutput& candidates,
                                  const SamplingParameters& sampling_params,
                                  torch::Tensor gumbel_noise,
                                  int64_t vocab_size,
                                  torch::Tensor token_ids_out) {
  CHECK_EQ(candidates.candidate_ids.dim(), 3);
  CHECK_EQ(candidates.edge_logits.dim(), 4);
  const int64_t batch_size = candidates.candidate_ids.size(0);
  const int64_t num_steps = candidates.candidate_ids.size(1);
  const int64_t top_k = candidates.candidate_ids.size(2);
  CHECK_EQ(candidates.edge_logits.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k, top_k}));
  const bool is_probabilistic_draft = gumbel_noise.defined();
  if (is_probabilistic_draft) {
    CHECK_EQ(gumbel_noise.sizes(),
             torch::IntArrayRef({batch_size, num_steps, top_k}));
  }
  const torch::Device device = candidates.edge_logits.device();
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  torch::Tensor edge_logits = candidates.edge_logits.to(torch::kFloat32);
  if (sampling_params.temperatures().defined()) {
    apply_temperatures(edge_logits, sampling_params.temperatures());
  }
  // log_softmax is monotone per row, so argmax(log_probs + noise) matches
  // argmax(logits + noise); normalize only when q is reported.
  torch::Tensor edge_log_probs =
      is_probabilistic_draft ? torch::log_softmax(edge_logits, /*dim=*/-1)
                             : edge_logits;

  torch::Tensor token_ids = std::move(token_ids_out);
  if (!token_ids.defined()) {
    token_ids = torch::empty({batch_size, num_steps},
                             candidates.candidate_ids.options());
  } else {
    CHECK_EQ(token_ids.sizes(), torch::IntArrayRef({batch_size, num_steps}));
  }
  torch::Tensor draft_logits;
  if (is_probabilistic_draft) {
    draft_logits = torch::full({batch_size, num_steps, vocab_size},
                               -std::numeric_limits<float>::infinity(),
                               float_options);
  }
  torch::Tensor previous_indices =
      torch::zeros({batch_size}, candidates.candidate_ids.options());

  torch::Tensor greedy_rows;
  torch::Tensor zero_log_probs;
  if (is_probabilistic_draft &&
      sampling_params.sampling_mode() == SamplingMode::MIXED) {
    greedy_rows = greedy_mask(sampling_params, device);
    zero_log_probs = torch::zeros({batch_size, 1}, float_options);
  }

  for (int64_t step = 0; step < num_steps; ++step) {
    torch::Tensor gather_indices = previous_indices.view({batch_size, 1, 1})
                                       .expand({batch_size, 1, top_k});
    torch::Tensor row_log_probs =
        edge_log_probs.select(/*dim=*/1, /*index=*/step)
            .gather(/*dim=*/1, gather_indices)
            .squeeze(/*dim=*/1);
    // gumbel_argmax consumes its noise buffer in place; each step's slice
    // has no later reader, so no clone.
    torch::Tensor sampled_indices =
        is_probabilistic_draft
            ? gumbel_argmax(row_log_probs,
                            gumbel_noise.select(/*dim=*/1, /*index=*/step))
            : greedy_sample(row_log_probs);

    torch::Tensor step_candidates =
        candidates.candidate_ids.select(/*dim=*/1, /*index=*/step);
    torch::Tensor sampled_tokens =
        index_select_2d(step_candidates, /*dim=*/1, sampled_indices);
    if (is_probabilistic_draft) {
      torch::Tensor step_log_probs = row_log_probs;
      if (sampling_params.sampling_mode() == SamplingMode::MIXED) {
        // Greedy rows embed a log-space one-hot (q = 1).
        torch::Tensor one_hot =
            torch::full({batch_size, top_k},
                        -std::numeric_limits<float>::infinity(),
                        float_options);
        one_hot.scatter_(
            /*dim=*/1, sampled_indices.view({batch_size, 1}), zero_log_probs);
        step_log_probs = torch::where(greedy_rows, one_hot, step_log_probs);
      }
      draft_logits.select(/*dim=*/1, step)
          .scatter_(/*dim=*/1, step_candidates, step_log_probs);
    }
    token_ids.select(/*dim=*/1, step).copy_(sampled_tokens);
    previous_indices = sampled_indices;
  }

  return DraftProposal(std::move(token_ids), std::move(draft_logits));
}

torch::Tensor broadcast_gumbel_noise(int64_t batch_size,
                                     const SamplingParameters& sampling,
                                     int64_t num_steps,
                                     int64_t num_classes,
                                     const torch::Device& device,
                                     ProcessGroup* sampling_group) {
  if (sampling.sampling_mode() == SamplingMode::GREEDY) {
    return torch::Tensor();
  }
  CHECK_GT(batch_size, 0);
  torch::Tensor noise =
      gumbel_noise(sampling, {batch_size, num_steps, num_classes}, device);
  sync_sampling_tensor(noise, sampling_group);
  return noise;
}
namespace {

int64_t next_pow2_block_table_cols(int64_t cols) {
  CHECK_GE(cols, 0);
  int64_t padded = 16;
  while (padded < cols) {
    padded *= 2;
  }
  CHECK_LT(padded, 1LL << 28)
      << "DFlash2 block-table columns exceed graph capacity: " << cols;
  return padded;
}

}  // namespace

struct DFlash2WorkerImpl::DraftGraphResources {
  std::unique_ptr<Stream> capture_stream;
#if defined(USE_NPU)
  c10_npu::MempoolId_t pool = c10_npu::graph_pool_handle();
#endif
};

struct DFlash2WorkerImpl::DraftGraph {
  LlmForwardInput input;
  LlmForwardInput query;
  torch::Tensor accepted_tokens;
  torch::Tensor embeddings;
  torch::Tensor base_positions;
  torch::Tensor noise;
  DraftProposal output;
#if defined(USE_NPU)
  c10_npu::NPUGraph graph;
#endif
};

DFlash2WorkerImpl::DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                                     const torch::Device& device,
                                     const runtime::Options& options)
    : DFlashWorkerImpl(parallel_args, device, options) {
  prelaunch_graph_enabled_ = ExecutionConfig::get_instance().enable_graph();
}

DFlash2WorkerImpl::~DFlash2WorkerImpl() { discard_draft_prelaunch(); }

DraftProposal DFlash2WorkerImpl::run_decode_draft(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input) {
  if (pending_draft_.has_value()) {
    if (pending_draft_matches(input)) {
      if (pending_draft_->next_template.has_value()) {
        prepared_template_ = std::move(pending_draft_->next_template);
      }
      if (pending_draft_->target_input.has_value()) {
        validate_input = std::move(*pending_draft_->target_input);
        validate_input.input_params.meta.batch_id =
            input.input_params.meta.batch_id;
        auto& staged = validate_input.input_params.embedding;
        const auto& current = input.input_params.embedding;
        if (staged.linear_state_ids != current.linear_state_ids ||
            staged.linear_state_read_ids != current.linear_state_read_ids) {
          c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
          CHECK(prepare_stream_->wait_event(
              validate_input.runtime.metadata_ready_event))
              << "failed to wait speculative metadata ready event";
          staged.linear_state_ids = current.linear_state_ids;
          staged.linear_state_read_ids = current.linear_state_read_ids;
          staged.linear_state_indices = torch::Tensor();
          staged.linear_state_read_indices = torch::Tensor();
          materialize_linear_state_indices(staged, device_.unwrap());
          auto& metadata = *validate_input.input_params.attn_metadata;
          metadata.linear_state_indices = staged.linear_state_indices;
          metadata.linear_state_read_indices = staged.linear_state_read_indices;
          validate_input.runtime.metadata_ready_event =
              prepare_stream_->record_event_or_sync();
        }
      } else {
        prepare_validate_inputs(input, validate_input);
      }
      DraftProposal block = std::move(pending_draft_->proposal);
      // Keep producer tensors alive through this iteration's target verify.
      auto retained = std::make_shared<LlmForwardInput>(
          std::move(pending_draft_->identity));
      retained->input_params.embedding.input_embedding =
          std::move(pending_draft_->retained_embeddings);
      retained->runtime.metadata_ready_event = std::move(pending_draft_->ready);
      block.keep_alive().emplace_back(std::move(retained));
      pending_draft_.reset();
      return block;
    }
    discard_draft_prelaunch();
  }
  LlmForwardInput query_input;
  prepare_query_inputs(input, query_input);
  // Proposal and target verification have distinct state and attention modes.
  // Warmup forwards keep ordinary chunk metadata.
  if (draft_impl_->context_.dflash_xfia_enabled() &&
      !input.input_params.meta.is_graph_warmup) {
    query_input.input_params.is_dflash_proposal = true;
  }

  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_speculative_tokens, 0);
  const torch::Tensor anchors = anchor_token_ids(input);

  const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
  const int64_t selector_top_k = draft_args.dflash2_selector_top_k();
  c10::StreamGuard noise_guard = compute_stream_->set_stream_guard();
  SamplingParameters draft_sampling_params = input.sampling_params;
  prepare_draft_sampling(is_probabilistic_draft_, draft_sampling_params);
  draft_sampling_params =
      draft_sampling_params.to(draft_impl_->device(), torch::kFloat32);
  torch::Tensor gumbel_noise = broadcast_gumbel_noise(batch_size,
                                                      draft_sampling_params,
                                                      num_speculative_tokens,
                                                      selector_top_k,
                                                      draft_impl_->device(),
                                                      sampling_process_group_);

  DraftProposal block = execute_draft_query(draft_sampling_params,
                                            std::move(query_input),
                                            anchors,
                                            gumbel_noise,
                                            false);
  prepare_validate_inputs(input, validate_input);
  return block;
}

DraftProposal DFlash2WorkerImpl::execute_draft_query(
    const SamplingParameters& sampling_params,
    LlmForwardInput query_input,
    const torch::Tensor& anchor_token_ids,
    const torch::Tensor& gumbel_noise,
    bool prepared,
    Stream* execution_stream) {
  Stream& stream =
      execution_stream == nullptr ? *compute_stream_ : *execution_stream;
  Timer timer;
  const int32_t batch_size = query_input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  query_input.skip_sampling_for_logits_only = true;
  query_input.return_selected_hidden = true;
  LlmForwardInput processed_input;
  if (prepared) {
    processed_input = std::move(query_input);
  } else {
    draft_impl_->prepare_work_before_execute_on_stream(
        query_input,
        processed_input,
        *prepare_stream_,
        /*record_ready_event=*/prepare_stream_.get() != compute_stream_.get());
  }
  draft_impl_->set_hierarchy_layer_synchronizer(processed_input.input_params);
  std::optional<ForwardOutput> draft_output =
      draft_impl_->execute_no_sync_on_stream(
          processed_input, stream, ForwardCompletion::CALLER_MANAGED);
  CHECK(draft_output.has_value());
  CHECK(draft_output->logits.defined());
  CHECK(draft_output->selected_hidden.defined());
  const int64_t num_rows = draft_output->logits.size(0);
  CHECK_EQ(num_rows, static_cast<int64_t>(batch_size) * num_speculative_tokens);
  torch::Tensor unary_logits = draft_output->logits.view(
      {batch_size, num_speculative_tokens, draft_output->logits.size(-1)});
  torch::Tensor hidden_states = draft_output->selected_hidden.view(
      {batch_size,
       num_speculative_tokens,
       draft_output->selected_hidden.size(-1)});

  DraftProposal draft_proposal;
  {
    c10::StreamGuard stream_guard = stream.set_stream_guard();
    DFlash2CandidateOutput candidates = draft_impl_->dflash2_candidates(
        hidden_states, unary_logits, anchor_token_ids);
    draft_proposal = sample_dflash2_path(
        candidates, sampling_params, gumbel_noise, unary_logits.size(-1));
  }
  draft_proposal.keep_alive() = take_retained_inputs(*draft_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft_proposal;
}

bool DFlash2WorkerImpl::can_use_prelaunch_graph(
    const LlmForwardInput& input) const {
  if (!draft_impl_->context_.dflash_xfia_enabled() ||
      !prelaunch_graph_enabled_ || input.input_params.meta.num_sequences <= 0) {
    return false;
  }
  // Prelaunch is restricted to DP=1, so the shared global capacity is also
  // the request-count limit for this independently captured draft graph.
  const uint32_t graph_capacity = npu::acl_graph_max_global_batch_size(
      options_.max_seqs_per_batch(),
      std::max<int32_t>(
          1,
          ExecutionConfig::get_instance().acl_graph_decode_batch_size_limit()),
      parallel_args_.dp_size());
  return static_cast<uint32_t>(input.input_params.meta.num_sequences) <=
         graph_capacity;
}

bool DFlash2WorkerImpl::can_prepare_without_compute_stream_wait(
    const LlmForwardInput& input) const {
  // Reuse the owner's actual enablement, including A3/Python, overlap,
  // DP/CP and PD role restrictions. The outer worker stages
  // fresh metadata; it owns neither the leaf graph's persistent buffers nor
  // its KV cache. Do not extend this exemption to the leaf workers.
  return can_use_prelaunch_graph(input) &&
         input.input_params.meta.batch_forward_type.is_decode() &&
         can_stage_prelaunch_input(input);
}

bool DFlash2WorkerImpl::can_stage_prelaunch_input(
    const LlmForwardInput& input) const {
  if (!draft_impl_->context_.dflash_xfia_enabled() ||
      input.sampling_params.seeds().defined() ||
      input.input_params.meta.is_graph_warmup ||
      !input.input_params.multi_block_tables.empty() ||
      input.input_params.meta.requires_host_restore ||
      !input.input_params.block_copy.swap_blocks.empty() ||
      (input.input_params.block_copy.src_block_indices.defined() &&
       input.input_params.block_copy.src_block_indices.numel() > 0) ||
      !input.transfer_kv_infos.empty() || !input.json_object_states.empty() ||
      !input.json_object_state_snapshots.empty() ||
      input.input_params.embedding.embedding_ids.empty() ||
      input.input_params.embedding.request_ids.size() !=
          input.input_params.embedding.embedding_ids.size()) {
    return false;
  }
  return true;
}

bool DFlash2WorkerImpl::prepare_draft_prelaunch(const LlmForwardInput& input) {
  if (!can_stage_prelaunch_input(input)) {
    return false;
  }
  // A matching pending draft can supply the following draft's template too.
  // Its full lookahead block mapping was checked before consumption.
  if (prepared_template_.has_value()) {
    return true;
  }
  const int32_t width = options_.num_speculative_tokens() + 1;
  const auto rows = specBuilder::make_decode_row_context(input);
  if (rows.model_managed_multiblock) {
    return false;
  }
  // Capacity is checked using only known host geometry, before validation.
  if (!specBuilder::has_decode_lookahead_capacity(
          rows,
          /*num_tokens=*/2 * width,
          options_.block_size(),
          draft_impl_->context_.get_model_args().max_position_embeddings())) {
    return false;
  }
  // Noise is independent of the accepted prefix. Schedule any TP consensus
  // while preparing the template, before rejection sampling is submitted.
  DraftTemplate prepared;
  c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
  prepared.sampling = input.sampling_params;
  prepare_draft_sampling(is_probabilistic_draft_, prepared.sampling);
  prepared.sampling = prepared.sampling.to(device_.unwrap(), torch::kFloat32);
  prepared.noise = broadcast_gumbel_noise(
      rows.num_sequences,
      prepared.sampling,
      width - 1,
      draft_impl_->context_.get_model_args().dflash2_selector_top_k(),
      draft_impl_->device(),
      sampling_process_group_);
  LlmForwardInput future = input.clone();
  future.positions_host = input.positions_host.clone();
  future.positions_host.add_(width);
  for (int32_t& length : future.input_params.attention.host.kv_seq_lens) {
    length += width;
  }
  LlmForwardInput query;
  prepare_query_inputs(future, query);
  query.input_params.is_dflash_proposal = true;
  query.skip_sampling_for_logits_only = true;
  query.return_selected_hidden = true;
  // The query builder has allocated fresh metadata buffers on prepare_stream_.
  // It reads no in-flight draft outputs and performs no KV swaps/restores.
  // Only its explicit input-ready event is needed; waiting for the entire
  // compute stream here would serialize staging behind the pending draft.
  draft_impl_->prepare_work_before_execute_on_stream(
      query,
      prepared.query,
      *prepare_stream_,
      /*record_ready_event=*/true,
      /*restore_linear_state=*/true,
      /*wait_for_compute_stream=*/false);
  prepared.base_positions =
      prepared.query.positions.view({rows.num_sequences, width}).select(1, 0) -
      width;
  if (can_use_prelaunch_graph(input)) {
    const torch::Tensor table =
        prepared.query.input_params.attention.device.block_tables;
    const int64_t cols = next_pow2_block_table_cols(table.size(1));
    torch::Tensor padded = torch::zeros({table.size(0), cols}, table.options());
    padded.slice(1, 0, table.size(1)).copy_(table);
    prepared.query.input_params.attention.device.block_tables =
        std::move(padded);
  }
  // Covers the final staging operations as well as the earlier template.
  prepared.query.runtime.metadata_ready_event = prepare_stream_->record_event();
  prepared_template_ = std::move(prepared);
  return true;
}

void DFlash2WorkerImpl::launch_draft_prelaunch(
    const LlmForwardInput& input,
    const SampleOutput& output,
    const torch::Tensor& accepted_host) {
  CHECK(prepared_template_.has_value());
  CHECK(!pending_draft_.has_value());
  c10::StreamGuard guard = compute_stream_->set_stream_guard();
  DraftTemplate prepared = std::move(*prepared_template_);
  prepared_template_.reset();
  LlmForwardInput& query = prepared.query;
  CHECK(compute_stream_->wait_event(query.runtime.metadata_ready_event));
  const torch::Tensor& noise = prepared.noise;
  const torch::Tensor& base_positions = prepared.base_positions;
  auto staged_query = std::make_shared<LlmForwardInput>(query.clone());
  PendingDraft pending;
  pending.accepted_host = accepted_host;
  pending.retained_embeddings = output.embeddings;
  pending.identity = input.clone();
  pending.identity.sampling_params = std::move(prepared.sampling);
  if (can_use_prelaunch_graph(input)) {
    pending.proposal = replay_prelaunch_graph(
        pending.identity, std::move(query), output, noise, base_positions);
  } else {
    pending.proposal = execute_prelaunch_body(pending.identity,
                                              query,
                                              output.next_tokens,
                                              output.embeddings,
                                              base_positions,
                                              noise,
                                              *compute_stream_);
  }
  pending.proposal.keep_alive().emplace_back(std::move(staged_query));
  pending.retained.emplace_back(std::make_shared<std::vector<torch::Tensor>>(
      std::initializer_list<torch::Tensor>{
          noise, base_positions, output.next_tokens, output.embeddings}));
  pending.ready = compute_stream_->record_event();
  CHECK(pending.ready != nullptr);
  pending_draft_ = std::move(pending);
}

DraftProposal DFlash2WorkerImpl::execute_prelaunch_body(
    const LlmForwardInput& input,
    LlmForwardInput& query,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& embeddings,
    const torch::Tensor& base_positions,
    const torch::Tensor& noise,
    Stream& stream) {
  c10::StreamGuard stream_guard = stream.set_stream_guard();
  const int64_t batch = accepted_tokens.size(0);
  const int64_t width = accepted_tokens.size(1);
  dflash_async::NextDraftInputs next = dflash_async::prepare_next_draft(
      accepted_tokens,
      base_positions,
      query.input_params.attention.device.block_tables,
      mask_token_id_,
      options_.block_size());
  ModelOutput written =
      draft_impl_->write_context_kv(embeddings.reshape({batch * width, -1}),
                                    next.context_positions,
                                    next.context_slots,
                                    query.input_params);
  CHECK(written.hidden_states.defined());
  query.token_ids.copy_(next.query_tokens);
  query.positions.copy_(next.query_positions);
  query.input_params.attention.device.new_cache_slots.copy_(next.query_slots);
  query.input_params.attention.device.kv_seq_lens.copy_(next.kv_lengths);
  query.input_params.attn_metadata.reset();
  query.runtime.metadata_ready_event.reset();
  query.runtime.device_tensors_ready = true;
  return execute_draft_query(input.sampling_params,
                             query.clone(),
                             next.anchor_tokens,
                             noise,
                             /*prepared=*/true,
                             &stream);
}

DraftProposal DFlash2WorkerImpl::replay_prelaunch_graph(
    const LlmForwardInput& input,
    LlmForwardInput query,
    const SampleOutput& output,
    const torch::Tensor& noise,
    const torch::Tensor& base_positions) {
#if defined(USE_NPU)
  torch::InferenceMode inference_guard;
  const int64_t batch = output.next_tokens.size(0);
  const torch::Tensor source_table =
      query.input_params.attention.device.block_tables;
  const int64_t source_cols = source_table.size(1);
  const int64_t table_cols = next_pow2_block_table_cols(source_cols);
  const uint64_t mode =
      (static_cast<uint64_t>(input.sampling_params.sampling_mode()) << 1) |
      input.sampling_params.temperatures().defined();
  const uint64_t key = (static_cast<uint64_t>(batch) << 32) |
                       (static_cast<uint64_t>(table_cols) << 4) | mode;
  auto iterator = draft_graphs_.find(key);
  const bool first_capture = iterator == draft_graphs_.end();
  if (first_capture) {
    if (draft_graph_resources_ == nullptr) {
      draft_graph_resources_ = std::make_unique<DraftGraphResources>();
      draft_graph_resources_->capture_stream = device_.get_stream_from_pool();
    }
    auto entry = std::make_unique<DraftGraph>();
    entry->input = input.clone();
    if (input.sampling_params.do_sample().defined()) {
      entry->input.sampling_params.do_sample() =
          input.sampling_params.do_sample().clone();
    }
    if (input.sampling_params.temperatures().defined()) {
      entry->input.sampling_params.temperatures() =
          input.sampling_params.temperatures().clone();
    }
    entry->query = query.clone();
    entry->query.input_params.attention.device.block_tables =
        torch::zeros({batch, table_cols}, source_table.options());
    entry->accepted_tokens = torch::empty_like(output.next_tokens);
    entry->embeddings = torch::empty_like(output.embeddings);
    entry->base_positions = torch::empty({batch}, query.positions.options());
    entry->noise = noise.defined() ? torch::empty_like(noise) : torch::Tensor();
    iterator = draft_graphs_.emplace(key, std::move(entry)).first;
  }
  DraftGraph& entry = *iterator->second;
  entry.accepted_tokens.copy_(output.next_tokens, /*non_blocking=*/true);
  entry.embeddings.copy_(output.embeddings, /*non_blocking=*/true);
  entry.base_positions.copy_(base_positions, /*non_blocking=*/true);
  if (noise.defined()) {
    entry.noise.copy_(noise, /*non_blocking=*/true);
  }
  entry.query.input_params.attention.device.block_tables.copy_(
      source_table, /*non_blocking=*/true);
  if (entry.input.sampling_params.do_sample().defined()) {
    entry.input.sampling_params.do_sample().copy_(
        input.sampling_params.require_do_sample());
  }
  if (input.sampling_params.temperatures().defined()) {
    entry.input.sampling_params.temperatures().copy_(
        input.sampling_params.temperatures());
  }
  if (first_capture) {
    auto& capture_lock =
        npu::DeviceCaptureLock::get_instance().get_lock(device_.index());
    std::lock_guard<std::mutex> lock(capture_lock);
    Stream& capture_stream = *draft_graph_resources_->capture_stream;
    capture_stream.wait_stream(*compute_stream_);
    c10::StreamGuard capture_guard = capture_stream.set_stream_guard();
    // Only the first use of a bucket warms/captures. Context writes use the
    // same positions, but sampling consumes the noise buffer in place.
    for (int32_t step = 0; step < 2; ++step) {
      entry.output = execute_prelaunch_body(entry.input,
                                            entry.query,
                                            entry.accepted_tokens,
                                            entry.embeddings,
                                            entry.base_positions,
                                            entry.noise,
                                            capture_stream);
    }
    CHECK_EQ(capture_stream.synchronize(), 0);
    // Keep verifier outputs outside the shared pool: an earlier graph's
    // temporary storage can alias a later graph's captured output. Replays
    // are serialized, but target validation retains the proposal tensors.
    torch::Tensor token_ids = torch::empty_like(entry.output.token_ids());
    std::optional<torch::Tensor> logits;
    if (entry.output.logits().has_value()) {
      logits = torch::empty_like(*entry.output.logits());
    }
    // Share only the draft pool and capture stream. The target has its own
    // pool, and every bucket's input buffers were allocated before capture.
    entry.graph.capture_begin(draft_graph_resources_->pool,
                              ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    entry.output = execute_prelaunch_body(entry.input,
                                          entry.query,
                                          entry.accepted_tokens,
                                          entry.embeddings,
                                          entry.base_positions,
                                          entry.noise,
                                          capture_stream);
    token_ids.copy_(entry.output.token_ids());
    if (logits.has_value()) {
      logits->copy_(*entry.output.logits());
    }
    auto retained = std::move(entry.output.keep_alive());
    entry.output = DraftProposal(std::move(token_ids), std::move(logits));
    entry.output.keep_alive() = std::move(retained);
    entry.graph.capture_end();
  }
  if (first_capture && noise.defined()) {
    // Causal warmups can select different rows, so even two in-place sampling
    // passes do not restore the original noise for the first replay.
    compute_stream_->wait_stream(*draft_graph_resources_->capture_stream);
    entry.noise.copy_(noise, /*non_blocking=*/true);
  }
  entry.graph.replay();
  return entry.output;
#else
  LOG(FATAL) << "DFlash2 prelaunch graph requires NPU";
  return {};
#endif
}

bool DFlash2WorkerImpl::pending_draft_matches(
    const LlmForwardInput& input) const {
  const PendingDraft& pending = *pending_draft_;
  const auto& old = pending.identity.input_params;
  const auto& current = input.input_params;
  if (!can_stage_prelaunch_input(input) ||
      old.embedding.embedding_ids != current.embedding.embedding_ids ||
      old.embedding.request_ids != current.embedding.request_ids ||
      old.parallel.dp_global_batch_generations !=
          current.parallel.dp_global_batch_generations ||
      old.parallel.dp_global_token_nums !=
          current.parallel.dp_global_token_nums ||
      pending.identity.sampling_params.sampling_mode() !=
          input.sampling_params.sampling_mode()) {
    return false;
  }
  const auto previous = specBuilder::make_decode_row_context(pending.identity);
  const auto next = specBuilder::make_decode_row_context(input);
  const int64_t width = pending.accepted_host.size(1);
  const int64_t* tokens = pending.accepted_host.const_data_ptr<int64_t>();
  for (int32_t seq = 0; seq < next.num_sequences; ++seq) {
    int64_t count = 0;
    while (count < width && tokens[seq * width + count] >= 0) {
      ++count;
    }
    if (count == 0 || next.positions[seq] != previous.positions[seq] + count ||
        next.token_ids[seq] != tokens[seq * width + count - 1]) {
      return false;
    }
    const int64_t lookahead =
        pending.next_template.has_value() ? 2 * width : width;
    const int64_t pages =
        (next.positions[seq] + lookahead + options_.block_size() - 1) /
        options_.block_size();
    if (pages > next.block_table_stride ||
        pages > previous.block_table_stride) {
      return false;
    }
    for (int64_t page = 0; page < pages; ++page) {
      if (next.block_tables[seq * next.block_table_stride + page] !=
          previous.block_tables[seq * previous.block_table_stride + page]) {
        return false;
      }
    }
  }
  return true;
}

void DFlash2WorkerImpl::finish_draft_prelaunch(const LlmForwardInput& input) {
  if (!pending_draft_.has_value()) {
    return;
  }
  // Keep next-target preparation overlapped with the running draft. Only
  // after that preparation, fence all context/query KV writes before this
  // rank returns accepted tokens. The scheduler may end any row (EOS, length,
  // stop sequence or cancellation) and immediately reuse its blocks. Retaining
  // tensors or waiting when a stale draft is consumed does not pin those
  // blocks.
  prepare_next_target(input);
  CHECK(pending_draft_->ready->synchronize())
      << "Failed to retire DFlash2 prelaunch KV writes before publishing "
         "tokens";
}

void DFlash2WorkerImpl::prepare_next_target(const LlmForwardInput& input) {
  if (!pending_draft_.has_value() || !can_use_prelaunch_graph(input) ||
      input.sampling_params.unique_token_ids().defined() ||
      input.sampling_params.filter_mask().defined() ||
      input.sampling_params.filter_bitmask().defined()) {
    return;
  }
  PendingDraft& pending = *pending_draft_;
  CHECK(!pending.target_input.has_value());
  // run_validate has waited only for rejection sampling and its pinned D2H.
  // The prelaunched draft is still queued/running on compute_stream_. Build
  // the next verify metadata on prepare_stream_ using the exact host prefix.
  const auto rows = specBuilder::make_decode_row_context(input);
  const int64_t width = pending.accepted_host.size(1);
  const int64_t* accepted = pending.accepted_host.const_data_ptr<int64_t>();
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> kv_lengths;
  tokens.reserve(rows.num_sequences);
  positions.reserve(rows.num_sequences);
  kv_lengths.reserve(rows.num_sequences);
  for (int32_t seq = 0; seq < rows.num_sequences; ++seq) {
    int64_t count = 0;
    while (count < width && accepted[seq * width + count] >= 0) {
      ++count;
    }
    CHECK_GT(count, 0);
    tokens.emplace_back(
        static_cast<int32_t>(accepted[seq * width + count - 1]));
    const int32_t position = rows.positions[seq] + static_cast<int32_t>(count);
    positions.emplace_back(position);
    kv_lengths.emplace_back(position + 1);
  }
  LlmForwardInput future = input.clone();
  future.token_ids_host = specBuilder::make_cpu_int_tensor(tokens);
  future.positions_host = specBuilder::make_cpu_int_tensor(positions);
  future.input_params.attention.host.kv_seq_lens = std::move(kv_lengths);
  future.runtime.device_tensors_ready = false;
  future.runtime.input_host_buffer_has_layout = false;
  future.runtime.metadata_ready_event.reset();
  future.input_params.attn_metadata.reset();
  pending.target_input.emplace();
  prepare_validate_inputs(future, *pending.target_input);
  // Finish the target worker's input preparation and the C++ -> Python
  // attention metadata here too. In particular, materializing the recurrent
  // validity mask on compute_stream_ would synchronize it behind the draft.
  LlmForwardInput processed_target;
  impl_->prepare_work_before_execute_on_stream(
      *pending.target_input,
      processed_target,
      *prepare_stream_,
      /*record_ready_event=*/false,
      /*restore_linear_state=*/false,
      /*wait_for_compute_stream=*/false);
  {
    c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
    processed_target.input_params.attn_metadata =
        std::make_shared<layer::AttentionMetadata>(
            layer::AttentionMetadataBuilder::build(
                ModelParamsView(processed_target.input_params),
                impl_->context_.get_model_args().enable_mla(),
                std::nullopt,
                device_.unwrap()));
    processed_target.runtime.metadata_ready_event =
        prepare_stream_->record_event();
    CHECK(processed_target.runtime.metadata_ready_event != nullptr);
  }
  pending.target_input = std::move(processed_target);
  if (prepare_draft_prelaunch(future)) {
    pending.next_template = std::move(prepared_template_);
    prepared_template_.reset();
  }
}

void DFlash2WorkerImpl::discard_draft_prelaunch() {
  if (pending_draft_.has_value()) {
    CHECK(pending_draft_->ready->synchronize())
        << "Failed to drain stale DFlash2 prelaunch";
    if (pending_draft_->target_input.has_value()) {
      CHECK(pending_draft_->target_input->runtime.metadata_ready_event
                ->synchronize())
          << "Failed to drain stale DFlash2 target preparation";
    }
    if (pending_draft_->next_template.has_value()) {
      CHECK(pending_draft_->next_template->query.runtime.metadata_ready_event
                ->synchronize())
          << "Failed to drain stale DFlash2 draft template";
    }
    pending_draft_.reset();
  }
  prepared_template_.reset();
}

}  // namespace xllm
