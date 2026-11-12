/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <unordered_map>

#include "core/framework/speculative/draft_proposal.h"
#include "runtime/dflash_worker_impl.h"

namespace xllm {

class ProcessGroup;

// Gumbel-noise walk of the DFlash2 candidate tree; undefined noise (greedy
// draft) degrades to the raw argmax. TP noise consensus is the caller's.
DraftProposal sample_dflash2_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    torch::Tensor gumbel_noise,
    int64_t vocab_size,
    torch::Tensor token_ids_out = torch::Tensor());
// Selector noise for the DFlash2 candidate walk; only a deterministic draft
// (SamplingMode::GREEDY) returns an undefined tensor — callers key
// is_probabilistic_draft off that. A MIXED batch still draws noise (the
// 0.0 greedy-row sentinel keeps its greedy rows). With a non-null
// sampling_group the noise is broadcast from rank 0 so every rank walks the
// same candidates. batch_size is the row count the noise must match — the
// caller's own geometry, not derived from the sampling params.
torch::Tensor broadcast_gumbel_noise(int64_t batch_size,
                                     const SamplingParameters& sampling,
                                     int64_t num_steps,
                                     int64_t num_classes,
                                     const torch::Device& device,
                                     ProcessGroup* sampling_group);

class DFlash2WorkerImpl final : public DFlashWorkerImpl {
 public:
  DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                    const torch::Device& device,
                    const runtime::Options& options);

  ~DFlash2WorkerImpl() override;

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

 protected:
  bool can_prepare_without_compute_stream_wait(
      const LlmForwardInput& input) const override;
  DraftProposal run_decode_draft(const LlmForwardInput& input,
                                 LlmForwardInput& validate_input) override;

  // DFlash2 overlaps the next draft and the next verify metadata with this
  // step's target work. Both hooks own whatever they stage; discarding must
  // drain it before the scheduler reuses the blocks it wrote.
  bool prepare_draft_prelaunch(const LlmForwardInput& input) override;
  void launch_draft_prelaunch(const LlmForwardInput& input,
                              const SampleOutput& output,
                              const torch::Tensor& accepted_host) override;
  void discard_draft_prelaunch() override;
  void finish_draft_prelaunch(const LlmForwardInput& input) override;

 private:
  void prepare_next_target(const LlmForwardInput& input);
  bool can_stage_prelaunch_input(const LlmForwardInput& input) const;
  bool can_use_prelaunch_graph(const LlmForwardInput& input) const;
  DraftProposal execute_draft_query(const SamplingParameters& sampling_params,
                                    LlmForwardInput query_input,
                                    const torch::Tensor& anchor_token_ids,
                                    const torch::Tensor& gumbel_noise,
                                    bool prepared,
                                    Stream* execution_stream = nullptr);
  bool pending_draft_matches(const LlmForwardInput& input) const;
  DraftProposal execute_prelaunch_body(const LlmForwardInput& input,
                                       LlmForwardInput& query,
                                       const torch::Tensor& accepted_tokens,
                                       const torch::Tensor& embeddings,
                                       const torch::Tensor& base_positions,
                                       const torch::Tensor& noise,
                                       Stream& stream);
  // Bucketed ACL-graph replay of the prelaunch draft; first use captures.
  DraftProposal replay_prelaunch_graph(const LlmForwardInput& input,
                                       LlmForwardInput query,
                                       const SampleOutput& output,
                                       const torch::Tensor& noise,
                                       const torch::Tensor& base_positions);
  struct DraftGraph;
  struct DraftGraphResources;
  std::unique_ptr<DraftGraphResources> draft_graph_resources_;
  std::unordered_map<uint64_t, std::unique_ptr<DraftGraph>> draft_graphs_;
  bool prelaunch_graph_enabled_ = false;

  struct DraftTemplate {
    LlmForwardInput query;
    SamplingParameters sampling;
    torch::Tensor noise;
    torch::Tensor base_positions;
  };
  struct PendingDraft {
    LlmForwardInput identity;
    torch::Tensor accepted_host;
    torch::Tensor retained_embeddings;
    DraftProposal proposal;
    // Retained no-sync draft inputs; they must outlive the verify stream sync.
    std::vector<std::shared_ptr<const void>> retained;
    StreamEventPtr ready;
    std::optional<LlmForwardInput> target_input;
    std::optional<DraftTemplate> next_template;
  };
  std::optional<DraftTemplate> prepared_template_;
  std::optional<PendingDraft> pending_draft_;
};

}  // namespace xllm
