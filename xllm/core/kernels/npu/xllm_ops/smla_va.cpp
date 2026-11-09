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

#include <torch/torch.h>

#include <array>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {
constexpr int64_t kDim0 = 0;
constexpr int64_t kDim1 = 1;
constexpr int64_t kDim2 = 2;

void check_query_shape(const torch::Tensor& q, const std::string& layout) {
  TORCH_CHECK(layout == "TND" || layout == "BSND",
              "SMLA layout_q must be \"TND\" or \"BSND\", got ",
              layout);
  TORCH_CHECK(q.numel() > 0, "SMLA query must not be empty");
  TORCH_CHECK(q.dim() == (layout == "TND" ? 3 : 4),
              "SMLA query rank ",
              q.dim(),
              " does not match layout ",
              layout);
}

int64_t get_kv_head_num(const std::optional<torch::Tensor>& ori_kv,
                        const std::optional<torch::Tensor>& cmp_kv,
                        const std::string& layout_kv_str) {
  TORCH_CHECK(layout_kv_str == "TND" || layout_kv_str == "PA_BBND",
              "SMLA layout_kv must be \"TND\" or \"PA_BBND\", got ",
              layout_kv_str);
  TORCH_CHECK(ori_kv.has_value() || cmp_kv.has_value(),
              "SMLA needs at least one of ori_kv / cmp_kv");
  const int64_t expected_dim = layout_kv_str == "TND" ? 3 : 4;
  int64_t heads = -1;
  for (const auto* candidate : {&ori_kv, &cmp_kv}) {
    if (!candidate->has_value()) {
      continue;
    }
    const auto& kv = candidate->value();
    TORCH_CHECK(kv.dim() == expected_dim,
                "SMLA layout_kv ",
                layout_kv_str,
                " expects a rank-",
                expected_dim,
                " kv tensor, got rank ",
                kv.dim());
    for (const int64_t size : kv.sizes()) {
      TORCH_CHECK(
          size > 0, "SMLA KV dimensions must be positive, got ", kv.sizes());
    }
    const int64_t current_heads =
        layout_kv_str == "TND" ? kv.size(kDim1) : kv.size(kDim2);
    TORCH_CHECK(heads < 0 || heads == current_heads,
                "SMLA original and compressed KV head counts must match");
    heads = current_heads;
  }
  return heads;
}

std::tuple<torch::Tensor, torch::Tensor> make_smla_outputs(
    const torch::Tensor& q,
    const std::optional<torch::Tensor>& ori_kv,
    const std::optional<torch::Tensor>& cmp_kv,
    const std::string& layout_q_str,
    const std::string& layout_kv_str,
    bool return_softmax_lse) {
  check_query_shape(q, layout_q_str);
  int64_t kv_head_num = get_kv_head_num(ori_kv, cmp_kv, layout_kv_str);
  const int64_t query_head_num = q.size(layout_q_str == "TND" ? kDim1 : kDim2);
  TORCH_CHECK(query_head_num % kv_head_num == 0,
              "SMLA query heads must be divisible by KV heads: ",
              query_head_num,
              " % ",
              kv_head_num);
  torch::Tensor atten_out = torch::empty_like(q);
  torch::Tensor softmax_lse;
  if (!return_softmax_lse) {
    softmax_lse = torch::empty({0}, q.options().dtype(torch::kFloat32));
    return {atten_out, softmax_lse};
  }

  if (layout_q_str == "BSND") {
    softmax_lse = torch::empty({q.size(kDim0),
                                kv_head_num,
                                q.size(kDim1),
                                q.size(kDim2) / kv_head_num},
                               q.options().dtype(torch::kFloat32));
  } else {
    softmax_lse =
        torch::empty({kv_head_num, q.size(kDim0), q.size(kDim1) / kv_head_num},
                     q.options().dtype(torch::kFloat32));
  }
  return {atten_out, softmax_lse};
}

}  // namespace

torch::Tensor smla_va_metadata(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const std::optional<torch::Tensor>& cu_seqlens_q,
    const std::optional<torch::Tensor>& cu_seqlens_ori_kv,
    const std::optional<torch::Tensor>& cu_seqlens_cmp_kv,
    const std::optional<torch::Tensor>& seqused_q,
    const std::optional<torch::Tensor>& seqused_ori_kv,
    const std::optional<torch::Tensor>& seqused_cmp_kv,
    const std::optional<torch::Tensor>& cmp_residual_kv,
    const std::optional<torch::Tensor>& ori_topk_length,
    const std::optional<torch::Tensor>& cmp_topk_length,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_ori_kv,
    int64_t max_seqlen_cmp_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv) {
  torch::Device output_device(torch::kPrivateUse1);
  const std::array<const std::optional<torch::Tensor>*, 9> tensor_args = {
      &cu_seqlens_q,
      &cu_seqlens_ori_kv,
      &cu_seqlens_cmp_kv,
      &seqused_q,
      &seqused_ori_kv,
      &seqused_cmp_kv,
      &cmp_residual_kv,
      &ori_topk_length,
      &cmp_topk_length};
  for (const auto* arg : tensor_args) {
    if (arg->has_value()) {
      output_device = arg->value().device();
      break;
    }
  }

  torch::Tensor output =
      torch::zeros({kDsaMetadataBufferElements},
                   torch::dtype(torch::kInt32).device(output_device));
  auto cu_seqlens_q_val =
      detail::materialize_int32_metadata(cu_seqlens_q, output_device);
  auto cu_seqlens_ori_kv_val =
      detail::materialize_int32_metadata(cu_seqlens_ori_kv, output_device);
  auto cu_seqlens_cmp_kv_val =
      detail::materialize_int32_metadata(cu_seqlens_cmp_kv, output_device);
  auto seqused_q_val =
      detail::materialize_int32_metadata(seqused_q, output_device);
  auto seqused_ori_kv_val =
      detail::materialize_int32_metadata(seqused_ori_kv, output_device);
  auto seqused_cmp_kv_val =
      detail::materialize_int32_metadata(seqused_cmp_kv, output_device);
  auto cmp_residual_kv_val =
      detail::materialize_int32_metadata(cmp_residual_kv, output_device);
  auto ori_topk_length_val =
      detail::materialize_int32_metadata(ori_topk_length, output_device);
  auto cmp_topk_length_val =
      detail::materialize_int32_metadata(cmp_topk_length, output_device);

  // Workspace creation consumes layout strings synchronously; queued execution
  // retains only the executor.
  const std::string layout_q_str(layout_q);
  const std::string layout_kv_str(layout_kv);
  const char* layout_q_ptr = layout_q_str.c_str();
  const char* layout_kv_ptr = layout_kv_str.c_str();

  EXEC_NPU_CMD(aclnnSmlaVaMetadata,
               cu_seqlens_q_val,
               cu_seqlens_ori_kv_val,
               cu_seqlens_cmp_kv_val,
               seqused_q_val,
               seqused_ori_kv_val,
               seqused_cmp_kv_val,
               cmp_residual_kv_val,
               ori_topk_length_val,
               cmp_topk_length_val,
               num_heads_q,
               num_heads_kv,
               head_dim,
               batch_size,
               max_seqlen_q,
               max_seqlen_ori_kv,
               max_seqlen_cmp_kv,
               ori_topk,
               cmp_topk,
               cmp_ratio,
               ori_mask_mode,
               cmp_mask_mode,
               ori_win_left,
               ori_win_right,
               layout_q_ptr,
               layout_kv_ptr,
               has_ori_kv,
               has_cmp_kv,
               output);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> smla_va(
    const torch::Tensor& q,
    const std::optional<torch::Tensor>& ori_kv,
    const std::optional<torch::Tensor>& cmp_kv,
    const std::optional<torch::Tensor>& ori_sparse_indices,
    const std::optional<torch::Tensor>& cmp_sparse_indices,
    const std::optional<torch::Tensor>& ori_block_table,
    const std::optional<torch::Tensor>& cmp_block_table,
    const std::optional<torch::Tensor>& cu_seqlens_q,
    const std::optional<torch::Tensor>& cu_seqlens_ori_kv,
    const std::optional<torch::Tensor>& cu_seqlens_cmp_kv,
    const std::optional<torch::Tensor>& seqused_q,
    const std::optional<torch::Tensor>& seqused_ori_kv,
    const std::optional<torch::Tensor>& seqused_cmp_kv,
    const std::optional<torch::Tensor>& cmp_residual_kv,
    const std::optional<torch::Tensor>& ori_topk_length,
    const std::optional<torch::Tensor>& cmp_topk_length,
    const std::optional<torch::Tensor>& sinks,
    const std::optional<torch::Tensor>& metadata,
    double softmax_scale,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    int64_t topk_value_mode,
    bool return_softmax_lse) {
  std::string layout_q_str = std::string(layout_q);
  std::string layout_kv_str = std::string(layout_kv);
  char* layout_q_ptr = layout_q_str.data();
  char* layout_kv_ptr = layout_kv_str.data();

  auto [atten_out, softmax_lse] = make_smla_outputs(
      q, ori_kv, cmp_kv, layout_q_str, layout_kv_str, return_softmax_lse);

  EXEC_NPU_CMD(aclnnSmlaVa,
               q,
               ori_kv,
               cmp_kv,
               ori_sparse_indices,
               cmp_sparse_indices,
               ori_block_table,
               cmp_block_table,
               cu_seqlens_q,
               cu_seqlens_ori_kv,
               cu_seqlens_cmp_kv,
               seqused_q,
               seqused_ori_kv,
               seqused_cmp_kv,
               cmp_residual_kv,
               ori_topk_length,
               cmp_topk_length,
               sinks,
               metadata,
               softmax_scale,
               cmp_ratio,
               ori_mask_mode,
               cmp_mask_mode,
               ori_win_left,
               ori_win_right,
               layout_q_ptr,
               layout_kv_ptr,
               topk_value_mode,
               return_softmax_lse,
               atten_out,
               softmax_lse);
  return std::tuple<torch::Tensor, torch::Tensor>(atten_out, softmax_lse);
}

}  // namespace xllm::kernel::npu
