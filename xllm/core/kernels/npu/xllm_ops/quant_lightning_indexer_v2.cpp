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

#include <glog/logging.h>

#include <string>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

std::tuple<torch::Tensor, torch::Tensor>
construct_quant_lightning_indexer_v2_va_output(const torch::Tensor& query,
                                               const torch::Tensor& key,
                                               int64_t sparse_count,
                                               const std::string& query_layout,
                                               const std::string& key_layout) {
  CHECK(sparse_count > 0)
      << "sparse count should be greater than 0, but now is " << sparse_count;
  const auto output_size = detail::qli_output_size(
      query, key, sparse_count, query_layout, key_layout);
  torch::Tensor sparse_indices_out =
      torch::zeros(output_size, query.options().dtype(torch::kInt));
  torch::Tensor sparse_values_out =
      torch::zeros({0}, query.options().dtype(torch::kBFloat16));
  return std::tuple<torch::Tensor, torch::Tensor>(sparse_indices_out,
                                                  sparse_values_out);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
quant_lightning_indexer_v2_va(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weights,
    const torch::Tensor& query_dequant_scale,
    const torch::Tensor& key_dequant_scale,
    int64_t topk,
    int64_t quant_mode,
    const std::optional<torch::Tensor>& candidate_topk_index,
    const std::optional<torch::Tensor>& cu_seqlens_q,
    const std::optional<torch::Tensor>& cu_seqlens_k,
    const std::optional<torch::Tensor>& seqused_q,
    const std::optional<torch::Tensor>& seqused_k,
    const std::optional<torch::Tensor>& cmp_residual_k,
    const std::optional<torch::Tensor>& block_table,
    const std::optional<torch::Tensor>& output_idx_offset,
    const std::optional<torch::Tensor>& metadata,
    int64_t max_seqlen_q,
    c10::string_view layout_q,
    c10::string_view layout_k,
    int64_t mask_mode,
    int64_t cmp_ratio,
    int64_t candidate_mode,
    int64_t candidate_topk_blocks,
    int64_t candidate_block_size) {
  CHECK(query.numel() > 0) << "Tensor query is empty.";
  CHECK(key.numel() > 0) << "Tensor key is empty.";
  CHECK(quant_mode == 2)
      << "Aurora QLI V2 currently supports INT8 quant_mode=2";
  CHECK(query.scalar_type() == torch::kInt8 &&
        key.scalar_type() == torch::kInt8)
      << "QLI V2 query/key must be INT8";
  CHECK(weights.scalar_type() == torch::kHalf &&
        query_dequant_scale.scalar_type() == torch::kHalf &&
        key_dequant_scale.scalar_type() == torch::kHalf)
      << "QLI V2 weights/scales must be FP16";
  CHECK(candidate_mode >= 1 && candidate_mode <= 3) << "Invalid candidate_mode";
  CHECK(candidate_mode != 2 || candidate_topk_index.has_value())
      << "Consumer requires candidate blocks";

  std::string query_layout_str = std::string(layout_q);
  std::string key_layout_str = std::string(layout_k);

  const c10::OptionalDeviceGuard device_guard(c10::Device(query.device()));

  const int64_t return_value_attr = 0;
  std::tuple<torch::Tensor, torch::Tensor> indexer_output =
      construct_quant_lightning_indexer_v2_va_output(
          query, key, topk, query_layout_str, key_layout_str);
  torch::Tensor sparse_indices_out = std::get<0>(indexer_output);
  torch::Tensor sparse_values_out = std::get<1>(indexer_output);

  torch::Tensor candidate_topk_index_out;
  if (candidate_mode == 1) {
    const auto candidate_size = detail::qli_output_size(
        query, key, candidate_topk_blocks, query_layout_str, key_layout_str);
    candidate_topk_index_out =
        torch::zeros(candidate_size, query.options().dtype(torch::kInt));
  } else {
    candidate_topk_index_out =
        torch::zeros({0}, query.options().dtype(torch::kInt));
  }

  if (query.device().is_meta()) {
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
        sparse_indices_out, sparse_values_out, candidate_topk_index_out);
  }

  // ACLNN reads these layout strings synchronously through its char* ABI.
  char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
  char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());
  // ACLNN cannot inspect these tensor strides; pass the first-dimension strides
  // explicitly.
  const int64_t key_stride0 = key.stride(0);
  const int64_t key_dequant_scale_stride0 = key_dequant_scale.stride(0);

  EXEC_NPU_CMD(aclnnQuantLightningIndexerV2Va,
               query,
               key,
               weights,
               query_dequant_scale,
               key_dequant_scale,
               cu_seqlens_q,
               cu_seqlens_k,
               seqused_q,
               seqused_k,
               cmp_residual_k,
               block_table,
               output_idx_offset,
               metadata,
               candidate_topk_index,
               topk,
               quant_mode,
               max_seqlen_q,
               query_layout_ptr,
               key_layout_ptr,
               mask_mode,
               cmp_ratio,
               return_value_attr,
               candidate_mode,
               candidate_topk_blocks,
               candidate_block_size,
               key_stride0,
               key_dequant_scale_stride0,
               sparse_indices_out,
               sparse_values_out,
               candidate_topk_index_out);

  return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
      sparse_indices_out, sparse_values_out, candidate_topk_index_out);
}

}  // namespace xllm::kernel::npu
