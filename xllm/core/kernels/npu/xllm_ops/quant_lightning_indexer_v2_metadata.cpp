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

#include <optional>
#include <string>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor quant_lightning_indexer_v2_metadata_va(
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    int64_t topk,
    int64_t quant_mode,
    const std::optional<torch::Tensor>& cu_seqlens_q,
    const std::optional<torch::Tensor>& cu_seqlens_k,
    const std::optional<torch::Tensor>& seqused_q,
    const std::optional<torch::Tensor>& seqused_k,
    const std::optional<torch::Tensor>& cmp_residual_k,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    c10::string_view layout_q,
    c10::string_view layout_k,
    int64_t mask_mode,
    int64_t cmp_ratio,
    c10::string_view device) {
  torch::Device output_device(torch::kPrivateUse1);
  if (cu_seqlens_q.has_value()) {
    output_device = cu_seqlens_q.value().device();
  } else if (cu_seqlens_k.has_value()) {
    output_device = cu_seqlens_k.value().device();
  } else if (seqused_q.has_value()) {
    output_device = seqused_q.value().device();
  } else if (seqused_k.has_value()) {
    output_device = seqused_k.value().device();
  } else if (cmp_residual_k.has_value()) {
    output_device = cmp_residual_k.value().device();
  } else {
    output_device = torch::Device(std::string(device));
  }

  torch::Tensor output = at::zeros(
      {kDsaMetadataBufferElements},
      torch::TensorOptions().dtype(torch::kInt).device(output_device));
  // Missing optionals must stay nullptr; an empty tensor is rejected as a
  // null-buffer aclTensor.

  if (output.device().is_meta()) {
    return output;
  }

  std::string layout_q_str = std::string(layout_q);
  std::string layout_k_str = std::string(layout_k);
  char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
  char* layout_k_ptr = const_cast<char*>(layout_k_str.c_str());

  EXEC_NPU_CMD(aclnnQuantLightningIndexerV2MetadataVa,
               cu_seqlens_q,
               cu_seqlens_k,
               seqused_q,
               seqused_k,
               cmp_residual_k,
               num_heads_q,
               num_heads_k,
               head_dim,
               topk,
               quant_mode,
               batch_size,
               max_seqlen_q,
               max_seqlen_k,
               layout_q_ptr,
               layout_k_ptr,
               mask_mode,
               cmp_ratio,
               output);

  return output;
}

}  // namespace xllm::kernel::npu
