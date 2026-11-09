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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor> rms_norm_cast_va(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    double epsilon) {
  CHECK(x.scalar_type() == torch::kHalf || x.scalar_type() == torch::kBFloat16)
      << "rms_norm_cast_va only supports float16 and bfloat16 input";
  CHECK(x.scalar_type() == gamma.scalar_type())
      << "x and gamma must have the same dtype";
  CHECK(gamma.dim() == 1 && x.dim() >= 1 && x.size(-1) == gamma.size(0))
      << "gamma must be 1D and match the last dimension of x";

  torch::Tensor y = torch::empty(x.sizes(), x.options());
  torch::Tensor y_fp32 =
      torch::empty(x.sizes(), x.options().dtype(torch::kFloat));
  EXEC_NPU_CMD(aclnnRmsNormCastVa, x, gamma, epsilon, y, y_fp32);
  return {y, y_fp32};
}

}  // namespace xllm::kernel::npu
