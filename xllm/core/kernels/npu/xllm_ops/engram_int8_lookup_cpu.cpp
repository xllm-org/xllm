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

#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

constexpr int64_t kGroupSize = 32;
constexpr int64_t kParallelMinRows = 256;
constexpr int64_t kParallelGrain = 128;
// Cap the grain argument for very large batches; backend partition sizes remain
// implementation-dependent.
constexpr int64_t kParallelGrainMax = kParallelGrain * 4;

#if defined(__aarch64__)
uint16x4_t to_bfloat16(float32x4_t values) {
  const auto bits = vreinterpretq_u32_f32(values);
  const auto bias = vaddq_u32(vdupq_n_u32(0x7fff),
                              vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1)));
  const uint16x4_t rounded = vshrn_n_u32(vaddq_u32(bits, bias), 16);
  const uint16x4_t not_nan_mask = vmovn_u32(vceqq_f32(values, values));
  // Match c10::BFloat16's canonical NaN and round-to-nearest-even.
  return vbsl_u16(not_nan_mask, rounded, vdup_n_u16(0x7fc0));
}
#endif

}  // namespace

void engram_int8_lookup_cpu(const torch::Tensor& weight,
                            const torch::Tensor& scale,
                            const torch::Tensor& ids,
                            torch::Tensor& output) {
  for (const auto& tensor : {weight, scale, ids, output}) {
    TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous(),
                "Engram CPU lookup requires contiguous CPU tensors");
  }
  TORCH_CHECK(
      weight.scalar_type() == torch::kChar &&
          scale.scalar_type() == torch::kFloat &&
          ids.scalar_type() == torch::kLong &&
          output.scalar_type() == torch::kBFloat16,
      "Engram CPU lookup expects INT8 weight, FP32 scale, INT64 IDs and "
      "BF16 output");
  TORCH_CHECK(weight.dim() == 2 && scale.dim() == 2 && ids.dim() == 1 &&
                  output.dim() == 2,
              "Invalid Engram CPU lookup dimensions");
  const int64_t rows = weight.size(0);
  const int64_t num_ids = ids.numel();
  const int64_t width = weight.size(1);
  const int64_t groups = width / kGroupSize;
  TORCH_CHECK(width > 0 && width % kGroupSize == 0 && scale.size(0) == rows &&
                  scale.size(1) == groups && output.size(0) == num_ids &&
                  output.size(1) == width,
              "Invalid Engram CPU lookup shapes");
  const auto* codes = weight.const_data_ptr<int8_t>();
  const auto* scales = scale.const_data_ptr<float>();
  const auto* indices = ids.const_data_ptr<int64_t>();
  auto* out = output.mutable_data_ptr<torch::BFloat16>();
  // Check IDs before parallel_for: worker exceptions cannot safely cross its
  // unwind boundary.
  for (int64_t i = 0; i < num_ids; ++i) {
    const int64_t row = indices[i];
    TORCH_CHECK(row >= 0 && row < rows, "Engram CPU lookup ID outside shard");
  }
  const int64_t num_threads =
      std::max<int64_t>(1, static_cast<int64_t>(torch::get_num_threads()));
  const int64_t grain =
      num_ids < kParallelMinRows
          ? num_ids
          : std::min(std::max(kParallelGrain, num_ids / (num_threads * 4)),
                     kParallelGrainMax);
  at::parallel_for(0, num_ids, grain, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t row = indices[i];
      for (int64_t g = 0; g < groups; ++g) {
        const float factor = scales[row * groups + g];
#if defined(__aarch64__)
        for (int64_t j = 0; j < kGroupSize; j += 8) {
          const int64_t column = g * kGroupSize + j;
          const auto values = vmovl_s8(vld1_s8(codes + row * width + column));
          const auto low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(values)));
          const auto high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(values)));
          auto* destination =
              reinterpret_cast<uint16_t*>(out + i * width + column);
          vst1_u16(destination, to_bfloat16(vmulq_n_f32(low, factor)));
          vst1_u16(destination + 4, to_bfloat16(vmulq_n_f32(high, factor)));
        }
#else
        for (int64_t j = 0; j < kGroupSize; ++j) {
          const int64_t column = g * kGroupSize + j;
          out[i * width + column] =
              torch::BFloat16(float(codes[row * width + column]) * factor);
        }
#endif
      }
    }
  });
}

}  // namespace xllm::kernel::npu
