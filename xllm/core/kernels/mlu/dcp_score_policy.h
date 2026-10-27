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

#pragma once

#include <torch/types.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

namespace xllm::kernel {

inline constexpr int64_t kDcpScoreHeads = 32;
inline constexpr int64_t kDcpScoreHeadDim = 128;

struct DcpScorePolicy {
  int64_t block_n;
  int32_t num_stages;
  int64_t slot_cap;
  int64_t page_shift;
  uint32_t programs;
  int64_t row_cap;
  bool row_owned;
  bool small_rows;
  bool preload_counts;
  bool prefetch_rows;
  bool program_proof;
  bool narrow;
};

inline DcpScorePolicy dcp_score_policy(const torch::Tensor& query,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& cache,
                                       const torch::Tensor& slots,
                                       const torch::Tensor& counts,
                                       int64_t cores,
                                       int64_t physical_block_size) {
  const int64_t rows = slots.size(0);
  const int64_t width = slots.size(1);
  const int64_t cache_slots = cache.size(0);
  const bool row_owned = rows >= cores;
  const bool small_rows = rows <= 128;
  int64_t block_n = 512;
  int32_t num_stages = 99;
  if (query.stride(-1) != 1 || weights.stride(-1) != 1 ||
      cache.stride(-1) != 1 || slots.stride(-1) != 1 ||
      counts.stride(-1) != 1) {
    block_n = 32;
    num_stages = 3;
  } else if (width < 256) {
    block_n = 128;
    num_stages = 3;
  } else if (!row_owned || small_rows) {
    block_n = 256;
    num_stages = !row_owned || width <= 4096 ? 5 : 99;
  }
  const int64_t slot_cap =
      static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(width)));
  int64_t page_shift = 0;
  for (int64_t page = physical_block_size; page > 1; page >>= 1) {
    ++page_shift;
  }
  const uint32_t programs = static_cast<uint32_t>(
      std::min(cores, rows * ((width + block_n - 1) / block_n)));
  const int64_t row_cap =
      rows <= 16384
          ? static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(rows)))
          : 0;
  const int64_t bound = row_cap == 0 ? rows : row_cap;
  // Widen before multiplication: valid int64 strides can overflow int64 byte
  // bounds even when no corresponding large allocation is made (zero strides).
  const auto wide = [](int64_t value) { return static_cast<__int128>(value); };
  constexpr int64_t kLimit = std::numeric_limits<int32_t>::max();
  const bool narrow = cache_slots <= kLimit &&
                      (wide(bound - 1) * query.stride(0) +
                       wide(kDcpScoreHeads - 1) * query.stride(1) +
                       wide(kDcpScoreHeadDim - 1) * query.stride(2)) *
                              4 <
                          kLimit &&
                      (wide(bound - 1) * weights.stride(0) +
                       wide(kDcpScoreHeads - 1) * weights.stride(1)) *
                              4 <
                          kLimit &&
                      (wide(cache_slots - 1) * cache.stride(0) +
                       wide(kDcpScoreHeadDim - 1) * cache.stride(1)) *
                              4 <
                          kLimit &&
                      (wide(bound - 1) * slots.stride(0) +
                       (wide(slot_cap) + block_n - 1) * slots.stride(1)) *
                              4 <
                          kLimit &&
                      wide(bound - 1) * counts.stride(0) * 4 < kLimit &&
                      (wide(bound) * width + block_n) * 4 < kLimit;
  return {block_n,
          num_stages,
          slot_cap,
          page_shift,
          programs,
          row_cap,
          row_owned,
          small_rows,
          row_owned && rows > 128 && rows <= 16384,
          row_owned && rows > cores && small_rows,
          rows >= 2 * cores && small_rows && width <= 4096,
          narrow};
}

}  // namespace xllm::kernel
