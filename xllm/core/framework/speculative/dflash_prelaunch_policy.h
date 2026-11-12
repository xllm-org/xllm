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

#include <cstdint>
#include <string>
#include <string_view>

#include "core/common/types.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model_loader/hf_model_args.h"
#include "core/util/tensor_helper.h"

namespace xllm {

// Scheduler reservations and worker launches must use the same eligibility.
template <typename Options>
bool supports_dflash2_prelaunch(std::string_view algorithm,
                                const Options& options,
                                int32_t dp_size,
                                int32_t cp_size,
                                bool is_ascend_a3,
                                bool is_python_model) {
  return SpeculativeConfig::is_dflash2_algorithm(algorithm) && is_ascend_a3 &&
         is_python_model && options.enable_schedule_overlap() && dp_size == 1 &&
         cp_size == 1 &&
         (!options.enable_disagg_pd() ||
          options.instance_role() == InstanceRole::DECODE);
}

// XFIA is optional: unsupported draft geometry retains ordinary attention.
inline bool supports_dflash2_xfia(const std::string& draft_model_path,
                                  const std::string& backend,
                                  int32_t block_size,
                                  std::string_view kv_cache_dtype) {
  if (block_size != 128 || kv_cache_dtype != "auto") {
    return false;
  }
  ModelArgs draft_args;
  if (!load_hf_model_args(draft_model_path, backend, &draft_args)) {
    return false;
  }
  const auto dtype = try_get_scalar_type_from_string(draft_args.dtype());
  return !draft_args.enable_mla() && draft_args.head_dim() == 128 &&
         (dtype == torch::kFloat16 || dtype == torch::kBFloat16);
}

}  // namespace xllm
