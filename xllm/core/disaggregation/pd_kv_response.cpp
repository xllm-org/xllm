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

#include "core/disaggregation/pd_kv_response.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/framework/request/sequence_kv_state.h"
#include "core/kv_cache/block/block.h"

namespace xllm {

void append_pd_kv_transfer_groups(
    const KVCacheState& kv_state,
    bool include_linear_state,
    proto::DisaggResponse* response,
    std::vector<int32_t>* virtual_memory_block_ids) {
  if (kv_state.has_multi_block_export()) {
    // Iterate borrowed slices directly, without allocating an export-view
    // vector or copying reference-counted Block handles.
    for (const BlockType block_type : kMultiBlockExportOrder) {
      const auto blocks = kv_state.blocks(block_type);
      if (blocks.empty()) {
        continue;
      }
      proto::KVTransferGroup* group = response->add_groups();
      group->set_group_id(cache_group_id(block_type));
      // P uses this cursor to skip grouped blocks already cached on D.
      group->set_remote_shared_num(
          static_cast<uint32_t>(kv_state.shared_blocks_num(block_type)));
      group->mutable_ids()->Reserve(static_cast<int32_t>(blocks.size()));
      for (const auto& block : blocks) {
        const bool valid = block.is_valid();
        CHECK(valid || block_type == BlockType::SWA)
            << "Decode allocated an invalid grouped KV block under a "
            << "non-SWA leaf, group_id=" << cache_group_id(block_type);
        // Expired SWA slots must preserve their logical indices on the wire.
        const int64_t block_id = valid ? block.id() : -1;
        group->add_ids(static_cast<uint64_t>(block_id));
      }
    }
  } else {
    const auto blocks = kv_state.blocks(BlockType::KV);
    const size_t shared_num = kv_state.shared_blocks_num(BlockType::KV);
    CHECK_LE(shared_num, blocks.size());
    proto::KVTransferGroup* group = response->add_groups();
    group->set_group_id(cache_group_id(BlockType::KV));
    group->set_remote_shared_num(static_cast<uint32_t>(shared_num));
    group->mutable_ids()->Reserve(
        static_cast<int32_t>(blocks.size() - shared_num));
    if (virtual_memory_block_ids != nullptr) {
      virtual_memory_block_ids->reserve(virtual_memory_block_ids->size() +
                                        blocks.size() - shared_num);
    }
    for (size_t i = shared_num; i < blocks.size(); ++i) {
      const int32_t block_id = blocks[i].id();
      group->add_ids(static_cast<uint64_t>(block_id));
      if (virtual_memory_block_ids != nullptr) {
        virtual_memory_block_ids->push_back(block_id);
      }
    }
  }

  if (include_linear_state) {
    const int32_t linear_state_id = kv_state.get_linear_block_id();
    CHECK_GE(linear_state_id, 0)
        << "Decode did not allocate a linear-state slot.";
    proto::KVTransferGroup* group = response->add_groups();
    group->set_group_id(cache_group_id(BlockType::LINEAR));
    group->add_ids(static_cast<uint64_t>(linear_state_id));
  }
}

bool append_pd_pull_source_groups(const KVCacheState& kv_state,
                                  bool include_linear_state,
                                  proto::DisaggGenerationsRequest* request) {
  if (!kv_state.has_multi_block_export()) {
    const Slice<Block> blocks = kv_state.blocks(BlockType::KV);
    proto::KVTransferGroup* group = request->add_source_groups();
    group->set_group_id(cache_group_id(BlockType::KV));
    group->mutable_ids()->Reserve(static_cast<int32_t>(blocks.size()));
    for (const Block& block : blocks) {
      CHECK(block.is_valid());
      group->add_ids(static_cast<uint64_t>(block.id()));
    }
  } else {
    for (const BlockType block_type : kMultiBlockExportOrder) {
      const Slice<Block> blocks = kv_state.blocks(block_type);
      if (blocks.empty()) {
        continue;
      }
      if (block_type != BlockType::SWA) {
        proto::KVTransferGroup* group = request->add_source_groups();
        group->set_group_id(cache_group_id(block_type));
        group->mutable_ids()->Reserve(static_cast<int32_t>(blocks.size()));
        for (const Block& block : blocks) {
          if (block.is_valid()) {
            group->add_ids(static_cast<uint64_t>(block.id()));
          }
        }
        continue;
      }

      const auto& transfer_info = kv_state.transfer_kv_info();
      const KVTransferMapping* swa_mapping = nullptr;
      if (transfer_info.has_value()) {
        for (const KVTransferMapping& mapping : transfer_info->mappings) {
          if (mapping.group_id != cache_group_id(BlockType::SWA)) {
            continue;
          }
          if (swa_mapping != nullptr) {
            LOG(ERROR) << "Duplicate SWA PULL mapping, request_id="
                       << transfer_info->request_id;
            return false;
          }
          swa_mapping = &mapping;
        }
      }

      if (swa_mapping == nullptr) {
        LOG(ERROR) << "Missing SWA PULL mapping, request_id="
                   << (transfer_info.has_value() ? transfer_info->request_id
                                                 : std::string());
        return false;
      }
      if (blocks.size() != swa_mapping->remote_ids.size()) {
        LOG(ERROR) << "SWA PULL logical table length mismatch, request_id="
                   << transfer_info->request_id << ", source=" << blocks.size()
                   << ", receiver=" << swa_mapping->remote_ids.size();
        return false;
      }
      for (size_t i = 0; i < blocks.size(); ++i) {
        if (swa_mapping->remote_ids[i] ==
            std::numeric_limits<uint64_t>::max()) {
          continue;
        }
        if (!blocks[i].is_valid()) {
          LOG(ERROR) << "Invalid required SWA source block, request_id="
                     << transfer_info->request_id << ", index=" << i;
          return false;
        }
      }

      proto::KVTransferGroup* group = request->add_source_groups();
      group->set_group_id(cache_group_id(BlockType::SWA));
      group->mutable_ids()->Reserve(static_cast<int32_t>(blocks.size()));
      for (size_t i = 0; i < blocks.size(); ++i) {
        if (swa_mapping->remote_ids[i] ==
            std::numeric_limits<uint64_t>::max()) {
          continue;
        }
        group->add_ids(static_cast<uint64_t>(blocks[i].id()));
      }
    }
  }

  if (include_linear_state) {
    const int32_t linear_state_id = kv_state.get_linear_block_id();
    CHECK_GE(linear_state_id, 0)
        << "Prefill did not allocate a linear-state slot.";
    proto::KVTransferGroup* group = request->add_source_groups();
    group->set_group_id(cache_group_id(BlockType::LINEAR));
    group->add_ids(static_cast<uint64_t>(linear_state_id));
  }
  return true;
}

}  // namespace xllm
