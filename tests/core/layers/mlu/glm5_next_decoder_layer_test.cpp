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

#include "layers/mlu/glm5_next/glm5_next_decoder_layer.h"

#include <gtest/gtest.h>

#include "framework/model/model_args.h"

namespace xllm::layer {
namespace {

TEST(Glm5NextDecoderLayerTest, ResolvesAttentionAndMlpRolesIndependently) {
  ModelArgs args;
  args.layer_types(
      {"linear_attention", "deepseek_sparse_attention", "linear_attention"});
  args.mlp_layer_types({"dense", "sparse", "sparse"});

  const Glm5NextLayerRole layer0 = resolve_glm5_next_layer_role(args, 0);
  EXPECT_EQ(layer0.attention, Glm5NextAttentionRole::KDA);
  EXPECT_EQ(layer0.mlp, Glm5NextMlpRole::DENSE);

  const Glm5NextLayerRole layer1 = resolve_glm5_next_layer_role(args, 1);
  EXPECT_EQ(layer1.attention, Glm5NextAttentionRole::DSA);
  EXPECT_EQ(layer1.mlp, Glm5NextMlpRole::SPARSE);

  const Glm5NextLayerRole layer2 = resolve_glm5_next_layer_role(args, 2);
  EXPECT_EQ(layer2.attention, Glm5NextAttentionRole::KDA);
  EXPECT_EQ(layer2.mlp, Glm5NextMlpRole::SPARSE);
}

}  // namespace
}  // namespace xllm::layer
