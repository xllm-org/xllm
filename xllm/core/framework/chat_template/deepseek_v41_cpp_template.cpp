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

#include "framework/chat_template/deepseek_v41_cpp_template.h"

#include "framework/chat_template/deepseek_chat_template.h"

namespace xllm {

DeepseekV41CppTemplate::DeepseekV41CppTemplate(const TokenizerArgs& args)
    : bos_token_(args.bos_token().empty() ? deepseek_chat::kBosToken
                                          : args.bos_token()) {}

std::optional<std::string> DeepseekV41CppTemplate::apply(
    const ChatMessages& messages) const {
  return apply(messages, {}, nlohmann::ordered_json::object());
}

}  // namespace xllm
