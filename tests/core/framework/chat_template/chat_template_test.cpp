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

#include "framework/chat_template/chat_template.h"

#include <gtest/gtest.h>

#include "core/framework/config/model_config.h"
#include "framework/chat_template/chat_template_factory.h"
#include "framework/chat_template/deepseek_v32_cpp_template.h"
#include "framework/chat_template/deepseek_v41_cpp_template.h"
#include "framework/chat_template/deepseek_v4_cpp_template.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "function_call/function_call_parser.h"

namespace xllm {
namespace {

class ScopedUseCppChatTemplate final {
 public:
  explicit ScopedUseCppChatTemplate(bool enabled)
      : old_value_(ModelConfig::get_instance().use_cpp_chat_template()) {
    ModelConfig::get_instance().use_cpp_chat_template(enabled);
  }

  ~ScopedUseCppChatTemplate() {
    ModelConfig::get_instance().use_cpp_chat_template(old_value_);
  }

 private:
  bool old_value_;
};

TEST(ChatTemplateFactory, DeepseekV32FallsBackToJinjaWhenFlagDisabled) {
  ScopedUseCppChatTemplate scoped_flag(/*enabled=*/false);
  TokenizerArgs args;

  std::unique_ptr<ChatTemplate> impl =
      ChatTemplateFactory::create(args, /*model_type=*/"deepseek_v32");

  ASSERT_TRUE(impl != nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV32CppTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV4FallsBackToJinjaWhenFlagDisabled) {
  ScopedUseCppChatTemplate scoped_flag(/*enabled=*/false);
  TokenizerArgs args;

  std::unique_ptr<ChatTemplate> impl =
      ChatTemplateFactory::create(args, /*model_type=*/"deepseek_v4");

  ASSERT_TRUE(impl != nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV32CppTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV4UsesCppTemplateWhenFlagEnabled) {
  ScopedUseCppChatTemplate scoped_flag(/*enabled=*/true);
  TokenizerArgs args;

  std::unique_ptr<ChatTemplate> impl =
      ChatTemplateFactory::create(args, /*model_type=*/"deepseek_v4");

  ASSERT_TRUE(impl != nullptr);
  EXPECT_EQ(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV32CppTemplate*>(impl.get()), nullptr);
  EXPECT_NE(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV41UsesCppTemplateWhenEnabled) {
  ScopedUseCppChatTemplate scoped_flag(true);
  auto impl = ChatTemplateFactory::create(TokenizerArgs{}, "deepseek_v41");
  ASSERT_NE(impl, nullptr);
  ASSERT_NE(dynamic_cast<DeepseekV41CppTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
  const auto prompt = impl->apply({{"user", "hello"}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("<｜System｜>Reasoning Effort: 50"),
            std::string::npos);
}

TEST(ChatTemplateFactory, DeepseekV41UsesJinjaWhenDisabled) {
  ScopedUseCppChatTemplate scoped_flag(false);
  auto impl = ChatTemplateFactory::create(TokenizerArgs{}, "deepseek_v41");
  ASSERT_NE(impl, nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV41CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, NonDeepseekModelUsesJinjaWhenFlagEnabled) {
  ScopedUseCppChatTemplate scoped_flag(/*enabled=*/true);
  TokenizerArgs args;

  ModelConfig::get_instance().use_cpp_chat_template(false);
  std::unique_ptr<ChatTemplate> impl =
      ChatTemplateFactory::create(args, /*model_type=*/"qwen3");

  ASSERT_TRUE(impl != nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV32CppTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekToolDialectMatchesAutoParser) {
  ScopedUseCppChatTemplate scoped_flag(true);
  for (const auto& model :
       {std::string("deepseek_v4"), std::string("deepseek_v41")}) {
    SCOPED_TRACE(model);
    auto encoder = ChatTemplateFactory::create(TokenizerArgs{}, model);
    ASSERT_NE(encoder, nullptr);
    Message message("assistant", "");
    Message::ToolCall call;
    call.id = "lookup-call";
    call.type = "function";
    call.function.name = "lookup";
    call.function.arguments = R"({"city":"北京","days":2})";
    message.tool_calls = Message::ToolCallVec{call};
    const auto prompt = encoder->apply({message}, {}, {{"thinking", false}});
    ASSERT_TRUE(prompt.has_value());
    const std::vector<JsonTool> tools{JsonTool(
        "function", JsonFunction("lookup", "Lookup", {{"type", "object"}}))};
    function_call::FunctionCallParser parser(
        tools,
        function_call::FunctionCallParser::get_parser_auto("auto", model));
    ASSERT_TRUE(parser.has_tool_call(*prompt));
    const auto [text, calls] = parser.parse_non_stream(*prompt);
    ASSERT_EQ(calls.size(), 1);
    EXPECT_EQ(*calls[0].name, "lookup");
    EXPECT_EQ(nlohmann::json::parse(calls[0].parameters),
              nlohmann::json::parse(call.function.arguments));
  }
}

}  // namespace
}  // namespace xllm
