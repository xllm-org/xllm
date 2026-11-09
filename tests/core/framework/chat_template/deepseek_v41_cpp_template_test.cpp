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

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace {

DeepseekV41CppTemplate make_encoder() {
  return DeepseekV41CppTemplate(TokenizerArgs{});
}

Message make_tool_calls(const std::vector<std::string>& ids) {
  Message message("assistant", "");
  Message::ToolCallVec calls;
  for (const auto& id : ids) {
    Message::ToolCall call;
    call.id = id;
    call.type = "function";
    call.function.name = "lookup";
    call.function.arguments = R"({"text":"hello","value":1})";
    calls.push_back(std::move(call));
  }
  message.tool_calls = std::move(calls);
  return message;
}

Message make_tool_result(const std::string& id, const std::string& content) {
  Message message("tool", content);
  message.tool_call_id = id;
  return message;
}

TEST(DeepseekV41CppTemplate, EmptyConversationContainsOnlyBos) {
  const auto prompt = make_encoder().apply({});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt, "<｜begin▁of▁sentence｜>");
}

TEST(DeepseekV41CppTemplate, SnapshotsCustomBosToken) {
  TokenizerArgs args;
  args.bos_token("CUSTOM");
  const DeepseekV41CppTemplate encoder(args);
  args.bos_token("CHANGED");
  const auto prompt =
      encoder.apply({{"user", "hello"}}, {}, {{"thinking", false}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt, "CUSTOM<｜User｜>hello<｜Assistant｜></think>");
}

TEST(DeepseekV41CppTemplate, ChatModePreservesInitialSystemToken) {
  const auto prompt = make_encoder().apply(
      {{"system", "rules"}, {"user", "hi"}}, {}, {{"thinking", false}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜><｜System｜>rules"
            "<｜User｜>hi<｜Assistant｜></think>");
}

TEST(DeepseekV41CppTemplate, MidConversationSystemTokenIsPreserved) {
  const auto prompt = make_encoder().apply({{"system", "rules"},
                                            {"assistant", "answer"},
                                            {"system", "next"},
                                            {"user", "hi"}},
                                           {},
                                           {{"thinking", false}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜><｜System｜>rulesanswer"
            "<｜end▁of▁sentence｜><｜System｜>next"
            "<｜User｜>hi<｜Assistant｜></think>");
}

TEST(DeepseekV41CppTemplate, DefaultRequestUsesNumericHighBudget) {
  const auto prompt = make_encoder().apply({{"user", "hello"}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 "
            "(range 1-100, the higher the value, the more thorough the "
            "reasoning)\n\n<｜User｜>hello<｜Assistant｜><think>");
}

TEST(DeepseekV41CppTemplate, ReasoningBudgetsPreserveNamedAndNumericTiers) {
  for (const auto& [effort, budget] :
       std::vector<std::pair<std::string, std::string>>{{"low", "25"},
                                                        {"high", "50"},
                                                        {"xhigh", "75"},
                                                        {"max", "100"},
                                                        {"1", "1"},
                                                        {"42", "42"},
                                                        {"100", "100"}}) {
    SCOPED_TRACE(effort);
    const auto prompt = make_encoder().apply(
        {{"user", "hello"}}, {}, {{"reasoning_effort", effort}});
    ASSERT_TRUE(prompt.has_value());
    EXPECT_EQ(*prompt,
              "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: " + budget +
                  " (range 1-100, the higher the value, the "
                  "more thorough the reasoning)\n\n"
                  "<｜User｜>hello<｜Assistant｜><think>");
  }
}

TEST(DeepseekV41CppTemplate, NoneEffortAndExplicitFalseDisableThinking) {
  for (const auto& kwargs : std::vector<nlohmann::ordered_json>{
           {{"reasoning_effort", "none"}},
           {{"thinking", false}, {"reasoning_effort", "max"}},
           {{"enable_thinking", false}, {"reasoning_effort", "high"}}}) {
    const auto prompt = make_encoder().apply({{"user", "hello"}}, {}, kwargs);
    ASSERT_TRUE(prompt.has_value());
    EXPECT_EQ(*prompt,
              "<｜begin▁of▁sentence｜><｜User｜>hello"
              "<｜Assistant｜></think>");
  }
}

TEST(DeepseekV41CppTemplate, ExplicitModeHasPriorityOverThinkingBoolean) {
  const auto prompt = make_encoder().apply(
      {{"user", "hello"}}, {}, {{"thinking_mode", "chat"}, {"thinking", true}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello"
            "<｜Assistant｜></think>");
}

TEST(DeepseekV41CppTemplate, ToolCallsUseSpacedDsmlDialect) {
  const auto prompt =
      make_encoder().apply({make_tool_calls({"a"})}, {}, {{"thinking", false}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜>\n\n<｜DSML｜ calls>\n"
            "<｜DSML｜ invoke name=\"lookup\">\n"
            "<｜DSML｜ parameter name=\"text\" string=\"true\">hello"
            "</｜DSML｜ parameter>\n"
            "<｜DSML｜ parameter name=\"value\" string=\"false\">1"
            "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
            "</｜DSML｜ calls><｜end▁of▁sentence｜>");
}

TEST(DeepseekV41CppTemplate, UnknownToolResultsFollowKnownAndRemainStable) {
  const auto prompt = make_encoder().apply({make_tool_calls({"a", "b"}),
                                            make_tool_result("z", "Z1"),
                                            make_tool_result("b", "B"),
                                            make_tool_result("z", "Z2"),
                                            make_tool_result("a", "A")},
                                           {},
                                           {{"thinking", false}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_TRUE(prompt->ends_with(
      "<｜User｜><tool_result>A</tool_result>\n\n"
      "<tool_result>B</tool_result>\n\n<tool_result>Z1</tool_result>\n\n"
      "<tool_result>Z2</tool_result><｜Assistant｜></think>"));
}

TEST(DeepseekV41CppTemplate, OrderedDuplicateResultsKeepStableOrder) {
  const auto prompt = make_encoder().apply({make_tool_calls({"a", "b"}),
                                            make_tool_result("a", "A1"),
                                            make_tool_result("a", "A2"),
                                            make_tool_result("b", "B")},
                                           {},
                                           {{"thinking", false}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_TRUE(prompt->ends_with(
      "<｜User｜><tool_result>A1</tool_result>\n\n"
      "<tool_result>A2</tool_result>\n\n<tool_result>B</tool_result>"
      "<｜Assistant｜></think>"));
}

TEST(DeepseekV41CppTemplate, ToolSortingPreservesTextBlockPositions) {
  const auto prompt = make_encoder().apply({make_tool_calls({"a", "b"}),
                                            make_tool_result("b", "B"),
                                            {"user", "middle"},
                                            make_tool_result("a", "A")},
                                           {},
                                           {{"thinking", false}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_TRUE(
      prompt->ends_with("<｜User｜><tool_result>A</tool_result>\n\nmiddle\n\n"
                        "<tool_result>B</tool_result><｜Assistant｜></think>"));
}

TEST(DeepseekV41CppTemplate, ToolSchemaPreservesDeferLoading) {
  for (const bool deferred : {false, true}) {
    JsonTool tool;
    tool.function.name = "lookup";
    tool.function.parameters = {{"type", "object"}};
    tool.function.defer_loading = deferred;
    const auto prompt = make_encoder().apply(
        {{"user", "hello"}}, {tool}, {{"thinking", false}});
    ASSERT_TRUE(prompt.has_value());
    EXPECT_NE(prompt->find(std::string("\"defer_loading\":") +
                           (deferred ? "true" : "false")),
              std::string::npos);
  }
}

TEST(DeepseekV41CppTemplate, ToolsPreservePriorThinking) {
  Message reply("assistant", "answer");
  reply.reasoning_content = "historical reasoning";
  const ChatMessages messages{{"user", "first"}, reply, {"user", "next"}};
  const auto without_tools = make_encoder().apply(messages);
  ASSERT_TRUE(without_tools.has_value());
  EXPECT_EQ(without_tools->find("historical reasoning"), std::string::npos);
  JsonTool tool;
  tool.function.name = "lookup";
  tool.function.parameters = {{"type", "object"}};
  const auto with_tools = make_encoder().apply(messages, {tool}, {});
  ASSERT_TRUE(with_tools.has_value());
  EXPECT_NE(with_tools->find("historical reasoning"), std::string::npos);
}

TEST(DeepseekV41CppTemplate, MalformedToolArgumentsDoNotAffectNextRender) {
  Message bad = make_tool_calls({"a"});
  bad.tool_calls->front().function.arguments = "{invalid";
  const auto encoder = make_encoder();
  const auto malformed = encoder.apply({bad});
  ASSERT_TRUE(malformed.has_value());
  EXPECT_NE(malformed->find("{invalid"), std::string::npos);
  const auto valid =
      encoder.apply({{"user", "hello"}}, {}, {{"thinking", false}});
  ASSERT_TRUE(valid.has_value());
  EXPECT_EQ(*valid,
            "<｜begin▁of▁sentence｜><｜User｜>hello"
            "<｜Assistant｜></think>");
}

}  // namespace
}  // namespace xllm
