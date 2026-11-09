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

#include "function_call/deepseekv41_detector.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "function_call/function_call_parser.h"

namespace xllm {
namespace function_call {
namespace {

const std::string kCall =
    "<｜DSML｜ calls><｜DSML｜ invoke name=\"weather\">"
    "<｜DSML｜ parameter name=\"city\" string=\"true\">北京"
    "</｜DSML｜ parameter>"
    "<｜DSML｜ parameter name=\"days\" string=\"false\">2"
    "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";

const std::string kUnspacedCall =
    "<｜DSML｜tool_calls><｜DSML｜invoke name=\"weather\">"
    "<｜DSML｜parameter name=\"city\" string=\"true\">北京"
    "</｜DSML｜parameter>"
    "<｜DSML｜parameter name=\"days\" string=\"false\">2"
    "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";

std::vector<JsonTool> weather_tools() {
  return {JsonTool("function",
                   JsonFunction("weather",
                                "Weather",
                                {{"type", "object"},
                                 {"properties",
                                  {{"city", {{"type", "string"}}},
                                   {"days", {{"type", "integer"}}}}}}))};
}

void check_stream(BaseFormatDetector& detector,
                  const std::vector<std::string>& chunks) {
  const auto tools = weather_tools();
  std::string text =
      detector.parse_streaming_increment("前文 ", tools).normal_text;
  std::string arguments;
  int32_t names = 0;
  for (const auto& chunk : chunks) {
    auto result = detector.parse_streaming_increment(chunk, tools);
    text += result.normal_text;
    for (const auto& call : result.calls) {
      if (call.name.has_value()) {
        EXPECT_EQ(*call.name, "weather");
        ++names;
      }
      arguments += call.parameters;
    }
  }
  EXPECT_EQ(text, "前文");
  EXPECT_EQ(names, 1);
  const auto args = nlohmann::json::parse(arguments);
  EXPECT_EQ(args["city"], "北京");
  EXPECT_EQ(args["days"], 2);
}

TEST(DeepSeekV41DetectorTest, ParsesSpacedTagsAndTypedUtf8Arguments) {
  DeepSeekV41Detector detector;
  const auto result =
      detector.detect_and_parse("前文 " + kCall, weather_tools());
  EXPECT_EQ(result.normal_text, "前文");
  ASSERT_EQ(result.calls.size(), 1);
  ASSERT_TRUE(result.calls[0].name.has_value());
  EXPECT_EQ(*result.calls[0].name, "weather");
  const auto args = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_EQ(args["city"], "北京");
  EXPECT_EQ(args["days"], 2);
}

TEST(DeepSeekV41DetectorTest, DetectsUnwrappedSpacedInvoke) {
  DeepSeekV41Detector detector;
  EXPECT_TRUE(detector.has_tool_call("<｜DSML｜ invoke name=\"weather\">"));
  const auto begin = kCall.find("<｜DSML｜ invoke");
  const auto end = kCall.find("</｜DSML｜ calls>");
  const auto result = detector.detect_and_parse(
      kCall.substr(begin, end - begin), weather_tools());
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(*result.calls[0].name, "weather");
}

TEST(DeepSeekV41DetectorTest, StreamsTagAndArgumentByteBoundaries) {
  const std::string input = kCall;
  for (size_t split = 0; split <= input.size(); ++split) {
    SCOPED_TRACE(split);
    DeepSeekV41Detector detector;
    check_stream(detector, {input.substr(0, split), input.substr(split)});
  }
}

TEST(DeepSeekSharedDetectorTest, StreamsClosingTagsOneByteAtATime) {
  auto stream_bytes = [](BaseFormatDetector& detector,
                         const std::string& input) {
    std::vector<std::string> chunks;
    for (char byte : input) {
      chunks.emplace_back(1, byte);
    }
    check_stream(detector, chunks);
  };
  auto replace_all = [](std::string& input,
                        const std::string& before,
                        const std::string& after) {
    size_t pos = 0;
    while ((pos = input.find(before, pos)) != std::string::npos) {
      input.replace(pos, before.size(), after);
      pos += after.size();
    }
  };
  DeepSeekV41Detector v41;
  stream_bytes(v41, kCall);
  std::string v4_call = kCall;
  replace_all(v4_call, "｜DSML｜ ", "｜DSML｜");
  replace_all(v4_call, "｜calls>", "｜tool_calls>");
  DeepSeekV4Detector v4;
  stream_bytes(v4, v4_call);
  replace_all(v4_call, "tool_calls>", "function_calls>");
  DeepSeekV32Detector v32;
  stream_bytes(v32, v4_call);
}

TEST(DeepSeekV41DetectorTest, StreamsArgumentsBeforeInvokeCloses) {
  DeepSeekV41Detector detector;
  const auto tools = weather_tools();
  detector.parse_streaming_increment(
      "<｜DSML｜ calls><｜DSML｜ invoke name=\"weather\">"
      "<｜DSML｜ parameter name=\"city\" string=\"true\">B",
      tools);
  const auto progress = detector.parse_streaming_increment("e", tools);
  std::string arguments;
  for (const auto& call : progress.calls) {
    arguments += call.parameters;
  }
  EXPECT_FALSE(arguments.empty());
  const auto tail = detector.parse_streaming_increment(
      "ijing</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>", tools);
  for (const auto& call : tail.calls) {
    arguments += call.parameters;
  }
  EXPECT_EQ(nlohmann::json::parse(arguments)["city"], "Beijing");
}

TEST(DeepSeekV41DetectorTest, StreamingPreservesParameterOrderAndValues) {
  DeepSeekV41Detector detector;
  const auto tools = weather_tools();
  const std::string input =
      "<｜DSML｜ calls><｜DSML｜ invoke name=\"weather\">"
      "<｜DSML｜ parameter name=\"z_last\" string=\"true\"> 北京 "
      "</｜DSML｜ parameter>"
      "<｜DSML｜ parameter name=\"a_first\" string=\"false\">1.25"
      "</｜DSML｜ parameter>"
      "<｜DSML｜ parameter name=\"boolean\" string=\"false\">true"
      "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
  std::string arguments;
  for (char byte : input) {
    const auto result =
        detector.parse_streaming_increment(std::string(1, byte), tools);
    for (const auto& call : result.calls) {
      arguments += call.parameters;
    }
  }
  const auto parsed = nlohmann::json::parse(arguments);
  EXPECT_EQ(parsed["z_last"], " 北京 ");
  EXPECT_EQ(parsed["a_first"], 1.25);
  EXPECT_EQ(parsed["boolean"], true);
}

TEST(DeepSeekV41DetectorTest, StreamsJsonParametersInsideInvoke) {
  const std::string input =
      "<｜DSML｜ calls><｜DSML｜ invoke name=\"weather\">"
      "{\"city\":\"北京\",\"days\":2}"
      "</｜DSML｜ invoke></｜DSML｜ calls>";
  for (size_t split = 0; split <= input.size(); ++split) {
    SCOPED_TRACE(split);
    DeepSeekV41Detector detector;
    check_stream(detector, {input.substr(0, split), input.substr(split)});
  }
}

TEST(DeepSeekV41DetectorTest, RetainsSharedJsonFallback) {
  DeepSeekV41Detector detector;
  const auto result = detector.detect_and_parse(
      R"({"tool_calls":[{"name":"weather","arguments":{"city":"北京","days":2}}]})",
      weather_tools());
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(*result.calls[0].name, "weather");
  EXPECT_EQ(nlohmann::json::parse(result.calls[0].parameters)["days"], 2);
}

TEST(DeepSeekV41DetectorTest, RetainsSharedTrailingJsonHintNormalization) {
  DeepSeekV41Detector detector;
  const auto result =
      detector.detect_and_parse("前文\n```json\n" + kCall, weather_tools());
  EXPECT_EQ(result.normal_text, "前文");
  ASSERT_EQ(result.calls.size(), 1);
}

TEST(DeepSeekV41DetectorTest, AutoFactoryUsesV41Dialect) {
  const auto selected =
      FunctionCallParser::get_parser_auto("auto", "deepseek_v41");
  EXPECT_EQ(selected, "deepseekv41");
  FunctionCallParser parser(weather_tools(), selected);
  EXPECT_TRUE(parser.has_tool_call(kCall));
  const auto [text, calls] = parser.parse_non_stream(kCall);
  ASSERT_EQ(calls.size(), 1);
  EXPECT_EQ(*calls[0].name, "weather");
  EXPECT_EQ(nlohmann::json::parse(calls[0].parameters)["city"], "北京");
}

TEST(DeepSeekV41DetectorTest, PreservesV4AndV32PartialJsonStreaming) {
  const std::string expected = R"({"days":[1,2,3]})";
  for (const auto& dialect :
       {std::string("deepseekv4"), std::string("deepseekv32")}) {
    SCOPED_TRACE(dialect);
    FunctionCallParser parser(weather_tools(), dialect);
    std::string progress;
    const std::vector<std::string> chunks = {
        "<｜DSML｜invoke name=\"weather\">"
        "<｜DSML｜parameter name=\"days\" string=\"false\">[1,",
        "2,",
        "3]</｜DSML｜parameter></｜DSML｜invoke>"};
    for (size_t i = 0; i < chunks.size(); ++i) {
      const auto result = parser.parse_streaming_increment(chunks[i]);
      for (const auto& call : result.calls) {
        progress += call.parameters;
      }
      EXPECT_EQ(expected.substr(0, progress.size()), progress);
      if (i == 1) {
        EXPECT_FALSE(progress.empty());
        EXPECT_NE(progress.find('1'), std::string::npos);
      }
    }
    EXPECT_EQ(progress, expected);
    EXPECT_EQ(nlohmann::json::parse(progress)["days"],
              nlohmann::json::array({1, 2, 3}));
  }
}

TEST(DeepSeekV41DetectorTest, DefersNonStringValueUntilParameterCloses) {
  FunctionCallParser parser(weather_tools(), "deepseekv41");
  for (const auto& chunk :
       {std::string("<｜DSML｜ invoke name=\"weather\">"
                    "<｜DSML｜ parameter name=\"days\" string=\"false\">12"),
        std::string("e-"),
        std::string("3")}) {
    const auto result = parser.parse_streaming_increment(chunk);
    for (const auto& call : result.calls) {
      EXPECT_TRUE(call.parameters.empty());
    }
  }
  const auto result = parser.parse_streaming_increment(
      "</｜DSML｜ parameter></｜DSML｜ invoke>");
  std::string arguments;
  for (const auto& call : result.calls) {
    arguments += call.parameters;
  }
  EXPECT_EQ(arguments, R"({"days":0.012})");
  EXPECT_EQ(nlohmann::json::parse(arguments)["days"], 0.012);
}

TEST(DeepSeekV41DetectorTest, CachedPatternsPreserveCustomCallBoundaries) {
  DeepSeekV4Detector standard;
  DeepSeekV41Detector spaced;
  DeepSeekV32Detector custom("<custom-calls>", "</custom-calls>");
  const auto result = custom.detect_and_parse(
      "<custom-calls><｜DSML｜invoke name=\"weather\">"
      "<｜DSML｜parameter name=\"days\" string=\"false\">2"
      "</｜DSML｜parameter></｜DSML｜invoke></custom-calls>",
      weather_tools());
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(nlohmann::json::parse(result.calls[0].parameters)["days"], 2);
  check_stream(standard, {kUnspacedCall});
  check_stream(spaced, {kCall});
}

TEST(DeepSeekV41DetectorTest, CachedPatternsPreserveCustomInvocationTags) {
  class CustomDetector : public DeepSeekV32Detector {
   public:
    CustomDetector()
        : DeepSeekV32Detector("<custom-calls>",
                              "</custom-calls>",
                              "run",
                              "arg") {}
  };
  DeepSeekV4Detector standard;
  DeepSeekV41Detector spaced;
  CustomDetector custom;
  const auto result = custom.detect_and_parse(
      "<custom-calls><｜DSML｜run name=\"weather\">"
      "<｜DSML｜arg name=\"days\" string=\"false\">2"
      "</｜DSML｜arg></｜DSML｜run></custom-calls>",
      weather_tools());
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(nlohmann::json::parse(result.calls[0].parameters)["days"], 2);
  check_stream(standard, {kUnspacedCall});
  check_stream(spaced, {kCall});
}

}  // namespace
}  // namespace function_call
}  // namespace xllm
