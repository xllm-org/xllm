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

#include "framework/chat_template/deepseek_chat_template.h"

#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "framework/chat_template/deepseek_v41_cpp_template.h"
#include "framework/chat_template/deepseek_v4_cpp_template.h"

namespace xllm {
namespace {

constexpr const char* kEosToken = "<｜end▁of▁sentence｜>";
constexpr const char* kThinkingStartToken = "<think>";
constexpr const char* kThinkingEndToken = "</think>";
constexpr const char* kDsmlToken = "｜DSML｜";
template <bool IsV41>
constexpr const char* kDsmlSeparator = IsV41 ? " " : "";

constexpr const char* kSystemSpToken = "<｜System｜>";
constexpr const char* kUserSpToken = "<｜User｜>";
constexpr const char* kAssistantSpToken = "<｜Assistant｜>";
constexpr const char* kLatestReminderSpToken = "<｜latest_reminder｜>";

constexpr const char* kToolCallsBlockName = "tool_calls";

const std::unordered_map<std::string, std::string>& task_sp_tokens() {
  static const std::unordered_map<std::string, std::string> kMap = {
      {"action", "<｜action｜>"},
      {"query", "<｜query｜>"},
      {"authority", "<｜authority｜>"},
      {"domain", "<｜domain｜>"},
      {"title", "<｜title｜>"},
      {"read_url", "<｜read_url｜>"},
  };
  return kMap;
}

constexpr const char* kToolsTemplate =
    "## Tools\n"
    "\n"
    "You have access to a set of tools to help "
    "answer the user's question. You can invoke "
    "tools by writing a "
    "\"<{dsml_token}{dsml_sep}{block_name}>\" block like the "
    "following:\n"
    "\n"
    "<{dsml_token}{dsml_sep}{block_name}>\n"
    "<{dsml_token}{dsml_sep}invoke name=\"$TOOL_NAME\">\n"
    "<{dsml_token}{dsml_sep}parameter "
    "name=\"$PARAMETER_NAME\" "
    "string=\"true|false\">"
    "$PARAMETER_VALUE"
    "</{dsml_token}{dsml_sep}parameter>\n"
    "...\n"
    "</{dsml_token}{dsml_sep}invoke>\n"
    "<{dsml_token}{dsml_sep}invoke name=\"$TOOL_NAME2\">\n"
    "...\n"
    "</{dsml_token}{dsml_sep}invoke>\n"
    "</{dsml_token}{dsml_sep}{block_name}>\n"
    "\n"
    "String parameters should be specified as is "
    "and set `string=\"true\"`. For all other types "
    "(numbers, booleans, arrays, objects), pass the "
    "value in JSON format and set "
    "`string=\"false\"`.\n"
    "\n"
    "If thinking_mode is enabled (triggered by "
    "{thinking_start_token}), you MUST output your "
    "complete reasoning inside "
    "{thinking_start_token}..."
    "{thinking_end_token} BEFORE any tool calls or "
    "final response.\n"
    "\n"
    "Otherwise, output directly after "
    "{thinking_end_token} with tool calls or final "
    "response.\n"
    "\n"
    "### Available Tool Schemas\n"
    "\n"
    "{tool_schemas}\n"
    "\n"
    "You MUST strictly follow the above defined "
    "tool name and parameter schemas to invoke "
    "tool calls.\n";

constexpr const char* kResponseFormatTemplate =
    "## Response Format:\n\n"
    "You MUST strictly adhere to the following "
    "schema to reply:\n{schema}";

constexpr const char* kToolCallTemplate =
    "<{dsml_token}{dsml_sep}invoke name=\"{name}\">\n"
    "{arguments}\n"
    "</{dsml_token}{dsml_sep}invoke>";

constexpr const char* kReasoningEffortPromptHigh =
    "Reasoning Effort: Absolute maximum with no "
    "shortcuts permitted.\n"
    "You MUST be very thorough in your thinking "
    "and comprehensively decompose the problem to "
    "resolve the root cause, rigorously "
    "stress-testing your logic against all "
    "potential paths, edge cases, and adversarial "
    "scenarios.\n"
    "Explicitly write out your entire deliberation "
    "process, documenting every intermediate step, "
    "considered alternative, and rejected "
    "hypothesis to ensure absolutely no assumption "
    "is left unchecked.\n\n";

constexpr const char* kReasoningEffortPromptMax =
    "Reasoning Effort: Beyond maximum — "
    "exhaustive, relentless, and "
    "uncompromising.\n"
    "You MUST reason with the utmost depth and "
    "rigor, leaving absolutely nothing to chance: "
    "exhaustively decompose the problem into its "
    "most fundamental components, trace every "
    "causal chain to its root, and resolve the "
    "underlying cause rather than any surface "
    "symptom.\n"
    "Do not stop reasoning until you have "
    "independently verified the solution from "
    "multiple angles and are certain that no "
    "assumption remains unchecked and no error "
    "remains undiscovered.\n\n";

constexpr const char* kReasoningEffortNone = "none";
constexpr const char* kReasoningEffortLowTier = "low";
constexpr const char* kReasoningEffortHighTier = "high";
constexpr const char* kReasoningEffortMaxTier = "max";

constexpr const char* kThinkingModeThinking = "thinking";
constexpr const char* kThinkingModeChat = "chat";

constexpr const char* kRoleSystem = "system";
constexpr const char* kRoleDeveloper = "developer";
constexpr const char* kRoleUser = "user";
constexpr const char* kRoleTool = "tool";
constexpr const char* kRoleAssistant = "assistant";
constexpr const char* kRoleLatestReminder = "latest_reminder";
constexpr const char* kRoleDirectSearchResults = "direct_search_results";

std::string to_json(const nlohmann::ordered_json& value) {
  try {
    return value.dump(
        /*indent=*/-1,
        /*indent_char=*/' ',
        /*ensure_ascii=*/false,
        nlohmann::json::error_handler_t::replace);
  } catch (const std::exception&) {
    return value.dump(
        /*indent=*/-1,
        /*indent_char=*/' ',
        /*ensure_ascii=*/true,
        nlohmann::json::error_handler_t::replace);
  }
}

nlohmann::ordered_json parse_json_object_or_empty(const std::string& text) {
  if (text.empty()) {
    return nlohmann::ordered_json::object();
  }
  try {
    nlohmann::ordered_json parsed = nlohmann::json::parse(text);
    if (parsed.is_object()) {
      return parsed;
    }
    return nlohmann::ordered_json::object();
  } catch (const std::exception&) {
    nlohmann::ordered_json fallback = nlohmann::ordered_json::object();
    fallback["arguments"] = text;
    return fallback;
  }
}

bool get_thinking_enabled(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("thinking") && kwargs["thinking"].is_boolean()) {
    return kwargs["thinking"].get<bool>();
  }
  if (kwargs.contains("enable_thinking") &&
      kwargs["enable_thinking"].is_boolean()) {
    return kwargs["enable_thinking"].get<bool>();
  }
  return false;
}

std::string get_reasoning_effort(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("reasoning_effort") &&
      kwargs["reasoning_effort"].is_string()) {
    return kwargs["reasoning_effort"].get<std::string>();
  }
  return "";
}

std::string resolve_reasoning_effort(const std::string& reasoning_effort) {
  if (reasoning_effort == kReasoningEffortNone) {
    return "";
  }
  if (reasoning_effort == kReasoningEffortMaxTier) {
    return kReasoningEffortMaxTier;
  }
  if (reasoning_effort == "minimal" ||
      reasoning_effort == kReasoningEffortLowTier ||
      reasoning_effort == "medium") {
    return kReasoningEffortLowTier;
  }
  return kReasoningEffortHighTier;
}

const char* reasoning_effort_prompt(const std::string& resolved_effort) {
  if (resolved_effort == kReasoningEffortMaxTier) {
    return kReasoningEffortPromptMax;
  }
  if (resolved_effort == kReasoningEffortHighTier) {
    return kReasoningEffortPromptHigh;
  }
  return "";
}

constexpr const char* kReasoningEffortTemplate =
    "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n";

constexpr const char* kEffortWeightLow = "25";
constexpr const char* kEffortWeightHigh = "50";
constexpr const char* kEffortWeightXHigh = "75";
constexpr const char* kEffortWeightMax = "100";

std::string resolve_reasoning_effort_v41(const std::string& reasoning_effort) {
  if (reasoning_effort == kReasoningEffortNone) {
    return "";
  }
  if (reasoning_effort == "low") {
    return kEffortWeightLow;
  }
  if (reasoning_effort == "high") {
    return kEffortWeightHigh;
  }
  if (reasoning_effort == "xhigh") {
    return kEffortWeightXHigh;
  }
  if (reasoning_effort == "max") {
    return kEffortWeightMax;
  }
  if (!reasoning_effort.empty()) {
    size_t consumed = 0;
    try {
      int32_t budget = std::stoi(reasoning_effort, &consumed);
      if (consumed == reasoning_effort.size() && budget >= 1 && budget <= 100) {
        return std::to_string(budget);
      }
    } catch (const std::exception&) {
    }
    LOG_FIRST_N(WARNING, 1)
        << "Unrecognized reasoning effort \"" << reasoning_effort
        << "\"; falling back to " << kEffortWeightHigh;
  }
  return kEffortWeightHigh;
}

std::string reasoning_effort_prompt_v41(const std::string& resolved_effort) {
  if (resolved_effort.empty()) {
    return "";
  }
  return absl::StrReplaceAll(kReasoningEffortTemplate,
                             {{"{budget}", resolved_effort}});
}

bool has_explicit_thinking_flag(const nlohmann::ordered_json& kwargs) {
  return (kwargs.contains("thinking") && kwargs["thinking"].is_boolean()) ||
         (kwargs.contains("enable_thinking") &&
          kwargs["enable_thinking"].is_boolean());
}

inline std::string get_thinking_mode(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("thinking_mode") && kwargs["thinking_mode"].is_string()) {
    return kwargs["thinking_mode"].get<std::string>();
  }
  const std::string reasoning_effort = get_reasoning_effort(kwargs);
  if (reasoning_effort == kReasoningEffortNone) {
    return kThinkingModeChat;
  }
  if (!has_explicit_thinking_flag(kwargs)) {
    return kThinkingModeThinking;
  }
  return get_thinking_enabled(kwargs) ? kThinkingModeThinking
                                      : kThinkingModeChat;
}

nlohmann::ordered_json tool_calls_from_openai_format(
    const Message::ToolCallVec& tool_calls,
    size_t& prompt_capacity) {
  nlohmann::ordered_json out = nlohmann::ordered_json::array();
  out.get_ref<nlohmann::ordered_json::array_t&>().reserve(tool_calls.size());
  for (const Message::ToolCall& tc : tool_calls) {
    prompt_capacity +=
        tc.function.name.size() + tc.function.arguments.size() + 128;
    nlohmann::ordered_json item;
    item["id"] = tc.id;
    item["name"] = tc.function.name;
    item["arguments"] = parse_json_object_or_empty(tc.function.arguments);
    out.emplace_back(std::move(item));
  }
  return out;
}

template <bool IsV41>
std::string encode_arguments_to_dsml(const nlohmann::ordered_json& tool_call) {
  const auto arguments_it = tool_call.find("arguments");
  if (arguments_it == tool_call.end()) {
    return "";
  }
  const nlohmann::ordered_json* arguments = &*arguments_it;
  nlohmann::ordered_json parsed_arguments;
  if (arguments->is_string()) {
    parsed_arguments =
        parse_json_object_or_empty(arguments->get_ref<const std::string&>());
    arguments = &parsed_arguments;
  }
  std::ostringstream oss;
  bool first = true;
  for (auto it = arguments->begin(); it != arguments->end(); ++it) {
    if (!first) {
      oss << "\n";
    }
    first = false;
    const nlohmann::ordered_json& value = it.value();
    bool is_string = value.is_string();
    oss << "<" << kDsmlToken << kDsmlSeparator<IsV41> << "parameter name=\""
        << it.key() << "\" string=\"" << (is_string ? "true" : "false")
        << "\">";
    if (is_string) {
      oss << value.get_ref<const std::string&>();
    } else {
      oss << to_json(value);
    }
    oss << "</" << kDsmlToken << kDsmlSeparator<IsV41> << "parameter>";
  }
  return oss.str();
}

template <bool IsV41>
int32_t find_last_user_index(const nlohmann::ordered_json& messages) {
  for (int32_t idx = static_cast<int32_t>(messages.size()) - 1; idx >= 0;
       --idx) {
    std::string role = messages[idx].value("role", "");
    if (role == kRoleUser || role == kRoleDeveloper ||
        (IsV41 && role == kRoleSystem && idx > 0)) {
      return idx;
    }
  }
  return -1;
}

std::string get_text_content(const Message::Content& content) {
  if (std::holds_alternative<std::string>(content)) {
    return std::get<std::string>(content);
  }
  return "";
}

template <bool IsV41>
std::string render_tools(const nlohmann::ordered_json& tools) {
  std::vector<std::string> schemas;
  schemas.reserve(tools.size());
  for (const nlohmann::ordered_json& tool : tools) {
    const auto function = tool.find("function");
    if (function != tool.end()) {
      schemas.emplace_back(to_json(*function));
    }
  }
  return absl::StrReplaceAll(
      kToolsTemplate,
      {{"{tool_schemas}", absl::StrJoin(schemas, "\n")},
       {"{dsml_sep}", kDsmlSeparator<IsV41>},
       {"{block_name}", IsV41 ? "calls" : kToolCallsBlockName},
       {"{dsml_token}", kDsmlToken},
       {"{thinking_start_token}", kThinkingStartToken},
       {"{thinking_end_token}", kThinkingEndToken}});
}

template <bool IsV41>
std::string render_tool_calls(const nlohmann::ordered_json& tool_calls) {
  if (tool_calls.empty()) {
    return "";
  }
  const char* block_name = IsV41 ? "calls" : kToolCallsBlockName;
  std::vector<std::string> chunks;
  chunks.reserve(tool_calls.size());
  for (const nlohmann::ordered_json& tc : tool_calls) {
    std::string invoke = absl::StrReplaceAll(
        kToolCallTemplate,
        {{"{dsml_token}", kDsmlToken},
         {"{dsml_sep}", kDsmlSeparator<IsV41>},
         {"{name}", tc.value("name", "")},
         {"{arguments}", encode_arguments_to_dsml<IsV41>(tc)}});
    chunks.emplace_back(std::move(invoke));
  }
  std::string block = std::string("<") + kDsmlToken + kDsmlSeparator<IsV41> +
                      std::string(block_name) + ">\n" +
                      absl::StrJoin(chunks, "\n") + "\n</" + kDsmlToken +
                      kDsmlSeparator<IsV41> + std::string(block_name) + ">";
  return "\n\n" + block;
}

nlohmann::ordered_json merge_tool_messages(nlohmann::ordered_json messages) {
  nlohmann::ordered_json merged = nlohmann::json::array();
  merged.get_ref<nlohmann::ordered_json::array_t&>().reserve(messages.size());

  for (auto& msg : messages) {
    std::string role = msg.value("role", "");

    if (role == kRoleTool) {
      nlohmann::ordered_json tool_block;
      tool_block["type"] = "tool_result";
      const auto tool_call_id = msg.find("tool_call_id");
      tool_block["tool_use_id"] = tool_call_id == msg.end()
                                      ? nlohmann::ordered_json("")
                                      : std::move(*tool_call_id);
      tool_block["content"] = std::move(msg["content"]);

      if (!merged.empty() && merged.back().value("role", "") == kRoleUser &&
          merged.back().contains("content_blocks")) {
        merged.back()["content_blocks"].emplace_back(std::move(tool_block));
      } else {
        nlohmann::ordered_json new_msg;
        new_msg["role"] = kRoleUser;
        new_msg["content_blocks"] = nlohmann::ordered_json::array();
        new_msg["content_blocks"].emplace_back(std::move(tool_block));
        merged.emplace_back(std::move(new_msg));
      }
    } else if (role == kRoleUser) {
      nlohmann::ordered_json text_block;
      text_block["type"] = "text";
      text_block["text"] = std::move(msg["content"]);

      if (!merged.empty() && merged.back().value("role", "") == kRoleUser &&
          merged.back().contains("content_blocks") &&
          !merged.back().contains("task")) {
        merged.back()["content_blocks"].emplace_back(std::move(text_block));
      } else {
        nlohmann::ordered_json new_msg;
        new_msg["role"] = kRoleUser;
        new_msg["content"] = text_block["text"];
        new_msg["content_blocks"] = nlohmann::ordered_json::array();
        new_msg["content_blocks"].emplace_back(std::move(text_block));
        for (const auto& key : {"task", "wo_eos", "mask"}) {
          if (msg.contains(key)) {
            new_msg[key] = std::move(msg[key]);
          }
        }
        merged.emplace_back(std::move(new_msg));
      }
    } else {
      merged.emplace_back(std::move(msg));
    }
  }
  return merged;
}

template <bool IsV41>
nlohmann::ordered_json sort_tool_results_by_call_order(
    nlohmann::ordered_json messages) {
  std::unordered_map<std::string, int32_t> last_tc_order;
  constexpr int32_t kUnknownOrder =
      IsV41 ? std::numeric_limits<int32_t>::max() : 0;

  for (auto& msg : messages) {
    std::string role = msg.value("role", "");
    if (role == kRoleAssistant && msg.contains("tool_calls") &&
        msg["tool_calls"].is_array()) {
      last_tc_order.clear();
      last_tc_order.reserve(msg["tool_calls"].size());
      int32_t idx = 0;
      for (const auto& tc : msg["tool_calls"]) {
        std::string tc_id;
        if (tc.contains("id")) {
          tc_id = tc.value("id", "");
        } else if (tc.contains("function") && tc["function"].contains("id")) {
          tc_id = tc["function"].value("id", "");
        }
        if (!tc_id.empty()) {
          last_tc_order[std::move(tc_id)] = idx;
        }
        ++idx;
      }
    } else if (role == kRoleUser && msg.contains("content_blocks")) {
      if (last_tc_order.empty()) {
        continue;
      }
      auto& blocks = msg["content_blocks"];
      int32_t tool_count = 0;
      int32_t previous_order = -1;
      bool strictly_ordered = true;
      for (const auto& b : blocks) {
        if (b.value("type", "") == "tool_result") {
          ++tool_count;
          const auto order_it = last_tc_order.find(b.value("tool_use_id", ""));
          const int32_t order = order_it != last_tc_order.end()
                                    ? order_it->second
                                    : kUnknownOrder;
          strictly_ordered =
              strictly_ordered &&
              (IsV41 ? previous_order <= order : previous_order < order);
          previous_order = order;
        }
      }
      if (tool_count > 1 && !strictly_ordered) {
        std::vector<nlohmann::ordered_json> tool_blocks;
        tool_blocks.reserve(tool_count);
        std::vector<size_t> tool_positions;
        tool_positions.reserve(tool_count);
        for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
          auto& b = blocks[block_idx];
          if (b.value("type", "") == "tool_result") {
            tool_positions.emplace_back(block_idx);
            tool_blocks.emplace_back(std::move(b));
          }
        }
        const auto compare = [&](const nlohmann::ordered_json& a,
                                 const nlohmann::ordered_json& b) {
          auto ia = last_tc_order.find(a.value("tool_use_id", ""));
          auto ib = last_tc_order.find(b.value("tool_use_id", ""));
          int32_t oa = ia != last_tc_order.end() ? ia->second : kUnknownOrder;
          int32_t ob = ib != last_tc_order.end() ? ib->second : kUnknownOrder;
          return oa < ob;
        };
        if constexpr (IsV41) {
          std::stable_sort(tool_blocks.begin(), tool_blocks.end(), compare);
        } else {
          std::sort(tool_blocks.begin(), tool_blocks.end(), compare);
        }
        for (size_t idx = 0; idx < tool_positions.size(); ++idx) {
          blocks[tool_positions[idx]] = std::move(tool_blocks[idx]);
        }
      }
    }
  }
  return messages;
}

template <bool IsV41>
nlohmann::ordered_json drop_thinking_messages(nlohmann::ordered_json messages) {
  const int32_t last_user_idx = find_last_user_index<IsV41>(messages);
  auto& message_array = messages.get_ref<nlohmann::ordered_json::array_t&>();
  size_t retained_count = 0;

  for (int32_t idx = 0; idx < static_cast<int32_t>(messages.size()); ++idx) {
    auto& message = messages[idx];
    std::string role = message.value("role", "");
    if (role == kRoleUser || role == kRoleSystem || role == kRoleTool ||
        role == kRoleLatestReminder || role == kRoleDirectSearchResults ||
        idx >= last_user_idx) {
    } else if (role == kRoleAssistant) {
      message.erase("reasoning");
    } else {
      continue;
    }
    if (retained_count != static_cast<size_t>(idx)) {
      message_array[retained_count] = std::move(message);
    }
    ++retained_count;
  }
  message_array.resize(retained_count);
  return messages;
}

nlohmann::ordered_json normalize_messages(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& tools,
    size_t& prompt_capacity) {
  nlohmann::ordered_json out = nlohmann::json::array();
  out.get_ref<nlohmann::ordered_json::array_t&>().reserve(messages.size() +
                                                          !tools.empty());
  prompt_capacity = 0;
  for (const Message& message : messages) {
    nlohmann::ordered_json item;
    item["role"] = message.role;
    std::string content = get_text_content(message.content);
    prompt_capacity += content.size() + 64;
    item["content"] = std::move(content);
    if (message.reasoning_content.has_value()) {
      item["reasoning"] = message.reasoning_content.value();
      prompt_capacity += message.reasoning_content->size();
    }
    if (message.tool_calls.has_value()) {
      item["tool_calls"] = tool_calls_from_openai_format(
          message.tool_calls.value(), prompt_capacity);
    }
    if (message.tool_call_id.has_value()) {
      item["tool_call_id"] = message.tool_call_id.value();
    }
    out.emplace_back(std::move(item));
  }
  if (!tools.empty()) {
    nlohmann::ordered_json tools_json = nlohmann::json::array();
    tools_json.get_ref<nlohmann::ordered_json::array_t&>().reserve(
        tools.size());
    for (const xllm::JsonTool& tool : tools) {
      prompt_capacity +=
          tool.function.name.size() + tool.function.description.size() + 256;
      nlohmann::ordered_json openai_tool;
      openai_tool["type"] = "function";
      nlohmann::ordered_json function;
      function["name"] = tool.function.name;
      function["description"] = tool.function.description;
      function["parameters"] = tool.function.parameters;
      if (tool.function.defer_loading.has_value()) {
        function["defer_loading"] = *tool.function.defer_loading;
      }
      openai_tool["function"] = std::move(function);
      tools_json.emplace_back(std::move(openai_tool));
    }
    nlohmann::ordered_json sys_msg;
    sys_msg["role"] = kRoleSystem;
    sys_msg["tools"] = std::move(tools_json);
    out.insert(out.begin(), std::move(sys_msg));
  }
  return out;
}

template <bool IsV41>
std::string render_message(const nlohmann::ordered_json& messages,
                           int32_t index,
                           int32_t last_user_idx,
                           const std::string& thinking_mode,
                           bool drop_thinking,
                           const std::string& reasoning_effort) {
  std::string prompt;
  const nlohmann::ordered_json& msg = messages[index];
  std::string role = msg.value("role", "");
  const std::string empty_content;
  const auto content_it = msg.find("content");
  const std::string& content = content_it == msg.end()
                                   ? empty_content
                                   : content_it->get_ref<const std::string&>();

  const auto tools = msg.find("tools");
  const bool has_tools =
      tools != msg.end() && tools->is_array() && !tools->empty();
  const auto response_format = msg.find("response_format");
  const bool has_response_format =
      response_format != msg.end() && !response_format->is_null();

  if constexpr (IsV41) {
    const std::string effort =
        index == 0 && thinking_mode == kThinkingModeThinking
            ? reasoning_effort_prompt_v41(reasoning_effort)
            : std::string();
    if (index == 0 && (!effort.empty() || role == kRoleSystem)) {
      prompt += kSystemSpToken;
    }
    prompt += effort;
  } else {
    if (index == 0 && thinking_mode == kThinkingModeThinking) {
      prompt += reasoning_effort_prompt(reasoning_effort);
    }
  }

  if (role == kRoleSystem) {
    if constexpr (IsV41) {
      if (index > 0) {
        prompt += kSystemSpToken;
      }
    }
    prompt += content;
    if (has_tools) {
      prompt += "\n\n";
      prompt += render_tools<IsV41>(*tools);
    }
    if (has_response_format) {
      prompt += "\n\n";
      prompt += absl::StrReplaceAll(kResponseFormatTemplate,
                                    {{"{schema}", to_json(*response_format)}});
    }
  } else if (role == kRoleDeveloper) {
    if (content.empty()) {
      LOG(FATAL) << "Developer message content is empty";
    }
    prompt += kUserSpToken;
    prompt += content;
    if (has_tools) {
      prompt += "\n\n";
      prompt += render_tools<IsV41>(*tools);
    }
    if (has_response_format) {
      prompt += "\n\n";
      prompt += absl::StrReplaceAll(kResponseFormatTemplate,
                                    {{"{schema}", to_json(*response_format)}});
    }
  } else if (role == kRoleUser) {
    prompt += kUserSpToken;
    if (msg.contains("content_blocks") && msg["content_blocks"].is_array()) {
      bool first_part = true;
      for (const auto& block : msg["content_blocks"]) {
        std::string btype = block.value("type", "");
        if (btype != "text" && btype != "tool_result") {
          continue;
        }
        if (!first_part) {
          prompt += "\n\n";
        }
        first_part = false;
        if (btype == "text") {
          const auto text = block.find("text");
          if (text != block.end()) {
            prompt += text->get_ref<const std::string&>();
          }
        } else {
          const auto tool_content = block.find("content");
          const std::string& tc =
              tool_content == block.end()
                  ? empty_content
                  : tool_content->get_ref<const std::string&>();
          prompt += "<tool_result>";
          prompt += tc;
          prompt += "</tool_result>";
        }
      }
    } else {
      prompt += content;
    }
  } else if (role == kRoleLatestReminder) {
    prompt += kLatestReminderSpToken;
    prompt += content;
  } else if (role == kRoleTool) {
    LOG(FATAL) << "deepseek_v4 merges tool messages into user; please "
                  "preprocess with merge_tool_messages()";
  } else if (role == kRoleAssistant) {
    std::string tc_content;

    const auto tool_calls = msg.find("tool_calls");
    if (tool_calls != msg.end() && tool_calls->is_array() &&
        !tool_calls->empty()) {
      tc_content = render_tool_calls<IsV41>(*tool_calls);
    }

    bool prev_has_task = index - 1 >= 0 && messages[index - 1].contains("task");

    if (thinking_mode == kThinkingModeThinking && !prev_has_task) {
      if (!drop_thinking || index > last_user_idx) {
        const auto reasoning_it = msg.find("reasoning");
        const std::string& reasoning =
            reasoning_it == msg.end()
                ? empty_content
                : reasoning_it->get_ref<const std::string&>();
        prompt += reasoning;
        prompt += kThinkingEndToken;
      }
    }

    bool wo_eos = msg.contains("wo_eos") && msg["wo_eos"].is_boolean() &&
                  msg["wo_eos"].get<bool>();
    prompt += content;
    prompt += tc_content;
    if (!wo_eos) {
      prompt += kEosToken;
    }
  } else {
    LOG(FATAL) << "Unknown role: " + role;
  }

  if (index + 1 < static_cast<int32_t>(messages.size()) &&
      messages[index + 1].value("role", "") != kRoleAssistant &&
      messages[index + 1].value("role", "") != kRoleLatestReminder) {
    return prompt;
  }

  std::string task;
  if (msg.contains("task") && msg["task"].is_string()) {
    task = msg["task"].get<std::string>();
  }

  if (!task.empty()) {
    auto it = task_sp_tokens().find(task);
    if (it == task_sp_tokens().end()) {
      LOG(FATAL) << "Invalid task: " + task;
    }
    if (task != "action") {
      prompt += it->second;
    } else {
      prompt += kAssistantSpToken;
      prompt += (thinking_mode != kThinkingModeThinking) ? kThinkingEndToken
                                                         : kThinkingStartToken;
      prompt += it->second;
    }
  } else if (role == kRoleUser || role == kRoleDeveloper ||
             (IsV41 && role == kRoleSystem && index > 0)) {
    prompt += kAssistantSpToken;
    if (thinking_mode == kThinkingModeThinking &&
        (!drop_thinking || index >= last_user_idx)) {
      prompt += kThinkingStartToken;
    } else {
      prompt += kThinkingEndToken;
    }
  }

  return prompt;
}

template <bool IsV41>
std::optional<std::string> apply_format(
    const std::string& bos_token,
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) {
  const char* stage = "normalize_messages";
  try {
    size_t prompt_capacity = 0;
    nlohmann::ordered_json normalized =
        normalize_messages(messages, json_tools, prompt_capacity);
    std::string thinking_mode = get_thinking_mode(chat_template_kwargs);
    std::string reasoning_effort =
        IsV41 ? resolve_reasoning_effort_v41(
                    get_reasoning_effort(chat_template_kwargs))
              : resolve_reasoning_effort(
                    get_reasoning_effort(chat_template_kwargs));

    stage = "merge_tool_messages";
    normalized = merge_tool_messages(std::move(normalized));
    stage = "sort_tool_results_by_call_order";
    normalized = sort_tool_results_by_call_order<IsV41>(std::move(normalized));

    bool drop_thinking = false;
    if (!normalized.empty()) {
      std::string last_role = normalized.back().value("role", "");
      drop_thinking = (last_role == kRoleUser);
    }

    if (drop_thinking) {
      for (const auto& m : normalized) {
        if (m.contains("tools")) {
          drop_thinking = false;
          break;
        }
      }
    }

    if (thinking_mode == kThinkingModeThinking && drop_thinking) {
      stage = "drop_thinking_messages";
      normalized = drop_thinking_messages<IsV41>(std::move(normalized));
    }

    const int32_t last_user_idx = find_last_user_index<IsV41>(normalized);
    std::string prompt;
    prompt.reserve(bos_token.size() + prompt_capacity);
    prompt = bos_token;

    stage = "render_message";
    for (int32_t idx = 0; idx < static_cast<int32_t>(normalized.size());
         ++idx) {
      if constexpr (IsV41) {
        try {
          prompt += render_message<IsV41>(normalized,
                                          idx,
                                          last_user_idx,
                                          thinking_mode,
                                          drop_thinking,
                                          reasoning_effort);
        } catch (const std::exception& e) {
          LOG(ERROR) << "Failed to apply DeepSeek V4.1 native template (stage: "
                     << stage << ", message index: " << idx
                     << ", role: " << normalized[idx].value("role", "")
                     << "): " << e.what();
          return std::nullopt;
        }
      } else {
        prompt += render_message<IsV41>(normalized,
                                        idx,
                                        last_user_idx,
                                        thinking_mode,
                                        drop_thinking,
                                        reasoning_effort);
      }
    }
    return prompt;
  } catch (const std::exception& e) {
    if constexpr (IsV41) {
      LOG(ERROR) << "Failed to apply DeepSeek V4.1 native template (stage: "
                 << stage << "): " << e.what();
    } else {
      LOG(ERROR) << "Failed to apply DeepSeek V4 native template: " << e.what();
    }
    return std::nullopt;
  }
}

}  // namespace

std::optional<std::string> DeepseekV4CppTemplate::apply(
    const ChatMessages& messages,
    const std::vector<JsonTool>& json_tools,
    const nlohmann::ordered_json& kwargs) const {
  return apply_format<false>(bos_token_, messages, json_tools, kwargs);
}

std::optional<std::string> DeepseekV41CppTemplate::apply(
    const ChatMessages& messages,
    const std::vector<JsonTool>& json_tools,
    const nlohmann::ordered_json& kwargs) const {
  return apply_format<true>(bos_token_, messages, json_tools, kwargs);
}

}  // namespace xllm
