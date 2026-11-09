include(cc_binary)
include(cc_library)

# Base class (independent of concrete implementations)
cc_library (
  NAME
    chat_template
  HDRS
    chat_template.h
  SRCS
    chat_template.cpp
  DEPS
    :common
    nlohmann_json::nlohmann_json
)

cc_library (
  NAME
    jinja_chat_template
  HDRS
    jinja_chat_template.h
  SRCS
    jinja_chat_template.cpp
  DEPS
    :chat_template
    :minja
    :tokenizer
    nlohmann_json::nlohmann_json
    glog::glog
)

cc_library (
  NAME
    deepseek_v32_cpp_template
  HDRS
    deepseek_v32_cpp_template.h
  SRCS
    deepseek_v32_cpp_template.cpp
  DEPS
    :chat_template
    :tokenizer
    nlohmann_json::nlohmann_json
    glog::glog
    absl::strings
)

cc_library (
  NAME
    deepseek_chat_template
  HDRS
    deepseek_chat_template.h
  SRCS
    deepseek_chat_template.cpp
  DEPS
    :chat_template
    :tokenizer
    glog::glog
    absl::strings
)

cc_library (
  NAME
    deepseek_v4_cpp_template
  HDRS
    deepseek_v4_cpp_template.h
  SRCS
    deepseek_v4_cpp_template.cpp
  DEPS
    :deepseek_chat_template
    :tokenizer
)

cc_library (
  NAME
    deepseek_v41_cpp_template
  HDRS
    deepseek_v41_cpp_template.h
  SRCS
    deepseek_v41_cpp_template.cpp
  DEPS
    :deepseek_chat_template
    :tokenizer
)

# Factory (depends on all implementations)
cc_library (
  NAME
    chat_template_factory
  HDRS
    chat_template_factory.h
  SRCS
    chat_template_factory.cpp
  DEPS
    :chat_template
    :config
    :jinja_chat_template
    :deepseek_v32_cpp_template
    :deepseek_v4_cpp_template
    :deepseek_v41_cpp_template
    nlohmann_json::nlohmann_json
    glog::glog
)

# Request-path hop 2: the chat-template render LLMRequestFactory performs on
# Qwen2.5 Jinja and DeepSeek V3.2/V4 native templates. Compare Release builds.
cc_binary(
  NAME
    chat_template_benchmark
  SRCS
    chat_template_benchmark.cpp
  DEPS
    :jinja_chat_template
    :deepseek_v32_cpp_template
    :deepseek_v4_cpp_template
    :deepseek_v41_cpp_template
    nlohmann_json::nlohmann_json
    benchmark::benchmark
    benchmark::benchmark_main
)
