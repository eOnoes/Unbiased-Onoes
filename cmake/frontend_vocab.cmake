# Build a narrowly patched copy; the pinned llama.cpp checkout stays untouched.
# Its public API rejects unknown architectures before its vocab_only fast path.
file(READ "${STRATA_GGML_DIR}/src/llama-model.cpp" _vocab_src)
set(_vocab_anchor "    if (arch == LLM_ARCH_UNKNOWN) {\n        throw std::runtime_error(\"unknown model architecture: '\" + ml.get_arch_name() + \"'\");\n    }")
string(FIND "${_vocab_src}" "${_vocab_anchor}" _vocab_at)
if(_vocab_at EQUAL -1)
  message(FATAL_ERROR "llama-model.cpp vocabulary-only patch anchor changed; review the pin")
endif()
string(REPLACE "${_vocab_anchor}"
  "    if (arch == LLM_ARCH_UNKNOWN && params.vocab_only && ml.get_arch_name() == \"deepseek41\") {\n        return new strata_vocab_only_model(params);\n    }\n${_vocab_anchor}"
  _vocab_src "${_vocab_src}")
string(REPLACE "static llama_model * llama_model_mapping"
  "#include \"vocab_only_model.inc\"\n\nstatic llama_model * llama_model_mapping" _vocab_src "${_vocab_src}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/llama-model-vocab.cpp" "${_vocab_src}")
get_target_property(_llama_sources llama SOURCES)
list(REMOVE_ITEM _llama_sources llama-model.cpp)
set_property(TARGET llama PROPERTY SOURCES "${_llama_sources}")
target_sources(llama PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/llama-model-vocab.cpp")
target_include_directories(llama PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/frontend")
