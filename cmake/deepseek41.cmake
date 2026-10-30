# Official Strata 8183c6b67162340c09b67360860177c577a65f2e, Project Maya MIT.
# Kept separate from deepseek4; registered in strata-poly's existing child protocol.
option(STRATA_DEEPSEEK41 "Build the distinct DeepSeek V4.1 GGUF implementation" OFF)
if(STRATA_DEEPSEEK41)
  if(NOT STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP OR NOT TARGET strata-poly)
    message(FATAL_ERROR "Initial DeepSeek41 port requires the CUDA strata-poly target")
  endif()
  set(_ds src/dsv41)
  add_library(strata_deepseek41 STATIC
    ${_ds}/ds4_main.cpp ${_ds}/ds4_model.cu ${_ds}/ds4_kernels.cu ${_ds}/ds4_vision.cu
    ${_ds}/ds4_gguf.cu ${_ds}/ds4_gguf_cpu.cpp ${_ds}/ds4_iq.cu ${_ds}/ds4_iq_cpu.cpp
    ${_ds}/ds4_tq.cu ${_ds}/ds4_tq_cpu.cpp ${_ds}/ds4_gg_cpu.cpp ${_ds}/ds4_packinfo.cpp
    ${_ds}/vendor/exl3.cu ${_ds}/vendor/sampler.cu ${_ds}/vendor/moe_win.cu
    ${_ds}/vendor/exl3_cpu.cpp ${_ds}/vendor/exl3_legacy.cu)
  target_include_directories(strata_deepseek41 PUBLIC include PRIVATE
    ${STRATA_GGML_DIR}/ggml/src ${STRATA_GGML_DIR}/ggml/include third_party/dsv41-ggml)
  target_compile_definitions(strata_deepseek41 PRIVATE STRATA_DSV41_GGML_CPU=1
    DS4_BUILD_ID="unbiased-2049dd2-dirty-strata-8183c6b")
  target_link_libraries(strata_deepseek41 PUBLIC CUDA::cudart CUDA::cublas ggml-cpu ggml-base Threads::Threads)
  # Preserve donor rounding: no --use_fast_math for this architecture.
  target_compile_options(strata_deepseek41 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  target_sources(strata-poly PRIVATE src/poly/deepseek41_dispatch.cpp)
  target_compile_definitions(strata-poly PRIVATE STRATA_DEEPSEEK41=1)
  target_link_libraries(strata-poly PRIVATE strata_deepseek41)
  add_executable(deepseek41_iq_test tests/dsv41/iq_expert_test.cpp)
  target_include_directories(deepseek41_iq_test PRIVATE include ${STRATA_GGML_DIR}/ggml/include)
  target_link_libraries(deepseek41_iq_test PRIVATE strata_deepseek41)
  add_executable(deepseek41_slot_test tests/dsv41/slot_test.cpp)
  target_include_directories(deepseek41_slot_test PRIVATE src/dsv41)
  add_executable(deepseek41_kernels_test tools/dsv41_lab/ds4_lab.cu)
  target_link_libraries(deepseek41_kernels_test PRIVATE strata_deepseek41)
endif()
