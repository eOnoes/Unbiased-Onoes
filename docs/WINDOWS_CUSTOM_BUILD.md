# Building the preserved Windows custom port

These are manual build instructions, not commands executed by the preservation
task. Use a developer command prompt for the intended installed MSVC toolset,
CMake, Ninja, the pinned llama.cpp submodule, and CUDA 13.3 for SM89. Clone and
submodule acquisition are explicit preparation steps; configure must not fetch
new runtime source. See `CUSTOM_BASELINE.md` for provenance and limits.

From the repository root, a CPU-only frontend/test configuration is:

```bat
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_FRONTEND=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR=third_party/llama.cpp -DGGML_CCACHE=OFF -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /EHsc /utf-8 /D_WIN32_WINNT=0x0A00"
cmake --build build-cpu --target unbiased-strata engine_test windows_child_test literals_test picture_cache_test gguf_reader_test gguf_split_test gguf_geometry_test model_family_test deepseek41_types_test
python -B -m unittest discover -s tools -p "test_deepseek41*.py" -v
```

Set `STRATA_TEST_PYTHON` to the actual Python executable before invoking
`engine_test`; it uses a fake child only. Native reader tests use small temporary
fixtures, and the Python unit suite does not call the model/inference harness.

The historical CUDA build used MSVC 14.50 and nvcc 13.3 with
`--allow-unsupported-compiler`; that workaround is recorded, not endorsed as a
portable supported toolchain. Maya's current successful build uses MSVC 14.44.
Select and qualify one toolchain deliberately before a new build. An equivalent
configuration for the existing custom source, assuming nvcc is on PATH, is:

```bat
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_DEEPSEEK41=ON -DSTRATA_FRONTEND=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=third_party/llama.cpp -DGGML_CCACHE=OFF -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /EHsc /utf-8 /D_WIN32_WINNT=0x0A00" -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler -Xcompiler=/utf-8 -Xcompiler=/bigobj"
cmake --build build-cuda --target strata strata-poly deepseek41_iq_test deepseek41_slot_test deepseek41_kernels_test
```

Building tests is not permission to run GPU/model tests during another model's
session. The IQ1_M fast-chunked prefill numerical failure remains unresolved;
only the qualified `--prefill-chunk 0` window path is admitted. Neither a fresh
compile nor a successful synthetic test establishes model quality or a new
context/checkpoint's qualification.
