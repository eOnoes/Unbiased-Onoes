# DeepSeek V4.1 in UnbiasedStrata

This opt-in CUDA path is distinct from `deepseek4`. It ports official
[Strata revision 8183c6b](https://github.com/Niko1221/Strata/tree/8183c6b67162340c09b67360860177c577a65f2e/src/dsv41),
derived from Project Maya under MIT (notices in `third_party/project-maya`).
It runs inside **UnbiasedStrata's `strata-poly` child**, using the existing
INFO/READY/GEN/PP/T/DONE/STOP/QUIT protocol. No other engine executable is used.

Build with the installed MSVC, Ninja and CUDA toolkit:

```
cmake -S . -B build-cuda -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_DEEPSEEK41=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=<this-tree>/third_party/llama.cpp
cmake --build build-cuda --target strata-poly deepseek41_iq_test deepseek41_kernels_test
```

Portable manual commands and historical compiler caveats are in
[`WINDOWS_CUSTOM_BUILD.md`](WINDOWS_CUSTOM_BUILD.md). Local absolute-path
recipes remain under ignored `evidence/`. CPU-only frontend builds need no
CUDA. The pinned llama library is built as C++17; the frontend remains C++20.

Prepare the exact four DASLab IQ1_M shards and shared IQ4_NL Engram file:

```
py -3.14 tools/deepseek41_prepare.py <bundle-directory> <index-directory-outside-bundle>
```

This checks the bundle's actual metadata, geometry, quant histogram, spans,
source identifiers and hash arrays, then writes a small tensor-offset index,
Engram hash data and tokenizer files. It does **not** copy, merge or requantize
weights. All offsets still refer to the original five GGUFs. The index is not
the Q2_0 expert pack. Full-file weight hashes are not verified by this step.

Launch (all paths explicit; use an available loopback port):

```
unbiased-strata --engine <this-tree>/build-cuda/strata-poly.exe -m <first-shard> --host 127.0.0.1 --port 18141 --alias daslab-deepseek41-iq1-m-unbiased -c 1024 -- --architecture deepseek41 --model <first-shard> --engram <Engram-file> --pack <index-directory> --max-context 1024 --prefill-chunk 0 --dspark none --vision off --mmap-experts on --ssd-cache 0 --expert-profile none
```

The frontend's vocabulary-only adapter preserves `general.architecture`;
the llama library still refuses full `deepseek41` inference. Set
`STRATA_ENGINE_LOG` to save the actual child protocol and its DONE timings.

The port uses the donor's shared KV/index source layers, candidate selections,
0/2/1 compression, hyper-connections, RoPE, Engram hash history and reset logic.
CPU expert products use the pinned ggml-cpu type traits. IQ1_S, IQ1_M, IQ2_XXS
and IQ2_S stay in their original formats. External IQ4_NL tables are mapped
read-only and read in bounded rows, with a bounded cache (`DS4_ENGRAM_CACHE_MB`).
Windows buffered file/mmap reads support the E: exFAT bundle. That is separate
from the standing D:-only rule for Qwen Flash-Next PLE/direct I/O.

Qualification and limits:

- `deepseek41_iq_test --bundle-mix`: all seven actual gate/up/down combinations,
  CPU and GPU, compared with independent ggml decoding and double arithmetic.
- `deepseek41_kernels_test --tokens 1,2,16 --ctx 4096 --reps 20`: numerical
  attention, index selection, routing, hyper-connections, 2/1 compressors,
  Engram gate/dequantization, and bit-exact IQ4_NL row decoding.
- Windows process tests cover quoting/Unicode, bounded asynchronous writes,
  cancellation, read deadlines, liveness, failed launches and job teardown.
- Fast chunked FP16 prefill is refused: the donor IQ1_M test failed its existing
  tolerance on this GPU. Only the tested 16-token window path is enabled.
- Initial integration is CUDA-only. Vision, DSpark, multi-GPU, persistent slots
  and other contexts/checkpoints are not qualified by this exact-bundle run.

Use `tools/deepseek41_run.py` for the bounded resource-instrumented exact-file
test. It refuses to overwrite an earlier `evidence/exact-run`, verifies health
and alias, and computes PPS/TPS only from successful DONE counts/times. Runtime
results and any blockers stay under ignored `evidence/`. A sanitized historical
summary is in [`CUSTOM_BASELINE.md`](CUSTOM_BASELINE.md), and the offline receipt
test uses a small tracked fixture rather than raw local telemetry. The run helper
still has machine-specific defaults and is not a GPU-ownership supervisor; do
not run it while Maya or another inference backend is active. Compilation or a
passing synthetic test alone is not proof of exact-model inference.
