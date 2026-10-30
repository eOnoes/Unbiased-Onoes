# One Strata entry point for the actual Unbiased build and Maya

Design investigation prepared 2026-10-30; sanitized for source control. See [CUSTOM_BASELINE.md](CUSTOM_BASELINE.md) for the subsequent preservation commit and checks. `<CUSTOM_ROOT>`, `<MAYA_ROOT>`, and other angle-bracket paths are local checkout placeholders. Investigation and design only. No source tree was edited, pulled, reset or built. No engine, inference request, benchmark, GPU probe, model load, or process restart was run. Model weight payloads were not read or hashed. Existing source, small configuration files, build metadata and saved receipts were inspected. The running GLM service at `127.0.0.1:8088` was left alone.

## Recommendation

**Build one Strata product around the custom Unbiased C++ frontend and model-aware child backends. Keep the current Maya GLM runner as an isolated backend initially. Preserve the already integrated custom DeepSeek41 runner, and retain the existing Qwen runners.** Do not start by merging their GPU memory managers, kernels or cache formats.

The important discovery is that the actual custom work is already partway through this combination. It is in `<CUSTOM_ROOT>`, not the clean upstream checkout or the similarly named earlier Windows build attempt. That custom worktree has a working Win32 process adapter and a Maya-derived DeepSeek V4.1 implementation linked inside its own `strata-poly` child. Its saved local run proves a short exact-file generation; it does not establish coding quality or long-context readiness.

The next useful integration is therefore **the currently working Maya GLM runner behind that C++ frontend**, plus reliable metadata selection, capability reporting and single-GPU process ownership. This can give the owner one launcher/API while honestly retaining model-specific runtimes. A unified experience does not require all families to use the same attention/cache implementation.

There is a lower-risk first delivery if preserving today's GLM behavior is the immediate priority: put the common Strata supervisor in front of **intact backend services over private loopback HTTP**, retaining Maya's own Python tokenizer/template/parser alongside its runner and the custom Unbiased frontend alongside its children. That avoids treating similar token protocols as full frontend compatibility. It changes the GLM channel's serving-policy description (Python remains there), so label it explicitly. The C++-only GLM channel is the recommended destination after prompt/token/protocol parity, not a prerequisite for the first unified selector. The intact-service first step preserves the working frontend pairing; the later C++ step can reuse the custom Win32 transport.

## 1. Exact local identities

| Role | Checkout / build | Verified identity and state |
|---|---|---|
| Actual custom Unbiased development tree | `<CUSTOM_ROOT>` | Investigation snapshot: registered Git worktree; branch `codex/unbiased-deepseek41`; base `2049dd2a292ffc38fd1cc006be04664ba15de5bb`; 8 modified tracked files plus substantial untracked implementation, tests, donor files and evidence. **That original base commit alone did not identify the custom build; the preservation commit now records its source.** |
| Original Unbiased repository | `<BASE_ROOT>` | `main`, same base commit; origin `https://github.com/robertkarl/UnbiasedStrata.git`; only untracked `SECURITY.md`. `git worktree list --porcelain` led to the actual custom worktree above. |
| Earlier Windows candidate | `<EARLIER_WINDOWS_CANDIDATE>` | Detached at the same base; origin is the E: local repository; tracked tree clean. Three Windows configure/build directories exist; no final runtime executables were found there. **Do not substitute this for the custom worktree.** |
| Unbiased llama.cpp submodule | `third_party/llama.cpp` | `a6de9af3b19b5e3a5397f2c835a29a13c37664bd`, robertkarl/llama.cpp branch `UnbiasedStrata`; based on ggml-org `3cf03257f219afbe7334045ff7c6a06ac68c627d`, with tool-parser fixes. |
| Installed Maya | `<MAYA_ROOT>` | `VERSION` 1.1.41; `main`, `9030b73661b41d947117f7161fc6a11e14a5107a`; origin `https://github.com/mw00/project-maya.git`; Git status clean. |
| Current GLM executable | Maya `build\maya.exe` | Present, 87,301,632 bytes. `build\MAYA-BUILD.json`: CUDA 13.3, SM89, target **maya only**, source fingerprint `1b0aef122e8955e0`; local pinned llama.cpp source `3cf03257...`. `maya-ds4.exe` was not present in this build directory. |
| Custom executables | Custom `build-cpu\unbiased-strata.exe`; `build-cuda\{strata-poly,strata}.exe` | All present. Saved continuation build identity records hashes and source provenance; these binaries were not executed or rehashed in this investigation. |

The custom Git changes are `CMakeLists.txt`, `frontend/{engine.hpp,main.cpp,strata_context.cpp}`, adapter tests, `include/strata/artifact/gguf_reader.hpp`, and `src/poly/main.cpp`. Added work includes `frontend/windows_child.hpp`, `cmake/{deepseek41,frontend_vocab}.cmake`, `src/poly/deepseek41_dispatch.cpp`, `src/dsv41/`, `include/strata/dsv41/`, exact-bundle preparation/audit tooling, numerical tests and evidence. None may be discarded during integration.

I independently checked the 62 small source files named in `evidence/continuation-build-identity.json`: **62/62 match the recorded source hashes**. Supporting result: [managed baseline provenance](CUSTOM_BASELINE.md). This corroborates the implementation snapshot behind the saved run, without rerunning it. It does not authenticate full model files or reproduce runtime results.

Earlier review copies under `<OLDER_REVIEW_COPIES>` include UnbiasedStrata 1 at `b124b59...` and upstream Strata at `296b59...`; neither is the requested current custom engine. The untracked E: `SECURITY.md` describes an older `46c46ec...` audit and older pins; its claims must not be treated as certification of the current custom build.

## 2. What “Unbiased” actually means here

The base is **UnbiasedStrata 2**, derived from PolyStrata `v0.1.45-poly.1`, commit `d0cd84dc6bd009d1d9e2191e4e28cc76b80f1ee3`, through import `0197ae86`. It replaces the Python installer/server/web layer with a smaller C++ server built from pinned llama.cpp HTTP, chat, template and parser code. It removes runtime installers/downloaders, self-update, server-side MCP/tool execution and the web UI. Offline Python preparation/test tools remain.

This is an architectural/offline-serving policy, **not a guarantee of uncensored weights, zero numerical error, unquantized execution, or zero expert approximation**. In the clean base, `git diff 0197ae86 HEAD -- src include` contains only a CLI `@FILE` input change in `src/poly/main.cpp`; it did not replace the inherited inference math. The public provenance description agrees with this local inspection: [Unbiased relationship to upstream](https://github.com/robertkarl/UnbiasedStrata/blob/2049dd2a292ffc38fd1cc006be04664ba15de5bb/UPSTREAM.md).

Preserve specifically:

- The pinned C++ tokenizer/template/parsers, including schema-order-independent tool arguments. Local `UPSTREAM.md:88` onward documents the two llama.cpp parser commits; literal handling and image-cache tests also matter for multi-turn agents.
- No weight replacement or runtime fetching; tools returned to the client rather than executed by the serving engine.
- The custom Win32 transport, unchanged original-file offsets, I64 bounds checking and distinct V4.1 graph.
- Honest advertised capabilities: the existing frontend warns that requested grammars cannot constrain the child sampler (`frontend/strata_context.cpp:337` in the base; nearby in the custom tree). Do not advertise enforced JSON schema or required tool calls merely because the prompt/parser understands them.

There are inherited approximation switches. Base `src/program/generate.cpp:9380` onward can enable route-tail expert skipping by default under certain Qwen cache/hardware conditions; `src/core/expert_source.cpp:3850` onward omits those contributions. `strata-poly` has explicit `Options::tail_skip`, default 0 (`src/poly/engine.hpp:78`; `engine.cpp:2158`). A full-computation profile must set `STRATA_ROUTE_TAIL_SKIP=0` and reject overrides that change it. The fork's name is insufficient evidence that this has already happened.

## 3. What is already combined in the owner's custom tree

The custom V4.1 port uses official Strata donor `8183c6b67162340c09b67360860177c577a65f2e`, whose V4.1 engine is Maya-derived. It is **not the currently installed Maya v1.1.41 source**. Copyright notice records Maya v1.1.17 origins and later changes through v1.1.26; the exact donor pin is the useful reproducibility boundary.

Implemented seams:

| Seam | Exact local reference | Meaning |
|---|---|---|
| Distinct opt-in build | `cmake/deepseek41.cmake:3` | `STRATA_DEEPSEEK41=ON`, CUDA-only, links `strata_deepseek41` into `strata-poly`; preserves donor rounding, no `--use_fast_math` for this architecture. |
| Explicit dispatch | `src/poly/main.cpp:1114`; `src/poly/deepseek41_dispatch.cpp:9` | `--architecture deepseek41`; checks backbone/Engram architecture, matching source identifiers and tensor bounds. This is not automatic model detection yet. |
| Native Windows child | `frontend/windows_child.hpp:20`; `frontend/engine.hpp:83` | UTF-8/UTF-16 path handling, CRT quoting, restricted inherited handles, hidden child, kill-on-close job, asynchronous bounded writes, read deadlines and child teardown. Reuse this work. |
| Vocabulary-only architecture | `cmake/frontend_vocab.cmake:1`; `frontend/vocab_only_model.inc` | Allows deepseek41 vocabulary/template loading while full llama inference remains refused. It currently generates a patched build copy of pinned llama-model.cpp; it does not edit the submodule. |
| Exact artifact adapter | `tools/deepseek41_prepare.py:1`; `tools/deepseek41_bundle_audit.py` | Checks the exact five-file bundle; produces metadata/tensor offsets and Engram hashes; no merge, requantization, or weight copy. |
| Qualified prefill restriction | `src/poly/deepseek41_dispatch.cpp:13` | Rejects nonzero `--prefill-chunk`; forces window prefill because the IQ1_M FP16 chunk test failed its existing tolerance. Must survive all integration work. |
| Distinct kernel namespace | `include/strata/dsv41/exl3.hpp:27` | Already separates donor EXL3 as `strata::dsv41k::exl3`; shows why adding more shared symbols requires review. |

Saved evidence is ordered: `evidence/RESULTS.md` is the earlier blocked audit; **`CONTINUATION-RESULTS.md` supersedes its runtime status**, and `CODEX-CLEANUP-REVIEW.md` corrects units and validates result parsing. Do not quote only the earlier failure or only the later headline.

The preserved local result is `DONE 64 256 38192.0 7592.7 length 0 0 0 0 0`: **6.703 prompt tokens/s, 8.429 generated tokens/s**, 1,024 context, greedy, 256 input / 64 output, no speculation. Startup was 842.49 seconds. It was a repeated-text raw completion, not a coding-quality test. Weight caches were warmed during startup; OS cache was uncontrolled. Logs record 3,277/15,360 experts in VRAM and substantial CPU expert work. The frontend's 8.304 TPS uses 63 timed intervals; it is not a separate run or regression.

Saved tests include five CPU regressions, seven protocol cases, Win32 transport cases, mixed expert numerical tests and T=1/2/16 attention/routing/compressor/Engram checks. These are historical receipts, not tests rerun today. Vision, DSpark, multi-GPU, persistent slots, larger contexts and end-to-end answer quality remain unqualified for this port.

## 4. The integration boundary

Both applications already have the same fundamental process contract:

```text
start: backend.exe --serve <backend-specific arguments>
out:   INFO key=value ...
       READY <actual_context> stop [other capabilities]
in:    GEN <max_new> [sampling keys] <token_ids>
out:   PP / REUSED / RESUME ...
       T <token_id>
       DONE <counts, timings, finish, optional extension fields>
in:    STOP / QUIT
```

References: custom `frontend/engine.hpp:3`, `:111` onward and `:383` onward; installed Maya `serve/server.py:787`, `:811`, `:1224`; Maya GLM `src/program/generate.cpp:991`; Maya DeepSeek `src/ds4/ds4_main.cpp:7`.

**Protocol similarity is a promising seam, not proven interchangeability.** The custom parser expects untagged single-sequence token/DONE lines. Current Maya supports request IDs, pause/fairness, multi-sequence decoding, extra STAT/VRAM/LP lines, per-request controls and richer reasoning handling. Start with one sequence and `--parallel 1`; explicitly negotiate/validate the implemented subset. Do not hand an unsupported key to an engine and hope it ignores it safely.

Current frontend defects to address for generalization:

1. `frontend/strata_context.cpp:1265` infers `engine_is_poly` from the executable filename. It affects sampling-key forwarding (`:389`) and public identity. Replace with an explicit backend descriptor/capabilities; renaming Maya to “strata” is not an adapter.
2. `frontend/main.cpp:198` starts the child before vocabulary loading at `:217`. Introduce CPU-only model/admission inspection before spawning any GPU process.
3. `gen_header` clamps top-k to 64 and suppresses some keys by its poly Boolean. Validate sampling support per backend; preserve omission/default semantics, seed, stop IDs and penalties. Match Maya's current temperature 1.0 / top-p 0.95 / reasoning max deliberately rather than inheriting unrelated frontend defaults (`min_p` is 0.05 unless set).
4. Frontend startup, cancellation, schema handling and health must be truthful for the selected backend. `READY` context, not requested CLI context, is the allocation fact. Production admission should reject unexpected context reduction unless the user profile explicitly permits it.

## 5. Proposed one-entry-point architecture

```text
Strata launcher / model selector / one public localhost API
    |
    +-- metadata admission + explicit registry + one GPU lease
    |
    +-- exactly one C++ Unbiased frontend worker
           |
           +-- qwen4exp -> existing custom strata child
           +-- qwen35moe -> existing custom strata-poly child
           +-- deepseek41 + exact DASLab bundle -> existing custom strata-poly V4.1 dispatch
           +-- glm5-next + Maya pack -> pinned Maya v1.1.41 maya child
           +-- other exact family/format -> only a separately qualified adapter
```

The supervisor/selector is proposed, not present today. Keep worker and GPU child separate; no GPU libraries need to be loaded by the supervisor. A worker owns templates/tokenization and the line-protocol child. A stable public API can proxy to a private loopback worker while it reports loading/switching/unavailable. Alternatively, first deliver a CLI launcher with an explicit restart gap on the same port; that is a smaller initial gate but must not be presented as seamless hot switching.

Use the existing Win32 job transport for worker/child lifetime; test nested-job ownership if the new supervisor also uses a job. Do not have two independent supervisors restart the same child. No automatic attachment or takeover of the current Maya process: its Python server owns it. Moving GLM behind the C++ frontend is a later, explicitly scheduled stop/start after the owner finishes testing.

The C++ frontend can start `maya.exe --serve --glm-pack ...` directly, preserving Unbiased's no-Python inference path. A separate optional browser client can talk to the shared API. Reusing Maya's complete Python server would preserve its dashboard and extra APIs fastest, but changes the meaning of the Unbiased serving policy; treat that as a distinct transitional option, not the recommended final design.

Maya currently exposes Chat Completions, Responses, Anthropic Messages and a dashboard (`serve/server.py:6` and `serve/responses.py`). Custom Unbiased explicitly registers Chat Completions and Completions plus health/models/tokenizer endpoints (`frontend/strata_context.cpp`, route table at its end); it does not register equivalent Responses/Anthropic routes. **API parity and dashboard parity are separate deliverables.** Verify what the owner's overnight builder client actually uses. Return a clear unsupported endpoint/capability error until a needed dialect has been implemented and tested.

## 6. Detection and compatibility

Detection must read actual GGUF metadata/tensor directories or safetensors headers/config, without loading weights. Build a registry keyed by architecture, geometry, quant-type histogram, required tensor names, tokenizer/template identity, pack schema and model source/revision. Validate all shard indices, matching sources, table formats, shapes and byte bounds before starting CUDA. Filename and friendly model name are display hints only.

For packs, retain source identity: Maya's current pack `sources.json` lists filenames, sizes and modification times, which is useful invalidation data but not cryptographic identity. A new shared model manifest should also record known publisher digests, the exact preparer/version and metadata fingerprint. Full artifact verification is a separate offline task; it was not performed here. Do not assume two `index.txt` files with “v3” are interchangeable across forks.

Offer `--backend` / `--architecture` override for ambiguous metadata, but require a compatible registry entry. An override must never authorize interpreting deepseek41 as deepseek4, ignore a shape/quantization mismatch, or turn an unqualified artifact into a supported one. Refuse unknown combinations with the observed keys and missing requirements.

| Precise model / artifact | Existing evidence | Initial routing recommendation and remaining gate |
|---|---|---|
| GLM-5.3-Flash Maya-S24, three original GGUF shards plus current pack | Current installed Maya configuration and startup logs; the owner reports 21–24 TPS in active testing. Not benchmarked here. | `maya.exe --glm-pack`; retain 65,536 context and current placement. C++ tokenizer/template/protocol pairing still needs qualification. Do not assert native `strata-poly` accepts this exact pack/quant mixture. |
| GLM-5.3-Flash GSQ-RCO 3.0bit-q4kattn GGUF | Base fork documents remote-machine acceptance and numerical comparisons, `docs/MODEL-FILES.md`. | Existing `strata-poly` optional registered profile after local validation; not interchangeable with Maya-S24. |
| DeepSeek-V4.1-Flash-1017 DASLab IQ1_M, **four backbone shards plus external IQ4_NL Engram** | Actual custom local 1K-context 256/64 receipt and numerical gates described above. Backbone contains mixed IQ1_S/IQ1_M/IQ2_XXS/IQ2_S and other tensor types, not only IQ1_M. | Preserve custom `strata-poly --architecture deepseek41`, window prefill only; answer quality and useful contexts still need testing. |
| DeepSeek-V4.1-Flash-1017 Maya-S / Maya-XS GGUF or supported EXL3 | Installed Maya source has packers and separate `maya-ds4` target. No `maya-ds4.exe` in the current build; no local inference qualification in this review. | Proposed separately built Maya DS4 adapter. Current packer requires `deepseek41`, handles IQ4_NL/MXFP8/Q6_K tables and multiple expert types; source coverage is not proof this exact DASLab bundle is faster or numerically fixed. |
| DeepSeek-V4-Flash-0731 UD-IQ2_XXS, `deepseek4` | Base fork documented acceptance on other hardware. | Existing `strata-poly` deepseek4 class; separate profile and validation. No V4.1 alias. |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, `qwen4exp`, pack + separate MTP | Base fork's documented tested configuration. | Preserve `strata` path; local custom Windows binaries exist, but original-family inference was not qualified by the DeepSeek continuation. |
| the owner's Flash-Next GSQ-RCO IQ3_S / Coder variants | Existing local model/pack/test folders found; older upstream `0.1.44.1` release executable exists in `<OLDER_QWEN_RELEASE_BUILD>`. It is not the custom Unbiased binary. | Exact artifacts require custom-loader, format, PLE, prompt, cache and speculation checks before declaring support. |
| Qwen3.6-35B-A3B UD-Q3_K_M / UD-IQ4_XS GGUF with MTP, `qwen35moe` | Base fork documents exact-file acceptance and numerical findings. | Existing `strata-poly` path; local Windows family regression and context qualification still required. FP8/safetensors does not follow from GGUF support. |
| Qwen3.8-Flash-Next CYBER-FROST Q5_K_M GGUF | Maya source ships one exact pinned `tools/qwen/download.json`; text-only path through `maya`. | Optional Maya Qwen backend after tests. It is a fine-tuned model/artifact, not a substitute for the owner's original weights. |
| Qwen3.8-27B dense, other Qwen/DeepSeek/GLM generations, arbitrary FP8/safetensors | Not established by the inspected runners. | Unsupported until a correct exact-family adapter is qualified. A separately pinned llama.cpp or another backend could later join the same registry; “all Qwen” is not an existing capability. |

Exact code anchors: custom `src/poly/engine.cpp:984` recognizes `glm5-next`/`glm5next`, `qwen35moe`, `gemma4`, `deepseek4`, `lfm2moe`; custom `src/poly/main.cpp:1114` adds deepseek41; Maya `tools/iq_pack.py:99`, `:360` recognizes qwen4exp/GLM aliases; `tools/ds4/pack_gguf.py:429`–`:433` distinguishes deepseek41 and sidecar Engram. Canonicalizing the two GLM spellings is safe only while retaining their corresponding metadata-key prefixes.

## 7. Memory, files and caches on this machine

Use the owner's stated RTX 4080 Super with 32 GB, Ryzen 7900X and 128 GB RAM. Retain the working 48 GiB pagefile as an existing environment setting, not a new capacity claim. Runtime admission must examine available physical RAM **and commit**, pinned memory, fixed allocations, context buffers and reserve. Pagefile capacity does not make page-locked experts freely pageable or turn storage into RAM bandwidth.

Current GLM startup evidence (`maya-maya-s24.log:947` onward): 65,536 context; 32,768 DSA positions resident, 0.74 GB pinned latent cache; about 5.24 GB dense weights, 0.66 GB state/KV, 22.38 GB expert pool and 66.04 GB pinned RAM tier in that startup. The logs' GB convention must be retained rather than silently calling those GiB. The warmed expert set is 86.0 GB. These are log observations from an existing start, not a fresh resource sample or worst-case budget.

The saved exact DASLab bundle totals 231,316,411,072 bytes (215.43 GiB), with about 103 GiB of external Engram tables. Its qualified path reads bounded rows with a 64 MiB cache; it never loads the full table. The saved run used E: exFAT buffered I/O and had a 14-minute startup. This is historical evidence, **not the desired new storage layout**. The requested future layout puts models and disk-backed tables on `D:\models`. Plan a separately verified move/copy only after checking actual free space; do not copy 215 GiB based on stale capacity receipts. No files were moved here.

Qwen Flash-Next PLE/direct-I/O files belong on D: as well. `docs/MEMORY_LAYOUT.md` distinguishes arena (pinned whole expert set), resident complement and mmap CPU miss paths. CPU/GPU placement changes arithmetic reduction paths; page-cache warmth can hide storage costs. Record anonymous/pinned/commit, mapped file cache, VRAM fixed/cache/scratch, logical reads and physical storage measurements separately.

All cache state is **backend-specific**: Maya GLM DSA/KDA recurrent state and streamed latent cache; Qwen GDN/QSA/PLE state; DeepSeek41 shared compressed KV/index state and Engram history. A shared frontend may save conversation messages, but may not reuse raw KV/session blobs across backends, models, quantizations or incompatible builds. Namespace caches by full model/backend/preparer/context/precision identity, validate on reuse, and start clean on a backend switch.

## 8. Fidelity policy and qualification

Define two distinct notions: (a) preserving the selected quantized checkpoint's full computation, and (b) measuring that quant's quality against a higher-precision reference. A wrapper can preserve (a); it cannot restore bits already lost in Maya-S24 or IQ1_M. No speed result here establishes overnight-builder intelligence.

Recommended default is a named full-computation profile, with effective settings logged and exposed. Explicitly disable expert-drop controls in both aliases, clear conflicting inherited environment and prohibit request-level bypass:

- Strata `STRATA_ROUTE_TAIL_SKIP=0`; `strata-poly --tail-skip 0`; audit route-resident draft bias/control vectors and leave experimental model-changing settings off.
- Maya GLM: trunk keep 8 with no per-layer reduction map; disk skip weight 0 / disk skip rank 8; miss skip weight 0. Exact code is `src/core/glm_fast_path.cu:1902`, `:5593`; docs list request-level `strata_tune` overrides.
- Maya/custom DeepSeek: disk/RAM expert-drop weights off; keep the custom fast-prefill refusal. Maya `src/ds4/ds4_model.cu:1721` and custom `src/dsv41/ds4_model.cu:1048` describe lost route contributions explicitly.

Maya aliases every `MAYA_*` to `STRATA_*`, with MAYA values winning (`include/strata/platform/env_alias.hpp:20`). Sanitizing only one prefix is insufficient. The older custom DS4 additionally uses `DS4_*`; descriptors must name all relevant aliases. Profile/environment construction should be explicit rather than inherit unrelated tuning from a shell.

Do not confuse dead-work optimizations with approximate expert pruning. For example, Maya GLM prefill tail skip updates caches and omits unread terminal outputs when no MTP/split consumer exists (`src/core/glm_prefill.cu:2120`); its source says decode separately computes the final prompt token. Maya DeepSeek prompt skip mode 1 claims dependency-preserving work removal; mode 2 changes the answer (`docs/SETTINGS.md:705`). Test state/logit equivalence for the supported profile rather than declaring every option containing “skip” either safe or unsafe.

**Registry exclusion:** the known-failing DeepSeek41 IQ1_M fast-chunked FP16 prefill combination is denied even when architecture metadata matches. Only window prefill is qualified until the original numerical test passes without relaxing its tolerance.

Qualification must include:

1. **Adapter fidelity first:** capture a fixed set of messages/tools to rendered prompt bytes and token IDs; compare existing frontend with the new C++ frontend. Check BOS/EOS, assistant continuation, reasoning on/off/max, tool argument ordering, literal control-token text and multi-turn tool results. Different tokenizer/templates can cost quality without any kernel error.
2. **Same-engine parity:** identical backend binary, checkpoint, context, quantized KV, sampling and fixed placement; raw child tokens versus API tokens, counts, stop reasons, cancellation and cache reuse. Greedy repeatability with placement pinned; distinguish near-tie floating-point changes from changed routes.
3. **Numerical tests:** teacher-forced logits/distributions on the exact weights against a qualified reference, CPU versus GPU experts, forced cache misses, warm/cold tiers, recurrent rollback, chunk boundaries and supported speculative windows. Use the inherited correctness harness and retained custom numerical tests. Investigate failures; do not loosen a tolerance to claim a pass.
4. **Builder quality:** held-out repo edits, tests that actually execute, multi-file tool loops, long-context retrieval/dependency tasks, recovery from a failed tool and completion under realistic output limits. Compare task success, build/test success, retry counts and elapsed completion time before TPS. Avoid accepting the repeated-text DeepSeek receipt as a quality pass.
5. **Performance after correctness:** TTFT, prompt speed at representative 1K/8K/32K/64K input, decode speed, successful task time, p50/p95 latency, peak resources, misses/read traffic, warm versus cold startup, cancellation latency and sustained overnight stability. Test larger context only after memory admission. Use paired runs of the same file and precision; never compare distinct quants/fine-tunes as engine speed alone.

Speculation can be an opt-in later gate; verify target distributions, rollback and observed answer quality. Existing Unbiased receipts already report non-bitwise-inert windows and cache-placement effects. A policy must say whether reproducible bits or reference-tolerant full computation is required; “exact” without that definition is misleading.

## 9. Build, dependency and license boundaries

Process isolation avoids the immediate ABI/kernel collision problem. Maya and Unbiased both define `strata_*` libraries/namespaces, compile selected ggml CUDA sources and have separate expert planners/samplers. A single CMake `add_subdirectory` of both whole projects would duplicate targets/symbols and couple incompatible state assumptions. Custom dsv41 has already namespaced donor kernels; use that as evidence of the work a deeper import entails.

Both current builds use CUDA 13.3 / SM89, but toolchains differ: Maya's cache names VS2022 MSVC 14.44; custom's cache names VS18 MSVC 14.50 and its saved command uses `--allow-unsupported-compiler`. The custom frontend fixes pinned llama C++17 versus frontend C++20 `char8_t` issues. Do not flatten global compiler flags or rebuild it with Maya's flags casually. Private DLL paths per worker/backend, matching runtime/static CRT linkage and driver requirements need clean-machine verification. Maya's CMake uses static CUDA runtime on Windows, while custom dsv41 links CUDA::cudart; preserve separate-process linkage initially.

Pinned ggml base is close: Maya uses ggml-org `3cf03257`; Unbiased submodule is that base plus parser changes. This helps frontend interoperability but **does not prove newer Maya tensor-type extensions, packed formats or CUDA kernels can share the old libraries**. Maya has additional ggml patches/vendor headers and newer EXL3 half-bit trellis support. Preserve donor pin, namespace and format version independently.

The custom `cmake/frontend_vocab.cmake` generates a patched copy at configure time. That is explicit existing custom behavior and conflicts with the base project's stated preference against configure-time replacement. For a reproducible release, carry the vocabulary-only change as an auditable pinned submodule/fork commit or hash and record the generated source with a fail-closed anchor. This is a provenance decision, not a reason to discard working custom code.

Retain MIT notices for Strata, Unbiased, Maya and ggml/llama.cpp; Maya explicitly permits modification/merging subject to retaining notice ([pinned Maya license](https://github.com/mw00/project-maya/blob/9030b73661b41d947117f7161fc6a11e14a5107a/LICENSE)). The custom donor's `third_party/project-maya/{LICENSE,NOTICE.md}` and dsv41-ggml license already record provenance. Model licenses are separate from engine licenses; use exact artifact/revision licenses (Maya includes Qwen Community and DeepSeek license files). An engine merger does not grant additional weight redistribution rights. No license-driven dependency replacement is needed for this local design; distribution requires a complete notice/artifact inventory.

## 10. Single-GPU lifecycle and failure rules

Proposed supervisor states: `idle -> validating -> stopping/draining -> releasing -> loading -> ready -> draining`, with explicit `failed`. An atomic Windows named mutex/lease records ownership for all managed backends. Refuse startup while an externally owned inference process occupies the target profile/device; do not kill it. The currently running GLM session remains externally owned during this investigation.

On switch: stop admissions; let accepted work finish by default; require explicit cancellation policy for active work; send STOP and drain to DONE if cancelling; QUIT; wait for the owned child and worker to exit; verify cleanup and available resources before starting the next backend. Device-wide GPU accounting can be observed later, but must not be misrepresented as per-process usage on WDDM. Job teardown must affect only owned descendants. Readiness requires expected build/backend/model identity and actual context, not merely an open port or spawned PID.

Use load deadlines suitable for large disk-backed models (the custom saved load took 14 minutes), bounded queues, progress-aware stall detection, client disconnect cancellation and explicit protocol errors. A malformed DONE, missing READY, unknown quant or lost child fails the request and closes its stream clearly. Retry/restart only with bounded policy and a new readiness check; do not replay tools or resume partial responses invisibly.

No silent fallbacks: no smaller quant, expert drop, reduced top-k, lower context, different fine-tune, different family or unqualified prefill. A CPU expert miss path that computes all declared contributions can be allowed and reported; it is different from skipping the expert. Fallback to another backend is allowed only for the **same registered artifact and semantics**, with prior qualification, visible identity and a clean request boundary. If those conditions fail, report unsupported/unavailable.

## 11. Phased implementation gates and rollback

| Phase | Concrete work | Exit gate |
|---|---|---|
| 0: freeze identities | Capture actual dirty custom sources, donor/submodule pins, build recipes and evidence; preserve working Maya config/build/profile; select exact model IDs. | Reconstructable source snapshot, no reliance on base SHA alone. Work must occur in a new integration branch/worktree when authorized, not overwrite current testing. |
| 1: descriptor and metadata admission | CPU-only registry, family/version/format checks, explicit overrides, resource estimate and capability schema; replace binary-name sniffing. | Synthetic metadata/shard/unknown-quant/mismatch tests and no GPU allocation before valid admission. |
| 2a: intact-service bridge | Optional safest first delivery: supervisor proxies the selected exact backend service over private loopback; retain Maya's Python frontend and custom Unbiased frontend respectively. Keep dependency environments separate. | Same backend behavior through the common API, one GPU owner, transparent capability errors; explicitly report that the GLM channel still uses Python. |
| 2b: native Maya GLM adapter | Reuse custom Win32 transport and C++ prompt/token/parser layer; pin installed Maya child; parallel 1; same model/config. | Fake-protocol tests first, then scheduled same-binary prompt/token/cancel/context parity and GLM quality checks after the owner's session is free. Promote only after parity; intact-service profile remains rollback. |
| 3: common launcher/lifecycle | One public API, one worker/child, GPU lease, loading/error state, clean switch; selected browser/client integration. | Repeated GLM -> Qwen -> DeepSeek41 -> GLM transitions, no orphan/overlap, truthful identities, no session-blob sharing and explicit missing-capability responses. |
| 4: model qualification | Exact Qwen artifacts; DeepSeek useful-context/quality tests; retain IQ1_M fast-prefill prohibition. D: model/table storage plan with verified capacity and integrity. | Per-model signed-off capability/context/precision matrix and held-out builder task results. Historical 1K DeepSeek receipt alone cannot pass this gate. |
| 5: selective runtime consolidation | Only if measured need justifies it: import GLM runtime as another explicitly isolated library/entry into custom tree; later review current Maya DS4 improvements independently. | Kernel/cache numerical parity, source/license/build reproducibility, meaningful performance benefit and unchanged fallback/fidelity policy. Optional, not needed for one user entry point. |

Rollback is a model/backend manifest selection to frozen working binaries/configs. Keep existing Maya 8088 launcher/config and custom DeepSeek binaries, evidence and index intact. Store new caches/profiles under separate build/model keys; no in-place pack conversion. A schema migration must preserve the previous manifest. Reverting the frontend/backend profile must not need redownloading or rewriting weights.

## 12. Remaining uncertainties and decisions

- This investigation located **the actual dirty custom worktree** and checked its source snapshot, but did not rerun any build/inference test or assert its saved performance is current after reboot.
- GLM current Maya setup is observed working; placing its runner behind the custom C++ tokenizer/template stack is proposed and untested. Current saved configuration has MTP off, context 65,536, GPU 0, 1,500 MiB reserve and 16 GB RAM headroom; do not enable drafting as an unannounced integration optimization.
- The exact Qwen files the owner wants are broader than the fork's validated set. Use the matrix and register specific artifacts; a dense 27B path is a separate capability decision.
- Determine the builder client's actual API dialect, tools, structured-output expectations, reasoning format and context/output requirements. Chat Completions can be the first common contract; Responses/Anthropic/dashboard require explicit feature work if required.
- Decide the fidelity default: recommended full computation of the selected quant, no silent expert pruning, with honest floating-point tolerances; optionally expose separately named experimental speed profiles later.
- DeepSeek current Maya DS4 has much newer code, quant/Engram and CPU optimization work. It may be worth a controlled comparison, but it is not presently built locally, its IQ1_M fast-prefill behavior is not qualified here, and no source inspection proves a speed gain. Preserve the custom exact-file adapter and refusal until the same-bundle tests demonstrate otherwise.
- D: capacity, physical NVMe performance, current free RAM/commit and exact storage migration are not freshly measured. The saved E: bundle placement is historical and should not silently become the new default.

**No active blocker prevented completion of this design investigation.** Native Windows read-only Git metadata required approved sandbox escalation; no automatic-review denial remains. Implementation and scheduling disruptive model tests are outside the authorized design-only scope. The practical outcome is a defined path to one Strata launcher/API, preserving the custom Unbiased work while making Maya GLM another explicit backend.
