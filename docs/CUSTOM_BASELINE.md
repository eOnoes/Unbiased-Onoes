# Managed custom baseline

Preserved locally on 2026-10-30. This is a source-preservation checkpoint, not a
new inference benchmark, security certification, release, or deployment.

## Identity

- Baseline source commit: `e014c6f55c2547e6c025f0c4f6102f070f31620d`.
- Branch at preservation: `codex/unbiased-deepseek41`.
- Original Unbiased base: `2049dd2a292ffc38fd1cc006be04664ba15de5bb`.
- Pinned llama.cpp submodule: `a6de9af3b19b5e3a5397f2c835a29a13c37664bd`.
- DeepSeek41 donor: official Strata `8183c6b67162340c09b67360860177c577a65f2e`.
- Donor Maya origin and licensing: `third_party/project-maya/NOTICE.md` and
  `LICENSE`; retained dsv41-ggml header and license under `third_party/`.
- Existing remote remains the upstream `robertkarl/UnbiasedStrata` repository.
  No user-owned publishing destination is configured or assumed by this work.

The original checkpoint contains the reviewed existing Win32 child transport,
distinct opt-in DeepSeek41 runtime, I64/type handling, original-quant Engram and
expert code, tests, offline preparation/audit tools, and attribution. No original
runtime arithmetic or prefill refusal was changed during preservation. All 62
source files named by the historical continuation build receipt matched their
recorded SHA-256 hashes before preservation. The reviewed source candidates
contained no detected credentials, personal home paths, emails, or binary data;
this bounded scan is not a guarantee that all upstream history is secret-free.

## Historical qualification, not rerun on preservation day

The retained local continuation records show one exact five-file DASLab IQ1_M
plus external IQ4_NL Engram run through this frontend and its own strata-poly
child: context 1024, greedy, 256 input tokens, 64 output tokens, no speculation.
The protocol receipt was:

```
DONE 64 256 38192.0 7592.7 length 0 0 0 0 0
```

That is 6.703 prompt tokens/s and 8.429 generated tokens/s using the child's
timing convention. Startup took 842.49 seconds, with startup-warmed weight
caches and uncontrolled OS cache. The output repeated its input; it establishes
bounded generation and timing, not builder intelligence or sustained throughput.
The sanitized tracked fixture retains only the protocol, metrics and minimal
health/model identity needed by the offline result-parser test.

The existing IQ1_M fast-chunked FP16 prefill test failed its tolerance on the
local GPU. Dispatch rejects nonzero `--prefill-chunk` and forces the tested
window path. This combination remains excluded from qualification even when
architectural metadata matches. Vision, DSpark, multi-GPU, persistent slots,
larger contexts and end-to-end answer quality remain unqualified for this port.
Full-file model hashes were not verified by the historical structural audit.

## Checks rerun during preservation

On 2026-10-30, offline Python `test_deepseek41*.py` passed 15 tests. Existing
CPU-only executables `model_family_test`, `gguf_reader_test`, `gguf_split_test`,
`gguf_geometry_test`, `deepseek41_types_test`, `literals_test`, and
`picture_cache_test` all exited zero. These were existing binaries, not rebuilt.
Temporary fixtures were confined to a separate task workspace. No model data,
GPU tests, inference requests, server lifecycle, installation or build ran.

## Included and excluded material

Only enumerated reviewed source, tests, build integration and license files
entered the baseline commit. Raw `evidence/` and downloaded `references/`
remain on disk and are ignored. Builds, binary caches, GGUF weights,
model-derived offset indexes/tokenizers/hash arrays, raw responses/prompts,
machine telemetry, local absolute-path recipes and downloaded reference Git
repositories are not part of the managed commit. Nothing excluded was deleted.
Small sanitized records and portable build instructions live in `docs/`.

## Known management follow-ups

The historical exact-run harness has machine-specific bundle/toolkit defaults;
it is a local laboratory helper, not the proposed single-GPU supervisor. Its
process-name checks are not a complete GPU ownership lock and do not protect
against every other backend, including Maya. Do not use it while another model
is active. The new supervisor requires an explicit device lease and admission.
No harness main function was invoked during preservation.

The donor build ID is a historical static string, not the new commit identity.
Use this source commit plus the pinned donor/submodule and actual build recipe
for provenance until build-identity modernization is reviewed. The generated
vocabulary-only llama source must also be recorded or pinned for releases.

No commits were pushed, remote destinations changed, or GitHub Actions run.
Next gate: CPU-only backend descriptors and metadata admission, preserving all
known exclusion rules before any model/API switching work.
