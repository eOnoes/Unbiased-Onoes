# One entry point, intact backends

`python strata.py` is the single local control/API entry point. This first stage
uses a small standard-library Python gateway; each backend keeps its existing
frontend/tokenizer/chat template/tool parser and inference engine. It does not
merge kernels, replace quants, or put a new tokenizer in front of Maya.

## Current qualification

The metadata, policy, lifecycle and HTTP behavior have CPU/synthetic/fake-worker
tests. No real model was loaded through this controller, no native worker was
launched for this integration test, and no GPU/cancellation/performance or builder
quality result is claimed. Every supplied example profile is deliberately
`pending`, with incomplete artifact/binary identities. An explicit load refuses
pending profiles before model inspection or worker creation. Do not mark one
approved merely because architecture metadata matches.

The existing DeepSeek41 IQ1_M numerical failure excludes fast-chunked FP16
prefill. Its only registered custom mode is `window`, `prefill_chunk=0`, context
1024. The native dispatch refusal remains untouched. Larger contexts and other
checkpoints need new qualification. GLM initially retains context 65536 and MTP
off. Source/version dependency is pinned in `config/maya.lock.json` and enforced
by the registry; no dependency/model download or installation happens here.

## Local setup after the benchmarking session is available

Keep all real paths, keys, model metadata receipts and runtime config local:

```text
copy config/profiles.example.json profiles.local.json
python strata.py check profiles.local.json
python strata.py inspect D:/models/<exact-first-shard.gguf>
```

`check` validates only profile schema/family/mode; it explicitly reports that no
artifact admission or model load occurred. `inspect` reads bounded GGUF v3
headers and tensor directories, never tensor payloads. Run it for every shard
and Engram sidecar. Register artifacts in exact shard order, each with `path`,
`role` (`backbone` or `engram`), exact `size` and `header_sha256`. Register the
architecture-specific `block_count`, `embedding_length`, `expert_count` and
`expert_used_count` under `expected_metadata`, plus the actual architecture's
prefix and any additional required metadata. Metadata aliases do not authorize
running DeepSeek41 through DeepSeek4. Unknown quantization types fail closed.

Register prepared-pack hashes in `pack_files` as basename-to-SHA256 entries.
Maya requires `dense.bin`, `index.txt`, `native_experts.txt`, `sources.json` and
`compat-bf16.json` when present; its source names/sizes and parent directory
must agree with the registered originals. DeepSeek41 requires `ds4.txt`,
`tensors.txt`, `engram.bin`, and an index model directory matching the registered
backbone directory. This prevents silently reusing an old E: index after D:
staging or pairing a valid checkpoint header with a different prepared pack.

Header identity is not full weight-file integrity. Record verified publisher
weight hashes and exact evidence for each artifact in the local qualification
record before approval. The initial controller also enforces Windows resource
headroom, D:-only model/table paths, GPU 0, source/binary pins, exact context and
no expert-drop/control-vector options. Preserve the existing metadata offset
index; never convert it to a specialized Q2_0 pack or rewrite weights.
Native backend DLL directories may be listed in `library_dirs`; they must be
explicit existing absolute paths. Keep each backend's matching CUDA runtime
and other dependencies separate and record them during qualification.

When scheduled real-backend tests pass, a reviewer can set that local profile's
`qualification` to `approved` and identify the reviewed `qualification_record`.
The file is trusted local configuration, never accepted through the HTTP API.

Set `STRATA_CONTROL_KEY` privately to a random token of at least 24 characters
(do not commit or log it), then:

```text
python strata.py serve profiles.local.json --port 8099
```

The service always binds 127.0.0.1; controller port 8088 is refused. Startup loads
no model and does not adopt an existing service. Only `/health` is unauthenticated.
Other endpoints require `Authorization: Bearer <STRATA_CONTROL_KEY>` and reject
browser Origin headers/untrusted Host headers. Send JSON `{"model":"<id>"}` to
`POST /admin/load`, `{}` to `/admin/unload`; inspect `/admin/status` and
`/v1/models`. Loads are explicit, not triggered by an inference request's model
field. A different requested model returns an error rather than silently switching.

The controller creates a private loopback service on an ephemeral port. Maya is
launched directly through its pinned `serve/server.py`, not its installer or
self-updating launcher. A generated runtime config preserves engine arguments,
sampling, tokenizer and model identity while binding loopback and disabling
backend model management/MCP; the original config is never edited. Backend
mutable logs/settings go under the ignored `.strata-runtime/` directory. Unbiased
uses its preserved C++ frontend and child unchanged. Executable hashes must
match the locally registered frontend/engine before any worker starts.
Each worker gets its own temporary backend API key, distinct from the public
controller key; direct unauthenticated calls to its private port are refused.
Generated runtime configs may contain that transient key and must remain local.

## API and ownership contract

- Exactly one managed GPU lease and worker. A named Windows mutex survives
  across HTTP request threads; a worker-only kill-on-close job contains owned
  descendants. External compute owners/known inference processes block loading;
  ownership/accounting probe failures fail closed. Nothing discovered is killed.
  The lease coordinates managed controllers; it cannot prevent an independent
  application from taking GPU memory after a successful probe. Real WDDM owner
  accounting and this external race remain explicit qualification limits.
- Resource checks run only for an explicit approved future load, never at import,
  controller startup, schema checks or fake-worker tests. WDDM ownership/resource
  behavior and nested jobs remain part of real Windows qualification.
- Switching requires explicit unload followed by load. Unload refuses active
  requests; no implicit cancellation/replay. Actual allocated context and alias
  must match before readiness. A dead worker needs explicit cleanup/reload.
- Original request JSON and response/tool-stream bytes pass through. The gateway
  does not render chat templates, reorder arguments, replay partial tools or fall
  back to a different engine/model/quant/context. Client disconnect closes the
  owned upstream request; verify backend cancellation during the scheduled gate.
- Endpoint support follows the selected frontend. Maya exposes its existing chat,
  Responses and Anthropic APIs; Unbiased exposes its existing chat/completion and
  tokenizer/template APIs. Unsupported routes fail explicitly. The dashboard and
  backend management endpoints are not exposed by this first gateway stage.
- Background generation, constrained schema/grammar/required-tool claims and
  per-request engine-tuning extensions are denied until qualified. Unbiased
  explicit `top_k` outside 1..64 is rejected instead of silently clamped. Message,
  reasoning and ordinary tool fields stay backend-owned.
- All inherited MAYA_/STRATA_/DS4_ tuning is cleared. Full-computation controls
  override environment aliases; target expert pruning is disabled. This preserves
  the selected quantized checkpoint, not its original higher-precision quality.

## Tests and next gate

```text
python -B -m unittest discover -s tests/control -v
python -B -m unittest discover -s tools -p "test_deepseek41*.py" -v
```

Controller tests use tiny synthetic GGUFs and ephemeral in-process HTTP workers;
they do not read real weights, call nvidia-smi, touch 8088, launch native inference,
or take over live processes. Confined temporary directories may be selected by
the runner when a sandbox restricts the default temp path.

Before real activation, schedule an idle-GPU window: verify exact D: artifact
identity/full-file integrity and build provenance; establish driver-specific
external-owner checks; test native/nested job lifecycle; load one approved backend
at a time; compare direct versus proxied templates/tool streams/stop behavior;
test mid-prefill and mid-generation disconnects, exact allocated context, release
and reload; then qualify quality and representative long-context tasks. Do not
claim production readiness before these pass. Rollback keeps the original
working Maya launcher/config and preserved Unbiased builds; no model or cache
format migration is required.
