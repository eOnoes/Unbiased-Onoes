# Canonical repository

The maintained source is https://github.com/eOnoes/Unbiased-Onoes on `main`.
The initial snapshot preserves the current custom Windows/DeepSeek41 source,
attributed to the upstream and donor revisions in `SOURCE_SNAPSHOT.json`.
The original local repositories, branches, builds and raw receipts are retained
as historical rollback material; they are not additional publishing destinations.

This repository started empty. Its first commit is a reviewed source snapshot,
not a rewrite of an existing remote branch. Original history was not imported
because it contains optional model-derived binary assets and historical media.
Required copyright/license notices and the pinned llama.cpp gitlink are retained.
No executable binaries, weights, raw telemetry, downloaded references or build
caches are published. Small draft-vocabulary/expert-profile assets listed in
the snapshot manifest remain external model-preparation inputs; existing local
runtime artifacts are preserved, and source compilation does not regenerate or
claim to validate them automatically.

`CUSTOM_BASELINE.md` describes the earlier local preservation stage. Its statements
about no configured publishing destination apply to that stage. This document
records the later explicitly authorized canonical destination. `MAYA_INTEGRATION.md`
is the staged design. No new GPU/model runtime validation is implied by publishing.

The next gate is CPU-only exact artifact/format/execution-mode admission and a
thin controller retaining intact backend frontends. The failed DeepSeek41 IQ1_M
fast-chunked FP16 mode remains denied. Existing Maya testing is independently
owned and must not be stopped, adopted or restarted by source-management work.
