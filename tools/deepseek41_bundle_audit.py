"""Structural audit of the DASLab IQ1_M bundle; never loads an inference engine.

Reads GGUF headers, the small integer hash tensors, and three 144-byte rows per
Engram table. A successful audit is NOT a weight-integrity or runtime-support claim.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import struct

from gguf_reader import BLOCK_GEOMETRY, GGUFFile

STEM = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M"
ENGRAM = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
EXPERT_TYPES = {"IQ1_S": 83, "IQ1_M": 15, "IQ2_XXS": 20, "IQ2_S": 2}
BACKBONE_TYPES = dict(EXPERT_TYPES, F32=489, Q5_K=244, Q8_0=80,
                      BF16=71, I32=1, I64=3)
GEOMETRY = dict(BLOCK_GEOMETRY, I32=(1, 4), I64=(1, 8))


def payload_bytes(t):
    block, size = GEOMETRY[t.type_name]
    if not t.shape or any(d <= 0 for d in t.shape) or t.shape[0] % block:
        raise ValueError(f"{t.name}: row is not whole blocks")
    return math.prod(t.shape) // block * size


def validate_contract(meta, tensors, engram_meta, engram_tensors):
    """Exact selected build contract, separate from file I/O for mutation tests."""
    def need(ok, message):
        if not ok:
            raise ValueError(message)

    expected = {
        "general.architecture": "deepseek41", "deepseek41.block_count": 40,
        "deepseek41.embedding_length": 5120, "deepseek41.expert_count": 384,
        "deepseek41.expert_used_count": 6, "deepseek41.expert_feed_forward_length": 2304,
        "deepseek41.attention.compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        "deepseek41.kv_source_layer_ids": [2, 8, 14, 20],
        "deepseek41.index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "deepseek41.engram.layer_ids": [1, 14],
        "deepseek41.engram.num_embeddings": [384006168, 384016682],
        "deepseek41.engram_head_dim": 256, "deepseek41.engram_n_heads": 8,
        "deepseek41.engram_max_ngram_size": 4, "deepseek41.engram.file": ENGRAM,
    }
    for key, value in expected.items():
        need(meta.get(key) == value, f"metadata mismatch: {key}")
    need(engram_meta.get("general.architecture") == "deepseek41-engram", "Engram architecture")
    source = meta.get("deepseek41.engram.source")
    need(bool(source) and source == engram_meta.get("deepseek41.engram.source"), "Engram source mismatch")
    need(len(tensors) == 1008, "backbone tensor count")
    need(collections.Counter(t.type_name for t in tensors) == BACKBONE_TYPES, "backbone type histogram")
    by_name = {t.name: t for t in tensors}
    need(len(by_name) == len(tensors), "duplicate backbone tensor")
    experts = []
    for layer in range(40):
        for role in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            t = by_name.get(name)
            shape = [2304, 5120, 384] if role == "down" else [5120, 2304, 384]
            need(t is not None and t.shape == shape, f"expert shape/missing: {name}")
            experts.append(t)
    need(collections.Counter(t.type_name for t in experts) == EXPERT_TYPES, "expert type histogram")
    need(len(engram_tensors) == 2, "Engram table count")
    tables = {t.name: t for t in engram_tensors}
    for layer, rows in zip([1, 14], [384006168, 384016682]):
        t = tables.get(f"blk.{layer}.engram_embd.weight")
        need(t is not None and t.shape == [256, rows] and t.type_name == "IQ4_NL", "Engram table shape/type")


def audit(root):
    paths = [root / f"{STEM}-{i:05d}-of-00004.gguf" for i in range(1, 5)] + [root / ENGRAM]
    files = [GGUFFile(p) for p in paths]
    tensors = [t for g in files[:4] for t in g.tensors]
    validate_contract(files[0].metadata, tensors, files[4].metadata, files[4].tensors)
    receipts = []
    for i, g in enumerate(files):
        if i < 4:
            for key, value in {"split.count": 4, "split.no": i, "split.tensors.count": 1008}.items():
                if g.metadata.get(key) != value:
                    raise ValueError(f"{g.path.name}: {key}")
        size = g.path.stat().st_size
        previous_end = 0
        for t in sorted(g.tensors, key=lambda t: t.offset):
            end = t.offset + payload_bytes(t)
            if t.offset % g.alignment or t.offset < previous_end or g.data_start + end > size:
                raise ValueError(f"{t.name}: overlapping, unaligned or truncated payload")
            previous_end = end
        with g.path.open("rb") as f:
            header_hash = hashlib.sha256(f.read(g.data_start)).hexdigest()
        receipts.append(dict(file=g.path.name, size=size, version=g.version,
                             header_bytes=g.data_start, header_sha256=header_hash,
                             tensor_count=len(g.tensors),
                             tensor_types=dict(collections.Counter(t.type_name for t in g.tensors))))

    def ints(name, fmt):
        for g in files[:4]:
            for t in g.tensors:
                if t.name == name:
                    with g.path.open("rb") as f:
                        f.seek(g.data_start + t.offset)
                        return list(struct.unpack("<" + str(math.prod(t.shape)) + fmt, f.read(payload_bytes(t))))
        raise ValueError(f"missing {name}")

    tm = ints("engram_token_map.weight", "i")
    primes = ints("engram_primes.weight", "q")
    offsets = ints("engram_offsets.weight", "q")
    multipliers = ints("engram_multipliers.weight", "q")
    if len(tm) != 129280 or min(tm) < 0 or max(tm) != 99091 or len(multipliers) != 8:
        raise ValueError("Engram hash arrays")
    if len(primes) != 48 or len(offsets) != 48:
        raise ValueError("Engram primes/offsets length")
    for e, rows in enumerate([384006168, 384016682]):
        at = 0
        for j in range(24):
            k = e * 24 + j
            if primes[k] <= 0 or offsets[k] != at:
                raise ValueError("Engram bucket offsets")
            at += primes[k]
        if at != rows:
            raise ValueError("Engram buckets do not tile table")
    samples = []
    g = files[4]
    with g.path.open("rb") as f:
        for t in g.tensors:
            for row in (0, t.shape[1] // 2, t.shape[1] - 1):
                f.seek(g.data_start + t.offset + row * 144)
                raw = f.read(144)
                if len(raw) != 144:
                    raise ValueError("short Engram row read")
                samples.append(dict(tensor=t.name, row=row, sha256=hashlib.sha256(raw).hexdigest()))
    return dict(structural_audit="PASS", runtime_compatible=False,
                runtime_compatibility="NOT IMPLEMENTED; this audit does not authorize a full load",
                full_file_hashes_verified=False, total_bytes=sum(r["size"] for r in receipts),
                files=receipts, engram_hash_arrays="PASS", sparse_row_samples=samples)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path)
    args = ap.parse_args()
    result = json.dumps(audit(args.bundle), indent=2)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)
