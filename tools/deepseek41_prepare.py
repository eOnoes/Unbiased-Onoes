"""Build only the small offset/hash index for the exact DASLab bundle.

All weight files stay read-only in their original location. No requantization.
The strict bundle audit runs before the pinned upstream metadata translator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from deepseek41_bundle_audit import audit, STEM, ENGRAM

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bundle', type=pathlib.Path)
    ap.add_argument('output', type=pathlib.Path)
    a = ap.parse_args()
    source, out = a.bundle.resolve(), a.output.resolve()
    if out == source or source in out.parents:
        ap.error('output must be outside the read-only bundle')
    result = audit(source)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'bundle-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    cmd = [sys.executable, str(pathlib.Path(__file__).parent / 'dsv41' / 'pack_gguf.py'),
           '--model', str(source), '--out', str(out), '--shards',
           *[f'{STEM}-{i:05d}-of-00004.gguf' for i in range(1, 5)],
           '--engram', str(source / ENGRAM)]
    subprocess.run(cmd, check=True)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (out / 'ds4.txt', out / 'tensors.txt', out / 'engram.bin')}
    (out / 'index-receipt.json').write_text(json.dumps(dict(command=cmd, files=files,
        donor='Niko1221/Strata@8183c6b67162340c09b67360860177c577a65f2e',
        weights_copied=False, weights_requantized=False), indent=2) + '\n')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
