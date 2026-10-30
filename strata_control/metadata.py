"""Bounded GGUF metadata/directory inspection. Never reads tensor payloads."""
import hashlib
import struct
from collections import Counter
from pathlib import Path


class AdmissionError(ValueError):
    pass


GEOMETRY = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
            6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84),
            11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
            16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
            20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
            15: (256, 292), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8),
            28: (1, 8), 29: (256, 56), 30: (1, 2), 39: (32, 17),
            40: (64, 36), 42: (64, 18)}
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
           7: '?', 10: 'Q', 11: 'q', 12: 'd'}
ARCHITECTURES = {'glm5-next': 'glm53', 'glm5next': 'glm53',
                 'qwen4exp': 'qwen-flash-next', 'qwen35moe': 'qwen35moe',
                 'deepseek4': 'deepseek4', 'deepseek41': 'deepseek41'}


def inspect_gguf(path, limit=64 * 1024 * 1024):
    path = Path(path)
    size = path.stat().st_size
    kept = {}
    with path.open('rb') as f:
        def read(n):
            if n < 0 or f.tell() + n > min(size, limit):
                raise AdmissionError('GGUF header exceeds its bounds or inspection limit')
            b = f.read(n)
            if len(b) != n:
                raise AdmissionError('truncated GGUF header')
            return b

        def num(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize('<' + fmt)))[0]

        def string(retain=True):
            n = num('Q')
            b = read(n)
            return b.decode('utf-8', errors='strict') if retain else None

        def value(kind, retain):
            if kind in SCALARS:
                v = num(SCALARS[kind])
                return v if retain else None
            if kind == 8:
                return string(retain)
            if kind == 9:
                subtype, n = num('I'), num('Q')
                if subtype == 9 or n > 2_000_000 or (retain and n > 4096):
                    raise AdmissionError('unsupported or oversized GGUF metadata array')
                if not retain and subtype in SCALARS:
                    read(n * struct.calcsize('<' + SCALARS[subtype]))
                    return None
                items = [value(subtype, retain) for _ in range(n)]
                return items if retain else None
            raise AdmissionError('unknown GGUF metadata value type')

        if read(4) != b'GGUF' or num('I') != 3:
            raise AdmissionError('only GGUF v3 is admitted')
        nt, nkv = num('Q'), num('Q')
        if nt > 100_000 or nkv > 100_000:
            raise AdmissionError('GGUF directory exceeds inspection limits')
        seen = set()
        for _ in range(nkv):
            key = string()
            if key in seen:
                raise AdmissionError('duplicate metadata key')
            seen.add(key)
            retain = (key in ('general.architecture', 'general.alignment') or
                      key.startswith('split.') or
                      (not key.startswith('tokenizer.') and any(t in key for t in (
                          'block_count', 'embedding_length', 'expert_count', 'expert_used_count',
                          'compress_ratios', 'source_layer_ids', 'engram.source'))))
            v = value(num('I'), retain)
            if retain:
                kept[key] = v
        tensors = []
        names = set()
        for _ in range(nt):
            name, nd = string(), num('I')
            if name in names or not 1 <= nd <= 4:
                raise AdmissionError('duplicate tensor or invalid tensor rank')
            names.add(name)
            shape = [num('Q') for _ in range(nd)]
            typ, off = num('I'), num('Q')
            if not all(0 < x < 2**48 for x in shape) or typ not in GEOMETRY:
                raise AdmissionError('unsupported tensor shape or quantization type')
            block, width = GEOMETRY[typ]
            if shape[0] % block:
                raise AdmissionError('tensor row is not whole quantization blocks')
            elements = 1
            for dim in shape:
                elements *= dim
            tensors.append({'name': name, 'shape': shape, 'type': typ,
                            'offset': off, 'bytes': elements // block * width})
        align = kept.get('general.alignment', 32)
        if not isinstance(align, int) or align < 1 or align > 4096 or align & (align - 1):
            raise AdmissionError('invalid GGUF alignment')
        start = (f.tell() + align - 1) // align * align
        previous = 0
        for t in sorted(tensors, key=lambda x: x['offset']):
            if t['offset'] % align or t['offset'] < previous or start + t['offset'] + t['bytes'] > size:
                raise AdmissionError('overlapping, unaligned or truncated tensor payload')
            previous = t['offset'] + t['bytes']
        if start > min(size, limit):
            raise AdmissionError('invalid GGUF data start')
        f.seek(0)
        fingerprint = hashlib.sha256(read(start)).hexdigest()
    return {'architecture': kept.get('general.architecture'), 'metadata': kept,
            'header_sha256': fingerprint, 'size': size, 'tensors': tensors,
            'quant_types': dict(Counter(str(t['type']) for t in tensors))}


def validate_bundle(profile):
    artifacts = profile['artifacts']
    if not artifacts:
        raise AdmissionError('an explicit artifact set is required')
    observed = []
    names = set()
    backbone = []
    sidecars = []
    for a in artifacts:
        result = inspect_gguf(a['path'])
        if result['size'] != a['size'] or result['header_sha256'] != a['header_sha256']:
            raise AdmissionError('artifact identity differs from the registered header/size')
        if a['role'] == 'backbone':
            backbone.append(result)
            for t in result['tensors']:
                if t['name'] in names:
                    raise AdmissionError('duplicate tensor across backbone shards')
                names.add(t['name'])
        elif a['role'] == 'engram':
            sidecars.append(result)
        else:
            raise AdmissionError('unsupported artifact role')
        observed.append(result)
    if not backbone:
        raise AdmissionError('missing backbone')
    meta = backbone[0]['metadata']
    arch = meta.get('general.architecture')
    family = ARCHITECTURES.get(arch)
    if family != profile['family']:
        raise AdmissionError('metadata architecture does not match the selected backend family')
    required = [arch + '.' + x for x in ('block_count', 'embedding_length', 'expert_count', 'expert_used_count')]
    expected = profile['expected_metadata']
    if not all(k in expected for k in required):
        raise AdmissionError('registered geometry is incomplete')
    for key, value in expected.items():
        if meta.get(key) != value:
            raise AdmissionError('metadata mismatch: ' + key)
    count = meta.get('split.count', 1)
    if count != len(backbone):
        raise AdmissionError('missing or extra backbone shards')
    for i, shard in enumerate(backbone):
        sm = shard['metadata']
        if count > 1 and (sm.get('split.count') != count or sm.get('split.no') != i or
                          sm.get('split.tensors.count') != len(names)):
            raise AdmissionError('inconsistent split identity or shard order')
        if sm.get('general.architecture', arch) != arch:
            raise AdmissionError('mixed backbone architectures')
        for key in required:
            if key in sm and sm[key] != meta.get(key):
                raise AdmissionError('inconsistent shard geometry')
    if not names or not any('.ffn_' in x and '_exps.weight' in x for x in names):
        raise AdmissionError('routed-expert tensors are required')
    if family == 'deepseek41':
        exact = {'deepseek41.block_count': 40, 'deepseek41.embedding_length': 5120,
                 'deepseek41.expert_count': 384, 'deepseek41.expert_used_count': 6,
                 'deepseek41.attention.compress_ratios': [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
                 'deepseek41.kv_source_layer_ids': [2, 8, 14, 20],
                 'deepseek41.index_source_layer_ids': [2, 8, 14, 20, 24, 28, 32, 36]}
        if len(backbone) != 4 or len(names) != 1008 or len(sidecars) != 1:
            raise AdmissionError('custom DeepSeek41 supports only the qualified four-shard/Engram bundle')
        for key, val in exact.items():
            if meta.get(key) != val:
                raise AdmissionError('unqualified DeepSeek41 geometry: ' + key)
        side = sidecars[0]
        if side['architecture'] != 'deepseek41-engram' or not meta.get('deepseek41.engram.source') or side['metadata'].get('deepseek41.engram.source') != meta['deepseek41.engram.source']:
            raise AdmissionError('Engram architecture/source mismatch')
        ts = sorted(side['tensors'], key=lambda t: t['name'])
        if [(t['type'], t['shape']) for t in ts] != [(20, [256, 384006168]), (20, [256, 384016682])]:
            raise AdmissionError('unqualified Engram layout')
    elif sidecars:
        raise AdmissionError('unexpected Engram sidecar')
    return observed
