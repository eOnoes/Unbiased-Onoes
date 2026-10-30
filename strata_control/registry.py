"""Explicit local profiles, pinned backend identity and full-computation policy."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from .metadata import AdmissionError, validate_bundle

MAYA_PIN = '9030b73661b41d947117f7161fc6a11e14a5107a'
BACKENDS = {
    'maya-glm': ('glm53', {'native'}),
    'unbiased-flash': ('qwen-flash-next', {'native'}),
    'unbiased-qwen35': ('qwen35moe', {'native'}),
    'unbiased-deepseek4': ('deepseek4', {'native'}),
    'unbiased-deepseek41': ('deepseek41', {'window'}),
}
SAFE_ENV = {'STRATA_ROUTE_TAIL_SKIP': '0', 'MAYA_GLM_TRUNK_KEEP': '8',
            'STRATA_GLM_TRUNK_KEEP': '8', 'MAYA_GLM_TRUNK_KEEP_MAP': '',
            'STRATA_GLM_TRUNK_KEEP_MAP': '', 'MAYA_GLM_DISK_SKIP_W': '0',
            'STRATA_GLM_DISK_SKIP_W': '0', 'MAYA_GLM_DISK_SKIP_FROM': '8',
            'STRATA_GLM_DISK_SKIP_FROM': '8', 'MAYA_GLM_MISS_SKIP_W': '0',
            'STRATA_GLM_MISS_SKIP_W': '0', 'MAYA_DS4_DISK_SKIP_W': '0',
            'STRATA_DS4_DISK_SKIP_W': '0', 'MAYA_DS4_RAM_SKIP_W': '0',
            'STRATA_DS4_RAM_SKIP_W': '0', 'DS4_DISK_SKIP_W': '0', 'DS4_RAM_SKIP_W': '0'}
CAPS = {
    'maya-glm': ['/v1/chat/completions', '/v1/responses', '/v1/messages'],
    'unbiased': ['/v1/chat/completions', '/v1/completions', '/completion', '/tokenize', '/detokenize', '/apply-template'],
}


def load_profiles(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if obj.get('schema') != 1 or not isinstance(obj.get('profiles'), list):
        raise AdmissionError('profiles schema must be 1')
    result = {}
    for raw in obj['profiles']:
        p = dict(raw)
        name = p.get('id', '')
        backend = p.get('backend')
        if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,80}', name) or name in result or backend not in BACKENDS:
            raise AdmissionError('invalid/duplicate profile or unknown backend')
        fam, modes = BACKENDS[backend]
        if p.get('family') != fam or p.get('mode') not in modes:
            raise AdmissionError('unqualified model family/execution mode')
        if not isinstance(p.get('context'), int) or p['context'] < 128:
            raise AdmissionError('explicit context required')
        if backend == 'unbiased-deepseek41' and p['context'] != 1024:
            raise AdmissionError('custom DeepSeek41 has only 1024-context qualification')
        if p.get('gpu') != 0 or p.get('precision_policy') != 'full-computation':
            raise AdmissionError('initial controller supports only GPU 0 and full computation')
        if p.get('prefill_chunk', 0) != 0 and backend == 'unbiased-deepseek41':
            raise AdmissionError('IQ1_M fast-chunked FP16 prefill is prohibited')
        result[name] = p
    return result


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def validate_pack(profile, pack, required):
    pack = Path(pack).resolve()
    registered = profile.get('pack_files', {})
    required = set(required)
    if (pack / 'compat-bf16.json').is_file():
        required.add('compat-bf16.json')
    if not required.issubset(registered):
        raise AdmissionError('exact prepared-pack file identities are required')
    for name, digest in registered.items():
        if Path(name).name != name or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise AdmissionError('invalid prepared-pack identity')
        if sha256_file(pack / name) != digest:
            raise AdmissionError('prepared-pack identity mismatch')
    first = Path(profile['artifacts'][0]['path']).resolve()
    if 'sources.json' in required:
        sources = json.loads((pack / 'sources.json').read_text())['files']
        expected = {Path(a['path']).name: a['size'] for a in profile['artifacts'] if a['role'] == 'backbone'}
        if pack.parent != first.parent or {x['name']: x['size'] for x in sources} != expected:
            raise AdmissionError('Maya pack sources differ from registered artifacts')
    if 'ds4.txt' in required:
        model_dir = [line.split(' ', 1)[1] for line in (pack / 'ds4.txt').read_text().splitlines()
                     if line.startswith('model_dir ')]
        if len(model_dir) != 1 or Path(model_dir[0]).resolve() != first.parent:
            raise AdmissionError('DeepSeek prepared index points to another model directory')


def clean_environment(extra=None):
    # Never inherit model tuning or the controller key from the parent shell.
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(('MAYA_', 'STRATA_', 'DS4_'))}
    for key, value in (extra or {}).items():
        if key not in ('MAYA_GLM_NO_MTP', 'MAYA_GLM_RAM_HEADROOM_GB', 'DS4_ENGRAM_CACHE_MB', 'DS4_WIN_DIRECT'):
            raise AdmissionError('unreviewed backend environment setting: ' + key)
        if key == 'MAYA_GLM_NO_MTP' and str(value) != '1':
            raise AdmissionError('the initial GLM profile keeps MTP disabled')
        if key == 'MAYA_GLM_RAM_HEADROOM_GB' and float(value) < 16:
            raise AdmissionError('GLM RAM headroom must preserve at least 16 GB')
        env[key] = str(value)
    env.update(SAFE_ENV)
    env['CUDA_VISIBLE_DEVICES'] = '0'
    return env


def validate_options(args, context, backend):
    if not all(isinstance(a, str) for a in args):
        raise AdmissionError('engine arguments must be literal strings')
    denied = ('--control-vector', '--cvec-', '--remote-', '--peer-', '--layer-split', '--spec-follow')
    for arg in args:
        if arg.startswith(denied) or arg.startswith('--tail-skip=') or arg.startswith('--prefill-chunk='):
            raise AdmissionError('unqualified/approximating engine option')
    def vals(flag):
        return [args[i + 1] if i + 1 < len(args) else '' for i, x in enumerate(args) if x == flag]
    if vals('--max-context') != [str(context)]:
        raise AdmissionError('engine context must equal the registered context exactly')
    if vals('--tail-skip') not in ([], ['0']):
        raise AdmissionError('expert pruning prohibited')
    if backend == 'unbiased-deepseek41' and vals('--prefill-chunk') != ['0']:
        raise AdmissionError('IQ1_M requires explicit window prefill')
    if '--parallel' in args and vals('--parallel') != ['1']:
        raise AdmissionError('only single-sequence workers are qualified')
    return vals


def admit(profile):
    if profile.get('qualification') != 'approved' or not profile.get('qualification_record'):
        raise AdmissionError('profile awaits explicit exact-artifact/transport qualification')
    validate_bundle(profile)
    if os.name == 'nt' and any(Path(a['path']).drive.upper() != 'D:' for a in profile['artifacts']):
        raise AdmissionError('managed model files and disk-backed tables must be on D:')
    require_resources(profile)
    binaries = profile.get('binaries', {})
    if not binaries:
        raise AdmissionError('pinned executable identities required')
    for name, item in binaries.items():
        if not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', '')) or sha256_file(item['path']) != item['sha256']:
            raise AdmissionError('backend executable identity mismatch: ' + name)
    if profile['backend'] == 'maya-glm':
        root = Path(profile['source_root']).resolve()
        observed = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if observed != MAYA_PIN or (root / 'VERSION').read_text().strip() != '1.1.41':
            raise AdmissionError('Maya source pin/version mismatch')
        if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True):
            raise AdmissionError('tracked Maya source is dirty')
        cfg = json.loads(Path(profile['service_config']).read_text(encoding='utf-8'))
        vals = validate_options(cfg['args'], profile['context'], 'maya-glm')
        if not vals('--glm-pack') or vals('--parallel') != ['1'] or cfg.get('gpu') != [0]:
            raise AdmissionError('Maya profile must name its GLM pack, GPU 0 and parallel 1')
        validate_pack(profile, vals('--glm-pack')[0], {'dense.bin', 'index.txt', 'native_experts.txt', 'sources.json'})
        installed = (cfg.get('installer') or {}).get('gguf')
        if installed and Path(installed).resolve() != Path(profile['artifacts'][0]['path']).resolve():
            raise AdmissionError('Maya config names another artifact')
        validate_tokenizer(profile, cfg, root)
        if sha256_file(cfg['exe']) != binaries['engine']['sha256']:
            raise AdmissionError('Maya config points to another engine')
        for key in cfg.get('env', {}):
            if key not in SAFE_ENV and key not in ('MAYA_GLM_NO_MTP', 'MAYA_GLM_RAM_HEADROOM_GB'):
                raise AdmissionError('Maya config contains unreviewed tuning')
        if str(cfg.get('env', {}).get('MAYA_GLM_NO_MTP')) != '1':
            raise AdmissionError('initial Maya GLM profile preserves MTP off')
        return cfg
    args = profile['engine_args']
    vals = validate_options(args, profile['context'], profile['backend'])
    first = Path(profile['artifacts'][0]['path']).resolve()
    flag = '--native' if profile['backend'] == 'unbiased-flash' else '--model'
    if len(vals(flag)) != 1 or Path(vals(flag)[0]).resolve() != first:
        raise AdmissionError('engine arguments name another artifact')
    if profile['backend'] == 'unbiased-deepseek41' and vals('--architecture') != ['deepseek41']:
        raise AdmissionError('distinct DeepSeek41 dispatch required')
    if profile['backend'] == 'unbiased-deepseek41':
        if len(vals('--pack')) != 1:
            raise AdmissionError('DeepSeek41 requires one exact prepared index')
        validate_pack(profile, vals('--pack')[0], {'ds4.txt', 'tensors.txt', 'engram.bin'})
        sidecars = [a['path'] for a in profile['artifacts'] if a['role'] == 'engram']
        if len(vals('--engram')) != 1 or Path(vals('--engram')[0]).resolve() != Path(sidecars[0]).resolve():
            raise AdmissionError('engine arguments name another Engram file')
    return None


def require_resources(profile):
    req = profile.get('requirements', {})
    floor = (80, 96) if profile['backend'] == 'maya-glm' else (32, 40)
    ram, commit, vram = (req.get(k, 0) for k in ('free_ram_gib', 'free_commit_gib', 'free_vram_mib'))
    if ram < floor[0] or commit < floor[1] or vram < 28000:
        raise AdmissionError('explicit reviewed RAM/commit/VRAM headroom is required')
    if os.name != 'nt':
        raise AdmissionError('real backend resource admission is initially Windows-only')
    import ctypes
    from ctypes import wintypes as w
    class Memory(ctypes.Structure):
        _fields_ = [('length', w.DWORD), ('load', w.DWORD)] + [(x, ctypes.c_uint64) for x in (
            'total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_ext')]
    m = Memory(); m.length = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise AdmissionError('cannot inspect RAM/commit headroom')
    if m.avail_phys < ram * 2**30 or m.avail_page < commit * 2**30:
        raise AdmissionError('RAM or Windows commit headroom is below the registered budget')
    try:
        free = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True, timeout=10)
        if float(free.strip()) < vram:
            raise AdmissionError('free VRAM is below the registered budget')
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        raise AdmissionError('cannot inspect available VRAM') from e


def worker_command(profile, port, runtime_dir, config=None, backend_key=''):
    extra = dict(profile.get('environment', {}))
    if profile['backend'] == 'maya-glm':
        for k in ('MAYA_GLM_NO_MTP', 'MAYA_GLM_RAM_HEADROOM_GB'):
            if k in config.get('env', {}):
                extra.setdefault(k, config['env'][k])
        extra.setdefault('MAYA_GLM_NO_MTP', '1')
        extra.setdefault('MAYA_GLM_RAM_HEADROOM_GB', '16')
    env = clean_environment(extra)
    libraries = profile.get('library_dirs', [])
    if any(not Path(p).is_absolute() or not Path(p).is_dir() for p in libraries):
        raise AdmissionError('backend library directories must be explicit existing absolute paths')
    if libraries:
        env['PATH'] = os.pathsep.join(libraries + [env.get('PATH', '')])
    b = profile['binaries']
    if profile['backend'] == 'maya-glm':
        cfg = dict(config)
        cfg.update(host='127.0.0.1', port=port, model_name=profile['id'],
                   model_management=False, idle_unload_s=0)
        cfg.pop('api_key', None)
        cfg['_strata_controller'] = {'schema': 1, 'profile': profile['id']}
        for key in list(cfg):
            if key.startswith('mcp'):
                del cfg[key]
        cfg['env'] = {k: v for k, v in env.items() if k.startswith(('MAYA_', 'STRATA_', 'DS4_'))}
        # Keep mutable service files away from the working Maya config/source.
        cfg['log'] = str(runtime_dir / 'maya-engine.log')
        cp = runtime_dir / 'maya-worker.json'
        cleanup_runtime_config(runtime_dir, profile['id'])
        cp.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        # Pinned Maya --api-key defaults to STRATA_API_KEY (server.py:5728).
        # Set it only after config serialization; never persist it in JSON.
        env['STRATA_API_KEY'] = backend_key
        cmd = [b['frontend']['path'], str(Path(profile['source_root']) / 'serve/server.py'),
               '--engine', 'strata', '--config', str(cp), '--host', '127.0.0.1', '--port', str(port)]
        return cmd, profile['source_root'], env, cp
    cmd = [b['frontend']['path'], '--engine', b['engine']['path'], '-m', profile['artifacts'][0]['path'],
           '--host', '127.0.0.1', '--port', str(port), '--alias', profile['id'],
           '-c', str(profile['context'])]
    if backend_key:
        cmd += ['--api-key', backend_key]
    cmd += ['--', *profile['engine_args']]
    return cmd, profile.get('source_root', str(Path(b['frontend']['path']).parent)), env, None


def cleanup_runtime_config(directory, profile_id):
    """Remove only an identifiable controller-owned config, never a tree."""
    directory = Path(directory)
    path = directory / 'maya-worker.json'
    if directory.is_symlink() or path.is_symlink():
        raise AdmissionError('refusing linked runtime configuration')
    if not path.exists():
        return
    if path.stat().st_size > 1024 * 1024:
        raise AdmissionError('unrecognized runtime configuration requires owner review')
    try:
        cfg = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as e:
        raise AdmissionError('unrecognized runtime configuration requires owner review') from e
    marker = cfg.get('_strata_controller')
    modern = marker == {'schema': 1, 'profile': profile_id}
    # Exact f45813c-generated shape; enables migration of its credential file
    # only within this registered profile's controller runtime directory.
    legacy = (marker is None and cfg.get('host') == '127.0.0.1' and
              cfg.get('model_name') == profile_id and cfg.get('model_management') is False and
              cfg.get('log') == str(directory / 'maya-engine.log') and
              isinstance(cfg.get('port'), int) and isinstance(cfg.get('args'), list))
    if not (modern or legacy):
        raise AdmissionError('unowned runtime configuration requires owner review')
    path.unlink()


def validate_tokenizer(profile, cfg, source_root):
    """Bind bounded tokenizer assets and the effective pinned GLM template."""
    identity = profile.get('tokenizer_identity', {})
    declared = Path(identity.get('root', ''))
    if not identity.get('root') or not declared.is_absolute() or declared.is_symlink():
        raise AdmissionError('explicit unlinked tokenizer root required')
    root = declared.resolve()
    if not cfg.get('tokenizer') or Path(cfg['tokenizer']).resolve() != root:
        raise AdmissionError('tokenizer directory differs from the registered profile')
    if identity.get('artifact_header_sha256') != profile['artifacts'][0]['header_sha256']:
        raise AdmissionError('tokenizer identity is not tied to this artifact')
    files = identity.get('files', {})
    required = {'vocab.json', 'merges.txt', 'tokenizer.json', 'token_type.json', 'chat_template.jinja'}
    if not required.issubset(files):
        raise AdmissionError('complete bounded tokenizer/template identities required')
    total = 0
    for name, entry in files.items():
        if Path(name).name != name or root.is_symlink() or (root / name).is_symlink():
            raise AdmissionError('invalid tokenizer asset path')
        p = root / name
        size = p.stat().st_size; total += size
        if size != entry.get('size') or size > 32 * 1024 * 1024 or total > 64 * 1024 * 1024:
            raise AdmissionError('tokenizer asset size exceeds its registered bounds')
        if sha256_file(p) != entry.get('sha256'):
            raise AdmissionError('tokenizer/template identity mismatch')
    # Mirror pinned server.py:5510 chat_template_path without importing its
    # service. MAYA_PACK_TEMPLATE is cleared by clean_environment. The swap is
    # conditional on the pack's exact GLM signature, not merely its family name.
    packed = root / 'chat_template.jinja'
    if packed.stat().st_size > 1024 * 1024:
        raise AdmissionError('pack chat template exceeds bounded identity limit')
    head = packed.read_text(encoding='utf-8', errors='replace')
    glm = Path(source_root) / 'serve/glm5_chat_template.jinja'
    template = glm if (glm.exists() and head.startswith('[gMASK]<sop>') and
                       "reasoning_effort in ['low', 'high']" in head) else packed
    if template.stat().st_size > 1024 * 1024 or sha256_file(template) != identity.get('effective_template_sha256'):
        raise AdmissionError('effective Maya GLM template identity mismatch')
