"""Bounded exact-file UnbiasedStrata load + one reproducible PPS/TPS test.

stdlib only. Owns/cleans up only its frontend (whose Win32 job owns the child).
Never copies model weights. Resource snapshots are observations, not estimates.
"""
from __future__ import annotations
import ctypes as c
from ctypes import wintypes as w
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'evidence' / 'exact-run'
BUNDLE = Path('E:/Echo-DB/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M')
FIRST = BUNDLE / 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf'
ENGRAM = BUNDLE / 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'
ALIAS = 'daslab-deepseek41-iq1-m-unbiased'
PORT = 18141

def ready_for_exact_run(health, models):
    return (health.get('status') == 'ok' and health.get('loaded') is True
            and any(x.get('id') == ALIAS for x in models.get('data', [])))


def exact_done_metrics(protocol):
    """Accept only the single, complete 256/64 non-speculative run requested here."""
    done = [l.split() for l in protocol.splitlines() if l.startswith('DONE ')]
    if len(done) != 1 or len(done[0]) != 11:
        raise RuntimeError('expected one complete engine DONE timing receipt')
    d = done[0]
    try:
        n, pp = int(d[1]), int(d[2])
        ppms, tms = float(d[3]), float(d[4])
        counters = [int(x) for x in d[6:]]
    except ValueError as e:
        raise RuntimeError('malformed DONE counts/timings') from e
    if (n != 64 or pp != 256 or d[5] != 'length' or any(counters)
            or not all(math.isfinite(t) and t > 0 for t in (ppms, tms))):
        raise RuntimeError('DONE does not match the complete uncached 256/64 exact run')
    return dict(generated_tokens=n, prompt_ms=ppms, decode_ms=tms, finish=d[5],
                pps=pp*1000/ppms, tps=n*1000/tms)


class Memory(c.Structure):
    _fields_ = [('length', w.DWORD), ('load', w.DWORD)] + [(s, c.c_ulonglong) for s in
        ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_ext')]

class Process(c.Structure):
    _fields_ = [('size', w.DWORD), ('usage', w.DWORD), ('pid', w.DWORD), ('heap', c.c_size_t),
        ('module', w.DWORD), ('threads', w.DWORD), ('parent', w.DWORD), ('priority', w.LONG),
        ('flags', w.DWORD), ('name', w.WCHAR * 260)]

K = c.WinDLL('kernel32', use_last_error=True)
K.CreateToolhelp32Snapshot.restype = w.HANDLE
K.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
K.Process32FirstW.argtypes = [w.HANDLE, c.POINTER(Process)]
K.Process32NextW.argtypes = [w.HANDLE, c.POINTER(Process)]
K.CloseHandle.argtypes = [w.HANDLE]
K.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
K.OpenProcess.restype = w.HANDLE

def processes():
    h = K.CreateToolhelp32Snapshot(2, 0)
    p = Process(); p.size = c.sizeof(p)
    items = []
    try:
        ok = K.Process32FirstW(h, c.byref(p))
        while ok:
            items.append(dict(pid=p.pid, parent=p.parent, name=p.name))
            ok = K.Process32NextW(h, c.byref(p))
    finally:
        K.CloseHandle(h)
    return items

def process_memory(pid):
    # PROCESS_MEMORY_COUNTERS_EX and IO_COUNTERS (read bytes include buffered
    # reads; they do not prove physical drive bytes or mmap fault traffic).
    class PM(c.Structure):
        _fields_ = [('cb', w.DWORD), ('faults', w.DWORD)] + [(s, c.c_size_t) for s in
            ('peak_working_set', 'working_set', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged',
             'pagefile', 'peak_pagefile', 'private_bytes')]
    class IO(c.Structure):
        _fields_ = [(s, c.c_ulonglong) for s in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
    h = K.OpenProcess(0x410, False, pid)
    if not h: return dict(pid=pid, unavailable=True)
    try:
        pm = PM(); pm.cb = c.sizeof(pm); io = IO()
        getmem = c.WinDLL('psapi').GetProcessMemoryInfo
        getmem.argtypes = [w.HANDLE, c.POINTER(PM), w.DWORD]
        K.GetProcessIoCounters.argtypes = [w.HANDLE, c.POINTER(IO)]
        ok = getmem(h, c.byref(pm), pm.cb)
        K.GetProcessIoCounters(h, c.byref(io))
        return dict(pid=pid, memory_ok=bool(ok), **{k: getattr(pm,k) for k,_ in pm._fields_},
                    io={k:getattr(io,k) for k,_ in io._fields_})
    finally: K.CloseHandle(h)

def snapshot(frontend=None):
    m = Memory(); m.length = c.sizeof(m)
    if not K.GlobalMemoryStatusEx(c.byref(m)): raise c.WinError(c.get_last_error())
    ps = processes()
    relevant = [p for p in ps if any(s in p['name'].lower() for s in ('strata','llama','python','nvcc'))]
    owned = [p['pid'] for p in ps if frontend and (p['pid']==frontend or p['parent']==frontend)]
    gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
    import shutil
    return dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        memory={k:getattr(m,k) for k,_ in m._fields_}, gpu=gpu,
        disk_free={d:shutil.disk_usage(d).free for d in ('D:/','E:/')},
        processes=relevant, owned=[process_memory(pid) for pid in owned])

def request(path, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(f'http://127.0.0.1:{PORT}'+path, data=data,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f: return json.load(f)

def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=False)  # refuse to overwrite a previous run
    before = snapshot(); save('before.json',before)
    if before['memory']['avail_phys'] < 32*2**30 or before['memory']['avail_page'] < 40*2**30:
        raise RuntimeError('insufficient fresh RAM/commit headroom')
    if any('strata' in p['name'].lower() or 'llama' in p['name'].lower() for p in before['processes']):
        raise RuntimeError('another inference runtime is present; review the process snapshot')
    with socket.socket() as s: s.bind(('127.0.0.1',PORT))
    cmd = [str(ROOT/'build-cpu/unbiased-strata.exe'),'--engine',str(ROOT/'build-cuda/strata-poly.exe'),
        '-m',str(FIRST),'--host','127.0.0.1','--port',str(PORT),'--alias',ALIAS,'-c','1024',
        '--','--architecture','deepseek41','--model',str(FIRST),'--engram',str(ENGRAM),
        '--pack',str(ROOT/'evidence/deepseek41-index'),'--max-context','1024','--gpus','0',
        '--reserve-mib','1536','--mmap-experts','on','--ssd-cache','0','--prefill-chunk','0',
        '--dspark','none','--vision','off','--expert-profile','none','--cpu-threads','12','--verbose']
    env = os.environ.copy()
    env['PATH']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin;'+env['PATH']
    env['STRATA_ENGINE_LOG']=str(OUT/'engine-protocol.log')
    env['DS4_ENGRAM_CACHE_MB']='64'
    env['DS4_WIN_DIRECT']='0'  # buffered mmap/file reads on E: exFAT USB
    save('command.json',dict(command=cmd,environment={k:env[k] for k in
        ('STRATA_ENGINE_LOG','DS4_ENGRAM_CACHE_MB','DS4_WIN_DIRECT')},
        tree_base='2049dd2a292ffc38fd1cc006be04664ba15de5bb',donor='8183c6b67162340c09b67360860177c577a65f2e'))
    start=time.monotonic(); proc=None
    result=dict(loaded=False,pps=None,tps=None)
    try:
        with (OUT/'stdout.log').open('wb') as so,(OUT/'stderr.log').open('wb') as se:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=so,stderr=se,
                creationflags=subprocess.CREATE_NO_WINDOW)
            save('pid.json',dict(frontend_pid=proc.pid))
            last=0
            while time.monotonic()-start < 900:
                if proc.poll() is not None: raise RuntimeError(f'frontend exited during load: {proc.returncode}')
                elapsed=time.monotonic()-start
                if elapsed-last>=15:
                    snap=snapshot(proc.pid)
                    with (OUT/'resources.jsonl').open('a') as f: f.write(json.dumps(snap)+'\n')
                    print(f'load {elapsed:.0f}s, RAM available {snap["memory"]["avail_phys"]/2**30:.1f} GiB; GPU {snap["gpu"]}',flush=True)
                    if snap['memory']['avail_phys']<8*2**30 or snap['memory']['avail_page']<8*2**30:
                        raise RuntimeError('resource safety floor reached')
                    last=elapsed
                try:
                    health=request('/health',timeout=1)
                    models=request('/v1/models',timeout=1)
                    if ready_for_exact_run(health, models):
                        result.update(loaded=True,load_seconds=time.monotonic()-start)
                        save('health.json',health); save('models.json',models); break
                except Exception: pass
                time.sleep(1)
            if not result['loaded']: raise RuntimeError('900-second load deadline reached')
            save('loaded-resources.json',snapshot(proc.pid))
            # Exercise the actual embedded Jinja before measuring the plain token prompt.
            template=request('/apply-template',dict(messages=[dict(role='user',content='Name the capital of France.')],
                chat_template_kwargs=dict(enable_thinking=False)))
            save('template.json',template)
            text=('Write a clear, detailed explanation of how a library organizes books. Discuss authors, titles, '
                  'catalogs, shelves, borrowing, returning, and how a visitor can find a specific subject. ')*12
            tokens=request('/tokenize',dict(content=text,add_special=True))['tokens'][:256]
            if len(tokens)!=256: raise RuntimeError('could not construct exactly 256 prompt tokens')
            body=dict(prompt=tokens,n_predict=64,temperature=0,seed=42,cache_prompt=False,stream=False)
            save('request.json',body)
            t=time.monotonic()
            reply=request('/completion',body,timeout=1800)
            save('response.json',reply)
            result.update(request_wall_seconds=time.monotonic()-t,prompt_tokens=256,
                cache_state='first inference in this process; OS cache state uncontrolled')
            save('after-request.json',snapshot(proc.pid))
            # The child DONE counts and times are the source of throughput.
            result.update(exact_done_metrics((OUT/'engine-protocol.log').read_text()))
            print(json.dumps(result),flush=True)
    except Exception as e:
        result['blocker']=str(e); print('BLOCKER: '+str(e),flush=True)
    finally:
        save('result.json',result)
        if proc and proc.poll() is None:
            # Only our process; closing its job handle terminates its own engine.
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()
        save('after-cleanup.json',snapshot())
    return 0 if result.get('pps') else 1

if __name__=='__main__': raise SystemExit(main())
