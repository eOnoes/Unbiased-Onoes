import copy
import http.client
from http.server import BaseHTTPRequestHandler
import json
import os
import select
import socket
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from strata_control.metadata import AdmissionError, inspect_gguf, validate_bundle
from strata_control.registry import load_profiles, validate_options, clean_environment, admit, worker_command, validate_pack, sha256_file
from strata_control.process import DeviceLease
from strata_control.server import Controller, Busy, HTTP, make_handler


def gguf(path, arch='qwen35moe', typ=0, truncate=False):
    def s(x):
        b = x.encode(); return struct.pack('<Q', len(b)) + b
    meta = {'general.architecture': arch, arch + '.block_count': 40,
            arch + '.embedding_length': 2048, arch + '.expert_count': 256,
            arch + '.expert_used_count': 8}
    out = b'GGUF' + struct.pack('<IQQ', 3, 1, len(meta))
    for k, v in meta.items():
        out += s(k) + struct.pack('<I', 8 if isinstance(v, str) else 4)
        out += s(v) if isinstance(v, str) else struct.pack('<I', v)
    out += s('blk.0.ffn_gate_exps.weight') + struct.pack('<IQIQ', 1, 4, typ, 0)
    out += b'\0' * (-len(out) % 32) + b'\0' * (15 if truncate else 16)
    path.write_bytes(out)
    return meta


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def profile(self):
        p = self.root / 'arbitrary-name.gguf'; meta = gguf(p)
        info = inspect_gguf(p)
        return {'family': 'qwen35moe', 'expected_metadata': {k: v for k, v in meta.items() if k != 'general.architecture'},
                'artifacts': [{'path': str(p), 'role': 'backbone', 'size': info['size'],
                               'header_sha256': info['header_sha256']}]}
    def test_metadata_selects_family_without_filename(self):
        p = self.profile(); self.assertEqual(len(validate_bundle(p)), 1)
    def test_architecture_override_cannot_alias(self):
        p = self.profile(); p['family'] = 'deepseek41'
        with self.assertRaises(AdmissionError): validate_bundle(p)
    def test_exact_identity_and_geometry_required(self):
        for change in ('hash', 'size', 'geometry'):
            p = self.profile()
            if change == 'hash': p['artifacts'][0]['header_sha256'] = '0' * 64
            if change == 'size': p['artifacts'][0]['size'] += 1
            if change == 'geometry': p['expected_metadata'].pop('qwen35moe.expert_count')
            with self.subTest(change=change), self.assertRaises(AdmissionError): validate_bundle(p)
    def test_unknown_type_truncated_payload_and_header_bound(self):
        p = self.root / 'bad.gguf'
        for kw in ({'typ': 999}, {'truncate': True}):
            gguf(p, **kw)
            with self.assertRaises(AdmissionError): inspect_gguf(p)
        gguf(p)
        with self.assertRaises(AdmissionError): inspect_gguf(p, limit=16)
    def test_shards_cannot_be_guessed(self):
        p = self.profile(); p['artifacts'].append(copy.deepcopy(p['artifacts'][0]))
        with self.assertRaises(AdmissionError): validate_bundle(p)


def profile():
    return {'id': 'fixture', 'backend': 'unbiased-qwen35', 'family': 'qwen35moe',
            'mode': 'native', 'context': 1024, 'gpu': 0, 'precision_policy': 'full-computation',
            'qualification': 'pending'}


class RegistryTests(unittest.TestCase):
    def test_prepared_index_identity_and_model_directory_must_match(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); pack = root / 'index'; pack.mkdir()
            (pack / 'ds4.txt').write_text('model_dir ' + str(root) + '\n')
            (pack / 'tensors.txt').write_text('file 0 original.gguf\n')
            (pack / 'engram.bin').write_bytes(b'DSE1')
            p = {'artifacts': [{'path': str(root / 'original.gguf')}],
                 'pack_files': {x.name: sha256_file(x) for x in pack.iterdir()}}
            validate_pack(p, pack, {'ds4.txt', 'tensors.txt', 'engram.bin'})
            (pack / 'ds4.txt').write_text('model_dir ' + str(root / 'other') + '\n')
            with self.assertRaises(AdmissionError): validate_pack(p, pack, {'ds4.txt'})
            p['pack_files']['ds4.txt'] = sha256_file(pack / 'ds4.txt')
            with self.assertRaises(AdmissionError): validate_pack(p, pack, {'ds4.txt'})
    def test_profiles_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'p.json'
            for change in ({'backend': 'auto'}, {'mode': 'fast'}, {'gpu': 1},
                           {'precision_policy': 'approximate'}, {'id': '../bad'}):
                p = profile(); p.update(change)
                path.write_text(json.dumps({'schema': 1, 'profiles': [p]}))
                with self.subTest(change=change), self.assertRaises(AdmissionError): load_profiles(path)
    def test_failed_iq1m_mode_is_excluded_before_model_reads(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'p.json'; p = profile()
            p.update(backend='unbiased-deepseek41', family='deepseek41', mode='window', prefill_chunk=16)
            path.write_text(json.dumps({'schema': 1, 'profiles': [p]}))
            with self.assertRaises(AdmissionError): load_profiles(path)
            p['prefill_chunk'] = 0; p['context'] = 65536
            path.write_text(json.dumps({'schema': 1, 'profiles': [p]}))
            with self.assertRaises(AdmissionError): load_profiles(path)
    def test_pending_qualification_never_inspects_weights_or_starts_workers(self):
        with patch('strata_control.registry.validate_bundle', side_effect=AssertionError('forbidden')):
            with self.assertRaises(AdmissionError): admit(profile())
    def test_full_computation_environment(self):
        with patch.dict(os.environ, {'MAYA_GLM_DISK_SKIP_W': '0.8', 'STRATA_GLM_TRUNK_KEEP': '2',
                                    'STRATA_CONTROL_KEY': 'fixture-control-value', 'DS4_UNKNOWN': '1'}):
            env = clean_environment()
        self.assertEqual(env['MAYA_GLM_DISK_SKIP_W'], '0')
        self.assertEqual(env['STRATA_GLM_TRUNK_KEEP'], '8')
        self.assertNotIn('DS4_UNKNOWN', env); self.assertNotIn('STRATA_CONTROL_KEY', env)
        with self.assertRaises(AdmissionError): clean_environment({'MAYA_GLM_NO_MTP': '0'})
    def test_bad_engine_modes_and_context_rejected(self):
        for args in (['--max-context', '512'], ['--max-context', '1024', '--tail-skip', '7'],
                     ['--max-context', '1024', '--prefill-chunk', '16']):
            with self.subTest(args=args), self.assertRaises(AdmissionError):
                validate_options(args, 1024, 'unbiased-deepseek41')
    def test_maya_command_keeps_its_frontend_and_does_not_mutate_original_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = profile(); p.update(backend='maya-glm', source_root=d,
                binaries={'frontend': {'path': 'python.exe'}, 'engine': {'path': 'maya.exe'}})
            cfg = {'args': ['--glm-pack', 'pack', '--max-context', '1024', '--parallel', '1'],
                   'env': {'MAYA_GLM_NO_MTP': '1', 'MAYA_GLM_RAM_HEADROOM_GB': '16'}, 'mcp': ['forbidden']}
            before = copy.deepcopy(cfg)
            cmd, cwd, env = worker_command(p, 18999, Path(d), cfg)
            self.assertIn(str(Path(d) / 'serve/server.py'), cmd)
            self.assertEqual(cfg, before); self.assertEqual(env['MAYA_GLM_NO_MTP'], '1')
            generated = json.loads((Path(d) / 'maya-worker.json').read_text())
            self.assertNotIn('mcp', generated); self.assertFalse(generated['model_management'])


class FakeWorker:
    def __init__(self, argv, cwd, env):
        self.dead = False; self.posts = []; self.stops = 0; self.partial = False
        self.wait_stream = False; self.cancelled = threading.Event()
        parent = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def auth(self):
                if self.headers.get('Authorization') != 'Bearer ' + argv[1]:
                    self.send_response(401); self.end_headers(); return False
                return True
            def do_GET(self):
                if not self.auth(): return
                obj = {'status': 'ok', 'loaded': True, 'max_context': 1024}
                if self.path == '/v1/models': obj = {'data': [{'id': 'fixture'}]}
                data = json.dumps(obj).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data)
            def do_POST(self):
                if not self.auth(): return
                raw = self.rfile.read(int(self.headers['Content-Length']))
                parent.posts.append(raw)
                data = b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\\"x\\":1}"}}]}}]}\n\ndata: [DONE]\n\n'
                if parent.wait_stream:
                    self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers(); self.wfile.write(b'data: pending\n\n'); self.wfile.flush()
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        ready, _, _ = select.select([self.connection], [], [], 0.05)
                        if ready and self.connection.recv(1, socket.MSG_PEEK) == b'':
                            parent.cancelled.set(); return
                    return
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data[:24] if parent.partial else data)
                self.wfile.flush()
        self.http = HTTP(('127.0.0.1', int(argv[0])), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True); self.thread.start()
    def poll(self): return 1 if self.dead else None
    def stop(self):
        self.stops += 1; self.dead = True
        self.http.shutdown(); self.http.server_close(); self.thread.join()


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.c = Controller({'fixture': profile()}, self.temp.name, validator=lambda p: None,
                            idle_probe=lambda: None, worker_factory=FakeWorker,
                            command_factory=lambda p, port, root, cfg, key: ([str(port), key], root, {}))
    def tearDown(self):
        self.c.unload(); self.temp.cleanup()
    def test_busy_load_does_not_stop_or_release_existing_worker(self):
        self.c.load('fixture', timeout=2); w = self.c.worker
        with self.assertRaises(Busy): self.c.load('fixture')
        self.assertIs(self.c.worker, w); self.assertEqual(w.stops, 0)
        self.assertEqual(self.c.status()['state'], 'ready')
    def test_external_owner_blocks_before_worker_start(self):
        self.c.idle_probe = lambda: (_ for _ in ()).throw(AdmissionError('external owner'))
        with self.assertRaises(AdmissionError): self.c.load('fixture')
        self.assertIsNone(self.c.worker); self.assertEqual(self.c.state, 'failed')
    def test_active_request_refuses_unload_and_model_switch(self):
        self.c.load('fixture', timeout=2)
        with self.c.request_slot('/v1/chat/completions', {'model': 'fixture'}):
            with self.assertRaises(Busy): self.c.unload()
            with self.assertRaises(Busy): self.c.load('fixture')
            self.assertEqual(self.c.worker.stops, 0)
        self.c.unload(); self.assertEqual(self.c.state, 'idle')
    def test_no_tuning_unsupported_grammar_or_topk_clamp(self):
        self.c.load('fixture', timeout=2)
        for obj in ({'strata_tune': {}}, {'response_format': {'type': 'json_schema'}}, {'top_k': 100}, {'background': True}):
            with self.subTest(obj=obj), self.assertRaises(AdmissionError):
                with self.c.request_slot('/v1/chat/completions', obj): pass
        with self.assertRaises(AdmissionError):
            with self.c.request_slot('/v1/responses', {}): pass
    def test_dead_worker_needs_explicit_recovery(self):
        self.c.load('fixture', timeout=2); self.c.worker.dead = True
        self.assertEqual(self.c.status()['state'], 'failed')
        with self.assertRaises(Busy):
            with self.c.request_slot('/v1/chat/completions', {}): pass
    def test_device_lease_excludes_other_controller_and_releases_across_threads(self):
        a, b = DeviceLease(self.temp.name), DeviceLease(self.temp.name)
        a.acquire()
        try:
            with self.assertRaises(AdmissionError): b.acquire()
        finally:
            t = threading.Thread(target=a.release); t.start(); t.join()
        b.acquire(); b.release()
    def test_proxy_preserves_body_and_tool_stream_without_replay(self):
        self.c.load('fixture', timeout=2)
        key = 'fixture-only-control-key-123456789'
        api = HTTP(('127.0.0.1', 0), make_handler(self.c, key))
        t = threading.Thread(target=api.serve_forever, daemon=True); t.start()
        try:
            conn = http.client.HTTPConnection('127.0.0.1', api.server_port)
            raw = b'{ "model":"fixture", "stream":true, "messages":[{"role":"user","content":"x"}] }'
            conn.request('POST', '/v1/chat/completions', raw, {'Authorization': 'Bearer ' + key})
            response = conn.getresponse(); data = response.read()
            self.assertEqual(response.status, 200); self.assertIn(b'[DONE]', data)
            self.assertEqual(self.c.worker.posts, [raw]); conn.close()
            direct = http.client.HTTPConnection('127.0.0.1', self.c.port)
            direct.request('POST', '/v1/chat/completions', raw)
            self.assertEqual(direct.getresponse().status, 401); direct.close()
            conn = http.client.HTTPConnection('127.0.0.1', api.server_port)
            conn.request('POST', '/v1/chat/completions', raw)
            self.assertEqual(conn.getresponse().status, 401); conn.close()
            self.assertEqual(len(self.c.worker.posts), 1)
            self.c.worker.partial = True
            conn = http.client.HTTPConnection('127.0.0.1', api.server_port)
            conn.request('POST', '/v1/chat/completions', raw, {'Authorization': 'Bearer ' + key})
            partial = conn.getresponse().read(); conn.close()
            self.assertNotIn(b'[DONE]', partial)
            self.assertEqual(len(self.c.worker.posts), 2)  # one failed stream, no replay
            self.c.worker.partial = False; self.c.worker.wait_stream = True
            conn = http.client.HTTPConnection('127.0.0.1', api.server_port)
            conn.request('POST', '/v1/chat/completions', raw, {'Authorization': 'Bearer ' + key})
            response = conn.getresponse(); self.assertEqual(response.read(1), b'd')
            response.close(); conn.close()
            self.assertTrue(self.c.worker.cancelled.wait(2), 'disconnect must close the owned upstream request')
            self.assertEqual(len(self.c.worker.posts), 3)
        finally:
            api.shutdown(); api.server_close(); t.join()


if __name__ == '__main__': unittest.main()
