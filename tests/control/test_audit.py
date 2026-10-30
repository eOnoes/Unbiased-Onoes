import copy
import ctypes
import http.client
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from strata_control.metadata import AdmissionError
from strata_control.process import Worker, JobAccounting
from strata_control.registry import (admit, worker_command, cleanup_runtime_config,
                                     sha256_file, validate_tokenizer)
from strata_control.server import (Controller, Busy, ready_backend, validated_route,
                                  Cancellation, RequestCancelled, forward_request, HTTP, make_handler)
from test_control import FakeWorker, profile

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/maya-1.1.41.json').read_text())


def controller(root, p=None, **kwargs):
    p = p or profile()
    defaults = dict(validator=lambda p: None, idle_probe=lambda: None,
                    release_probe=lambda p: None, worker_factory=FakeWorker,
                    command_factory=lambda p, port, run, cfg, key: ([str(port), key], run, {}, None),
                    secret_factory=lambda: 'fixture-only-worker-key')
    defaults.update(kwargs)
    return Controller({'fixture': p}, root, **defaults)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.p = profile(); self.p['backend'] = 'maya-glm'
    def test_source_derived_maya_contract_has_no_loaded_field(self):
        self.assertNotIn('loaded', FIXTURE['health'])
        self.assertTrue(ready_backend(self.p, FIXTURE['health'], FIXTURE['models']))
    def test_maya_unloaded_or_reloading_is_not_ready(self):
        for key, value in [('unloaded', True), ('status', 'reloading')]:
            health = dict(FIXTURE['health']); health[key] = value
            self.assertFalse(ready_backend(self.p, health, FIXTURE['models']))
    def test_wrong_model_context_version_or_model_listing_is_rejected(self):
        for target, key, value in [('health', 'model', 'other'), ('health', 'max_context', 4096),
                                   ('health', 'version', '1.1.17'), ('models', 'data', [])]:
            fixture = copy.deepcopy(FIXTURE); fixture[target][key] = value
            with self.subTest(key=key), self.assertRaises(AdmissionError):
                ready_backend(self.p, fixture['health'], fixture['models'])
        fixture = copy.deepcopy(FIXTURE); fixture['models']['data'][0]['meta']['n_ctx'] = 4096
        with self.assertRaises(AdmissionError): ready_backend(self.p, fixture['health'], fixture['models'])
    def test_native_contract_requires_loaded_and_service(self):
        health = {'status': 'ok', 'service': 'strata', 'loaded': True,
                  'model': 'fixture', 'max_context': 1024}
        models = {'object': 'list', 'data': [{'id': 'fixture'}]}
        self.assertTrue(ready_backend(profile(), health, models))
        health.pop('loaded'); self.assertFalse(ready_backend(profile(), health, models))
    def test_controller_loads_maya_fixture_contract(self):
        class MayaFake(FakeWorker):
            pass
        with tempfile.TemporaryDirectory() as root:
            c = controller(root, self.p)
            def get(path):
                return copy.deepcopy(FIXTURE['health' if path == '/health' else 'models'])
            c._get = get
            try: self.assertEqual(c.load('fixture', timeout=1)['state'], 'ready')
            finally: c.unload()


class RuntimeConfigTests(unittest.TestCase):
    def command(self, root):
        p = profile(); p.update(backend='maya-glm', source_root=str(root),
            binaries={'frontend': {'path': 'python.exe'}, 'engine': {'path': 'maya.exe'}})
        cfg = {'args': [], 'env': {}, 'api_key': 'old-fixture-key'}
        result = worker_command(p, 19111, root, cfg, 'fixture-only-worker-key')
        return p, result
    def test_key_is_environment_only_and_owned_config_can_be_removed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p, (_, _, env, cp) = self.command(root)
            self.assertEqual(env['STRATA_API_KEY'], 'fixture-only-worker-key')
            self.assertNotIn('fixture-only-worker-key', cp.read_text())
            self.assertNotIn('old-fixture-key', cp.read_text())
            self.assertNotIn('STRATA_API_KEY', json.loads(cp.read_text())['env'])
            cleanup_runtime_config(root, p['id']); self.assertFalse(cp.exists())
    def test_startup_removes_only_registered_owned_stale_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root / 'fixture'; run.mkdir()
            self.command(run); (run / 'keep.txt').write_text('keep')
            c = controller(root)
            self.assertFalse((run / 'maya-worker.json').exists())
            self.assertTrue((run / 'keep.txt').exists()); c.unload()
    def test_unowned_or_oversized_config_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cp = root / 'maya-worker.json'
            for content in ['{}', 'x' * (1024 * 1024 + 1)]:
                cp.write_text(content)
                with self.assertRaises(AdmissionError): cleanup_runtime_config(root, 'fixture')
                self.assertTrue(cp.exists())
    def test_legacy_controller_config_is_removed_without_reading_other_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cp = root / 'maya-worker.json'
            cp.write_text(json.dumps(dict(host='127.0.0.1', model_name='fixture', model_management=False,
                log=str(root / 'maya-engine.log'), port=19111, args=[], api_key='legacy-fixture-key')))
            cleanup_runtime_config(root, 'fixture'); self.assertFalse(cp.exists())
    def test_unload_and_failed_worker_creation_cleanup_config(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                def command(p, port, run, cfg, key):
                    self.command(run)
                    return [str(port), key], run, {}, run / 'maya-worker.json'
                factory = Mock(side_effect=OSError('fixture spawn failed')) if fail else FakeWorker
                c = controller(root, command_factory=command, worker_factory=factory)
                if fail:
                    with self.assertRaises(OSError): c.load('fixture', timeout=1)
                else:
                    c.load('fixture', timeout=1); c.unload()
                self.assertFalse((root / 'fixture/maya-worker.json').exists()); c.unload()


class TokenizerTests(unittest.TestCase):
    def fixture(self, root):
        tok = root / 'tokenizer'; tok.mkdir(); serve = root / 'serve'; serve.mkdir()
        files = {}
        for name in ('vocab.json', 'merges.txt', 'tokenizer.json', 'token_type.json', 'chat_template.jinja'):
            p = tok / name
            p.write_text("[gMASK]<sop> reasoning_effort in ['low', 'high']" if name == 'chat_template.jinja' else 'fixture-' + name)
            files[name] = {'size': p.stat().st_size, 'sha256': sha256_file(p)}
        template = serve / 'glm5_chat_template.jinja'; template.write_text('effective fixture template')
        p = profile(); p['artifacts'] = [{'header_sha256': 'a' * 64}]
        p['tokenizer_identity'] = dict(root=str(tok), files=files, artifact_header_sha256='a' * 64,
                                      effective_template_sha256=sha256_file(template))
        return p, {'tokenizer': str(tok)}
    def test_exact_assets_and_effective_template_are_required(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p, cfg = self.fixture(root)
            validate_tokenizer(p, cfg, root)
            for change in ('root', 'artifact', 'vocab', 'effective', 'missing'):
                q = copy.deepcopy(p); c = dict(cfg)
                if change == 'root': c['tokenizer'] = str(root)
                if change == 'artifact': q['tokenizer_identity']['artifact_header_sha256'] = 'b' * 64
                if change == 'vocab': q['tokenizer_identity']['files']['vocab.json']['sha256'] = 'b' * 64
                if change == 'effective': q['tokenizer_identity']['effective_template_sha256'] = 'b' * 64
                if change == 'missing': q['tokenizer_identity']['files'].pop('merges.txt')
                with self.subTest(change=change), self.assertRaises(AdmissionError): validate_tokenizer(q, c, root)
    def test_asset_size_limit_precedes_content_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p, cfg = self.fixture(root)
            with patch('strata_control.registry.sha256_file', side_effect=AssertionError('must not hash oversized asset')):
                p['tokenizer_identity']['files']['vocab.json']['size'] += 1
                with self.assertRaises(AdmissionError): validate_tokenizer(p, cfg, root)
    def test_effective_template_follows_pinned_conditional_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p, cfg = self.fixture(root)
            packed = root / 'tokenizer/chat_template.jinja'; packed.write_text('custom registered template')
            p['tokenizer_identity']['files']['chat_template.jinja'] = {
                'size': packed.stat().st_size, 'sha256': sha256_file(packed)}
            with self.assertRaises(AdmissionError): validate_tokenizer(p, cfg, root)
            p['tokenizer_identity']['effective_template_sha256'] = sha256_file(packed)
            validate_tokenizer(p, cfg, root)
    def test_null_installer_gguf_uses_validated_pack_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'VERSION').write_text('1.1.41')
            exe = root / 'engine.fixture'; exe.write_text('fixture')
            cfg = dict(args=['--glm-pack', str(root), '--max-context', '1024', '--parallel', '1'],
                       gpu=[0], installer={'gguf': None}, exe=str(exe), env={'MAYA_GLM_NO_MTP': '1'})
            cp = root / 'config.json'; cp.write_text(json.dumps(cfg))
            p = profile(); p.update(backend='maya-glm', qualification='approved', qualification_record='fixture',
                source_root=str(root), service_config=str(cp), artifacts=[{'path': r'D:\models\fixture.gguf'}],
                binaries={'engine': {'path': str(exe), 'sha256': sha256_file(exe)}})
            with patch('strata_control.registry.validate_bundle'), patch('strata_control.registry.require_resources'), \
                 patch('strata_control.registry.subprocess.check_output', side_effect=[
                     '9030b73661b41d947117f7161fc6a11e14a5107a', '']), \
                 patch('strata_control.registry.validate_pack') as pack, patch('strata_control.registry.validate_tokenizer'):
                self.assertEqual(admit(p), cfg)
                pack.assert_called_once()


class StopTests(unittest.TestCase):
    def worker(self, terminate=True, wait=0, query=True, active=0):
        w = Worker.__new__(Worker); w.proc = None; w.job = 101; w.process = 102; w._lock = threading.Lock()
        def accounting(job, kind, pointer, size, returned):
            ctypes.cast(pointer, ctypes.POINTER(JobAccounting)).contents.active = active
            return query
        w.k = SimpleNamespace(TerminateJobObject=Mock(return_value=terminate),
            WaitForSingleObject=Mock(return_value=wait), QueryInformationJobObject=Mock(side_effect=accounting),
            CloseHandle=Mock())
        return w
    def test_job_and_root_must_both_be_confirmed_stopped(self):
        w = self.worker(); self.assertEqual(w.stop(), {'owned_job_empty': True})
        self.assertIsNone(w.job); self.assertIsNone(w.process)
    def test_terminate_wait_query_and_descendant_failures_retain_handles(self):
        for kwargs in (dict(terminate=False), dict(wait=258), dict(query=False), dict(active=1)):
            w = self.worker(**kwargs)
            with self.subTest(kwargs=kwargs), patch('strata_control.process.time.monotonic', side_effect=[0, 6]):
                with self.assertRaises(AdmissionError) as failure: w.stop()
            self.assertIs(failure.exception.worker, w)
            self.assertEqual(w.job, 101); self.assertEqual(w.process, 102); w.k.CloseHandle.assert_not_called()
    def test_resource_release_failure_retains_worker_and_lease_until_retry(self):
        with tempfile.TemporaryDirectory() as d:
            c = controller(d); c.load('fixture', timeout=1); w = c.worker
            c.release_probe = Mock(side_effect=AdmissionError('fixture resource uncertainty'))
            with self.assertRaises(AdmissionError): c.unload()
            self.assertIs(c.worker, w); self.assertEqual(c.state, 'failed')
            self.assertTrue(c.lease.handle or c.lease.file)
            with self.assertRaises(Busy): c.load('fixture')
            c.release_probe = lambda p: None; c.unload(); self.assertEqual(c.state, 'idle')
    def test_stop_failure_retains_lease(self):
        with tempfile.TemporaryDirectory() as d:
            c = controller(d); c.load('fixture', timeout=1); stop = c.worker.stop
            c.worker.stop = Mock(side_effect=AdmissionError('fixture job failure'))
            with self.assertRaises(AdmissionError): c.unload()
            self.assertTrue(c.lease.handle or c.lease.file)
            c.worker.stop = stop; c.unload()
    def test_failed_readiness_keeps_lease_when_cleanup_is_uncertain(self):
        with tempfile.TemporaryDirectory() as d:
            owned = []
            def factory(*args):
                w = FakeWorker(*args); owned.append((w, w.stop))
                w.stop = Mock(side_effect=AdmissionError('fixture job failure'))
                return w
            c = controller(d, worker_factory=factory)
            c._get = Mock(side_effect=AdmissionError('fixture wrong model'))
            with self.assertRaises(AdmissionError): c.load('fixture', timeout=1)
            self.assertIs(c.worker, owned[0][0]); self.assertEqual(c.state, 'failed')
            self.assertTrue(c.lease.handle or c.lease.file)
            c.worker.stop = owned[0][1]; c.unload()
    def test_partial_constructor_failure_preserves_ownership_for_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            owned = []
            def factory(*args):
                worker = FakeWorker(*args); owned.append(worker)
                error = AdmissionError('fixture partial start'); error.worker = worker
                raise error
            c = controller(d, worker_factory=factory)
            with self.assertRaises(AdmissionError): c.load('fixture', timeout=1)
            self.assertTrue(owned[0].dead); self.assertIsNone(c.worker)
            self.assertFalse(c.lease.handle or c.lease.file); c.unload()


class CapabilityTests(unittest.TestCase):
    def test_messages_beta_query_body_and_headers_reach_backend_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p = profile(); p['backend'] = 'maya-glm'
            c = controller(d, p); c._get = lambda path: copy.deepcopy(FIXTURE['health' if path == '/health' else 'models'])
            c.load('fixture', timeout=1)
            key = 'fixture-only-controller-key-123456'
            api = HTTP(('127.0.0.1', 0), make_handler(c, key))
            thread = threading.Thread(target=api.serve_forever, daemon=True); thread.start()
            try:
                conn = http.client.HTTPConnection('127.0.0.1', api.server_port)
                raw = b'{ "tool_choice":{"type":"auto"},"messages":[] }'
                conn.request('POST', '/v1/messages?beta=true', raw, {
                    'Authorization': 'Bearer ' + key, 'anthropic-version': '2023-06-01',
                    'anthropic-beta': 'fixture-beta'})
                response = conn.getresponse(); self.assertEqual(response.status, 200); response.read(); conn.close()
                self.assertEqual(c.worker.targets, ['/v1/messages?beta=true'])
                self.assertEqual(c.worker.posts, [raw])
                self.assertEqual(c.worker.received_headers[0]['anthropic-version'], '2023-06-01')
                self.assertEqual(c.worker.received_headers[0]['anthropic-beta'], 'fixture-beta')
            finally:
                api.shutdown(); api.server_close(); thread.join(); c.unload()

    def test_dialect_specific_tools_queries_and_constraints(self):
        with tempfile.TemporaryDirectory() as d:
            p = profile(); p['backend'] = 'maya-glm'
            c = controller(d, p); c._get = lambda path: copy.deepcopy(FIXTURE['health' if path == '/health' else 'models'])
            c.load('fixture', timeout=1)
            try:
                for path, obj in [('/v1/messages?beta=true', {'tool_choice': {'type': 'auto'}}),
                                  ('/v1/messages', {'tool_choice': {'type': 'none'}}),
                                  ('/v1/responses', {'text': {'format': {'type': 'text'}}})]:
                    with c.request_slot(path, obj): pass
                bad = [('/v1/completions', {}), ('/v1/chat/completions', {'json_schema': {}}),
                       ('/v1/chat/completions', {'response_format': {}}),
                       ('/v1/messages', {'tool_choice': {'type': 'any'}}),
                       ('/v1/messages', {'tool_choice': {'type': 'tool', 'name': 'x'}}),
                       ('/v1/messages', {'tool_choice': 'auto'}),
                       ('/v1/responses', {'text': {'format': {'type': 'json_schema'}}}),
                       ('/v1/responses', {'tool_choice': 'required'})]
                for path, obj in bad:
                    with self.subTest(path=path, obj=obj), self.assertRaises(AdmissionError):
                        with c.request_slot(path, obj): pass
            finally: c.unload()
    def test_unvalidated_query_and_absolute_targets_are_rejected(self):
        for path in ('http://localhost/v1/messages?beta=true', '//localhost/v1/messages',
                     '/v1/messages?beta=true&beta=false', '/v1/messages?other=true',
                     '/v1/messages#x', '/v1/chat/completions?beta=true'):
            with self.subTest(path=path), self.assertRaises(AdmissionError): validated_route(path)


class CancellationTests(unittest.TestCase):
    def test_disconnect_before_connect_during_connect_upload_and_before_first_byte(self):
        for stage in ('before-connect', 'during-connect', 'upload', 'before-first-byte'):
            token = Cancellation(); transport = Mock()
            class Connection:
                sock = transport
                posts = 0
                uploads = 0
                def connect(self):
                    if stage == 'during-connect': token.cancel()
                def putrequest(self, *args): self.posts += 1
                def putheader(self, *args): pass
                def endheaders(self): pass
                def send(self, raw):
                    self.uploads += 1
                    if stage == 'upload': token.cancel()
                def getresponse(self):
                    if stage == 'before-first-byte': token.cancel()
                    return object()
            connection = Connection()
            if stage == 'before-connect': token.cancel()
            with self.subTest(stage=stage), self.assertRaises(RequestCancelled):
                forward_request(connection, '/v1/messages?beta=true', b'x' * 131072, {}, token)
            if stage in ('before-connect', 'during-connect'): self.assertEqual(connection.posts, 0)
            if stage != 'before-connect': transport.shutdown.assert_called_once()
            if stage == 'upload': self.assertEqual(connection.uploads, 1)
    def test_cancel_and_bind_are_thread_safe(self):
        token = Cancellation(); transport = Mock()
        thread = threading.Thread(target=token.cancel); thread.start(); thread.join()
        with self.assertRaises(RequestCancelled): token.bind(transport)
        transport.shutdown.assert_called_once()


if __name__ == '__main__': unittest.main()
