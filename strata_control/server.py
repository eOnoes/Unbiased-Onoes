"""One loopback API, explicit loads, intact backend HTTP, and no request replay."""
import argparse
from contextlib import contextmanager
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import select
import secrets
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit, parse_qsl
from .metadata import AdmissionError, inspect_gguf
from .process import DeviceLease, Worker, require_idle_gpu
from .registry import CAPS, admit, load_profiles, worker_command, cleanup_runtime_config, require_resources


def ready_backend(profile, health, models):
    """Contracts from pinned Maya 1.1.41 and the preserved C++ frontend."""
    if health.get('status') != 'ok':
        return False
    name, context = profile['id'], profile['context']
    if profile['backend'] == 'maya-glm':
        if health.get('unloaded') is not False:
            return False
        if health.get('version') != '1.1.41':
            raise AdmissionError('Maya health version mismatch')
    elif health.get('service') != 'strata' or health.get('loaded') is not True:
        return False
    if health.get('model') != name or health.get('max_context') != context:
        raise AdmissionError('backend health model/context mismatch')
    data = models.get('data', [])
    if models.get('object') != 'list' or len(data) != 1 or data[0].get('id') != name:
        raise AdmissionError('backend models identity mismatch')
    if profile['backend'] == 'maya-glm':
        model = data[0]
        if model.get('status', {}).get('value') != 'loaded':
            return False
        if model.get('context_length') != context or model.get('meta', {}).get('n_ctx') != context:
            raise AdmissionError('Maya model context mismatch')
    return True


def validated_route(target):
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or parsed.fragment or not target.startswith('/'):
        raise AdmissionError('invalid request target')
    if parsed.query:
        if parsed.path != '/v1/messages' or parse_qsl(parsed.query, keep_blank_values=True) not in (
                [('beta', 'true')], [('beta', 'false')]):
            raise AdmissionError('unsupported query parameters')
    return parsed.path


def validate_request_policy(path, obj):
    if obj.get('background') or any(k in obj for k in ('response_format', 'grammar', 'json_schema')):
        raise AdmissionError('background or constrained generation is not qualified')
    choice = obj.get('tool_choice')
    if path == '/v1/messages':
        if choice is not None and choice not in ({'type': 'auto'}, {'type': 'none'}):
            raise AdmissionError('Anthropic forced tool selection is not qualified')
    elif choice not in (None, 'auto', 'none'):
        raise AdmissionError('forced tool selection is not qualified')
    if path == '/v1/responses':
        text = obj.get('text', {})
        if not isinstance(text, dict) or ('format' in text and text['format'] != {'type': 'text'}):
            raise AdmissionError('Responses constrained text format is not qualified')


class RequestCancelled(OSError):
    pass


def forward_request(upstream, target, raw, headers, cancellation):
    cancellation.check()
    upstream.connect()
    cancellation.bind(upstream.sock)
    cancellation.check()
    upstream.putrequest('POST', target)
    for name, value in headers.items():
        upstream.putheader(name, value)
    upstream.putheader('Content-Length', str(len(raw)))
    cancellation.check()
    upstream.endheaders()
    for offset in range(0, len(raw), 65536):
        cancellation.check()
        upstream.send(raw[offset:offset + 65536])
    cancellation.check()
    response = upstream.getresponse()
    cancellation.check()
    return response


class Cancellation:
    """Bind transport atomically even when disconnect wins the connect race."""
    def __init__(self):
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.transport = None

    def check(self):
        if self.cancelled.is_set():
            raise RequestCancelled('client disconnected')

    def bind(self, transport):
        with self.lock:
            self.transport = transport
            if self.cancelled.is_set():
                self._shutdown()
                raise RequestCancelled('client disconnected during connect')

    def _shutdown(self):
        if self.transport:
            try:
                self.transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            self._shutdown()


class Busy(AdmissionError):
    pass


class Controller:
    def __init__(self, profiles, runtime_dir, validator=admit, idle_probe=require_idle_gpu,
                 worker_factory=Worker, command_factory=worker_command, release_probe=None,
                 secret_factory=lambda: secrets.token_urlsafe(32)):
        self.profiles = profiles
        self.runtime = Path(runtime_dir).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.validate, self.idle_probe = validator, idle_probe
        self.worker_factory, self.command_factory = worker_factory, command_factory
        self.release_probe = release_probe or self._resources_released
        self.secret_factory = secret_factory
        self.lock = threading.RLock()
        self.operation = threading.Lock()
        self.lease = DeviceLease(self.runtime)
        self.state, self.current, self.worker, self.port = 'idle', None, None, None
        self.active = 0
        self.error = None
        self.backend_key = None
        # Acquire the same global lease before inspecting/deleting stale owned
        # configs, so another controller's active configuration is never touched.
        self.lease.acquire()
        try:
            for name in profiles:
                cleanup_runtime_config(self.runtime / name, name)
        finally:
            self.lease.release()

    def _resources_released(self, profile):
        self.idle_probe()
        require_resources(profile)

    def _retire(self):
        """Keep worker and lease on any uncertain termination/resource result."""
        if self.worker:
            evidence = self.worker.stop()
            if not isinstance(evidence, dict) or evidence.get('owned_job_empty') is not True:
                raise AdmissionError('owned worker release is unverified; GPU lease retained')
            self.release_probe(self.current)
        if self.current:
            cleanup_runtime_config(self.runtime / self.current['id'], self.current['id'])
        self.worker, self.current, self.port, self.backend_key = None, None, None, None
        self.lease.release()

    def status(self):
        with self.lock:
            if self.state == 'ready' and self.worker.poll() is not None:
                self.state, self.error = 'failed', 'owned backend exited; explicit unload/reload required'
            return {'state': self.state, 'model': self.current['id'] if self.current else None,
                    'backend': self.current['backend'] if self.current else None,
                    'context': self.current['context'] if self.current else None,
                    'active_requests': self.active, 'error': self.error,
                    'qualification': 'CPU/fake-worker integration only; real backends require scheduled validation'}

    def load(self, name, timeout=900):
        if not self.operation.acquire(blocking=False):
            raise Busy('another lifecycle operation is in progress')
        owns_attempt = False
        try:
            with self.lock:
                if self.worker or self.active or self.state not in ('idle', 'failed'):
                    raise Busy('explicitly unload the previous model before loading another')
                if name not in self.profiles:
                    raise AdmissionError('unknown registered model')
                p = self.profiles[name]
                owns_attempt = True
                self.state, self.error = 'validating', None
            config = self.validate(p)  # metadata, exact profile, modes, source/binary identity
            self.lease.acquire()
            self.idle_probe()  # never terminate an external owner
            with socket.socket() as s:
                s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
            run = self.runtime / name
            run.mkdir(exist_ok=True)
            self.backend_key = self.secret_factory()
            with self.lock:
                self.current = p
            cmd, cwd, env, _ = self.command_factory(p, port, run, config, self.backend_key)
            with self.lock:
                self.state, self.current, self.port = 'loading', p, port
                try:
                    self.worker = self.worker_factory(cmd, cwd, env)
                except Exception as e:
                    self.worker = getattr(e, 'worker', None)
                    raise
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.worker.poll() is not None:
                    raise AdmissionError('owned worker exited before readiness')
                try:
                    health = self._get('/health')
                    models = self._get('/v1/models')
                    if ready_backend(p, health, models):
                        with self.lock:
                            self.state = 'ready'
                        return self.status()
                except (OSError, http.client.HTTPException, json.JSONDecodeError):
                    pass
                time.sleep(0.05)
            raise AdmissionError('worker load deadline expired; no model substitution attempted')
        except Exception:
            if owns_attempt:
                with self.lock:
                    self.state, self.error = 'failed', 'load failed; inspect local profile and backend qualification'
                    try:
                        self._retire()
                    except Exception as cleanup_error:
                        self.error = 'cleanup unverified; worker ownership and GPU lease retained'
                        raise AdmissionError(self.error) from cleanup_error
            raise
        finally:
            self.operation.release()

    def _get(self, path):
        c = http.client.HTTPConnection('127.0.0.1', self.port, timeout=1)
        try:
            c.request('GET', path, headers={'Authorization': 'Bearer ' + self.backend_key})
            r = c.getresponse()
            if r.status != 200:
                raise OSError('backend not ready')
            return json.loads(r.read(1024 * 1024))
        finally:
            c.close()

    def unload(self):
        if not self.operation.acquire(blocking=False):
            raise Busy('another lifecycle operation is in progress')
        try:
            with self.lock:
                if self.active:
                    raise Busy('requests are active; drain first, no implicit cancellation')
                self.state = 'stopping'
                try:
                    self._retire()
                except Exception:
                    self.state, self.error = 'failed', 'release unverified; worker ownership and GPU lease retained'
                    raise
                self.error = None
                self.state = 'idle'
            return self.status()
        finally:
            self.operation.release()

    @contextmanager
    def request_slot(self, path, obj):
        with self.lock:
            status = self.status()
            if status['state'] != 'ready':
                raise Busy('no ready backend')
            p = self.current
            allowed = CAPS['maya-glm' if p['backend'] == 'maya-glm' else 'unbiased']
            path = validated_route(path)
            if path not in allowed:
                raise AdmissionError('endpoint unsupported by this intact backend')
            if 'model' in obj and obj['model'] != p['id']:
                raise Busy('request model differs from the explicitly loaded model')
            if self.active:
                raise Busy('single-request admission is busy')
            if any(k.startswith(('strata_', 'maya_', 'ds4_')) for k in obj):
                raise AdmissionError('per-request engine tuning is prohibited')
            validate_request_policy(path, obj)
            if p['backend'] != 'maya-glm' and 'top_k' in obj and not 1 <= obj['top_k'] <= 64:
                raise AdmissionError('this backend supports explicit top_k 1..64; no silent clamp')
            self.active += 1
            port = self.port
        try:
            yield port
        finally:
            with self.lock:
                self.active -= 1


class HTTP(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(controller, key):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *args):
            pass  # no prompts, API keys or tool output in access logs

        def reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            if self.headers.get('Origin') or self.headers.get('Host', '').split(':')[0] not in ('127.0.0.1', 'localhost'):
                self.reply(403, {'error': 'untrusted origin or Host'})
                return False
            value = self.headers.get('Authorization', '')
            if not hmac.compare_digest(value, 'Bearer ' + key):
                self.reply(401, {'error': 'controller bearer key required'})
                return False
            return True

        def do_GET(self):
            if self.path == '/health':
                return self.reply(200, controller.status())
            if not self.authorized():
                return
            if self.path == '/v1/models':
                current = controller.status()
                return self.reply(200, {'object': 'list', 'data': [
                    {'id': p['id'], 'object': 'model', 'backend': p['backend'],
                     'loaded': current['model'] == p['id'] and current['state'] == 'ready',
                     'qualification': p.get('qualification', 'pending'),
                     'context': p['context'], 'precision_policy': p['precision_policy'],
                     'grammar_enforcement': False}
                    for p in controller.profiles.values()]})
            if self.path == '/admin/status':
                return self.reply(200, controller.status())
            self.reply(404, {'error': 'unknown endpoint'})

        def do_POST(self):
            if not self.authorized():
                return
            try:
                if self.headers.get('Transfer-Encoding'):
                    raise AdmissionError('chunked request bodies are unsupported')
                n = int(self.headers.get('Content-Length', '-1'))
                if not 0 <= n <= 16 * 1024 * 1024:
                    raise AdmissionError('request body size is invalid')
                raw = self.rfile.read(n)
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise AdmissionError('JSON object required')
                if self.path == '/admin/load':
                    return self.reply(200, controller.load(obj.get('model')))
                if self.path == '/admin/unload':
                    return self.reply(200, controller.unload())
                with controller.request_slot(self.path, obj) as port:
                    upstream = http.client.HTTPConnection('127.0.0.1', port, timeout=1800)
                    started = False
                    finished = threading.Event()
                    cancellation = Cancellation()
                    def watch_disconnect():
                        while not finished.wait(0.1):
                            try:
                                ready, _, _ = select.select([self.connection], [], [], 0)
                                if ready and self.connection.recv(1, socket.MSG_PEEK) == b'':
                                    cancellation.cancel()
                                    return
                            except OSError:
                                cancellation.cancel()
                                return
                    watcher = threading.Thread(target=watch_disconnect, daemon=True)
                    watcher.start()
                    try:
                        # Forward original JSON bytes. Never render templates or reformat tools.
                        headers = {'Content-Type': 'application/json',
                                   'Authorization': 'Bearer ' + controller.backend_key}
                        for name in ('anthropic-version', 'anthropic-beta'):
                            if self.headers.get(name):
                                headers[name] = self.headers[name]
                        r = forward_request(upstream, self.path, raw, headers, cancellation)
                        self.send_response(r.status)
                        self.send_header('Content-Type', r.getheader('Content-Type', 'application/json'))
                        self.send_header('Connection', 'close')
                        self.end_headers(); started = True
                        while chunk := r.read1(16384):
                            cancellation.check()
                            self.wfile.write(chunk); self.wfile.flush()
                    except (OSError, http.client.HTTPException):
                        # Partial tool streams are closed. No automatic replay or fallback.
                        if not started and not cancellation.cancelled.is_set():
                            self.reply(502, {'error': 'backend connection failed; request was not replayed'})
                    finally:
                        finished.set()
                        upstream.close()
                        watcher.join(timeout=1)
            except Busy as e:
                self.reply(409, {'error': str(e)})
            except (AdmissionError, ValueError, KeyError, TypeError) as e:
                self.reply(400, {'error': str(e)})
            except Exception:
                self.reply(503, {'error': 'backend unavailable; no retry or substitution attempted'})
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect'); inspect.add_argument('gguf')
    check = sub.add_parser('check'); check.add_argument('profiles')
    serve = sub.add_parser('serve'); serve.add_argument('profiles')
    serve.add_argument('--port', type=int, default=8099)
    serve.add_argument('--runtime-dir', default='.strata-runtime')
    a = ap.parse_args(argv)
    if a.command == 'inspect':
        result = inspect_gguf(a.gguf); result.pop('tensors')
        print(json.dumps(result, indent=2)); return 0
    profiles = load_profiles(a.profiles)
    if a.command == 'check':
        print(json.dumps({'profiles': list(profiles), 'schema_valid': True,
                          'model_load_performed': False, 'artifact_admission_performed': False}))
        return 0
    if a.port == 8088 or not 1024 <= a.port <= 65535:
        raise AdmissionError('choose a controller port distinct from live Maya 8088')
    key = os.environ.get('STRATA_CONTROL_KEY', '')
    if len(key) < 24:
        raise AdmissionError('set STRATA_CONTROL_KEY to a private token of at least 24 characters')
    controller = Controller(profiles, a.runtime_dir)
    http = HTTP(('127.0.0.1', a.port), make_handler(controller, key))
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http.server_close()
        if controller.active:
            # Stop the owned worker, verify job/resource release and remove only
            # our configuration. A failure never advertises a reusable lease.
            with controller.lock:
                controller._retire()
        else:
            controller.unload()
    return 0
