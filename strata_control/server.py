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
from .metadata import AdmissionError, inspect_gguf
from .process import DeviceLease, Worker, require_idle_gpu
from .registry import CAPS, admit, load_profiles, worker_command


class Busy(AdmissionError):
    pass


class Controller:
    def __init__(self, profiles, runtime_dir, validator=admit, idle_probe=require_idle_gpu,
                 worker_factory=Worker, command_factory=worker_command):
        self.profiles = profiles
        self.runtime = Path(runtime_dir).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.validate, self.idle_probe = validator, idle_probe
        self.worker_factory, self.command_factory = worker_factory, command_factory
        self.lock = threading.RLock()
        self.operation = threading.Lock()
        self.lease = DeviceLease(self.runtime)
        self.state, self.current, self.worker, self.port = 'idle', None, None, None
        self.active = 0
        self.error = None
        self.backend_key = None

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
            self.backend_key = secrets.token_urlsafe(32)
            cmd, cwd, env = self.command_factory(p, port, run, config, self.backend_key)
            with self.lock:
                self.state, self.current, self.port = 'loading', p, port
                self.worker = self.worker_factory(cmd, cwd, env)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.worker.poll() is not None:
                    raise AdmissionError('owned worker exited before readiness')
                try:
                    health = self._get('/health')
                    models = self._get('/v1/models')
                    ids = [x.get('id') for x in models.get('data', [])]
                    allocated = health.get('max_context', health.get('n_ctx'))
                    if health.get('status') == 'ok' and health.get('loaded') is True and name in ids:
                        if allocated != p['context']:
                            raise AdmissionError('actual allocated context differs or is unreported')
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
                    if self.worker:
                        self.worker.stop(); self.worker = None
                    self.current, self.port = None, None
                    self.backend_key = None
                self.lease.release()
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
                if self.worker:
                    self.worker.stop(); self.worker = None
                self.current, self.port, self.error = None, None, None
                self.backend_key = None
                self.lease.release()
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
            if path not in allowed:
                raise AdmissionError('endpoint unsupported by this intact backend')
            if 'model' in obj and obj['model'] != p['id']:
                raise Busy('request model differs from the explicitly loaded model')
            if self.active:
                raise Busy('single-request admission is busy')
            if any(k.startswith(('strata_', 'maya_', 'ds4_')) for k in obj):
                raise AdmissionError('per-request engine tuning is prohibited')
            if obj.get('background') or obj.get('response_format') or obj.get('grammar') or obj.get('tool_choice') not in (None, 'auto', 'none'):
                raise AdmissionError('background or constrained generation is not qualified')
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
                    disconnected = threading.Event()
                    transport = [None]
                    def watch_disconnect():
                        while not finished.wait(0.1):
                            try:
                                ready, _, _ = select.select([self.connection], [], [], 0)
                                if ready and self.connection.recv(1, socket.MSG_PEEK) == b'':
                                    disconnected.set()
                                    if transport[0]:
                                        transport[0].shutdown(socket.SHUT_RDWR)
                                    return
                            except OSError:
                                return
                    watcher = threading.Thread(target=watch_disconnect, daemon=True)
                    watcher.start()
                    try:
                        # Forward original JSON bytes. Never render templates or reformat tools.
                        upstream.request('POST', self.path, raw, {'Content-Type': 'application/json',
                                         'Authorization': 'Bearer ' + controller.backend_key})
                        transport[0] = upstream.sock
                        r = upstream.getresponse()
                        self.send_response(r.status)
                        self.send_header('Content-Type', r.getheader('Content-Type', 'application/json'))
                        self.send_header('Connection', 'close')
                        self.end_headers(); started = True
                        while chunk := r.read1(16384):
                            self.wfile.write(chunk); self.wfile.flush()
                    except (OSError, http.client.HTTPException):
                        # Partial tool streams are closed. No automatic replay or fallback.
                        if not started and not disconnected.is_set():
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
            # Process exit/job handles own only our workers; never target external PIDs.
            if controller.worker:
                controller.worker.stop()
            controller.lease.release()
        else:
            controller.unload()
    return 0
