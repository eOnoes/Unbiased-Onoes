"""Owned workers and a single-device lease; no process-by-name termination."""
import ctypes
import os
import signal
import subprocess
import threading
from pathlib import Path
from .metadata import AdmissionError


class DeviceLease:
    def __init__(self, directory, device=0):
        self.path = Path(directory) / ('gpu-' + str(device) + '.lock')
        self.handle = None
        self.file = None
        self.thread = None
        self.release_event = threading.Event()

    def acquire(self):
        if self.handle or self.file:
            raise AdmissionError('device lease already held')
        if os.name == 'nt':
            ready = threading.Event()
            result = []
            self.release_event.clear()
            def owner():
                # Windows mutex ownership belongs to a thread. Keep one owner
                # thread alive across HTTP requests and release on that thread.
                k = ctypes.WinDLL('kernel32', use_last_error=True)
                k.CreateMutexW.restype = ctypes.c_void_p
                k.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
                k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                h = k.CreateMutexW(None, False, 'Local\\UnbiasedOnoesGpu0')
                acquired = bool(h) and k.WaitForSingleObject(ctypes.c_void_p(h), 0) in (0, 128)
                self.handle = h if acquired else None
                result.append(acquired); ready.set()
                if acquired:
                    self.release_event.wait()
                    k.ReleaseMutex(ctypes.c_void_p(h)); self.handle = None
                if h:
                    k.CloseHandle(ctypes.c_void_p(h))
            self.thread = threading.Thread(target=owner, daemon=True)
            self.thread.start(); ready.wait()
            if not result[0]:
                self.thread.join(); self.thread = None
                raise AdmissionError('another controller owns GPU 0')
        else:
            import fcntl
            self.path.parent.mkdir(parents=True, exist_ok=True)
            f = self.path.open('a+')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                raise AdmissionError('another controller owns the device')
            self.file = f

    def release(self):
        if self.handle:
            self.release_event.set()
            self.thread.join(timeout=5)
            self.thread = None
        if self.file:
            self.file.close()
            self.file = None


def require_idle_gpu():
    # Read-only probe occurs only on an explicit future load, never at import or
    # controller startup. Failure/unsupported WDDM accounting fails closed.
    try:
        p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as e:
        raise AdmissionError('cannot establish external GPU ownership; load refused') from e
    lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    if lines:
        raise AdmissionError('GPU has external compute owners or unsupported ownership accounting')
    # WDDM can omit owners from compute-apps. Conservatively reject known
    # inference processes as well; never stop any discovered PID.
    if os.name == 'nt':
        try:
            names = subprocess.check_output(['tasklist', '/FO', 'CSV', '/NH'], text=True, timeout=10).lower()
        except (OSError, subprocess.SubprocessError) as e:
            raise AdmissionError('cannot inspect external inference ownership') from e
        if any('"' + x + '"' in names for x in ('maya.exe', 'maya-ds4.exe', 'maya-mm3.exe',
                                                  'strata.exe', 'strata-poly.exe', 'llama-server.exe')):
            raise AdmissionError('external inference runtime is active; no takeover permitted')


class Worker:
    def __init__(self, argv, cwd, env):
        self.proc = None
        self.process = self.job = None
        self._lock = threading.Lock()
        if os.name != 'nt':
            self.proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         start_new_session=True)
            return
        from ctypes import wintypes as w
        k = self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        class SI(ctypes.Structure):
            _fields_ = [('cb', w.DWORD), ('reserved', w.LPWSTR), ('desktop', w.LPWSTR), ('title', w.LPWSTR),
                        ('x', w.DWORD), ('y', w.DWORD), ('xs', w.DWORD), ('ys', w.DWORD),
                        ('xc', w.DWORD), ('yc', w.DWORD), ('fill', w.DWORD), ('flags', w.DWORD),
                        ('show', w.WORD), ('reserved2', w.WORD), ('resptr', ctypes.c_void_p),
                        ('stdin', w.HANDLE), ('stdout', w.HANDLE), ('stderr', w.HANDLE)]
        class PI(ctypes.Structure):
            _fields_ = [('process', w.HANDLE), ('thread', w.HANDLE), ('pid', w.DWORD), ('tid', w.DWORD)]
        class BASIC(ctypes.Structure):
            _fields_ = [('perprocess', ctypes.c_int64), ('perjob', ctypes.c_int64), ('flags', w.DWORD),
                        ('minws', ctypes.c_size_t), ('maxws', ctypes.c_size_t), ('active', w.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(x, ctypes.c_uint64) for x in ('ro', 'wo', 'oo', 'rb', 'wb', 'ob')]
        class EXT(ctypes.Structure):
            _fields_ = [('basic', BASIC), ('io', IO), ('pm', ctypes.c_size_t), ('jm', ctypes.c_size_t),
                        ('peakp', ctypes.c_size_t), ('peakj', ctypes.c_size_t)]
        k.CreateJobObjectW.restype = w.HANDLE
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        k.CreateProcessW.argtypes = [w.LPCWSTR, w.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
                                    w.BOOL, w.DWORD, ctypes.c_void_p, w.LPCWSTR,
                                    ctypes.POINTER(SI), ctypes.POINTER(PI)]
        for name in ('AssignProcessToJobObject',):
            getattr(k, name).argtypes = [w.HANDLE, w.HANDLE]
        k.ResumeThread.argtypes = [w.HANDLE]
        k.ResumeThread.restype = w.DWORD
        k.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        k.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        k.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
        k.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        k.CloseHandle.argtypes = [w.HANDLE]
        self.job = k.CreateJobObjectW(None, None)
        limits = EXT(); limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.job or not k.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.stop()
            raise OSError('cannot create owned worker job')
        si = SI(); si.cb = ctypes.sizeof(si); pi = PI()
        block = ctypes.create_unicode_buffer('\0'.join(k + '=' + v for k, v in sorted(env.items())) + '\0\0')
        cmd = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
        if not k.CreateProcessW(str(Path(argv[0]).resolve()), cmd, None, None, False,
                                0x08000000 | 0x00000004 | 0x00000400, block, str(cwd),
                                ctypes.byref(si), ctypes.byref(pi)):
            self.stop()
            raise OSError('cannot start owned worker')
        self.process = pi.process
        if not k.AssignProcessToJobObject(self.job, self.process):
            k.TerminateProcess(self.process, 1)
            k.CloseHandle(pi.thread)
            self.stop()
            raise OSError('cannot assign worker job; no unowned worker allowed')
        if k.ResumeThread(pi.thread) == 0xffffffff:
            k.CloseHandle(pi.thread); self.stop()
            raise OSError('cannot resume worker')
        k.CloseHandle(pi.thread)

    def poll(self):
        if self.proc:
            return self.proc.poll()
        if not self.process:
            return 1
        from ctypes import wintypes as w
        code = w.DWORD()
        self.k.GetExitCodeProcess(self.process, ctypes.byref(code))
        return None if code.value == 259 else code.value

    def stop(self):
        with self._lock:
            if self.proc:
                # Only the process group created by this Worker is targeted.
                if self.proc.poll() is None:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                        self.proc.wait(timeout=5)
                self.proc = None
            if self.job:
                self.k.TerminateJobObject(self.job, 1)
                if self.process:
                    self.k.WaitForSingleObject(self.process, 5000)
                self.k.CloseHandle(self.job); self.job = None
            if self.process:
                self.k.CloseHandle(self.process); self.process = None
