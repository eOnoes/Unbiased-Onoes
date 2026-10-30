// Win32 transport for the engine's unchanged line protocol.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata_frontend {
class WindowsChild {
public:
    ~WindowsChild() { quit(); }
    static std::wstring wide(const std::string & s) {
        int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), nullptr, 0);
        if (n == 0 && !s.empty()) return {};
        std::wstring w(n, L'\0');
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), w.data(), n);
        return w;
    }
    // CommandLineToArgv/CRT quoting, including empty args and trailing backslashes.
    static std::wstring quote(const std::wstring & s) {
        std::wstring q = L"\"";
        size_t slashes = 0;
        for (wchar_t c : s) {
            if (c == L'\\') { ++slashes; continue; }
            q.append(slashes * (c == L'\"' ? 2 : 1), L'\\');
            slashes = 0;
            if (c == L'\"') q += L'\\';
            q += c;
        }
        q.append(slashes * 2, L'\\');
        return q + L'\"';
    }
    bool start(const std::string & executable, const std::vector<std::string> & args, std::string & err) {
        if (process_) { err = "engine already started"; return false; }
        HANDLE child_in = nullptr, child_out = nullptr, child_err = nullptr;
        auto close = [](HANDLE & h) { if (h && h != INVALID_HANDLE_VALUE) CloseHandle(h); h = nullptr; };
        auto fail = [&](const char * op) {
            err = std::string(op) + ": Win32 error " + std::to_string(GetLastError());
            close(child_in); close(child_out); close(child_err); close(input_); close(output_); close(job_);
            return false;
        };
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        if (!CreatePipe(&child_in, &input_, &sa, 65536) || !CreatePipe(&output_, &child_out, &sa, 65536)) return fail("CreatePipe");
        if (!SetHandleInformation(input_, HANDLE_FLAG_INHERIT, 0) ||
            !SetHandleInformation(output_, HANDLE_FLAG_INHERIT, 0)) return fail("SetHandleInformation");
        if (!DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE), GetCurrentProcess(), &child_err,
                             0, TRUE, DUPLICATE_SAME_ACCESS)) return fail("DuplicateHandle(stderr)");
        job_ = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!job_ || !SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &limits, sizeof limits)) return fail("CreateJobObject");
        STARTUPINFOEXW si{};
        si.StartupInfo.cb = sizeof si;
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        si.StartupInfo.hStdInput = child_in; si.StartupInfo.hStdOutput = child_out; si.StartupInfo.hStdError = child_err;
        SIZE_T bytes = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
        std::vector<unsigned char> attr(bytes);
        si.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr.data());
        if (!InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &bytes)) return fail("InitializeProcThreadAttributeList");
        HANDLE inherited[] = {child_in, child_out, child_err};
        bool attr_ok = UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                                inherited, sizeof inherited, nullptr, nullptr) != FALSE;
        std::wstring cmd;
        for (const auto & a : args) { if (!cmd.empty()) cmd += L' '; cmd += quote(wide(a)); }
        const std::wstring exe = wide(executable);
        PROCESS_INFORMATION pi{};
        const bool created = attr_ok && CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE,
            CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &si.StartupInfo, &pi);
        const DWORD create_error = GetLastError();
        DeleteProcThreadAttributeList(si.lpAttributeList);
        if (!created) { SetLastError(create_error); return fail("CreateProcessW"); }
        if (!AssignProcessToJobObject(job_, pi.hProcess)) {
            const DWORD code = GetLastError();
            TerminateProcess(pi.hProcess, 1); WaitForSingleObject(pi.hProcess, INFINITE);
            CloseHandle(pi.hThread); CloseHandle(pi.hProcess); SetLastError(code);
            return fail("AssignProcessToJobObject");
        }
        process_ = pi.hProcess; pid_ = pi.dwProcessId;
        close(child_in); close(child_out); close(child_err);
        stopping_ = false; failed_ = false;
        writer_ = std::thread([this] { write_loop(); });
        if (ResumeThread(pi.hThread) == DWORD(-1)) {
            CloseHandle(pi.hThread); err = "ResumeThread failed"; quit(); return false;
        }
        CloseHandle(pi.hThread);
        return true;
    }
    bool send(const std::string & s) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // Bound abandoned requests even when the child has stopped reading stdin.
        if (stopping_ || failed_ || !input_ || queued_bytes_ + s.size() > 8 * 1024 * 1024) return false;
        queued_bytes_ += s.size(); queue_.push_back(s); wake_.notify_one(); return true;
    }
    // No blocking ReadFile unless PeekNamedPipe proved data is available.
    int read(char * dst, DWORD cap, int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        do {
            DWORD available = 0, n = 0;
            if (!output_ || !PeekNamedPipe(output_, nullptr, 0, nullptr, &available, nullptr)) return -1;
            if (available) return ReadFile(output_, dst, std::min(cap, available), &n, nullptr) && n ? (int) n : -1;
            if (timeout_ms <= 0 || std::chrono::steady_clock::now() >= deadline) return 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (true);
    }
    bool running(std::string & why) {
        std::lock_guard<std::mutex> lock(process_mutex_);
        if (!process_) return false;
        if (WaitForSingleObject(process_, 0) == WAIT_TIMEOUT) return true;
        DWORD code = 0; GetExitCodeProcess(process_, &code);
        why = "engine exited with code " + std::to_string(code); return false;
    }
    void quit() {
        std::lock_guard<std::mutex> lock(process_mutex_);
        if (!process_) return;
        send("QUIT\n");
        // A child blocked on stdout or stdin is still owned by this job only.
        if (WaitForSingleObject(process_, 2000) == WAIT_TIMEOUT) {
            TerminateJobObject(job_, 1); WaitForSingleObject(process_, 2000);
        }
        { std::lock_guard<std::mutex> qlock(queue_mutex_); stopping_ = true; queue_.clear(); queued_bytes_ = 0; }
        wake_.notify_all();
        if (writer_.joinable()) { CancelSynchronousIo(writer_.native_handle()); writer_.join(); }
        CloseHandle(input_); CloseHandle(output_); CloseHandle(process_); CloseHandle(job_);
        input_ = output_ = process_ = job_ = nullptr; pid_ = 0;
    }
    DWORD pid() const { return pid_; }
private:
    void write_loop() {
        for (;;) {
            std::string s;
            { std::unique_lock<std::mutex> lock(queue_mutex_);
              wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
              if (stopping_) return;
              s = std::move(queue_.front()); queue_.pop_front(); queued_bytes_ -= s.size(); }
            size_t off = 0;
            while (off < s.size()) {
                DWORD n = 0;
                if (!WriteFile(input_, s.data() + off, (DWORD) std::min<size_t>(65536, s.size() - off), &n, nullptr) || !n) {
                    failed_ = true; return;
                }
                off += n;
            }
        }
    }
    HANDLE process_ = nullptr, job_ = nullptr, input_ = nullptr, output_ = nullptr;
    DWORD pid_ = 0;
    std::thread writer_;
    std::mutex process_mutex_, queue_mutex_;
    std::condition_variable wake_;
    std::deque<std::string> queue_;
    size_t queued_bytes_ = 0;
    bool stopping_ = true;
    std::atomic<bool> failed_{false};
};
} // namespace strata_frontend
