// Real native child processes, no GPU/model/Python. Quoting, bounded writes,
// read deadlines, failed launch, concurrent liveness and child-only teardown.
#include "engine.hpp"
#include <iostream>
#include <thread>
#include <filesystem>

int wmain(int argc, wchar_t ** argv) {
    if (argc > 1 && std::wstring(argv[1]) == L"--echo") {
        if (argc != 6 || std::wstring(argv[2]) != L"" || std::wstring(argv[3]) != L"a b\\" ||
            std::wstring(argv[4]) != L"x\"y\\\\" || std::wstring(argv[5]) != L"\u03a9") return 9;
        std::printf("INFO quote=ok\nREADY 512 stop\n"); std::fflush(stdout);
        char b[1024];
        while (std::fgets(b, sizeof b, stdin)) if (std::strncmp(b, "QUIT", 4) == 0) break;
        return 0;
    }
    if (argc > 1 && std::wstring(argv[1]) == L"--stall") {
        std::printf("READY 512 stop\n"); std::fflush(stdout);
        Sleep(60000); return 0;
    }
    wchar_t self[32768]; GetModuleFileNameW(nullptr, self, 32768);
    const auto u = std::filesystem::path(self).u8string();
    const std::string exe(reinterpret_cast<const char *>(u.data()), u.size());
    int fail = 0;
    auto ck = [&](bool ok) { if (!ok) ++fail; };
    std::string err;
    {
        strata_frontend::Engine e;
        ck(e.start(exe, {exe, "--echo", "", "a b\\", "x\"y\\\\", "\xCE\xA9"}, err));
        ck(e.wait_ready(err, 3)); ck(e.info().count("quote") == 1);
        strata_frontend::EngineLine line;
        const auto t = e.now_ms(); ck(!e.next_line(line, 50)); ck(e.now_ms() - t < 500);
        e.quit();
    }
    {
        strata_frontend::Engine e;
        ck(!e.start(exe + ".absent", {exe}, err)); ck(!err.empty());
    }
    {
        strata_frontend::Engine e;
        ck(e.start(exe, {exe, "--stall"}, err)); ck(e.wait_ready(err, 3));
        const DWORD pid = e.pid();
        const auto t = e.now_ms();
        ck(e.send_request(std::string(1024 * 1024, 'x'), {1}));
        ck(e.now_ms() - t < 500);
        std::atomic<bool> stop{false};
        std::thread poller([&] { std::string why; while (!stop) { e.reap_exit(why); Sleep(1); } });
        e.quit(); stop = true; poller.join();
        HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, pid);
        ck(!h || WaitForSingleObject(h, 0) == WAIT_OBJECT_0);
        if (h) CloseHandle(h);
        ck(e.now_ms() - t < 5000);
    }
    std::cout << "windows_child_test: " << fail << " failures\n";
    return fail ? 1 : 0;
}
