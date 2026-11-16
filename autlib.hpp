// Bridge to the external affine-equivalence library
//   https://github.com/zskiley/Affine-and-linear-equivalence-algorithms
//
// For a function F_2^n -> F_2 the library returns generators of the group of
// affine maps A with f(Ax) = f(x) or f(Ax) = f(x) + 1.  Its search cannot be
// interrupted, and on some inputs (bent functions in 12 or more variables,
// for one) it runs for a very long time.  It is therefore run in a child
// process -- this same executable, started with --aut-worker -- which the
// solver polls while it goes on searching, and which is stopped when its time
// is up or when the solver is done.
//
// The library is C++20 and optional.  It is used only if NORMALITY_ALEQ is
// defined, and then autgroup.cpp (the one file that includes its headers) and
// the library's two source files have to be compiled and linked as well.
// Nothing it returns is trusted: the driver checks every map on all points.
#pragma once
#include "group.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifdef NORMALITY_ALEQ
// autgroup.cpp.  table[x] = f(x) on m variables.  Each generator comes as the
// m images of the unit vectors followed by the image of 0; `order` is the
// order of the group they generate.
bool aleq_generators(const std::vector<uint32_t>& table, int m, std::vector<std::vector<uint32_t>>& gens,
                     double& order);
#endif

static bool aut_library_built() {
#ifdef NORMALITY_ALEQ
    return true;
#else
    return false;
#endif
}

static std::string g_aut_self;  // path of this executable, for starting the worker
static void aut_set_self(const char* argv0) {
#ifdef _WIN32
    char buf[4096];
    DWORD len = GetModuleFileNameA(NULL, buf, sizeof buf);
    g_aut_self = len > 0 && len < sizeof buf ? std::string(buf, len) : std::string(argv0);
#else
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof buf - 1);
    g_aut_self = len > 0 ? std::string(buf, (size_t)len) : std::string(argv0);
#endif
}

// The child process: read f from in_path, write the generators to out_path.
// It ends itself shortly after `budget` seconds, in case the parent is gone
// by then and cannot stop it.
static int aut_worker(const char* in_path, const char* out_path, double budget) {
#ifdef NORMALITY_ALEQ
    if (budget > 0)
        std::thread([budget] {
            std::this_thread::sleep_for(std::chrono::duration<double>(budget + 5.0));
            std::_Exit(9);
        }).detach();
    FILE* fi = fopen(in_path, "rb");
    if (!fi) return 3;
    int32_t m = 0;
    if (fread(&m, sizeof m, 1, fi) != 1 || m < 1 || m > 30) { fclose(fi); return 3; }
    std::vector<u64> F(nwords(m));
    size_t got = fread(F.data(), sizeof(u64), F.size(), fi);
    fclose(fi);
    if (got != F.size()) return 3;
    std::vector<uint32_t> table((size_t)1 << m);
    for (size_t x = 0; x < table.size(); ++x) table[x] = getbit(F.data(), (u32)x);
    std::vector<std::vector<uint32_t>> gens;
    double order = 0;
    if (!aleq_generators(table, m, gens, order)) return 4;
    FILE* fo = fopen(out_path, "w");
    if (!fo) return 3;
    fprintf(fo, "ok %zu %.17g\n", gens.size(), order);
    for (const auto& g : gens) {
        for (uint32_t c : g) fprintf(fo, "%u ", c);
        fprintf(fo, "\n");
    }
    return fclose(fo) == 0 ? 0 : 3;
#else
    (void)in_path;
    (void)out_path;
    (void)budget;
    return 5;
#endif
}

#ifdef NORMALITY_ALEQ
// One run of the library in a child process.
struct AutLibJob : AutJob {
    typedef std::chrono::steady_clock clock;
    std::string in, out;
    double budget = 0;
    int m = 0;
    clock::time_point t0;
    bool running = false;
#ifdef _WIN32
    HANDLE hproc = NULL, hthread = NULL;
#else
    pid_t child = -1;
#endif
    double elapsed() const { return std::chrono::duration<double>(clock::now() - t0).count(); }
    void kill_child() {
        if (!running) return;
#ifdef _WIN32
        TerminateProcess(hproc, 9);
        WaitForSingleObject(hproc, 5000);
#else
        kill(child, SIGKILL);
        int status = 0;
        waitpid(child, &status, 0);
#endif
        running = false;
    }
    void close_all() {
        kill_child();
#ifdef _WIN32
        if (hproc) CloseHandle(hproc);
        if (hthread) CloseHandle(hthread);
        hproc = hthread = NULL;
#endif
        if (!in.empty()) std::remove(in.c_str());
        if (!out.empty()) std::remove(out.c_str());
        in.clear();
        out.clear();
    }
    ~AutLibJob() override { close_all(); }

    bool start(const u64* F, int mm, double bud) {
        static std::atomic<unsigned> counter(0);
        m = mm;
        budget = bud;
        std::string dir;
#ifdef _WIN32
        {
            char buf[MAX_PATH + 2];
            DWORD len = GetTempPathA(sizeof buf, buf);
            dir = len > 0 && len < sizeof buf ? std::string(buf, len) : std::string(".\\");
        }
        unsigned long pid = GetCurrentProcessId();
#else
        {
            const char* t = getenv("TMPDIR");
            dir = t && *t ? std::string(t) + "/" : std::string("/tmp/");
        }
        unsigned long pid = (unsigned long)getpid();
#endif
        std::string base = dir + "normality_aut_" + std::to_string(pid) + "_" + std::to_string(counter.fetch_add(1));
        in = base + ".in";
        out = base + ".out";
        FILE* f = fopen(in.c_str(), "wb");
        if (!f) return false;
        int32_t m32 = m;
        bool ok = fwrite(&m32, sizeof m32, 1, f) == 1 && fwrite(F, sizeof(u64), nwords(m), f) == nwords(m);
        ok &= fclose(f) == 0;
        if (!ok) return false;
        t0 = clock::now();
#ifdef _WIN32
        std::string cmd = "\"" + g_aut_self + "\" --aut-worker \"" + in + "\" \"" + out + "\" " + std::to_string(budget);
        std::vector<char> cl(cmd.begin(), cmd.end());
        cl.push_back(0);
        STARTUPINFOA si;
        PROCESS_INFORMATION pi;
        ZeroMemory(&si, sizeof si);
        si.cb = sizeof si;
        if (!CreateProcessA(g_aut_self.c_str(), cl.data(), NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi))
            return false;
        hproc = pi.hProcess;
        hthread = pi.hThread;
#else
        child = fork();
        if (child < 0) return false;
        if (child == 0) {
            execl(g_aut_self.c_str(), g_aut_self.c_str(), "--aut-worker", in.c_str(), out.c_str(),
                  std::to_string(budget).c_str(), (char*)NULL);
            _exit(127);
        }
#endif
        running = true;
        return true;
    }

    int poll(std::vector<AffMap>& gens, std::string& note) override {
        char buf[160];
        if (!running) { note = "library: not running"; return -1; }
        int code = -1;
        bool finished = false;
#ifdef _WIN32
        if (WaitForSingleObject(hproc, 0) == WAIT_OBJECT_0) {
            DWORD ec = 1;
            GetExitCodeProcess(hproc, &ec);
            finished = true;
            code = (int)ec;
        }
#else
        int status = 0;
        if (waitpid(child, &status, WNOHANG) == child) {
            finished = true;
            code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
        }
#endif
        double sec = elapsed();
        if (!finished) {
            if (sec < budget) return 0;
            close_all();
            snprintf(buf, sizeof buf, "library: no answer within %.0f s, stopped", budget);
            note = buf;
            return -1;
        }
        running = false;
        if (code != 0) {
            close_all();
            snprintf(buf, sizeof buf, "library: worker failed (exit code %d)", code);
            note = buf;
            return -1;
        }
        bool ok = false;
        double order = 0;
        FILE* f = fopen(out.c_str(), "r");
        if (f) {
            char tag[8] = "";
            unsigned long long ng = 0;
            if (fscanf(f, "%7s %llu %lf", tag, &ng, &order) == 3 && !strcmp(tag, "ok")) {
                ok = true;
                for (unsigned long long i = 0; i < ng && ok; ++i) {
                    AffMap a;
                    for (int k = 0; k <= m && ok; ++k) {
                        unsigned v = 0;
                        ok = fscanf(f, "%u", &v) == 1;
                        if (k < m) a.col.push_back(v); else a.t = v;
                    }
                    if (ok) gens.push_back(a);
                }
            }
            fclose(f);
        }
        close_all();
        if (!ok) { note = "library: unreadable answer"; return -1; }
        snprintf(buf, sizeof buf, "library: affine group of order %.0f, found in %.2f s", order, sec);
        note = buf;
        return 1;
    }
    void stop(std::string& note) override {
        if (running) note = "library: still computing when the search ended, stopped";
        close_all();
    }
};
#endif

// An AutStart (see group.hpp): begins computing the automorphism group of f
// with the library; null if that is not possible.
static AutJob* aut_library_start(const u64* F, int m, double budget) {
#ifdef NORMALITY_ALEQ
    AutLibJob* job = new AutLibJob();
    if (!job->start(F, m, budget)) {
        delete job;
        return nullptr;
    }
    return job;
#else
    (void)F; (void)m; (void)budget;
    return nullptr;
#endif
}

// Blocking convenience: wait for the job (used by the tests).
static bool aut_library(const u64* F, int m, double budget, std::vector<AffMap>& gens, std::string& note) {
    AutJob* job = aut_library_start(F, m, budget);
    if (!job) { note = "equivalence library not available"; return false; }
    int st;
    while ((st = job->poll(gens, note)) == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    delete job;
    return st == 1;
}
