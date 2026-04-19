// ============================================================================
// SILICON_CONTROLLER.CPP — Windows Host Controller for Living Silicon VM
//
// Connects to the Living Silicon kernel via Hyper-V serial pipe.
// Sends commands, receives telemetry and benchmark results.
//
// Build: cl /O2 /EHsc /std:c++20 silicon_controller.cpp /link /OUT:silicon_controller.exe
// Or:    g++ -O2 -std=c++20 -o silicon_controller.exe silicon_controller.cpp
//
// Usage: silicon_controller.exe [--pipe \\.\pipe\SiliconSerial] [--csv results.csv]
// ============================================================================

#ifndef _WIN32
#error "This controller is Windows-only (Hyper-V host side)"
#endif

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>

// ── Protocol constants (must match kernel ring_protocol.hpp) ──
enum class MsgType : uint32_t {
    NOP              = 0x00000000,
    CMD_PING         = 0x00000001,
    CMD_START_BENCH  = 0x00000010,
    CMD_RUN_PDE      = 0x00000011,
    CMD_RUN_TESTS    = 0x00000012,
    CMD_SET_GA       = 0x00000020,
    CMD_SHUTDOWN     = 0x000000FF,
    RSP_PONG         = 0x80000001,
    RSP_BENCH_RESULT = 0x80000010,
    RSP_BENCH_DONE   = 0x80000011,
    RSP_TEST_RESULT  = 0x80000012,
    RSP_TELEMETRY    = 0x80000020,
    RSP_GA_STATE     = 0x80000021,
    RSP_ERROR        = 0x8000FFFF,
};

#pragma pack(push, 1)
struct Message {
    MsgType  type;
    uint32_t seq;
    uint32_t payload_len;
    uint32_t flags;
    union {
        struct {
            float mutation_rate;
            float crossover_rate;
            uint32_t population_size;
            uint32_t generations;
            float phase_delta;
            float coupling_strength;
        } ga;
        struct {
            char engine_name[24];
            float time_ms;
            float us_per_step;
            float ga_fitness;
            uint32_t steps;
        } bench;
        struct {
            char test_name[24];
            uint32_t passed;
            uint32_t failed;
            float max_error;
        } test;
        struct {
            float energy;
            float coherence;
            float nd_balance;
            float system_pressure;
            uint64_t total_steps;
            float steps_per_sec;
        } telemetry;
        struct {
            uint32_t pde_id;
            uint32_t n_steps;
            float dt_override;
        } pde_cmd;
        uint8_t raw[48];
    } payload;
};
#pragma pack(pop)

static_assert(sizeof(Message) == 64, "Message must be 64 bytes");

// ============================================================================
// Serial Pipe Connection
// ============================================================================
class PipeConnection {
    HANDLE hPipe_ = INVALID_HANDLE_VALUE;

public:
    bool connect(const char* pipeName) {
        printf("[CTRL] Connecting to %s ...\n", pipeName);

        // Wait for pipe to become available
        for (int attempt = 0; attempt < 30; attempt++) {
            hPipe_ = CreateFileA(
                pipeName,
                GENERIC_READ | GENERIC_WRITE,
                0, nullptr,
                OPEN_EXISTING,
                0, nullptr);

            if (hPipe_ != INVALID_HANDLE_VALUE) {
                printf("[CTRL] Connected to Living Silicon!\n");
                // Set pipe mode to message mode if supported
                DWORD mode = PIPE_READMODE_BYTE;
                SetNamedPipeHandleState(hPipe_, &mode, nullptr, nullptr);
                return true;
            }

            DWORD err = GetLastError();
            if (err == ERROR_PIPE_BUSY) {
                WaitNamedPipeA(pipeName, 2000);
            } else {
                printf("[CTRL] Waiting for kernel... (attempt %d/30, err=%lu)\n",
                       attempt+1, err);
                Sleep(1000);
            }
        }
        printf("[CTRL] FAILED to connect.\n");
        return false;
    }

    bool send(const Message& msg) {
        DWORD written = 0;
        return WriteFile(hPipe_, &msg, sizeof(msg), &written, nullptr) && written == sizeof(msg);
    }

    bool recv(Message& msg, DWORD timeout_ms = 5000) {
        // Use overlapped I/O for timeout
        DWORD available = 0;
        auto start = std::chrono::steady_clock::now();

        while (true) {
            if (PeekNamedPipe(hPipe_, nullptr, 0, nullptr, &available, nullptr)) {
                if (available >= sizeof(Message)) {
                    DWORD read = 0;
                    if (ReadFile(hPipe_, &msg, sizeof(msg), &read, nullptr) && read == sizeof(msg))
                        return true;
                }
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if ((DWORD)elapsed >= timeout_ms) return false;
            Sleep(10);
        }
    }

    void disconnect() {
        if (hPipe_ != INVALID_HANDLE_VALUE) {
            CloseHandle(hPipe_);
            hPipe_ = INVALID_HANDLE_VALUE;
        }
    }

    ~PipeConnection() { disconnect(); }
};

// ============================================================================
// Result Logger
// ============================================================================
class ResultLogger {
    FILE* csv_ = nullptr;

public:
    bool open(const char* path) {
        csv_ = fopen(path, "w");
        if (!csv_) return false;
        fprintf(csv_, "timestamp,type,engine,time_ms,us_per_step,ga_fitness,steps,energy,coherence,nd_balance,pressure,total_steps,steps_per_sec\n");
        fflush(csv_);
        return true;
    }

    void log_bench(const Message& m) {
        if (!csv_) return;
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        fprintf(csv_, "%lld,bench,%.24s,%.2f,%.2f,%.0f,%u,,,,,,,\n",
                (long long)now,
                m.payload.bench.engine_name,
                m.payload.bench.time_ms,
                m.payload.bench.us_per_step,
                m.payload.bench.ga_fitness,
                m.payload.bench.steps);
        fflush(csv_);
    }

    void log_telemetry(const Message& m) {
        if (!csv_) return;
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        fprintf(csv_, "%lld,telemetry,,,,,,%f,%f,%f,%f,%llu,%.1f\n",
                (long long)now,
                m.payload.telemetry.energy,
                m.payload.telemetry.coherence,
                m.payload.telemetry.nd_balance,
                m.payload.telemetry.system_pressure,
                (unsigned long long)m.payload.telemetry.total_steps,
                m.payload.telemetry.steps_per_sec);
        fflush(csv_);
    }

    void log_test(const Message& m) {
        if (!csv_) return;
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        fprintf(csv_, "%lld,test,%.24s,,,,pass=%u fail=%u maxerr=%e,,,,,,,\n",
                (long long)now,
                m.payload.test.test_name,
                m.payload.test.passed,
                m.payload.test.failed,
                m.payload.test.max_error);
        fflush(csv_);
    }

    ~ResultLogger() { if (csv_) fclose(csv_); }
};

// ============================================================================
// Console UI
// ============================================================================
static void print_banner() {
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║  LIVING SILICON — Host Controller                       ║\n");
    printf("  ║  Windows 11 ↔ Bare-Metal Physics Kernel                 ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

static void print_bench_result(const Message& m) {
    printf("  %-24s %8.1f ms  %8.1f us/step  fitness=%.0f  [%u steps]\n",
           m.payload.bench.engine_name,
           m.payload.bench.time_ms,
           m.payload.bench.us_per_step,
           m.payload.bench.ga_fitness,
           m.payload.bench.steps);
}

static void print_telemetry(const Message& m) {
    printf("  [TELEM] E=%.4f C=%.4f ND=%.4f P=%.4f | %llu steps @ %.0f/s\n",
           m.payload.telemetry.energy,
           m.payload.telemetry.coherence,
           m.payload.telemetry.nd_balance,
           m.payload.telemetry.system_pressure,
           (unsigned long long)m.payload.telemetry.total_steps,
           m.payload.telemetry.steps_per_sec);
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    const char* pipeName = "\\\\.\\pipe\\SiliconSerial";
    const char* csvPath  = "C:\\Silicon\\results.csv";

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pipe") == 0 && i+1 < argc) pipeName = argv[++i];
        if (strcmp(argv[i], "--csv") == 0 && i+1 < argc)  csvPath = argv[++i];
    }

    print_banner();

    // Connect
    PipeConnection pipe;
    if (!pipe.connect(pipeName)) return 1;

    // Open CSV logger
    ResultLogger logger;
    if (logger.open(csvPath)) {
        printf("[CTRL] Logging to %s\n", csvPath);
    }

    // Send ping
    Message ping{};
    ping.type = MsgType::CMD_PING;
    ping.seq = 1;
    pipe.send(ping);

    // Wait for pong
    Message rsp{};
    if (pipe.recv(rsp, 3000) && rsp.type == MsgType::RSP_PONG) {
        printf("[CTRL] Kernel responded: PONG (seq=%u)\n", rsp.seq);
    }

    // Start full benchmark
    printf("\n[CTRL] Starting full benchmark...\n\n");
    printf("  %-24s %10s %14s %10s %10s\n", "ENGINE", "TIME", "us/STEP", "FITNESS", "STEPS");
    printf("  %-24s %10s %14s %10s %10s\n", "------", "----", "-------", "-------", "-----");

    Message bench_cmd{};
    bench_cmd.type = MsgType::CMD_START_BENCH;
    bench_cmd.seq = 2;
    pipe.send(bench_cmd);

    // Receive loop
    int bench_count = 0;
    float total_time = 0;
    bool running = true;

    while (running) {
        Message m{};
        if (!pipe.recv(m, 60000)) {
            printf("[CTRL] Timeout waiting for response.\n");
            break;
        }

        switch (m.type) {
            case MsgType::RSP_BENCH_RESULT:
                print_bench_result(m);
                logger.log_bench(m);
                bench_count++;
                total_time += m.payload.bench.time_ms;
                break;

            case MsgType::RSP_BENCH_DONE:
                printf("\n  ═══════════════════════════════════════════════════\n");
                printf("  BENCHMARK COMPLETE: %d engines, %.1f ms total\n",
                       bench_count, total_time);
                printf("  ═══════════════════════════════════════════════════\n");
                running = false;
                break;

            case MsgType::RSP_TELEMETRY:
                print_telemetry(m);
                logger.log_telemetry(m);
                break;

            case MsgType::RSP_TEST_RESULT:
                printf("  [TEST] %s: %u pass / %u fail (maxerr=%e)\n",
                       m.payload.test.test_name,
                       m.payload.test.passed,
                       m.payload.test.failed,
                       m.payload.test.max_error);
                logger.log_test(m);
                break;

            case MsgType::RSP_ERROR:
                printf("  [ERROR] Kernel error (seq=%u)\n", m.seq);
                break;

            default:
                printf("  [???] Unknown message type: 0x%08X\n", (uint32_t)m.type);
                break;
        }
    }

    printf("\n[CTRL] Results saved to %s\n", csvPath);
    printf("[CTRL] Session complete.\n");
    return 0;
}
