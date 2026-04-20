// ============================================================================
// RING_PROTOCOL.HPP — Lock-Free SPSC Ring Buffer for VM ↔ Host Communication
//
// Pure C++20 bare-metal compatible. No libc, no OS, no allocations.
// Memory-mapped at physical address RING_BASE_ADDR.
//
// Protocol:
//   Host (Windows) writes COMMANDS → Ring → Kernel reads
//   Kernel writes TELEMETRY → Ring → Host reads
//
// Architecture: Single-Producer Single-Consumer (SPSC) with acquire/release
// ============================================================================
#pragma once

#include "../compat.hpp"

namespace silicon {

// ── Physical addresses (must match linker.ld reserved region) ──
static constexpr uintptr_t RING_BASE_ADDR = 0x200000;  // 2 MiB
static constexpr size_t    RING_SIZE       = 2 * 1024 * 1024;  // 2 MiB total

// ── Message types ──
enum class MsgType : uint32_t {
    NOP              = 0x00000000,

    // Host → Kernel commands
    CMD_PING         = 0x00000001,
    CMD_START_BENCH  = 0x00000010,  // Start full benchmark
    CMD_RUN_PDE      = 0x00000011,  // Run specific PDE engine
    CMD_RUN_TESTS    = 0x00000012,  // Run self-test suite
    CMD_SET_GA       = 0x00000020,  // Set GA parameters
    CMD_SHUTDOWN     = 0x000000FF,  // Halt kernel

    // Kernel → Host responses
    RSP_PONG         = 0x80000001,
    RSP_BENCH_RESULT = 0x80000010,  // Benchmark result row
    RSP_BENCH_DONE   = 0x80000011,  // All benchmarks complete
    RSP_TEST_RESULT  = 0x80000012,  // Test pass/fail
    RSP_TELEMETRY    = 0x80000020,  // Periodic telemetry
    RSP_GA_STATE     = 0x80000021,  // GA state update
    RSP_ERROR        = 0x8000FFFF,  // Error message
};

// ── Fixed-size message (64 bytes — cache-line aligned) ──
struct alignas(64) Message {
    MsgType  type;          // 4 bytes
    uint32_t seq;           // 4 bytes — sequence number
    uint32_t payload_len;   // 4 bytes — valid bytes in payload
    uint32_t flags;         // 4 bytes — reserved
    union {
        // GA parameters (CMD_SET_GA)
        struct {
            float mutation_rate;
            float crossover_rate;
            uint32_t population_size;
            uint32_t generations;
            float phase_delta;
            float coupling_strength;
        } ga;

        // Benchmark result (RSP_BENCH_RESULT)
        struct {
            char engine_name[24];   // PDE engine name
            float time_ms;          // Total time
            float us_per_step;      // Microseconds per step
            float ga_fitness;       // Final GA fitness
            uint32_t steps;         // Steps executed
        } bench;

        // Test result (RSP_TEST_RESULT)
        struct {
            char test_name[24];
            uint32_t passed;
            uint32_t failed;
            float max_error;
        } test;

        // Telemetry (RSP_TELEMETRY)
        struct {
            float energy;
            float coherence;
            float nd_balance;
            float system_pressure;
            uint64_t total_steps;
            float steps_per_sec;
        } telemetry;

        // PDE selection (CMD_RUN_PDE)
        struct {
            uint32_t pde_id;        // 0-25
            uint32_t n_steps;
            float dt_override;      // 0 = use default
        } pde_cmd;

        // Raw bytes
        uint8_t raw[48];
    } payload;
};

static_assert(sizeof(Message) == 64, "Message must be 64 bytes (cache-line)");

// ── Ring header (first 64 bytes of shared memory) ──
struct alignas(64) RingHeader {
    // Magic: "SiLiCoN\0" = 0x4E6F43694C6953
    uint64_t magic;

    // Indices (each on own cache line to avoid false sharing)
    // Host→Kernel ring (commands)
    volatile uint32_t cmd_write_idx;    // Host writes, Kernel reads
    volatile uint32_t cmd_read_idx;     // Kernel writes, Host reads
    uint32_t cmd_capacity;              // Number of message slots

    // Kernel→Host ring (responses)
    volatile uint32_t rsp_write_idx;    // Kernel writes, Host reads
    volatile uint32_t rsp_read_idx;     // Host writes, Kernel reads
    uint32_t rsp_capacity;              // Number of message slots

    // Status
    volatile uint32_t kernel_alive;     // Heartbeat counter (++every 1s)
    volatile uint32_t kernel_state;     // 0=booting, 1=ready, 2=running, 3=error
    uint64_t boot_timestamp;            // TSC at boot

    uint8_t _pad[4];                    // Align to 64 bytes
};

static_assert(sizeof(RingHeader) == 64, "RingHeader must be 64 bytes");

// ── Ring layout in memory ──
// [0x200000] RingHeader (64 bytes)
// [0x200040] Command ring: cmd_capacity × Message (64 bytes each)
// [halfway]  Response ring: rsp_capacity × Message (64 bytes each)

static constexpr uint32_t CMD_RING_SLOTS = (RING_SIZE / 2 - sizeof(RingHeader)) / sizeof(Message);
static constexpr uint32_t RSP_RING_SLOTS = (RING_SIZE / 2) / sizeof(Message);

static constexpr uint64_t RING_MAGIC = 0x4E6F43694C695300ULL;  // "SiLiCoN\0"

// ============================================================================
// Ring Controller (bare-metal kernel side)
// ============================================================================
class KernelRing {
    volatile RingHeader* hdr_;
    volatile Message*    cmd_ring_;
    volatile Message*    rsp_ring_;

public:
    void init() {
        hdr_ = reinterpret_cast<volatile RingHeader*>(RING_BASE_ADDR);

        // Initialize header
        hdr_->magic = RING_MAGIC;
        hdr_->cmd_write_idx = 0;
        hdr_->cmd_read_idx = 0;
        hdr_->cmd_capacity = CMD_RING_SLOTS;
        hdr_->rsp_write_idx = 0;
        hdr_->rsp_read_idx = 0;
        hdr_->rsp_capacity = RSP_RING_SLOTS;
        hdr_->kernel_alive = 0;
        hdr_->kernel_state = 0;  // booting
        hdr_->boot_timestamp = 0;

        // Pointers to ring data
        cmd_ring_ = reinterpret_cast<volatile Message*>(
            RING_BASE_ADDR + sizeof(RingHeader));
        rsp_ring_ = reinterpret_cast<volatile Message*>(
            RING_BASE_ADDR + RING_SIZE / 2);

        // Zero all slots
        for (uint32_t i = 0; i < CMD_RING_SLOTS; i++) {
            auto* m = const_cast<Message*>(&cmd_ring_[i]);
            for (int b = 0; b < 64; b++)
                reinterpret_cast<volatile uint8_t*>(m)[b] = 0;
        }
        for (uint32_t i = 0; i < RSP_RING_SLOTS; i++) {
            auto* m = const_cast<Message*>(&rsp_ring_[i]);
            for (int b = 0; b < 64; b++)
                reinterpret_cast<volatile uint8_t*>(m)[b] = 0;
        }

        hdr_->kernel_state = 1;  // ready
    }

    // Read command from host (non-blocking)
    bool read_cmd(Message& out) {
        uint32_t ri = hdr_->cmd_read_idx;
        uint32_t wi = hdr_->cmd_write_idx;
        if (ri == wi) return false;  // empty

        // Memory barrier (acquire)
        asm volatile("mfence" ::: "memory");

        const volatile Message& slot = cmd_ring_[ri % hdr_->cmd_capacity];
        // Copy out
        const auto* src = reinterpret_cast<const volatile uint8_t*>(&slot);
        auto* dst = reinterpret_cast<uint8_t*>(&out);
        for (int i = 0; i < 64; i++) dst[i] = src[i];

        // Advance read index
        asm volatile("mfence" ::: "memory");
        hdr_->cmd_read_idx = ri + 1;
        return true;
    }

    // Write response to host (non-blocking)
    bool write_rsp(const Message& msg) {
        uint32_t wi = hdr_->rsp_write_idx;
        uint32_t ri = hdr_->rsp_read_idx;
        if (wi - ri >= hdr_->rsp_capacity) return false;  // full

        volatile Message& slot = rsp_ring_[wi % hdr_->rsp_capacity];
        const auto* src = reinterpret_cast<const uint8_t*>(&msg);
        auto* dst = reinterpret_cast<volatile uint8_t*>(&slot);
        for (int i = 0; i < 64; i++) dst[i] = src[i];

        // Memory barrier (release)
        asm volatile("mfence" ::: "memory");
        hdr_->rsp_write_idx = wi + 1;
        return true;
    }

    // Convenience: send benchmark result
    void send_bench_result(const char* name, float time_ms, float us_step,
                          float fitness, uint32_t steps, uint32_t seq) {
        Message m{};
        m.type = MsgType::RSP_BENCH_RESULT;
        m.seq = seq;
        m.payload_len = sizeof(m.payload.bench);
        // Copy name
        for (int i = 0; i < 24 && name[i]; i++)
            m.payload.bench.engine_name[i] = name[i];
        m.payload.bench.time_ms = time_ms;
        m.payload.bench.us_per_step = us_step;
        m.payload.bench.ga_fitness = fitness;
        m.payload.bench.steps = steps;
        write_rsp(m);
    }

    // Send telemetry
    void send_telemetry(float energy, float coherence, float nd_bal,
                       float pressure, uint64_t total_steps, float sps, uint32_t seq) {
        Message m{};
        m.type = MsgType::RSP_TELEMETRY;
        m.seq = seq;
        m.payload.telemetry.energy = energy;
        m.payload.telemetry.coherence = coherence;
        m.payload.telemetry.nd_balance = nd_bal;
        m.payload.telemetry.system_pressure = pressure;
        m.payload.telemetry.total_steps = total_steps;
        m.payload.telemetry.steps_per_sec = sps;
        write_rsp(m);
    }

    // Heartbeat (call every ~1 second)
    void heartbeat() {
        hdr_->kernel_alive++;
    }

    void set_state(uint32_t state) {
        hdr_->kernel_state = state;
    }
};

// ============================================================================
// Host Ring Controller (Windows side — uses memcpy semantics)
// ============================================================================
#ifndef BARE_METAL
class HostRing {
    RingHeader* hdr_;
    Message*    cmd_ring_;
    Message*    rsp_ring_;

public:
    bool attach(void* shared_mem) {
        hdr_ = reinterpret_cast<RingHeader*>(shared_mem);
        if (hdr_->magic != RING_MAGIC) return false;

        cmd_ring_ = reinterpret_cast<Message*>(
            reinterpret_cast<uint8_t*>(shared_mem) + sizeof(RingHeader));
        rsp_ring_ = reinterpret_cast<Message*>(
            reinterpret_cast<uint8_t*>(shared_mem) + RING_SIZE / 2);
        return true;
    }

    // Send command to kernel
    bool send_cmd(const Message& msg) {
        uint32_t wi = hdr_->cmd_write_idx;
        uint32_t ri = hdr_->cmd_read_idx;
        if (wi - ri >= hdr_->cmd_capacity) return false;

        Message& slot = cmd_ring_[wi % hdr_->cmd_capacity];
        memcpy(&slot, &msg, sizeof(Message));
        _mm_mfence();
        hdr_->cmd_write_idx = wi + 1;
        return true;
    }

    // Read response from kernel
    bool read_rsp(Message& out) {
        uint32_t ri = hdr_->rsp_read_idx;
        uint32_t wi = hdr_->rsp_write_idx;
        if (ri == wi) return false;

        _mm_mfence();
        memcpy(&out, &rsp_ring_[ri % hdr_->rsp_capacity], sizeof(Message));
        _mm_mfence();
        hdr_->rsp_read_idx = ri + 1;
        return true;
    }

    uint32_t kernel_heartbeat() const { return hdr_->kernel_alive; }
    uint32_t kernel_state() const { return hdr_->kernel_state; }
};
#endif // !BARE_METAL

}  // namespace silicon
