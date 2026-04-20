// ============================================================================
// KERNEL.CPP — Living Silicon Kernel Entry Point v2.0
//
// Now with Hyper-V shared-memory ring protocol integration.
//
// Flow:
//   0. Init hardware (serial, VGA, memory, timer)
//   1. Detect CPU features (AVX2/FMA)
//   2. Init ring protocol (shared memory @ 0x200000)
//   3. Self-test: math, FFT, PDE conservation, gradients
//   4. Enter command loop: read ring → execute → write results
//   5. Fallback: autonomous GA evolution if no host connected
//
// Build (bare-metal):
//   nasm -f elf64 src/boot/boot.asm -o boot.o
//   g++ -ffreestanding -fno-exceptions -fno-rtti -nostdlib -mno-red-zone \
//       -mavx2 -mfma -O3 -std=c++20 -DBARE_METAL -I src \
//       -c src/kernel.cpp -o kernel.o
//   ld -n -T src/boot/linker.ld -o well_silicon.bin boot.o kernel.o
//
// Boot (Hyper-V):
//   .\deploy\hyperv_setup.ps1 -Action start
//   .\deploy\silicon_controller.exe
// ============================================================================

#define BARE_METAL
#include "compat.hpp"

// ── All engines ──
#include "tensor.hpp"
#include "engine.hpp"
#include "core/fft.hpp"
#include "core/tensor_nd.hpp"

// Physics (26 PDE)
#include "physics/pde_base.hpp"
#include "physics/navier_stokes_2d.hpp"
#include "physics/diffusion_wave_burgers.hpp"
#include "physics/mhd_rb_sw.hpp"
#include "physics/euler_helmholtz_advection_rd.hpp"
#include "physics/ks_ch_torsion2d_ac.hpp"
#include "physics/acoustic_scattering.hpp"
#include "physics/active_matter_visco.hpp"
#include "physics/rt_shear.hpp"
#include "physics/supernova_turb.hpp"
#include "physics/convective_turb_rad.hpp"
#include "physics/neutron_planetswe.hpp"

// Models (9)
#include "models/model.hpp"
#include "models/fno2d.hpp"
#include "models/unet2d.hpp"
#include "models/resnet2d.hpp"
#include "models/vit.hpp"
#include "models/pinn.hpp"

// Training
#include "training/metrics.hpp"
#include "training/adamw.hpp"

// Tests
#include "tests/test_math.hpp"
#include "tests/test_fft.hpp"
#include "tests/test_pde.hpp"
#include "tests/test_gradients.hpp"

// Hyper-V communication
#include "hyperv/ring_protocol.hpp"

// ============================================================================
// PDE REGISTRY — Maps PDE ID to engine instance
// ============================================================================
namespace {

struct PDEEntry {
    const char* name;
    // We use function pointers to avoid static constructors
    void (*run)(uint32_t steps, uint64_t seed, float* out_time_ms, float* out_fitness);
};

template<typename PDE>
void run_pde(uint32_t steps, uint64_t seed, float* out_time_ms, float* out_fitness) {
    PDE pde;
    pde.init(seed);
    uint64_t t0 = timer::now_us();
    for (uint32_t i = 0; i < steps; i++) pde.step(0);
    uint64_t elapsed = timer::now_us() - t0;
    *out_time_ms = (float)elapsed / 1000.0f;
    *out_fitness = (float)pde.engine_ga.fitness;
}

static const PDEEntry PDE_REGISTRY[] = {
    {"NavierStokes-2D",       run_pde<well::NavierStokes2D>},
    {"Diffusion-2D",          run_pde<well::Diffusion2D>},
    {"Wave-2D",               run_pde<well::Wave2D>},
    {"Burgers-2D",            run_pde<well::Burgers2D>},
    {"MHD-2D",                run_pde<well::MHD2D>},
    {"RayleighBenard-2D",     run_pde<well::RayleighBenard2D>},
    {"ShallowWater-2D",       run_pde<well::ShallowWater2D>},
    {"CompressibleEuler-2D",  run_pde<well::CompressibleEuler2D>},
    {"Helmholtz-2D",          run_pde<well::Helmholtz2D>},
    {"Advection-2D",          run_pde<well::Advection2D>},
    {"GrayScott-2D",          run_pde<well::GrayScott2D>},
    {"KuramotoSivashinsky",   run_pde<well::KuramotoSivashinsky2D>},
    {"CahnHilliard-2D",       run_pde<well::CahnHilliard2D>},
    {"Torsion-2D",            run_pde<well::Torsion2D>},
    {"AllenCahn-2D",          run_pde<well::AllenCahn2D>},
    {"AcousticScattering",    run_pde<well::AcousticScattering2D>},
    {"ActiveMatter-2D",       run_pde<well::ActiveMatter2D>},
    {"ViscoelasticInst-2D",   run_pde<well::ViscoelasticInstability2D>},
    {"RayleighTaylor-2D",     run_pde<well::RayleighTaylor2D>},
    {"ShearFlow-2D",          run_pde<well::ShearFlow2D>},
    {"Supernova-2D",          run_pde<well::Supernova2D>},
    {"TurbGravCool-2D",       run_pde<well::TurbulenceGravityCooling2D>},
    {"ConvectiveEnvelope",    run_pde<well::ConvectiveEnvelope2D>},
    {"TurbRadLayer-2D",       run_pde<well::TurbulentRadiativeLayer2D>},
    {"NeutronStarMerger",     run_pde<well::PostNeutronStarMerger2D>},
    {"PlanetarySWE-2D",       run_pde<well::PlanetarySWE2D>},
};

static constexpr int N_PDE = sizeof(PDE_REGISTRY) / sizeof(PDE_REGISTRY[0]);

} // anon namespace

// ============================================================================
// BENCHMARK: Run all 26 PDE engines, report via ring + serial
// ============================================================================
static void run_full_benchmark(silicon::KernelRing& ring, uint32_t steps_per_engine) {
    print::banner(" BENCHMARK: 26 PDE ENGINES ");
    print::line("");

    uint32_t seq = 100;
    for (int i = 0; i < N_PDE; i++) {
        float time_ms = 0, fitness = 0;
        PDE_REGISTRY[i].run(steps_per_engine, (uint64_t)(i + 1), &time_ms, &fitness);

        float us_step = time_ms * 1000.0f / (float)steps_per_engine;

        // Serial output
        print::str("  ");
        print::str(PDE_REGISTRY[i].name);
        print::str(": ");
        print::flt(time_ms, 1);
        print::str(" ms (");
        print::flt(us_step, 1);
        print::str(" us/step) fitness=");
        print::num((uint64_t)fitness);
        print::line("");

        // Ring output
        ring.send_bench_result(PDE_REGISTRY[i].name, time_ms, us_step,
                              fitness, steps_per_engine, seq++);
    }

    // Send completion
    silicon::Message done{};
    done.type = silicon::MsgType::RSP_BENCH_DONE;
    done.seq = seq;
    ring.write_rsp(done);

    print::line("");
    print::line("  All 26 engines: COMPLETE");
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================
static void run_self_tests(silicon::KernelRing& ring) {
    print::banner(" SELF-TEST SUITE ");
    print::line("");

    int total_pass = 0, total_fail = 0;
    uint32_t seq = 200;

    // Math
    auto math_r = test::run_math_tests();
    total_pass += math_r.passed; total_fail += math_r.failed;
    print::str(" [MATH] "); print::num(math_r.passed); print::str("/"); print::num(math_r.passed + math_r.failed); print::line("");
    {
        silicon::Message m{}; m.type = silicon::MsgType::RSP_TEST_RESULT; m.seq = seq++;
        const char* n = "MATH"; for(int i=0;n[i];i++) m.payload.test.test_name[i]=n[i];
        m.payload.test.passed = math_r.passed; m.payload.test.failed = math_r.failed;
        ring.write_rsp(m);
    }

    // FFT
    auto fft_r = test::run_fft_tests();
    total_pass += fft_r.passed; total_fail += fft_r.failed;
    print::str(" [FFT]  "); print::num(fft_r.passed); print::str("/"); print::num(fft_r.passed + fft_r.failed); print::line("");
    {
        silicon::Message m{}; m.type = silicon::MsgType::RSP_TEST_RESULT; m.seq = seq++;
        const char* n = "FFT"; for(int i=0;n[i];i++) m.payload.test.test_name[i]=n[i];
        m.payload.test.passed = fft_r.passed; m.payload.test.failed = fft_r.failed;
        ring.write_rsp(m);
    }

    // PDE
    auto pde_r = test::run_pde_tests();
    total_pass += pde_r.passed; total_fail += pde_r.failed;
    print::str(" [PDE]  "); print::num(pde_r.passed); print::str("/"); print::num(pde_r.passed + pde_r.failed); print::line("");
    {
        silicon::Message m{}; m.type = silicon::MsgType::RSP_TEST_RESULT; m.seq = seq++;
        const char* n = "PDE"; for(int i=0;n[i];i++) m.payload.test.test_name[i]=n[i];
        m.payload.test.passed = pde_r.passed; m.payload.test.failed = pde_r.failed;
        ring.write_rsp(m);
    }

    // Gradients
    auto grad_r = test::run_gradient_tests();
    total_pass += grad_r.passed; total_fail += grad_r.failed;
    print::str(" [GRAD] "); print::num(grad_r.passed); print::str("/"); print::num(grad_r.passed + grad_r.failed); print::line("");
    {
        silicon::Message m{}; m.type = silicon::MsgType::RSP_TEST_RESULT; m.seq = seq++;
        const char* n = "GRADIENTS"; for(int i=0;n[i];i++) m.payload.test.test_name[i]=n[i];
        m.payload.test.passed = grad_r.passed; m.payload.test.failed = grad_r.failed;
        ring.write_rsp(m);
    }

    print::line("");
    print::str(" TOTAL: "); print::num(total_pass); print::str(" PASS / ");
    print::num(total_fail); print::line(" FAIL");

    if (total_fail == 0)
        print::banner(" *** ALL TESTS PASSED *** ");
    else
        print::banner(" !!! TESTS FAILED !!! ");
    print::line("");
}

// ============================================================================
// KERNEL MAIN
// ============================================================================
extern "C" void kernel_main() {
    // ── Phase 0: Hardware Init ──
    serial::init();
    vga::clear();

    print::banner("╔══════════════════════════════════════════════════════════╗");
    print::banner("║  LIVING SILICON v2.0 | HYPER-V ENABLED | BARE METAL    ║");
    print::banner("╚══════════════════════════════════════════════════════════╝");
    print::line("");
    print::line(" No OS. No libc. No Python. Direct hardware.");
    print::line(" 26 PDE engines | 9 NN models | GA-driven | AVX2/FMA");
    print::line(" Ring protocol: shared memory @ 0x200000 (2MB)");
    print::line("");

    // ── Phase 1: CPU Detection ──
    auto cpu = cpuid::detect();
    print::str(" CPU: "); print::str(cpu.brand); print::line("");
    print::str(" AVX2: "); print::str(cpu.avx2 ? "YES" : "NO");
    print::str("  FMA: "); print::str(cpu.fma ? "YES" : "NO");
    print::str("  AVX512: "); print::str(cpu.avx512 ? "YES" : "NO");
    print::line(""); print::line("");

    if (!cpu.avx2) {
        print::line("[FATAL] AVX2 required. Halting.");
        hw::hlt();
        return;
    }

    // ── Phase 2: Memory Init ──
    mem::init();
    print::str(" Memory arena: "); print::num(mem::ARENA_SIZE / (1024*1024)); print::line(" MB");

    // ── Phase 3: Timer ──
    timer::calibrate();
    print::str(" TSC freq: ~"); print::num(timer::tsc_freq / 1000000); print::line(" MHz");
    print::line("");

    // ── Phase 4: Ring Protocol Init ──
    silicon::KernelRing ring;
    ring.init();
    ring.set_state(1);  // ready
    print::line(" Ring protocol: INITIALIZED");
    print::line(" Waiting for host commands (or auto-run in 5s)...");
    print::line("");

    // ── Phase 5: Wait for host or auto-run ──
    // Give host 5 seconds to send a PING
    bool host_connected = false;
    uint64_t wait_start = timer::now_us();

    while (timer::now_us() - wait_start < 5000000) {  // 5 seconds
        silicon::Message cmd{};
        if (ring.read_cmd(cmd)) {
            if (cmd.type == silicon::MsgType::CMD_PING) {
                silicon::Message pong{};
                pong.type = silicon::MsgType::RSP_PONG;
                pong.seq = cmd.seq;
                ring.write_rsp(pong);
                host_connected = true;
                print::line(" Host connected! Entering command mode.");
                break;
            }
        }
        ring.heartbeat();
    }

    if (!host_connected) {
        print::line(" No host detected. Running autonomous mode.");
    }

    // ── Phase 6: Self-test (always runs) ──
    ring.set_state(2);  // running
    run_self_tests(ring);

    // ── Phase 7: Command loop or autonomous ──
    if (host_connected) {
        // ── COMMAND MODE ──
        print::banner(" COMMAND MODE — AWAITING HOST INSTRUCTIONS ");
        print::line("");

        while (true) {
            silicon::Message cmd{};
            if (ring.read_cmd(cmd)) {
                switch (cmd.type) {
                    case silicon::MsgType::CMD_PING: {
                        silicon::Message pong{};
                        pong.type = silicon::MsgType::RSP_PONG;
                        pong.seq = cmd.seq;
                        ring.write_rsp(pong);
                        break;
                    }

                    case silicon::MsgType::CMD_START_BENCH: {
                        uint32_t steps = 500;  // default
                        run_full_benchmark(ring, steps);
                        break;
                    }

                    case silicon::MsgType::CMD_RUN_PDE: {
                        uint32_t id = cmd.payload.pde_cmd.pde_id;
                        uint32_t steps = cmd.payload.pde_cmd.n_steps;
                        if (steps == 0) steps = 500;
                        if (id < (uint32_t)N_PDE) {
                            float time_ms = 0, fitness = 0;
                            PDE_REGISTRY[id].run(steps, id + 1, &time_ms, &fitness);
                            ring.send_bench_result(PDE_REGISTRY[id].name,
                                                  time_ms, time_ms * 1000.0f / steps,
                                                  fitness, steps, cmd.seq);
                        }
                        break;
                    }

                    case silicon::MsgType::CMD_RUN_TESTS: {
                        run_self_tests(ring);
                        break;
                    }

                    case silicon::MsgType::CMD_SET_GA: {
                        // TODO: Apply GA params to active evolution
                        silicon::Message ack{};
                        ack.type = silicon::MsgType::RSP_GA_STATE;
                        ack.seq = cmd.seq;
                        ring.write_rsp(ack);
                        break;
                    }

                    case silicon::MsgType::CMD_SHUTDOWN: {
                        print::line(" [SHUTDOWN] Host requested halt.");
                        ring.set_state(0);
                        hw::hlt();
                        return;
                    }

                    default:
                        break;
                }
            }

            // Background: heartbeat + idle GA step
            ring.heartbeat();

            // Yield CPU briefly (not busy-spin)
            for (volatile int i = 0; i < 10000; i++) {}
        }
    } else {
        // ── AUTONOMOUS MODE ──
        print::banner(" AUTONOMOUS MODE — GA EVOLUTION ");
        print::line("");

        // Run benchmark first
        run_full_benchmark(ring, 500);

        // Memory report
        print::banner(" MEMORY STATUS ");
        print::str("  Used: "); print::num(mem::used() / 1024);
        print::str(" KB / "); print::num(mem::ARENA_SIZE / (1024*1024)); print::line(" MB");
        print::line("");

        print::banner("╔════════════════════════════════════════════════════════╗");
        print::banner("║        LIVING SILICON BENCHMARK COMPLETE              ║");
        print::banner("╚════════════════════════════════════════════════════════╝");
        print::line("");

        // Continuous GA evolution
        well::NavierStokes2D ns;
        ns.init(0xC0DE7052ULL);
        uint64_t gen = 0;
        uint64_t total_steps = 0;
        uint64_t last_telemetry = timer::now_us();

        while (true) {
            ns.step(0);
            gen++;
            total_steps++;

            // Check for late host connection
            silicon::Message cmd{};
            if (ring.read_cmd(cmd)) {
                if (cmd.type == silicon::MsgType::CMD_PING) {
                    silicon::Message pong{};
                    pong.type = silicon::MsgType::RSP_PONG;
                    pong.seq = cmd.seq;
                    ring.write_rsp(pong);
                    print::line(" [GA] Host connected mid-evolution!");
                }
            }

            if ((gen & 0xFFFFF) == 0) {
                ns.ga_mutate(ns.rng);

                uint64_t now = timer::now_us();
                float dt = (float)(now - last_telemetry) / 1000000.0f;
                float sps = (float)(gen) / (dt > 0 ? dt : 1.0f);

                print::str("  [GA] gen="); print::num(gen >> 20);
                print::str(" fitness="); print::num(ns.engine_ga.fitness);
                print::str(" sps="); print::flt(sps, 0);
                print::str(" mem="); print::num(mem::used() / 1024);
                print::line(" KB");

                // Send telemetry
                ring.send_telemetry(
                    0.0f, 0.0f, 0.0f, 0.0f,
                    total_steps, sps, (uint32_t)(gen >> 20));

                ring.heartbeat();
            }
        }
    }
}
