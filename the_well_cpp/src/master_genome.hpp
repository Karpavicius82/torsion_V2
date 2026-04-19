// ============================================================================
// MASTER_GENOME.HPP — Living Silicon 36-Parameter Hierarchical DNA
//
// 4 mutation levels × 8-12 parameters = 36 total
// Each level mutates at different frequency (cascade):
//   L0: Every 128 ticks   — SIMD hot-path (reflexes)
//   L1: Every 4096 ticks   — field topology (movement)
//   L2: Every 65536 ticks  — PDE control (strategy)
//   L3: Every 1M ticks     — meta-GA (evolution of evolution)
//
// Fits: 36 × uint8_t = 36 bytes = less than 1 cache line
// Overhead: ~0.5% throughput on i7-6700HQ
//
// Copyright 2026 — Living Silicon. Pure C++.
// ============================================================================
#pragma once

#include "compat.hpp"

namespace silicon {

// ── Mutation cascade timing ──
constexpr uint32_t L0_EPOCH = 0x7F;       // 128 ticks
constexpr uint32_t L1_EPOCH = 0xFFF;      // 4096 ticks
constexpr uint32_t L2_EPOCH = 0xFFFF;     // 65536 ticks
constexpr uint32_t L3_EPOCH = 0xFFFFF;    // ~1M ticks

// ── Parameter count per level ──
constexpr int L0_COUNT = 8;   // SIMD hot-path
constexpr int L1_COUNT = 8;   // Field topology
constexpr int L2_COUNT = 12;  // PDE + resource control
constexpr int L3_COUNT = 8;   // Meta-GA
constexpr int GENOME_SIZE = L0_COUNT + L1_COUNT + L2_COUNT + L3_COUNT; // 36

// ── Named parameter indices ──
// L0: SIMD Hot-Path [0-7]
constexpr int P_DELTA_A       = 0;   // Phase rotation speed, variant A
constexpr int P_DELTA_B       = 1;   // Phase rotation speed, variant B
constexpr int P_COUPLING_A    = 2;   // Phase-mag coupling, variant A
constexpr int P_COUPLING_B    = 3;   // Phase-mag coupling, variant B
constexpr int P_BLEND_A       = 4;   // Spatial viscosity, variant A
constexpr int P_BLEND_B       = 5;   // Spatial viscosity, variant B
constexpr int P_DECAY         = 6;   // Magnitude decay shift (0-15)
constexpr int P_INJECT_SCALE  = 7;   // Injection amplitude scale

// L1: Field Topology [8-15]
constexpr int P_ND_THRESHOLD  = 8;   // N-D bucket activation threshold
constexpr int P_ND_BANDWIDTH  = 9;   // N-D frequency band selector
constexpr int P_COH_TARGET    = 10;  // Target coherence level
constexpr int P_ENERGY_TARGET = 11;  // Target energy level
constexpr int P_FIELD_SYMMETRY= 12;  // Symmetry breaking parameter
constexpr int P_COUPLE_RANGE  = 13;  // Coupling range modifier
constexpr int P_PHASE_BIAS    = 14;  // Global phase offset
constexpr int P_DAMPING       = 15;  // Field damping rate

// L2: PDE + Resource Control [16-27]
constexpr int P_PDE_VISCOSITY = 16;  // Viscosity multiplier
constexpr int P_PDE_DIFFUSION = 17;  // Diffusion multiplier
constexpr int P_PDE_DT        = 18;  // Timestep scale
constexpr int P_PDE_REYNOLDS  = 19;  // Reynolds number adj
constexpr int P_PDE_FORCING   = 20;  // Forcing amplitude
constexpr int P_PDE_NOISE     = 21;  // Noise injection level
constexpr int P_FOCUS         = 22;  // Focus intensity (0=spread, 255=laser)
constexpr int P_SCOUT_TARGET  = 23;  // Scout lane target PDE
constexpr int P_LANE_MAP_LO   = 24;  // Lane assignment map (lanes 0-3)
constexpr int P_LANE_MAP_HI   = 25;  // Lane assignment map (lanes 4-7)
constexpr int P_REBALANCE_THR = 26;  // Rebalance trigger threshold
constexpr int P_RESERVE_L2    = 27;  // Reserved

// L3: Meta-GA [28-35]
constexpr int P_MUT_RATE_L0   = 28;  // L0 mutation aggressiveness
constexpr int P_MUT_RATE_L1   = 29;  // L1 mutation aggressiveness
constexpr int P_MUT_RATE_L2   = 30;  // L2 mutation aggressiveness
constexpr int P_CROSSOVER     = 31;  // Crossover blend strength
constexpr int P_STAGNATION    = 32;  // Stagnation trigger count
constexpr int P_EXPLORE_BIAS  = 33;  // Exploration vs exploitation
constexpr int P_FIT_W_ENERGY  = 34;  // Fitness weight: energy
constexpr int P_FIT_W_COHER   = 35;  // Fitness weight: coherence

// ============================================================================
// RNG (xoshiro128**)
// ============================================================================
struct Rng32 {
    uint32_t s[4];

    void seed(uint64_t v) {
        s[0] = (uint32_t)(v ^ 0xDEADBEEFUL);
        s[1] = (uint32_t)(v >> 16);
        s[2] = (uint32_t)(v >> 32);
        s[3] = (uint32_t)((v >> 48) ^ 0xCAFEUL);
        // Warm up
        for (int i = 0; i < 8; i++) next();
    }

    uint32_t next() {
        auto rotl = [](uint32_t x, int k) -> uint32_t {
            return (x << k) | (x >> (32 - k));
        };
        uint32_t r = rotl(s[1] * 5, 7) * 9;
        uint32_t t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 11);
        return r;
    }

    uint32_t range(uint32_t max) { return next() % max; }

    // Random delta: -mag to +mag
    int32_t delta(int mag) {
        return (int32_t)(range((uint32_t)(2 * mag + 1))) - mag;
    }
};

// ============================================================================
// PROPRIOCEPTIVE STATE — what the system "feels"
// ============================================================================
struct Proprioception {
    float energy;         // Total |mag| across all lanes
    float coherence;      // Phase alignment (0=chaos, 1=locked)
    float nd_balance;     // N-D population ratio (target: ~0.5)
    float pressure;       // Loop latency (TSC-measured, lower=faster)
    float fitness;        // Composite fitness
    uint64_t total_ticks; // Lifetime tick counter
    uint32_t mutations;   // Lifetime mutation counter
    uint32_t gen;         // Current generation
};

// ============================================================================
// MASTER GENOME
// ============================================================================
struct alignas(64) MasterGenome {
    // ── The DNA: 36 parameters, each uint8_t [0-255] ──
    uint8_t params[GENOME_SIZE];

    // ── Fitness tracking ──
    float fitness;
    float best_fitness;
    float ema_fitness;
    uint32_t stagnation;
    uint32_t improvements;
    uint32_t total_mutations;
    Rng32 rng;

    // ── Lane assignment cache ──
    uint8_t lane_task[8];  // What each lane is doing (PDE id or 0xFF=engine)

    // ── Initialize with sane defaults ──
    void init(uint64_t seed) {
        rng.seed(seed);
        fitness = 0; best_fitness = 0; ema_fitness = 0;
        stagnation = 0; improvements = 0; total_mutations = 0;

        // L0: SIMD defaults (from original V26)
        params[P_DELTA_A]      = 17;    // original delta
        params[P_DELTA_B]      = 25;    // variant
        params[P_COUPLING_A]   = 48;    // original coupling
        params[P_COUPLING_B]   = 64;    // variant
        params[P_BLEND_A]      = 160;   // original blend
        params[P_BLEND_B]      = 192;   // variant
        params[P_DECAY]        = 4;     // decay shift
        params[P_INJECT_SCALE] = 128;   // mid-range injection

        // L1: Field topology
        params[P_ND_THRESHOLD]  = 128;
        params[P_ND_BANDWIDTH]  = 64;
        params[P_COH_TARGET]    = 128;  // ~50% coherence target
        params[P_ENERGY_TARGET] = 128;  // mid-range energy target
        params[P_FIELD_SYMMETRY]= 128;  // neutral symmetry
        params[P_COUPLE_RANGE]  = 128;
        params[P_PHASE_BIAS]    = 128;  // no bias
        params[P_DAMPING]       = 32;

        // L2: PDE + resource
        params[P_PDE_VISCOSITY] = 128;  // 1.0x multiplier
        params[P_PDE_DIFFUSION] = 128;
        params[P_PDE_DT]        = 128;
        params[P_PDE_REYNOLDS]  = 128;
        params[P_PDE_FORCING]   = 64;
        params[P_PDE_NOISE]     = 16;
        params[P_FOCUS]         = 0;    // full spread
        params[P_SCOUT_TARGET]  = 0;
        params[P_LANE_MAP_LO]   = 0x76543210 & 0xFF; // sequential
        params[P_LANE_MAP_HI]   = 0;
        params[P_REBALANCE_THR] = 128;
        params[P_RESERVE_L2]    = 0;

        // L3: Meta-GA
        params[P_MUT_RATE_L0]   = 128;  // medium aggression
        params[P_MUT_RATE_L1]   = 96;
        params[P_MUT_RATE_L2]   = 64;
        params[P_CROSSOVER]     = 64;   // 25% crossover blend
        params[P_STAGNATION]    = 4;    // trigger after 4 epochs
        params[P_EXPLORE_BIAS]  = 128;  // balanced
        params[P_FIT_W_ENERGY]  = 128;  // equal energy weight
        params[P_FIT_W_COHER]   = 128;  // equal coherence weight

        // Default lane assignments: all on engine
        for (int i = 0; i < 8; i++) lane_task[i] = 0xFF;
    }

    // ── Read parameter as float [0.0, 1.0] ──
    float norm(int idx) const { return (float)params[idx] / 255.0f; }

    // ── Read parameter as float [lo, hi] ──
    float scaled(int idx, float lo, float hi) const {
        return lo + norm(idx) * (hi - lo);
    }

    // ── Read parameter as int [0, max] ──
    int irange(int idx, int max) const {
        return (int)params[idx] * max / 255;
    }

    // ── Mutate one parameter within a level ──
    void mutate_level(int level_start, int level_count, int aggression) {
        int idx = level_start + (int)rng.range((uint32_t)level_count);
        int mag = 1 + (aggression * (int)rng.range(8)) / 255;
        int d = rng.delta(mag);
        int v = (int)params[idx] + d;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        params[idx] = (uint8_t)v;
        total_mutations++;
    }

    // ── Cascade mutation: called every tick ──
    void cascade_mutate(uint64_t tick) {
        // L0: every 128 ticks
        if ((tick & L0_EPOCH) == 0) {
            int agg = params[P_MUT_RATE_L0];
            if (stagnation >= (uint32_t)params[P_STAGNATION] ||
                rng.range(256) < (uint32_t)agg / 4) {
                mutate_level(0, L0_COUNT, agg);
            }
        }

        // L1: every 4096 ticks
        if ((tick & L1_EPOCH) == 0) {
            int agg = params[P_MUT_RATE_L1];
            if (stagnation >= (uint32_t)params[P_STAGNATION] * 2 ||
                rng.range(256) < (uint32_t)agg / 8) {
                mutate_level(L0_COUNT, L1_COUNT, agg);
            }
        }

        // L2: every 65536 ticks
        if ((tick & L2_EPOCH) == 0) {
            int agg = params[P_MUT_RATE_L2];
            if (stagnation >= (uint32_t)params[P_STAGNATION] * 8 ||
                rng.range(256) < (uint32_t)agg / 16) {
                mutate_level(L0_COUNT + L1_COUNT, L2_COUNT, agg);
            }
        }

        // L3: every ~1M ticks (meta-GA mutates itself!)
        if ((tick & L3_EPOCH) == 0) {
            // L3 always has a small chance to mutate
            if (rng.range(100) < 5) {
                mutate_level(L0_COUNT + L1_COUNT + L2_COUNT, L3_COUNT, 64);
            }
        }
    }

    // ── Compute composite fitness from proprioception ──
    float evaluate(const Proprioception& prop) {
        float w_e = norm(P_FIT_W_ENERGY);
        float w_c = norm(P_FIT_W_COHER);
        float w_sum = w_e + w_c + 0.001f;
        w_e /= w_sum; w_c /= w_sum;

        // Energy fitness: closer to target = better
        float e_target = scaled(P_ENERGY_TARGET, 10000.0f, 10000000.0f);
        float e_diff = (prop.energy > 0.001f)
                     ? 1.0f - fabsf(prop.energy - e_target) / (e_target + 1.0f)
                     : 0.0f;
        if (e_diff < 0) e_diff = 0;

        // Coherence fitness: closer to target = better
        float c_target = norm(P_COH_TARGET);
        float c_diff = 1.0f - fabsf(prop.coherence - c_target);
        if (c_diff < 0) c_diff = 0;

        // ND balance fitness: target ~50%
        float nd_fit = 1.0f - fabsf(prop.nd_balance - 0.5f) * 2.0f;

        // Composite
        float fit = (w_e * e_diff + w_c * c_diff) * 100.0f + nd_fit * 20.0f;

        // Update tracking
        float prev_ema = ema_fitness;
        ema_fitness = ema_fitness * 0.875f + fit * 0.125f;
        fitness = fit;

        if (fit > best_fitness) {
            best_fitness = fit;
            stagnation = 0;
            improvements++;
        } else {
            stagnation++;
        }

        return fit;
    }

    // ── Crossover from a better genome ──
    void crossover_from(const MasterGenome& better) {
        float blend = norm(P_CROSSOVER);
        for (int i = 0; i < GENOME_SIZE; i++) {
            float v = (1.0f - blend) * (float)params[i]
                    + blend * (float)better.params[i];
            params[i] = (uint8_t)(v + 0.5f);
        }
        stagnation = 0;
    }

    // ── Resource allocation ──
    // Returns how many lanes should focus on the primary task
    int focus_lanes() const {
        int f = params[P_FOCUS];  // 0-255
        // 0   = 1 lane (spread)
        // 128 = 4 lanes
        // 255 = 8 lanes (laser focus)
        return 1 + (f * 7) / 255;
    }

    // Is lane N the scout lane?
    bool is_scout(int lane) const {
        return lane == 7 && params[P_FOCUS] < 240;
    }

    // ── Dump state to serial/printf ──
    void print_summary() const {
        #ifdef BARE_METAL
        print::str("  MasterGenome: ");
        print::num(GENOME_SIZE);
        print::str(" params, fit=");
        print::flt(fitness, 1);
        print::str(" best=");
        print::flt(best_fitness, 1);
        print::str(" mut=");
        print::num(total_mutations);
        print::str(" stag=");
        print::num(stagnation);
        print::line("");
        #else
        fprintf(stderr, "  MasterGenome: %d params, fit=%.1f best=%.1f mut=%u stag=%u\n",
                GENOME_SIZE, fitness, best_fitness, total_mutations, stagnation);
        fprintf(stderr, "  L0[delta=%d/%d coup=%d/%d blend=%d/%d decay=%d]\n",
                params[0],params[1],params[2],params[3],params[4],params[5],params[6]);
        fprintf(stderr, "  L1[nd_thr=%d coh_tgt=%d e_tgt=%d damp=%d]\n",
                params[8],params[10],params[11],params[15]);
        fprintf(stderr, "  L2[focus=%d scout=%d visc=%d diff=%d]\n",
                params[22],params[23],params[16],params[17]);
        fprintf(stderr, "  L3[mut_l0=%d mut_l1=%d mut_l2=%d explore=%d]\n",
                params[28],params[29],params[30],params[33]);
        #endif
    }
};

// ============================================================================
// Size verification (compile-time)
// ============================================================================
static_assert(GENOME_SIZE == 36, "Genome must be exactly 36 parameters");
static_assert(sizeof(MasterGenome) <= 128, "MasterGenome must fit in 2 cache lines");

} // namespace silicon
