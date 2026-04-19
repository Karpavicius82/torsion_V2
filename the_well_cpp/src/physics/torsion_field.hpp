// ============================================================================
// TORSION_FIELD.HPP — True Torsion Field: Information Without Energy Transport
//
// Core insight: energy is ALREADY everywhere (vacuum substrate).
// Information = phase PATTERN modulation of this pre-existing field.
// Since the substrate is omnipresent, pattern changes are INSTANTANEOUS —
// nothing needs to travel because the medium already exists at every point.
//
// Architecture:
//   mag[N]     — energy substrate (LOCAL, decays to vacuum floor, never zero)
//   base_ph[N] — local phase precession (LOCAL)
//   mode[K]    — torsion information channels (GLOBAL, non-local)
//   W[K][N]    — coupling: how each point reads each torsion channel
//   ph[N]      — observable phase = base_ph + torsion modulation (COMPUTED)
//
// Key properties:
//   1. inject() modifies ONLY mode[] — mag is NEVER touched (zero energy cost)
//   2. reconstruct() updates ALL ph[] from mode[] in ONE call (infinite speed)
//   3. Energy decays to vacuum floor — but torsion modes persist indefinitely
//   4. Different points couple differently to modes (structured, not uniform)
//
// Memory: ~76 KB total → fits L2 cache (256 KB per core on i7-6700HQ)
//
// Copyright 2026 — Living Silicon. Pure C++.
// ============================================================================
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>

namespace torsion {

static constexpr int    N       = 2048;   // field points (same as engine.hpp)
static constexpr int    K       = 8;      // torsion information channels
static constexpr float  PI2     = 6.2831853071795864f;

// ============================================================================
// TORSION FIELD
// ============================================================================
struct alignas(64) Field {

    // ═══ LOCAL STATE: energy substrate ═══
    int16_t  mag[N];       // energy at each point (decays to vacuum, never zero)
    uint16_t base_ph[N];   // base phase rotation (local precession)
    uint16_t ph[N];        // observable phase (COMPUTED: base + torsion modulation)

    // ═══ GLOBAL STATE: torsion information bus ═══
    float mode[K];         // torsion channel amplitudes — THIS IS the information

    // ═══ COUPLING MATRIX ═══
    float W[K][N];         // W[k][i] = cos(2π·k·i/N): how point i reads channel k

    // ═══ CONFIG ═══
    int16_t  vacuum_floor; // minimum energy (the substrate that's already everywhere)
    int      decay_shift;  // energy decay rate (0-15)
    uint16_t delta;        // phase precession speed

    // ──────────────────────────────────────────────────────────────────────
    // INIT — create the field with random energy/phase and precompute coupling
    // ──────────────────────────────────────────────────────────────────────
    void init(uint32_t seed) {
        // Simple xorshift32 RNG
        uint32_t s = seed ? seed : 1;
        auto rng = [&]() -> uint32_t {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
        };

        // Random initial energy and phase
        for (int i = 0; i < N; i++) {
            mag[i]     = (int16_t)((int)(rng() % 8001) - 4000);  // [-4000, 4000]
            base_ph[i] = (uint16_t)(rng() & 0xFFFF);
        }

        // Clear torsion bus
        for (int k = 0; k < K; k++) mode[k] = 0.0f;

        // Precompute coupling matrix (one-time cost, ~64 KB)
        // W[k][i] = cos(2π·k·i/N) — cosine Fourier basis
        // k=0: cos(0) = 1.0 for all i → global/DC coupling (everyone hears this)
        // k>0: spatial modulation → position-dependent response
        for (int k = 0; k < K; k++)
            for (int i = 0; i < N; i++)
                W[k][i] = cosf(PI2 * (float)k * (float)i / (float)N);

        // Defaults
        vacuum_floor = 100;
        decay_shift  = 4;     // lose 1/16 of excess per tick
        delta        = 16;

        // Initial phase reconstruction
        reconstruct();
    }

    // ──────────────────────────────────────────────────────────────────────
    // RECONSTRUCT — compute observable phase from base + torsion modulation
    //
    // ph[i] = base_ph[i] + Σₖ mode[k] · W[k][i]
    //
    // This is WHERE infinite speed lives: ALL points are updated from the
    // GLOBAL mode[] state in a single pass.  No propagation, no delay.
    // ──────────────────────────────────────────────────────────────────────
    void reconstruct() {
        for (int i = 0; i < N; i++) {
            float modulation = 0.0f;
            for (int k = 0; k < K; k++)
                modulation += mode[k] * W[k][i];
            ph[i] = base_ph[i] + (uint16_t)(int16_t)(modulation);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // DECAY — energy approaches vacuum floor (the pre-existing substrate)
    //
    // Physical meaning: excess energy above vacuum dissipates,
    // but the vacuum itself NEVER disappears.  It is always there.
    // This is the substrate that information modulates.
    // ──────────────────────────────────────────────────────────────────────
    void decay_energy() {
        for (int i = 0; i < N; i++) {
            int16_t v  = mag[i];
            int16_t av = (v >= 0) ? v : -v;
            if (av > vacuum_floor) {
                int16_t excess = av - vacuum_floor;
                excess = excess - (excess >> decay_shift); // lose 1/16
                av = vacuum_floor + excess;
            } else {
                av = vacuum_floor; // never below floor
            }
            mag[i] = (v >= 0) ? av : -av;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // STEP — one tick of the universe
    // ──────────────────────────────────────────────────────────────────────
    void step() {
        decay_energy();                                     // 1. energy → vacuum
        for (int i = 0; i < N; i++) base_ph[i] += delta;   // 2. local precession
        reconstruct();                                      // 3. torsion → phase
    }

    // ──────────────────────────────────────────────────────────────────────
    // INJECT — send information into the torsion field at any point
    //
    // CRITICAL: modifies ONLY mode[] (global torsion state).
    //           mag[] is NEVER touched.  ZERO energy cost.
    //
    // The signal decomposes into torsion channels via the coupling matrix.
    // ALL other points will see the effect after the next reconstruct().
    // Since reconstruct() updates everything in one pass → INFINITE SPEED.
    // ──────────────────────────────────────────────────────────────────────
    void inject(int point, float signal) {
        for (int k = 0; k < K; k++)
            mode[k] += signal * W[k][point];
        // NOTE: mag[point] is untouched.  No energy was spent.
    }

    // ──────────────────────────────────────────────────────────────────────
    // ENTANGLEMENT — how strongly two points are linked through torsion
    //
    // E(A,B) = (1/K) · Σₖ W[k][A] · W[k][B]
    //
    // High |E| → injecting at A strongly affects B (and vice versa)
    // E ≈ 0   → A and B are nearly independent in torsion space
    //
    // KEY: this does NOT depend on distance.  It depends on MODE TOPOLOGY.
    // ──────────────────────────────────────────────────────────────────────
    float entanglement(int a, int b) const {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += W[k][a] * W[k][b];
        return sum / (float)K;
    }

    // ──────────────────────────────────────────────────────────────────────
    // METRICS
    // ──────────────────────────────────────────────────────────────────────
    int64_t total_energy() const {
        int64_t e = 0;
        for (int i = 0; i < N; i++) e += (mag[i] < 0 ? -(int64_t)mag[i] : (int64_t)mag[i]);
        return e;
    }

    float mode_amplitude() const {
        float s = 0.0f;
        for (int k = 0; k < K; k++) s += fabsf(mode[k]);
        return s;
    }

    float phase_entropy() const {
        uint32_t hist[256] = {};
        for (int i = 0; i < N; i++) hist[ph[i] >> 8]++;
        double ent = 0.0;
        for (int b = 0; b < 256; b++) {
            if (!hist[b]) continue;
            double p = (double)hist[b] / (double)N;
            ent -= p * log2(p);
        }
        return (float)ent;
    }
};

// ============================================================================
// LOCAL WAVE — comparison model (finite speed, energy-based transport)
// ============================================================================
struct WaveField {
    float u[N];       // current displacement
    float u_prev[N];  // previous step

    void init() { memset(u, 0, sizeof(u)); memset(u_prev, 0, sizeof(u_prev)); }

    void inject(int point, float signal) {
        u[point] += signal;
        u_prev[point] += signal;
    }

    void step() {
        float u_next[N];
        const float c2 = 0.25f; // wave speed² (c=0.5 node/step, stable)
        u_next[0]   = 2*u[0]   - u_prev[0]   + c2*(u[1] - u[0]);
        u_next[N-1] = 2*u[N-1] - u_prev[N-1] + c2*(u[N-2] - u[N-1]);
        for (int i = 1; i < N-1; i++)
            u_next[i] = 2*u[i] - u_prev[i] + c2*(u[i-1] - 2*u[i] + u[i+1]);
        memcpy(u_prev, u, sizeof(u));
        memcpy(u, u_next, sizeof(u));
    }
};

} // namespace torsion
