// ============================================================================
// ACOUSTIC_SCATTERING.HPP — 2D Acoustic Scattering (Living Silicon / GA)
//
// PDE: ∂²p/∂t² = c²(∂²p/∂x² + ∂²p/∂y²) + source
// Boundary: circular scatterer (rigid body), absorbing outer boundary
//
// GA parameters: wave_speed, source_freq, source_amplitude, damping
// ============================================================================
#pragma once
#include "pde_base.hpp"

namespace well {

struct AcousticScattering2D : PDE2D {
    // Fields: pressure, velocity_x, velocity_y
    float p[N2D][N2D]     = {};    // pressure
    float vx[N2D][N2D]    = {};    // x-velocity
    float vy[N2D][N2D]    = {};    // y-velocity
    float p_prev[N2D][N2D] = {};   // previous pressure (leapfrog)

    // Scatterer geometry
    float sc_cx, sc_cy, sc_radius;

    // GA-tunable
    struct AcousticGA {
        float wave_speed  = 1.0f;
        float src_freq    = 2.0f;
        float src_amp     = 1.0f;
        float damping     = 0.01f;
        float absorb_coef = 0.95f;
    } ga;

    const char* name() const override { return "AcousticScattering-2D"; }
    int num_fields() const override { return 3; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        sc_cx = 0.5f; sc_cy = 0.5f; sc_radius = 0.12f;

        // Zero fields
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                p[j][i] = 0; vx[j][i] = 0; vy[j][i] = 0;
                p_prev[j][i] = 0;
            }
    }

    bool inside_scatterer(int i, int j) const {
        float x = (float)i / (float)N2D;
        float y = (float)j / (float)N2D;
        float dx = x - sc_cx, dy = y - sc_cy;
        return (dx*dx + dy*dy) < sc_radius * sc_radius;
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float c = ga.wave_speed;
        float c2 = c * c;
        float dt = dt_ > 0 ? dt_ : dx / (c * 2.0f);  // CFL
        float r = c2 * dt * dt / (dx * dx);

        // Source: plane wave from left boundary
        float freq = ga.src_freq;
        float omega = 2.0f * 3.14159265f * freq;
        float t_now = tick * dt;
        float src = ga.src_amp * sinf(omega * t_now);

        // Wave equation: p_next = 2*p - p_prev + r*(laplacian(p))
        float p_next[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j) {
            for (int i = 1; i < N2D-1; ++i) {
                if (inside_scatterer(i, j)) {
                    p_next[j][i] = 0;  // rigid boundary
                    continue;
                }

                float lap = p[j][i-1] + p[j][i+1] + p[j-1][i] + p[j+1][i]
                          - 4.0f * p[j][i];

                p_next[j][i] = 2.0f * p[j][i] - p_prev[j][i]
                             + r * lap
                             - ga.damping * dt * (p[j][i] - p_prev[j][i]);
            }
        }

        // Source injection (left boundary strip)
        for (int j = 1; j < N2D-1; ++j) {
            p_next[j][1] += dt * dt * src;
            p_next[j][2] += dt * dt * src * 0.5f;
        }

        // Absorbing boundary (Mur first-order)
        float abc = (c * dt - dx) / (c * dt + dx);
        for (int j = 0; j < N2D; ++j) {
            // Right
            p_next[j][N2D-1] = p[j][N2D-2] + abc * (p_next[j][N2D-2] - p[j][N2D-1]);
            // Left (partial — source side)
            p_next[j][0] = p[j][1] + abc * (p_next[j][1] - p[j][0]);
        }
        for (int i = 0; i < N2D; ++i) {
            p_next[0][i] = p[1][i] + abc * (p_next[1][i] - p[0][i]);
            p_next[N2D-1][i] = p[N2D-2][i] + abc * (p_next[N2D-2][i] - p[N2D-1][i]);
        }

        // Swap
        memcpy(p_prev, p, sizeof(p));
        memcpy(p, p_next, sizeof(p));
        tick++;

        // Update fitness: higher = more scattered energy
        float scattered_energy = 0;
        for (int j = 0; j < N2D; ++j)
            for (int i = N2D/2; i < N2D; ++i)  // right half
                scattered_energy += p[j][i] * p[j][i];
        engine_ga.fitness = (int)(scattered_energy * 1000.0f);
    }

    void write_field(int field, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j * N2D + i;
                if (field == 0) out[idx] = p[j][i];
                else if (field == 1) out[idx] = vx[j][i];
                else out[idx] = vy[j][i];
            }
    }

    void ga_mutate(Rng& rng) override {
        PDE2D::ga_mutate(rng);
        ga.wave_speed  *= 1.0f + rng.normal(0, 0.05f);
        ga.src_freq    *= 1.0f + rng.normal(0, 0.03f);
        ga.damping     *= 1.0f + rng.normal(0, 0.05f);
        if (ga.wave_speed < 0.1f) ga.wave_speed = 0.1f;
        if (ga.damping < 0.001f) ga.damping = 0.001f;
    }
};

} // namespace well
