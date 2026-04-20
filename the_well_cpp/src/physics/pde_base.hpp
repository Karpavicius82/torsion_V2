// ============================================================================
// PDE_BASE.HPP — Abstract base for all 2D Physics Engines
//
// Every PDE engine:
//   1. Maintains a 2D field (or multi-component field)
//   2. advance() — one timestep of the PDE integrator
//   3. energy() — total energy or relevant conserved quantity
//   4. generate_trajectory() — produces training data for ML models
//   5. GA-driven parameter evolution (like Living Silicon engine.hpp)
//
// All engines generate data on-the-fly. No HDF5. No disk I/O.
// ============================================================================
#pragma once

#include "../tensor.hpp"

namespace well {

// ── GA-driven parameter ──
struct GAParam {
    float value;
    float min_val, max_val;
    float mutation_scale;

    void init(float v, float lo, float hi, float ms = 0.1f) {
        value = v; min_val = lo; max_val = hi; mutation_scale = ms;
    }

    void mutate(Rng& rng) {
        float delta = rng.normal(0.0f, mutation_scale);
        value += delta;
        if (value < min_val) value = min_val;
        if (value > max_val) value = max_val;
    }
};

// ── Per-lane GA state (shared across all PDEs) ──
struct EngineGA {
    static constexpr int MAX_PARAMS = 8;

    GAParam params[MAX_PARAMS];
    int n_params = 0;

    float fitness = 0.0f, best_fitness = 0.0f;
    float ema_fitness = 0.0f;
    uint32_t mutations = 0, stagnation = 0, improvements = 0;
    Rng rng;

    void add_param(float v, float lo, float hi, float ms = 0.1f) {
        if (n_params < MAX_PARAMS) {
            params[n_params].init(v, lo, hi, ms);
            n_params++;
        }
    }

    void evaluate(float fit) {
        float prev_ema = ema_fitness;
        ema_fitness = ema_fitness * 0.875f + fit * 0.125f;
        float slope = ema_fitness - prev_ema;
        fitness = fit;

        if (fit > best_fitness) {
            best_fitness = fit;
            stagnation = 0;
            improvements++;
        } else {
            stagnation++;
        }

        // Mutate if stagnating or randomly
        if (stagnation >= 3 || (slope <= 0.0f && (rng.next() % 100) < 5)) {
            int pi = rng.next() % (uint32_t)n_params;
            params[pi].mutate(rng);
            mutations++;
            stagnation = 0;
        }
    }

    // Crossover from a better genome
    void crossover_from(const EngineGA& better, float blend = 0.25f) {
        for (int i = 0; i < n_params; ++i) {
            params[i].value = (1.0f - blend) * params[i].value +
                              blend * better.params[i].value;
        }
        stagnation = 0;
    }
};

// ── Abstract PDE Base ──
struct PDE2D {
    int NX, NY;                    // spatial grid
    int n_components;              // number of field components
    float dx, dy, dt;              // spatial/temporal resolution
    uint64_t tick;                 // current simulation time

    float* fields;                 // [n_components * NY * NX]
    float* fields_prev;            // for multi-step integrators

    EngineGA ga;

    virtual ~PDE2D() {
        if (fields) aligned_free_impl(fields);
        if (fields_prev) aligned_free_impl(fields_prev);
    }

    void alloc_fields(int nx, int ny, int nc) {
        NX = nx; NY = ny; n_components = nc;
        int total = nc * ny * nx;
        int alloc_n = (total + 7) & ~7;
        fields = (float*)aligned_alloc_impl(32, alloc_n * sizeof(float));
        fields_prev = (float*)aligned_alloc_impl(32, alloc_n * sizeof(float));
        memset(fields, 0, alloc_n * sizeof(float));
        memset(fields_prev, 0, alloc_n * sizeof(float));
        tick = 0;
    }

    // Field component accessor
    float* component(int c) { return fields + c * NY * NX; }
    const float* component(int c) const { return fields + c * NY * NX; }
    float& at(int c, int y, int x) { return fields[(c * NY + y) * NX + x]; }

    // Periodic boundary helper
    int px(int x) const { return ((x % NX) + NX) % NX; }
    int py(int y) const { return ((y % NY) + NY) % NY; }

    // ── Interface ──
    virtual const char* pde_name() const = 0;
    virtual void initialize(uint64_t seed) = 0;
    virtual void advance() = 0;
    virtual float energy() const = 0;
    virtual float compute_fitness() const = 0;

    // Run one tick with GA evaluation
    void step() {
        advance();
        tick++;

        // GA epoch every 64 ticks
        if ((tick & 0x3F) == 0) {
            float fit = compute_fitness();
            ga.evaluate(fit);
        }
    }

    // ── Trajectory generation for ML ──
    // Runs the engine for `n_frames` frames (each `stride` ticks apart),
    // writes subsampled field to output.
    // output: [n_frames, n_components, out_h, out_w]
    void generate_trajectory(float* output, int n_frames, int stride,
                              int out_h, int out_w) {
        int sub_y = NY / out_h;
        int sub_x = NX / out_w;

        for (int f = 0; f < n_frames; ++f) {
            for (int s = 0; s < stride; ++s) step();

            // Subsample each component
            for (int c = 0; c < n_components; ++c) {
                float* src = component(c);
                float* dst = output + ((f * n_components + c) * out_h) * out_w;

                for (int oy = 0; oy < out_h; ++oy) {
                    for (int ox = 0; ox < out_w; ++ox) {
                        dst[oy * out_w + ox] = src[(oy * sub_y) * NX + (ox * sub_x)];
                    }
                }
            }
        }
    }

    // Normalize field to [-1, 1] range
    void normalize_field() {
        int total = n_components * NY * NX;
        float mx = 0.0f;
        for (int i = 0; i < total; ++i) {
            float a = fabsf(fields[i]);
            if (a > mx) mx = a;
        }
        if (mx > 1e-10f) {
            float inv = 1.0f / mx;
            for (int i = 0; i < total; ++i) fields[i] *= inv;
        }
    }
};

} // namespace well
