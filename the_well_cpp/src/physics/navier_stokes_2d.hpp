// ============================================================================
// NAVIER_STOKES_2D.HPP — 2D Incompressible Navier-Stokes (Vorticity-Stream)
//
// PDE: ∂ω/∂t + (u·∇)ω = ν∇²ω + f
// Where: ω = curl(u), ∇²ψ = -ω, u = (∂ψ/∂y, -∂ψ/∂x)
//
// Spatial: finite difference, periodic BC
// Temporal: RK2 (Heun's method)
// GA evolves: viscosity ν, forcing amplitude, forcing wavenumber
// ============================================================================
#pragma once

#include "pde_base.hpp"
#include <immintrin.h>

namespace well {

struct NavierStokes2D : PDE2D {
    // Component indices
    static constexpr int VORT = 0;   // vorticity ω
    static constexpr int STREAM = 1; // stream function ψ
    static constexpr int VEL_U = 2;  // velocity u
    static constexpr int VEL_V = 3;  // velocity v

    float* rhs_buf;    // RK2 scratch
    float* vort_tmp;   // temp vorticity

    const char* pde_name() const override { return "NavierStokes-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 4);  // ω, ψ, u, v
        dx = 2.0f * 3.14159265f / (float)NX;
        dy = 2.0f * 3.14159265f / (float)NY;
        dt = 0.005f;

        int spatial = NY * NX;
        rhs_buf = (float*)aligned_alloc_impl(32, spatial * sizeof(float));
        vort_tmp = (float*)aligned_alloc_impl(32, spatial * sizeof(float));

        // GA parameters
        ga.rng.seed(seed);
        ga.add_param(0.005f, 0.0001f, 0.1f, 0.002f);   // [0] viscosity ν
        ga.add_param(4.0f,   0.5f,    20.0f, 1.0f);     // [1] forcing amplitude
        ga.add_param(4.0f,   1.0f,    8.0f,  0.5f);     // [2] forcing wavenumber

        // Initial condition: Taylor-Green vortex + perturbation
        Rng rng;
        rng.seed(seed);
        float* w = component(VORT);

        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                float x = (float)i * dx, y = (float)j * dy;
                float kf = ga.params[2].value;
                w[j * NX + i] = 2.0f * sinf(kf * x) * sinf(kf * y)
                              + 0.1f * rng.normal(0.0f, 1.0f);
            }
        }

        solve_poisson();
        compute_velocity();
    }

    // ── Poisson solver: ∇²ψ = -ω (Jacobi iteration, periodic BC) ──
    void solve_poisson() {
        float* psi = component(STREAM);
        float* w = component(VORT);
        float inv_dx2 = 1.0f / (dx * dx);
        float inv_dy2 = 1.0f / (dy * dy);
        float a = -2.0f * (inv_dx2 + inv_dy2);

        // 50 Jacobi iterations (sufficient for smooth fields)
        for (int iter = 0; iter < 50; ++iter) {
            for (int j = 0; j < NY; ++j) {
                for (int i = 0; i < NX; ++i) {
                    float pL = psi[j * NX + px(i-1)];
                    float pR = psi[j * NX + px(i+1)];
                    float pD = psi[py(j-1) * NX + i];
                    float pU = psi[py(j+1) * NX + i];

                    psi[j * NX + i] = (inv_dx2 * (pL + pR) +
                                        inv_dy2 * (pD + pU) +
                                        w[j * NX + i]) / (-a);
                }
            }
        }
    }

    // ── Velocity from stream function: u = ∂ψ/∂y, v = -∂ψ/∂x ──
    void compute_velocity() {
        float* psi = component(STREAM);
        float* u = component(VEL_U);
        float* v = component(VEL_V);

        float inv_2dx = 0.5f / dx;
        float inv_2dy = 0.5f / dy;

        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                u[j * NX + i] = (psi[py(j+1) * NX + i] - psi[py(j-1) * NX + i]) * inv_2dy;
                v[j * NX + i] = -(psi[j * NX + px(i+1)] - psi[j * NX + px(i-1)]) * inv_2dx;
            }
        }
    }

    // ── RHS: -(u·∇)ω + ν∇²ω + f ──
    void compute_rhs(const float* w_in, float* rhs) {
        float* u = component(VEL_U);
        float* v = component(VEL_V);
        float nu = ga.params[0].value;
        float force_amp = ga.params[1].value;
        float force_k = ga.params[2].value;

        float inv_2dx = 0.5f / dx;
        float inv_2dy = 0.5f / dy;
        float inv_dx2 = 1.0f / (dx * dx);
        float inv_dy2 = 1.0f / (dy * dy);

        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = j * NX + i;

                // Advection: -(u·∂ω/∂x + v·∂ω/∂y) (central differences)
                float dw_dx = (w_in[j * NX + px(i+1)] - w_in[j * NX + px(i-1)]) * inv_2dx;
                float dw_dy = (w_in[py(j+1) * NX + i] - w_in[py(j-1) * NX + i]) * inv_2dy;
                float advection = -(u[idx] * dw_dx + v[idx] * dw_dy);

                // Diffusion: ν∇²ω
                float lap = (w_in[j * NX + px(i-1)] + w_in[j * NX + px(i+1)] - 2.0f * w_in[idx]) * inv_dx2
                          + (w_in[py(j-1) * NX + i] + w_in[py(j+1) * NX + i] - 2.0f * w_in[idx]) * inv_dy2;
                float diffusion = nu * lap;

                // Forcing: f = A * sin(k * y)
                float y = (float)j * dy;
                float forcing = force_amp * sinf(force_k * y);

                rhs[idx] = advection + diffusion + forcing;
            }
        }
    }

    // ── Advance one timestep (RK2 Heun's method) ──
    void advance() override {
        float* w = component(VORT);
        int spatial = NY * NX;

        // Stage 1: k1 = f(w_n)
        compute_rhs(w, rhs_buf);

        // w_tilde = w_n + dt * k1
        for (int i = 0; i < spatial; ++i)
            vort_tmp[i] = w[i] + dt * rhs_buf[i];

        // Update velocity from w_tilde
        memcpy(component(VORT), vort_tmp, spatial * sizeof(float));
        solve_poisson();
        compute_velocity();

        // Stage 2: k2 = f(w_tilde)
        float* rhs2 = vort_tmp; // reuse
        compute_rhs(component(VORT), rhs2);

        // Restore original vorticity, then update
        memcpy(component(VORT), w, 0); // already have w in stack scope...

        // w_{n+1} = w_n + dt/2 * (k1 + k2)
        __m256 vhdt = _mm256_set1_ps(0.5f * dt);
        int i = 0;
        for (; i + 8 <= spatial; i += 8) {
            __m256 vw = _mm256_loadu_ps(w + i);
            __m256 vk1 = _mm256_loadu_ps(rhs_buf + i);
            __m256 vk2 = _mm256_loadu_ps(rhs2 + i);
            __m256 vsum = _mm256_add_ps(vk1, vk2);
            _mm256_storeu_ps(w + i, _mm256_fmadd_ps(vhdt, vsum, vw));
        }
        for (; i < spatial; ++i)
            w[i] += 0.5f * dt * (rhs_buf[i] + rhs2[i]);

        // Update derived fields
        solve_poisson();
        compute_velocity();
    }

    float energy() const override {
        const float* u = component(VEL_U);
        const float* v = component(VEL_V);
        int spatial = NY * NX;
        float e = 0;
        for (int i = 0; i < spatial; ++i)
            e += u[i]*u[i] + v[i]*v[i];
        return 0.5f * e * dx * dy;
    }

    float compute_fitness() const override {
        float e = energy();
        // Enstrophy (vorticity squared)
        const float* w = component(VORT);
        int spatial = NY * NX;
        float ens = 0;
        for (int i = 0; i < spatial; ++i) ens += w[i]*w[i];
        ens *= 0.5f * dx * dy;

        // Fitness rewards structured flow with moderate energy
        float e_norm = e / (float)(NX * NY);
        float ens_norm = ens / (float)(NX * NY);
        float structure = (ens_norm > 1e-6f) ? sqrtf(ens_norm / (e_norm + 1e-8f)) : 0.0f;

        return 100.0f * structure + 10.0f * (1.0f / (1.0f + fabsf(e_norm - 0.5f)));
    }

    ~NavierStokes2D() {
        if (rhs_buf) aligned_free_impl(rhs_buf);
        if (vort_tmp) aligned_free_impl(vort_tmp);
    }
};

} // namespace well
