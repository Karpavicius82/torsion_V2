// ============================================================================
// DIFFUSION_2D.HPP — 2D Heat/Diffusion Equation
// PDE: ∂u/∂t = D∇²u + source
// GA evolves: diffusion D, source amplitude, source wavenumber
// ============================================================================
#pragma once
#include "pde_base.hpp"
#include <immintrin.h>

namespace well {

struct Diffusion2D : PDE2D {
    const char* pde_name() const override { return "Diffusion-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 1);
        dx = 1.0f / (float)NX;
        dy = 1.0f / (float)NY;
        dt = 0.0001f;

        ga.rng.seed(seed);
        ga.add_param(0.01f, 0.001f, 0.1f, 0.005f);   // [0] diffusivity D
        ga.add_param(5.0f,  0.5f,   20.0f, 1.0f);     // [1] source amplitude
        ga.add_param(3.0f,  1.0f,   8.0f,  0.5f);     // [2] source wavenumber

        Rng rng; rng.seed(seed);
        float* u = component(0);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float x = (float)i * dx, y = (float)j * dy;
                u[j * NX + i] = sinf(3.14159f * 2.0f * x) * sinf(3.14159f * 2.0f * y)
                              + 0.1f * rng.normal();
            }
    }

    void advance() override {
        float* u = component(0);
        float D = ga.params[0].value;
        float src_amp = ga.params[1].value;
        float src_k = ga.params[2].value;
        float inv_dx2 = 1.0f / (dx * dx);
        float inv_dy2 = 1.0f / (dy * dy);
        float t = (float)tick * dt;

        // Copy to prev
        memcpy(fields_prev, fields, NY * NX * sizeof(float));
        float* up = fields_prev;

        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                float lap = (up[j*NX+px(i-1)] + up[j*NX+px(i+1)] - 2.0f*up[j*NX+i]) * inv_dx2
                          + (up[py(j-1)*NX+i] + up[py(j+1)*NX+i] - 2.0f*up[j*NX+i]) * inv_dy2;

                float x = (float)i * dx, y = (float)j * dy;
                float source = src_amp * sinf(src_k * x) * cosf(src_k * y) * cosf(2.0f * t);

                u[j*NX+i] = up[j*NX+i] + dt * (D * lap + source);
            }
        }
    }

    float energy() const override {
        const float* u = component(0);
        float e = 0;
        for (int i = 0; i < NY*NX; ++i) e += u[i]*u[i];
        return 0.5f * e * dx * dy;
    }

    float compute_fitness() const override {
        float e = energy();
        float e_norm = e / (float)(NX*NY);
        // Reward moderate energy with spatial structure
        float grad_e = 0;
        const float* u = component(0);
        for (int j = 1; j < NY-1; ++j)
            for (int i = 1; i < NX-1; ++i) {
                float gx = u[j*NX+i+1] - u[j*NX+i-1];
                float gy = u[(j+1)*NX+i] - u[(j-1)*NX+i];
                grad_e += gx*gx + gy*gy;
            }
        return 50.0f * sqrtf(grad_e / (float)(NX*NY)) + 10.0f / (1.0f + fabsf(e_norm - 0.3f));
    }
};

// ============================================================================
// WAVE_2D.HPP — 2D Wave Equation
// PDE: ∂²u/∂t² = c²∇²u - damping·∂u/∂t
// GA evolves: wave speed c, damping, source frequency
// ============================================================================
struct Wave2D : PDE2D {
    static constexpr int U_FIELD = 0;
    static constexpr int V_FIELD = 1;  // velocity ∂u/∂t

    const char* pde_name() const override { return "Wave-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 2);  // u, v
        dx = 1.0f / (float)NX;
        dy = 1.0f / (float)NY;
        dt = 0.0005f;

        ga.rng.seed(seed);
        ga.add_param(1.0f, 0.1f, 5.0f, 0.2f);    // [0] wave speed c
        ga.add_param(0.01f, 0.0f, 0.1f, 0.005f);  // [1] damping
        ga.add_param(3.0f, 1.0f, 8.0f, 0.5f);     // [2] source wavenumber

        Rng rng; rng.seed(seed);
        float* u = component(U_FIELD);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float x = (float)i * dx, y = (float)j * dy;
                float k = ga.params[2].value;
                u[j*NX+i] = sinf(k * 3.14159f * x) * sinf(k * 3.14159f * y);
            }
    }

    void advance() override {
        float* u = component(U_FIELD);
        float* v = component(V_FIELD);
        float c = ga.params[0].value;
        float c2 = c * c;
        float damp = ga.params[1].value;
        float inv_dx2 = 1.0f / (dx*dx);
        float inv_dy2 = 1.0f / (dy*dy);

        // Symplectic leapfrog: update v first, then u
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = j * NX + i;
                float lap = (u[j*NX+px(i-1)] + u[j*NX+px(i+1)] - 2.0f*u[idx]) * inv_dx2
                          + (u[py(j-1)*NX+i] + u[py(j+1)*NX+i] - 2.0f*u[idx]) * inv_dy2;
                v[idx] += dt * (c2 * lap - damp * v[idx]);
            }
        }

        // Update u
        __m256 vdt = _mm256_set1_ps(dt);
        int n = NY * NX, k = 0;
        for (; k + 8 <= n; k += 8) {
            __m256 vu = _mm256_loadu_ps(u + k);
            __m256 vv = _mm256_loadu_ps(v + k);
            _mm256_storeu_ps(u + k, _mm256_fmadd_ps(vdt, vv, vu));
        }
        for (; k < n; ++k) u[k] += dt * v[k];
    }

    float energy() const override {
        const float* u = component(U_FIELD);
        const float* v = component(V_FIELD);
        float c = ga.params[0].value;
        float e = 0;
        for (int i = 0; i < NY*NX; ++i) e += v[i]*v[i] + c*c*u[i]*u[i];
        return 0.5f * e * dx * dy;
    }

    float compute_fitness() const override {
        float e = energy();
        // Measure wave coherence via spatial correlation
        const float* u = component(U_FIELD);
        float corr = 0;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX-1; ++i)
                corr += u[j*NX+i] * u[j*NX+i+1];
        corr /= (float)(NX*NY);
        return 50.0f * fabsf(corr) + 10.0f / (1.0f + fabsf(logf(e + 1e-10f)));
    }
};

// ============================================================================
// BURGERS_2D.HPP — 2D Viscous Burgers' Equation
// PDE: ∂u/∂t + u·∂u/∂x + v·∂u/∂y = ν∇²u
//      ∂v/∂t + u·∂v/∂x + v·∂v/∂y = ν∇²v
// GA evolves: viscosity ν, initial amplitude, wavenumber
// ============================================================================
struct Burgers2D : PDE2D {
    static constexpr int U_COMP = 0;
    static constexpr int V_COMP = 1;

    const char* pde_name() const override { return "Burgers-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 2);
        dx = 2.0f * 3.14159265f / (float)NX;
        dy = 2.0f * 3.14159265f / (float)NY;
        dt = 0.001f;

        ga.rng.seed(seed);
        ga.add_param(0.01f, 0.001f, 0.1f, 0.003f);  // [0] viscosity ν
        ga.add_param(1.0f,  0.1f,   5.0f, 0.3f);     // [1] amplitude
        ga.add_param(2.0f,  1.0f,   6.0f, 0.5f);     // [2] wavenumber

        Rng rng; rng.seed(seed);
        float* u = component(U_COMP);
        float* v = component(V_COMP);
        float amp = ga.params[1].value;
        float k = ga.params[2].value;

        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float x = (float)i * dx, y = (float)j * dy;
                u[j*NX+i] = amp * sinf(k*x) * cosf(k*y) + 0.05f * rng.normal();
                v[j*NX+i] = -amp * cosf(k*x) * sinf(k*y) + 0.05f * rng.normal();
            }
    }

    void advance() override {
        float* u = component(U_COMP);
        float* v = component(V_COMP);
        float nu = ga.params[0].value;
        float inv_2dx = 0.5f / dx, inv_2dy = 0.5f / dy;
        float inv_dx2 = 1.0f / (dx*dx), inv_dy2 = 1.0f / (dy*dy);
        int spatial = NY * NX;

        memcpy(fields_prev, fields, 2 * spatial * sizeof(float));
        float* up = fields_prev;
        float* vp = fields_prev + spatial;

        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float uc = up[idx], vc = vp[idx];

                // Advection
                float du_dx = (up[j*NX+px(i+1)] - up[j*NX+px(i-1)]) * inv_2dx;
                float du_dy = (up[py(j+1)*NX+i] - up[py(j-1)*NX+i]) * inv_2dy;
                float dv_dx = (vp[j*NX+px(i+1)] - vp[j*NX+px(i-1)]) * inv_2dx;
                float dv_dy = (vp[py(j+1)*NX+i] - vp[py(j-1)*NX+i]) * inv_2dy;

                // Diffusion
                float lap_u = (up[j*NX+px(i-1)] + up[j*NX+px(i+1)] - 2.0f*uc) * inv_dx2
                            + (up[py(j-1)*NX+i] + up[py(j+1)*NX+i] - 2.0f*uc) * inv_dy2;
                float lap_v = (vp[j*NX+px(i-1)] + vp[j*NX+px(i+1)] - 2.0f*vc) * inv_dx2
                            + (vp[py(j-1)*NX+i] + vp[py(j+1)*NX+i] - 2.0f*vc) * inv_dy2;

                u[idx] = uc + dt * (-uc*du_dx - vc*du_dy + nu*lap_u);
                v[idx] = vc + dt * (-uc*dv_dx - vc*dv_dy + nu*lap_v);
            }
        }
    }

    float energy() const override {
        const float* u = component(U_COMP);
        const float* v = component(V_COMP);
        float e = 0;
        for (int i = 0; i < NY*NX; ++i) e += u[i]*u[i] + v[i]*v[i];
        return 0.5f * e * dx * dy;
    }

    float compute_fitness() const override {
        float e = energy();
        // Reward shock formation (high gradients) with stability
        float max_grad = 0;
        const float* u = component(U_COMP);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX-1; ++i) {
                float g = fabsf(u[j*NX+i+1] - u[j*NX+i]) / dx;
                if (g > max_grad) max_grad = g;
            }
        float shock_score = (max_grad > 1.0f) ? logf(max_grad) : 0.0f;
        return 30.0f * shock_score + 20.0f / (1.0f + fabsf(e - 1.0f));
    }
};

} // namespace well
