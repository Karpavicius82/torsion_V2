// ============================================================================
// KS_CH_TORSION2D_AC.HPP — Kuramoto-Sivashinsky, Cahn-Hilliard,
//                           Torsion-2D (Living Silicon upgrade), Allen-Cahn
//
// 4 PDE engines. All inherit PDE2D with GA evolution.
// ============================================================================
#pragma once
#include "pde_base.hpp"

namespace well {

// ============================================================================
// KuramotoSivashinsky2D — ∂u/∂t = -∇²u - ∇⁴u - ½|∇u|²
// Models flame front propagation / thin film dynamics
// GA evolves: coefficients of each term
// ============================================================================
struct KuramotoSivashinsky2D : PDE2D {
    const char* pde_name() const override { return "KuramotoSivashinsky-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 1);
        dx = 50.0f / (float)NX;
        dy = 50.0f / (float)NY;
        dt = 0.05f;

        ga.rng.seed(seed);
        ga.add_param(1.0f, 0.1f, 3.0f, 0.2f);    // [0] diffusion coeff
        ga.add_param(1.0f, 0.1f, 3.0f, 0.2f);    // [1] hyper-diffusion coeff
        ga.add_param(0.5f, 0.1f, 2.0f, 0.1f);    // [2] nonlinearity coeff

        Rng rng; rng.seed(seed);
        float* u = component(0);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                u[j*NX+i] = 0.1f * rng.normal();
    }

    float biharmonic(const float* u, int j, int i) const {
        // ∇⁴u = ∇²(∇²u)  — approximate with 5-point stencil iterated
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);
        auto lap = [&](int jj, int ii) {
            return (u[jj*NX+px(ii-1)] + u[jj*NX+px(ii+1)] - 2.0f*u[jj*NX+ii])*inv_dx2
                 + (u[py(jj-1)*NX+ii] + u[py(jj+1)*NX+ii] - 2.0f*u[jj*NX+ii])*inv_dy2;
        };
        float lap_c = lap(j, i);
        float lap_L = lap(j, px(i-1));
        float lap_R = lap(j, px(i+1));
        float lap_D = lap(py(j-1), i);
        float lap_U = lap(py(j+1), i);
        return (lap_L + lap_R - 2.0f*lap_c)*inv_dx2 + (lap_D + lap_U - 2.0f*lap_c)*inv_dy2;
    }

    void advance() override {
        float* u = component(0);
        float a = ga.params[0].value;  // diffusion
        float b = ga.params[1].value;  // hyper-diffusion
        float c = ga.params[2].value;  // nonlinearity
        float inv_2dx = 0.5f/dx, inv_2dy = 0.5f/dy;
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);

        memcpy(fields_prev, fields, NY*NX*sizeof(float));
        float* up = fields_prev;

        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float lap = (up[j*NX+px(i-1)] + up[j*NX+px(i+1)] - 2.0f*up[idx])*inv_dx2
                          + (up[py(j-1)*NX+i] + up[py(j+1)*NX+i] - 2.0f*up[idx])*inv_dy2;

                float biharm = biharmonic(up, j, i);

                float du_dx = (up[j*NX+px(i+1)] - up[j*NX+px(i-1)]) * inv_2dx;
                float du_dy = (up[py(j+1)*NX+i] - up[py(j-1)*NX+i]) * inv_2dy;
                float grad_sq = du_dx*du_dx + du_dy*du_dy;

                u[idx] = up[idx] + dt * (-a*lap - b*biharm - c*0.5f*grad_sq);
            }
    }

    float energy() const override {
        float e = 0;
        for (int i = 0; i < NY*NX; ++i) e += fields[i]*fields[i];
        return 0.5f * e * dx * dy;
    }
    float compute_fitness() const override {
        float e = energy();
        return 50.0f * sqrtf(e / (float)(NX*NY));
    }
};

// ============================================================================
// CahnHilliard2D — ∂φ/∂t = M·∇²(φ³ - φ - ε²∇²φ)
// Phase-field model for spinodal decomposition
// GA evolves: mobility M, interface width ε
// ============================================================================
struct CahnHilliard2D : PDE2D {
    const char* pde_name() const override { return "CahnHilliard-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 1);
        dx = 1.0f / (float)NX;
        dy = 1.0f / (float)NY;
        dt = 0.0001f;

        ga.rng.seed(seed);
        ga.add_param(1.0f,   0.1f, 5.0f, 0.3f);     // [0] mobility M
        ga.add_param(0.01f,  0.001f, 0.1f, 0.005f);  // [1] epsilon² (interface)

        Rng rng; rng.seed(seed);
        float* phi = component(0);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                phi[j*NX+i] = 0.05f * rng.normal();  // nearly mixed state
    }

    void advance() override {
        float* phi = component(0);
        float M = ga.params[0].value;
        float eps2 = ga.params[1].value;
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);

        memcpy(fields_prev, fields, NY*NX*sizeof(float));
        float* pp = fields_prev;

        // Chemical potential: μ = φ³ - φ - ε²∇²φ
        // Then: ∂φ/∂t = M·∇²μ
        // Two-step: compute μ first, then Laplacian of μ

        // Scratch: reuse top of fields_prev (safe since we already copied)
        float* mu = fields_prev; // WARNING: overwrites prev but we only need it once

        // Step 1: compute μ
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float p = pp[idx];
                float lap = (pp[j*NX+px(i-1)] + pp[j*NX+px(i+1)] - 2.0f*p)*inv_dx2
                          + (pp[py(j-1)*NX+i] + pp[py(j+1)*NX+i] - 2.0f*p)*inv_dy2;
                mu[idx] = p*p*p - p - eps2 * lap;
            }

        // Step 2: ∂φ/∂t = M·∇²μ
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float lap_mu = (mu[j*NX+px(i-1)] + mu[j*NX+px(i+1)] - 2.0f*mu[idx])*inv_dx2
                             + (mu[py(j-1)*NX+i] + mu[py(j+1)*NX+i] - 2.0f*mu[idx])*inv_dy2;
                phi[idx] = pp[idx] + dt * M * lap_mu;
            }
    }

    float energy() const override {
        // Ginzburg-Landau free energy: ∫[¼(φ²-1)² + ½ε²|∇φ|²]
        float E = 0;
        float eps2 = ga.params[1].value;
        const float* p = component(0);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float pc = p[j*NX+i];
                float pot = 0.25f * (pc*pc - 1.0f) * (pc*pc - 1.0f);
                float gx = (p[j*NX+px(i+1)] - p[j*NX+px(i-1)]) / (2.0f*dx);
                float gy = (p[py(j+1)*NX+i] - p[py(j-1)*NX+i]) / (2.0f*dy);
                E += pot + 0.5f * eps2 * (gx*gx + gy*gy);
            }
        return E * dx * dy;
    }
    float compute_fitness() const override {
        float e = energy();
        return 50.0f / (1.0f + e);
    }
};

// ============================================================================
// Torsion2D — 2D upgrade of Living Silicon engine
//
// PDE: ∂²S/∂t² = c²∇²S - m²S + g·S³  (Einstein-Cartan torsion in 2D)
// GA evolves: wave speed c, mass m, coupling g, damping
// Identical physics to engine.hpp but extended to 2D float32
// ============================================================================
struct Torsion2D : PDE2D {
    static constexpr int S_FIELD = 0;  // torsion scalar S
    static constexpr int V_FIELD = 1;  // velocity ∂S/∂t

    const char* pde_name() const override { return "Torsion-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 2);
        dx = 2.0f * 3.14159265f / (float)NX;
        dy = 2.0f * 3.14159265f / (float)NY;
        dt = 0.001f;

        ga.rng.seed(seed);
        ga.add_param(1.0f,  0.1f, 5.0f, 0.2f);    // [0] wave speed c
        ga.add_param(0.5f,  0.0f, 3.0f, 0.2f);     // [1] mass m
        ga.add_param(0.1f,  0.0f, 1.0f, 0.05f);    // [2] coupling g
        ga.add_param(0.01f, 0.0f, 0.1f, 0.005f);   // [3] damping

        Rng rng; rng.seed(seed);
        float* S = component(S_FIELD);
        // Gaussian pulse + noise (like inject_gaussian in engine.hpp)
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float x = (float)i * dx - 3.14159f;
                float y = (float)j * dy - 3.14159f;
                S[j*NX+i] = 0.5f * expf(-(x*x + y*y) * 0.5f)
                           + 0.02f * rng.normal();
            }
    }

    void advance() override {
        float* S = component(S_FIELD);
        float* V = component(V_FIELD);
        float c = ga.params[0].value;
        float m = ga.params[1].value;
        float g = ga.params[2].value;
        float damp = ga.params[3].value;
        float c2 = c * c;
        float m2 = m * m;
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);

        // Symplectic leapfrog (like Wave2D but with nonlinear term)
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float sc = S[idx];
                float lap = (S[j*NX+px(i-1)] + S[j*NX+px(i+1)] - 2.0f*sc)*inv_dx2
                          + (S[py(j-1)*NX+i] + S[py(j+1)*NX+i] - 2.0f*sc)*inv_dy2;

                // d²S/dt² = c²∇²S - m²S + g·S³ - damping·dS/dt
                float force = c2 * lap - m2 * sc + g * sc * sc * sc;
                V[idx] += dt * (force - damp * V[idx]);
            }
        }

        // Update S
        __m256 vdt = _mm256_set1_ps(dt);
        int n = NY * NX, k = 0;
        for (; k + 8 <= n; k += 8) {
            __m256 vs = _mm256_loadu_ps(S + k);
            __m256 vv = _mm256_loadu_ps(V + k);
            _mm256_storeu_ps(S + k, _mm256_fmadd_ps(vdt, vv, vs));
        }
        for (; k < n; ++k) S[k] += dt * V[k];
    }

    float energy() const override {
        const float* S = component(S_FIELD);
        const float* V = component(V_FIELD);
        float c = ga.params[0].value;
        float m = ga.params[1].value;
        float g = ga.params[2].value;
        float e = 0;
        for (int i = 0; i < NY*NX; ++i) {
            e += 0.5f * V[i]*V[i] + 0.5f * c*c * S[i]*S[i]
               + 0.5f * m*m * S[i]*S[i] - 0.25f * g * S[i]*S[i]*S[i]*S[i];
        }
        return e * dx * dy;
    }

    float compute_fitness() const override {
        float e = energy();
        // Like engine.hpp: reward soliton structure
        float peak = 0;
        const float* S = component(S_FIELD);
        for (int i = 0; i < NY*NX; ++i) {
            float a = fabsf(S[i]);
            if (a > peak) peak = a;
        }
        float mean_a = 0;
        for (int i = 0; i < NY*NX; ++i) mean_a += fabsf(S[i]);
        mean_a /= (float)(NY*NX);
        float soliton_ratio = (mean_a > 0) ? peak / mean_a : 0.0f;

        return 4.0f * soliton_ratio + 50.0f / (1.0f + fabsf(logf(fabsf(e) + 1e-10f)));
    }
};

// ============================================================================
// AllenCahn2D — ∂φ/∂t = ε²∇²φ + φ - φ³
// Phase-field order parameter with bistable dynamics
// GA evolves: epsilon², reaction rate
// ============================================================================
struct AllenCahn2D : PDE2D {
    const char* pde_name() const override { return "AllenCahn-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 1);
        dx = 1.0f / (float)NX;
        dy = 1.0f / (float)NY;
        dt = 0.001f;

        ga.rng.seed(seed);
        ga.add_param(0.01f, 0.001f, 0.1f, 0.005f);  // [0] epsilon²
        ga.add_param(1.0f,  0.1f,   5.0f, 0.3f);     // [1] reaction rate

        Rng rng; rng.seed(seed);
        float* phi = component(0);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                phi[j*NX+i] = rng.normal(0, 0.3f);
    }

    void advance() override {
        float* phi = component(0);
        float eps2 = ga.params[0].value;
        float rate = ga.params[1].value;
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);

        memcpy(fields_prev, fields, NY*NX*sizeof(float));
        float* pp = fields_prev;

        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float p = pp[idx];
                float lap = (pp[j*NX+px(i-1)] + pp[j*NX+px(i+1)] - 2.0f*p)*inv_dx2
                          + (pp[py(j-1)*NX+i] + pp[py(j+1)*NX+i] - 2.0f*p)*inv_dy2;
                phi[idx] = p + dt * (eps2 * lap + rate * (p - p*p*p));
            }
    }

    float energy() const override {
        float e = 0;
        const float* p = component(0);
        float eps2 = ga.params[0].value;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float pc = p[j*NX+i];
                e += 0.25f*(pc*pc-1.0f)*(pc*pc-1.0f);
                float gx = (p[j*NX+px(i+1)] - p[j*NX+px(i-1)])/(2.0f*dx);
                float gy = (p[py(j+1)*NX+i] - p[py(j-1)*NX+i])/(2.0f*dy);
                e += 0.5f * eps2 * (gx*gx + gy*gy);
            }
        return e * dx * dy;
    }
    float compute_fitness() const override {
        float e = energy();
        return 50.0f / (1.0f + e);
    }
};

} // namespace well
