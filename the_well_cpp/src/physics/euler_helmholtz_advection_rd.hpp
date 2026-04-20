// ============================================================================
// EULER_HELMHOLTZ_ADVECTION_RD.HPP — CompressibleEuler, Helmholtz,
//                                     Advection, and Reaction-Diffusion 2D
//
// 4 PDE engines in one file. All inherit PDE2D with GA evolution.
// ============================================================================
#pragma once
#include "pde_base.hpp"

namespace well {

// ============================================================================
// CompressibleEuler2D — ∂ρ/∂t + ∇·(ρu) = 0
//                       ∂(ρu)/∂t + ∇·(ρu⊗u + pI) = 0
//                       p = (γ-1) ρ e  (ideal gas)
// ============================================================================
struct CompressibleEuler2D : PDE2D {
    static constexpr int DENSITY = 0;  // ρ
    static constexpr int MOM_X = 1;    // ρu
    static constexpr int MOM_Y = 2;    // ρv
    static constexpr int ENERGY_F = 3; // E = ρe + ½ρ(u²+v²)

    const char* pde_name() const override { return "CompressibleEuler-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 4);
        dx = 1.0f / (float)NX;
        dy = 1.0f / (float)NY;
        dt = 0.0001f;

        ga.rng.seed(seed);
        ga.add_param(1.4f, 1.1f, 2.0f, 0.05f);     // [0] gamma γ
        ga.add_param(1.0f, 0.1f, 5.0f, 0.3f);       // [1] density perturbation
        ga.add_param(1.0f, 0.1f, 5.0f, 0.3f);       // [2] pressure ratio

        Rng rng; rng.seed(seed);
        float* rho = component(DENSITY);
        float* E = component(ENERGY_F);
        float gamma = ga.params[0].value;
        float amp = ga.params[1].value;

        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float x = (float)i * dx, y = (float)j * dy;
                float r = (x < 0.5f && y < 0.5f) ? 1.0f * amp : 0.125f * amp;
                float p = (x < 0.5f && y < 0.5f) ? 1.0f * ga.params[2].value : 0.1f;
                rho[j*NX+i] = r;
                E[j*NX+i] = p / (gamma - 1.0f);
            }
    }

    void advance() override {
        float* rho = component(DENSITY);
        float* mu = component(MOM_X);
        float* mv = component(MOM_Y);
        float* E = component(ENERGY_F);
        float gamma = ga.params[0].value;
        float inv_2dx = 0.5f/dx, inv_2dy = 0.5f/dy;
        int sp = NY * NX;

        memcpy(fields_prev, fields, 4 * sp * sizeof(float));
        float* rp = fields_prev;
        float* mup = fields_prev + sp;
        float* mvp = fields_prev + 2*sp;
        float* Ep = fields_prev + 3*sp;

        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float r = rp[idx]; if (r < 1e-6f) r = 1e-6f;
                float u = mup[idx] / r;
                float v = mvp[idx] / r;
                float ek = 0.5f * r * (u*u + v*v);
                float p = (gamma - 1.0f) * (Ep[idx] - ek);
                if (p < 1e-6f) p = 1e-6f;

                // Density: ∂ρ/∂t = -∂(ρu)/∂x - ∂(ρv)/∂y
                float d_rhou_dx = (mup[j*NX+px(i+1)] - mup[j*NX+px(i-1)]) * inv_2dx;
                float d_rhov_dy = (mvp[py(j+1)*NX+i] - mvp[py(j-1)*NX+i]) * inv_2dy;
                rho[idx] = rp[idx] - dt * (d_rhou_dx + d_rhov_dy);
                if (rho[idx] < 1e-6f) rho[idx] = 1e-6f;

                // Momentum X
                float rR = rp[j*NX+px(i+1)]; if (rR < 1e-6f) rR = 1e-6f;
                float rL = rp[j*NX+px(i-1)]; if (rL < 1e-6f) rL = 1e-6f;
                float pR = (gamma-1.0f)*(Ep[j*NX+px(i+1)] - 0.5f*mup[j*NX+px(i+1)]*mup[j*NX+px(i+1)]/rR);
                float pL = (gamma-1.0f)*(Ep[j*NX+px(i-1)] - 0.5f*mup[j*NX+px(i-1)]*mup[j*NX+px(i-1)]/rL);
                mu[idx] = mup[idx] - dt * (u*d_rhou_dx + (pR - pL) * inv_2dx);

                // Momentum Y (simplified)
                mv[idx] = mvp[idx] - dt * (v*d_rhov_dy +
                    ((gamma-1.0f)*(Ep[py(j+1)*NX+i] - Ep[py(j-1)*NX+i])) * inv_2dy * 0.5f);

                // Energy
                float dEu = (Ep[j*NX+px(i+1)]*mup[j*NX+px(i+1)]/rR -
                             Ep[j*NX+px(i-1)]*mup[j*NX+px(i-1)]/rL) * inv_2dx;
                E[idx] = Ep[idx] - dt * dEu;
            }
        }
    }

    float energy() const override {
        const float* E = component(ENERGY_F);
        float e = 0;
        for (int i = 0; i < NY*NX; ++i) e += E[i];
        return e * dx * dy;
    }

    float compute_fitness() const override {
        float e = energy();
        return 50.0f / (1.0f + fabsf(logf(fabsf(e) + 1e-10f)));
    }
};

// ============================================================================
// Helmholtz2D — ∂u/∂t = D∇²u - k²u + f
// Damped wave with source
// ============================================================================
struct Helmholtz2D : PDE2D {
    const char* pde_name() const override { return "Helmholtz-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 1);
        dx = 1.0f / (float)NX;
        dy = 1.0f / (float)NY;
        dt = 0.0001f;

        ga.rng.seed(seed);
        ga.add_param(0.01f, 0.001f, 0.1f, 0.005f);  // [0] diffusivity D
        ga.add_param(5.0f,  1.0f,   20.0f, 1.0f);    // [1] k (damping)
        ga.add_param(2.0f,  0.5f,   10.0f, 0.5f);    // [2] source amp

        Rng rng; rng.seed(seed);
        float* u = component(0);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                u[j*NX+i] = rng.normal(0, 0.5f);
    }

    void advance() override {
        float* u = component(0);
        float D = ga.params[0].value;
        float k2 = ga.params[1].value * ga.params[1].value;
        float src = ga.params[2].value;
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);
        float t = (float)tick * dt;

        memcpy(fields_prev, fields, NY*NX*sizeof(float));
        float* up = fields_prev;

        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float lap = (up[j*NX+px(i-1)] + up[j*NX+px(i+1)] - 2.0f*up[idx])*inv_dx2
                          + (up[py(j-1)*NX+i] + up[py(j+1)*NX+i] - 2.0f*up[idx])*inv_dy2;
                float x = (float)i*dx, y = (float)j*dy;
                float forcing = src * sinf(3.0f*x + t) * cosf(2.0f*y - 0.5f*t);
                u[idx] = up[idx] + dt * (D*lap - k2*up[idx] + forcing);
            }
    }

    float energy() const override {
        float e = 0;
        for (int i = 0; i < NY*NX; ++i) e += fields[i]*fields[i];
        return 0.5f * e * dx * dy;
    }
    float compute_fitness() const override { return 50.0f / (1.0f + energy()); }
};

// ============================================================================
// Advection2D — ∂u/∂t + c_x·∂u/∂x + c_y·∂u/∂y = 0
// Linear transport with GA-driven velocities
// ============================================================================
struct Advection2D : PDE2D {
    const char* pde_name() const override { return "Advection-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 1);
        dx = 1.0f / (float)NX;
        dy = 1.0f / (float)NY;
        dt = 0.001f;

        ga.rng.seed(seed);
        ga.add_param(1.0f,  -3.0f, 3.0f, 0.3f);   // [0] velocity cx
        ga.add_param(0.5f,  -3.0f, 3.0f, 0.3f);   // [1] velocity cy

        Rng rng; rng.seed(seed);
        float* u = component(0);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float x = (float)i*dx - 0.5f, y = (float)j*dy - 0.5f;
                u[j*NX+i] = expf(-50.0f*(x*x + y*y));
            }
    }

    void advance() override {
        float* u = component(0);
        float cx = ga.params[0].value, cy = ga.params[1].value;
        float inv_2dx = 0.5f / dx, inv_2dy = 0.5f / dy;

        memcpy(fields_prev, fields, NY*NX*sizeof(float));
        float* up = fields_prev;

        // Upwind scheme
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float du_dx, du_dy;
                if (cx > 0) du_dx = (up[idx] - up[j*NX+px(i-1)]) / dx;
                else        du_dx = (up[j*NX+px(i+1)] - up[idx]) / dx;
                if (cy > 0) du_dy = (up[idx] - up[py(j-1)*NX+i]) / dy;
                else        du_dy = (up[py(j+1)*NX+i] - up[idx]) / dy;
                u[idx] = up[idx] - dt * (cx*du_dx + cy*du_dy);
            }
    }

    float energy() const override {
        float e = 0;
        for (int i = 0; i < NY*NX; ++i) e += fields[i]*fields[i];
        return e * dx * dy;
    }
    float compute_fitness() const override {
        // Reward shape preservation
        float e = energy();
        float peak = 0;
        for (int i = 0; i < NY*NX; ++i) if (fields[i] > peak) peak = fields[i];
        return 30.0f * peak + 20.0f / (1.0f + fabsf(e - 0.03f));
    }
};

// ============================================================================
// GrayScott2D — Reaction-Diffusion (Gray-Scott model)
// PDE: ∂u/∂t = Du∇²u - uv² + F(1-u)
//      ∂v/∂t = Dv∇²v + uv² - (F+k)v
// GA evolves: Du, Dv, F, k
// ============================================================================
struct GrayScott2D : PDE2D {
    static constexpr int U_COMP = 0;
    static constexpr int V_COMP = 1;

    const char* pde_name() const override { return "GrayScott-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 2);
        dx = 1.0f / (float)NX;
        dy = 1.0f / (float)NY;
        dt = 1.0f;  // Gray-Scott uses larger timesteps

        ga.rng.seed(seed);
        ga.add_param(0.16f,  0.01f, 0.5f, 0.02f);   // [0] Du
        ga.add_param(0.08f,  0.005f, 0.25f, 0.01f);  // [1] Dv
        ga.add_param(0.04f,  0.01f, 0.08f, 0.005f);  // [2] F (feed rate)
        ga.add_param(0.06f,  0.03f, 0.07f, 0.003f);  // [3] k (kill rate)

        float* u = component(U_COMP);
        float* v = component(V_COMP);
        Rng rng; rng.seed(seed);

        // Initialize: u=1 everywhere, v=0 with seed patches
        for (int i = 0; i < NY*NX; ++i) u[i] = 1.0f;
        memset(v, 0, NY*NX*sizeof(float));

        // Seed patches of v
        for (int p = 0; p < 5; ++p) {
            int cx = 20 + (int)rng.next() % (NX-40);
            int cy = 20 + (int)rng.next() % (NY-40);
            for (int dj = -5; dj <= 5; ++dj)
                for (int di = -5; di <= 5; ++di) {
                    int ii = px(cx+di), jj = py(cy+dj);
                    u[jj*NX+ii] = 0.5f;
                    v[jj*NX+ii] = 0.25f + 0.01f * rng.normal();
                }
        }
    }

    void advance() override {
        float* u = component(U_COMP);
        float* v = component(V_COMP);
        float Du = ga.params[0].value;
        float Dv = ga.params[1].value;
        float F = ga.params[2].value;
        float k = ga.params[3].value;
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);
        int sp = NY * NX;

        memcpy(fields_prev, fields, 2*sp*sizeof(float));
        float* up = fields_prev;
        float* vp = fields_prev + sp;

        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float uc = up[idx], vc = vp[idx];

                float lap_u = (up[j*NX+px(i-1)] + up[j*NX+px(i+1)] - 2.0f*uc)*inv_dx2
                            + (up[py(j-1)*NX+i] + up[py(j+1)*NX+i] - 2.0f*uc)*inv_dy2;
                float lap_v = (vp[j*NX+px(i-1)] + vp[j*NX+px(i+1)] - 2.0f*vc)*inv_dx2
                            + (vp[py(j-1)*NX+i] + vp[py(j+1)*NX+i] - 2.0f*vc)*inv_dy2;

                float uvv = uc * vc * vc;
                u[idx] = uc + dt * (Du*lap_u - uvv + F*(1.0f - uc));
                v[idx] = vc + dt * (Dv*lap_v + uvv - (F+k)*vc);
            }
    }

    float energy() const override {
        float e = 0;
        const float* v = component(V_COMP);
        for (int i = 0; i < NY*NX; ++i) e += v[i]*v[i];
        return e * dx * dy;
    }

    float compute_fitness() const override {
        // Reward Turing patterns (high spatial variation of v)
        const float* v = component(V_COMP);
        float grad_e = 0;
        for (int j = 1; j < NY-1; ++j)
            for (int i = 1; i < NX-1; ++i) {
                float gx = v[j*NX+i+1] - v[j*NX+i-1];
                float gy = v[(j+1)*NX+i] - v[(j-1)*NX+i];
                grad_e += gx*gx + gy*gy;
            }
        return 50.0f * sqrtf(grad_e / (float)(NX*NY));
    }
};

} // namespace well
