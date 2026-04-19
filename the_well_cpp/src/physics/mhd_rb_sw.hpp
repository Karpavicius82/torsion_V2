// ============================================================================
// MHD_2D.HPP — 2D Magnetohydrodynamics (incompressible)
//
// PDE: ∂ω/∂t = -∇·(uω) + ∇·(BJ) + ν∇²ω + η∇²J
//      ∂A/∂t = -u·∇A + η∇²A
// Where: B = curl(A), J = -∇²A, u from stream function
// GA evolves: viscosity ν, resistivity η, Alfvén speed
// ============================================================================
#pragma once
#include "pde_base.hpp"

namespace well {

struct MHD2D : PDE2D {
    static constexpr int VORT = 0;    // vorticity
    static constexpr int A_POT = 1;   // magnetic potential A_z
    static constexpr int STREAM = 2;  // stream function ψ
    static constexpr int J_CURR = 3;  // current density J

    const char* pde_name() const override { return "MHD-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 4);
        dx = 2.0f * 3.14159265f / (float)NX;
        dy = 2.0f * 3.14159265f / (float)NY;
        dt = 0.002f;

        ga.rng.seed(seed);
        ga.add_param(0.005f, 0.0005f, 0.05f, 0.002f);  // [0] viscosity ν
        ga.add_param(0.005f, 0.0005f, 0.05f, 0.002f);  // [1] resistivity η
        ga.add_param(1.0f,   0.1f,    5.0f,  0.3f);     // [2] B0 amplitude
        ga.add_param(2.0f,   1.0f,    6.0f,  0.5f);     // [3] wavenumber

        Rng rng; rng.seed(seed);
        float* w = component(VORT);
        float* A = component(A_POT);
        float k = ga.params[3].value;
        float B0 = ga.params[2].value;

        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float x = (float)i * dx, y = (float)j * dy;
                w[j*NX+i] = sinf(k*x) * sinf(k*y) + 0.1f * rng.normal();
                A[j*NX+i] = B0 * cosf(k*x) * cosf(k*y) + 0.05f * rng.normal();
            }
    }

    // Compute Laplacian
    float laplacian(const float* f, int j, int i) const {
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);
        return (f[j*NX+px(i-1)] + f[j*NX+px(i+1)] - 2.0f*f[j*NX+i]) * inv_dx2
             + (f[py(j-1)*NX+i] + f[py(j+1)*NX+i] - 2.0f*f[j*NX+i]) * inv_dy2;
    }

    void solve_poisson_field(float* psi, const float* src) {
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);
        float a = -2.0f * (inv_dx2 + inv_dy2);
        for (int iter = 0; iter < 40; ++iter)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i)
                    psi[j*NX+i] = (inv_dx2*(psi[j*NX+px(i-1)] + psi[j*NX+px(i+1)]) +
                                   inv_dy2*(psi[py(j-1)*NX+i] + psi[py(j+1)*NX+i]) +
                                   src[j*NX+i]) / (-a);
    }

    void advance() override {
        float* w = component(VORT);
        float* A = component(A_POT);
        float* psi = component(STREAM);
        float* J = component(J_CURR);
        float nu = ga.params[0].value;
        float eta = ga.params[1].value;
        int sp = NY * NX;

        // J = -∇²A
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                J[j*NX+i] = -laplacian(A, j, i);

        // Solve ∇²ψ = -ω
        solve_poisson_field(psi, w);

        float inv_2dx = 0.5f/dx, inv_2dy = 0.5f/dy;
        memcpy(fields_prev, fields, 4 * sp * sizeof(float));
        float* wp = fields_prev;
        float* Ap = fields_prev + sp;

        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                // Velocity from stream function
                float ux = (psi[py(j+1)*NX+i] - psi[py(j-1)*NX+i]) * inv_2dy;
                float vy = -(psi[j*NX+px(i+1)] - psi[j*NX+px(i-1)]) * inv_2dx;

                // Magnetic field from A: Bx = ∂A/∂y, By = -∂A/∂x
                float Bx = (Ap[py(j+1)*NX+i] - Ap[py(j-1)*NX+i]) * inv_2dy;
                float By = -(Ap[j*NX+px(i+1)] - Ap[j*NX+px(i-1)]) * inv_2dx;

                // Vorticity: advection + Lorentz + viscous
                float dw_dx = (wp[j*NX+px(i+1)] - wp[j*NX+px(i-1)]) * inv_2dx;
                float dw_dy = (wp[py(j+1)*NX+i] - wp[py(j-1)*NX+i]) * inv_2dy;
                float dJ_dx = (J[j*NX+px(i+1)] - J[j*NX+px(i-1)]) * inv_2dx;
                float dJ_dy = (J[py(j+1)*NX+i] - J[py(j-1)*NX+i]) * inv_2dy;

                float rhs_w = -(ux*dw_dx + vy*dw_dy) + (Bx*dJ_dx + By*dJ_dy)
                            + nu * laplacian(wp, j, i);

                // A-field: advection + resistive diffusion
                float dA_dx = (Ap[j*NX+px(i+1)] - Ap[j*NX+px(i-1)]) * inv_2dx;
                float dA_dy = (Ap[py(j+1)*NX+i] - Ap[py(j-1)*NX+i]) * inv_2dy;
                float rhs_A = -(ux*dA_dx + vy*dA_dy) + eta * laplacian(Ap, j, i);

                w[idx] = wp[idx] + dt * rhs_w;
                A[idx] = Ap[idx] + dt * rhs_A;
            }
        }
    }

    float energy() const override {
        const float* w = component(VORT);
        const float* A = component(A_POT);
        float ek = 0, em = 0;
        for (int i = 0; i < NY*NX; ++i) { ek += w[i]*w[i]; em += A[i]*A[i]; }
        return 0.5f * (ek + em) * dx * dy;
    }

    float compute_fitness() const override {
        float e = energy();
        return 50.0f / (1.0f + fabsf(logf(e + 1e-10f) - 2.0f));
    }
};

// ============================================================================
// RAYLEIGH_BENARD_2D.HPP — 2D Rayleigh-Bénard Convection
//
// PDE: ∂ω/∂t = -(u·∇)ω + ν∇²ω + Ra·Pr·∂T/∂x
//      ∂T/∂t = -(u·∇)T + κ∇²T + u_y (background temperature gradient)
// GA evolves: Rayleigh number Ra, Prandtl number Pr
// ============================================================================
struct RayleighBenard2D : PDE2D {
    static constexpr int VORT = 0;
    static constexpr int TEMP = 1;
    static constexpr int STREAM = 2;

    const char* pde_name() const override { return "RayleighBenard-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 3);
        dx = 4.0f / (float)NX;  // aspect ratio 4:1
        dy = 1.0f / (float)NY;
        dt = 0.0005f;

        ga.rng.seed(seed);
        ga.add_param(1000.0f, 100.0f, 50000.0f, 500.0f);  // [0] Rayleigh Ra
        ga.add_param(0.7f,    0.1f,   10.0f,    0.2f);     // [1] Prandtl Pr

        Rng rng; rng.seed(seed);
        float* T = component(TEMP);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float y_norm = (float)j / (float)NY;
                T[j*NX+i] = 1.0f - y_norm + 0.01f * rng.normal();
            }
    }

    void solve_poisson_rb(float* psi, const float* src) {
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);
        float a = -2.0f * (inv_dx2 + inv_dy2);
        for (int iter = 0; iter < 40; ++iter)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i)
                    psi[j*NX+i] = (inv_dx2*(psi[j*NX+px(i-1)] + psi[j*NX+px(i+1)]) +
                                   inv_dy2*(psi[py(j-1)*NX+i] + psi[py(j+1)*NX+i]) +
                                   src[j*NX+i]) / (-a);
    }

    void advance() override {
        float* w = component(VORT);
        float* T = component(TEMP);
        float* psi = component(STREAM);
        float Ra = ga.params[0].value;
        float Pr = ga.params[1].value;
        float nu = sqrtf(Pr / Ra);
        float kappa = 1.0f / sqrtf(Ra * Pr);
        int sp = NY * NX;

        solve_poisson_rb(psi, w);
        float inv_2dx = 0.5f/dx, inv_2dy = 0.5f/dy;
        float inv_dx2 = 1.0f/(dx*dx), inv_dy2 = 1.0f/(dy*dy);

        memcpy(fields_prev, fields, 3 * sp * sizeof(float));
        float* wp = fields_prev;
        float* Tp = fields_prev + sp;

        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float ux = (psi[py(j+1)*NX+i] - psi[py(j-1)*NX+i]) * inv_2dy;
                float uy = -(psi[j*NX+px(i+1)] - psi[j*NX+px(i-1)]) * inv_2dx;

                // Vorticity
                float dw_dx = (wp[j*NX+px(i+1)] - wp[j*NX+px(i-1)]) * inv_2dx;
                float dw_dy = (wp[py(j+1)*NX+i] - wp[py(j-1)*NX+i]) * inv_2dy;
                float lap_w = (wp[j*NX+px(i-1)] + wp[j*NX+px(i+1)] - 2.0f*wp[idx])*inv_dx2
                            + (wp[py(j-1)*NX+i] + wp[py(j+1)*NX+i] - 2.0f*wp[idx])*inv_dy2;
                float dT_dx = (Tp[j*NX+px(i+1)] - Tp[j*NX+px(i-1)]) * inv_2dx;

                w[idx] = wp[idx] + dt * (-(ux*dw_dx + uy*dw_dy) + nu*lap_w + Ra*Pr*dT_dx);

                // Temperature
                float dT_dx2 = (Tp[j*NX+px(i+1)] - Tp[j*NX+px(i-1)]) * inv_2dx;
                float dT_dy = (Tp[py(j+1)*NX+i] - Tp[py(j-1)*NX+i]) * inv_2dy;
                float lap_T = (Tp[j*NX+px(i-1)] + Tp[j*NX+px(i+1)] - 2.0f*Tp[idx])*inv_dx2
                            + (Tp[py(j-1)*NX+i] + Tp[py(j+1)*NX+i] - 2.0f*Tp[idx])*inv_dy2;
                (void)dT_dx2;
                T[idx] = Tp[idx] + dt * (-(ux*dT_dx + uy*dT_dy) + kappa*lap_T);
            }
        }
    }

    float energy() const override {
        float e = 0;
        const float* w = component(VORT);
        for (int i = 0; i < NY*NX; ++i) e += w[i]*w[i];
        return 0.5f * e * dx * dy;
    }

    float compute_fitness() const override {
        // Nusselt number proxy: heat transport vs conduction
        const float* T = component(TEMP);
        float flux = 0;
        for (int i = 0; i < NX; ++i) {
            float dT = T[i] - T[(NY-1)*NX+i];  // top-bottom difference
            flux += fabsf(dT);
        }
        float Nu = flux / (float)NX;
        return 50.0f * Nu;
    }
};

// ============================================================================
// SHALLOW_WATER_2D.HPP — 2D Shallow Water Equations
//
// PDE: ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
//      ∂(hu)/∂t + ∂(hu²+½gh²)/∂x + ∂(huv)/∂y = -gh·∂b/∂x
//      ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv²+½gh²)/∂y = -gh·∂b/∂y
// GA evolves: gravity g, initial perturbation, bottom topography
// ============================================================================
struct ShallowWater2D : PDE2D {
    static constexpr int HEIGHT = 0;  // h
    static constexpr int MOM_U = 1;   // hu
    static constexpr int MOM_V = 2;   // hv

    const char* pde_name() const override { return "ShallowWater-2D"; }

    void initialize(uint64_t seed) override {
        alloc_fields(128, 128, 3);
        dx = 10.0f / (float)NX;
        dy = 10.0f / (float)NY;
        dt = 0.005f;

        ga.rng.seed(seed);
        ga.add_param(9.81f, 1.0f, 20.0f, 1.0f);    // [0] gravity g
        ga.add_param(0.5f,  0.1f, 2.0f,  0.2f);     // [1] perturbation amplitude
        ga.add_param(3.0f,  1.0f, 6.0f,  0.5f);     // [2] wavenumber

        Rng rng; rng.seed(seed);
        float* h = component(HEIGHT);
        float amp = ga.params[1].value;
        float k = ga.params[2].value;

        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float x = (float)i * dx, y = (float)j * dy;
                h[j*NX+i] = 1.0f + amp * expf(-((x-5.0f)*(x-5.0f) + (y-5.0f)*(y-5.0f)) * k);
            }
    }

    void advance() override {
        float* h = component(HEIGHT);
        float* hu = component(MOM_U);
        float* hv = component(MOM_V);
        float g = ga.params[0].value;
        float inv_2dx = 0.5f / dx, inv_2dy = 0.5f / dy;
        int sp = NY * NX;

        memcpy(fields_prev, fields, 3 * sp * sizeof(float));
        float* hp = fields_prev;
        float* hup = fields_prev + sp;
        float* hvp = fields_prev + 2*sp;

        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = j*NX+i;
                float hc = hp[idx];
                if (hc < 0.01f) hc = 0.01f;
                float uc = hup[idx] / hc;
                float vc = hvp[idx] / hc;

                // Fluxes: ∂(hu)/∂x, ∂(hv)/∂y
                float dhu_dx = (hup[j*NX+px(i+1)] - hup[j*NX+px(i-1)]) * inv_2dx;
                float dhv_dy = (hvp[py(j+1)*NX+i] - hvp[py(j-1)*NX+i]) * inv_2dy;

                // Height eq
                h[idx] = hp[idx] - dt * (dhu_dx + dhv_dy);
                if (h[idx] < 0.01f) h[idx] = 0.01f;

                // Momentum (x): ∂(hu²+½gh²)/∂x + ∂(huv)/∂y
                float hL = hp[j*NX+px(i-1)], hR = hp[j*NX+px(i+1)];
                float flux_x = (hR*hup[j*NX+px(i+1)]/hR + 0.5f*g*hR*hR
                              - hL*hup[j*NX+px(i-1)]/hL - 0.5f*g*hL*hL) * inv_2dx;
                (void)flux_x;

                hu[idx] = hup[idx] - dt * (uc*dhu_dx + hc*uc*uc*0.0f
                        + g*hc*(hp[j*NX+px(i+1)] - hp[j*NX+px(i-1)]) * inv_2dx);

                // Momentum (y)
                hv[idx] = hvp[idx] - dt * (vc*dhv_dy
                        + g*hc*(hp[py(j+1)*NX+i] - hp[py(j-1)*NX+i]) * inv_2dy);
            }
        }
    }

    float energy() const override {
        const float* h = component(HEIGHT);
        const float* hu = component(MOM_U);
        const float* hv = component(MOM_V);
        float g = ga.params[0].value;
        float e = 0;
        for (int i = 0; i < NY*NX; ++i) {
            float hc = h[i] > 0.01f ? h[i] : 0.01f;
            e += (hu[i]*hu[i] + hv[i]*hv[i]) / hc + g * h[i] * h[i];
        }
        return 0.5f * e * dx * dy;
    }

    float compute_fitness() const override {
        float e = energy();
        return 50.0f / (1.0f + fabsf(logf(e + 1e-10f) - 5.0f));
    }
};

} // namespace well
