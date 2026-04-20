// ============================================================================
// ACTIVE_MATTER_VISCO.HPP — Active Matter + Viscoelastic Instability
//
// Active Matter (Vicsek-like continuum):
//   ∂ρ/∂t = -∇·(ρv) + D_ρ ∇²ρ
//   ∂θ/∂t = -v·∇θ + D_θ ∇²θ + noise
//   v = v₀(cosθ, sinθ)  — self-propulsion
//
// Viscoelastic (Oldroyd-B):
//   ∂u/∂t + (u·∇)u = -∇p + ∇·(η_s ∇u) + ∇·τ
//   ∂τ/∂t + (u·∇)τ = τ·∇u + (∇u)^T·τ - τ/λ + η_p/λ (∇u + ∇u^T)
//
// GA evolves: v₀, D_ρ, D_θ, η_s, η_p, λ, noise_strength
// ============================================================================
#pragma once
#include "pde_base.hpp"

namespace well {

// ── Active Matter 2D ──
struct ActiveMatter2D : PDE2D {
    float rho[N2D][N2D] = {};   // density
    float theta[N2D][N2D] = {}; // orientation angle
    float vx[N2D][N2D] = {};
    float vy[N2D][N2D] = {};

    struct ActiveGA {
        float v0 = 0.5f;         // self-propulsion speed
        float D_rho = 0.01f;     // density diffusion
        float D_theta = 0.1f;    // orientation diffusion
        float noise = 0.3f;      // angular noise strength
    } ga;

    const char* name() const override { return "ActiveMatter-2D"; }
    int num_fields() const override { return 4; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                rho[j][i] = 1.0f + rng.normal(0, 0.1f);
                theta[j][i] = rng.uniform() * 6.2831853f;
                vx[j][i] = ga.v0 * cosf(theta[j][i]);
                vy[j][i] = ga.v0 * sinf(theta[j][i]);
            }
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float dt = dt_ > 0 ? dt_ : 0.1f * dx;

        float rho_new[N2D][N2D], theta_new[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j) {
            for (int i = 1; i < N2D-1; ++i) {
                // Velocity from orientation
                float vxi = ga.v0 * cosf(theta[j][i]);
                float vyi = ga.v0 * sinf(theta[j][i]);

                // Density: ∂ρ/∂t = -∇·(ρv) + D∇²ρ
                float drho_dx = (rho[j][i+1]*vx[j][i+1] - rho[j][i-1]*vx[j][i-1]) / (2.0f*dx);
                float drho_dy = (rho[j+1][i]*vy[j+1][i] - rho[j-1][i]*vy[j-1][i]) / (2.0f*dx);
                float lap_rho = (rho[j][i-1] + rho[j][i+1] + rho[j-1][i] + rho[j+1][i]
                               - 4.0f*rho[j][i]) / (dx*dx);
                rho_new[j][i] = rho[j][i] + dt * (-drho_dx - drho_dy + ga.D_rho * lap_rho);

                // Orientation: ∂θ/∂t = -v·∇θ + D∇²θ + noise
                float dtheta_dx = (theta[j][i+1] - theta[j][i-1]) / (2.0f*dx);
                float dtheta_dy = (theta[j+1][i] - theta[j-1][i]) / (2.0f*dx);
                float lap_theta = (theta[j][i-1] + theta[j][i+1] + theta[j-1][i] + theta[j+1][i]
                                 - 4.0f*theta[j][i]) / (dx*dx);

                // Mean-field alignment: average neighbor orientation
                float avg_sin = sinf(theta[j][i-1]) + sinf(theta[j][i+1]) +
                                sinf(theta[j-1][i]) + sinf(theta[j+1][i]);
                float avg_cos = cosf(theta[j][i-1]) + cosf(theta[j][i+1]) +
                                cosf(theta[j-1][i]) + cosf(theta[j+1][i]);
                float target_angle = atan2f(avg_sin, avg_cos);

                float alignment = sinf(target_angle - theta[j][i]);

                theta_new[j][i] = theta[j][i] + dt * (
                    -vxi * dtheta_dx - vyi * dtheta_dy
                    + ga.D_theta * lap_theta
                    + alignment
                    + ga.noise * rng.normal(0, 1.0f)
                );

                // Clamp density
                if (rho_new[j][i] < 0.01f) rho_new[j][i] = 0.01f;
            }
        }

        // Periodic BC
        for (int j = 0; j < N2D; ++j) {
            rho_new[j][0] = rho_new[j][N2D-2]; rho_new[j][N2D-1] = rho_new[j][1];
            theta_new[j][0] = theta_new[j][N2D-2]; theta_new[j][N2D-1] = theta_new[j][1];
        }
        for (int i = 0; i < N2D; ++i) {
            rho_new[0][i] = rho_new[N2D-2][i]; rho_new[N2D-1][i] = rho_new[1][i];
            theta_new[0][i] = theta_new[N2D-2][i]; theta_new[N2D-1][i] = theta_new[1][i];
        }

        memcpy(rho, rho_new, sizeof(rho));
        memcpy(theta, theta_new, sizeof(theta));

        // Update velocities
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                vx[j][i] = ga.v0 * cosf(theta[j][i]);
                vy[j][i] = ga.v0 * sinf(theta[j][i]);
            }
        tick++;

        // Fitness: order parameter (alignment)
        float sx = 0, sy = 0;
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                sx += cosf(theta[j][i]);
                sy += sinf(theta[j][i]);
            }
        float order = sqrtf(sx*sx + sy*sy) / (float)(N2D*N2D);
        engine_ga.fitness = (int)(order * 10000.0f);
    }

    void write_field(int field, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j * N2D + i;
                switch (field) {
                    case 0: out[idx] = rho[j][i]; break;
                    case 1: out[idx] = theta[j][i]; break;
                    case 2: out[idx] = vx[j][i]; break;
                    default: out[idx] = vy[j][i]; break;
                }
            }
    }

    void ga_mutate(Rng& r) override {
        PDE2D::ga_mutate(r);
        ga.v0     *= 1.0f + r.normal(0, 0.05f);
        ga.D_rho  *= 1.0f + r.normal(0, 0.05f);
        ga.D_theta *= 1.0f + r.normal(0, 0.05f);
        ga.noise  *= 1.0f + r.normal(0, 0.05f);
        if (ga.v0 < 0.01f) ga.v0 = 0.01f;
    }
};

// ── Viscoelastic Instability (Oldroyd-B) ──
struct ViscoelasticInstability2D : PDE2D {
    float ux[N2D][N2D] = {};
    float uy[N2D][N2D] = {};
    float press[N2D][N2D] = {};
    // Polymer stress tensor (symmetric): τxx, τxy, τyy
    float txx[N2D][N2D] = {};
    float txy[N2D][N2D] = {};
    float tyy[N2D][N2D] = {};

    struct ViscoGA {
        float eta_s = 0.1f;    // solvent viscosity
        float eta_p = 0.5f;    // polymer viscosity
        float lambda_ = 1.0f;  // relaxation time
        float Wi = 2.0f;       // Weissenberg number
    } ga;

    const char* name() const override { return "ViscoelasticInstability-2D"; }
    int num_fields() const override { return 5; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        // Couette-like base flow (shear)
        for (int j = 0; j < N2D; ++j) {
            float y = (float)j / (float)N2D;
            for (int i = 0; i < N2D; ++i) {
                ux[j][i] = y;  // linear shear profile
                uy[j][i] = rng.normal(0, 0.001f);
                txx[j][i] = 2.0f * ga.eta_p * ga.lambda_ * 1.0f;  // steady state
                txy[j][i] = ga.eta_p;
                tyy[j][i] = 0;
            }
        }
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float dt = dt_ > 0 ? dt_ : 0.01f * dx;

        float ux_new[N2D][N2D], uy_new[N2D][N2D];
        float txx_new[N2D][N2D], txy_new[N2D][N2D], tyy_new[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j) {
            for (int i = 1; i < N2D-1; ++i) {
                // Velocity gradients
                float dudx = (ux[j][i+1] - ux[j][i-1]) / (2.0f*dx);
                float dudy = (ux[j+1][i] - ux[j-1][i]) / (2.0f*dx);
                float dvdx = (uy[j][i+1] - uy[j][i-1]) / (2.0f*dx);
                float dvdy = (uy[j+1][i] - uy[j-1][i]) / (2.0f*dx);

                // Laplacians
                float lap_u = (ux[j][i-1] + ux[j][i+1] + ux[j-1][i] + ux[j+1][i]
                             - 4.0f*ux[j][i]) / (dx*dx);
                float lap_v = (uy[j][i-1] + uy[j][i+1] + uy[j-1][i] + uy[j+1][i]
                             - 4.0f*uy[j][i]) / (dx*dx);

                // Stress divergence
                float dtxx_dx = (txx[j][i+1] - txx[j][i-1]) / (2.0f*dx);
                float dtxy_dx = (txy[j][i+1] - txy[j][i-1]) / (2.0f*dx);
                float dtxy_dy = (txy[j+1][i] - txy[j-1][i]) / (2.0f*dx);
                float dtyy_dy = (tyy[j+1][i] - tyy[j-1][i]) / (2.0f*dx);

                // Momentum
                float advU = ux[j][i]*dudx + uy[j][i]*dudy;
                float advV = ux[j][i]*dvdx + uy[j][i]*dvdy;

                ux_new[j][i] = ux[j][i] + dt * (-advU + ga.eta_s*lap_u + dtxx_dx + dtxy_dy);
                uy_new[j][i] = uy[j][i] + dt * (-advV + ga.eta_s*lap_v + dtxy_dx + dtyy_dy);

                // Oldroyd-B constitutive equation
                float inv_lam = 1.0f / ga.lambda_;
                // Upper-convected derivative
                float adv_txx = ux[j][i]*(txx[j][i+1]-txx[j][i-1])/(2.0f*dx)
                              + uy[j][i]*(txx[j+1][i]-txx[j-1][i])/(2.0f*dx);
                float stretch_xx = 2.0f*(txx[j][i]*dudx + txy[j][i]*dudy);
                txx_new[j][i] = txx[j][i] + dt * (-adv_txx + stretch_xx
                              - inv_lam*txx[j][i] + 2.0f*ga.eta_p*inv_lam*dudx);

                float adv_txy = ux[j][i]*(txy[j][i+1]-txy[j][i-1])/(2.0f*dx)
                              + uy[j][i]*(txy[j+1][i]-txy[j-1][i])/(2.0f*dx);
                float stretch_xy = txx[j][i]*dvdx + tyy[j][i]*dudy;
                txy_new[j][i] = txy[j][i] + dt * (-adv_txy + stretch_xy
                              - inv_lam*txy[j][i] + ga.eta_p*inv_lam*(dudy+dvdx));

                float adv_tyy = ux[j][i]*(tyy[j][i+1]-tyy[j][i-1])/(2.0f*dx)
                              + uy[j][i]*(tyy[j+1][i]-tyy[j-1][i])/(2.0f*dx);
                float stretch_yy = 2.0f*(txy[j][i]*dvdx + tyy[j][i]*dvdy);
                tyy_new[j][i] = tyy[j][i] + dt * (-adv_tyy + stretch_yy
                              - inv_lam*tyy[j][i] + 2.0f*ga.eta_p*inv_lam*dvdy);
            }
        }

        // Boundary: no-slip top/bottom, periodic left/right
        for (int i = 0; i < N2D; ++i) {
            ux_new[0][i] = 0; ux_new[N2D-1][i] = 1.0f;
            uy_new[0][i] = 0; uy_new[N2D-1][i] = 0;
        }
        for (int j = 0; j < N2D; ++j) {
            ux_new[j][0] = ux_new[j][N2D-2]; ux_new[j][N2D-1] = ux_new[j][1];
            uy_new[j][0] = uy_new[j][N2D-2]; uy_new[j][N2D-1] = uy_new[j][1];
        }

        memcpy(ux, ux_new, sizeof(ux)); memcpy(uy, uy_new, sizeof(uy));
        memcpy(txx, txx_new, sizeof(txx));
        memcpy(txy, txy_new, sizeof(txy));
        memcpy(tyy, tyy_new, sizeof(tyy));
        tick++;

        // Fitness: stress magnitude (higher = more instability)
        float stress_energy = 0;
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i)
                stress_energy += txx[j][i]*txx[j][i] + 2.0f*txy[j][i]*txy[j][i] + tyy[j][i]*tyy[j][i];
        engine_ga.fitness = (int)(stress_energy / (float)(N2D*N2D) * 100.0f);
    }

    void write_field(int field, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j * N2D + i;
                switch (field) {
                    case 0: out[idx] = ux[j][i]; break;
                    case 1: out[idx] = uy[j][i]; break;
                    case 2: out[idx] = txx[j][i]; break;
                    case 3: out[idx] = txy[j][i]; break;
                    default: out[idx] = tyy[j][i]; break;
                }
            }
    }

    void ga_mutate(Rng& r) override {
        PDE2D::ga_mutate(r);
        ga.eta_s   *= 1.0f + r.normal(0, 0.05f);
        ga.eta_p   *= 1.0f + r.normal(0, 0.05f);
        ga.lambda_ *= 1.0f + r.normal(0, 0.05f);
        if (ga.eta_s < 0.001f) ga.eta_s = 0.001f;
        if (ga.lambda_ < 0.01f) ga.lambda_ = 0.01f;
    }
};

} // namespace well
