// ============================================================================
// RT_SHEAR.HPP — Rayleigh-Taylor Instability + Shear Flow (Kelvin-Helmholtz)
//
// RT: density-stratified fluid under gravity → mixing instability
//   ∂ρ/∂t + ∇·(ρu) = 0
//   ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + ρg
//   Atwood number A = (ρ₂-ρ₁)/(ρ₂+ρ₁) → GA-tunable
//
// Shear Flow (Kelvin-Helmholtz):
//   Same equations but initial condition = velocity shear layer
//   u₁ = +U, u₂ = -U with smooth transition
//
// GA-driven: Atwood, viscosity, gravity, shear velocity
// ============================================================================
#pragma once
#include "pde_base.hpp"

namespace well {

struct RayleighTaylor2D : PDE2D {
    float rho[N2D][N2D] = {};
    float ux[N2D][N2D]  = {};
    float uy[N2D][N2D]  = {};
    float press[N2D][N2D] = {};

    struct RTGA {
        float rho_heavy = 3.0f;
        float rho_light = 1.0f;
        float gravity   = 1.0f;
        float viscosity = 0.005f;
        float perturb_amp = 0.05f;
    } ga;

    const char* name() const override { return "RayleighTaylor-2D"; }
    int num_fields() const override { return 3; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        float mid = 0.5f;
        for (int j = 0; j < N2D; ++j) {
            float y = (float)j / (float)N2D;
            for (int i = 0; i < N2D; ++i) {
                float x = (float)i / (float)N2D;
                // Interface: heavy on top, light on bottom + perturbation
                float interface = mid + ga.perturb_amp *
                    (cosf(2.0f*3.14159f*x) + 0.5f*cosf(4.0f*3.14159f*x) +
                     0.25f*cosf(6.0f*3.14159f*x));
                float smooth = 0.5f * (1.0f + tanhf((y - interface) * 40.0f));
                rho[j][i] = ga.rho_light + (ga.rho_heavy - ga.rho_light) * smooth;
                ux[j][i] = rng.normal(0, 0.001f);
                uy[j][i] = rng.normal(0, 0.001f);
                press[j][i] = ga.rho_heavy * ga.gravity * (1.0f - y);  // hydrostatic
            }
        }
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float dt = dt_ > 0 ? dt_ : 0.25f * dx * dx / (ga.viscosity + 1e-8f);
        dt = fminf(dt, 0.001f);

        float rho_new[N2D][N2D], ux_new[N2D][N2D], uy_new[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j) {
            for (int i = 1; i < N2D-1; ++i) {
                float r = rho[j][i];
                float inv_r = 1.0f / (r + 1e-12f);

                // Advection (upwind)
                float u = ux[j][i], v = uy[j][i];
                float dudx = (ux[j][i+1] - ux[j][i-1]) / (2.0f*dx);
                float dudy = (ux[j+1][i] - ux[j-1][i]) / (2.0f*dx);
                float dvdx = (uy[j][i+1] - uy[j][i-1]) / (2.0f*dx);
                float dvdy = (uy[j+1][i] - uy[j-1][i]) / (2.0f*dx);

                // Laplacians
                float lap_u = (ux[j][i-1]+ux[j][i+1]+ux[j-1][i]+ux[j+1][i]-4.0f*ux[j][i])/(dx*dx);
                float lap_v = (uy[j][i-1]+uy[j][i+1]+uy[j-1][i]+uy[j+1][i]-4.0f*uy[j][i])/(dx*dx);

                // Pressure gradient (approximate)
                float dpdx = (press[j][i+1]-press[j][i-1])/(2.0f*dx);
                float dpdy = (press[j+1][i]-press[j-1][i])/(2.0f*dx);

                // Momentum
                ux_new[j][i] = u + dt * (-u*dudx - v*dudy - inv_r*dpdx + ga.viscosity*lap_u);
                uy_new[j][i] = v + dt * (-u*dvdx - v*dvdy - inv_r*dpdy + ga.viscosity*lap_v - ga.gravity);

                // Density transport
                float drhodx = (rho[j][i+1]-rho[j][i-1])/(2.0f*dx);
                float drhody = (rho[j+1][i]-rho[j-1][i])/(2.0f*dx);
                rho_new[j][i] = r + dt * (-u*drhodx - v*drhody - r*(dudx + dvdy));
                if (rho_new[j][i] < 0.1f) rho_new[j][i] = 0.1f;
            }
        }

        // BC: walls top/bottom, periodic sides
        for (int i = 0; i < N2D; ++i) {
            ux_new[0][i] = 0; ux_new[N2D-1][i] = 0;
            uy_new[0][i] = 0; uy_new[N2D-1][i] = 0;
            rho_new[0][i] = ga.rho_light; rho_new[N2D-1][i] = ga.rho_heavy;
        }
        for (int j = 0; j < N2D; ++j) {
            ux_new[j][0] = ux_new[j][N2D-2]; ux_new[j][N2D-1] = ux_new[j][1];
            uy_new[j][0] = uy_new[j][N2D-2]; uy_new[j][N2D-1] = uy_new[j][1];
            rho_new[j][0] = rho_new[j][N2D-2]; rho_new[j][N2D-1] = rho_new[j][1];
        }

        memcpy(rho, rho_new, sizeof(rho));
        memcpy(ux, ux_new, sizeof(ux));
        memcpy(uy, uy_new, sizeof(uy));
        tick++;

        // Fitness: mixing layer thickness
        float mixing = 0;
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                float f = (rho[j][i] - ga.rho_light) / (ga.rho_heavy - ga.rho_light);
                mixing += f * (1.0f - f);
            }
        engine_ga.fitness = (int)(mixing);
    }

    void write_field(int f, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j * N2D + i;
                if (f == 0) out[idx] = rho[j][i];
                else if (f == 1) out[idx] = ux[j][i];
                else out[idx] = uy[j][i];
            }
    }

    void ga_mutate(Rng& r) override {
        PDE2D::ga_mutate(r);
        ga.gravity *= 1.0f + r.normal(0, 0.05f);
        ga.viscosity *= 1.0f + r.normal(0, 0.05f);
        ga.perturb_amp *= 1.0f + r.normal(0, 0.05f);
        if (ga.viscosity < 0.001f) ga.viscosity = 0.001f;
    }
};

// ── Shear Flow (Kelvin-Helmholtz) ──
struct ShearFlow2D : PDE2D {
    float rho[N2D][N2D] = {};
    float ux[N2D][N2D]  = {};
    float uy[N2D][N2D]  = {};
    float tracer[N2D][N2D] = {};  // passive scalar

    struct ShearGA {
        float U_shear = 1.0f;     // shear velocity
        float delta = 0.05f;      // shear layer thickness
        float viscosity = 0.002f;
        float rho_ratio = 2.0f;   // density ratio
        float perturb = 0.01f;
    } ga;

    const char* name() const override { return "ShearFlow-2D"; }
    int num_fields() const override { return 4; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        for (int j = 0; j < N2D; ++j) {
            float y = (float)j / (float)N2D;
            for (int i = 0; i < N2D; ++i) {
                float x = (float)i / (float)N2D;
                // Two shear layers
                float f1 = tanhf((y - 0.25f) / ga.delta);
                float f2 = tanhf((0.75f - y) / ga.delta);
                ux[j][i] = ga.U_shear * 0.5f * (f1 + f2);
                uy[j][i] = ga.perturb * sinf(4.0f * 3.14159f * x) *
                           (expf(-((y-0.25f)*(y-0.25f))/(ga.delta*ga.delta)) +
                            expf(-((y-0.75f)*(y-0.75f))/(ga.delta*ga.delta)));
                rho[j][i] = 1.0f + (ga.rho_ratio - 1.0f) * 0.5f * (1.0f + f1);
                tracer[j][i] = y < 0.5f ? 0.0f : 1.0f;
            }
        }
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float dt = dt_ > 0 ? dt_ : 0.2f*dx;

        float rho_n[N2D][N2D], ux_n[N2D][N2D], uy_n[N2D][N2D], tr_n[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j)
            for (int i = 1; i < N2D-1; ++i) {
                float u = ux[j][i], v = uy[j][i], r = rho[j][i];

                // Central differences
                float dudx = (ux[j][i+1]-ux[j][i-1])/(2.0f*dx);
                float dudy = (ux[j+1][i]-ux[j-1][i])/(2.0f*dx);
                float dvdx = (uy[j][i+1]-uy[j][i-1])/(2.0f*dx);
                float dvdy = (uy[j+1][i]-uy[j-1][i])/(2.0f*dx);
                float lap_u = (ux[j][i-1]+ux[j][i+1]+ux[j-1][i]+ux[j+1][i]-4.0f*u)/(dx*dx);
                float lap_v = (uy[j][i-1]+uy[j][i+1]+uy[j-1][i]+uy[j+1][i]-4.0f*v)/(dx*dx);

                ux_n[j][i] = u + dt*(-u*dudx - v*dudy + ga.viscosity*lap_u);
                uy_n[j][i] = v + dt*(-u*dvdx - v*dvdy + ga.viscosity*lap_v);

                // Density
                float drhodx = (rho[j][i+1]-rho[j][i-1])/(2.0f*dx);
                float drhody = (rho[j+1][i]-rho[j-1][i])/(2.0f*dx);
                rho_n[j][i] = r + dt*(-u*drhodx - v*drhody - r*(dudx+dvdy));

                // Tracer
                float dtrdx = (tracer[j][i+1]-tracer[j][i-1])/(2.0f*dx);
                float dtrdy = (tracer[j+1][i]-tracer[j-1][i])/(2.0f*dx);
                tr_n[j][i] = tracer[j][i] + dt*(-u*dtrdx - v*dtrdy);
            }

        // Periodic BC
        for (int j = 0; j < N2D; ++j) {
            ux_n[j][0]=ux_n[j][N2D-2]; ux_n[j][N2D-1]=ux_n[j][1];
            uy_n[j][0]=uy_n[j][N2D-2]; uy_n[j][N2D-1]=uy_n[j][1];
            rho_n[j][0]=rho_n[j][N2D-2]; rho_n[j][N2D-1]=rho_n[j][1];
            tr_n[j][0]=tr_n[j][N2D-2]; tr_n[j][N2D-1]=tr_n[j][1];
        }
        for (int i = 0; i < N2D; ++i) {
            ux_n[0][i]=ux_n[N2D-2][i]; ux_n[N2D-1][i]=ux_n[1][i];
            uy_n[0][i]=uy_n[N2D-2][i]; uy_n[N2D-1][i]=uy_n[1][i];
            rho_n[0][i]=rho_n[N2D-2][i]; rho_n[N2D-1][i]=rho_n[1][i];
            tr_n[0][i]=tr_n[N2D-2][i]; tr_n[N2D-1][i]=tr_n[1][i];
        }

        memcpy(rho, rho_n, sizeof(rho)); memcpy(ux, ux_n, sizeof(ux));
        memcpy(uy, uy_n, sizeof(uy)); memcpy(tracer, tr_n, sizeof(tracer));
        tick++;

        // Fitness: vorticity magnitude
        float vort = 0;
        for (int j = 1; j < N2D-1; ++j)
            for (int i = 1; i < N2D-1; ++i) {
                float w = (uy[j][i+1]-uy[j][i-1])/(2.0f*dx)
                        - (ux[j+1][i]-ux[j-1][i])/(2.0f*dx);
                vort += w * w;
            }
        engine_ga.fitness = (int)(sqrtf(vort));
    }

    void write_field(int f, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j * N2D + i;
                switch (f) {
                    case 0: out[idx] = rho[j][i]; break;
                    case 1: out[idx] = ux[j][i]; break;
                    case 2: out[idx] = uy[j][i]; break;
                    default: out[idx] = tracer[j][i]; break;
                }
            }
    }

    void ga_mutate(Rng& r) override {
        PDE2D::ga_mutate(r);
        ga.U_shear *= 1.0f + r.normal(0, 0.05f);
        ga.viscosity *= 1.0f + r.normal(0, 0.05f);
    }
};

} // namespace well
