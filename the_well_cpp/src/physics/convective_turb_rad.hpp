// ============================================================================
// CONVECTIVE_TURB_RAD.HPP â€” Convective Envelope + Turbulent Radiative Layer
//
// Convective envelope (stellar interior, anelastic approx):
//   âˆ‡Â·(Ïâ‚€ u) = 0
//   Ïâ‚€(âˆ‚u/âˆ‚t + uÂ·âˆ‡u) = -âˆ‡p' + Ï'g + Î¼âˆ‡Â²u
//   âˆ‚T'/âˆ‚t + uÂ·âˆ‡(Tâ‚€+T') = Îºâˆ‡Â²T' + Q
//   Ï' = -Ïâ‚€Î±T'
//
// Turbulent radiative layer (radiation hydro, flux-limited diffusion):
//   âˆ‚E_r/âˆ‚t = âˆ‡Â·(D_râˆ‡E_r) - Îº_abs Ï c E_r + Îº_abs Ï c aTâ´
//   + full hydro equations with radiative source
//
// GA: Rayleigh number, Prandtl, opacity, radiative diffusion
// ============================================================================
#pragma once
#include "pde_base.hpp"

namespace well {

struct ConvectiveEnvelope2D : PDE2D {
    float ux[N2D][N2D] = {};
    float uy[N2D][N2D] = {};
    float T[N2D][N2D]  = {};     // temperature perturbation
    float psi[N2D][N2D] = {};    // stream function

    struct ConvGA {
        float Ra = 1e4f;    // Rayleigh number
        float Pr = 0.7f;    // Prandtl number
        float T_top = 0.0f;
        float T_bot = 1.0f;
        float alpha = 1.0f; // thermal expansion
    } ga;

    const char* name() const override { return "ConvectiveEnvelope-2D"; }
    int num_fields() const override { return 3; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        for (int j = 0; j < N2D; ++j) {
            float y = (float)j / (float)N2D;
            for (int i = 0; i < N2D; ++i) {
                T[j][i] = ga.T_bot - (ga.T_bot - ga.T_top) * y
                         + rng.normal(0, 0.01f);
                ux[j][i] = rng.normal(0, 0.001f);
                uy[j][i] = rng.normal(0, 0.001f);
            }
        }
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float nu = ga.Pr;
        float kappa = 1.0f;
        float dt = dt_ > 0 ? dt_ : 0.25f * dx * dx / fmaxf(nu, kappa);
        dt = fminf(dt, 0.005f);

        float ux_n[N2D][N2D], uy_n[N2D][N2D], T_n[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j)
            for (int i = 1; i < N2D-1; ++i) {
                float u = ux[j][i], v = uy[j][i], Tv = T[j][i];

                float dudx = (ux[j][i+1]-ux[j][i-1])/(2.0f*dx);
                float dudy = (ux[j+1][i]-ux[j-1][i])/(2.0f*dx);
                float dvdx = (uy[j][i+1]-uy[j][i-1])/(2.0f*dx);
                float dvdy = (uy[j+1][i]-uy[j-1][i])/(2.0f*dx);
                float dTdx = (T[j][i+1]-T[j][i-1])/(2.0f*dx);
                float dTdy = (T[j+1][i]-T[j-1][i])/(2.0f*dx);

                float lap_u = (ux[j][i-1]+ux[j][i+1]+ux[j-1][i]+ux[j+1][i]-4.0f*u)/(dx*dx);
                float lap_v = (uy[j][i-1]+uy[j][i+1]+uy[j-1][i]+uy[j+1][i]-4.0f*v)/(dx*dx);
                float lap_T = (T[j][i-1]+T[j][i+1]+T[j-1][i]+T[j+1][i]-4.0f*Tv)/(dx*dx);

                // Boussinesq: buoyancy = Ra * Pr * T in y-direction
                float buoyancy = ga.Ra * ga.Pr * Tv;

                ux_n[j][i] = u + dt*(-u*dudx - v*dudy + nu*lap_u);
                uy_n[j][i] = v + dt*(-u*dvdx - v*dvdy + nu*lap_v + buoyancy);
                T_n[j][i]  = Tv + dt*(-u*dTdx - v*dTdy + kappa*lap_T);
            }

        // BC: no-slip top/bottom, periodic sides
        for (int i = 0; i < N2D; ++i) {
            ux_n[0][i] = 0; ux_n[N2D-1][i] = 0;
            uy_n[0][i] = 0; uy_n[N2D-1][i] = 0;
            T_n[0][i] = ga.T_bot; T_n[N2D-1][i] = ga.T_top;
        }
        for (int j = 0; j < N2D; ++j) {
            ux_n[j][0]=ux_n[j][N2D-2]; ux_n[j][N2D-1]=ux_n[j][1];
            uy_n[j][0]=uy_n[j][N2D-2]; uy_n[j][N2D-1]=uy_n[j][1];
            T_n[j][0]=T_n[j][N2D-2]; T_n[j][N2D-1]=T_n[j][1];
        }

        memcpy(ux, ux_n, sizeof(ux)); memcpy(uy, uy_n, sizeof(uy));
        memcpy(T, T_n, sizeof(T));
        tick++;

        // Fitness: Nusselt number proxy (vertical heat flux)
        float flux = 0;
        int jm = N2D / 2;
        for (int i = 0; i < N2D; ++i)
            flux += uy[jm][i] * T[jm][i];
        engine_ga.fitness = (int)(fabsf(flux) * 100.0f);
    }

    void write_field(int f, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j*N2D+i;
                if (f==0) out[idx]=ux[j][i];
                else if (f==1) out[idx]=uy[j][i];
                else out[idx]=T[j][i];
            }
    }
    void ga_mutate(Rng& r) override { PDE2D::ga_mutate(r);
        ga.Ra *= 1.0f+r.normal(0,0.1f); ga.Pr *= 1.0f+r.normal(0,0.05f);
        if (ga.Ra < 100) ga.Ra = 100; if (ga.Pr < 0.01f) ga.Pr = 0.01f;
    }
};

// â”€â”€ Turbulent Radiative Layer â”€â”€
struct TurbulentRadiativeLayer2D : PDE2D {
    float rho[N2D][N2D] = {};
    float ux[N2D][N2D]  = {};
    float uy[N2D][N2D]  = {};
    float T[N2D][N2D]   = {};
    float E_rad[N2D][N2D] = {};   // radiation energy density

    struct RadGA {
        float opacity    = 1.0f;
        float rad_diff   = 0.1f;    // radiation diffusion coefficient
        float viscosity  = 0.005f;
        float T_bot = 2.0f;
        float T_top = 0.5f;
    } ga;

    const char* name() const override { return "TurbulentRadiativeLayer-2D"; }
    int num_fields() const override { return 5; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        for (int j = 0; j < N2D; ++j) {
            float y = (float)j / (float)N2D;
            for (int i = 0; i < N2D; ++i) {
                rho[j][i] = 1.0f;
                T[j][i] = ga.T_bot + (ga.T_top - ga.T_bot) * y + rng.normal(0, 0.01f);
                ux[j][i] = rng.normal(0, 0.01f);
                uy[j][i] = rng.normal(0, 0.01f);
                E_rad[j][i] = T[j][i] * T[j][i] * T[j][i] * T[j][i];  // aTâ´
            }
        }
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float dt = dt_ > 0 ? dt_ : 0.1f * dx * dx / (ga.rad_diff + ga.viscosity + 1e-8f);
        dt = fminf(dt, 0.002f);

        float rho_n[N2D][N2D], ux_n[N2D][N2D], uy_n[N2D][N2D];
        float T_n[N2D][N2D], Er_n[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j)
            for (int i = 1; i < N2D-1; ++i) {
                float u = ux[j][i], v = uy[j][i], r = rho[j][i], Tv = T[j][i];
                float Er = E_rad[j][i];

                float dudx = (ux[j][i+1]-ux[j][i-1])/(2.0f*dx);
                float dvdy = (uy[j+1][i]-uy[j-1][i])/(2.0f*dx);
                float lap_u = (ux[j][i-1]+ux[j][i+1]+ux[j-1][i]+ux[j+1][i]-4.0f*u)/(dx*dx);
                float lap_v = (uy[j][i-1]+uy[j][i+1]+uy[j-1][i]+uy[j+1][i]-4.0f*v)/(dx*dx);
                float lap_T = (T[j][i-1]+T[j][i+1]+T[j-1][i]+T[j+1][i]-4.0f*Tv)/(dx*dx);
                float lap_Er = (E_rad[j][i-1]+E_rad[j][i+1]+E_rad[j-1][i]+E_rad[j+1][i]-4.0f*Er)/(dx*dx);

                float dTdx = (T[j][i+1]-T[j][i-1])/(2.0f*dx);
                float dTdy = (T[j+1][i]-T[j-1][i])/(2.0f*dx);
                float dudy = (ux[j+1][i]-ux[j-1][i])/(2.0f*dx);
                float dvdx = (uy[j][i+1]-uy[j][i-1])/(2.0f*dx);

                // Radiation exchange: absorption/emission
                float T4 = Tv*Tv*Tv*Tv;
                float exchange = ga.opacity * r * (T4 - Er);

                // Hydro
                float drhodx = (rho[j][i+1]-rho[j][i-1])/(2.0f*dx);
                float drhody = (rho[j+1][i]-rho[j-1][i])/(2.0f*dx);
                rho_n[j][i] = r + dt*(-u*drhodx - v*drhody - r*(dudx+dvdy));
                ux_n[j][i] = u + dt*(-u*dudx - v*dudy + ga.viscosity*lap_u);
                uy_n[j][i] = v + dt*(-u*dvdx - v*dvdy + ga.viscosity*lap_v);
                T_n[j][i]  = Tv + dt*(-u*dTdx - v*dTdy + ga.viscosity*lap_T - exchange/r);
                Er_n[j][i] = Er + dt*(ga.rad_diff*lap_Er + exchange);

                if (rho_n[j][i] < 0.01f) rho_n[j][i] = 0.01f;
                if (T_n[j][i] < 0.01f) T_n[j][i] = 0.01f;
                if (Er_n[j][i] < 0) Er_n[j][i] = 0;
            }

        // BC
        for (int i = 0; i < N2D; ++i) {
            T_n[0][i] = ga.T_bot; T_n[N2D-1][i] = ga.T_top;
            ux_n[0][i]=0; ux_n[N2D-1][i]=0; uy_n[0][i]=0; uy_n[N2D-1][i]=0;
        }
        for (int j = 0; j < N2D; ++j) {
            ux_n[j][0]=ux_n[j][N2D-2]; ux_n[j][N2D-1]=ux_n[j][1];
            uy_n[j][0]=uy_n[j][N2D-2]; uy_n[j][N2D-1]=uy_n[j][1];
            T_n[j][0]=T_n[j][N2D-2]; T_n[j][N2D-1]=T_n[j][1];
            rho_n[j][0]=rho_n[j][N2D-2]; rho_n[j][N2D-1]=rho_n[j][1];
            Er_n[j][0]=Er_n[j][N2D-2]; Er_n[j][N2D-1]=Er_n[j][1];
        }

        memcpy(rho, rho_n, sizeof(rho)); memcpy(ux, ux_n, sizeof(ux));
        memcpy(uy, uy_n, sizeof(uy)); memcpy(T, T_n, sizeof(T));
        memcpy(E_rad, Er_n, sizeof(E_rad));
        tick++;

        float ke = 0;
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i)
                ke += 0.5f*rho[j][i]*(ux[j][i]*ux[j][i]+uy[j][i]*uy[j][i]);
        engine_ga.fitness = (int)(ke * 100.0f);
    }

    void write_field(int f, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j*N2D+i;
                switch(f) { case 0: out[idx]=rho[j][i]; break;
                    case 1: out[idx]=ux[j][i]; break;
                    case 2: out[idx]=uy[j][i]; break;
                    case 3: out[idx]=T[j][i]; break;
                    default: out[idx]=E_rad[j][i]; break; }
            }
    }
    void ga_mutate(Rng& r) override { PDE2D::ga_mutate(r);
        ga.opacity *= 1.0f+r.normal(0,0.05f); ga.rad_diff *= 1.0f+r.normal(0,0.05f);
    }
};

} // namespace well
