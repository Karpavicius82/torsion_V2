// ============================================================================
// SUPERNOVA_TURB.HPP â€” Supernova Explosion + Turbulence with Gravity Cooling
//
// Supernova (Sedov-Taylor blast wave):
//   Euler equations with point-source energy injection
//   âˆ‚Ï/âˆ‚t + âˆ‡Â·(Ïu) = 0
//   âˆ‚(Ïu)/âˆ‚t + âˆ‡Â·(ÏuâŠ—u + pI) = 0
//   âˆ‚E/âˆ‚t + âˆ‡Â·((E+p)u) = 0
//   p = (Î³-1)(E - 0.5Ï|u|Â²)
//
// Turbulence + gravity + cooling:
//   Same compressible Euler + gravitational source + radiative cooling Î›(T)
//   Forcing: stochastic solenoidal driving at large scales
//
// GA: blast energy, Î³, cooling rate, forcing amplitude
// ============================================================================
#pragma once
#include "pde_base.hpp"

namespace well {

struct Supernova2D : PDE2D {
    float rho[N2D][N2D] = {};
    float mx[N2D][N2D]  = {};   // momentum x
    float my[N2D][N2D]  = {};   // momentum y
    float E[N2D][N2D]   = {};   // total energy

    struct SNGA {
        float gamma = 5.0f/3.0f;
        float blast_energy = 10.0f;
        float blast_radius = 0.05f;
        float ambient_rho = 1.0f;
        float ambient_p = 0.01f;
    } ga;

    const char* name() const override { return "Supernova-2D"; }
    int num_fields() const override { return 4; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        float cx = 0.5f, cy = 0.5f;
        for (int j = 0; j < N2D; ++j) {
            float y = (float)j / (float)N2D;
            for (int i = 0; i < N2D; ++i) {
                float x = (float)i / (float)N2D;
                rho[j][i] = ga.ambient_rho;
                mx[j][i] = 0; my[j][i] = 0;

                float r = sqrtf((x-cx)*(x-cx) + (y-cy)*(y-cy));
                if (r < ga.blast_radius) {
                    // Point explosion: high energy in center
                    float e_blast = ga.blast_energy / (3.14159f * ga.blast_radius * ga.blast_radius);
                    E[j][i] = e_blast;
                } else {
                    E[j][i] = ga.ambient_p / (ga.gamma - 1.0f);
                }
            }
        }
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float gm = ga.gamma;
        float gm1 = gm - 1.0f;

        // Compute maximum wave speed for CFL
        float max_cs = 0;
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                float r = rho[j][i]; if (r < 1e-8f) r = 1e-8f;
                float ke = 0.5f * (mx[j][i]*mx[j][i] + my[j][i]*my[j][i]) / r;
                float p = gm1 * (E[j][i] - ke);
                if (p < 0) p = 1e-10f;
                float cs = sqrtf(gm * p / r) + sqrtf(mx[j][i]*mx[j][i]+my[j][i]*my[j][i])/r;
                if (cs > max_cs) max_cs = cs;
            }
        float dt = dt_ > 0 ? dt_ : 0.3f * dx / (max_cs + 1e-10f);
        dt = fminf(dt, 0.01f);

        float rho_n[N2D][N2D], mx_n[N2D][N2D], my_n[N2D][N2D], E_n[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j) {
            for (int i = 1; i < N2D-1; ++i) {
                float r = rho[j][i]; if (r < 1e-8f) r = 1e-8f;
                float u = mx[j][i] / r;
                float v = my[j][i] / r;
                float ke = 0.5f * r * (u*u + v*v);
                float p = gm1 * (E[j][i] - ke);
                if (p < 0) p = 1e-10f;

                // Flux differences (Lax-Friedrichs-like)
                auto pres = [&](int jj, int ii) -> float {
                    float rr = rho[jj][ii]; if (rr < 1e-8f) rr = 1e-8f;
                    float kk = 0.5f*(mx[jj][ii]*mx[jj][ii]+my[jj][ii]*my[jj][ii])/rr;
                    float pp = gm1*(E[jj][ii]-kk); return pp > 0 ? pp : 1e-10f;
                };

                // x-direction fluxes
                float fr_l = mx[j][i-1], fr_r = mx[j][i+1];
                float fmx_l = mx[j][i-1]*mx[j][i-1]/fmaxf(rho[j][i-1],1e-8f) + pres(j,i-1);
                float fmx_r = mx[j][i+1]*mx[j][i+1]/fmaxf(rho[j][i+1],1e-8f) + pres(j,i+1);

                // Simplified: central difference + artificial viscosity
                float d_rho_x = (mx[j][i+1] - mx[j][i-1]) / (2.0f*dx);
                float d_rho_y = (my[j+1][i] - my[j-1][i]) / (2.0f*dx);

                float d_mx_x = (fmx_r - fmx_l) / (2.0f*dx);
                float fmy_b = mx[j-1][i]*my[j-1][i]/fmaxf(rho[j-1][i],1e-8f);
                float fmy_t = mx[j+1][i]*my[j+1][i]/fmaxf(rho[j+1][i],1e-8f);
                float d_mx_y = (fmy_t - fmy_b) / (2.0f*dx);

                float d_my_x = (mx[j][i+1]*my[j][i+1]/fmaxf(rho[j][i+1],1e-8f) -
                                mx[j][i-1]*my[j][i-1]/fmaxf(rho[j][i-1],1e-8f)) / (2.0f*dx);
                float fvy_b = my[j-1][i]*my[j-1][i]/fmaxf(rho[j-1][i],1e-8f)+pres(j-1,i);
                float fvy_t = my[j+1][i]*my[j+1][i]/fmaxf(rho[j+1][i],1e-8f)+pres(j+1,i);
                float d_my_y = (fvy_t - fvy_b) / (2.0f*dx);

                float d_E_x = ((E[j][i+1]+pres(j,i+1))*mx[j][i+1]/fmaxf(rho[j][i+1],1e-8f) -
                               (E[j][i-1]+pres(j,i-1))*mx[j][i-1]/fmaxf(rho[j][i-1],1e-8f)) / (2.0f*dx);
                float d_E_y = ((E[j+1][i]+pres(j+1,i))*my[j+1][i]/fmaxf(rho[j+1][i],1e-8f) -
                               (E[j-1][i]+pres(j-1,i))*my[j-1][i]/fmaxf(rho[j-1][i],1e-8f)) / (2.0f*dx);

                // Artificial viscosity
                float nu_art = 0.1f * dx * max_cs;
                float lap_r = (rho[j][i-1]+rho[j][i+1]+rho[j-1][i]+rho[j+1][i]-4.0f*rho[j][i])/(dx*dx);
                float lap_mx = (mx[j][i-1]+mx[j][i+1]+mx[j-1][i]+mx[j+1][i]-4.0f*mx[j][i])/(dx*dx);
                float lap_my = (my[j][i-1]+my[j][i+1]+my[j-1][i]+my[j+1][i]-4.0f*my[j][i])/(dx*dx);
                float lap_E = (E[j][i-1]+E[j][i+1]+E[j-1][i]+E[j+1][i]-4.0f*E[j][i])/(dx*dx);

                rho_n[j][i] = rho[j][i] + dt * (-d_rho_x - d_rho_y + nu_art*lap_r);
                mx_n[j][i]  = mx[j][i]  + dt * (-d_mx_x - d_mx_y + nu_art*lap_mx);
                my_n[j][i]  = my[j][i]  + dt * (-d_my_x - d_my_y + nu_art*lap_my);
                E_n[j][i]   = E[j][i]   + dt * (-d_E_x - d_E_y + nu_art*lap_E);

                if (rho_n[j][i] < 1e-6f) rho_n[j][i] = 1e-6f;
                if (E_n[j][i] < 1e-10f) E_n[j][i] = 1e-10f;
            }
        }

        // Outflow BC
        for (int k = 0; k < N2D; ++k) {
            rho_n[0][k]=rho_n[1][k]; rho_n[N2D-1][k]=rho_n[N2D-2][k];
            rho_n[k][0]=rho_n[k][1]; rho_n[k][N2D-1]=rho_n[k][N2D-2];
            mx_n[0][k]=mx_n[1][k]; mx_n[N2D-1][k]=mx_n[N2D-2][k];
            mx_n[k][0]=mx_n[k][1]; mx_n[k][N2D-1]=mx_n[k][N2D-2];
            my_n[0][k]=my_n[1][k]; my_n[N2D-1][k]=my_n[N2D-2][k];
            my_n[k][0]=my_n[k][1]; my_n[k][N2D-1]=my_n[k][N2D-2];
            E_n[0][k]=E_n[1][k]; E_n[N2D-1][k]=E_n[N2D-2][k];
            E_n[k][0]=E_n[k][1]; E_n[k][N2D-1]=E_n[k][N2D-2];
        }

        memcpy(rho, rho_n, sizeof(rho)); memcpy(mx, mx_n, sizeof(mx));
        memcpy(my, my_n, sizeof(my)); memcpy(E, E_n, sizeof(E));
        tick++;

        // Fitness: shock front radius
        float max_grad = 0;
        for (int j = 1; j < N2D-1; ++j)
            for (int i = 1; i < N2D-1; ++i) {
                float gx = rho[j][i+1]-rho[j][i-1];
                float gy = rho[j+1][i]-rho[j-1][i];
                float g = gx*gx + gy*gy;
                if (g > max_grad) max_grad = g;
            }
        engine_ga.fitness = (int)(sqrtf(max_grad) * 1000.0f);
    }

    void write_field(int f, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j*N2D+i;
                switch(f) {
                    case 0: out[idx]=rho[j][i]; break;
                    case 1: out[idx]=mx[j][i]; break;
                    case 2: out[idx]=my[j][i]; break;
                    default: out[idx]=E[j][i]; break;
                }
            }
    }
    void ga_mutate(Rng& r) override { PDE2D::ga_mutate(r);
        ga.blast_energy *= 1.0f+r.normal(0,0.05f);
        ga.gamma = 1.0f + fmaxf(0.01f, (ga.gamma-1.0f)*(1.0f+r.normal(0,0.02f)));
    }
};

// â”€â”€ Turbulence with Gravity + Cooling â”€â”€
struct TurbulenceGravityCooling2D : PDE2D {
    float rho[N2D][N2D] = {};
    float ux[N2D][N2D]  = {};
    float uy[N2D][N2D]  = {};
    float temp[N2D][N2D] = {};  // temperature

    struct TurbGA {
        float viscosity  = 0.005f;
        float gravity    = 1.0f;
        float cooling    = 0.1f;
        float forcing    = 0.5f;
        float T_floor    = 0.1f;
    } ga;

    const char* name() const override { return "TurbulenceGravityCooling-2D"; }
    int num_fields() const override { return 4; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        for (int j = 0; j < N2D; ++j) {
            float y = (float)j / (float)N2D;
            for (int i = 0; i < N2D; ++i) {
                rho[j][i] = 1.0f;
                ux[j][i] = rng.normal(0, 0.1f);
                uy[j][i] = rng.normal(0, 0.1f);
                temp[j][i] = 1.0f - 0.5f*y;  // temperature gradient
            }
        }
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float dt = dt_ > 0 ? dt_ : 0.25f*dx*dx/(ga.viscosity+1e-8f);
        dt = fminf(dt, 0.005f);

        float rho_n[N2D][N2D], ux_n[N2D][N2D], uy_n[N2D][N2D], T_n[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j)
            for (int i = 1; i < N2D-1; ++i) {
                float u = ux[j][i], v = uy[j][i], r = rho[j][i], T = temp[j][i];

                float dudx = (ux[j][i+1]-ux[j][i-1])/(2.0f*dx);
                float dudy = (ux[j+1][i]-ux[j-1][i])/(2.0f*dx);
                float dvdx = (uy[j][i+1]-uy[j][i-1])/(2.0f*dx);
                float dvdy = (uy[j+1][i]-uy[j-1][i])/(2.0f*dx);
                float dTdx = (temp[j][i+1]-temp[j][i-1])/(2.0f*dx);
                float dTdy = (temp[j+1][i]-temp[j-1][i])/(2.0f*dx);

                float lap_u = (ux[j][i-1]+ux[j][i+1]+ux[j-1][i]+ux[j+1][i]-4.0f*u)/(dx*dx);
                float lap_v = (uy[j][i-1]+uy[j][i+1]+uy[j-1][i]+uy[j+1][i]-4.0f*v)/(dx*dx);
                float lap_T = (temp[j][i-1]+temp[j][i+1]+temp[j-1][i]+temp[j+1][i]-4.0f*T)/(dx*dx);
                float lap_r = (rho[j][i-1]+rho[j][i+1]+rho[j-1][i]+rho[j+1][i]-4.0f*r)/(dx*dx);

                // Stochastic forcing (large scale)
                float fx = ga.forcing * sinf(2.0f*3.14159f*((float)j/(float)N2D)) * rng.normal(0,1.0f);
                float fy = ga.forcing * cosf(2.0f*3.14159f*((float)i/(float)N2D)) * rng.normal(0,1.0f);

                // Cooling function: Î›(T) = cooling * TÂ²
                float cooling_rate = ga.cooling * T * T;

                rho_n[j][i] = r + dt*(-u*(rho[j][i+1]-rho[j][i-1])/(2.0f*dx)
                             -v*(rho[j+1][i]-rho[j-1][i])/(2.0f*dx) - r*(dudx+dvdy));
                ux_n[j][i] = u + dt*(-u*dudx-v*dudy + ga.viscosity*lap_u + fx);
                uy_n[j][i] = v + dt*(-u*dvdx-v*dvdy + ga.viscosity*lap_v - ga.gravity + fy);
                T_n[j][i]  = T + dt*(-u*dTdx-v*dTdy + ga.viscosity*lap_T - cooling_rate);

                if (rho_n[j][i] < 0.01f) rho_n[j][i] = 0.01f;
                if (T_n[j][i] < ga.T_floor) T_n[j][i] = ga.T_floor;
            }

        // Periodic BC
        for (int j = 0; j < N2D; ++j) {
            rho_n[j][0]=rho_n[j][N2D-2]; rho_n[j][N2D-1]=rho_n[j][1];
            ux_n[j][0]=ux_n[j][N2D-2]; ux_n[j][N2D-1]=ux_n[j][1];
            uy_n[j][0]=uy_n[j][N2D-2]; uy_n[j][N2D-1]=uy_n[j][1];
            T_n[j][0]=T_n[j][N2D-2]; T_n[j][N2D-1]=T_n[j][1];
        }
        for (int i = 0; i < N2D; ++i) {
            rho_n[0][i]=rho_n[N2D-2][i]; rho_n[N2D-1][i]=rho_n[1][i];
            ux_n[0][i]=ux_n[N2D-2][i]; ux_n[N2D-1][i]=ux_n[1][i];
            uy_n[0][i]=uy_n[N2D-2][i]; uy_n[N2D-1][i]=uy_n[1][i];
            T_n[0][i]=T_n[N2D-2][i]; T_n[N2D-1][i]=T_n[1][i];
        }

        memcpy(rho, rho_n, sizeof(rho)); memcpy(ux, ux_n, sizeof(ux));
        memcpy(uy, uy_n, sizeof(uy)); memcpy(temp, T_n, sizeof(temp));
        tick++;

        // Fitness: kinetic energy
        float ke = 0;
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i)
                ke += 0.5f*rho[j][i]*(ux[j][i]*ux[j][i]+uy[j][i]*uy[j][i]);
        engine_ga.fitness = (int)(ke);
    }

    void write_field(int f, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j*N2D+i;
                switch(f) { case 0: out[idx]=rho[j][i]; break;
                    case 1: out[idx]=ux[j][i]; break;
                    case 2: out[idx]=uy[j][i]; break;
                    default: out[idx]=temp[j][i]; break; }
            }
    }
    void ga_mutate(Rng& r) override { PDE2D::ga_mutate(r);
        ga.forcing *= 1.0f+r.normal(0,0.05f); ga.cooling *= 1.0f+r.normal(0,0.05f);
    }
};

} // namespace well
