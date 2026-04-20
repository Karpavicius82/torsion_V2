// ============================================================================
// NEUTRON_PLANETSWE.HPP â€” Post Neutron Star Merger + Planetary Shallow Water
//
// Neutron star merger (relativistic MHD, simplified):
//   âˆ‚Ï/âˆ‚t + âˆ‡Â·(Ïu) = 0
//   âˆ‚(Ïu)/âˆ‚t + âˆ‡Â·(ÏuâŠ—u + (p+BÂ²/2)I - BâŠ—B) = -Ïâˆ‡Î¦
//   âˆ‚B/âˆ‚t = âˆ‡Ã—(uÃ—B) + Î·âˆ‡Â²B
//   âˆ‚E/âˆ‚t + ... = neutrino_cooling
//   Lorentz factor Î“ â‰ˆ 1 (mildly relativistic)
//
// Planetary shallow water (rotating shallow water with topography):
//   âˆ‚h/âˆ‚t + âˆ‡Â·(hu) = 0
//   âˆ‚u/âˆ‚t + (uÂ·âˆ‡)u = -gâˆ‡(h+b) + fÃ—u + Î½âˆ‡Â²u
//   f = fâ‚€ + Î²y (Î²-plane approx, Coriolis)
//   b = topography
//
// GA: magnetic field strength, resistivity, Coriolis, topography scale
// ============================================================================
#pragma once
#include "pde_base.hpp"

namespace well {

struct PostNeutronStarMerger2D : PDE2D {
    float rho[N2D][N2D]  = {};
    float ux[N2D][N2D]   = {};
    float uy[N2D][N2D]   = {};
    float Bx[N2D][N2D]   = {};   // magnetic field
    float By[N2D][N2D]   = {};
    float E[N2D][N2D]    = {};   // total energy

    struct NeutronGA {
        float B0 = 1.0f;         // initial B magnitude
        float eta = 0.01f;       // resistivity
        float gamma = 4.0f/3.0f; // relativistic gas
        float grav_strength = 1.0f;
        float neutrino_cool = 0.01f;
    } ga;

    const char* name() const override { return "PostNeutronStarMerger-2D"; }
    int num_fields() const override { return 5; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        float cx = 0.5f, cy = 0.5f;
        for (int j = 0; j < N2D; ++j) {
            float y = (float)j / (float)N2D;
            for (int i = 0; i < N2D; ++i) {
                float x = (float)i / (float)N2D;
                float r = sqrtf((x-cx)*(x-cx) + (y-cy)*(y-cy));

                // Dense remnant at center
                rho[j][i] = 0.1f + 10.0f * expf(-r*r / (0.05f*0.05f));

                // Angular momentum â†’ rotation
                float theta = atan2f(y-cy, x-cx);
                float v_rot = 0.5f * fminf(r / 0.1f, 1.0f);
                ux[j][i] = -v_rot * sinf(theta) + rng.normal(0, 0.01f);
                uy[j][i] =  v_rot * cosf(theta) + rng.normal(0, 0.01f);

                // Toroidal magnetic field
                Bx[j][i] = -ga.B0 * sinf(theta) * expf(-r*r/(0.1f*0.1f));
                By[j][i] =  ga.B0 * cosf(theta) * expf(-r*r/(0.1f*0.1f));

                float p0 = 0.01f + rho[j][i] * 0.1f;
                float ke = 0.5f * rho[j][i] * (ux[j][i]*ux[j][i]+uy[j][i]*uy[j][i]);
                float me = 0.5f * (Bx[j][i]*Bx[j][i]+By[j][i]*By[j][i]);
                E[j][i] = p0/(ga.gamma-1.0f) + ke + me;
            }
        }
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float gm1 = ga.gamma - 1.0f;

        // CFL
        float max_v = 0;
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                float r = fmaxf(rho[j][i], 1e-8f);
                float v = sqrtf(ux[j][i]*ux[j][i]+uy[j][i]*uy[j][i]);
                float B2 = Bx[j][i]*Bx[j][i]+By[j][i]*By[j][i];
                float va = sqrtf(B2/r);
                if (v+va > max_v) max_v = v+va;
            }
        float dt = dt_ > 0 ? dt_ : 0.3f*dx/(max_v+1e-8f);
        dt = fminf(dt, 0.005f);

        float rho_n[N2D][N2D], ux_n[N2D][N2D], uy_n[N2D][N2D];
        float Bx_n[N2D][N2D], By_n[N2D][N2D], E_n[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j)
            for (int i = 1; i < N2D-1; ++i) {
                float r = fmaxf(rho[j][i], 1e-8f);
                float u = ux[j][i], v = uy[j][i];
                float bx = Bx[j][i], by = By[j][i];
                float B2 = bx*bx + by*by;
                float ke = 0.5f*r*(u*u+v*v);
                float me = 0.5f*B2;
                float p = gm1*(E[j][i]-ke-me); if (p < 1e-10f) p = 1e-10f;
                float ptot = p + 0.5f*B2;

                // Gravity: central potential
                float x = (float)i/(float)N2D - 0.5f;
                float y = (float)j/(float)N2D - 0.5f;
                float rad = sqrtf(x*x+y*y) + 0.01f;
                float gx = -ga.grav_strength * x / (rad*rad*rad);
                float gy = -ga.grav_strength * y / (rad*rad*rad);

                // Laplacians (for diffusion)
                float lap_bx = (Bx[j][i-1]+Bx[j][i+1]+Bx[j-1][i]+Bx[j+1][i]-4.0f*bx)/(dx*dx);
                float lap_by = (By[j][i-1]+By[j][i+1]+By[j-1][i]+By[j+1][i]-4.0f*by)/(dx*dx);

                // Simplified fluxes (central diff)
                float dudx = (ux[j][i+1]-ux[j][i-1])/(2.0f*dx);
                float dudy = (ux[j+1][i]-ux[j-1][i])/(2.0f*dx);
                float dvdx = (uy[j][i+1]-uy[j][i-1])/(2.0f*dx);
                float dvdy = (uy[j+1][i]-uy[j-1][i])/(2.0f*dx);

                // Mass
                float drhodx = (rho[j][i+1]-rho[j][i-1])/(2.0f*dx);
                float drhody = (rho[j+1][i]-rho[j-1][i])/(2.0f*dx);
                rho_n[j][i] = r + dt*(-u*drhodx - v*drhody - r*(dudx+dvdy));

                // Momentum (MHD: magnetic stress)
                float dBxdx = (Bx[j][i+1]-Bx[j][i-1])/(2.0f*dx);
                float dBxdy = (Bx[j+1][i]-Bx[j-1][i])/(2.0f*dx);
                float dBydx = (By[j][i+1]-By[j][i-1])/(2.0f*dx);
                float dBydy = (By[j+1][i]-By[j-1][i])/(2.0f*dx);

                float lap_u = (ux[j][i-1]+ux[j][i+1]+ux[j-1][i]+ux[j+1][i]-4.0f*u)/(dx*dx);
                float lap_v = (uy[j][i-1]+uy[j][i+1]+uy[j-1][i]+uy[j+1][i]-4.0f*v)/(dx*dx);

                ux_n[j][i] = u + dt*(-u*dudx - v*dudy -(1.0f/r)*(ptot-p)*dBxdx
                            + (bx*dBxdx+by*dBxdy)/r + 0.01f*lap_u + gx);
                uy_n[j][i] = v + dt*(-u*dvdx - v*dvdy -(1.0f/r)*(ptot-p)*dBydy
                            + (bx*dBydx+by*dBydy)/r + 0.01f*lap_v + gy);

                // Induction: dB/dt = curl(uÃ—B) + Î·âˆ‡Â²B
                Bx_n[j][i] = bx + dt*((u*dBydy - v*dBydx) + ga.eta*lap_bx);
                By_n[j][i] = by + dt*(-(u*dBxdy - v*dBxdx) + ga.eta*lap_by);

                // Energy
                float cooling = ga.neutrino_cool * p * p / r;
                E_n[j][i] = E[j][i] + dt*(-cooling);

                if (rho_n[j][i] < 1e-6f) rho_n[j][i] = 1e-6f;
            }

        // Outflow BC
        for (int k = 0; k < N2D; ++k) {
            rho_n[0][k]=rho_n[1][k]; rho_n[N2D-1][k]=rho_n[N2D-2][k];
            rho_n[k][0]=rho_n[k][1]; rho_n[k][N2D-1]=rho_n[k][N2D-2];
            ux_n[0][k]=ux_n[1][k]; ux_n[N2D-1][k]=ux_n[N2D-2][k];
            ux_n[k][0]=ux_n[k][1]; ux_n[k][N2D-1]=ux_n[k][N2D-2];
            uy_n[0][k]=uy_n[1][k]; uy_n[N2D-1][k]=uy_n[N2D-2][k];
            uy_n[k][0]=uy_n[k][1]; uy_n[k][N2D-1]=uy_n[k][N2D-2];
            Bx_n[0][k]=Bx_n[1][k]; Bx_n[N2D-1][k]=Bx_n[N2D-2][k];
            Bx_n[k][0]=Bx_n[k][1]; Bx_n[k][N2D-1]=Bx_n[k][N2D-2];
            By_n[0][k]=By_n[1][k]; By_n[N2D-1][k]=By_n[N2D-2][k];
            By_n[k][0]=By_n[k][1]; By_n[k][N2D-1]=By_n[k][N2D-2];
            E_n[0][k]=E_n[1][k]; E_n[N2D-1][k]=E_n[N2D-2][k];
            E_n[k][0]=E_n[k][1]; E_n[k][N2D-1]=E_n[k][N2D-2];
        }

        memcpy(rho,rho_n,sizeof(rho)); memcpy(ux,ux_n,sizeof(ux));
        memcpy(uy,uy_n,sizeof(uy)); memcpy(Bx,Bx_n,sizeof(Bx));
        memcpy(By,By_n,sizeof(By)); memcpy(E,E_n,sizeof(E));
        tick++;

        float me = 0;
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i)
                me += Bx[j][i]*Bx[j][i]+By[j][i]*By[j][i];
        engine_ga.fitness = (int)(me);
    }

    void write_field(int f, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j*N2D+i;
                switch(f) { case 0: out[idx]=rho[j][i]; break;
                    case 1: out[idx]=ux[j][i]; break;
                    case 2: out[idx]=uy[j][i]; break;
                    case 3: out[idx]=Bx[j][i]; break;
                    default: out[idx]=By[j][i]; break; }
            }
    }
    void ga_mutate(Rng& r) override { PDE2D::ga_mutate(r);
        ga.B0 *= 1.0f+r.normal(0,0.05f); ga.eta *= 1.0f+r.normal(0,0.05f);
    }
};

// â”€â”€ Planetary Shallow Water (Î²-plane with topography) â”€â”€
struct PlanetarySWE2D : PDE2D {
    float h[N2D][N2D]   = {};   // fluid depth
    float ux[N2D][N2D]  = {};
    float uy[N2D][N2D]  = {};
    float topo[N2D][N2D] = {};  // bottom topography

    struct PlanetGA {
        float g = 9.81f;
        float f0 = 1.0f;          // Coriolis parameter
        float beta = 0.5f;        // Î²-plane
        float H0 = 1.0f;          // mean depth
        float viscosity = 0.001f;
        float topo_scale = 0.2f;
    } ga;

    const char* name() const override { return "PlanetarySWE-2D"; }
    int num_fields() const override { return 3; }

    void init(uint64_t seed) override {
        PDE2D::init(seed);
        // Generate random topography
        for (int j = 0; j < N2D; ++j) {
            float y = (float)j / (float)N2D;
            for (int i = 0; i < N2D; ++i) {
                float x = (float)i / (float)N2D;
                // Superposition of modes for interesting terrain
                topo[j][i] = ga.topo_scale * (
                    0.3f * cosf(4.0f*3.14159f*x) * sinf(2.0f*3.14159f*y)
                    + 0.2f * sinf(6.0f*3.14159f*x+1.0f) * cosf(4.0f*3.14159f*y)
                    + 0.1f * cosf(8.0f*3.14159f*x) * cosf(8.0f*3.14159f*y));

                h[j][i] = ga.H0 - topo[j][i] + rng.normal(0, 0.01f);
                if (h[j][i] < 0.01f) h[j][i] = 0.01f;

                // Initial geostrophic flow
                float f = ga.f0 + ga.beta * (y - 0.5f);
                ux[j][i] = -0.1f * cosf(2.0f*3.14159f*y);
                uy[j][i] = 0.1f * sinf(2.0f*3.14159f*x);
            }
        }
    }

    void step(float dt_) override {
        float dx = 1.0f / (float)N2D;
        float max_c = sqrtf(ga.g * ga.H0 * 2.0f);
        float dt = dt_ > 0 ? dt_ : 0.3f * dx / (max_c + 1.0f);
        dt = fminf(dt, 0.01f);

        float h_n[N2D][N2D], ux_n[N2D][N2D], uy_n[N2D][N2D];

        for (int j = 1; j < N2D-1; ++j) {
            float y = (float)j / (float)N2D;
            float f = ga.f0 + ga.beta * (y - 0.5f);

            for (int i = 1; i < N2D-1; ++i) {
                float u = ux[j][i], v = uy[j][i], hv = h[j][i];

                float dhdx = (h[j][i+1]-h[j][i-1])/(2.0f*dx);
                float dhdy = (h[j+1][i]-h[j-1][i])/(2.0f*dx);
                float dudx = (ux[j][i+1]-ux[j][i-1])/(2.0f*dx);
                float dudy = (ux[j+1][i]-ux[j-1][i])/(2.0f*dx);
                float dvdx = (uy[j][i+1]-uy[j][i-1])/(2.0f*dx);
                float dvdy = (uy[j+1][i]-uy[j-1][i])/(2.0f*dx);

                float dbdx = (topo[j][i+1]-topo[j][i-1])/(2.0f*dx);
                float dbdy = (topo[j+1][i]-topo[j-1][i])/(2.0f*dx);

                float lap_u = (ux[j][i-1]+ux[j][i+1]+ux[j-1][i]+ux[j+1][i]-4.0f*u)/(dx*dx);
                float lap_v = (uy[j][i-1]+uy[j][i+1]+uy[j-1][i]+uy[j+1][i]-4.0f*v)/(dx*dx);

                // Continuity
                h_n[j][i] = hv + dt*(-u*dhdx - v*dhdy - hv*(dudx+dvdy));

                // Momentum + Coriolis + gravity + topography
                ux_n[j][i] = u + dt*(-u*dudx - v*dudy + f*v
                            - ga.g*(dhdx+dbdx) + ga.viscosity*lap_u);
                uy_n[j][i] = v + dt*(-u*dvdx - v*dvdy - f*u
                            - ga.g*(dhdy+dbdy) + ga.viscosity*lap_v);

                if (h_n[j][i] < 0.001f) h_n[j][i] = 0.001f;
            }
        }

        // Periodic BC
        for (int j = 0; j < N2D; ++j) {
            h_n[j][0]=h_n[j][N2D-2]; h_n[j][N2D-1]=h_n[j][1];
            ux_n[j][0]=ux_n[j][N2D-2]; ux_n[j][N2D-1]=ux_n[j][1];
            uy_n[j][0]=uy_n[j][N2D-2]; uy_n[j][N2D-1]=uy_n[j][1];
        }
        for (int i = 0; i < N2D; ++i) {
            h_n[0][i]=h_n[N2D-2][i]; h_n[N2D-1][i]=h_n[1][i];
            ux_n[0][i]=ux_n[N2D-2][i]; ux_n[N2D-1][i]=ux_n[1][i];
            uy_n[0][i]=uy_n[N2D-2][i]; uy_n[N2D-1][i]=uy_n[1][i];
        }

        memcpy(h, h_n, sizeof(h)); memcpy(ux, ux_n, sizeof(ux));
        memcpy(uy, uy_n, sizeof(uy));
        tick++;

        // Fitness: vorticity
        float vort = 0;
        for (int j = 1; j < N2D-1; ++j)
            for (int i = 1; i < N2D-1; ++i) {
                float w = (uy[j][i+1]-uy[j][i-1])/(2.0f*dx)
                        - (ux[j+1][i]-ux[j-1][i])/(2.0f*dx);
                vort += fabsf(w);
            }
        engine_ga.fitness = (int)(vort);
    }

    void write_field(int f, float* out) const override {
        for (int j = 0; j < N2D; ++j)
            for (int i = 0; i < N2D; ++i) {
                int idx = j*N2D+i;
                if (f==0) out[idx]=h[j][i];
                else if (f==1) out[idx]=ux[j][i];
                else out[idx]=uy[j][i];
            }
    }
    void ga_mutate(Rng& r) override { PDE2D::ga_mutate(r);
        ga.f0 *= 1.0f+r.normal(0,0.05f); ga.beta *= 1.0f+r.normal(0,0.05f);
    }
};

} // namespace well
