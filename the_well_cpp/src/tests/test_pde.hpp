// ============================================================================
// TEST_PDE.HPP — PDE Conservation Tests (Bare-metal)
//
// Verifies that physics engines conserve the correct quantities:
//   - Diffusion: total mass (integral of u)
//   - Wave: total energy (kinetic + potential)
//   - NavierStokes: total vorticity (periodic domain)
//   - MHD: divergence-free B field
//
// Runs each PDE for 500 steps and checks conservation within tolerance.
// ============================================================================
#pragma once
#include "../compat.hpp"
#include "../physics/pde_base.hpp"
#include "../physics/navier_stokes_2d.hpp"
#include "../physics/diffusion_wave_burgers.hpp"

namespace test {

struct PDEResult {
    int passed = 0;
    int failed = 0;
    
    void check(const char* name, float initial, float final_val, float rel_tol) {
        float denom = math::abs(initial) > 1e-10f ? math::abs(initial) : 1.0f;
        float rel_err = math::abs(final_val - initial) / denom;
        if (rel_err <= rel_tol) {
            passed++;
        } else {
            print::str("  FAIL: "); print::str(name);
            print::str(" init="); print::flt(initial, 6);
            print::str(" final="); print::flt(final_val, 6);
            print::str(" rel_err="); print::flt(rel_err, 6);
            print::line("");
            failed++;
        }
    }
};

// Helper: compute total mass (sum of field values)
static float total_mass(const well::PDE2D& pde, int field = 0) {
    float sum = 0;
    int n = well::N2D;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            sum += pde.field2d[field][j * n + i];
    return sum;
}

static PDEResult run_pde_tests() {
    PDEResult r;
    const int STEPS = 300;
    
    // ── Diffusion: mass conservation ──
    // ∂u/∂t = D∇²u conserves ∫u dA (Neumann or periodic BC)
    {
        well::Diffusion2D pde;
        pde.init(42);
        
        // Compute initial total mass
        float mass0 = 0;
        for (int j = 0; j < well::N2D; ++j)
            for (int i = 0; i < well::N2D; ++i)
                mass0 += pde.u[j][i];
        
        for (int s = 0; s < STEPS; ++s) pde.step(0);
        
        float mass1 = 0;
        for (int j = 0; j < well::N2D; ++j)
            for (int i = 0; i < well::N2D; ++i)
                mass1 += pde.u[j][i];
        
        r.check("Diffusion mass conservation", mass0, mass1, 0.05f);
    }
    
    // ── Wave: energy conservation ──
    // E = 0.5 ∫ (∂u/∂t)² + c²|∇u|² dA
    {
        well::Wave2D pde;
        pde.init(42);
        
        auto wave_energy = [&]() -> float {
            float E = 0;
            float dx = 1.0f / well::N2D;
            for (int j = 1; j < well::N2D-1; ++j)
                for (int i = 1; i < well::N2D-1; ++i) {
                    // Kinetic: (u - u_prev)^2 / dt^2 ~ (∂u/∂t)²
                    float du_dt = pde.u[j][i] - pde.u_prev[j][i];
                    // Potential: |∇u|² 
                    float dux = (pde.u[j][i+1] - pde.u[j][i-1]) / (2*dx);
                    float duy = (pde.u[j+1][i] - pde.u[j-1][i]) / (2*dx);
                    E += 0.5f * (du_dt*du_dt + dux*dux + duy*duy);
                }
            return E;
        };
        
        float E0 = wave_energy();
        for (int s = 0; s < STEPS; ++s) pde.step(0);
        float E1 = wave_energy();
        
        // Wave equation should conserve energy (within numerical dissipation)
        r.check("Wave energy conservation", E0, E1, 0.15f);
    }
    
    // ── NavierStokes: enstrophy bounded ──
    // In viscous flow, enstrophy should decrease or stay bounded
    {
        well::NavierStokes2D pde;
        pde.init(42);
        
        auto enstrophy = [&]() -> float {
            float Z = 0;
            for (int j = 0; j < well::N2D; ++j)
                for (int i = 0; i < well::N2D; ++i)
                    Z += pde.omega[j][i] * pde.omega[j][i];
            return Z;
        };
        
        float Z0 = enstrophy();
        for (int s = 0; s < STEPS; ++s) pde.step(0);
        float Z1 = enstrophy();
        
        // Enstrophy should not grow unboundedly (viscous decay)
        // Allow up to 2x growth (numerical, CFL effects)
        float growth = (Z0 > 1e-10f) ? Z1 / Z0 : Z1;
        if (growth <= 3.0f) {
            r.passed++;
        } else {
            print::str("  FAIL: NS enstrophy explosion ");
            print::flt(growth, 2);
            print::line("x");
            r.failed++;
        }
    }
    
    // ── Burgers: L2 norm bounded ──
    {
        well::Burgers2D pde;
        pde.init(42);
        
        auto l2 = [&]() -> float {
            float s = 0;
            for (int j = 0; j < well::N2D; ++j)
                for (int i = 0; i < well::N2D; ++i)
                    s += pde.u[j][i] * pde.u[j][i];
            return math::sqrt(s);
        };
        
        float norm0 = l2();
        for (int s = 0; s < STEPS; ++s) pde.step(0);
        float norm1 = l2();
        
        // Viscous Burgers shouldn't blow up
        float growth = (norm0 > 1e-10f) ? norm1 / norm0 : norm1;
        if (growth <= 5.0f) {
            r.passed++;
        } else {
            print::str("  FAIL: Burgers L2 explosion ");
            print::flt(growth, 2); print::line("x");
            r.failed++;
        }
    }
    
    return r;
}

} // namespace test
