// ============================================================================
// TEST_FFT.HPP — FFT Unit Tests (Bare-metal)
//
// Tests:
//  1. Forward + Inverse = identity (round-trip)
//  2. Parseval's theorem (energy conservation)
//  3. Known signal: single frequency sinusoid → peak at correct bin
//  4. DC signal → all energy in bin 0
//  5. 2D FFT round-trip
// ============================================================================
#pragma once
#include "../compat.hpp"
#include "../core/fft.hpp"

namespace test {

struct FFTResult {
    int passed = 0;
    int failed = 0;
    
    void check(const char* name, float err, float tol) {
        if (err <= tol) {
            passed++;
        } else {
            print::str("  FAIL: "); print::str(name);
            print::str(" err="); print::flt(err, 8);
            print::str(" tol="); print::flt(tol, 8);
            print::line("");
            failed++;
        }
    }
    
    void test_pass(const char* name) {
        print::str("  PASS: "); print::str(name); print::line("");
        passed++;
    }
};

static FFTResult run_fft_tests() {
    FFTResult r;
    const int N = 64;  // test size
    
    // ── Test 1: Forward + Inverse = Identity ──
    {
        float data[2 * N];
        float original[2 * N];
        
        // Fill with known signal
        for (int i = 0; i < N; ++i) {
            data[2*i]   = math::sin(2.0f * math::PI * 3.0f * i / N) + 0.5f * math::cos(2.0f * math::PI * 7.0f * i / N);
            data[2*i+1] = 0.0f;
            original[2*i]   = data[2*i];
            original[2*i+1] = 0.0f;
        }
        
        well::fft::fft_1d(data, N, false);   // forward
        well::fft::fft_1d(data, N, true);    // inverse
        
        float max_err = 0;
        for (int i = 0; i < N; ++i) {
            float err = math::abs(data[2*i] - original[2*i]);
            if (err > max_err) max_err = err;
            err = math::abs(data[2*i+1] - original[2*i+1]);
            if (err > max_err) max_err = err;
        }
        r.check("FFT roundtrip (N=64)", max_err, 1e-4f);
    }
    
    // ── Test 2: Parseval's Theorem ──
    // sum|x[n]|^2 = (1/N) sum|X[k]|^2
    {
        float data[2 * N];
        float time_energy = 0;
        
        for (int i = 0; i < N; ++i) {
            data[2*i]   = math::sin(2.0f * math::PI * 5.0f * i / N);
            data[2*i+1] = 0.0f;
            time_energy += data[2*i] * data[2*i];
        }
        
        well::fft::fft_1d(data, N, false);
        
        float freq_energy = 0;
        for (int i = 0; i < N; ++i) {
            freq_energy += data[2*i]*data[2*i] + data[2*i+1]*data[2*i+1];
        }
        freq_energy /= (float)N;
        
        float rel_err = math::abs(freq_energy - time_energy) / (time_energy + 1e-10f);
        r.check("Parseval's theorem", rel_err, 1e-4f);
    }
    
    // ── Test 3: Single frequency detection ──
    {
        float data[2 * N];
        int target_bin = 5;
        
        for (int i = 0; i < N; ++i) {
            data[2*i]   = math::cos(2.0f * math::PI * target_bin * i / N);
            data[2*i+1] = 0.0f;
        }
        
        well::fft::fft_1d(data, N, false);
        
        // Find peak bin
        float max_mag = 0;
        int peak_bin = -1;
        for (int i = 0; i < N; ++i) {
            float mag = data[2*i]*data[2*i] + data[2*i+1]*data[2*i+1];
            if (mag > max_mag) { max_mag = mag; peak_bin = i; }
        }
        
        if (peak_bin == target_bin) {
            r.test_pass("Peak at correct bin");
        } else {
            r.check("Peak at correct bin", (float)math::abs((float)(peak_bin - target_bin)), 0.0f);
        }
    }
    
    // ── Test 4: DC signal ──
    {
        float data[2 * N];
        float dc_val = 3.7f;
        
        for (int i = 0; i < N; ++i) {
            data[2*i]   = dc_val;
            data[2*i+1] = 0.0f;
        }
        
        well::fft::fft_1d(data, N, false);
        
        // X[0] should be N * dc_val, all others ~ 0
        float dc_expected = N * dc_val;
        float dc_err = math::abs(data[0] - dc_expected);
        r.check("DC component (bin 0)", dc_err, 1e-2f);
        
        float non_dc_max = 0;
        for (int i = 1; i < N; ++i) {
            float mag = math::sqrt(data[2*i]*data[2*i] + data[2*i+1]*data[2*i+1]);
            if (mag > non_dc_max) non_dc_max = mag;
        }
        r.check("Non-DC bins ~0", non_dc_max, 1e-2f);
    }
    
    // ── Test 5: 2D FFT roundtrip ──
    {
        const int M = 16;  // 16x16
        float data[2 * M * M];
        float orig[2 * M * M];
        
        for (int j = 0; j < M; ++j)
            for (int i = 0; i < M; ++i) {
                int idx = j * M + i;
                data[2*idx]   = math::sin(2.0f*math::PI*i/M) * math::cos(2.0f*math::PI*j/M);
                data[2*idx+1] = 0.0f;
                orig[2*idx]   = data[2*idx];
                orig[2*idx+1] = 0.0f;
            }
        
        well::fft::fft_2d(data, M, M, false);
        well::fft::fft_2d(data, M, M, true);
        
        float max_err = 0;
        for (int i = 0; i < M * M; ++i) {
            float err = math::abs(data[2*i] - orig[2*i]);
            if (err > max_err) max_err = err;
        }
        r.check("2D FFT roundtrip (16x16)", max_err, 1e-3f);
    }
    
    return r;
}

} // namespace test
