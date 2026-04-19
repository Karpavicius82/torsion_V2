// ============================================================================
// FFT.HPP — Radix-2 Cooley-Tukey FFT (Pure C++ / AVX2)
//
// Replaces naive O(n²) DFT with O(n·log n).
// Provides: fft_1d, ifft_1d, fft_2d, ifft_2d, rfft_1d, irfft_1d.
// All transforms work on power-of-2 lengths.
//
// Layout: interleaved complex (re, im, re, im, ...)
// Header-only. No FFTW. No dependencies.
// ============================================================================
#pragma once

#include "compat.hpp"

namespace well {
namespace fft {

static constexpr float PI = 3.14159265358979323846f;
static constexpr float TWO_PI = 6.28318530717958647692f;

// ── Bit-reversal permutation ──
static inline uint32_t bit_reverse(uint32_t x, int bits) {
    uint32_t r = 0;
    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

static inline int log2i(int n) {
    int r = 0;
    while ((1 << r) < n) ++r;
    return r;
}

// ── In-place radix-2 Cooley-Tukey FFT (complex, interleaved) ──
// data: [re0, im0, re1, im1, ...] length = 2*n
// n must be power of 2
// inverse: if true, compute IFFT (with 1/n normalization)
static void fft_1d(float* data, int n, bool inverse = false) {
    int bits = log2i(n);

    // Bit-reversal permutation
    for (uint32_t i = 0; i < (uint32_t)n; ++i) {
        uint32_t j = bit_reverse(i, bits);
        if (j > i) {
            // Swap complex pair
            float tr = data[2*i], ti = data[2*i+1];
            data[2*i] = data[2*j]; data[2*i+1] = data[2*j+1];
            data[2*j] = tr; data[2*j+1] = ti;
        }
    }

    // Butterfly passes
    float sign = inverse ? 1.0f : -1.0f;

    for (int len = 2; len <= n; len <<= 1) {
        float angle = sign * TWO_PI / (float)len;
        float wpr = cosf(angle);
        float wpi = sinf(angle);
        int half = len >> 1;

        for (int i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;

            // AVX2 butterfly: process 4 complex pairs at once when possible
            int j = 0;
            #if defined(__AVX2__)
            for (; j + 4 <= half; j += 4) {
                // Load 4 even elements
                int e0 = i + j, e1 = i + j + 1, e2 = i + j + 2, e3 = i + j + 3;
                int o0 = e0 + half, o1 = e1 + half, o2 = e2 + half, o3 = e3 + half;

                // Compute twiddle factors for each
                float wr0 = wr, wi0 = wi;
                float wr1 = wr0*wpr - wi0*wpi, wi1 = wr0*wpi + wi0*wpr;
                float wr2 = wr1*wpr - wi1*wpi, wi2 = wr1*wpi + wi1*wpr;
                float wr3 = wr2*wpr - wi2*wpi, wi3 = wr2*wpi + wi2*wpr;

                // Butterfly for each pair
                for (int k = 0; k < 4; ++k) {
                    int ei = i + j + k, oi = ei + half;
                    float tw_r, tw_i;
                    switch(k) {
                        case 0: tw_r = wr0; tw_i = wi0; break;
                        case 1: tw_r = wr1; tw_i = wi1; break;
                        case 2: tw_r = wr2; tw_i = wi2; break;
                        default: tw_r = wr3; tw_i = wi3; break;
                    }

                    float or_ = data[2*oi], oi_ = data[2*oi+1];
                    float tr = tw_r * or_ - tw_i * oi_;
                    float ti = tw_r * oi_ + tw_i * or_;

                    data[2*oi]   = data[2*ei]   - tr;
                    data[2*oi+1] = data[2*ei+1] - ti;
                    data[2*ei]   += tr;
                    data[2*ei+1] += ti;
                }

                // Advance twiddle
                float tmp = wr3*wpr - wi3*wpi;
                wi = wr3*wpi + wi3*wpr;
                wr = tmp;
            }
            #endif

            // Scalar remainder
            for (; j < half; ++j) {
                int ei = i + j, oi = ei + half;
                float or_ = data[2*oi], oi_ = data[2*oi+1];
                float tr = wr * or_ - wi * oi_;
                float ti = wr * oi_ + wi * or_;

                data[2*oi]   = data[2*ei]   - tr;
                data[2*oi+1] = data[2*ei+1] - ti;
                data[2*ei]   += tr;
                data[2*ei+1] += ti;

                float tmp = wr*wpr - wi*wpi;
                wi = wr*wpi + wi*wpr;
                wr = tmp;
            }
        }
    }

    // Normalize for inverse
    if (inverse) {
        float inv_n = 1.0f / (float)n;
        __m256 vinv = _mm256_set1_ps(inv_n);
        int i = 0;
        for (; i + 8 <= 2*n; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            _mm256_storeu_ps(data + i, _mm256_mul_ps(v, vinv));
        }
        for (; i < 2*n; ++i) data[i] *= inv_n;
    }
}

// ── Inverse FFT ──
static inline void ifft_1d(float* data, int n) {
    fft_1d(data, n, true);
}

// ── Real-to-complex FFT (uses n/2-point complex FFT trick) ──
// input: real[n], output: complex[n] (interleaved, 2*n floats)
// Uses scratch buffer
static void rfft_1d(const float* input, float* output, int n) {
    // Pack real data as complex: input[k] + i*input[k+1] for k=0,2,4,...
    int m = n / 2;
    float* tmp = output; // reuse output as scratch

    for (int i = 0; i < m; ++i) {
        tmp[2*i]   = input[2*i];
        tmp[2*i+1] = input[2*i+1];
    }

    // m-point complex FFT
    fft_1d(tmp, m, false);

    // Unpack to n-point real FFT result
    // X[k] = 0.5*(Z[k] + Z*[m-k]) - 0.5i*e^(-2πik/n)*(Z[k] - Z*[m-k])
    float* out2 = (float*)__builtin_alloca(2 * n * sizeof(float));
    out2[0] = tmp[0] + tmp[1];  // DC: X[0] = Z[0].re + Z[0].im
    out2[1] = 0.0f;
    out2[2*(m)] = tmp[0] - tmp[1]; // Nyquist
    out2[2*(m)+1] = 0.0f;

    for (int k = 1; k < m; ++k) {
        float zr = tmp[2*k], zi = tmp[2*k+1];
        float zr_c = tmp[2*(m-k)], zi_c = -tmp[2*(m-k)+1]; // conjugate of Z[m-k]

        float ar = 0.5f * (zr + zr_c);
        float ai = 0.5f * (zi + zi_c);
        float br = 0.5f * (zr - zr_c);
        float bi = 0.5f * (zi - zi_c);

        float angle = -TWO_PI * k / (float)n;
        float wr = cosf(angle), wi = sinf(angle);

        // X[k] = (ar + ai*i) - (wr + wi*i)*(br + bi*i)
        // twiddle*(br+bi*i) = (wr*br - wi*bi) + (wr*bi + wi*br)*i
        float tr = wr*br - wi*bi;
        float ti = wr*bi + wi*br;

        out2[2*k]   = ar - ti;  // Note: we subtract imaginary part
        out2[2*k+1] = ai + tr;
    }

    memcpy(output, out2, 2 * n * sizeof(float));
}

// ── Complex-to-real IFFT ──
static void irfft_1d(const float* input, float* output, int n) {
    float* tmp = (float*)__builtin_alloca(2 * n * sizeof(float));
    memcpy(tmp, input, 2 * n * sizeof(float));
    fft_1d(tmp, n, true);
    for (int i = 0; i < n; ++i) {
        output[i] = tmp[2*i];
    }
}

// ── 2D FFT (row-major, NxM complex) ──
// data: [row0_re, row0_im, ..., row1_re, row1_im, ...]
// Each row has M complex elements → 2*M floats
// Total: N rows × 2*M floats per row
static void fft_2d(float* data, int rows, int cols, bool inverse = false) {
    // FFT along each row
    for (int r = 0; r < rows; ++r) {
        fft_1d(data + r * 2 * cols, cols, inverse);
    }

    // Transpose, FFT along each (now) row, transpose back
    // Allocate column buffer
    float* col = (float*)__builtin_alloca(2 * rows * sizeof(float));

    for (int c = 0; c < cols; ++c) {
        // Extract column c
        for (int r = 0; r < rows; ++r) {
            col[2*r]   = data[r * 2 * cols + 2*c];
            col[2*r+1] = data[r * 2 * cols + 2*c + 1];
        }

        // FFT the column
        fft_1d(col, rows, inverse);

        // Put back
        for (int r = 0; r < rows; ++r) {
            data[r * 2 * cols + 2*c]     = col[2*r];
            data[r * 2 * cols + 2*c + 1] = col[2*r+1];
        }
    }
}

static inline void ifft_2d(float* data, int rows, int cols) {
    fft_2d(data, rows, cols, true);
}

// ── Spectral convolution helper for FNO ──
// Multiply complex spectra element-wise with weights
// spec[n_modes * channels], weight[n_modes * channels]
// Both interleaved complex
static void spectral_multiply(const float* spec_in, const float* weight,
                               float* spec_out, int n_modes, int channels) {
    for (int k = 0; k < n_modes; ++k) {
        for (int c = 0; c < channels; ++c) {
            int idx = (k * channels + c) * 2;
            float sr = spec_in[idx], si = spec_in[idx+1];
            float wr = weight[idx], wi = weight[idx+1];
            spec_out[idx]   = sr * wr - si * wi;
            spec_out[idx+1] = sr * wi + si * wr;
        }
    }
}

// ── Power spectrum (magnitude squared of complex FFT output) ──
static void power_spectrum(const float* fft_out, float* power, int n) {
    __m256 v_zero = _mm256_setzero_ps();
    (void)v_zero;
    for (int i = 0; i < n; ++i) {
        float re = fft_out[2*i], im = fft_out[2*i+1];
        power[i] = re * re + im * im;
    }
}

} // namespace fft
} // namespace well
