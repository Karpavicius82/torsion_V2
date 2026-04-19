// ============================================================================
// METRICS.HPP â€” Physics-Aware Training Metrics (Pure C++ / AVX2)
//
// Replaces WandB/TensorBoard metrics with:
//   VRMSE  â€” Variance-weighted RMSE
//   Spectral Loss â€” error in frequency domain
//   Correlation â€” Pearson correlation coefficient
//   Energy Error â€” relative error in total energy
//   Gradient Norm â€” spatial gradient magnitude
// ============================================================================
#pragma once

#include "../tensor.hpp"
#include "../core/fft.hpp"
#include <immintrin.h>

namespace well {
namespace metrics {

// â”€â”€ VRMSE: âˆš(MSE / Var(target)) â”€â”€
static float vrmse(const float* pred, const float* target, int n) {
    float mse = 0, mean = 0, var = 0;

    // Mean
    for (int i = 0; i < n; ++i) mean += target[i];
    mean /= (float)n;

    // MSE and variance
    __m256 acc_mse = _mm256_setzero_ps();
    __m256 acc_var = _mm256_setzero_ps();
    __m256 vmean = _mm256_set1_ps(mean);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vp = _mm256_loadu_ps(pred + i);
        __m256 vt = _mm256_loadu_ps(target + i);
        __m256 d = _mm256_sub_ps(vp, vt);
        acc_mse = _mm256_fmadd_ps(d, d, acc_mse);
        __m256 dm = _mm256_sub_ps(vt, vmean);
        acc_var = _mm256_fmadd_ps(dm, dm, acc_var);
    }

    // Horizontal sum
    float a[8];
    _mm256_storeu_ps(a, acc_mse);
    for (int k = 0; k < 8; ++k) mse += a[k];
    _mm256_storeu_ps(a, acc_var);
    for (int k = 0; k < 8; ++k) var += a[k];

    for (; i < n; ++i) {
        float d = pred[i] - target[i]; mse += d*d;
        float dm = target[i] - mean; var += dm*dm;
    }

    mse /= (float)n;
    var /= (float)n;

    return (var > 1e-12f) ? sqrtf(mse / var) : sqrtf(mse);
}

// â”€â”€ Spectral Loss: MSE in frequency domain â”€â”€
static float spectral_loss(const float* pred, const float* target, int n) {
    // Allocate complex buffers
    int alloc_n = ((2 * n) + 7) & ~7;
    float* fft_pred = (float*)aligned_alloc_impl(32, alloc_n * sizeof(float));
    float* fft_tgt  = (float*)aligned_alloc_impl(32, alloc_n * sizeof(float));

    // Pack as complex
    for (int i = 0; i < n; ++i) {
        fft_pred[2*i] = pred[i]; fft_pred[2*i+1] = 0;
        fft_tgt[2*i]  = target[i]; fft_tgt[2*i+1] = 0;
    }

    // FFT
    fft::fft_1d(fft_pred, n);
    fft::fft_1d(fft_tgt, n);

    // MSE of power spectra
    float loss = 0;
    for (int i = 0; i < n; ++i) {
        float pr = fft_pred[2*i]*fft_pred[2*i] + fft_pred[2*i+1]*fft_pred[2*i+1];
        float tr = fft_tgt[2*i]*fft_tgt[2*i] + fft_tgt[2*i+1]*fft_tgt[2*i+1];
        float d = sqrtf(pr) - sqrtf(tr);
        loss += d * d;
    }

    aligned_free_impl(fft_pred);
    aligned_free_impl(fft_tgt);

    return loss / (float)n;
}

// â”€â”€ Pearson Correlation â”€â”€
static float correlation(const float* pred, const float* target, int n) {
    float mp = 0, mt = 0;
    for (int i = 0; i < n; ++i) { mp += pred[i]; mt += target[i]; }
    mp /= (float)n; mt /= (float)n;

    float cov = 0, vp = 0, vt = 0;
    for (int i = 0; i < n; ++i) {
        float dp = pred[i] - mp, dt = target[i] - mt;
        cov += dp * dt;
        vp += dp * dp;
        vt += dt * dt;
    }

    float denom = sqrtf(vp * vt);
    return (denom > 1e-12f) ? cov / denom : 0.0f;
}

// â”€â”€ Energy Error: |E_pred - E_target| / E_target â”€â”€
static float energy_error(const float* pred, const float* target, int n) {
    float ep = 0, et = 0;
    for (int i = 0; i < n; ++i) { ep += pred[i]*pred[i]; et += target[i]*target[i]; }
    return (et > 1e-12f) ? fabsf(ep - et) / et : fabsf(ep - et);
}

// â”€â”€ Gradient Norm: mean |âˆ‡f| for 2D field â”€â”€
static float gradient_norm(const float* field, int H, int W) {
    float gn = 0;
    for (int j = 1; j < H-1; ++j)
        for (int i = 1; i < W-1; ++i) {
            float gx = field[j*W+i+1] - field[j*W+i-1];
            float gy = field[(j+1)*W+i] - field[(j-1)*W+i];
            gn += sqrtf(gx*gx + gy*gy);
        }
    return gn / (float)((H-2) * (W-2));
}

// â”€â”€ Max Absolute Error â”€â”€
static float max_error(const float* pred, const float* target, int n) {
    float mx = 0;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(pred[i] - target[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// â”€â”€ L1 Loss â”€â”€
static float l1_loss(const float* pred, const float* target, int n) {
    float sum = 0;
    for (int i = 0; i < n; ++i) sum += fabsf(pred[i] - target[i]);
    return sum / (float)n;
}

} // namespace metrics
} // namespace well
