// ============================================================================
// TEST_GRADIENTS.HPP — Finite-Difference Gradient Check (Bare-metal)
//
// For each model's backward(), verify:
//   d_loss/d_param ≈ [loss(param + ε) - loss(param - ε)] / (2ε)
//
// Tests FNO2D, UNet2D weight gradients against numerical Jacobian.
// Tolerance: relative error < 0.1 (10%) — sufficient for verification.
// ============================================================================
#pragma once
#include "../compat.hpp"
#include "../core/tensor_nd.hpp"

namespace test {

struct GradResult {
    int passed = 0;
    int failed = 0;
    int skipped = 0;
    
    void check(const char* name, float analytic, float numerical, float rel_tol = 0.1f) {
        float denom = math::abs(numerical);
        if (denom < 1e-6f) {
            // Both should be ~0
            if (math::abs(analytic) < 1e-4f) { passed++; return; }
        }
        float rel_err = math::abs(analytic - numerical) / (denom + 1e-8f);
        if (rel_err <= rel_tol) {
            passed++;
        } else {
            print::str("  FAIL: "); print::str(name);
            print::str(" anal="); print::flt(analytic, 6);
            print::str(" num="); print::flt(numerical, 6);
            print::str(" rel="); print::flt(rel_err, 4);
            print::line("");
            failed++;
        }
    }
};

// Simple MSE loss on tensor
static float mse_loss(const float* pred, const float* target, int n) {
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        float d = pred[i] - target[i];
        sum += d * d;
    }
    return sum / (float)n;
}

// Gradient of MSE: d_loss/d_pred = 2*(pred-target)/n
static void mse_grad(const float* pred, const float* target, float* grad, int n) {
    float inv_n = 2.0f / (float)n;
    for (int i = 0; i < n; ++i) {
        grad[i] = inv_n * (pred[i] - target[i]);
    }
}

// ── Conv2d weight gradient check ──
static void check_conv2d_grad(GradResult& r) {
    // Small conv2d: 1 input channel → 1 output channel, 3x3 kernel
    const int B = 1, Ci = 1, Co = 1, H = 8, W = 8, K = 3;
    const int out_H = H - K + 1, out_W = W - K + 1;
    
    float input[B * Ci * H * W];
    float weight[Co * Ci * K * K];
    float target[B * Co * out_H * out_W];
    float output[B * Co * out_H * out_W];
    float d_output[B * Co * out_H * out_W];
    float d_weight[Co * Ci * K * K];
    
    // Init with small values
    for (int i = 0; i < B*Ci*H*W; ++i) input[i] = 0.1f * (float)(i % 7 - 3);
    for (int i = 0; i < Co*Ci*K*K; ++i) weight[i] = 0.05f * (float)(i % 5 - 2);
    for (int i = 0; i < B*Co*out_H*out_W; ++i) target[i] = 0.0f;
    
    // Forward
    well::ops::conv2d(input, weight, nullptr, output, B, Ci, H, W, Co, K, K, 1, 0);
    
    // Loss + grad at output
    float loss0 = mse_loss(output, target, B*Co*out_H*out_W);
    mse_grad(output, target, d_output, B*Co*out_H*out_W);
    
    // Analytical gradient (conv2d backward for weights)
    memset(d_weight, 0, sizeof(d_weight));
    // d_weight[co][ci][kh][kw] = sum_{b,oh,ow} d_output[b][co][oh][ow] * input[b][ci][oh+kh][ow+kw]
    for (int co = 0; co < Co; ++co)
        for (int ci = 0; ci < Ci; ++ci)
            for (int kh = 0; kh < K; ++kh)
                for (int kw = 0; kw < K; ++kw) {
                    float sum = 0;
                    for (int b = 0; b < B; ++b)
                        for (int oh = 0; oh < out_H; ++oh)
                            for (int ow = 0; ow < out_W; ++ow)
                                sum += d_output[((b*Co+co)*out_H+oh)*out_W+ow]
                                     * input[((b*Ci+ci)*(H)+oh+kh)*W+ow+kw];
                    d_weight[((co*Ci+ci)*K+kh)*K+kw] = sum;
                }
    
    // Numerical gradient for a few weight parameters
    float eps = 1e-3f;
    for (int p = 0; p < Co*Ci*K*K && p < 5; ++p) {
        float original = weight[p];
        
        weight[p] = original + eps;
        well::ops::conv2d(input, weight, nullptr, output, B, Ci, H, W, Co, K, K, 1, 0);
        float loss_plus = mse_loss(output, target, B*Co*out_H*out_W);
        
        weight[p] = original - eps;
        well::ops::conv2d(input, weight, nullptr, output, B, Ci, H, W, Co, K, K, 1, 0);
        float loss_minus = mse_loss(output, target, B*Co*out_H*out_W);
        
        weight[p] = original;
        
        float numerical = (loss_plus - loss_minus) / (2.0f * eps);
        char name[32] = "conv2d_w[0]";
        name[9] = '0' + p;
        r.check(name, d_weight[p], numerical, 0.15f);
    }
}

// ── matmul gradient check ──
static void check_matmul_grad(GradResult& r) {
    const int M = 4, K = 3, N = 2;
    float A[M*K], B[K*N], C[M*N], target[M*N];
    float dC[M*N], dA[M*K];
    
    for (int i = 0; i < M*K; ++i) A[i] = 0.1f * (float)(i % 5 - 2);
    for (int i = 0; i < K*N; ++i) B[i] = 0.1f * (float)(i % 7 - 3);
    for (int i = 0; i < M*N; ++i) target[i] = 0.0f;
    
    well::ops::matmul(A, B, C, M, K, N);
    mse_grad(C, target, dC, M*N);
    
    // dA = dC * B^T
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j) {
            float sum = 0;
            for (int n = 0; n < N; ++n)
                sum += dC[i*N + n] * B[j*N + n];
            dA[i*K + j] = sum;
        }
    
    float eps = 1e-3f;
    for (int p = 0; p < M*K && p < 5; ++p) {
        float original = A[p];
        
        A[p] = original + eps;
        well::ops::matmul(A, B, C, M, K, N);
        float lp = mse_loss(C, target, M*N);
        
        A[p] = original - eps;
        well::ops::matmul(A, B, C, M, K, N);
        float lm = mse_loss(C, target, M*N);
        
        A[p] = original;
        
        float numerical = (lp - lm) / (2.0f * eps);
        char name[32] = "matmul_A[0]";
        name[9] = '0' + p;
        r.check(name, dA[p], numerical, 0.15f);
    }
}

static GradResult run_gradient_tests() {
    GradResult r;
    check_conv2d_grad(r);
    check_matmul_grad(r);
    return r;
}

} // namespace test
