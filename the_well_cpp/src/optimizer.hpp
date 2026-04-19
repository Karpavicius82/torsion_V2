// ============================================================================
// OPTIMIZER.HPP — Adam optimizer (pure C++ / AVX2)
//
// Replaces PyTorch's torch.optim.Adam.
// Implements bias-corrected first and second moment estimates.
// AVX2-accelerated parameter updates.
// ============================================================================
#pragma once

#include "tensor.hpp"
#include "models/model.hpp"
#include <cmath>

namespace well {

struct Adam {
    float lr;
    float beta1, beta2, eps;
    float weight_decay;
    int   step_count;

    // Per-parameter moment buffers
    struct Moment {
        Tensor m;  // first moment (mean)
        Tensor v;  // second moment (variance)
    };
    Moment* moments;
    int n_params;

    void init(Model& model, float learning_rate = 1e-3f,
              float b1 = 0.9f, float b2 = 0.999f,
              float epsilon = 1e-8f, float wd = 0.0f) {
        lr = learning_rate;
        beta1 = b1; beta2 = b2; eps = epsilon;
        weight_decay = wd;
        step_count = 0;

        n_params = model.num_params();
        moments = new Moment[n_params];
        for (int i = 0; i < n_params; ++i) {
            int sz = model.param(i).size();
            moments[i].m = Tensor::alloc(sz);
            moments[i].v = Tensor::alloc(sz);
        }
    }

    void step(Model& model) {
        step_count++;
        float bc1 = 1.0f - powf(beta1, (float)step_count);
        float bc2 = 1.0f - powf(beta2, (float)step_count);
        float lr_t = lr * sqrtf(bc2) / bc1;

        __m256 vb1 = _mm256_set1_ps(beta1);
        __m256 vb2 = _mm256_set1_ps(beta2);
        __m256 v1mb1 = _mm256_set1_ps(1.0f - beta1);
        __m256 v1mb2 = _mm256_set1_ps(1.0f - beta2);
        __m256 vlr = _mm256_set1_ps(-lr_t);
        __m256 veps = _mm256_set1_ps(eps);
        __m256 vwd = _mm256_set1_ps(weight_decay);

        for (int p = 0; p < n_params; ++p) {
            Param& par = model.param(p);
            Moment& mom = moments[p];
            int sz = par.size();

            float* w = par.weight.data;
            float* g = par.grad.data;
            float* m = mom.m.data;
            float* v = mom.v.data;

            int i = 0;
            for (; i + 8 <= sz; i += 8) {
                __m256 vg = _mm256_loadu_ps(g + i);
                __m256 vw = _mm256_loadu_ps(w + i);

                // Weight decay
                if (weight_decay > 0)
                    vg = _mm256_fmadd_ps(vwd, vw, vg);

                // m = beta1 * m + (1 - beta1) * g
                __m256 vm = _mm256_loadu_ps(m + i);
                vm = _mm256_fmadd_ps(vb1, vm, _mm256_mul_ps(v1mb1, vg));
                _mm256_storeu_ps(m + i, vm);

                // v = beta2 * v + (1 - beta2) * g^2
                __m256 vv = _mm256_loadu_ps(v + i);
                vv = _mm256_fmadd_ps(vb2, vv, _mm256_mul_ps(v1mb2, _mm256_mul_ps(vg, vg)));
                _mm256_storeu_ps(v + i, vv);

                // w += -lr_t * m / (sqrt(v) + eps)
                __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(vv), veps);
                __m256 update = _mm256_mul_ps(vlr, _mm256_div_ps(vm, denom));
                _mm256_storeu_ps(w + i, _mm256_add_ps(vw, update));
            }

            // Scalar tail
            for (; i < sz; ++i) {
                float gi = g[i];
                if (weight_decay > 0) gi += weight_decay * w[i];
                m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
                v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
                w[i] += -lr_t * m[i] / (sqrtf(v[i]) + eps);
            }
        }
    }

    void release() {
        for (int i = 0; i < n_params; ++i) {
            moments[i].m.release();
            moments[i].v.release();
        }
        delete[] moments;
        moments = nullptr;
    }
};

// ── Cosine Annealing LR Scheduler ──
struct CosineScheduler {
    float base_lr, min_lr;
    int total_epochs;
    int warmup_epochs;

    void init(float lr, int epochs, int warmup = 5, float min = 1e-6f) {
        base_lr = lr; min_lr = min;
        total_epochs = epochs; warmup_epochs = warmup;
    }

    float get_lr(int epoch) const {
        if (epoch < warmup_epochs) {
            // Linear warmup
            return base_lr * (float)(epoch + 1) / (float)warmup_epochs;
        }
        // Cosine decay
        float progress = (float)(epoch - warmup_epochs) /
                          (float)(total_epochs - warmup_epochs);
        return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf(3.14159265f * progress));
    }

    void apply(Adam& opt, int epoch) {
        opt.lr = get_lr(epoch);
    }
};

} // namespace well
