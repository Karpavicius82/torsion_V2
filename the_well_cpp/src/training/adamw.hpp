// ============================================================================
// ADAMW.HPP â€” AdamW Optimizer with Decoupled Weight Decay (Pure C++ / AVX2)
//
// Differences from Adam:
//   1. Weight decay is decoupled from gradient update (Loshchilov & Hutter)
//   2. Warmup schedule (linear ramp over warmup_steps)
//   3. Cosine annealing with warm restarts
//   4. Gradient clipping (max norm)
//   5. GA-tunable: lr_mult, wd_mult, beta1, beta2
// ============================================================================
#pragma once

#include "../models/model.hpp"
#include <immintrin.h>

namespace well {

struct AdamW {
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    float max_grad_norm = 1.0f;

    int step_count = 0;

    // Momentum buffers per parameter
    struct PState {
        Tensor m;   // first moment
        Tensor v;   // second moment
        int size;
    };
    PState* states = nullptr;
    int n_params = 0;

    void init(Model& model, float lr_, float beta1_, float beta2_,
              float eps_, float wd_) {
        lr = lr_; beta1 = beta1_; beta2 = beta2_;
        eps = eps_; weight_decay = wd_;
        step_count = 0;
        n_params = model.num_params();
        states = new PState[n_params];

        for (int i = 0; i < n_params; ++i) {
            Param& p = model.param(i);
            states[i].size = p.size();
            states[i].m = Tensor::alloc(p.size());
            states[i].v = Tensor::alloc(p.size());
            states[i].m.zero();
            states[i].v.zero();
        }
    }

    // Gradient clipping (global norm)
    float clip_grad_norm(Model& model) {
        float total_norm = 0;
        for (int i = 0; i < n_params; ++i) {
            Param& p = model.param(i);
            for (int j = 0; j < p.size(); ++j)
                total_norm += p.grad[j] * p.grad[j];
        }
        total_norm = sqrtf(total_norm);

        if (total_norm > max_grad_norm) {
            float scale = max_grad_norm / (total_norm + 1e-12f);
            for (int i = 0; i < n_params; ++i) {
                Param& p = model.param(i);
                p.grad.scale(scale);
            }
        }
        return total_norm;
    }

    void step(Model& model) {
        step_count++;
        float bc1 = 1.0f - powf(beta1, (float)step_count);
        float bc2 = 1.0f - powf(beta2, (float)step_count);
        float lr_t = lr * sqrtf(bc2) / bc1;

        __m256 vb1 = _mm256_set1_ps(beta1);
        __m256 v1mb1 = _mm256_set1_ps(1.0f - beta1);
        __m256 vb2 = _mm256_set1_ps(beta2);
        __m256 v1mb2 = _mm256_set1_ps(1.0f - beta2);
        __m256 vlr = _mm256_set1_ps(-lr_t);
        __m256 veps = _mm256_set1_ps(eps);
        __m256 vwd = _mm256_set1_ps(1.0f - lr * weight_decay);

        for (int i = 0; i < n_params; ++i) {
            Param& p = model.param(i);
            PState& s = states[i];
            int n = p.size();

            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 vg = _mm256_loadu_ps(p.grad.data + j);
                __m256 vm = _mm256_loadu_ps(s.m.data + j);
                __m256 vv = _mm256_loadu_ps(s.v.data + j);
                __m256 vw = _mm256_loadu_ps(p.weight.data + j);

                // m = Î²1Â·m + (1-Î²1)Â·g
                vm = _mm256_fmadd_ps(vb1, vm, _mm256_mul_ps(v1mb1, vg));
                // v = Î²2Â·v + (1-Î²2)Â·gÂ²
                vv = _mm256_fmadd_ps(vb2, vv, _mm256_mul_ps(v1mb2, _mm256_mul_ps(vg, vg)));

                _mm256_storeu_ps(s.m.data + j, vm);
                _mm256_storeu_ps(s.v.data + j, vv);

                // AdamW: weight decay is decoupled
                // w = w * (1 - lr*wd) - lr_t * m / (sqrt(v) + eps)
                vw = _mm256_mul_ps(vw, vwd);
                __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(vv), veps);
                vw = _mm256_fmadd_ps(vlr, _mm256_div_ps(vm, denom), vw);

                _mm256_storeu_ps(p.weight.data + j, vw);
            }

            // Scalar tail
            for (; j < n; ++j) {
                float g = p.grad[j];
                s.m[j] = beta1 * s.m[j] + (1.0f - beta1) * g;
                s.v[j] = beta2 * s.v[j] + (1.0f - beta2) * g * g;

                p.weight[j] *= (1.0f - lr * weight_decay);
                p.weight[j] -= lr_t * s.m[j] / (sqrtf(s.v[j]) + eps);
            }
        }
    }

    void release() {
        if (states) {
            for (int i = 0; i < n_params; ++i) {
                states[i].m.release();
                states[i].v.release();
            }
            delete[] states;
            states = nullptr;
        }
    }
};

// â”€â”€ LR Schedulers â”€â”€

struct WarmupCosineScheduler {
    float base_lr;
    float min_lr;
    int warmup_steps;
    int total_steps;

    void init(float lr, int total, int warmup, float min_lr_ = 1e-6f) {
        base_lr = lr; total_steps = total;
        warmup_steps = warmup; min_lr = min_lr_;
    }

    float get_lr(int step) const {
        if (step < warmup_steps) {
            // Linear warmup
            return base_lr * (float)(step + 1) / (float)warmup_steps;
        }
        // Cosine decay
        float progress = (float)(step - warmup_steps) /
                         (float)(total_steps - warmup_steps);
        if (progress > 1.0f) progress = 1.0f;
        return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf(3.14159265f * progress));
    }

    void apply(AdamW& opt, int step) {
        opt.lr = get_lr(step);
    }
};

struct CosineWarmRestart {
    float base_lr;
    float min_lr;
    int T_0;      // Initial restart period
    int T_mult;   // Period multiplier after each restart

    void init(float lr, int period, int mult = 2, float min_lr_ = 1e-6f) {
        base_lr = lr; T_0 = period; T_mult = mult; min_lr = min_lr_;
    }

    float get_lr(int step) const {
        // Find current period
        int t_cur = step;
        int t_i = T_0;
        while (t_cur >= t_i) {
            t_cur -= t_i;
            t_i *= T_mult;
        }
        float progress = (float)t_cur / (float)t_i;
        return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf(3.14159265f * progress));
    }

    void apply(AdamW& opt, int step) {
        opt.lr = get_lr(step);
    }
};

} // namespace well
