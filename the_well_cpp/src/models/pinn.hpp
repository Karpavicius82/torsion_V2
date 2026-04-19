// ============================================================================
// PINN.HPP â€” Physics-Informed Neural Network (Pure C++)
//
// Architecture: MLP with physics residual in loss function
// Loss = Î± * MSE(pred, target) + Î² * PDE_residual(pred)
// PDE residual computed via finite differences on the predicted field
// GA evolves: Î±/Î² ratio, hidden size, learning rate multiplier
// ============================================================================
#pragma once

#include "model.hpp"
#include "../core/tensor_nd.hpp"

namespace well {

struct PINN : Model {
    static constexpr int HIDDEN = 64;
    static constexpr int N_LAYERS = 4;

    Param weights[N_LAYERS];
    Param biases[N_LAYERS];
    Tensor layer_cache[N_LAYERS];

    int in_dim_, out_dim_;
    int H_, W_;

    // Physics loss weight (GA-tunable)
    float physics_weight = 0.1f;

    void init(int in_ch, int out_ch, int H, int W, Rng& rng) {
        in_dim_ = in_ch * H * W;
        out_dim_ = out_ch * H * W;
        H_ = H; W_ = W;

        // Layer sizes: in_dim â†’ HIDDEN â†’ HIDDEN â†’ HIDDEN â†’ out_dim
        int layer_sizes[N_LAYERS + 1];
        layer_sizes[0] = in_dim_;
        for (int l = 1; l < N_LAYERS; ++l) layer_sizes[l] = HIDDEN;
        layer_sizes[N_LAYERS] = out_dim_;

        for (int l = 0; l < N_LAYERS; ++l) {
            int fan_in = layer_sizes[l], fan_out = layer_sizes[l + 1];
            float std = sqrtf(2.0f / (float)(fan_in + fan_out));
            weights[l] = Param::alloc(fan_in, fan_out);
            biases[l] = Param::alloc(fan_out);
            for (int i = 0; i < weights[l].size(); ++i)
                weights[l].weight[i] = rng.normal(0, std);
            biases[l].weight.zero();
            layer_cache[l] = Tensor::alloc(fan_out);
        }
    }

    const char* name() const override { return "PINN"; }

    int param_count() const override {
        int t = 0;
        for (int l = 0; l < N_LAYERS; ++l) t += weights[l].size() + biases[l].size();
        return t;
    }
    int num_params() const override { return N_LAYERS * 2; }
    Param& param(int idx) override {
        int l = idx / 2, w = idx % 2;
        return w == 0 ? weights[l] : biases[l];
    }

    void forward(const Tensor& input, Tensor& output,
                 int batch, int history, int width) override {
        (void)batch; (void)history; (void)width;

        // Flatten input
        const float* x = input.data;
        int current_dim = in_dim_;

        // For single-sample, iterate through MLP layers
        Tensor act = Tensor::alloc(in_dim_);
        memcpy(act.data, x, in_dim_ * sizeof(float));

        for (int l = 0; l < N_LAYERS; ++l) {
            int fan_out = (l < N_LAYERS - 1) ? HIDDEN : out_dim_;
            Tensor next = Tensor::alloc(fan_out);

            // Linear: next = act Ã— W + b
            ops::matmul(act.data, weights[l].weight.data, next.data,
                       1, current_dim, fan_out);
            for (int i = 0; i < fan_out; ++i)
                next[i] += biases[l].weight[i];

            // Cache pre-activation
            memcpy(layer_cache[l].data, next.data, fan_out * sizeof(float));

            // Activation (GELU for hidden, linear for output)
            if (l < N_LAYERS - 1)
                ops::gelu_inplace(next.data, fan_out);

            act.release();
            act = next;
            current_dim = fan_out;
        }

        memcpy(output.data, act.data, out_dim_ * sizeof(float));
        act.release();
    }

    // Compute PDE residual: âˆ‚Â²u/âˆ‚xÂ² + âˆ‚Â²u/âˆ‚yÂ² (Laplacian)
    // Used as additional physics loss term
    float pde_residual(const Tensor& pred, float dx = 1.0f, float dy = 1.0f) const {
        int H = H_, W = W_;
        float inv_dx2 = 1.0f / (dx * dx);
        float inv_dy2 = 1.0f / (dy * dy);
        float residual = 0;

        for (int j = 1; j < H - 1; ++j) {
            for (int i = 1; i < W - 1; ++i) {
                float lap = (pred[j*W+i-1] + pred[j*W+i+1] - 2.0f*pred[j*W+i]) * inv_dx2
                          + (pred[(j-1)*W+i] + pred[(j+1)*W+i] - 2.0f*pred[j*W+i]) * inv_dy2;
                residual += lap * lap;
            }
        }
        return residual / (float)((H-2) * (W-2));
    }

    void backward(const Tensor& d_output, const Tensor& input,
                  int batch, int history, int width) override {
        (void)batch; (void)history; (void)width;

        // Backprop through MLP
        // Output layer gradient
        int l = N_LAYERS - 1;
        Tensor d_act = Tensor::alloc(out_dim_);
        memcpy(d_act.data, d_output.data, out_dim_ * sizeof(float));

        for (int ll = N_LAYERS - 1; ll >= 0; --ll) {
            int fan_in = (ll == 0) ? in_dim_ : HIDDEN;
            int fan_out = (ll < N_LAYERS - 1) ? HIDDEN : out_dim_;

            // GELU backward (hidden layers) or identity (output)
            if (ll < N_LAYERS - 1) {
                for (int i = 0; i < fan_out; ++i)
                    d_act[i] *= ops::gelu_deriv(layer_cache[ll].data[i]);
            }

            // Bias gradient
            for (int i = 0; i < fan_out; ++i)
                biases[ll].grad[i] += d_act[i];

            // Weight gradient: d_W += x^T Ã— d_act
            const float* x_in = (ll == 0) ? input.data : layer_cache[ll > 0 ? ll - 1 : 0].data;
            for (int i = 0; i < fan_in; ++i)
                for (int j = 0; j < fan_out; ++j)
                    weights[ll].grad.at(i, j) += x_in[i] * d_act[j];

            // Propagate gradient: d_prev = d_act Ã— W^T
            if (ll > 0) {
                Tensor d_prev = Tensor::alloc(fan_in);
                for (int i = 0; i < fan_in; ++i) {
                    float sum = 0;
                    for (int j = 0; j < fan_out; ++j)
                        sum += d_act[j] * weights[ll].weight.at(i, j);
                    d_prev[i] = sum;
                }
                d_act.release();
                d_act = d_prev;
            }
        }
        d_act.release();
    }

    void release() {
        for (int l = 0; l < N_LAYERS; ++l) {
            weights[l].release(); biases[l].release();
            layer_cache[l].release();
        }
    }
};

} // namespace well
