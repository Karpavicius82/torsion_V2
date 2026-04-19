// ============================================================================
// CONV_NET.HPP — Dilated 1D ConvNet for PDE surrogate modeling
//
// Replaces PolymathicAI's DilatedResNet with pure C++.
//
// Architecture:
//   1. Lifting: [history] → [hidden] per node
//   2. 6× Dilated Conv blocks (dilation 1,2,4,8,4,2) with residual
//   3. Projection: [hidden] → [1]
//
// Captures long-range spatial dependencies via exponentially growing
// receptive field, matching the torsion PDE wavefront propagation.
// ============================================================================
#pragma once

#include "model.hpp"
#include <cmath>

namespace well {

struct ConvNet1D : Model {
    static constexpr int HIDDEN  = 24;
    static constexpr int KERNEL  = 5;
    static constexpr int BLOCKS  = 6;
    static constexpr int DILATIONS[BLOCKS] = {1, 2, 4, 8, 4, 2};

    Param lift;                    // [history, HIDDEN]
    Param proj;                    // [HIDDEN]
    Param conv_w[BLOCKS];          // [HIDDEN * HIDDEN * KERNEL]
    Param conv_b[BLOCKS];          // [HIDDEN]

    Tensor layer_buf;              // [HIDDEN, width]
    Tensor res_buf;                // [HIDDEN, width]

    int history_ = 0, width_ = 0;

    void init(int history, int width, Rng& rng) {
        history_ = history;
        width_ = width;

        float ls = sqrtf(2.0f / (float)(history + HIDDEN));
        lift = Param::alloc(history, HIDDEN);
        for (int i = 0; i < lift.size(); ++i) lift.weight[i] = rng.normal(0, ls);

        float ps = sqrtf(2.0f / (float)HIDDEN);
        proj = Param::alloc(HIDDEN);
        for (int i = 0; i < HIDDEN; ++i) proj.weight[i] = rng.normal(0, ps);

        float cs = sqrtf(2.0f / (float)(HIDDEN * KERNEL));
        for (int b = 0; b < BLOCKS; ++b) {
            conv_w[b] = Param::alloc(HIDDEN * HIDDEN * KERNEL);
            for (int i = 0; i < conv_w[b].size(); ++i)
                conv_w[b].weight[i] = rng.normal(0, cs);
            conv_b[b] = Param::alloc(HIDDEN);
            conv_b[b].weight.zero();
        }

        layer_buf = Tensor::alloc(HIDDEN, width);
        res_buf   = Tensor::alloc(HIDDEN, width);
    }

    const char* name() const override { return "DilatedConvNet-1D"; }

    int param_count() const override {
        int total = lift.size() + proj.size();
        for (int b = 0; b < BLOCKS; ++b)
            total += conv_w[b].size() + conv_b[b].size();
        return total;
    }

    int num_params() const override { return 2 + BLOCKS * 2; }

    Param& param(int idx) override {
        if (idx == 0) return lift;
        if (idx == 1) return proj;
        idx -= 2;
        return (idx % 2 == 0) ? conv_w[idx / 2] : conv_b[idx / 2];
    }

    static inline float relu(float x) { return x > 0 ? x : 0; }

    void forward(const Tensor& input, Tensor& output,
                 int batch, int history, int width) override {
        (void)batch; (void)history;

        // Lifting
        for (int j = 0; j < width; ++j) {
            for (int h = 0; h < HIDDEN; ++h) {
                float sum = 0;
                for (int t = 0; t < history_; ++t)
                    sum += input[t * width + j] * lift.weight.at(t, h);
                layer_buf.data[h * width + j] = relu(sum);
            }
        }

        // Dilated conv blocks with residual
        for (int b = 0; b < BLOCKS; ++b) {
            int dil = DILATIONS[b];

            // Save residual
            memcpy(res_buf.data, layer_buf.data, HIDDEN * width * sizeof(float));

            // Conv1d with dilation
            for (int oc = 0; oc < HIDDEN; ++oc) {
                for (int j = 0; j < width; ++j) {
                    float sum = conv_b[b].weight[oc];
                    for (int ic = 0; ic < HIDDEN; ++ic) {
                        for (int k = 0; k < KERNEL; ++k) {
                            int idx = j + (k - KERNEL / 2) * dil;
                            if (idx >= 0 && idx < width) {
                                sum += res_buf.data[ic * width + idx] *
                                       conv_w[b].weight[(oc * HIDDEN + ic) * KERNEL + k];
                            }
                        }
                    }
                    layer_buf.data[oc * width + j] = sum;
                }
            }

            // ReLU + residual
            for (int i = 0; i < HIDDEN * width; ++i)
                layer_buf.data[i] = relu(layer_buf.data[i]) + res_buf.data[i];
        }

        // Projection
        for (int j = 0; j < width; ++j) {
            float sum = 0;
            for (int h = 0; h < HIDDEN; ++h)
                sum += layer_buf.data[h * width + j] * proj.weight[h];
            output[j] = sum;
        }
    }

    void backward(const Tensor& d_output, const Tensor& input,
                  int batch, int history, int width) override {
        (void)batch; (void)history;

        // Projection gradient
        for (int h = 0; h < HIDDEN; ++h) {
            float g = 0;
            for (int j = 0; j < width; ++j)
                g += d_output[j] * layer_buf.data[h * width + j];
            proj.grad[h] += g;
        }

        // Backprop through layers (simplified: gradient flow through residuals)
        Tensor d_layer = Tensor::alloc(HIDDEN, width);
        for (int h = 0; h < HIDDEN; ++h)
            for (int j = 0; j < width; ++j)
                d_layer.data[h * width + j] = d_output[j] * proj.weight[h];

        // Conv block gradients (approximate for conv weights)
        for (int b = BLOCKS - 1; b >= 0; --b) {
            for (int oc = 0; oc < HIDDEN; ++oc) {
                // Bias gradient
                float bg = 0;
                for (int j = 0; j < width; ++j)
                    bg += d_layer.data[oc * width + j];
                conv_b[b].grad[oc] += bg;

                // Weight gradients (correlation with input)
                int dil = DILATIONS[b];
                for (int ic = 0; ic < HIDDEN; ++ic) {
                    for (int k = 0; k < KERNEL; ++k) {
                        float wg = 0;
                        for (int j = 0; j < width; ++j) {
                            int idx = j + (k - KERNEL / 2) * dil;
                            if (idx >= 0 && idx < width)
                                wg += d_layer.data[oc * width + j] *
                                      layer_buf.data[ic * width + idx];
                        }
                        conv_w[b].grad[(oc * HIDDEN + ic) * KERNEL + k] += wg;
                    }
                }
            }
        }

        // Lifting gradient
        for (int t = 0; t < history_; ++t) {
            for (int h = 0; h < HIDDEN; ++h) {
                float g = 0;
                for (int j = 0; j < width; ++j)
                    g += d_layer.data[h * width + j] * input[t * width + j];
                lift.grad.at(t, h) += g;
            }
        }

        d_layer.release();
    }

    void release() {
        lift.release(); proj.release();
        layer_buf.release(); res_buf.release();
        for (int b = 0; b < BLOCKS; ++b) {
            conv_w[b].release(); conv_b[b].release();
        }
    }
};

constexpr int ConvNet1D::DILATIONS[ConvNet1D::BLOCKS];

} // namespace well
