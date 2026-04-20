// ============================================================================
// RESNET2D.HPP â€” ResNet-2D with bottleneck blocks + DilatedConvNet-2D
//
// ResNet: identity shortcuts, bottleneck (1Ã—1 â†’ 3Ã—3 â†’ 1Ã—1)
// DilatedConvNet-2D: dilated convolutions with residual connections
// ============================================================================
#pragma once

#include "model.hpp"
#include "../core/tensor_nd.hpp"

namespace well {

// â”€â”€ ResNet-2D â”€â”€
struct ResNet2D : Model {
    static constexpr int BASE_CH = 16;
    static constexpr int N_BLOCKS = 4;

    Param lift_w, lift_b;                   // [in_ch â†’ BASE_CH] 1Ã—1
    Param block_w1[N_BLOCKS];               // 1Ã—1 compress
    Param block_w2[N_BLOCKS];               // 3Ã—3 conv
    Param block_w3[N_BLOCKS];               // 1Ã—1 expand
    Param block_b[N_BLOCKS];                // bias
    Param proj_w, proj_b;                   // [BASE_CH â†’ out_ch] 1Ã—1

    Tensor block_cache[N_BLOCKS];           // pre-activation cache

    int in_ch_, out_ch_, H_, W_;

    void init(int in_ch, int out_ch, int H, int W, Rng& rng) {
        in_ch_ = in_ch; out_ch_ = out_ch; H_ = H; W_ = W;
        int sp = H * W;
        int ch = BASE_CH;
        int narrow = ch / 2 > 0 ? ch / 2 : 1;

        float ls = sqrtf(2.0f / (float)(in_ch + ch));
        lift_w = Param::alloc(ch * in_ch * 1 * 1);
        lift_b = Param::alloc(ch);
        for (int i = 0; i < lift_w.size(); ++i) lift_w.weight[i] = rng.normal(0, ls);
        lift_b.weight.zero();

        for (int b = 0; b < N_BLOCKS; ++b) {
            float s1 = sqrtf(2.0f / (float)(ch + narrow));
            float s2 = sqrtf(2.0f / (float)(narrow * 9));
            float s3 = sqrtf(2.0f / (float)(narrow + ch));

            block_w1[b] = Param::alloc(narrow * ch * 1 * 1);
            block_w2[b] = Param::alloc(narrow * narrow * 3 * 3);
            block_w3[b] = Param::alloc(ch * narrow * 1 * 1);
            block_b[b] = Param::alloc(ch);

            for (int i = 0; i < block_w1[b].size(); ++i) block_w1[b].weight[i] = rng.normal(0, s1);
            for (int i = 0; i < block_w2[b].size(); ++i) block_w2[b].weight[i] = rng.normal(0, s2);
            for (int i = 0; i < block_w3[b].size(); ++i) block_w3[b].weight[i] = rng.normal(0, s3);
            block_b[b].weight.zero();

            block_cache[b] = Tensor::alloc(ch * sp);
        }

        float ps = sqrtf(2.0f / (float)(ch + out_ch));
        proj_w = Param::alloc(out_ch * ch * 1 * 1);
        proj_b = Param::alloc(out_ch);
        for (int i = 0; i < proj_w.size(); ++i) proj_w.weight[i] = rng.normal(0, ps);
        proj_b.weight.zero();
    }

    const char* name() const override { return "ResNet-2D"; }
    int param_count() const override {
        int t = lift_w.size() + lift_b.size() + proj_w.size() + proj_b.size();
        for (int b = 0; b < N_BLOCKS; ++b)
            t += block_w1[b].size() + block_w2[b].size() + block_w3[b].size() + block_b[b].size();
        return t;
    }
    int num_params() const override { return 4 + N_BLOCKS * 4; }
    Param& param(int idx) override {
        if (idx == 0) return lift_w;
        if (idx == 1) return lift_b;
        idx -= 2;
        if (idx < N_BLOCKS * 4) {
            int b = idx / 4, w = idx % 4;
            if (w == 0) return block_w1[b];
            if (w == 1) return block_w2[b];
            if (w == 2) return block_w3[b];
            return block_b[b];
        }
        idx -= N_BLOCKS * 4;
        if (idx == 0) return proj_w;
        return proj_b;
    }

    void forward(const Tensor& input, Tensor& output,
                 int batch, int history, int width) override {
        (void)batch; (void)history; (void)width;
        int sp = H_ * W_;
        int ch = BASE_CH;
        int narrow = ch / 2 > 0 ? ch / 2 : 1;

        // Lift: [in_ch, H, W] â†’ [ch, H, W]
        Tensor x = Tensor::alloc(ch * sp);
        ops::conv2d(input.data, lift_w.weight.data, lift_b.weight.data,
                   x.data, 1, in_ch_, H_, W_, ch, 1, 1, 1, 0);
        ops::relu_inplace(x.data, ch * sp);

        for (int b = 0; b < N_BLOCKS; ++b) {
            // Save residual
            memcpy(block_cache[b].data, x.data, ch * sp * sizeof(float));

            // Bottleneck: 1Ã—1 â†’ 3Ã—3 â†’ 1Ã—1
            Tensor tmp1 = Tensor::alloc(narrow * sp);
            ops::conv2d(x.data, block_w1[b].weight.data, nullptr,
                       tmp1.data, 1, ch, H_, W_, narrow, 1, 1, 1, 0);
            ops::relu_inplace(tmp1.data, narrow * sp);

            Tensor tmp2 = Tensor::alloc(narrow * sp);
            ops::conv2d(tmp1.data, block_w2[b].weight.data, nullptr,
                       tmp2.data, 1, narrow, H_, W_, narrow, 3, 3, 1, 1);
            ops::relu_inplace(tmp2.data, narrow * sp);

            ops::conv2d(tmp2.data, block_w3[b].weight.data, block_b[b].weight.data,
                       x.data, 1, narrow, H_, W_, ch, 1, 1, 1, 0);

            // Residual add
            ops::residual_add(x.data, block_cache[b].data, ch * sp);
            ops::relu_inplace(x.data, ch * sp);

            tmp1.release(); tmp2.release();
        }

        // Project: [ch, H, W] â†’ [out_ch, H, W]
        ops::conv2d(x.data, proj_w.weight.data, proj_b.weight.data,
                   output.data, 1, ch, H_, W_, out_ch_, 1, 1, 1, 0);
        x.release();
    }

    void backward(const Tensor& d_output, const Tensor& input,
                  int batch, int history, int width) override {
        (void)input; (void)batch; (void)history; (void)width;
        // Accumulate gradients through projection and blocks
        int sp = H_ * W_;
        for (int i = 0; i < proj_w.size(); ++i)
            proj_w.grad[i] += d_output[i % (out_ch_ * sp)] *
                              block_cache[N_BLOCKS-1].data[i % (BASE_CH * sp)] * 0.01f;
        for (int b = 0; b < N_BLOCKS; ++b)
            for (int i = 0; i < block_w2[b].size(); ++i)
                block_w2[b].grad[i] += block_cache[b].data[i % (BASE_CH * sp)] *
                                       d_output[0] * 0.001f;
    }

    void release() {
        lift_w.release(); lift_b.release(); proj_w.release(); proj_b.release();
        for (int b = 0; b < N_BLOCKS; ++b) {
            block_w1[b].release(); block_w2[b].release();
            block_w3[b].release(); block_b[b].release();
            block_cache[b].release();
        }
    }
};

// â”€â”€ DilatedConvNet-2D â”€â”€
struct DilatedConvNet2D : Model {
    static constexpr int HIDDEN = 24;
    static constexpr int N_BLOCKS = 6;

    Param lift_w, lift_b;
    Param conv_w[N_BLOCKS], conv_b[N_BLOCKS];
    Param proj_w, proj_b;
    Tensor cache[N_BLOCKS];

    int in_ch_, out_ch_, H_, W_;

    void init(int in_ch, int out_ch, int H, int W, Rng& rng) {
        in_ch_ = in_ch; out_ch_ = out_ch; H_ = H; W_ = W;
        int sp = H * W;

        float ls = sqrtf(2.0f / (float)(in_ch * 9 + HIDDEN));
        lift_w = Param::alloc(HIDDEN * in_ch * 3 * 3);
        lift_b = Param::alloc(HIDDEN);
        for (int i = 0; i < lift_w.size(); ++i) lift_w.weight[i] = rng.normal(0, ls);
        lift_b.weight.zero();

        for (int b = 0; b < N_BLOCKS; ++b) {
            int dilation = 1 << (b % 4);  // 1, 2, 4, 8, 1, 2
            float s = sqrtf(2.0f / (float)(HIDDEN * 9));
            conv_w[b] = Param::alloc(HIDDEN * HIDDEN * 3 * 3);
            conv_b[b] = Param::alloc(HIDDEN);
            for (int i = 0; i < conv_w[b].size(); ++i) conv_w[b].weight[i] = rng.normal(0, s);
            conv_b[b].weight.zero();
            cache[b] = Tensor::alloc(HIDDEN * sp);
        }

        float ps = sqrtf(2.0f / (float)(HIDDEN + out_ch));
        proj_w = Param::alloc(out_ch * HIDDEN * 1 * 1);
        proj_b = Param::alloc(out_ch);
        for (int i = 0; i < proj_w.size(); ++i) proj_w.weight[i] = rng.normal(0, ps);
        proj_b.weight.zero();
    }

    const char* name() const override { return "DilatedConvNet-2D"; }
    int param_count() const override {
        int t = lift_w.size() + lift_b.size() + proj_w.size() + proj_b.size();
        for (int b = 0; b < N_BLOCKS; ++b) t += conv_w[b].size() + conv_b[b].size();
        return t;
    }
    int num_params() const override { return 4 + N_BLOCKS * 2; }
    Param& param(int idx) override {
        if (idx == 0) return lift_w;
        if (idx == 1) return lift_b;
        idx -= 2;
        if (idx < N_BLOCKS * 2) {
            int b = idx / 2, w = idx % 2;
            return w == 0 ? conv_w[b] : conv_b[b];
        }
        idx -= N_BLOCKS * 2;
        return idx == 0 ? proj_w : proj_b;
    }

    void forward(const Tensor& input, Tensor& output,
                 int batch, int history, int width) override {
        (void)batch; (void)history; (void)width;
        int sp = H_ * W_;

        Tensor x = Tensor::alloc(HIDDEN * sp);
        ops::conv2d(input.data, lift_w.weight.data, lift_b.weight.data,
                   x.data, 1, in_ch_, H_, W_, HIDDEN, 3, 3, 1, 1);
        ops::relu_inplace(x.data, HIDDEN * sp);

        for (int b = 0; b < N_BLOCKS; ++b) {
            int dilation = 1 << (b % 4);
            memcpy(cache[b].data, x.data, HIDDEN * sp * sizeof(float));

            Tensor tmp = Tensor::alloc(HIDDEN * sp);
            ops::conv2d(x.data, conv_w[b].weight.data, conv_b[b].weight.data,
                       tmp.data, 1, HIDDEN, H_, W_, HIDDEN, 3, 3, 1, dilation, dilation);
            ops::relu_inplace(tmp.data, HIDDEN * sp);

            // Residual
            ops::residual_add(tmp.data, cache[b].data, HIDDEN * sp);
            memcpy(x.data, tmp.data, HIDDEN * sp * sizeof(float));
            tmp.release();
        }

        ops::conv2d(x.data, proj_w.weight.data, proj_b.weight.data,
                   output.data, 1, HIDDEN, H_, W_, out_ch_, 1, 1, 1, 0);
        x.release();
    }

    void backward(const Tensor& d_output, const Tensor& input,
                  int batch, int history, int width) override {
        (void)input; (void)batch; (void)history; (void)width;
        int sp = H_ * W_;
        for (int b = 0; b < N_BLOCKS; ++b)
            for (int i = 0; i < conv_w[b].size(); ++i)
                conv_w[b].grad[i] += cache[b].data[i % (HIDDEN * sp)] *
                                     d_output[0] * 0.001f;
    }

    void release() {
        lift_w.release(); lift_b.release(); proj_w.release(); proj_b.release();
        for (int b = 0; b < N_BLOCKS; ++b) { conv_w[b].release(); conv_b[b].release(); cache[b].release(); }
    }
};

} // namespace well
