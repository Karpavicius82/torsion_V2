// ============================================================================
// TENSOR_ND.HPP — N-D Tensor Operations for 2D/3D Physics
//
// Extends base Tensor with:
//   conv2d, conv2d_backward, matmul, pool2d, upsample2d, batchnorm,
//   transpose, reshape, softmax, layer_norm, add_bias, relu, gelu.
//
// All ops work on flat float32 buffers with explicit shape parameters.
// AVX2 SIMD where beneficial. Header-only.
// ============================================================================
#pragma once

#include "../tensor.hpp"

namespace well {
namespace ops {

// ── 2D Convolution (NCHW layout) ──
// input:  [batch, in_ch, H, W]
// kernel: [out_ch, in_ch, kH, kW]
// output: [batch, out_ch, oH, oW]
// oH = (H + 2*pad - kH) / stride + 1
static void conv2d(const float* input, const float* kernel, const float* bias,
                   float* output,
                   int batch, int in_ch, int H, int W,
                   int out_ch, int kH, int kW,
                   int stride = 1, int pad = 0, int dilation = 1) {
    int oH = (H + 2*pad - dilation*(kH-1) - 1) / stride + 1;
    int oW = (W + 2*pad - dilation*(kW-1) - 1) / stride + 1;

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_ch; ++oc) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float sum = bias ? bias[oc] : 0.0f;

                    for (int ic = 0; ic < in_ch; ++ic) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                int ih = oh * stride - pad + kh * dilation;
                                int iw = ow * stride - pad + kw * dilation;

                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int in_idx = ((b * in_ch + ic) * H + ih) * W + iw;
                                    int k_idx = ((oc * in_ch + ic) * kH + kh) * kW + kw;
                                    sum += input[in_idx] * kernel[k_idx];
                                }
                            }
                        }
                    }

                    output[((b * out_ch + oc) * oH + oh) * oW + ow] = sum;
                }
            }
        }
    }
}

// ── Conv2D backward (data gradient) ──
static void conv2d_backward_data(const float* d_output, const float* kernel,
                                  float* d_input,
                                  int batch, int in_ch, int H, int W,
                                  int out_ch, int kH, int kW,
                                  int stride, int pad, int dilation) {
    int oH = (H + 2*pad - dilation*(kH-1) - 1) / stride + 1;
    int oW = (W + 2*pad - dilation*(kW-1) - 1) / stride + 1;

    memset(d_input, 0, batch * in_ch * H * W * sizeof(float));

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_ch; ++oc) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float d_out = d_output[((b * out_ch + oc) * oH + oh) * oW + ow];

                    for (int ic = 0; ic < in_ch; ++ic) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                int ih = oh * stride - pad + kh * dilation;
                                int iw = ow * stride - pad + kw * dilation;

                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int k_idx = ((oc * in_ch + ic) * kH + kh) * kW + kw;
                                    d_input[((b * in_ch + ic) * H + ih) * W + iw] +=
                                        d_out * kernel[k_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Conv2D backward (weight gradient) ──
static void conv2d_backward_weight(const float* d_output, const float* input,
                                    float* d_kernel, float* d_bias,
                                    int batch, int in_ch, int H, int W,
                                    int out_ch, int kH, int kW,
                                    int stride, int pad, int dilation) {
    int oH = (H + 2*pad - dilation*(kH-1) - 1) / stride + 1;
    int oW = (W + 2*pad - dilation*(kW-1) - 1) / stride + 1;

    memset(d_kernel, 0, out_ch * in_ch * kH * kW * sizeof(float));
    if (d_bias) memset(d_bias, 0, out_ch * sizeof(float));

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_ch; ++oc) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float d_out = d_output[((b * out_ch + oc) * oH + oh) * oW + ow];
                    if (d_bias) d_bias[oc] += d_out;

                    for (int ic = 0; ic < in_ch; ++ic) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                int ih = oh * stride - pad + kh * dilation;
                                int iw = ow * stride - pad + kw * dilation;

                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int in_idx = ((b * in_ch + ic) * H + ih) * W + iw;
                                    int k_idx = ((oc * in_ch + ic) * kH + kh) * kW + kw;
                                    d_kernel[k_idx] += d_out * input[in_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Matrix multiply: C[M,N] = A[M,K] × B[K,N] ──
// AVX2 accelerated inner loop
static void matmul(const float* A, const float* B, float* C,
                   int M, int K, int N) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];
            __m256 va = _mm256_set1_ps(a_ik);
            int j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 vc = _mm256_loadu_ps(C + i * N + j);
                __m256 vb = _mm256_loadu_ps(B + k * N + j);
                _mm256_storeu_ps(C + i * N + j, _mm256_fmadd_ps(va, vb, vc));
            }
            for (; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// ── Matmul backward ──
// d_A[M,K] = d_C[M,N] × B^T[N,K]
// d_B[K,N] = A^T[K,M] × d_C[M,N]
static void matmul_backward(const float* d_C, const float* A, const float* B,
                             float* d_A, float* d_B, int M, int K, int N) {
    // d_A = d_C × B^T
    if (d_A) {
        memset(d_A, 0, M * K * sizeof(float));
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float dc = d_C[i * N + j];
                for (int k = 0; k < K; ++k)
                    d_A[i * K + k] += dc * B[k * N + j];
            }
    }
    // d_B = A^T × d_C
    if (d_B) {
        memset(d_B, 0, K * N * sizeof(float));
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < M; ++i) {
                float a = A[i * K + k];
                for (int j = 0; j < N; ++j)
                    d_B[k * N + j] += a * d_C[i * N + j];
            }
    }
}

// ── 2D Max Pooling (NCHW) ──
static void pool2d(const float* input, float* output, int* indices,
                   int batch, int ch, int H, int W,
                   int pool_h, int pool_w, int stride) {
    int oH = (H - pool_h) / stride + 1;
    int oW = (W - pool_w) / stride + 1;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < ch; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float mx = -1e30f;
                    int mx_idx = 0;

                    for (int ph = 0; ph < pool_h; ++ph) {
                        for (int pw = 0; pw < pool_w; ++pw) {
                            int ih = oh * stride + ph, iw = ow * stride + pw;
                            int idx = ((b * ch + c) * H + ih) * W + iw;
                            if (input[idx] > mx) { mx = input[idx]; mx_idx = idx; }
                        }
                    }

                    int out_idx = ((b * ch + c) * oH + oh) * oW + ow;
                    output[out_idx] = mx;
                    if (indices) indices[out_idx] = mx_idx;
                }
            }
        }
    }
}

// ── 2D Nearest-Neighbor Upsample (NCHW) ──
static void upsample2d(const float* input, float* output,
                       int batch, int ch, int H, int W, int scale) {
    int oH = H * scale, oW = W * scale;
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < ch; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    int ih = oh / scale, iw = ow / scale;
                    output[((b * ch + c) * oH + oh) * oW + ow] =
                        input[((b * ch + c) * H + ih) * W + iw];
                }
            }
        }
    }
}

// ── Batch Normalization (NCHW, inference-style with running stats) ──
static void batchnorm2d(float* data, const float* gamma, const float* beta,
                        int batch, int ch, int H, int W, float eps = 1e-5f) {
    int spatial = H * W;

    for (int c = 0; c < ch; ++c) {
        // Compute mean and variance across batch and spatial
        float mean = 0, var = 0;
        int count = batch * spatial;

        for (int b = 0; b < batch; ++b) {
            const float* ptr = data + (b * ch + c) * spatial;
            for (int i = 0; i < spatial; ++i) mean += ptr[i];
        }
        mean /= (float)count;

        for (int b = 0; b < batch; ++b) {
            const float* ptr = data + (b * ch + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                float d = ptr[i] - mean;
                var += d * d;
            }
        }
        var /= (float)count;

        float inv_std = 1.0f / sqrtf(var + eps);
        float g = gamma ? gamma[c] : 1.0f;
        float b_ = beta ? beta[c] : 0.0f;

        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vinv = _mm256_set1_ps(inv_std * g);
        __m256 vbeta = _mm256_set1_ps(b_);

        for (int b = 0; b < batch; ++b) {
            float* ptr = data + (b * ch + c) * spatial;
            int i = 0;
            for (; i + 8 <= spatial; i += 8) {
                __m256 v = _mm256_loadu_ps(ptr + i);
                v = _mm256_sub_ps(v, vmean);
                v = _mm256_fmadd_ps(v, vinv, vbeta);
                _mm256_storeu_ps(ptr + i, v);
            }
            for (; i < spatial; ++i) {
                ptr[i] = (ptr[i] - mean) * inv_std * g + b_;
            }
        }
    }
}

// ── Layer Normalization ──
static void layer_norm(float* data, const float* gamma, const float* beta,
                       int n, float eps = 1e-5f) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; ++i) mean += data[i];
    mean /= (float)n;
    for (int i = 0; i < n; ++i) { float d = data[i] - mean; var += d * d; }
    var /= (float)n;
    float inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < n; ++i) {
        float g = gamma ? gamma[i] : 1.0f;
        float b = beta ? beta[i] : 0.0f;
        data[i] = (data[i] - mean) * inv_std * g + b;
    }
}

// ── Activations ──
static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
static inline float relu_deriv(float x) { return x > 0.0f ? 1.0f : 0.0f; }

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
static inline float gelu_deriv(float x) {
    float t = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
    float s = 0.7978845608f * (1.0f + 0.134145f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * s;
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// ── ReLU in-place (AVX2) ──
static void relu_inplace(float* data, int n) {
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_max_ps(v, zero));
    }
    for (; i < n; ++i) data[i] = data[i] > 0 ? data[i] : 0;
}

// ── GELU in-place ──
static void gelu_inplace(float* data, int n) {
    for (int i = 0; i < n; ++i) data[i] = gelu(data[i]);
}

// ── Softmax over last dimension ──
static void softmax(float* data, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = data + r * cols;
        float mx = row[0];
        for (int c = 1; c < cols; ++c) if (row[c] > mx) mx = row[c];
        float sum = 0;
        for (int c = 0; c < cols; ++c) { row[c] = expf(row[c] - mx); sum += row[c]; }
        float inv = 1.0f / sum;
        for (int c = 0; c < cols; ++c) row[c] *= inv;
    }
}

// ── Add bias to each channel (NCHW) ──
static void add_bias_2d(float* data, const float* bias,
                        int batch, int ch, int H, int W) {
    int spatial = H * W;
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < ch; ++c) {
            float bv = bias[c];
            __m256 vb = _mm256_set1_ps(bv);
            float* ptr = data + (b * ch + c) * spatial;
            int i = 0;
            for (; i + 8 <= spatial; i += 8) {
                __m256 v = _mm256_loadu_ps(ptr + i);
                _mm256_storeu_ps(ptr + i, _mm256_add_ps(v, vb));
            }
            for (; i < spatial; ++i) ptr[i] += bv;
        }
    }
}

// ── Residual add: out[i] += x[i] (AVX2) ──
static void residual_add(float* out, const float* x, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vo = _mm256_loadu_ps(out + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(vo, vx));
    }
    for (; i < n; ++i) out[i] += x[i];
}

} // namespace ops
} // namespace well
