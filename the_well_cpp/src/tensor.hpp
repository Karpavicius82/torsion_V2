// ============================================================================
// TENSOR.HPP — Minimal N-D tensor for bare-metal ML
//
// Layout: contiguous float32 buffer with shape metadata.
// Supports: 1D, 2D, 3D, 4D.  No heap tricks, no RTTI.
// Aligned to 32 bytes for AVX2.
// ============================================================================
#pragma once

#include "compat.hpp"

static inline void* aligned_alloc_impl(size_t align, size_t sz) { return bare_alloc(sz, align); }
static inline void  aligned_free_impl(void* p) { bare_free(p); }

namespace well {

struct Tensor {
    float*   data   = nullptr;
    int      shape[4] = {0,0,0,0};
    int      ndim   = 0;
    int      size_  = 0;   // total elements
    bool     owned  = false;

    Tensor() = default;

    // Allocate
    static Tensor alloc(int d0, int d1 = 1, int d2 = 1, int d3 = 1) {
        Tensor t;
        t.shape[0] = d0; t.shape[1] = d1; t.shape[2] = d2; t.shape[3] = d3;
        t.ndim = (d3 > 1) ? 4 : (d2 > 1) ? 3 : (d1 > 1) ? 2 : 1;
        t.size_ = d0 * d1 * d2 * d3;
        // Pad to multiple of 8 for AVX2
        int alloc_n = (t.size_ + 7) & ~7;
        t.data = (float*)aligned_alloc_impl(32, alloc_n * sizeof(float));
        memset(t.data, 0, alloc_n * sizeof(float));
        t.owned = true;
        return t;
    }

    // View (non-owning)
    static Tensor view(float* ptr, int d0, int d1 = 1, int d2 = 1, int d3 = 1) {
        Tensor t;
        t.data = ptr;
        t.shape[0] = d0; t.shape[1] = d1; t.shape[2] = d2; t.shape[3] = d3;
        t.ndim = (d3 > 1) ? 4 : (d2 > 1) ? 3 : (d1 > 1) ? 2 : 1;
        t.size_ = d0 * d1 * d2 * d3;
        t.owned = false;
        return t;
    }

    void release() {
        if (owned && data) { aligned_free_impl(data); data = nullptr; }
    }

    // Accessors
    float& operator[](int i) { return data[i]; }
    float  operator[](int i) const { return data[i]; }
    float& at(int i, int j) { return data[i * shape[1] + j]; }
    float  at(int i, int j) const { return data[i * shape[1] + j]; }
    int    size() const { return size_; }

    // ── Fill ──
    void zero() { memset(data, 0, size_ * sizeof(float)); }
    void fill(float v) { for (int i = 0; i < size_; ++i) data[i] = v; }

    // ── BLAS-like ops (AVX2) ──

    // y = alpha * x + y (AXPY)
    void axpy(float alpha, const Tensor& x) {
        __m256 va = _mm256_set1_ps(alpha);
        int i = 0;
        for (; i + 8 <= size_; i += 8) {
            __m256 vx = _mm256_loadu_ps(x.data + i);
            __m256 vy = _mm256_loadu_ps(data + i);
            _mm256_storeu_ps(data + i, _mm256_fmadd_ps(va, vx, vy));
        }
        for (; i < size_; ++i) data[i] += alpha * x.data[i];
    }

    // dot product
    float dot(const Tensor& x) const {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= size_; i += 8) {
            __m256 va = _mm256_loadu_ps(data + i);
            __m256 vb = _mm256_loadu_ps(x.data + i);
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        float result = _mm_cvtss_f32(s);
        for (; i < size_; ++i) result += data[i] * x.data[i];
        return result;
    }

    // MSE loss between this and target
    float mse(const Tensor& target) const {
        float sum = 0;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= size_; i += 8) {
            __m256 a = _mm256_loadu_ps(data + i);
            __m256 b = _mm256_loadu_ps(target.data + i);
            __m256 d = _mm256_sub_ps(a, b);
            acc = _mm256_fmadd_ps(d, d, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        sum = _mm_cvtss_f32(s);
        for (; i < size_; ++i) { float d = data[i] - target.data[i]; sum += d * d; }
        return sum / size_;
    }

    // Element-wise scale
    void scale(float s) {
        __m256 vs = _mm256_set1_ps(s);
        int i = 0;
        for (; i + 8 <= size_; i += 8)
            _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), vs));
        for (; i < size_; ++i) data[i] *= s;
    }

    // Copy from another tensor
    void copy_from(const Tensor& src) {
        memcpy(data, src.data, size_ * sizeof(float));
    }

    // L2 norm
    float norm() const { return sqrtf(dot(*this)); }
};

// ── Simple PRNG for weight init ──
struct Rng {
    uint32_t s[4];
    void seed(uint64_t v) {
        s[0]=(uint32_t)(v^0xDEADBEEF); s[1]=(uint32_t)(v>>16);
        s[2]=(uint32_t)(v>>32); s[3]=(uint32_t)(v>>48)^0xCAFE;
    }
    uint32_t next() {
        auto rotl=[](uint32_t x,int k){return(x<<k)|(x>>(32-k));};
        uint32_t r=rotl(s[1]*5,7)*9, t=s[1]<<9;
        s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3];
        s[2]^=t; s[3]=rotl(s[3],11); return r;
    }
    // Uniform [-1, 1]
    float uniform() { return (float)(int32_t)next() / 2147483648.0f; }
    // Normal (Box-Muller)
    float normal(float mean = 0.0f, float std = 1.0f) {
        float u1 = (next() & 0x7FFFFFFFU) / 2147483648.0f + 1e-7f;
        float u2 = (next() & 0x7FFFFFFFU) / 2147483648.0f;
        return mean + std * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
};

} // namespace well
