// ============================================================================
// COMPAT.HPP — Bare-Metal / Hosted Compatibility Layer
//
// Include THIS instead of <cmath>, <cstring>, <cstdlib>, <cstdio>.
// In BARE_METAL mode: uses bare_metal.hpp (no libc).
// In hosted mode: uses standard libc.
//
// This is the ONLY bridge between the two worlds.
// ============================================================================
#pragma once

#ifdef BARE_METAL
    // ── Bare-metal: everything from our runtime ──
    #include "runtime/bare_metal.hpp"

    // Map libc math to our pure implementations
    #define sqrtf(x)    math::sqrt(x)
    #define sinf(x)     math::sin(x)
    #define cosf(x)     math::cos(x)
    #define tanf(x)     math::tan(x)
    #define expf(x)     math::exp(x)
    #define logf(x)     math::log(x)
    #define tanhf(x)    math::tanh(x)
    #define fabsf(x)    math::abs(x)
    #define fminf(a,b)  math::min(a,b)
    #define fmaxf(a,b)  math::max(a,b)
    #define atan2f(y,x) math::atan2(y,x)
    #define powf(b,e)   math::pow(b,e)

    // C++ std::math fallbacks
    namespace std {
        using ::size_t;
        static inline float sqrt(float x) { return math::sqrt(x); }
        static inline float sin(float x)  { return math::sin(x); }
        static inline float cos(float x)  { return math::cos(x); }
        static inline float tan(float x)  { return math::tan(x); }
        static inline float exp(float x)  { return math::exp(x); }
        static inline float log(float x)  { return math::log(x); }
        static inline float tanh(float x) { return math::tanh(x); }
        static inline float abs(float x)  { return math::abs(x); }
        static inline float fabs(float x) { return math::abs(x); }
        static inline float pow(float b, float e) { return math::pow(b,e); }
        static inline float atan2(float y, float x) { return math::atan2(y,x); }
        static inline float fmin(float a, float b) { return math::min(a,b); }
        static inline float fmax(float a, float b) { return math::max(a,b); }
    }

    // libc IO stubs
    #define fprintf(...) ((void)0)
    #define printf(...)  ((void)0)
    #define fflush(...)  ((void)0)
    #define fputc(...)   ((void)0)

    // libc memory: provided by bare_metal.hpp extern "C" memcpy/memset/memcmp/strcmp/strlen

    // Allocator
    static inline void* bare_alloc(size_t sz, size_t align = 32) { return mem::alloc(sz, align); }
    static inline void  bare_free(void*) {} // bump allocator, no free

    // Replace malloc/free/aligned
    #define malloc(sz)         mem::alloc(sz)
    #define free(p)            ((void)0)
    #define _aligned_malloc(sz,al) mem::alloc(sz,al)
    #define _aligned_free(p)   ((void)0)
    #define aligned_alloc(al,sz) mem::alloc(sz,al)

    // atoi/atof stubs (not needed in bare-metal, kernel has no CLI)
    static inline int atoi(const char*) { return 0; }
    static inline float atof(const char*) { return 0.0f; }

#else
    // ── Hosted mode: standard libc ──
    #include <cstdint>
    #include <cstdlib>
    #include <cstring>
    #include <cmath>
    #include <cstdio>
    #include <immintrin.h>

    #ifdef _WIN32
        #include <malloc.h>
        static inline void* bare_alloc(size_t sz, size_t align = 32) { return _aligned_malloc(sz, align); }
        static inline void  bare_free(void* p) { _aligned_free(p); }
    #else
        static inline void* bare_alloc(size_t sz, size_t align = 32) { return aligned_alloc(align, sz); }
        static inline void  bare_free(void* p) { free(p); }
    #endif
#endif
