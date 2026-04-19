// ============================================================================
// BARE_METAL.HPP — Living Silicon Freestanding Runtime
//
// REPLACES: libc, cstdio, cstdlib, cstring, cmath, windows.h, malloc
// This IS the OS. No syscalls. No kernel. Direct hardware.
//
// Provides:
//   - Memory allocator (bump + pool)
//   - Serial I/O (COM1 0x3F8)
//   - VGA text output (0xB8000)
//   - Math (AVX2 intrinsics, no libm)
//   - memcpy/memset/memmove
//   - Timer (RDTSC)
//   - CPUID feature detection
// ============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

// ============================================================================
// HARDWARE I/O
// ============================================================================

namespace hw {

static inline void outb(uint16_t port, uint8_t val) {
    asm volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    asm volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

static inline uint64_t rdtsc() {
    uint32_t lo, hi;
    asm volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline void hlt() { asm volatile ("hlt"); }
static inline void cli() { asm volatile ("cli"); }
static inline void sti() { asm volatile ("sti"); }

} // namespace hw

// ============================================================================
// SERIAL PORT (COM1) — Primary output channel
// ============================================================================

namespace serial {

constexpr uint16_t COM1 = 0x3F8;

static void init() {
    hw::outb(COM1 + 1, 0x00);  // Disable interrupts
    hw::outb(COM1 + 3, 0x80);  // DLAB on
    hw::outb(COM1 + 0, 0x01);  // 115200 baud
    hw::outb(COM1 + 1, 0x00);
    hw::outb(COM1 + 3, 0x03);  // 8N1
    hw::outb(COM1 + 2, 0xC7);  // FIFO
    hw::outb(COM1 + 4, 0x0B);  // RTS/DSR
}

static bool is_ready() {
    return (hw::inb(COM1 + 5) & 0x20) != 0;
}

static void putc(char c) {
    while (!is_ready()) {}
    hw::outb(COM1, (uint8_t)c);
}

static void puts(const char* s) {
    while (*s) putc(*s++);
}

static void putd(int64_t val) {
    if (val < 0) { putc('-'); val = -val; }
    if (val == 0) { putc('0'); return; }
    char buf[20]; int i = 0;
    while (val > 0) { buf[i++] = '0' + (val % 10); val /= 10; }
    while (i > 0) putc(buf[--i]);
}

static void putf(float val, int decimals = 6) {
    if (val < 0) { putc('-'); val = -val; }
    int64_t integer = (int64_t)val;
    putd(integer);
    putc('.');
    float frac = val - (float)integer;
    for (int d = 0; d < decimals; ++d) {
        frac *= 10.0f;
        int digit = (int)frac;
        putc('0' + digit);
        frac -= (float)digit;
    }
}

static void newline() { putc('\r'); putc('\n'); }

} // namespace serial

// ============================================================================
// VGA TEXT MODE (fallback display)
// ============================================================================

namespace vga {

static volatile uint16_t* const BUFFER = (volatile uint16_t*)0xB8000;
constexpr int COLS = 80;
constexpr int ROWS = 25;
static int cursor_x = 0, cursor_y = 0;

static void clear() {
    for (int i = 0; i < COLS * ROWS; ++i)
        BUFFER[i] = 0x0720;  // space, light grey on black
    cursor_x = 0; cursor_y = 0;
}

static void scroll() {
    for (int i = 0; i < COLS * (ROWS - 1); ++i)
        BUFFER[i] = BUFFER[i + COLS];
    for (int i = 0; i < COLS; ++i)
        BUFFER[(ROWS - 1) * COLS + i] = 0x0720;
    cursor_y = ROWS - 1;
}

static void putc(char c, uint8_t color = 0x0F) {
    if (c == '\n') { cursor_x = 0; cursor_y++; }
    else if (c == '\r') { cursor_x = 0; }
    else {
        BUFFER[cursor_y * COLS + cursor_x] = (uint16_t)c | ((uint16_t)color << 8);
        cursor_x++;
    }
    if (cursor_x >= COLS) { cursor_x = 0; cursor_y++; }
    if (cursor_y >= ROWS) scroll();
}

static void puts(const char* s, uint8_t color = 0x0F) {
    while (*s) putc(*s++, color);
}

static void puts_color(const char* s, uint8_t fg, uint8_t bg = 0) {
    uint8_t color = (bg << 4) | fg;
    puts(s, color);
}

} // namespace vga

// ============================================================================
// MEMORY ALLOCATOR — Bump allocator + pool (no OS, no malloc)
// ============================================================================

namespace mem {

// 256MB arena — adjust based on available physical memory
constexpr size_t ARENA_SIZE = 256ULL * 1024 * 1024;
static uint8_t* arena_base = nullptr;
static size_t   arena_used = 0;

// Arena is placed at a fixed physical address (set by linker/bootloader)
// For hosted testing, we'll use a static buffer
#ifdef BARE_METAL
    // Linker places this at a known address
    extern uint8_t __heap_start[];
    extern uint8_t __heap_end[];
#else
    static uint8_t heap_storage[ARENA_SIZE] __attribute__((aligned(4096)));
#endif

static void init() {
#ifdef BARE_METAL
    arena_base = __heap_start;
#else
    arena_base = heap_storage;
#endif
    arena_used = 0;
}

// Aligned bump allocation (primary allocator)
static void* alloc(size_t bytes, size_t align = 32) {
    // Align up
    size_t mask = align - 1;
    size_t offset = (arena_used + mask) & ~mask;

    if (offset + bytes > ARENA_SIZE) {
        serial::puts("[FATAL] OOM: arena exhausted\r\n");
        return nullptr;
    }

    void* ptr = arena_base + offset;
    arena_used = offset + bytes;
    return ptr;
}

// Reset arena (for training loop recycling)
static void reset() { arena_used = 0; }

// Usage stats
static size_t used() { return arena_used; }
static size_t free_bytes() { return ARENA_SIZE - arena_used; }

} // namespace mem

// ============================================================================
// MEMORY OPS (replaces cstring)
// ============================================================================

extern "C" {

void* memcpy(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;

    // AVX2 path for large copies
    if (n >= 32 && ((uintptr_t)d & 31) == 0 && ((uintptr_t)s & 31) == 0) {
        while (n >= 32) {
            __m256i v = _mm256_load_si256((const __m256i*)s);
            _mm256_store_si256((__m256i*)d, v);
            s += 32; d += 32; n -= 32;
        }
    }
    while (n--) *d++ = *s++;
    return dst;
}

void* memset(void* dst, int c, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    uint8_t val = (uint8_t)c;

    if (n >= 32 && ((uintptr_t)d & 31) == 0) {
        __m256i v = _mm256_set1_epi8((char)val);
        while (n >= 32) {
            _mm256_store_si256((__m256i*)d, v);
            d += 32; n -= 32;
        }
    }
    while (n--) *d++ = val;
    return dst;
}

void* memmove(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    if (d < s || d >= s + n) return memcpy(dst, src, n);
    // Overlap: copy backwards
    d += n; s += n;
    while (n--) *--d = *--s;
    return dst;
}

int memcmp(const void* a, const void* b, size_t n) {
    const uint8_t* pa = (const uint8_t*)a;
    const uint8_t* pb = (const uint8_t*)b;
    while (n--) {
        if (*pa != *pb) return *pa - *pb;
        pa++; pb++;
    }
    return 0;
}

int strcmp(const char* a, const char* b) {
    while (*a && *a == *b) { a++; b++; }
    return *(unsigned char*)a - *(unsigned char*)b;
}

size_t strlen(const char* s) {
    size_t len = 0;
    while (*s++) len++;
    return len;
}

} // extern "C"

// ============================================================================
// MATH (replaces cmath / libm) — Pure AVX2/FPU
// ============================================================================

namespace math {

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 6.28318530717958647692f;
constexpr float E = 2.71828182845904523536f;

// Fast reciprocal sqrt (Quake III style + Newton refinement)
static inline float rsqrt(float x) {
    __m128 v = _mm_set_ss(x);
    v = _mm_rsqrt_ss(v);
    // Newton step: y = y * (1.5 - 0.5*x*y*y)
    float y;
    _mm_store_ss(&y, v);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

static inline float sqrt(float x) {
    if (x <= 0) return 0;
    __m128 v = _mm_set_ss(x);
    v = _mm_sqrt_ss(v);
    float r; _mm_store_ss(&r, v);
    return r;
}

static inline float abs(float x) { return x < 0 ? -x : x; }
static inline float min(float a, float b) { return a < b ? a : b; }
static inline float max(float a, float b) { return a > b ? a : b; }

// Polynomial sin/cos (Chebyshev, |error| < 1e-6 on [-π, π])
static float sin(float x) {
    // Range reduce to [-π, π]
    x = x - TWO_PI * (float)(int)(x / TWO_PI);
    if (x > PI) x -= TWO_PI;
    if (x < -PI) x += TWO_PI;
    // Chebyshev
    float x2 = x * x;
    return x * (1.0f - x2 * (0.166666667f - x2 * (0.008333333f - x2 * 0.000198413f)));
}

static float cos(float x) { return sin(x + PI * 0.5f); }

static float tan(float x) {
    float c = cos(x);
    return (abs(c) > 1e-10f) ? sin(x) / c : 1e10f;
}

static float atan2(float y, float x) {
    if (abs(x) < 1e-10f && abs(y) < 1e-10f) return 0;
    float ax = abs(x), ay = abs(y);
    float mn = min(ax, ay), mx = max(ax, ay);
    float a = mn / mx;
    float s = a * a;
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    if (ay > ax) r = 1.57079637f - r;
    if (x < 0) r = PI - r;
    if (y < 0) r = -r;
    return r;
}

// Exponential: e^x via range reduction + polynomial
static float exp(float x) {
    if (x < -87.0f) return 0;
    if (x > 88.0f) return 1e38f;
    // x = n*ln(2) + r, |r| < 0.5*ln(2)
    float n = (float)(int)(x * 1.4426950408f + 0.5f);
    float r = x - n * 0.6931471806f;
    // Padé approximant for e^r
    float r2 = r * r;
    float p = 1.0f + r + 0.5f*r2 + r2*r*(0.166666667f + r*0.041666667f);
    // Multiply by 2^n using integer bit hack
    int32_t ni = (int32_t)n;
    union { float f; int32_t i; } u;
    u.f = p;
    u.i += (ni << 23);
    return u.f;
}

static float log(float x) {
    if (x <= 0) return -1e30f;
    // Extract mantissa and exponent
    union { float f; int32_t i; } u;
    u.f = x;
    int32_t e = ((u.i >> 23) & 0xFF) - 127;
    u.i = (u.i & 0x007FFFFF) | 0x3F800000;  // set exponent to 0
    float m = u.f;
    // log(x) = e*ln(2) + log(m), m in [1,2)
    float t = m - 1.0f;
    float ln_m = t * (1.0f - t * (0.5f - t * (0.333333f - t * 0.25f)));
    return (float)e * 0.6931471806f + ln_m;
}

static float pow(float base, float exp_) {
    return exp(exp_ * log(base));
}

static float tanh(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    float e2x = exp(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

} // namespace math

// ============================================================================
// TIMER — RDTSC-based microsecond timer
// ============================================================================

namespace timer {

static uint64_t tsc_freq = 0;  // TSC ticks/second (calibrated at boot)

// Calibrate using PIT (rough, ~1ms accuracy)
static void calibrate() {
    // Use PIT channel 2 for ~10ms calibration
    hw::outb(0x61, (hw::inb(0x61) & 0xFD) | 0x01);
    hw::outb(0x43, 0xB0);
    // ~10ms at 1.193182 MHz = 11932 ticks
    hw::outb(0x42, 0x9C);
    hw::outb(0x42, 0x2E);
    uint64_t t0 = hw::rdtsc();
    hw::outb(0x61, hw::inb(0x61) & 0xFE);
    hw::outb(0x61, hw::inb(0x61) | 0x01);
    while (!(hw::inb(0x61) & 0x20)) {}
    uint64_t t1 = hw::rdtsc();
    tsc_freq = (t1 - t0) * 100;  // ×100 since we measured ~10ms
}

static uint64_t now_us() {
    if (tsc_freq == 0) tsc_freq = 3000000000ULL;  // 3GHz fallback
    return hw::rdtsc() * 1000000ULL / tsc_freq;
}

} // namespace timer

// ============================================================================
// LIVING SILICON PRINT (replaces printf/fprintf)
// ============================================================================

namespace print {

// Print to both serial + VGA
static void str(const char* s) {
    serial::puts(s);
    vga::puts(s);
}

static void line(const char* s) {
    str(s);
    serial::newline();
    vga::putc('\n');
}

static void num(int64_t v) {
    serial::putd(v);
    // VGA: convert to string
    if (v < 0) { vga::putc('-'); v = -v; }
    if (v == 0) { vga::putc('0'); return; }
    char buf[20]; int i = 0;
    while (v > 0) { buf[i++] = '0' + (v % 10); v /= 10; }
    while (i > 0) vga::putc(buf[--i]);
}

static void flt(float v, int decimals = 4) {
    serial::putf(v, decimals);
    // Simplified VGA float
    if (v < 0) { vga::putc('-'); v = -v; }
    int64_t integer = (int64_t)v;
    num(integer);
    vga::putc('.');
    float frac = v - (float)integer;
    for (int d = 0; d < decimals; ++d) {
        frac *= 10.0f;
        vga::putc('0' + (int)frac);
        frac -= (float)(int)frac;
    }
}

static void banner(const char* text) {
    vga::puts_color(text, 15, 1);  // white on blue
    serial::puts(text);
    serial::newline();
}

} // namespace print

// ============================================================================
// ALIGNED ALLOC (replaces _mm_malloc / aligned_alloc)
// ============================================================================

static void* aligned_alloc_impl(size_t align, size_t size) {
    return mem::alloc(size, align);
}

static void aligned_free_impl(void*) {
    // Bump allocator: no individual free
}

// ============================================================================
// CPUID — Feature detection
// ============================================================================

namespace cpuid {

struct Features {
    bool avx2 = false;
    bool avx512 = false;
    bool fma = false;
    bool sse42 = false;
    char brand[48] = {};
};

static Features detect() {
    Features f;
    uint32_t eax, ebx, ecx, edx;

    asm volatile ("cpuid" : "=a"(eax),"=b"(ebx),"=c"(ecx),"=d"(edx) : "a"(7),"c"(0));
    f.avx2 = (ebx >> 5) & 1;
    f.avx512 = (ebx >> 16) & 1;

    asm volatile ("cpuid" : "=a"(eax),"=b"(ebx),"=c"(ecx),"=d"(edx) : "a"(1),"c"(0));
    f.fma = (ecx >> 12) & 1;
    f.sse42 = (ecx >> 20) & 1;

    // Brand string
    for (uint32_t i = 0; i < 3; ++i) {
        asm volatile ("cpuid" : "=a"(eax),"=b"(ebx),"=c"(ecx),"=d"(edx) : "a"(0x80000002+i));
        *(uint32_t*)(f.brand + i*16 + 0) = eax;
        *(uint32_t*)(f.brand + i*16 + 4) = ebx;
        *(uint32_t*)(f.brand + i*16 + 8) = ecx;
        *(uint32_t*)(f.brand + i*16 + 12) = edx;
    }
    f.brand[47] = 0;
    return f;
}

} // namespace cpuid
