// ============================================================================
// ENGINE.HPP — Living Silicon Torsion Field Engine
//
// Extracted from living_silicon_os.cpp v3.0.
// Provides: torsion PDE simulation, GA evolution, trajectory generation.
// Used as the DATA SOURCE for The Well C++ benchmark.
//
// The engine generates physics trajectories that surrogate models learn to
// predict.  This replaces the 15TB HDF5 dataset approach with live simulation.
// ============================================================================
#pragma once

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace well {
namespace engine {

constexpr int N = 2048;      // field nodes
constexpr int LANES = 8;     // parallel simulations
constexpr int ND = 64;       // N-D buckets

// ── AVX2 horizontal absolute sum (identical to V26) ──
static inline int64_t hsum_abs16(__m256i v) {
    __m256i a = _mm256_abs_epi16(v);
    __m256i lo = _mm256_unpacklo_epi16(a, _mm256_setzero_si256());
    __m256i hi = _mm256_unpackhi_epi16(a, _mm256_setzero_si256());
    __m256i s32 = _mm256_add_epi32(lo, hi);
    __m128i l = _mm256_castsi256_si128(s32);
    __m128i h = _mm256_extracti128_si256(s32, 1);
    __m128i s = _mm_add_epi32(l, h);
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    return _mm_cvtsi128_si32(s);
}

// ── Per-lane RNG ──
struct LaneRng {
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
    uint32_t range(uint32_t m) { return next() % m; }
};

static inline int abs_i(int x) { return x < 0 ? -x : x; }
static inline int min_i(int a, int b) { return a < b ? a : b; }
static inline int max_i(int a, int b) { return a > b ? a : b; }

// ── Lane state ──
struct alignas(32) Lane {
    int16_t  mag[N];
    uint16_t ph[N];
    uint8_t  nd[ND];
};

struct Genome {
    int32_t delta, coupling, blend, decay, nd_thr, inject_mag;
    int32_t fitness, best_fitness;
    uint32_t mutations;
    int32_t prev_e_bucket, ema, slope;
    uint32_t stagnation, improvements;
    LaneRng rng;
};

struct LaneStats {
    int64_t energy;
    int64_t coherence;
    int32_t nd_pop;
    int32_t peak;
};

// ── Engine state ──
static Lane      lanes[LANES];
static Genome    genomes[LANES];
static LaneStats stats[LANES];

// ── Advance one lane by one tick (SINGLE AVX2 PASS — identical to V26) ──
static void advance_lane(int lane, uint64_t tick) {
    auto& S = lanes[lane];
    auto& G = genomes[lane];

    const __m256i delta_v = _mm256_set1_epi16((int16_t)G.delta);
    const __m256i blend_v = _mm256_set1_epi16((int16_t)G.blend);
    const __m256i inv_bl  = _mm256_set1_epi16((int16_t)(256 - G.blend));
    const __m256i coup_v  = _mm256_set1_epi16((int16_t)G.coupling);
    const __m256i qtr     = _mm256_set1_epi16(16384);
    const int decay_s = G.decay;
    int64_t tick_energy = 0;

    for (int j = 0; j < N; j += 32) {
        auto* mp0 = (__m256i*)(S.mag + j);
        auto* mp1 = (__m256i*)(S.mag + j + 16);
        auto* pp0 = (__m256i*)(S.ph  + j);
        auto* pp1 = (__m256i*)(S.ph  + j + 16);

        __m256i m0 = _mm256_load_si256(mp0), m1 = _mm256_load_si256(mp1);
        __m256i p0 = _mm256_load_si256(pp0), p1 = _mm256_load_si256(pp1);

        p0 = _mm256_add_epi16(p0, delta_v);
        p1 = _mm256_add_epi16(p1, delta_v);

        __m256i pn0 = _mm256_alignr_epi8(p1, p0, 2);
        __m256i pn1 = _mm256_alignr_epi8(p0, p1, 2);
        __m256i mn0 = _mm256_alignr_epi8(m1, m0, 2);
        __m256i mn1 = _mm256_alignr_epi8(m0, m1, 2);

        __m256i pb0 = _mm256_add_epi16(_mm256_mulhi_epi16(p0,blend_v), _mm256_mulhi_epi16(pn0,inv_bl));
        __m256i pb1 = _mm256_add_epi16(_mm256_mulhi_epi16(p1,blend_v), _mm256_mulhi_epi16(pn1,inv_bl));
        __m256i mb0 = _mm256_add_epi16(_mm256_mulhi_epi16(m0,blend_v), _mm256_mulhi_epi16(mn0,inv_bl));
        __m256i mb1 = _mm256_add_epi16(_mm256_mulhi_epi16(m1,blend_v), _mm256_mulhi_epi16(mn1,inv_bl));

        __m256i sg0 = _mm256_srai_epi16(_mm256_add_epi16(pb0, qtr), 15);
        __m256i sg1 = _mm256_srai_epi16(_mm256_add_epi16(pb1, qtr), 15);
        __m256i c0 = _mm256_sub_epi16(_mm256_xor_si256(coup_v, sg0), sg0);
        __m256i c1 = _mm256_sub_epi16(_mm256_xor_si256(coup_v, sg1), sg1);
        mb0 = _mm256_adds_epi16(mb0, c0);
        mb1 = _mm256_adds_epi16(mb1, c1);

        if (decay_s > 0) {
            mb0 = _mm256_sub_epi16(mb0, _mm256_srai_epi16(mb0, decay_s));
            mb1 = _mm256_sub_epi16(mb1, _mm256_srai_epi16(mb1, decay_s));
        }

        _mm256_store_si256(mp0, mb0);
        _mm256_store_si256(mp1, mb1);
        _mm256_store_si256(pp0, pb0);
        _mm256_store_si256(pp1, pb1);

        tick_energy += hsum_abs16(mb0) + hsum_abs16(mb1);
    }

    stats[lane].energy = tick_energy;

    // N-D buckets
    int nd_count = 0;
    for (int i = 0; i < ND; ++i) {
        S.nd[i] = (abs_i((int)S.mag[i * (N / ND)]) > G.nd_thr) ? 1 : 0;
        nd_count += S.nd[i];
    }
    stats[lane].nd_pop = nd_count;

    // GA epoch (every 128 ticks)
    constexpr uint32_t EPOCH_MASK = 0x7F;
    if ((tick & EPOCH_MASK) == 0) {
        int64_t coh = 0;
        for (int j = 0; j < N - 1; ++j)
            coh += 65535 - abs_i((int)S.ph[j] - (int)S.ph[j + 1]);
        coh >>= 5;
        stats[lane].coherence = coh;

        int ct = (int)(coh >> 8);
        int eb = (int)(tick_energy >> 16);
        int ed = abs_i(eb - G.prev_e_bucket);
        G.prev_e_bucket = eb;

        int peak = 0;
        for (int i = 0; i < N; ++i) {
            int am = abs_i((int)S.mag[i]);
            if (am > peak) peak = am;
        }
        stats[lane].peak = peak;
        int ma = (int)(tick_energy / N);
        int ss = (ma > 0) ? min_i(peak / ma, 64) : 0;
        int es = 128 - min_i(abs_i(ed), 128);

        int fit = ss * 4 + ct + es;
        G.fitness = fit;
        int pe = G.ema;
        G.ema = (G.ema * 7 + fit) >> 3;
        G.slope = G.ema - pe;

        if (fit > G.best_fitness) { G.best_fitness = fit; G.stagnation = 0; G.improvements++; }
        else G.stagnation++;

        if (G.stagnation >= 2 || (G.slope <= 0 && G.rng.range(100) < 3)) {
            G.mutations++;
            int p = G.rng.range(4), mg = 1 + (int)G.rng.range(8);
            int d = (G.rng.next() & 1) ? 1 : -1;
            switch (p) {
                case 0: G.delta    = max_i(0, min_i(255, G.delta + d * mg)); break;
                case 1: G.coupling = max_i(0, min_i(255, G.coupling + d * mg)); break;
                case 2: G.blend    = max_i(0, min_i(255, G.blend + d * mg)); break;
                case 3: G.decay    = max_i(0, min_i(15, G.decay + d)); break;
            }
        }
    }
}

static void crossover() {
    int b = 0, w = 0;
    for (int i = 1; i < LANES; ++i) {
        if (genomes[i].best_fitness > genomes[b].best_fitness) b = i;
        if (genomes[i].best_fitness < genomes[w].best_fitness) w = i;
    }
    if (b == w || genomes[w].stagnation < 4) return;
    auto& gw = genomes[w]; auto& gb = genomes[b];
    gw.delta    = (3 * gw.delta + gb.delta) / 4;
    gw.coupling = (3 * gw.coupling + gb.coupling) / 4;
    gw.blend    = (3 * gw.blend + gb.blend) / 4;
    gw.decay    = (3 * gw.decay + gb.decay) / 4;
    gw.stagnation = 0;
}

static void initialize(uint64_t seed) {
    LaneRng init;
    init.seed(seed);
    memset(lanes, 0, sizeof(lanes));
    memset(stats, 0, sizeof(stats));
    for (int i = 0; i < LANES; ++i) {
        genomes[i] = {};
        genomes[i].delta = 17 + (int)init.range(20);
        genomes[i].coupling = 48 + (int)init.range(32);
        genomes[i].blend = 160 + (int)init.range(64);
        genomes[i].nd_thr = 128;
        genomes[i].inject_mag = 1000;
        genomes[i].rng.seed(seed ^ ((uint64_t)(i + 1) * 0x123456789ULL));
        // Random initial field
        for (int j = 0; j < N; ++j) {
            lanes[i].mag[j] = (int16_t)((init.next() & 0x7FFF) - 0x4000);
            lanes[i].ph[j]  = (uint16_t)init.next();
        }
    }
}

static void inject_gaussian() {
    int16_t sig[N];
    const int amp[] = {12000,11400,9600,7200,4800,2800,1400,600,200};
    for (int i = 0; i < N; ++i) {
        int dx = abs_i(i * 40 / N - 20);
        sig[i] = (dx < 9) ? (int16_t)amp[dx] : 0;
    }
    for (int l = 0; l < LANES; ++l) memcpy(lanes[l].mag, sig, sizeof(sig));
}

// ── Trajectory generation for ML training ──
// Runs the engine for `steps` ticks on `lane`, recording mag[] every `stride` ticks.
// Output: trajectory[steps/stride][N] as float32
static void generate_trajectory(float* out, int lane, int steps, int stride, uint64_t tick_base) {
    int idx = 0;
    for (int t = 0; t < steps; ++t) {
        advance_lane(lane, tick_base + t);
        if ((t % stride) == 0) {
            // Convert int16 → float32 normalized to [-1, 1]
            const float scale = 1.0f / 32768.0f;
            for (int j = 0; j < N; ++j) {
                out[idx * N + j] = (float)lanes[lane].mag[j] * scale;
            }
            ++idx;
        }
    }
}

static uint32_t total_mutations() {
    uint32_t t = 0;
    for (int i = 0; i < LANES; ++i) t += genomes[i].mutations;
    return t;
}

} // namespace engine
} // namespace well
