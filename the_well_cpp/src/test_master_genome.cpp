// ============================================================================
// TEST_MASTER_GENOME.CPP — Hosted test for 36-param MasterGenome + engine
//
// Build:
//   g++ -O3 -mavx2 -mfma -std=c++20 -I src -o test_genome src/test_master_genome.cpp
//
// Tests:
//   1. MasterGenome init + size verification
//   2. Cascade mutation distribution across levels
//   3. Engine integration with MasterGenome parameters
//   4. Proprioceptive feedback loop
//   5. Resource allocation (focus/spread)
//   6. Performance benchmark (ticks/s)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
static uint64_t now_us() {
    static LARGE_INTEGER freq = []{ LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (uint64_t)(c.QuadPart * 1000000ULL / freq.QuadPart);
}
#else
#include <sys/time.h>
static uint64_t now_us() {
    struct timeval tv; gettimeofday(&tv, nullptr);
    return (uint64_t)tv.tv_sec * 1000000ULL + tv.tv_usec;
}
#endif

// Include the genome
#include "master_genome.hpp"

// Inline engine (simplified for test — same SIMD hot-path)
namespace engine {

constexpr int N = 2048;
constexpr int LANES = 8;
constexpr int ND = 64;

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

struct alignas(32) Lane {
    int16_t  mag[N];
    uint16_t ph[N];
    uint8_t  nd[ND];
};

static Lane lanes[LANES];

static inline int abs_i(int x) { return x < 0 ? -x : x; }
static inline int min_i(int a, int b) { return a < b ? a : b; }

struct LaneResult {
    int64_t energy;
    int64_t coherence;
    int32_t nd_pop;
    int32_t peak;
};

// Hot-path: advance lane using MasterGenome parameters
static LaneResult advance_lane(int lane, uint64_t tick, const silicon::MasterGenome& g) {
    auto& S = lanes[lane];

    // Read params from genome — variant A or B based on lane parity
    int delta_idx    = (lane & 1) ? silicon::P_DELTA_B    : silicon::P_DELTA_A;
    int coupling_idx = (lane & 1) ? silicon::P_COUPLING_B : silicon::P_COUPLING_A;
    int blend_idx    = (lane & 1) ? silicon::P_BLEND_B    : silicon::P_BLEND_A;

    int16_t delta_val = (int16_t)g.params[delta_idx];
    int16_t blend_val = (int16_t)g.params[blend_idx];
    int16_t coup_val  = (int16_t)g.params[coupling_idx];
    int decay_s       = g.params[silicon::P_DECAY] & 0x0F;  // 0-15

    const __m256i delta_v = _mm256_set1_epi16(delta_val);
    const __m256i blend_v = _mm256_set1_epi16(blend_val);
    const __m256i inv_bl  = _mm256_set1_epi16((int16_t)(256 - blend_val));
    const __m256i coup_v  = _mm256_set1_epi16(coup_val);
    const __m256i qtr     = _mm256_set1_epi16(16384);
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

    // ND buckets
    int nd_threshold = g.params[silicon::P_ND_THRESHOLD];
    int nd_count = 0;
    for (int i = 0; i < ND; i++) {
        S.nd[i] = (abs_i((int)S.mag[i * (N / ND)]) > nd_threshold) ? 1 : 0;
        nd_count += S.nd[i];
    }

    // Coherence (every epoch)
    int64_t coh = 0;
    if ((tick & 0x7F) == 0) {
        for (int j = 0; j < N - 1; j++)
            coh += 65535 - abs_i((int)S.ph[j] - (int)S.ph[j + 1]);
        coh >>= 5;
    }

    LaneResult r;
    r.energy = tick_energy;
    r.coherence = coh;
    r.nd_pop = nd_count;

    int peak = 0;
    for (int i = 0; i < N; i++) {
        int am = abs_i((int)S.mag[i]);
        if (am > peak) peak = am;
    }
    r.peak = peak;

    return r;
}

static void initialize(uint64_t seed) {
    silicon::Rng32 rng;
    rng.seed(seed);
    memset(lanes, 0, sizeof(lanes));
    for (int i = 0; i < LANES; i++) {
        for (int j = 0; j < N; j++) {
            lanes[i].mag[j] = (int16_t)((rng.next() & 0x7FFF) - 0x4000);
            lanes[i].ph[j]  = (uint16_t)rng.next();
        }
    }
}

} // namespace engine

// ============================================================================
// TESTS
// ============================================================================

static int tests_passed = 0, tests_failed = 0;
#define CHECK(cond, name) do { \
    if (cond) { tests_passed++; fprintf(stderr, "  [PASS] %s\n", name); } \
    else { tests_failed++; fprintf(stderr, "  [FAIL] %s\n", name); } \
} while(0)

void test_genome_size() {
    fprintf(stderr, "\n═══ TEST 1: Genome Size ═══\n");
    silicon::MasterGenome g;
    g.init(42);

    CHECK(silicon::GENOME_SIZE == 36, "GENOME_SIZE == 36");
    CHECK(sizeof(g.params) == 36, "params array = 36 bytes");
    CHECK(sizeof(g) <= 128, "MasterGenome <= 128 bytes (2 cache lines)");
    fprintf(stderr, "  sizeof(MasterGenome) = %zu bytes\n", sizeof(g));
    fprintf(stderr, "  sizeof(params) = %zu bytes = %.1f cache lines\n",
            sizeof(g.params), (float)sizeof(g.params) / 64.0f);
}

void test_cascade_mutation() {
    fprintf(stderr, "\n═══ TEST 2: Cascade Mutation ═══\n");
    silicon::MasterGenome g;
    g.init(42);
    g.stagnation = 10;  // force mutations

    // Snapshot original
    uint8_t orig[36];
    memcpy(orig, g.params, 36);

    int l0_changes = 0, l1_changes = 0, l2_changes = 0, l3_changes = 0;

    // Run 2M ticks of cascade mutation
    for (uint64_t t = 0; t < 2000000; t++) {
        g.cascade_mutate(t);
    }

    for (int i = 0; i < silicon::L0_COUNT; i++)
        if (g.params[i] != orig[i]) l0_changes++;
    for (int i = silicon::L0_COUNT; i < silicon::L0_COUNT + silicon::L1_COUNT; i++)
        if (g.params[i] != orig[i]) l1_changes++;
    for (int i = silicon::L0_COUNT + silicon::L1_COUNT;
         i < silicon::L0_COUNT + silicon::L1_COUNT + silicon::L2_COUNT; i++)
        if (g.params[i] != orig[i]) l2_changes++;
    for (int i = silicon::L0_COUNT + silicon::L1_COUNT + silicon::L2_COUNT;
         i < silicon::GENOME_SIZE; i++)
        if (g.params[i] != orig[i]) l3_changes++;

    fprintf(stderr, "  L0 changes: %d/8\n", l0_changes);
    fprintf(stderr, "  L1 changes: %d/8\n", l1_changes);
    fprintf(stderr, "  L2 changes: %d/12\n", l2_changes);
    fprintf(stderr, "  L3 changes: %d/8\n", l3_changes);
    fprintf(stderr, "  Total mutations: %u\n", g.total_mutations);

    CHECK(l0_changes > 0, "L0 mutated (fastest)");
    CHECK(l0_changes >= l1_changes, "L0 >= L1 changes (cascade order)");
    CHECK(l1_changes >= l3_changes, "L1 >= L3 changes (cascade order)");
    CHECK(g.total_mutations > 1000, "Total mutations > 1000 in 2M ticks");

    float mut_per_sec_est = (float)g.total_mutations / 2.0f * 6.0f;  // 6M ticks/s
    fprintf(stderr, "  Estimated mut/s at 6M ticks/s: %.0f\n", mut_per_sec_est);
}

void test_engine_integration() {
    fprintf(stderr, "\n═══ TEST 3: Engine Integration ═══\n");
    silicon::MasterGenome g;
    g.init(0xC0DE7052ULL);
    engine::initialize(0xC0DE7052ULL);

    // Run 1000 ticks
    engine::LaneResult last;
    for (uint64_t t = 0; t < 1000; t++) {
        for (int l = 0; l < engine::LANES; l++) {
            last = engine::advance_lane(l, t, g);
        }
        g.cascade_mutate(t);
    }

    CHECK(last.energy > 0, "Engine produces energy");
    CHECK(last.nd_pop >= 0 && last.nd_pop <= engine::ND, "ND pop in valid range");

    fprintf(stderr, "  Energy: %lld\n", (long long)last.energy);
    fprintf(stderr, "  ND pop: %d/%d\n", last.nd_pop, engine::ND);
    fprintf(stderr, "  Peak mag: %d\n", last.peak);
}

void test_proprioception() {
    fprintf(stderr, "\n═══ TEST 4: Proprioceptive Feedback ═══\n");
    silicon::MasterGenome g;
    g.init(0xBEEF42ULL);
    engine::initialize(0xBEEF42ULL);

    float first_fitness = 0, last_fitness = 0;

    for (uint64_t t = 0; t < 100000; t++) {
        int64_t total_energy = 0;
        int64_t total_coh = 0;
        int total_nd = 0;

        for (int l = 0; l < engine::LANES; l++) {
            auto r = engine::advance_lane(l, t, g);
            total_energy += r.energy;
            total_coh += r.coherence;
            total_nd += r.nd_pop;
        }

        g.cascade_mutate(t);

        // Evaluate fitness every 128 ticks
        if ((t & 0x7F) == 0) {
            silicon::Proprioception prop;
            prop.energy = (float)total_energy;
            prop.coherence = (float)total_coh / (65535.0f * engine::N * engine::LANES);
            prop.nd_balance = (float)total_nd / (float)(engine::ND * engine::LANES);
            prop.pressure = 0.0f;
            prop.total_ticks = t;
            prop.mutations = g.total_mutations;

            float f = g.evaluate(prop);
            if (t < 256) first_fitness = f;
            last_fitness = f;
        }
    }

    CHECK(g.total_mutations > 0, "Mutations happened");
    CHECK(g.best_fitness > 0, "Best fitness > 0");

    fprintf(stderr, "  First fitness: %.2f\n", first_fitness);
    fprintf(stderr, "  Last fitness:  %.2f\n", last_fitness);
    fprintf(stderr, "  Best fitness:  %.2f\n", g.best_fitness);
    fprintf(stderr, "  Improvements:  %u\n", g.improvements);
    fprintf(stderr, "  Total mutations: %u\n", g.total_mutations);

    g.print_summary();
}

void test_resource_allocation() {
    fprintf(stderr, "\n═══ TEST 5: Resource Allocation ═══\n");
    silicon::MasterGenome g;
    g.init(42);

    g.params[silicon::P_FOCUS] = 0;
    CHECK(g.focus_lanes() == 1, "Focus=0 → 1 lane (full spread)");
    CHECK(g.is_scout(7), "Lane 7 is scout when focus < 240");

    g.params[silicon::P_FOCUS] = 128;
    int f128 = g.focus_lanes();
    fprintf(stderr, "  Focus=128 → %d lanes\n", f128);
    CHECK(f128 >= 3 && f128 <= 5, "Focus=128 → ~4 lanes");

    g.params[silicon::P_FOCUS] = 255;
    CHECK(g.focus_lanes() == 8, "Focus=255 → 8 lanes (laser)");
    CHECK(!g.is_scout(7), "No scout in laser mode");
}

void test_benchmark() {
    fprintf(stderr, "\n═══ TEST 6: Performance Benchmark ═══\n");
    silicon::MasterGenome g;
    g.init(0xC0DE7052ULL);
    engine::initialize(0xC0DE7052ULL);

    // Warmup
    for (uint64_t t = 0; t < 10000; t++) {
        for (int l = 0; l < engine::LANES; l++)
            engine::advance_lane(l, t, g);
    }

    // Benchmark: 100k ticks
    uint64_t t0 = now_us();
    uint64_t bench_ticks = 100000;

    for (uint64_t t = 0; t < bench_ticks; t++) {
        for (int l = 0; l < engine::LANES; l++)
            engine::advance_lane(l, t, g);
        g.cascade_mutate(t);
    }

    uint64_t elapsed = now_us() - t0;
    double sec = (double)elapsed / 1e6;
    double ticks_per_sec = (double)bench_ticks / sec;
    double lane_ticks_per_sec = ticks_per_sec * engine::LANES;
    double nodes_per_sec = lane_ticks_per_sec * engine::N;
    double gb_per_sec = nodes_per_sec * 4.0 / 1e9;  // 4 bytes per node (mag+ph)
    double mutations_per_sec = (double)g.total_mutations / sec;

    fprintf(stderr, "  Duration:         %.3f s\n", sec);
    fprintf(stderr, "  Ticks/s:          %.0f\n", ticks_per_sec);
    fprintf(stderr, "  Lane-ticks/s:     %.0f\n", lane_ticks_per_sec);
    fprintf(stderr, "  Nodes/s:          %.2f M\n", nodes_per_sec / 1e6);
    fprintf(stderr, "  Bandwidth:        %.1f GB/s\n", gb_per_sec);
    fprintf(stderr, "  Mutations/s:      %.0f\n", mutations_per_sec);
    fprintf(stderr, "  Genome mutations: %u\n", g.total_mutations);

    CHECK(ticks_per_sec > 100000, "Throughput > 100k ticks/s");
    CHECK(mutations_per_sec > 1000, "Mutations > 1k/s");

    // Final genome state
    fprintf(stderr, "\n  Final genome state:\n");
    g.print_summary();
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    fprintf(stderr,
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║    LIVING SILICON — 36-Param MasterGenome Test Suite    ║\n"
        "║    i7-6700HQ Target — L1:32KB L2:256KB L3:6MB          ║\n"
        "╚══════════════════════════════════════════════════════════╝\n");

    test_genome_size();
    test_cascade_mutation();
    test_engine_integration();
    test_proprioception();
    test_resource_allocation();
    test_benchmark();

    fprintf(stderr,
        "\n══════════════════════════════════════════════════════════\n"
        "  RESULTS: %d PASS / %d FAIL\n"
        "══════════════════════════════════════════════════════════\n\n",
        tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
