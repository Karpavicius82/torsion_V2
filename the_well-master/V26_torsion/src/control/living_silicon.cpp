#include "control/living_silicon.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <bit>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace antigravity::control::living {
namespace {

std::uint32_t xorshift32(std::uint32_t& rng) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

std::uint64_t exc_inh(const std::uint64_t current, const std::uint64_t excitatory, const std::uint64_t inhibitory) {
    return (current & ~inhibitory) | (excitatory & (~current) & (~inhibitory));
}

int clamp_i16(const int value) {
    return std::clamp(value, -32768, 32767);
}

// Correct cyclic phase distance on uint16 ring
std::uint16_t phase_distance(const std::uint16_t a, const std::uint16_t b) {
    const std::uint32_t d = a >= b ? static_cast<std::uint32_t>(a - b)
                                   : static_cast<std::uint32_t>(b - a);
    return static_cast<std::uint16_t>(std::min<std::uint32_t>(d, 65536U - d));
}

#if defined(__AVX2__)
std::int64_t sum_abs_16(__m256i vec) {
    alignas(32) std::array<std::int16_t, 16> tmp{};
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp.data()), _mm256_abs_epi16(vec));
    std::int64_t sum = 0;
    for (const auto value : tmp) {
        sum += value;
    }
    return sum;
}

std::int64_t sum_u16(__m256i vec) {
    alignas(32) std::array<std::uint16_t, 16> tmp{};
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp.data()), vec);
    std::int64_t sum = 0;
    for (const auto value : tmp) {
        sum += value;
    }
    return sum;
}
#endif

} // namespace

Engine::Engine() {
    initialize(0xC0FFEEULL);
}

void Engine::initialize(const std::uint64_t seed) {
    std::scoped_lock lock(mutex_);

    std::uint32_t rng = static_cast<std::uint32_t>(seed == 0 ? 0xC0FFEEU : seed);
    for (std::size_t lane = 0; lane < kThreads; ++lane) {
        auto& genome = genomes_[lane];
        genome.delta.store(17, std::memory_order_relaxed);
        genome.coupling.store(64, std::memory_order_relaxed);
        genome.threshold.store(8000, std::memory_order_relaxed);
        genome.blend.store(192, std::memory_order_relaxed);
        genome.decay.store(0, std::memory_order_relaxed);
        genome.inject_rate.store(64, std::memory_order_relaxed);
        genome.generation.store(0, std::memory_order_relaxed);
        genome.fitness.store(0, std::memory_order_relaxed);
        genome.best_fitness.store(0, std::memory_order_relaxed);
        genome.total_mutations.store(0, std::memory_order_relaxed);

        auto& state = data_[lane];
        auto& obs = obs_[lane];
        state.rng = 0xCAFE0000U + static_cast<std::uint32_t>(lane) ^ rng;
        state.tick_counter = 0;
        for (auto& word : state.nd) {
            word = (static_cast<std::uint64_t>(xorshift32(rng)) << 32) | xorshift32(rng);
        }
        for (std::size_t i = 0; i < kNodes; ++i) {
            state.mag[i] = static_cast<std::int16_t>((xorshift32(rng) & 0x7FFFU) - 0x4000U);
            state.ph[i] = static_cast<std::uint16_t>(xorshift32(rng));
        }
        obs = Observation{};
        ctrl_[lane] = ControllerState{};
    }
}

void Engine::inject(const std::size_t lane, const std::int16_t* signal, std::size_t n) {
    if (lane >= kThreads || signal == nullptr || n == 0) {
        return;
    }
    std::scoped_lock lock(mutex_);
    n = std::min(n, kNodes);
    std::copy_n(signal, n, data_[lane].mag.begin());

    // Track stimulus strength for fitness gating
    std::int64_t drive = 0;
    for (std::size_t i = 0; i < n; ++i) {
        drive += std::abs(static_cast<int>(signal[i]));
    }
    drive = static_cast<std::int64_t>(n) > 0 ? (drive / static_cast<std::int64_t>(n)) : 0;
    ctrl_[lane].recent_drive = static_cast<std::int32_t>(
        std::clamp<std::int64_t>(drive >> 6, 0, 1024));
}

void Engine::tick(const std::uint64_t ticks) {
    std::scoped_lock lock(mutex_);
    for (std::uint64_t i = 0; i < ticks; ++i) {
        // Compute collective membrane BEFORE lane advances
        if (enable_collective_) {
            compute_membrane();
        }
        for (std::size_t lane = 0; lane < kThreads; ++lane) {
            advance_lane(lane);
        }
        ++global_tick_;
        // Epoch-based crossover after all lanes advanced
        if (enable_collective_ && (global_tick_ & kEpochMask) == 0) {
            maybe_crossover();
        }
    }
}

void Engine::advance_lane(const std::size_t lane) {
    const auto started = std::chrono::steady_clock::now();
    auto& state = data_[lane];
    auto& obs = obs_[lane];
    auto& ctrl = ctrl_[lane];

    auto& genome = genomes_[lane];
    const auto delta = genome.delta.load(std::memory_order_relaxed);
    const auto base_coupling = genome.coupling.load(std::memory_order_relaxed);
    const auto threshold = genome.threshold.load(std::memory_order_relaxed);
    const auto base_blend = genome.blend.load(std::memory_order_relaxed);
    const auto decay = genome.decay.load(std::memory_order_relaxed);

    // Runtime modulation: effective_* from base genome + controller state
    const int mem = std::min<int>(ctrl.membrane_local, 4);
    const int stag = std::min<int>(ctrl.stagnation_epochs, 4);
    const auto coupling = static_cast<std::int16_t>(
        std::clamp<int>(base_coupling + mem * 8 + stag * 4, 0, 200));
    const auto blend = static_cast<std::int16_t>(
        std::clamp<int>(base_blend - mem * 8, 32, 224));

    std::int64_t tick_energy = 0;

#if defined(__AVX2__)
    const __m256i delta_v = _mm256_set1_epi16(delta);
    const __m256i coupling_v = _mm256_set1_epi16(coupling);
    const __m256i blend_v = _mm256_set1_epi16(blend);
    const __m256i inv_blend_v = _mm256_set1_epi16(256 - blend);
    const __m256i quarter_v = _mm256_set1_epi16(16384);

    for (std::size_t j = 0; j < kNodes; j += 32) {
        auto* mag_ptr = reinterpret_cast<__m256i*>(state.mag.data() + j);
        auto* ph_ptr = reinterpret_cast<__m256i*>(state.ph.data() + j);
        auto* mag_ptr_1 = reinterpret_cast<__m256i*>(state.mag.data() + j + 16);
        auto* ph_ptr_1 = reinterpret_cast<__m256i*>(state.ph.data() + j + 16);

        __m256i m0 = _mm256_load_si256(mag_ptr);
        __m256i m1 = _mm256_load_si256(mag_ptr_1);
        __m256i p0 = _mm256_load_si256(ph_ptr);
        __m256i p1 = _mm256_load_si256(ph_ptr_1);

        p0 = _mm256_add_epi16(p0, delta_v);
        p1 = _mm256_add_epi16(p1, delta_v);

        const __m256i pn0 = _mm256_alignr_epi8(p0, p1, 2);
        const __m256i pn1 = _mm256_alignr_epi8(p1, p0, 2);
        const __m256i mn0 = _mm256_alignr_epi8(m0, m1, 2);
        const __m256i mn1 = _mm256_alignr_epi8(m1, m0, 2);

        __m256i pb0 = _mm256_add_epi16(_mm256_mulhi_epi16(p0, blend_v), _mm256_mulhi_epi16(pn0, inv_blend_v));
        __m256i pb1 = _mm256_add_epi16(_mm256_mulhi_epi16(p1, blend_v), _mm256_mulhi_epi16(pn1, inv_blend_v));
        __m256i mb0 = _mm256_add_epi16(_mm256_mulhi_epi16(m0, blend_v), _mm256_mulhi_epi16(mn0, inv_blend_v));
        __m256i mb1 = _mm256_add_epi16(_mm256_mulhi_epi16(m1, blend_v), _mm256_mulhi_epi16(mn1, inv_blend_v));

        const __m256i sg0 = _mm256_srai_epi16(_mm256_add_epi16(pb0, quarter_v), 15);
        const __m256i sg1 = _mm256_srai_epi16(_mm256_add_epi16(pb1, quarter_v), 15);
        const __m256i c0 = _mm256_sub_epi16(_mm256_xor_si256(coupling_v, sg0), sg0);
        const __m256i c1 = _mm256_sub_epi16(_mm256_xor_si256(coupling_v, sg1), sg1);
        mb0 = _mm256_adds_epi16(mb0, c0);
        mb1 = _mm256_adds_epi16(mb1, c1);

        if (decay > 0) {
            const __m256i dec0 = _mm256_srai_epi16(mb0, decay);
            const __m256i dec1 = _mm256_srai_epi16(mb1, decay);
            mb0 = _mm256_sub_epi16(mb0, dec0);
            mb1 = _mm256_sub_epi16(mb1, dec1);
        }

        _mm256_store_si256(mag_ptr, mb0);
        _mm256_store_si256(mag_ptr_1, mb1);
        _mm256_store_si256(ph_ptr, pb0);
        _mm256_store_si256(ph_ptr_1, pb1);

        tick_energy += sum_abs_16(mb0) + sum_abs_16(mb1);
    }
#else
    for (std::size_t j = 0; j < kNodes; ++j) {
        const std::size_t next = (j + 1U) & (kNodes - 1U);
        const std::size_t prev = (j - 1U) & (kNodes - 1U);
        const auto p = static_cast<std::uint16_t>(state.ph[j] + delta);
        const auto pn = static_cast<std::uint16_t>(state.ph[next] + delta);
        const auto m = state.mag[j];
        const auto mn = state.mag[next];
        const auto mp = state.mag[prev];

        const auto pb = static_cast<std::uint16_t>((static_cast<std::uint32_t>(p) * blend + static_cast<std::uint32_t>(pn) * (256 - blend)) >> 8);
        auto mb = static_cast<int>((static_cast<int>(m) * blend + static_cast<int>(mn) * (256 - blend)) >> 8);
        const int sign = (static_cast<std::uint16_t>(pb + 16384U) >> 15) ? -1 : 1;
        mb = clamp_i16(mb + sign * coupling);

        // TORSIJOS PDE: kubinis narys gS^3 (solitonu formavimas)
        const int m_val = static_cast<int>(m);
        const int cubic = (m_val * m_val * m_val) >> 22;
        mb = clamp_i16(mb + (cubic >> 3));

        // TORSIJOS PDE: Laplasianas d^2S/dx^2 (bangos sklidimas)
        const int laplacian = (static_cast<int>(mp) - 2 * m_val + static_cast<int>(mn)) >> 3;
        mb = clamp_i16(mb + laplacian);

        if (decay > 0) {
            mb -= (mb >> decay);
        }

        state.mag[j] = static_cast<std::int16_t>(clamp_i16(mb));
        state.ph[j] = pb;
        tick_energy += std::abs(static_cast<int>(state.mag[j]));
    }
#endif

    obs.energy = tick_energy;

    // Coherence: separate pass with correct cyclic phase distance
    // Identical semantics for scalar and AVX2 paths
    std::int64_t tick_coherence = 0;
    for (std::size_t j = 0; j < kNodes; ++j) {
        const std::size_t next = (j + 1U) & (kNodes - 1U);
        tick_coherence += phase_distance(state.ph[j], state.ph[next]);
    }
    const auto normalized = std::clamp<std::int64_t>(65535 - (tick_coherence >> 5), 0, 65535);
    obs.coherence = normalized;

    // nd[] and nd_popcount BEFORE fitness/mutation
    std::uint64_t exc = 0;
    std::uint64_t inh = 0;
    for (int bit = 0; bit < 64; ++bit) {
        const auto idx = static_cast<std::size_t>((bit * 32) & (kNodes - 1));
        if (state.mag[idx] > threshold) exc |= (1ULL << bit);
        if (state.mag[idx] < -threshold) inh |= (1ULL << bit);
    }
    for (auto& word : state.nd) {
        word = exc_inh(word, exc, inh);
    }
    obs.nd_popcount = 0;
    for (const auto word : state.nd) {
        obs.nd_popcount += static_cast<std::int32_t>(std::popcount(word));
    }

    state.tick_counter += 1;

    // Mutate AFTER nd/obs are current (correct fitness calculation)
    maybe_mutate(lane, tick_energy);

    // Update controller membrane_local
    if (enable_collective_) {
        std::uint32_t mem_sum = 0;
        for (std::size_t j = 0; j < kNodes; ++j) {
            mem_sum += membrane_[j];
        }
        ctrl.membrane_local = static_cast<std::uint16_t>(mem_sum / kNodes);
    } else {
        ctrl.membrane_local = 0;
    }

    // Decay recent_drive (stimulus memory fades)
    ctrl.recent_drive = (ctrl.recent_drive * 7) / 8;

    // Mirror controller state to observation for telemetry
    obs.fitness_ema = ctrl.fitness_ema;
    obs.recent_drive = ctrl.recent_drive;
    obs.stagnation_epochs = ctrl.stagnation_epochs;
    obs.membrane_local = ctrl.membrane_local;
    obs.attention_hits = ctrl.attention_hits;

    const auto elapsed = std::chrono::steady_clock::now() - started;
    const auto elapsed_ns = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    const auto elapsed_us = static_cast<std::int32_t>(std::max<std::int64_t>(0, elapsed_ns / 1000));

    obs.tick_ns = elapsed_ns;
    obs.tick_ns_ema = obs.tick_ns_ema <= 0
        ? elapsed_ns
        : ((obs.tick_ns_ema * 7) + elapsed_ns) / 8;

    obs.pressure = obs.pressure <= 0
        ? elapsed_us
        : static_cast<std::int32_t>(((obs.pressure * 7) + elapsed_us) / 8);
}

void Engine::maybe_mutate(const std::size_t lane, const std::int64_t tick_energy) {
    auto& state = data_[lane];
    auto& obs = obs_[lane];
    auto& ctrl = ctrl_[lane];

    if ((state.tick_counter & kEpochMask) != 0) {
        return;
    }

    // TORSIJOS FITNESS: fizikinis prasmingumas
    // GA evoliucija ieskos parametru, kur:
    // 1. Solitonas FORMUOJASI (lokalizuota struktura = fantomas)
    // 2. Energija KONSERVUOJASI (ne issisklaido ir ne sprogsta)
    // 3. Fazine koherencija AUGA (bangos desningumas)
    const auto coherence_term = static_cast<std::int32_t>(obs.coherence >> 8);
    const auto energy_bucket = static_cast<std::int32_t>(tick_energy >> 16);
    const auto energy_delta = std::abs(energy_bucket - ctrl.prev_energy_bucket);
    ctrl.prev_energy_bucket = energy_bucket;

    // Solitono aptikimas: lokalizuota struktura (piku/vidurkio santykis)
    std::int32_t peak_val = 0;
    std::int64_t abs_sum = 0;
    for (std::size_t i = 0; i < kNodes; ++i) {
        const auto abs_m = std::abs(static_cast<int>(state.mag[i]));
        if (abs_m > peak_val) peak_val = abs_m;
        abs_sum += abs_m;
    }
    const auto mean_abs = static_cast<std::int32_t>(abs_sum / static_cast<std::int64_t>(kNodes));
    const auto soliton_score = (mean_abs > 0) ? std::min(peak_val / mean_abs, 64) : 0;

    // Energijos stabilumas: baudzia sprogima ir numirima
    const auto e_stability = 128 - std::min(std::abs(energy_delta), 128);

    const auto my_fitness = soliton_score * 4 + coherence_term + e_stability;

    auto& genome = genomes_[lane];
    const auto previous_fitness = genome.fitness.load(std::memory_order_relaxed);
    genome.fitness.store(my_fitness, std::memory_order_relaxed);
    const auto current_best = genome.best_fitness.load(std::memory_order_relaxed);

    // Update fitness EMA (alpha = 1/8)
    ctrl.prev_fitness_ema = ctrl.fitness_ema;
    ctrl.fitness_ema = (7 * ctrl.fitness_ema + my_fitness) / 8;
    const auto slope = ctrl.fitness_ema - ctrl.prev_fitness_ema;

    // Stagnation tracking
    const bool improving = my_fitness > previous_fitness;
    if (improving) {
        ctrl.stagnation_epochs = 0;
    } else if (ctrl.stagnation_epochs < 255) {
        ctrl.stagnation_epochs += 1;
    }

    // Single exploration mechanism: stagnation-based
    const bool explore = ctrl.stagnation_epochs >= 2;
    const bool should_mutate = explore
                            || ((state.rng & 0x1FU) == 0U && slope <= 0);  // rare random only when NOT trending up

    if (should_mutate) {
        auto mutate_field = [&](std::atomic<std::int16_t>& field, const int min_v, const int max_v, const int delta_mask, const int delta_bias) {
            const auto current = field.load(std::memory_order_relaxed);
            xorshift32(state.rng);
            // Wider mutations when deeply stagnant
            const int stag_boost = (ctrl.stagnation_epochs > 8) ? 2 : 1;
            auto next = current + static_cast<std::int16_t>(((state.rng & delta_mask) - delta_bias) * stag_boost);
            next = static_cast<std::int16_t>(std::clamp<int>(next, min_v, max_v));
            field.store(next, std::memory_order_relaxed);
        };

        xorshift32(state.rng);
        switch (state.rng % 6U) {
        case 0: mutate_field(genome.delta, 1, 100, 0x0F, 8); break;
        case 1: mutate_field(genome.coupling, 0, 200, 0x0F, 8); break;
        case 2: mutate_field(genome.blend, 32, 224, 0x1F, 16); break;
        case 3: mutate_field(genome.threshold, 1000, 20000, 0x1FF, 256); break;
        case 4: mutate_field(genome.decay, 0, 8, 0x03, 1); break;
        default: mutate_field(genome.inject_rate, 16, 256, 0x1F, 16); break;
        }

        genome.generation.fetch_add(1, std::memory_order_relaxed);
        genome.total_mutations.fetch_add(1, std::memory_order_relaxed);
        obs.mutations += 1;

        int expected = current_best;
        while (my_fitness > expected &&
               !genome.best_fitness.compare_exchange_weak(expected, my_fitness, std::memory_order_relaxed)) {
        }
        if (my_fitness > current_best) {
            obs.improvements += 1;
        }
    }

    // Injection: focused when collective, with effective_inject_rate
    if ((state.rng & 0x3FU) == 0U) {
        const auto base_inject = genome.inject_rate.load(std::memory_order_relaxed);
        const auto effective_inject_rate = static_cast<std::uint32_t>(
            std::clamp<int>(base_inject + (ctrl.stagnation_epochs >= 2 ? 16 : 0), 16, 256));
        for (std::uint32_t i = 0; i < effective_inject_rate; ++i) {
            const auto idx = xorshift32(state.rng) & (kNodes - 1U);
            if (enable_collective_ && membrane_[idx] >= 2) {
                state.mag[idx] = static_cast<std::int16_t>((state.rng & 0x7FFFU) - 0x4000U);
                ctrl.attention_hits += 1;
            } else if (!enable_collective_) {
                state.mag[idx] = static_cast<std::int16_t>((state.rng & 0x7FFFU) - 0x4000U);
            }
        }
    }
}

// ── Collective mechanisms ───────────────────────────────────────────

void Engine::compute_membrane() {
    // Disagreement score per node: how many lanes are in each half-period.
    // 0 = all agree, kThreads/2 = maximum split.
    for (std::size_t i = 0; i < kNodes; ++i) {
        std::uint8_t hi = 0;
        for (std::size_t lane = 0; lane < kThreads; ++lane) {
            hi += static_cast<std::uint8_t>((data_[lane].ph[i] >> 15) & 1U);
        }
        membrane_[i] = static_cast<std::uint8_t>(
            std::min<int>(static_cast<int>(hi), static_cast<int>(kThreads) - static_cast<int>(hi)));
    }
}

void Engine::maybe_crossover() {
    // Only every kCrossoverEpochs epochs
    const auto epoch_num = global_tick_ >> 7;  // divide by kEpochMask+1
    if ((epoch_num % kCrossoverEpochs) != 0) return;

    // Find best and worst lanes by combined fitness+coherence
    std::size_t best = 0, worst = 0;
    auto lane_score = [&](std::size_t lane) -> std::int64_t {
        return static_cast<std::int64_t>(genomes_[lane].fitness.load(std::memory_order_relaxed))
             + obs_[lane].coherence;
    };
    for (std::size_t lane = 1; lane < kThreads; ++lane) {
        if (lane_score(lane) > lane_score(best)) best = lane;
        if (lane_score(lane) < lane_score(worst)) worst = lane;
    }

    if (best == worst) return;
    // Only graft if worst is actually stagnating
    if (obs_[worst].stagnation_epochs < 4) return;

    // Conservative 25% graft — don't kill the worst, nudge it
    auto& gb = genomes_[best];
    auto& gw = genomes_[worst];
    gw.coupling.store(
        static_cast<std::int16_t>((3 * gw.coupling.load(std::memory_order_relaxed) + gb.coupling.load(std::memory_order_relaxed)) / 4),
        std::memory_order_relaxed);
    gw.blend.store(
        static_cast<std::int16_t>((3 * gw.blend.load(std::memory_order_relaxed) + gb.blend.load(std::memory_order_relaxed)) / 4),
        std::memory_order_relaxed);
    // 50% chance to inherit delta directly
    if ((data_[worst].rng & 1U) == 0U) {
        gw.delta.store(gb.delta.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    // Reset worst stagnation so it gets a fair chance
    obs_[worst].stagnation_epochs = 0;
}

Observation Engine::observation(const std::size_t lane) const {
    std::scoped_lock lock(mutex_);
    if (lane >= kThreads) {
        return {};
    }
    return obs_[lane];
}

GenomeSnapshot Engine::genome(const std::size_t lane) const {
    std::scoped_lock lock(mutex_);
    if (lane >= kThreads) {
        return {};
    }
    const auto& genome = genomes_[lane];
    return GenomeSnapshot{
        .delta = genome.delta.load(std::memory_order_relaxed),
        .coupling = genome.coupling.load(std::memory_order_relaxed),
        .threshold = genome.threshold.load(std::memory_order_relaxed),
        .blend = genome.blend.load(std::memory_order_relaxed),
        .decay = genome.decay.load(std::memory_order_relaxed),
        .inject_rate = genome.inject_rate.load(std::memory_order_relaxed),
        .generation = genome.generation.load(std::memory_order_relaxed),
        .fitness = genome.fitness.load(std::memory_order_relaxed),
        .best_fitness = genome.best_fitness.load(std::memory_order_relaxed),
        .total_mutations = genome.total_mutations.load(std::memory_order_relaxed),
    };
}

std::uint32_t Engine::total_mutations() const {
    std::scoped_lock lock(mutex_);
    std::uint32_t total = 0;
    for (const auto& genome : genomes_) {
        total += genome.total_mutations.load(std::memory_order_relaxed);
    }
    return total;
}

} // namespace antigravity::control::living
