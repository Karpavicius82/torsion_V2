#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace antigravity::control::living {

inline constexpr std::size_t kNodes = 2048;
inline constexpr std::size_t kThreads = 8;
inline constexpr std::uint64_t kEpochMask = 0x7f;
inline constexpr std::size_t kCrossoverEpochs = 4;  // crossover every N epochs

struct alignas(64) Genome {
    std::atomic<std::int16_t> delta{17};
    std::atomic<std::int16_t> coupling{64};
    std::atomic<std::int16_t> threshold{8000};
    std::atomic<std::int16_t> blend{192};
    std::atomic<std::int16_t> decay{0};
    std::atomic<std::int16_t> inject_rate{64};
    std::atomic<std::uint32_t> generation{0};
    std::atomic<std::int32_t> fitness{0};
    std::atomic<std::int32_t> best_fitness{0};
    std::atomic<std::uint32_t> total_mutations{0};
};

struct GenomeSnapshot {
    std::int16_t delta{17};
    std::int16_t coupling{64};
    std::int16_t threshold{8000};
    std::int16_t blend{192};
    std::int16_t decay{0};
    std::int16_t inject_rate{64};
    std::uint32_t generation{0};
    std::int32_t fitness{0};
    std::int32_t best_fitness{0};
    std::uint32_t total_mutations{0};
};

struct alignas(64) Observation {
    std::int64_t energy{0};
    std::int64_t coherence{0};
    std::int32_t nd_popcount{0};
    std::int32_t pressure{0};      // backward-compatible EMA in microseconds
    std::int64_t tick_ns{0};       // last lane tick duration in nanoseconds
    std::int64_t tick_ns_ema{0};   // EMA of lane tick duration in nanoseconds
    std::uint32_t mutations{0};
    std::uint32_t improvements{0};
    // Controller telemetry (mirrored from ControllerState)
    std::int32_t fitness_ema{0};
    std::int32_t recent_drive{0};       // stimulus strength (0=idle)
    std::uint16_t stagnation_epochs{0};
    std::uint16_t membrane_local{0};
    std::uint32_t attention_hits{0};
};

// Internal per-lane runtime state — reacts fast, does not touch Genome
struct alignas(64) ControllerState {
    std::int32_t fitness_ema{0};
    std::int32_t prev_fitness_ema{0};
    std::int32_t prev_energy_bucket{0};
    std::int32_t recent_drive{0};        // EMA of inject signal strength
    std::uint16_t stagnation_epochs{0};
    std::uint16_t membrane_local{0};
    std::uint32_t attention_hits{0};
};

struct alignas(64) ThreadState {
    std::array<std::int16_t, kNodes> mag{};
    std::array<std::uint16_t, kNodes> ph{};
    std::array<std::uint64_t, 4> nd{};
    std::uint32_t rng{0};
    std::uint64_t tick_counter{0};
};

class Engine {
public:
    Engine();

    void initialize(std::uint64_t seed);
    void inject(std::size_t lane, const std::int16_t* signal, std::size_t n);
    void tick(std::uint64_t ticks);

    void set_collective(bool enabled) { enable_collective_ = enabled; }
    [[nodiscard]] bool collective_enabled() const { return enable_collective_; }

    [[nodiscard]] Observation observation(std::size_t lane) const;
    [[nodiscard]] GenomeSnapshot genome(std::size_t lane) const;
    [[nodiscard]] std::uint32_t total_mutations() const;

private:
    void advance_lane(std::size_t lane);
    void maybe_mutate(std::size_t lane, std::int64_t tick_energy);
    void compute_membrane();
    void maybe_crossover();

    bool enable_collective_{false};  // feature flag — kill switch
    std::uint64_t global_tick_{0};
    std::array<std::uint8_t, kNodes> membrane_{};  // disagreement field

    mutable std::mutex mutex_;
    std::array<Genome, kThreads> genomes_{};
    std::array<Observation, kThreads> obs_{};
    std::array<ThreadState, kThreads> data_{};
    std::array<ControllerState, kThreads> ctrl_{};
};

} // namespace antigravity::control::living
