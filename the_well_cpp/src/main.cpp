// ============================================================================
// THE WELL C++ — Pure C++ Physics Simulation Benchmark
//
// Replaces PolymathicAI/the_well (Python/PyTorch) with bare-metal C++.
//
// Architecture:
//   1. Living Silicon engine GENERATES torsion field trajectories
//   2. Surrogate models LEARN to predict the next timestep
//   3. All training/evaluation in pure C++ with AVX2+FMA
//
// Models: FNO-1D, UNet-1D, DilatedConvNet-1D
//
// Build:
//   g++ -O3 -mavx2 -mfma -static -std=c++20 -o the_well_cpp src/main.cpp
//
// Usage:
//   ./the_well_cpp [model] [epochs] [lr]
//   model:  fno | unet | convnet   (default: fno)
//   epochs: number of training epochs (default: 50)
//   lr:     learning rate (default: 1e-3)
//
// Copyright 2026 — Karpavicius82.  Jokio Python.  Jokio PyTorch.
// ============================================================================

#ifdef _WIN32
  #define PLATFORM_WIN
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
#else
  #define PLATFORM_LINUX
  #include <sys/time.h>
  #include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "tensor.hpp"
#include "engine.hpp"
#include "optimizer.hpp"
#include "models/model.hpp"
#include "models/fno.hpp"
#include "models/unet.hpp"
#include "models/conv_net.hpp"

// ============================================================================
// PLATFORM
// ============================================================================

static uint64_t now_us() {
#ifdef PLATFORM_WIN
    static LARGE_INTEGER freq = []{ LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (uint64_t)(c.QuadPart * 1000000ULL / freq.QuadPart);
#else
    struct timeval tv; gettimeofday(&tv, nullptr);
    return (uint64_t)tv.tv_sec * 1000000ULL + tv.tv_usec;
#endif
}

static void enable_ansi() {
#ifdef PLATFORM_WIN
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD m = 0; GetConsoleMode(h, &m);
    SetConsoleMode(h, m | 0x0004);
#endif
}

// ============================================================================
// DATASET: Generated from Living Silicon Engine
// ============================================================================

namespace dataset {

// Subsample width: use every 8th node to make training tractable
constexpr int SUBSAMPLE = 32;
constexpr int WIDTH = well::engine::N / SUBSAMPLE;  // 64
constexpr int HISTORY = 4;    // input: 4 timesteps
constexpr int STRIDE  = 4;    // sample every 4 ticks
constexpr int SEQ_LEN = 64;   // trajectory length (in samples)
constexpr int TRAIN_SEQS = 32;   // number of training sequences
constexpr int EVAL_SEQS  = 8;    // number of eval sequences

struct Sample {
    float input[HISTORY * WIDTH];    // [history, width]
    float target[WIDTH];             // next timestep
};

static Sample train_data[TRAIN_SEQS * (SEQ_LEN - HISTORY)];
static Sample eval_data[EVAL_SEQS * (SEQ_LEN - HISTORY)];
static int train_count = 0;
static int eval_count  = 0;

// Generate dataset from engine
static void generate(bool verbose = true) {
    if (verbose) fprintf(stderr, "  Generating training data from Living Silicon engine...\n");

    well::engine::initialize(0xC0DE7052ULL);
    well::engine::inject_gaussian();

    // Warmup: run engine 1000 ticks to reach interesting dynamics
    for (int t = 0; t < 1000; ++t) {
        for (int l = 0; l < well::engine::LANES; ++l)
            well::engine::advance_lane(l, t);
    }

    uint64_t tick = 1000;

    // Generate full field trajectories, then subsample spatially
    auto extract_subsampled = [](int lane, float* out) {
        const float scale = 1.0f / 32768.0f;
        for (int j = 0; j < WIDTH; ++j)
            out[j] = (float)well::engine::lanes[lane].mag[j * SUBSAMPLE] * scale;
    };

    // Training data: from first half of lanes
    train_count = 0;
    for (int seq = 0; seq < TRAIN_SEQS; ++seq) {
        int lane = seq % well::engine::LANES;

        // Run engine to generate a sequence
        float frames[SEQ_LEN][WIDTH];
        for (int f = 0; f < SEQ_LEN; ++f) {
            for (int s = 0; s < STRIDE; ++s) {
                well::engine::advance_lane(lane, tick++);
            }
            extract_subsampled(lane, frames[f]);
        }

        // Create sliding window samples
        for (int i = 0; i + HISTORY < SEQ_LEN; ++i) {
            Sample& s = train_data[train_count++];
            for (int h = 0; h < HISTORY; ++h)
                memcpy(s.input + h * WIDTH, frames[i + h], WIDTH * sizeof(float));
            memcpy(s.target, frames[i + HISTORY], WIDTH * sizeof(float));
        }

        // Final crossover every 128 ticks
        if ((seq & 0xF) == 0) well::engine::crossover();
    }

    // Eval data: continue from current state
    eval_count = 0;
    for (int seq = 0; seq < EVAL_SEQS; ++seq) {
        int lane = seq % well::engine::LANES;
        float frames[SEQ_LEN][WIDTH];
        for (int f = 0; f < SEQ_LEN; ++f) {
            for (int s = 0; s < STRIDE; ++s)
                well::engine::advance_lane(lane, tick++);
            extract_subsampled(lane, frames[f]);
        }
        for (int i = 0; i + HISTORY < SEQ_LEN; ++i) {
            Sample& s = eval_data[eval_count++];
            for (int h = 0; h < HISTORY; ++h)
                memcpy(s.input + h * WIDTH, frames[i + h], WIDTH * sizeof(float));
            memcpy(s.target, frames[i + HISTORY], WIDTH * sizeof(float));
        }
    }

    if (verbose) {
        fprintf(stderr, "  Training samples: %d | Eval samples: %d | Width: %d\n",
                train_count, eval_count, WIDTH);
        fprintf(stderr, "  Engine state: energy=%lld  mutations=%u  fitness=%d\n",
                (long long)well::engine::stats[0].energy,
                well::engine::total_mutations(),
                well::engine::genomes[0].fitness);
    }
}

// Simple shuffle
static void shuffle(Sample* data, int count, well::Rng& rng) {
    for (int i = count - 1; i > 0; --i) {
        int j = rng.next() % (uint32_t)(i + 1);
        Sample tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

} // namespace dataset

// ============================================================================
// DASHBOARD
// ============================================================================

static void dashboard(int epoch, int total_epochs, float train_loss, float eval_loss,
                      float lr, float epoch_time_s, const char* model_name,
                      int params, const char* phase) {
    static float best_eval = 1e9f;
    if (eval_loss < best_eval && eval_loss > 0) best_eval = eval_loss;

    float progress = (float)(epoch + 1) / (float)total_epochs;
    int bar_width = 40;
    int filled = (int)(progress * bar_width);

    fprintf(stderr,
        "\033[H\033[44;1;37m THE WELL C++ | Living Silicon Benchmark | Pure C++/AVX2          \033[0m\n"
        " \033[36mPDE: d_tt S = c²·d_xx S - m²·S + g·S³  [Einstein-Cartan]\033[0m\n\n"
        " Model:  \033[1m%-20s\033[0m  Params: \033[33m%d\033[0m\n"
        " Phase:  \033[1;32m%-10s\033[0m\n\n"
        " Epoch:  \033[1m%d/%d\033[0m [",
        model_name, params, phase, epoch + 1, total_epochs);

    for (int i = 0; i < bar_width; ++i)
        fputc(i < filled ? '#' : '-', stderr);

    fprintf(stderr,
        "] %.0f%%\n\n"
        " Train Loss:  \033[33m%.6f\033[0m\n"
        " Eval  Loss:  \033[%sm%.6f\033[0m  (best: %.6f)\n"
        " Learning Rate: %.2e\n"
        " Epoch Time:    %.2f s\n\n",
        progress * 100,
        train_loss,
        eval_loss <= best_eval * 1.1f ? "1;32" : "31",
        eval_loss, best_eval,
        lr, epoch_time_s);

    // Engine status
    int best_lane = 0;
    for (int l = 1; l < well::engine::LANES; ++l)
        if (well::engine::stats[l].energy > well::engine::stats[best_lane].energy) best_lane = l;
    auto& st = well::engine::stats[best_lane];
    auto& G  = well::engine::genomes[best_lane];

    fprintf(stderr,
        "\033[44;1;37m LIVING SILICON ENGINE STATUS                                    \033[0m\n"
        " Energy:    \033[32m%-12lld\033[0m  Coherence: %-12lld\n"
        " Mutations: %-8u         Fitness:   %d\n"
        " Genome: delta=%-4d coupling=%-4d blend=%-4d decay=%d\n",
        (long long)st.energy, (long long)st.coherence,
        well::engine::total_mutations(), G.fitness,
        G.delta, G.coupling, G.blend, G.decay);
}

// ============================================================================
// TRAINING LOOP
// ============================================================================

static void train_model(well::Model* model, int epochs, float lr,
                        const char* model_name) {
    well::Adam opt;
    opt.init(*model, lr, 0.9f, 0.999f, 1e-8f, 1e-5f);

    well::CosineScheduler sched;
    sched.init(lr, epochs, 5);

    well::Rng shuffle_rng;
    shuffle_rng.seed(42);

    // CSV header
    printf("epoch,train_loss,eval_loss,lr,params,model,time_s\n");
    fflush(stdout);

    well::Tensor pred = well::Tensor::alloc(dataset::WIDTH);
    well::Tensor d_pred = well::Tensor::alloc(dataset::WIDTH);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        uint64_t t0 = now_us();

        // Update LR
        sched.apply(opt, epoch);

        // ── Train ──
        dataset::shuffle(dataset::train_data, dataset::train_count, shuffle_rng);
        float train_loss_sum = 0;
        int train_n = 0;

        for (int i = 0; i < dataset::train_count; ++i) {
            auto& sample = dataset::train_data[i];

            well::Tensor input = well::Tensor::view(sample.input,
                                    dataset::HISTORY, dataset::WIDTH);
            well::Tensor target = well::Tensor::view(sample.target, dataset::WIDTH);

            // Zero gradients
            model->zero_grad();

            // Forward
            pred.zero();
            model->forward(input, pred, 1, dataset::HISTORY, dataset::WIDTH);

            // MSE Loss
            float loss = pred.mse(target);
            train_loss_sum += loss;
            train_n++;

            // d_loss/d_pred = 2*(pred - target) / N
            float inv_n = 2.0f / (float)dataset::WIDTH;
            for (int j = 0; j < dataset::WIDTH; ++j)
                d_pred[j] = (pred[j] - target[j]) * inv_n;

            // Backward
            model->backward(d_pred, input, 1, dataset::HISTORY, dataset::WIDTH);

            // Gradient clipping (max norm = 1.0)
            float grad_norm = 0;
            for (int p = 0; p < model->num_params(); ++p)
                grad_norm += model->param(p).grad.dot(model->param(p).grad);
            grad_norm = sqrtf(grad_norm);
            if (grad_norm > 1.0f) {
                float scale = 1.0f / grad_norm;
                for (int p = 0; p < model->num_params(); ++p)
                    model->param(p).grad.scale(scale);
            }

            // Optimizer step
            opt.step(*model);
        }

        float train_loss = train_loss_sum / (float)train_n;

        // ── Eval ──
        float eval_loss_sum = 0;
        int eval_n = 0;

        for (int i = 0; i < dataset::eval_count; ++i) {
            auto& sample = dataset::eval_data[i];
            well::Tensor input = well::Tensor::view(sample.input,
                                    dataset::HISTORY, dataset::WIDTH);
            well::Tensor target = well::Tensor::view(sample.target, dataset::WIDTH);

            pred.zero();
            model->forward(input, pred, 1, dataset::HISTORY, dataset::WIDTH);
            eval_loss_sum += pred.mse(target);
            eval_n++;
        }

        float eval_loss = eval_loss_sum / (float)eval_n;
        float epoch_time = (float)(now_us() - t0) / 1e6f;

        // Dashboard
        dashboard(epoch, epochs, train_loss, eval_loss, opt.lr, epoch_time,
                  model_name, model->param_count(), "Training");

        // CSV
        printf("%d,%.8f,%.8f,%.2e,%d,%s,%.2f\n",
               epoch, train_loss, eval_loss, opt.lr,
               model->param_count(), model_name, epoch_time);
        fflush(stdout);
    }

    pred.release();
    d_pred.release();
    opt.release();
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    enable_ansi();
    fprintf(stderr, "\033[2J");

    // Parse args
    const char* model_name = (argc > 1) ? argv[1] : "fno";
    int epochs = (argc > 2) ? atoi(argv[2]) : 50;
    float lr   = (argc > 3) ? (float)atof(argv[3]) : 1e-3f;

    fprintf(stderr,
        "\033[44;1;37m THE WELL C++ — Pure C++/AVX2 Physics Benchmark                  \033[0m\n\n"
        " \033[1;36mNo Python. No PyTorch. No dependencies.\033[0m\n"
        " Model: \033[1m%s\033[0m | Epochs: %d | LR: %.1e\n\n",
        model_name, epochs, lr);

    // ── Generate dataset ──
    uint64_t gen_t0 = now_us();
    dataset::generate(true);
    float gen_time = (float)(now_us() - gen_t0) / 1e6f;
    fprintf(stderr, "  Data generated in %.2f s\n\n", gen_time);

    // ── Create model ──
    well::Rng init_rng;
    init_rng.seed(0xBEEF42ULL);

    if (strcmp(model_name, "fno") == 0) {
        well::FNO fno;
        fno.init(dataset::HISTORY, dataset::WIDTH, init_rng);
        fno.print_summary();
        train_model(&fno, epochs, lr, "FNO-1D");
        fno.release();

    } else if (strcmp(model_name, "unet") == 0) {
        well::UNet1D unet;
        unet.init(dataset::HISTORY, dataset::WIDTH, init_rng);
        unet.print_summary();
        train_model(&unet, epochs, lr, "UNet-1D");
        unet.release();

    } else if (strcmp(model_name, "convnet") == 0) {
        well::ConvNet1D cnet;
        cnet.init(dataset::HISTORY, dataset::WIDTH, init_rng);
        cnet.print_summary();
        train_model(&cnet, epochs, lr, "DilatedConvNet-1D");
        cnet.release();

    } else if (strcmp(model_name, "all") == 0) {
        fprintf(stderr, "\n\033[1;33m═══ BENCHMARK: ALL MODELS ═══\033[0m\n\n");

        // FNO
        {
            well::FNO fno;
            fno.init(dataset::HISTORY, dataset::WIDTH, init_rng);
            fno.print_summary();
            train_model(&fno, epochs, lr, "FNO-1D");
            fno.release();
        }

        // Regenerate data for fair comparison
        dataset::generate(false);
        init_rng.seed(0xBEEF42ULL);

        // ConvNet
        {
            well::ConvNet1D cnet;
            cnet.init(dataset::HISTORY, dataset::WIDTH, init_rng);
            cnet.print_summary();
            train_model(&cnet, epochs, lr, "DilatedConvNet-1D");
            cnet.release();
        }

        dataset::generate(false);
        init_rng.seed(0xBEEF42ULL);

        // UNet
        {
            well::UNet1D unet;
            unet.init(dataset::HISTORY, dataset::WIDTH, init_rng);
            unet.print_summary();
            train_model(&unet, epochs, lr, "UNet-1D");
            unet.release();
        }
    } else {
        fprintf(stderr, "\033[31mUnknown model: %s\033[0m\n", model_name);
        fprintf(stderr, "Available: fno, unet, convnet, all\n");
        return 1;
    }

    // ── Final summary ──
    fprintf(stderr,
        "\n\033[44;1;37m THE WELL C++ — COMPLETE                                         \033[0m\n"
        " \033[1;32mBenchmark finished. Results in CSV output.\033[0m\n"
        " Engine mutations: %u | Best fitness: %d\n\n",
        well::engine::total_mutations(),
        well::engine::genomes[0].best_fitness);

    return 0;
}
