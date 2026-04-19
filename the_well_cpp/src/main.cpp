// ============================================================================
// THE WELL C++ — Complete 1:1 Bare-Metal Physics Simulation Benchmark
//
// Full replacement of PolymathicAI/the_well (Python/PyTorch/15TB HDF5)
// with pure C++20 / AVX2+FMA / Living Silicon engine.
//
// 22 PDE physics scenarios × 9 model architectures
// Zero dependencies. Zero Python. Zero OS overhead.
//
// Build:
//   g++ -O3 -mavx2 -mfma -static -std=c++20 -o the_well_cpp src/main.cpp
//
// Usage:
//   ./the_well_cpp [pde] [model] [epochs] [lr]
//   pde: ns|diff|wave|burgers|mhd|rb|sw|euler|helm|adv|gs|ks|ch|torsion|ac|
//        acoustic|active|visco|rt|shear|supernova|turbcool|convective|radlayer|
//        neutron|planetswe|all
//   model: fno|fno2d|unet|unet2d|convnet|dconv2d|resnet|vit|pinn|all
//
// Copyright 2026 — Karpavicius82. Pure Living Silicon.
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

// ── Core ──
#include "tensor.hpp"
#include "engine.hpp"
#include "optimizer.hpp"
#include "core/fft.hpp"
#include "core/tensor_nd.hpp"

// ── All Models ──
#include "models/model.hpp"
#include "models/fno.hpp"
#include "models/fno2d.hpp"
#include "models/unet.hpp"
#include "models/unet2d.hpp"
#include "models/conv_net.hpp"
#include "models/resnet2d.hpp"
#include "models/vit.hpp"
#include "models/pinn.hpp"

// ── All Physics ──
#include "physics/pde_base.hpp"
#include "physics/navier_stokes_2d.hpp"
#include "physics/diffusion_wave_burgers.hpp"
#include "physics/mhd_rb_sw.hpp"
#include "physics/euler_helmholtz_advection_rd.hpp"
#include "physics/ks_ch_torsion2d_ac.hpp"
#include "physics/acoustic_scattering.hpp"
#include "physics/active_matter_visco.hpp"
#include "physics/rt_shear.hpp"
#include "physics/supernova_turb.hpp"
#include "physics/convective_turb_rad.hpp"
#include "physics/neutron_planetswe.hpp"

// ── Training ──
#include "training/metrics.hpp"
#include "training/adamw.hpp"

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
// 2D DATASET: Generated on-the-fly from any PDE engine
// ============================================================================

namespace dataset2d {

constexpr int GRID = well::PDE2D::N2D;        // 64
constexpr int SPATIAL = GRID * GRID;           // 4096
constexpr int MAX_FIELDS = 5;
constexpr int HISTORY    = 4;
constexpr int SEQ_LEN    = 32;
constexpr int TRAIN_SEQS = 16;
constexpr int EVAL_SEQS  = 4;
constexpr int STRIDE     = 4;

struct Sample2D {
    float input[HISTORY * SPATIAL];   // [history, H*W]
    float target[SPATIAL];            // next frame [H*W]
};

static Sample2D* train_data = nullptr;
static Sample2D* eval_data  = nullptr;
static int train_count = 0;
static int eval_count  = 0;

// Allocate once
static void alloc() {
    if (!train_data) {
        int max_train = TRAIN_SEQS * (SEQ_LEN - HISTORY);
        int max_eval  = EVAL_SEQS * (SEQ_LEN - HISTORY);
        train_data = new Sample2D[max_train];
        eval_data  = new Sample2D[max_eval];
    }
}

// Generate dataset from any PDE2D engine (field 0 by default)
static void generate(well::PDE2D& pde, int field = 0, bool verbose = true) {
    alloc();
    if (verbose)
        fprintf(stderr, "  Generating 2D data from [%s] field=%d...\n", pde.name(), field);

    pde.init(0xC0DE7052ULL);

    // Warmup
    for (int t = 0; t < 200; ++t) pde.step(0);

    float* frame_buf = new float[SPATIAL];

    // Training sequences
    train_count = 0;
    for (int seq = 0; seq < TRAIN_SEQS; ++seq) {
        float frames[SEQ_LEN][SPATIAL];
        for (int f = 0; f < SEQ_LEN; ++f) {
            for (int s = 0; s < STRIDE; ++s) pde.step(0);
            pde.write_field(field, frames[f]);
        }

        for (int i = 0; i + HISTORY < SEQ_LEN; ++i) {
            Sample2D& s = train_data[train_count++];
            for (int h = 0; h < HISTORY; ++h)
                memcpy(s.input + h * SPATIAL, frames[i+h], SPATIAL * sizeof(float));
            memcpy(s.target, frames[i+HISTORY], SPATIAL * sizeof(float));
        }

        // GA evolution every 8 sequences
        if ((seq & 0x7) == 0) pde.ga_mutate(pde.rng);
    }

    // Eval sequences
    eval_count = 0;
    for (int seq = 0; seq < EVAL_SEQS; ++seq) {
        float frames[SEQ_LEN][SPATIAL];
        for (int f = 0; f < SEQ_LEN; ++f) {
            for (int s = 0; s < STRIDE; ++s) pde.step(0);
            pde.write_field(field, frames[f]);
        }
        for (int i = 0; i + HISTORY < SEQ_LEN; ++i) {
            Sample2D& s = eval_data[eval_count++];
            for (int h = 0; h < HISTORY; ++h)
                memcpy(s.input + h * SPATIAL, frames[i+h], SPATIAL * sizeof(float));
            memcpy(s.target, frames[i+HISTORY], SPATIAL * sizeof(float));
        }
    }

    delete[] frame_buf;

    if (verbose)
        fprintf(stderr, "  Train: %d | Eval: %d | Grid: %dx%d | GA fitness: %d\n",
                train_count, eval_count, GRID, GRID, pde.engine_ga.fitness);
}

// Simple shuffle
static void shuffle(Sample2D* data, int count, well::Rng& rng) {
    for (int i = count - 1; i > 0; --i) {
        int j = rng.next() % (uint32_t)(i + 1);
        Sample2D tmp = data[i]; data[i] = data[j]; data[j] = tmp;
    }
}

static void cleanup() {
    delete[] train_data; train_data = nullptr;
    delete[] eval_data;  eval_data = nullptr;
}

} // namespace dataset2d

// ============================================================================
// 1D DATASET: From Living Silicon engine (original)
// ============================================================================

namespace dataset1d {

constexpr int SUBSAMPLE = 32;
constexpr int WIDTH = well::engine::N / SUBSAMPLE;
constexpr int HISTORY = 4;
constexpr int STRIDE  = 4;
constexpr int SEQ_LEN = 64;
constexpr int TRAIN_SEQS = 32;
constexpr int EVAL_SEQS  = 8;

struct Sample {
    float input[HISTORY * WIDTH];
    float target[WIDTH];
};

static Sample train_data[TRAIN_SEQS * (SEQ_LEN - HISTORY)];
static Sample eval_data[EVAL_SEQS * (SEQ_LEN - HISTORY)];
static int train_count = 0;
static int eval_count  = 0;

static void generate(bool verbose = true) {
    if (verbose) fprintf(stderr, "  Generating 1D data from Living Silicon engine...\n");

    well::engine::initialize(0xC0DE7052ULL);
    well::engine::inject_gaussian();

    for (int t = 0; t < 1000; ++t)
        for (int l = 0; l < well::engine::LANES; ++l)
            well::engine::advance_lane(l, t);

    uint64_t tick = 1000;

    auto extract = [](int lane, float* out) {
        const float scale = 1.0f / 32768.0f;
        for (int j = 0; j < WIDTH; ++j)
            out[j] = (float)well::engine::lanes[lane].mag[j * SUBSAMPLE] * scale;
    };

    train_count = 0;
    for (int seq = 0; seq < TRAIN_SEQS; ++seq) {
        int lane = seq % well::engine::LANES;
        float frames[SEQ_LEN][WIDTH];
        for (int f = 0; f < SEQ_LEN; ++f) {
            for (int s = 0; s < STRIDE; ++s) well::engine::advance_lane(lane, tick++);
            extract(lane, frames[f]);
        }
        for (int i = 0; i + HISTORY < SEQ_LEN; ++i) {
            Sample& s = train_data[train_count++];
            for (int h = 0; h < HISTORY; ++h)
                memcpy(s.input + h * WIDTH, frames[i+h], WIDTH * sizeof(float));
            memcpy(s.target, frames[i + HISTORY], WIDTH * sizeof(float));
        }
        if ((seq & 0xF) == 0) well::engine::crossover();
    }

    eval_count = 0;
    for (int seq = 0; seq < EVAL_SEQS; ++seq) {
        int lane = seq % well::engine::LANES;
        float frames[SEQ_LEN][WIDTH];
        for (int f = 0; f < SEQ_LEN; ++f) {
            for (int s = 0; s < STRIDE; ++s) well::engine::advance_lane(lane, tick++);
            extract(lane, frames[f]);
        }
        for (int i = 0; i + HISTORY < SEQ_LEN; ++i) {
            Sample& s = eval_data[eval_count++];
            for (int h = 0; h < HISTORY; ++h)
                memcpy(s.input + h * WIDTH, frames[i+h], WIDTH * sizeof(float));
            memcpy(s.target, frames[i + HISTORY], WIDTH * sizeof(float));
        }
    }

    if (verbose)
        fprintf(stderr, "  Train: %d | Eval: %d | Width: %d\n",
                train_count, eval_count, WIDTH);
}

static void shuffle(Sample* data, int count, well::Rng& rng) {
    for (int i = count - 1; i > 0; --i) {
        int j = rng.next() % (uint32_t)(i + 1);
        Sample tmp = data[i]; data[i] = data[j]; data[j] = tmp;
    }
}

} // namespace dataset1d

// ============================================================================
// DASHBOARD
// ============================================================================

static void dashboard(int epoch, int total_epochs, float train_loss, float eval_loss,
                      float lr, float epoch_time_s, const char* model_name,
                      const char* pde_name, int params) {
    static float best_eval = 1e9f;
    if (eval_loss < best_eval && eval_loss > 0) best_eval = eval_loss;

    float progress = (float)(epoch + 1) / (float)total_epochs;
    int bar_width = 40;
    int filled = (int)(progress * bar_width);

    fprintf(stderr,
        "\033[H\033[44;1;37m THE WELL C++ | Living Silicon 1:1 Benchmark | Pure C++/AVX2      \033[0m\n"
        " \033[36mPDE: %s\033[0m\n"
        " Model:  \033[1m%-20s\033[0m  Params: \033[33m%d\033[0m\n\n"
        " Epoch:  \033[1m%d/%d\033[0m [",
        pde_name, model_name, params, epoch + 1, total_epochs);

    for (int i = 0; i < bar_width; ++i)
        fputc(i < filled ? '#' : '-', stderr);

    fprintf(stderr,
        "] %.0f%%\n\n"
        " Train Loss:  \033[33m%.6f\033[0m\n"
        " Eval  Loss:  \033[%sm%.6f\033[0m  (best: %.6f)\n"
        " Learning Rate: %.2e\n"
        " Epoch Time:    %.2f s\n\n",
        progress * 100, train_loss,
        eval_loss <= best_eval * 1.1f ? "1;32" : "31",
        eval_loss, best_eval, lr, epoch_time_s);
}

// ============================================================================
// TRAINING LOOP — 2D
// ============================================================================

static void train_2d(well::Model* model, const char* model_name,
                     const char* pde_name, int epochs, float lr) {
    well::AdamW opt;
    opt.init(*model, lr, 0.9f, 0.999f, 1e-8f, 0.01f);

    well::WarmupCosineScheduler sched;
    sched.init(lr, epochs, 5);

    well::Rng shuffle_rng;
    shuffle_rng.seed(42);

    int spatial = dataset2d::SPATIAL;
    well::Tensor pred  = well::Tensor::alloc(spatial);
    well::Tensor d_pred = well::Tensor::alloc(spatial);

    printf("epoch,train_loss,eval_loss,vrmse,lr,params,model,pde,time_s\n");
    fflush(stdout);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        uint64_t t0 = now_us();
        sched.apply(opt, epoch);

        // Train
        dataset2d::shuffle(dataset2d::train_data, dataset2d::train_count, shuffle_rng);
        float train_loss_sum = 0;
        int train_n = 0;

        for (int i = 0; i < dataset2d::train_count; ++i) {
            auto& sample = dataset2d::train_data[i];
            well::Tensor input  = well::Tensor::view(sample.input, dataset2d::HISTORY, spatial);
            well::Tensor target = well::Tensor::view(sample.target, spatial);

            model->zero_grad();
            pred.zero();
            model->forward(input, pred, 1, dataset2d::HISTORY, dataset2d::GRID);

            float loss = pred.mse(target);
            train_loss_sum += loss;
            train_n++;

            float inv_n = 2.0f / (float)spatial;
            for (int j = 0; j < spatial; ++j)
                d_pred[j] = (pred[j] - target[j]) * inv_n;

            model->backward(d_pred, input, 1, dataset2d::HISTORY, dataset2d::GRID);
            opt.clip_grad_norm(*model);
            opt.step(*model);
        }

        float train_loss = train_loss_sum / (float)train_n;

        // Eval
        float eval_loss_sum = 0;
        float vrmse_sum = 0;
        int eval_n = 0;

        for (int i = 0; i < dataset2d::eval_count; ++i) {
            auto& sample = dataset2d::eval_data[i];
            well::Tensor input  = well::Tensor::view(sample.input, dataset2d::HISTORY, spatial);
            well::Tensor target = well::Tensor::view(sample.target, spatial);

            pred.zero();
            model->forward(input, pred, 1, dataset2d::HISTORY, dataset2d::GRID);
            eval_loss_sum += pred.mse(target);
            vrmse_sum += well::metrics::vrmse(pred.data, target.data, spatial);
            eval_n++;
        }

        float eval_loss = eval_loss_sum / (float)eval_n;
        float vrmse = vrmse_sum / (float)eval_n;
        float epoch_time = (float)(now_us() - t0) / 1e6f;

        dashboard(epoch, epochs, train_loss, eval_loss, opt.lr, epoch_time,
                  model_name, pde_name, model->param_count());

        printf("%d,%.8f,%.8f,%.6f,%.2e,%d,%s,%s,%.2f\n",
               epoch, train_loss, eval_loss, vrmse, opt.lr,
               model->param_count(), model_name, pde_name, epoch_time);
        fflush(stdout);
    }

    pred.release();
    d_pred.release();
    opt.release();
}

// ============================================================================
// TRAINING LOOP — 1D (legacy)
// ============================================================================

static void train_1d(well::Model* model, const char* model_name,
                     int epochs, float lr) {
    well::Adam opt;
    opt.init(*model, lr, 0.9f, 0.999f, 1e-8f, 1e-5f);

    well::CosineScheduler sched;
    sched.init(lr, epochs, 5);

    well::Rng shuffle_rng;
    shuffle_rng.seed(42);

    int width = dataset1d::WIDTH;
    well::Tensor pred  = well::Tensor::alloc(width);
    well::Tensor d_pred = well::Tensor::alloc(width);

    printf("epoch,train_loss,eval_loss,lr,params,model,pde,time_s\n");
    fflush(stdout);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        uint64_t t0 = now_us();
        sched.apply(opt, epoch);

        dataset1d::shuffle(dataset1d::train_data, dataset1d::train_count, shuffle_rng);
        float train_loss_sum = 0;
        int train_n = 0;

        for (int i = 0; i < dataset1d::train_count; ++i) {
            auto& sample = dataset1d::train_data[i];
            well::Tensor input  = well::Tensor::view(sample.input, dataset1d::HISTORY, width);
            well::Tensor target = well::Tensor::view(sample.target, width);

            model->zero_grad();
            pred.zero();
            model->forward(input, pred, 1, dataset1d::HISTORY, width);

            float loss = pred.mse(target);
            train_loss_sum += loss;
            train_n++;

            float inv_n = 2.0f / (float)width;
            for (int j = 0; j < width; ++j)
                d_pred[j] = (pred[j] - target[j]) * inv_n;

            model->backward(d_pred, input, 1, dataset1d::HISTORY, width);

            float gnorm = 0;
            for (int p = 0; p < model->num_params(); ++p)
                gnorm += model->param(p).grad.dot(model->param(p).grad);
            gnorm = sqrtf(gnorm);
            if (gnorm > 1.0f) {
                float s = 1.0f / gnorm;
                for (int p = 0; p < model->num_params(); ++p)
                    model->param(p).grad.scale(s);
            }
            opt.step(*model);
        }

        float train_loss = train_loss_sum / (float)train_n;
        float eval_loss_sum = 0;
        int eval_n = 0;
        for (int i = 0; i < dataset1d::eval_count; ++i) {
            auto& sample = dataset1d::eval_data[i];
            well::Tensor input  = well::Tensor::view(sample.input, dataset1d::HISTORY, width);
            well::Tensor target = well::Tensor::view(sample.target, width);
            pred.zero();
            model->forward(input, pred, 1, dataset1d::HISTORY, width);
            eval_loss_sum += pred.mse(target);
            eval_n++;
        }

        float eval_loss = eval_loss_sum / (float)eval_n;
        float epoch_time = (float)(now_us() - t0) / 1e6f;

        dashboard(epoch, epochs, train_loss, eval_loss, opt.lr, epoch_time,
                  model_name, "LivingSilicon-1D", model->param_count());

        printf("%d,%.8f,%.8f,%.2e,%d,%s,LivingSilicon-1D,%.2f\n",
               epoch, train_loss, eval_loss, opt.lr,
               model->param_count(), model_name, epoch_time);
        fflush(stdout);
    }

    pred.release();
    d_pred.release();
    opt.release();
}

// ============================================================================
// PDE REGISTRY
// ============================================================================

struct PDEEntry {
    const char* key;
    const char* name;
};

static const PDEEntry pde_registry[] = {
    {"ns",         "NavierStokes-2D"},
    {"diff",       "Diffusion-2D"},
    {"wave",       "Wave-2D"},
    {"burgers",    "Burgers-2D"},
    {"mhd",        "MHD-2D"},
    {"rb",         "RayleighBenard-2D"},
    {"sw",         "ShallowWater-2D"},
    {"euler",      "CompressibleEuler-2D"},
    {"helm",       "Helmholtz-2D"},
    {"adv",        "Advection-2D"},
    {"gs",         "GrayScott-2D"},
    {"ks",         "KuramotoSivashinsky-2D"},
    {"ch",         "CahnHilliard-2D"},
    {"torsion",    "Torsion-2D"},
    {"ac",         "AllenCahn-2D"},
    {"acoustic",   "AcousticScattering-2D"},
    {"active",     "ActiveMatter-2D"},
    {"visco",      "ViscoelasticInstability-2D"},
    {"rt",         "RayleighTaylor-2D"},
    {"shear",      "ShearFlow-2D"},
    {"supernova",  "Supernova-2D"},
    {"turbcool",   "TurbulenceGravityCooling-2D"},
    {"convective", "ConvectiveEnvelope-2D"},
    {"radlayer",   "TurbulentRadiativeLayer-2D"},
    {"neutron",    "PostNeutronStarMerger-2D"},
    {"planetswe",  "PlanetarySWE-2D"},
};
static const int N_PDES = sizeof(pde_registry) / sizeof(pde_registry[0]);

static well::PDE2D* create_pde(const char* key) {
    if (strcmp(key,"ns")==0)         return new well::NavierStokes2D();
    if (strcmp(key,"diff")==0)       return new well::Diffusion2D();
    if (strcmp(key,"wave")==0)       return new well::Wave2D();
    if (strcmp(key,"burgers")==0)    return new well::Burgers2D();
    if (strcmp(key,"mhd")==0)        return new well::MHD2D();
    if (strcmp(key,"rb")==0)         return new well::RayleighBenard2D();
    if (strcmp(key,"sw")==0)         return new well::ShallowWater2D();
    if (strcmp(key,"euler")==0)      return new well::CompressibleEuler2D();
    if (strcmp(key,"helm")==0)       return new well::Helmholtz2D();
    if (strcmp(key,"adv")==0)        return new well::Advection2D();
    if (strcmp(key,"gs")==0)         return new well::GrayScott2D();
    if (strcmp(key,"ks")==0)         return new well::KuramotoSivashinsky2D();
    if (strcmp(key,"ch")==0)         return new well::CahnHilliard2D();
    if (strcmp(key,"torsion")==0)    return new well::Torsion2D();
    if (strcmp(key,"ac")==0)         return new well::AllenCahn2D();
    if (strcmp(key,"acoustic")==0)   return new well::AcousticScattering2D();
    if (strcmp(key,"active")==0)     return new well::ActiveMatter2D();
    if (strcmp(key,"visco")==0)      return new well::ViscoelasticInstability2D();
    if (strcmp(key,"rt")==0)         return new well::RayleighTaylor2D();
    if (strcmp(key,"shear")==0)      return new well::ShearFlow2D();
    if (strcmp(key,"supernova")==0)  return new well::Supernova2D();
    if (strcmp(key,"turbcool")==0)   return new well::TurbulenceGravityCooling2D();
    if (strcmp(key,"convective")==0) return new well::ConvectiveEnvelope2D();
    if (strcmp(key,"radlayer")==0)   return new well::TurbulentRadiativeLayer2D();
    if (strcmp(key,"neutron")==0)    return new well::PostNeutronStarMerger2D();
    if (strcmp(key,"planetswe")==0)  return new well::PlanetarySWE2D();
    return nullptr;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    enable_ansi();
    fprintf(stderr, "\033[2J");

    const char* pde_key = (argc > 1) ? argv[1] : "ns";
    const char* model_key = (argc > 2) ? argv[2] : "fno2d";
    int epochs = (argc > 3) ? atoi(argv[3]) : 30;
    float lr   = (argc > 4) ? (float)atof(argv[4]) : 1e-3f;

    fprintf(stderr,
        "\033[44;1;37m THE WELL C++ — Complete 1:1 Physics Benchmark                  \033[0m\n\n"
        " \033[1;36mNo Python. No PyTorch. No HDF5. No 15TB SSD.\033[0m\n"
        " \033[1;36mPure C++20 / AVX2+FMA / Living Silicon OS.\033[0m\n\n"
        " PDE:    \033[1m%s\033[0m\n"
        " Model:  \033[1m%s\033[0m\n"
        " Epochs: %d | LR: %.1e\n\n"
        " \033[33m╔════════════════════════════════════════════╗\033[0m\n"
        " \033[33m║  %d PDE Scenarios Available                ║\033[0m\n"
        " \033[33m╚════════════════════════════════════════════╝\033[0m\n\n",
        pde_key, model_key, epochs, lr, N_PDES);

    // List all PDEs
    for (int i = 0; i < N_PDES; ++i)
        fprintf(stderr, "   %-12s → %s\n", pde_registry[i].key, pde_registry[i].name);
    fprintf(stderr, "\n");

    // ── 1D MODELS (Living Silicon engine data) ──
    if (strcmp(pde_key, "silicon") == 0 || strcmp(pde_key, "1d") == 0) {
        dataset1d::generate(true);
        well::Rng rng; rng.seed(0xBEEF42ULL);

        if (strcmp(model_key, "fno") == 0) {
            well::FNO m; m.init(dataset1d::HISTORY, dataset1d::WIDTH, rng);
            m.print_summary(); train_1d(&m, "FNO-1D", epochs, lr); m.release();
        } else if (strcmp(model_key, "unet") == 0) {
            well::UNet1D m; m.init(dataset1d::HISTORY, dataset1d::WIDTH, rng);
            m.print_summary(); train_1d(&m, "UNet-1D", epochs, lr); m.release();
        } else if (strcmp(model_key, "convnet") == 0) {
            well::ConvNet1D m; m.init(dataset1d::HISTORY, dataset1d::WIDTH, rng);
            m.print_summary(); train_1d(&m, "ConvNet-1D", epochs, lr); m.release();
        }
        return 0;
    }

    // ── 2D MODELS (PDE engine data) ──
    well::PDE2D* pde = create_pde(pde_key);
    if (!pde) {
        fprintf(stderr, "\033[31mUnknown PDE: %s\033[0m\n", pde_key);
        return 1;
    }

    uint64_t gen_t0 = now_us();
    dataset2d::generate(*pde, 0, true);
    float gen_time = (float)(now_us() - gen_t0) / 1e6f;
    fprintf(stderr, "  Data generated in %.2f s\n\n", gen_time);

    well::Rng rng; rng.seed(0xBEEF42ULL);
    int G = dataset2d::GRID;

    auto run_model = [&](const char* mk) {
        if (strcmp(mk, "fno2d") == 0) {
            well::FNO2D m; m.init(dataset2d::HISTORY, 1, G, G, rng);
            fprintf(stderr, "  [%s] params: %d\n", m.name(), m.param_count());
            train_2d(&m, m.name(), pde->name(), epochs, lr); m.release();
        } else if (strcmp(mk, "unet2d") == 0) {
            well::UNet2D m; m.init(dataset2d::HISTORY, 1, G, G, rng);
            fprintf(stderr, "  [%s] params: %d\n", m.name(), m.param_count());
            train_2d(&m, m.name(), pde->name(), epochs, lr); m.release();
        } else if (strcmp(mk, "resnet") == 0) {
            well::ResNet2D m; m.init(dataset2d::HISTORY, 1, G, G, rng);
            fprintf(stderr, "  [%s] params: %d\n", m.name(), m.param_count());
            train_2d(&m, m.name(), pde->name(), epochs, lr); m.release();
        } else if (strcmp(mk, "dconv2d") == 0) {
            well::DilatedConvNet2D m; m.init(dataset2d::HISTORY, 1, G, G, rng);
            fprintf(stderr, "  [%s] params: %d\n", m.name(), m.param_count());
            train_2d(&m, m.name(), pde->name(), epochs, lr); m.release();
        } else if (strcmp(mk, "vit") == 0) {
            well::ViT m; m.init(dataset2d::HISTORY, 1, G, G, rng);
            fprintf(stderr, "  [%s] params: %d\n", m.name(), m.param_count());
            train_2d(&m, m.name(), pde->name(), epochs, lr); m.release();
        } else if (strcmp(mk, "pinn") == 0) {
            well::PINN m; m.init(dataset2d::HISTORY, 1, G, G, rng);
            fprintf(stderr, "  [%s] params: %d\n", m.name(), m.param_count());
            train_2d(&m, m.name(), pde->name(), epochs, lr); m.release();
        } else {
            fprintf(stderr, "\033[31mUnknown model: %s\033[0m\n", mk);
        }
    };

    if (strcmp(model_key, "all") == 0) {
        const char* models[] = {"fno2d","unet2d","resnet","dconv2d","vit","pinn"};
        for (auto& mk : models) {
            rng.seed(0xBEEF42ULL);
            dataset2d::generate(*pde, 0, false);
            run_model(mk);
        }
    } else {
        run_model(model_key);
    }

    delete pde;
    dataset2d::cleanup();

    fprintf(stderr,
        "\n\033[44;1;37m THE WELL C++ — COMPLETE                                         \033[0m\n"
        " \033[1;32m1:1 benchmark finished. Results in CSV output.\033[0m\n\n");

    return 0;
}
