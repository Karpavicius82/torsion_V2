# The Well C++ — Pure C++/AVX2 Physics Benchmark

**Full C++ rewrite of [PolymathicAI/the_well](https://github.com/PolymathicAI/the_well).**

No Python. No PyTorch. No dependencies. Single static binary.

## Architecture

```
Living Silicon Engine (torsion PDE) → generates physics trajectories
       ↓
Surrogate Models (C++ / AVX2) → learn to predict next timestep
       ↓
Adam Optimizer → gradient descent with cosine LR scheduling
       ↓
CSV + ANSI Dashboard → results
```

## Models

| Model | Params | Description |
|-------|--------|-------------|
| **FNO-1D** | ~880 | Fourier Neural Operator — spectral convolution + bypass |
| **DilatedConvNet-1D** | ~17K | 6-block dilated conv with residual connections |
| **UNet-1D** | ~100K+ | 3-level encoder-decoder with skip connections |

## Physics Engine

Einstein-Cartan torsion PDE: `d_tt S = c²·d_xx S - m²·S + g·S³`

- 2048-node field × 8 parallel lanes
- AVX2 SIMD (32×int16 per instruction)
- Genetic Algorithm for parameter evolution
- Phase coherence + soliton detection

## Build

```bash
# Windows (MSYS2 UCRT64)
g++ -O3 -mavx2 -mfma -static -std=c++20 -Wall -Wextra -Werror \
    -Wno-unused-function -I src -o the_well_cpp.exe src/main.cpp

# Linux
g++ -O3 -mavx2 -mfma -static -std=c++20 -Wall -Wextra -Werror \
    -Wno-unused-function -I src -o the_well_cpp src/main.cpp -lpthread
```

## Usage

```bash
# Train FNO for 10 epochs
./the_well_cpp fno 10 0.001

# Train ConvNet
./the_well_cpp convnet 20 0.0005

# Train UNet
./the_well_cpp unet 10 0.001

# Benchmark all models
./the_well_cpp all 10 0.001

# CSV output to file
./the_well_cpp fno 50 0.001 > results.csv
```

## Source Files

```
src/
├── main.cpp              # Entry point, dataset generation, training loop
├── tensor.hpp            # AVX2-accelerated tensor ops (axpy, dot, mse)
├── engine.hpp            # Living Silicon torsion field engine
├── optimizer.hpp         # Adam optimizer + cosine LR scheduler
└── models/
    ├── model.hpp         # Abstract model base class
    ├── fno.hpp           # Fourier Neural Operator
    ├── unet.hpp          # U-Net 1D
    └── conv_net.hpp      # Dilated ConvNet
```

## Replaces

| Python (the_well) | C++ (this) |
|---|---|
| PyTorch `torch.nn` | Hand-written AVX2 kernels |
| `torch.optim.Adam` | `optimizer.hpp` (AVX2-fused) |
| HDF5 / 15TB datasets | Live engine simulation |
| Hydra configs | CLI arguments |
| WandB logging | CSV + ANSI dashboard |
| `pip install` | Single `g++` command |

## License

Copyright 2026 — Karpavicius82. Pure C/C++.
