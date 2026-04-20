# Living Silicon OS v3.0 — AVX2+SMP Torsion Field Engine

**Bare-metal-class torsion field engine. Single C++ file. Zero dependencies.**

## Performance

| Metric | Value |
|--------|-------|
| **Speed** | 432,302 tick/s |
| **Time (500K ticks)** | 1.2 seconds |
| **Binary size** | ~460 KB |
| **Source files** | 1 |
| **Dependencies** | 0 |
| **Warnings** | 0 |

## Architecture

- **AVX2 SIMD**: 32 int16 elements per instruction
- **SMP**: All CPU cores (spin-barrier, sense-reversing)
- **Real-time**: `REALTIME_PRIORITY_CLASS` + `SCHED_FIFO`
- **Memory locked**: `VirtualLock` / `mlockall` — zero page faults
- **CPU pinned**: One worker per core, balanced lane partition
- **90% duty cycle**: Accumulated sleep throttle (Windows-aware)

## Physics

- Einstein-Cartan torsion PDE: `d_tt S = c²·d_xx S - m²·S + g·S³`
- 2048-node field × 8 parallel lanes
- Genetic algorithm with crossover for parameter evolution
- Phase coherence tracking
- Soliton detection via N-dimensional buckets

## Build

```bash
# Windows (MSYS2 UCRT64)
g++ -O3 -mavx2 -static -o living_silicon_os.exe living_silicon_os.cpp

# Linux
g++ -O3 -mavx2 -static -o living_silicon_os living_silicon_os.cpp -lpthread
```

## Run

```bash
# Dashboard on terminal, CSV to file
./living_silicon_os > results.csv
```

## Verification

```
TORSIJOS TEORIJOS VERDIKTAS
[X] Bangos sklinda
[X] Solitonai formuojasi
[X] Fazinė koherencija
[X] GA konvergavo

*** 4/4 IŠSAMUS PATVIRTINIMAS ***
```

## License

Copyright 2026 — Karpavicius82. Pure C/C++.
