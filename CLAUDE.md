# Inference Engine — Claude Instructions

## Project Overview

A C++17 inference engine optimized for edge deployment (ARM64 Android), built
TDD-first with benchmarking as a first-class concern. Dev environment is WSL2
(x86_64) with an NVIDIA A2000 as a GPU reference and correctness oracle.

**Project name**: `inference-engine`  
**Namespace**: `ie::`  
**License**: Apache 2.0

---

## Architecture (5 Layers, strict bottom-up dependency)

```
Layer 5: Runtime API        (session, inference)
Layer 4: Model I/O          (parser, graph builder)
Layer 3: Graph IR           (nodes, scheduler)
Layer 2: Kernels            (GEMM, Conv, activation)
Layer 1: Memory primitives  (Tensor, Allocator)

Cross-cutting: Benchmark, Logging, Errors
```

Layer N may only depend on layers < N. Never introduce upward dependencies.

---

## Directory Structure

```
inference-engine/
├── .github/workflows/       # ci.yml, bench.yml, regression.yml
├── CMakeLists.txt
├── CMakePresets.json        # debug, release, asan, android-arm64, cuda
├── cmake/
│   ├── toolchains/android-arm64.cmake
│   └── modules/FindOpenBLAS.cmake
├── include/engine/          # public headers only
├── src/
│   ├── tensor/
│   ├── kernels/
│   │   ├── gemm/
│   │   ├── conv/
│   │   └── activation/
│   ├── quantization/
│   ├── graph/
│   ├── runtime/
│   └── gpu/                 # CUDA kernels, ENABLE_CUDA=OFF by default
├── tests/
│   ├── unit/                # GoogleTest, <100ms each, run on every commit
│   ├── integration/         # <30s total, run on every PR
│   └── correctness/         # vs cuBLAS/numpy, run nightly
├── benchmarks/
│   ├── microbench/          # per-kernel, per-size
│   ├── e2e/                 # full MobileNetV2 forward pass
│   ├── hardware_probe/      # peak GFLOPS + DRAM bandwidth, run once per machine
│   └── baselines/           # OpenBLAS, Eigen, cuBLAS wrappers
├── tools/
│   ├── onnx_to_flatbuffer.py
│   ├── roofline.py
│   ├── regression_check.py
│   └── bench_viz.py
├── third_party/             # googletest, benchmark, STREAM (vendored)
├── schema/                  # model.fbs flatbuffer schema
├── models/                  # .fb files (gitignored, schema committed)
├── scripts/
│   ├── setup_wsl.sh
│   ├── run_bench.sh
│   ├── profile.sh
│   └── run_android_bench.sh
└── docs/
    ├── ARCHITECTURE.md
    ├── BENCHMARK_METHODOLOGY.md
    ├── KERNEL_DESIGN.md
    └── assets/              # roofline.png, cpu_vs_gpu.png
```

---

## Build System

**Presets** (CMakePresets.json):
- `debug` — C++17, `-Wall -Wextra -Wpedantic -Werror`, no optimization
- `release` — `-O3 -march=native`, ccache if available
- `asan` — debug + AddressSanitizer
- `android-arm64` — NDK r26+, arm64-v8a ABI
- `cuda` — enables `ENABLE_CUDA=ON`, targets `sm_86` (A2000/Ampere)

**Key CMake options**:
- `ENABLE_AVX2` — gates `-mavx2 -mfma` and AVX2 kernel code
- `ENABLE_NEON` — gates NEON intrinsics (set automatically for android-arm64)
- `ENABLE_CUDA` — OFF by default; enables src/gpu/ and CUDA language support
- `ENABLE_OPENMP` — ON by default

Dependencies fetched via `FetchContent`: googletest, Google Benchmark, Eigen.  
`OpenBLAS` found via `find_package` with a fallback `FindOpenBLAS.cmake`.

---

## TDD Workflow (mandatory for every ticket)

Every ticket follows this exact commit cycle:

1. **Red** — Write failing GoogleTest. Commit: `test(<scope>): <subject>`
2. **Green** — Minimum code to pass. Commit: `feat(<scope>): <subject>`
3. **Refactor** — Clean up, no behavior change. Commit: `refactor(<scope>): <subject>`
4. **Bench** — Add Google Benchmark case. Commit: `bench(<scope>): <subject>`
5. **Doc** — Update docs/comments. Commit: `docs(<scope>): <subject>`

Never skip the Red step. If you find yourself writing implementation before
tests, stop and write the test first.

---

## Commit Conventions

Format: `<type>(<scope>): <subject>`

Types: `feat`, `fix`, `test`, `bench`, `perf`, `refactor`, `docs`, `build`, `ci`, `chore`

Scopes: `tensor`, `gemm`, `conv`, `quant`, `graph`, `runtime`, `gpu`, `ci`,
`bench`, `build`, `android`, `tools`, `dsp`

Branch naming: `<TICKET-ID>/<slug>` — e.g. `ENG-102/tensor-class`

---

## Coding Standards

- **C++17 only**. No C++20 features (Android NDK support constraint).
- **No exceptions in kernels**. Use `Result<T>` (expected-like type) at API boundaries.
- **No virtual dispatch in hot paths**. Use CRTP for compile-time kernel dispatch.
- **64-byte alignment** assumed for all owned `Tensor` buffers (AVX-512 / cache-line).
- **No heap allocation in kernel inner loops**. Pre-allocate in the caller.
- **Comments**: only when the WHY is non-obvious. No docstrings narrating what the code does.
- Code formatted with clang-format (LLVM style, 4-space indent). Run before every commit.

### Every kernel must have four implementations:
| Variant | Description |
|---|---|
| `naive_*` | Reference, no optimization. Used only in correctness tests. |
| `tiled_*` | Cache-blocked with `TilingConfig`. |
| `parallel_*` | Tiled + OpenMP static scheduling. |
| `simd_*` | Tiled + OpenMP + AVX2 (x86) or NEON (ARM). |

All four variants must produce numerically equivalent output within a documented epsilon.

---

## Key Interfaces

```cpp
// Tensor (Layer 1)
class Tensor {
    static Tensor create(Shape shape, DType dtype);          // owned, 64-byte aligned
    static Tensor view(const Tensor&, Shape, int64_t offset); // non-owning
    static Tensor external(void*, Shape, DType dtype);       // zero-copy
    template<typename T> T* data();
    const Shape& shape() const;
    DType dtype() const;
    int64_t numel() const;
};

// Kernels (Layer 2) — pure functions on Tensor views
namespace kernels {
    void gemm_fp32_naive(const Tensor& A, const Tensor& B, Tensor& C);
    void gemm_fp32_tiled(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig);
    void gemm_fp32_parallel(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig, int n_threads);
    void gemm_int8_fixed(const Tensor& A, const Tensor& B, Tensor& C, QuantParams);
}

// Session (Layer 5)
class Session {
    static Result<Session> load(const std::string& model_path);
    Result<Tensor> run(const Tensor& input);
    BenchmarkReport last_run_profile() const;
};
```

---

## Benchmarking Rules

- Framework: **Google Benchmark** (never raw `std::chrono` loops).
- Minimum: `--benchmark_min_time=1 --benchmark_repetitions=5` for dev runs; 20+ repetitions for nightly.
- Always report `GFLOPS` via `state.counters["GFLOPS"]`.
- Benchmark sizes: `{64, 128, 256, 512, 1024, 2048, 4096}` square + `{384, 768}` non-power-of-2.
- Every CPU benchmark reports ratio against: naive (our own), OpenBLAS, Eigen, and cuBLAS when GPU available.
- Pin CPU governor before benchmarking: `sudo cpupower frequency-set -g performance`
- Set `OPENBLAS_NUM_THREADS=1` when measuring single-core baselines.

**Regression gate**: PRs regressing >5% on median wall time are blocked unless
the commit message contains `perf-regression-expected: <reason>`.

---

## Performance Targets

| Metric | Target | Platform |
|---|---|---|
| FP32 GEMM 1024² (1 core) | ≥ 25 GFLOPS | WSL2 |
| FP32 GEMM 1024² (8 cores) | ≥ 150 GFLOPS | WSL2 |
| FP32 GEMM vs OpenBLAS | ≥ 0.5× | WSL2 |
| INT8 GEMM vs FP32 GEMM | ≥ 2.5× faster | WSL2 |
| MobileNetV2 latency | < 50 ms | Snapdragon 865 |
| MobileNetV2 INT8 accuracy drop | < 1% top-1 | ImageNet val |
| Cache miss reduction (tiled vs naive) | ≥ 40% | perf stat |

---

## Testing Tiers

| Tier | Location | Trigger | Time budget |
|---|---|---|---|
| Unit | `tests/unit/` | Every commit | < 100ms each |
| Integration | `tests/integration/` | Every PR | < 30s total |
| Correctness | `tests/correctness/` | Nightly | strict tolerances |
| Benchmarks | `benchmarks/` | Nightly + PR gate | persisted JSON |

Coverage target: >85% line coverage on `src/kernels/` and `src/tensor.cpp`.

---

## CUDA / A2000 Notes

- A2000 is **Ampere SM_86** with 3rd-gen Tensor Cores.
- Role: correctness oracle (diff against cuBLAS FP32) and performance ceiling reference.
- CUDA is **not** a deployment dependency — `ENABLE_CUDA=OFF` by default.
- Never install NVIDIA drivers inside WSL2; use the Windows host driver + DxGkrnl passthrough.
- Install CUDA from the WSL-specific repo (`wsl-ubuntu`), not the generic Ubuntu repo.
- Target `CMAKE_CUDA_ARCHITECTURES 86`.

---

## Android / Edge Notes

- Deployment target: `arm64-v8a`, NDK r26+.
- adb connection: prefer TCP (`adb connect <ip>:5555`) over USB from WSL2.
- Keep repo files under `~/` not `/mnt/c/` — filesystem crossings pollute benchmarks.
- NEON SIMD is gated behind `ENABLE_NEON`, set automatically by the android-arm64 preset.
- Hexagon DSP (ENG-705) is a stretch goal; do not block any other ticket on it.

---

## Epic Map (for context)

| Epic | Focus | Key Tickets |
|---|---|---|
| 1 | Foundation, Memory, Tooling | ENG-101 to ENG-104 |
| 2 | Benchmarking Infrastructure | ENG-201 to ENG-207 |
| 3 | Cache Optimization + Concurrency | ENG-301 to ENG-305 |
| 4 | Quantization + Fixed-Point | ENG-401 to ENG-405 |
| 5 | GPU Reference Baseline | ENG-501 to ENG-505 |
| 6 | Model Integration | ENG-601 to ENG-606 |
| 7 | Edge Deployment | ENG-701 to ENG-705 |
| 8 | Continuous Perf Monitoring | ENG-801 to ENG-803 |

**Epic 2 must ship before Epic 3 begins.** You cannot optimize what you cannot measure.
