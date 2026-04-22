# inference-engine

A C++17 inference engine optimized for edge deployment (ARM64 Android), built
TDD-first with benchmarking as a first-class concern.

![CI](https://github.com/varun997vn/hpc-learning/actions/workflows/ci.yml/badge.svg)

## Quick Start

### Prerequisites

Ubuntu 22.04+ (WSL2 recommended), CMake 3.20+, Ninja, GCC 12+ or Clang 15+.

```bash
sudo apt install build-essential cmake ninja-build ccache \
    libopenblas-dev libomp-dev clang-15 clang-format-15 clang-tidy-15 \
    python3-pip
pip install matplotlib numpy onnx onnxruntime flatbuffers
```

### Build

```bash
git clone https://github.com/varun997vn/hpc-learning.git inference-engine
cd inference-engine

# Debug build (tests included)
cmake --preset debug
cmake --build build/debug

# Run tests
ctest --preset debug

# Release build
cmake --preset release
cmake --build build/release
```

### Run the CLI

```bash
./build/debug/cli/engine_cli
```

### AddressSanitizer build

```bash
cmake --preset asan && cmake --build build/asan
ctest --test-dir build/asan --output-on-failure
```

### Android cross-compile

```bash
export ANDROID_NDK_ROOT=~/android/ndk/26.1.10909125
cmake --preset android-arm64
cmake --build build/android-arm64
```

## Running Benchmarks

> **Important**: Pin the CPU governor before benchmarking to reduce noise.
> ```bash
> sudo cpupower frequency-set -g performance
> ```

```bash
cmake --preset release && cmake --build build/release
./scripts/run_bench.sh
```

Results are written to `build/bench/results_<timestamp>.json`.

## Project Structure

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [CLAUDE.md](CLAUDE.md).

## CI / Branch Protection

All PRs must pass the 4-job CI matrix (`{debug,release} × {gcc-12,clang-15}`)
plus the Android cross-compile job before merging to `main`.
A performance regression gate blocks PRs that slow any benchmark by >5%
(override with `perf-regression-expected: <reason>` in the PR body).
