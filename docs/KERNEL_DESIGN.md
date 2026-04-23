# Kernel Design Notes

## ENG-303: `gemm_fp32_simd` — AVX2-Accelerated FP32 GEMM

### Overview

`gemm_fp32_simd` implements `C = alpha * A * B + beta * C` for FP32 matrices
using a two-path design:

- **AVX2 path** (`#ifdef __AVX2__`): 8×1 SIMD micro-kernel inside a
  tiled outer loop, with OpenMP static scheduling on the M-dimension.
- **Scalar fallback** (`#ifndef __AVX2__`): identical six-loop tiled structure
  without intrinsics, ensuring correctness on any platform.

### Tile Sizes (default `TilingConfig`)

| Parameter | Default | Rationale |
|---|---|---|
| `mc` | 64 | Fits a 64×64 FP32 A-panel in L1 (16 KB) |
| `nc` | 64 | Fits a 64×64 FP32 B/C-panel in L1 (16 KB) |
| `kc` | 64 | Together: 3 × 64×64 × 4 B = 48 KB — within 48 KB L1 on modern CPUs |

### AVX2 8×1 Micro-Kernel

The inner kernel processes one row `i` at a time, iterating over k in the
K-tile.  For each k step:

```
YMM registers used per k iteration:
  a_vec  = _mm256_set1_ps(alpha * A[i][k])   // broadcast scalar   (1 YMM)
  b_vec  = _mm256_loadu_ps(&B[k][j])          // load 8 B columns   (1 YMM)
  c_vec  = _mm256_loadu_ps(&C[i][j])          // load 8 C columns   (1 YMM)
  c_vec  = _mm256_fmadd_ps(a_vec, b_vec, c_vec)  // FMA
  _mm256_storeu_ps(&C[i][j], c_vec)           // store result
```

**Register budget**: 3 YMM per k step. The CPU has 16 YMM registers so this
leaves 13 free for the compiler's own use (loop counter, address calculations,
potential out-of-order window). AVX-512 is intentionally avoided to prevent
frequency downclocking on consumer CPUs that halve their clock when 512-bit
units are active.

**Why 8×1 instead of 8×8?** The 8×1 layout (one output row at a time) keeps
the accumulator count at 1 instead of 8, trading some IPC for simpler register
management and easier correctness. The tile blocking already provides the
cache reuse that an 8×8 kernel targets. ENG-304 will upgrade to an 8×8
register-blocked micro-kernel for an additional ~2× ILP gain.

### Tail Handling

When `N % 8 != 0`, the rightmost columns in each N-tile are handled by the
`scalar_tail()` inline helper:

```cpp
static inline void scalar_tail(float* c_row, const float* b_k,
                                float a_scaled, int64_t j_start, int64_t j_end);
```

This helper is shared by both the AVX2 and scalar fallback paths. Using
`_mm256_loadu_ps` (unaligned load) in the vector loop means no special
alignment preconditions are required for arbitrary pointer offsets, which is
correct for any owned or view `Tensor`.

### Alpha/Beta Semantics

1. **Beta pre-pass**: `C[i][j] *= beta` is applied once across all of C before
   any accumulation.  This ensures the tiled loop can unconditionally add to
   `C` without carrying `beta` into the inner kernel.
2. **Alpha folding**: `alpha` is multiplied into the A broadcast scalar
   (`alpha * A[i][k]`), so the inner FMA computes `c += (alpha*a) * b`
   in a single instruction.

### Correctness Tolerances

| Comparison | Tolerance |
|---|---|
| `gemm_fp32_simd` vs `gemm_fp32_naive` | < 1e-3 abs diff |
| Scalar fallback vs naive | < 1e-4 abs diff |

FMA reassociation can shift results by ~1e-6 per accumulation; for 1024×1024
matrices this sums to at most ~1e-3.

### Threading

`#pragma omp parallel for schedule(static)` on the outer M-tile loop.
Each thread owns a non-overlapping stripe of M-tile rows, so writes to `C`
never race. Thread count is set via `n_threads` argument using
`omp_set_num_threads()` before the parallel region.
