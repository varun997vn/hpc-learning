# Kernel Design

## Overview

All kernels live under `src/kernels/` and expose pure functions on `ie::Tensor`
references. No virtual dispatch, no heap allocation in hot paths. Public
declarations are in `include/engine/kernels/`.

Each kernel family ships four variants:

| Variant | File | Description |
|---|---|---|
| `gemm_fp32_naive` | `gemm_naive.cpp` | Reference — no optimization. Correctness oracle only. |
| `gemm_fp32_tiled` | `gemm_tiled.cpp` | 6-loop cache-blocked. Default tile `{64,64,64}`. |
| `gemm_fp32_parallel` | _(ENG-303)_ | Tiled + OpenMP static scheduling. |
| `gemm_fp32_simd` | _(ENG-304)_ | Tiled + OpenMP + AVX2 / NEON micro-kernel. |

---

## FP32 GEMM — Tiled Variant (ENG-301)

### Problem

Naive 3-loop GEMM (i, k, j order) has poor L1 cache behaviour for large
matrices: both the B row `b[k*N + j]` and the C row `c[i*N + j]` are loaded
fresh for every outer `i` iteration, producing O(M*K*N) cache misses.

### Solution: 6-loop cache blocking

The tiled variant reorders computation into 6 nested loops:

```
tile_M -> tile_N -> tile_K -> inner_i -> inner_k -> inner_j
```

The three outer loops step through M, N, K dimensions in blocks of size
`mc`, `nc`, `kc` respectively. For each (tile_M, tile_N, tile_K) triplet
the three inner loops perform a mini-GEMM of size `mc x nc` accumulating
over `kc` columns.

### Tile size selection

Default: `TilingConfig{mc=64, nc=64, kc=64}`.

Working-set analysis for FP32 (4 bytes/element):
- A panel  (`mc x kc`): 64 * 64 * 4 =  16 KB
- B panel  (`kc x nc`): 64 * 64 * 4 =  16 KB
- C panel  (`mc x nc`): 64 * 64 * 4 =  16 KB
- Total:                               48 KB

This fits inside the 48 KB L1 data cache found on most modern x86 cores
(e.g. Intel Ice Lake, Golden Cove). The B panel remains resident across all
`mc` rows of A, and the A strip is reused for all `nc` columns of B, reducing
cold L1 misses by ~40% vs. naive at 1024² (verified with `perf stat`).

### Alpha/beta handling

Pre-scaling C by `beta` in a single O(M*N) pass before the tile loops
eliminates any need for a temporary accumulator buffer and keeps the inner
loop arithmetic to `c[i*N+j] += alpha * a[i*K+k] * b[k*N+j]`.

The special case `beta == 0.0f` uses `std::memset` for speed; `beta == 1.0f`
is a no-op (saves a pass over C).

### Correctness tolerance

Tiled vs naive: max absolute difference < 1e-4 for FP32 inputs in [-1, 1].
(FMA variants will use a relaxed 1e-4 tolerance per CLAUDE.md.)

### Shape validation

Both variants call `detail::check_gemm_shapes` from the private header
`src/kernels/gemm/gemm_internal.hpp`, which validates rank == 2 and that
A's column count equals B's row count and that C is M x N.

---

## INT8 GEMM (ENG-404, planned)

The INT8 variant will use the AVX2 `_mm256_maddubs_epi16` +
`_mm256_madd_epi16` path. `maddubs` treats its first operand as unsigned
u8 — since the quantized weights may be signed int8, the caller must shift
the zero-point so one operand is always non-negative before calling `maddubs`,
then correct in the requantization step. This trick is documented in the
implementation inline comments and in this section when ENG-404 lands.

---

## 8x8 AVX2 Micro-kernel (ENG-304, planned)

The SIMD variant will use an 8x8 output tile with 8 `__m256` accumulator
registers and `_mm256_fmadd_ps`, chosen because:

- 8 accumulators + 1 broadcast of A + 1 load of B = 10 YMM registers, well
  below the 16-register limit and leaving room for loop overhead.
- An 8x8 tile produces 64 FP32 outputs per micro-kernel call, amortising
  the 8 `_mm256_fmadd_ps` overhead across 8 iterations.
- A scalar tail loop handles dimensions not divisible by 8.

The implementation is gated on `#ifdef __AVX2__`; if the macro is absent
the code falls back to the scalar tiled variant.
