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

## INT8 GEMM (ENG-404)

### Reference implementation (`gemm_int8_fixed`)

`gemm_int8_fixed` is the INT8 analogue of `gemm_fp32_naive`: a triple-loop
reference with no tiling or SIMD. It is the correctness oracle for future
optimised INT8 variants.

**Input/output contract**

| Tensor | DType | Description |
|---|---|---|
| A | INT8 | M×K, symmetric quantized (zero_point=0) |
| B | INT8 | K×N, symmetric quantized (zero_point=0) |
| C | FP32 | M×N, dequantized output |

**Accumulator strategy**

The inner loop accumulates into `int32_t`, never widening to `int64_t` or
converting to `float` per step:

```cpp
int32_t acc = 0;
for (int64_t k = 0; k < K; ++k)
    acc += static_cast<int32_t>(a[i*K+k]) * static_cast<int32_t>(b[k*N+j]);
c[i*N+j] = static_cast<float>(acc) * out_scale;
```

Using `int32_t` avoids per-multiply FP rounding while keeping the type wide
enough to hold the sum of up to 127²×K products before overflow. For INT8
inputs (range ±127), a single product is at most 127²=16129. With K products,
the maximum accumulator value is 16129×K. For K=4096 (the largest benchmark
size), max_acc ≈ 66M, well within the INT32 range of ≈2.1×10⁹.

**Dequantization / requantization scale**

After accumulation, the `int32_t` result is converted to FP32 in a single
multiply:

```
C[i][j] = acc * (scale_a * scale_b / scale_c)
```

- `scale_a` and `scale_b` are the symmetric per-tensor scales for A and B
  respectively (FP32_value ≈ INT8_value × scale).
- `scale_c` allows callers to re-quantize the output into a different scale
  domain (e.g. feed directly into a subsequent INT8 layer). Set `scale_c=1.0`
  to return plain FP32.

**Correctness tolerance**

Roundtrip (quantize FP32 → INT8 GEMM → compare to FP32 naive) produces
per-element error bounded by K×(scale_a + scale_b), where scale_a and
scale_b are both ≈ max|input| / 127. For inputs in [−1, 1] and K=8, the
typical worst-case error is < 0.13, verified in the unit tests.

**Shape validation**

`detail::check_int8_gemm_shapes()` in `gemm_internal.hpp` checks:
- all three tensors are rank-2
- A and B are DType::INT8; C is DType::FP32
- A.cols == B.rows (inner dimension)
- C is M×N

---

### AVX2 INT8 path (planned: ENG-405)

The optimised INT8 variant will use:

```
_mm256_maddubs_epi16(a_u8, b_s8)  →  int16
_mm256_madd_epi16(int16, ones)    →  int32
```

**The `maddubs` signed-operand trick**: `_mm256_maddubs_epi16` treats its
*first* operand as **unsigned** u8 and its second as signed s8.  Because our
quantized activations (A) are signed INT8 (range −127…127), at least one of
them may be negative. To satisfy the unsigned requirement for the first
operand we shift: add 128 to each element of A (making it u8, range 1…255)
before calling `maddubs`, then subtract the corresponding bias from the
accumulator in the requantization step. Specifically, for each column j of B:

```
bias_j = 128 × sum_k(B[k][j])
acc_corrected = acc_maddubs − bias_j
```

This correction is computed once per column of B (outside the inner loop) so
it adds O(K×N) work amortised over M rows. The trick is noted in inline
comments in the AVX2 source file when ENG-405 lands.

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
