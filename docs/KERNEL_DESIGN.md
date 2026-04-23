# Kernel Design Notes

## GEMM Variants

All GEMM functions compute `C = alpha * A * B + beta * C` for row-major FP32
tensors where A is [M, K], B is [K, N], and C is [M, N].

### gemm_fp32_naive

Triple-loop reference with no optimisations.  Used only in correctness tests
and as the performance baseline in benchmarks.  O(MKN) with no cache reuse.

### gemm_fp32_parallel (ENG-302)

Tiled + OpenMP static scheduling.

**Tile sizes (TilingConfig defaults: mc=nc=kc=64)**

Three tiles of 64x64 FP32 = 3 * 64 * 64 * 4 B = 48 KiB.  This fits in the
48 KiB L1 data cache found on Intel Golden Cove / Alder Lake P-cores and
leaves a margin for the instruction cache working set.

**Loop order**

```
tile_M (parallel) -> tile_K -> tile_N -> micro m -> micro n -> k
```

Parallelising the outermost M-tile loop with `schedule(static)` assigns
contiguous row-strips of C to each thread.  No two threads write the same
output element, so no locks or atomics are needed.

**beta pre-scaling**

`C` is multiplied by `beta` in a sequential pass before the parallel tiles
run.  This means the inner accumulation is always `C[i,j] += alpha * acc`
regardless of the K-tile boundary, avoiding a race on the beta contribution
when the K dimension spans more than one tile.

**No-OpenMP fallback**

When `_OPENMP` is not defined the `#pragma omp` directive is absent and the
function executes as a serial tiled loop, producing identical results.

**Numerical accuracy**

Max absolute difference vs `gemm_fp32_naive` is below 1e-4 for FP32 inputs
uniformly distributed in [-1, 1] at sizes up to 256x256.  The error budget
grows as O(sqrt(K)) due to floating-point accumulation order.

### gemm_fp32_tiled (ENG-302, serial)

Same tiled structure without OpenMP.  Not yet implemented.

### gemm_fp32_simd (ENG-304)

Tiled + OpenMP + AVX2 8x8 micro-kernel.  Not yet implemented.

The 8x8 register-blocked micro-kernel uses 8 `__m256` accumulator registers
(one per output row) plus at most 2 additional YMM registers for A broadcasts
and B loads, totalling 10 YMM registers — well under the 16-register YMM file
limit and leaving 6 registers for address arithmetic.  See ENG-304.
