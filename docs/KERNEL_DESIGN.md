# Kernel Design

## GEMM FP32

### Variant hierarchy

Every GEMM variant computes `C = alpha * A * B + beta * C` (row-major FP32).
Each builds on the previous; all must agree within epsilon 1e-5.

| Variant | Ticket | Description |
|---|---|---|
| `gemm_fp32_naive` | ENG-103 | Reference triple loop — correctness oracle only |
| `gemm_fp32_tiled` | ENG-302 | Cache-blocked 6-loop structure with `TilingConfig` |
| `gemm_fp32_parallel` | ENG-303 | Tiled + OpenMP static scheduling on the M-tile loop |
| `gemm_fp32_simd` | ENG-304 | Parallel + AVX2 8×8 register-blocked micro-kernel |

### Naive kernel (ENG-103)

The naive triple loop:

```
for i in [0, M):
  for j in [0, N):
    acc = 0
    for k in [0, K):
      acc += A[i,k] * B[k,j]
    C[i,j] = alpha * acc + beta * C[i,j]
```

Row pointer `a_row = A + i*K` is hoisted outside the j-loop to avoid
recomputing the base offset on every iteration.

Shape contract enforced at entry via `check_gemm_shapes`; all subsequent
variants reuse the same check.  The inner loops contain no allocations or
branches beyond the loop bounds.

### TilingConfig defaults (ENG-302)

`mc = nc = kc = 64`.  For FP32 (4 bytes), three 64×64 tiles occupy
3 × 64 × 64 × 4 = 48 KiB, fitting comfortably in a 32–64 KiB L1 data cache
with room for the instruction stream.

### AVX2 micro-kernel choice (ENG-304)

An 8×8 output tile is selected because:
- 8 `__m256` accumulator registers fit in 8 YMM registers.
- 2 more YMM registers hold the broadcast of A and the B panel row.
- Total: 10 YMM registers, well within the x86-64 limit of 16.
- Each `_mm256_fmadd_ps` processes 8 FP32 elements, matching the 8-wide
  AVX2 vector width exactly.

AVX-512 is explicitly avoided: 512-bit instructions trigger frequency
downclocking on many Intel consumer CPUs (EVEX transition penalty).

### INT8 GEMM (ENG-404)

`_mm256_maddubs_epi16` requires one operand to be unsigned (u8) and the other
signed (s8).  Because both activations and weights may be signed INT8 after
symmetric quantization, we shift the signed operand by adding 128 to make it
unsigned before the multiply, then correct with the requantization scale
factor.  This is documented inline in `src/kernels/gemm/gemm_int8.cpp`.
