#!/usr/bin/env bash
# ENG-305: Validate >=40% cache-miss reduction for tiled GEMM vs naive.
# Usage: ./scripts/profile.sh [preset] [size] [threshold]
# Requires: perf (Linux), built bench_gemm binary.
set -euo pipefail

PRESET="${1:-release}"
SIZE="${2:-512}"
THRESHOLD="${3:-0.40}"

BENCH="build/${PRESET}/benchmarks/microbench/bench_gemm"

if [ ! -x "$BENCH" ]; then
    echo "[profile] Building $PRESET preset..."
    cmake --build --preset "$PRESET" --target bench_gemm 2>/dev/null
fi

if ! command -v perf &>/dev/null; then
    echo "WARNING: 'perf' not available (WSL2/container). Skipping cache-miss check."
    exit 0
fi

run_perf() {
    local filter="$1"
    perf stat -e cache-misses,cache-references \
        "$BENCH" --benchmark_filter="${filter}/${SIZE}" \
                 --benchmark_min_time=2s \
                 --benchmark_repetitions=1 2>&1
}

echo "[profile] Profiling BM_GemmNaive_Square/${SIZE}..."
NAIVE_OUT="$(run_perf BM_GemmNaive_Square)"
echo "[profile] Profiling BM_GemmTiled_Square/${SIZE}..."
TILED_OUT="$(run_perf BM_GemmTiled_Square)"

python3 - <<PYEOF
import sys, re

def extract_misses(text):
    m = re.search(r'([\d,]+)\s+cache-misses', text)
    return int(m.group(1).replace(',', '')) if m else None

naive_out = """${NAIVE_OUT}"""
tiled_out = """${TILED_OUT}"""
threshold = float("${THRESHOLD}")

naive = extract_misses(naive_out)
tiled = extract_misses(tiled_out)

if naive is None or tiled is None:
    print("WARNING: could not extract cache-miss counts from perf output.")
    print("  naive perf output:", naive_out[-500:])
    sys.exit(0)

reduction = (naive - tiled) / max(naive, 1)
print(f"Naive cache misses : {naive:,}")
print(f"Tiled cache misses : {tiled:,}")
print(f"Reduction          : {reduction*100:.1f}%  (target >= {threshold*100:.0f}%)")

if reduction < threshold:
    print(f"FAIL: reduction {reduction*100:.1f}% < {threshold*100:.0f}%")
    sys.exit(1)
print("PASS")
PYEOF
