#!/usr/bin/env bash
# run_bench.sh — build and run all benchmark binaries, save JSON results.
#
# Usage:  ./scripts/run_bench.sh [<cmake-preset>]
# Default preset: release
#
# CPU governor pinning:
#   sudo cpupower frequency-set -g performance
#   Requires the cpupower tool (linux-tools package).
#   Falls back gracefully if unavailable (WSL2 may not expose cpufreq).
#
# OPENBLAS_NUM_THREADS=1:
#   Forces single-core baselines so OpenBLAS and our kernels are measured on
#   an equal footing.  Change to the desired thread count for multi-core runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:-release}"
RESULTS_DIR="${REPO_ROOT}/build/bench"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESULTS_DIR}"

# ---------- CPU governor -----------------------------------------------------
if command -v cpupower &>/dev/null; then
    echo "[bench] Pinning CPU to performance governor..."
    sudo cpupower frequency-set -g performance 2>/dev/null || \
        echo "[bench] WARNING: cpupower failed (WSL2 may not expose cpufreq); continuing."
else
    echo "[bench] cpupower not found; skipping governor pin (acceptable on WSL2)."
fi

# ---------- Build ------------------------------------------------------------
echo "[bench] Building preset '${PRESET}'..."
cmake --build --preset "${PRESET}"

# ---------- Single-core baseline env ----------------------------------------
export OPENBLAS_NUM_THREADS=1

# ---------- Run benchmarks ---------------------------------------------------
BENCH_BIN="${REPO_ROOT}/build/${PRESET}/benchmarks"

declare -A SUMMARY
RAN=0

for bench in \
    "${BENCH_BIN}/microbench/bench_tensor" \
    "${BENCH_BIN}/hardware_probe/bench_peak"; do

    [ -x "${bench}" ] || continue

    name="$(basename "${bench}")"
    out_file="${RESULTS_DIR}/${name}_${TIMESTAMP}.json"

    echo "[bench] Running ${name}..."
    "${bench}" \
        --benchmark_format=json \
        --benchmark_out="${out_file}" \
        --benchmark_min_time=1s \
        --benchmark_repetitions=5

    SUMMARY["${name}"]="${out_file}"
    RAN=$(( RAN + 1 ))
done

if [ "${RAN}" -eq 0 ]; then
    echo "[bench] ERROR: no benchmark binaries found under ${BENCH_BIN}" >&2
    echo "[bench] Did you build with preset '${PRESET}'?" >&2
    exit 1
fi

# ---------- Human-readable summary ------------------------------------------
echo ""
echo "========================================================"
echo " Benchmark results — ${TIMESTAMP}"
echo "========================================================"
for name in "${!SUMMARY[@]}"; do
    out="${SUMMARY[$name]}"
    echo ""
    echo "  [${name}]  ->  ${out}"
    if command -v python3 &>/dev/null; then
        python3 - "${out}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
benchmarks = [b for b in data.get("benchmarks", []) if b.get("run_type") != "aggregate"]
if not benchmarks:
    print("    (no benchmarks recorded)")
    sys.exit(0)
print(f"    {'Benchmark':<50} {'time (us)':>12}  {'bytes/s':>14}")
print(f"    {'-'*50} {'-'*12}  {'-'*14}")
for b in benchmarks:
    t_us = b.get("real_time", 0)
    unit = b.get("time_unit", "us")
    if unit == "ms":
        t_us = t_us * 1000
    elif unit == "ns":
        t_us = t_us / 1000
    bw = b.get("bytes_per_second", b.get("BytesProcessed", 0))
    bw_gb = bw / 1e9 if bw else 0
    bw_str = f"{bw_gb:.2f} GB/s" if bw_gb else "n/a"
    print(f"    {b['name']:<50} {t_us:>12.2f}  {bw_str:>14}")
PYEOF
    fi
done
echo ""
echo "[bench] JSON files saved to ${RESULTS_DIR}/"
