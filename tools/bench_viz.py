#!/usr/bin/env python3
"""ENG-203: Benchmark visualization tool for inference-engine.

Reads one or more Google Benchmark JSON files and produces:
  1. GFLOPS (or chosen metric) vs matrix size — one line per kernel variant.
  2. Speedup vs baseline — bar chart grouped by size (only when --baseline
     is given).
  3. Thread-scaling chart — GFLOPS vs thread count for parallel kernels
     (only when multiple thread counts are present in the data).

Chart 1 and Chart 2 are written to --output.
Chart 3 (thread scaling) is written to <stem>_thread_scaling<ext>.

The benchmark name is expected to follow the Google Benchmark convention:

    BM_<Variant>[_<Tag>]/<arg0>[/<arg1>]...

Examples:
    BM_GemmNaive_Square/256
    BM_GemmNaive_Square_Large/1024
    BM_GemmTiled_Square/512/threads:8

"Size" is extracted from the first numeric argument (/<number>).
"Threads" are extracted from a trailing "threads:<N>" argument when present.

Usage
-----
    python3 tools/bench_viz.py \\
        --bench build/bench/results_*.json \\
        [--baseline build/bench/baseline.json] \\
        --output docs/assets/cpu_vs_gpu.png \\
        [--metric GFLOPS] \\
        [--title "FP32 GEMM Performance"]
"""

import argparse
import glob
import json
import os
import sys


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
def _require_deps():
    missing = []
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    if missing:
        print(
            "ERROR: missing Python packages: "
            + ", ".join(missing)
            + "\nInstall with:  pip install "
            + " ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)


_require_deps()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading and parsing
# ---------------------------------------------------------------------------

def _expand_paths(patterns):
    paths = []
    for pat in patterns:
        expanded = sorted(glob.glob(pat))
        if expanded:
            paths.extend(expanded)
        elif os.path.isfile(pat):
            paths.append(pat)
    return paths


def _parse_name(name):
    """Parse a Google Benchmark name into (prefix, size, threads).

    Returns
    -------
    prefix  : str  — everything before the first '/' argument segment
    size    : int or None — first numeric argument value
    threads : int or None — value of 'threads:<N>' argument, or None
    """
    # Split into the function-name part and the argument parts.
    parts = name.split("/")
    prefix = parts[0]
    size = None
    threads = None

    for part in parts[1:]:
        if part.startswith("threads:"):
            try:
                threads = int(part.split(":")[1])
            except (IndexError, ValueError):
                pass
        elif size is None:
            try:
                size = int(part)
            except ValueError:
                pass

    return prefix, size, threads


def load_bench_data(patterns, metric):
    """Load benchmark data from JSON files.

    Returns
    -------
    dict: {prefix: {size: {threads: [values]}}}
        where values are the chosen metric values.
    """
    data = {}
    paths = _expand_paths(patterns)
    if not paths:
        print("WARNING: no benchmark JSON files matched the given patterns.", file=sys.stderr)
        return data

    for path in paths:
        try:
            with open(path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: skipping {path}: {exc}", file=sys.stderr)
            continue

        for bench in raw.get("benchmarks", []):
            if bench.get("run_type") == "aggregate":
                continue

            name = bench.get("name", "")
            prefix, size, threads = _parse_name(name)

            # Extract the requested metric value.
            value = _get_metric(bench, metric)
            if value is None:
                continue

            threads_key = threads if threads is not None else 1

            data.setdefault(prefix, {})
            data[prefix].setdefault(size, {})
            data[prefix][size].setdefault(threads_key, [])
            data[prefix][size][threads_key].append(value)

    # Reduce repetitions to their median.
    import statistics
    reduced = {}
    for prefix, sizes in data.items():
        reduced[prefix] = {}
        for size, threads_map in sizes.items():
            reduced[prefix][size] = {}
            for t, vals in threads_map.items():
                reduced[prefix][size][t] = statistics.median(vals)

    return reduced


def _get_metric(bench, metric):
    """Extract a numeric metric from a benchmark dict."""
    # Direct counter fields (GFLOPS, GB/s, etc.)
    if metric in bench:
        val = bench[metric]
        if val is not None:
            return float(val)
        return None

    # Standard time fields.
    if metric in ("real_time", "cpu_time"):
        val = bench.get(metric)
        if val is None:
            return None
        unit = bench.get("time_unit", "ns")
        val = float(val)
        if unit == "us":
            val *= 1e3
        elif unit == "ms":
            val *= 1e6
        elif unit == "s":
            val *= 1e9
        return val

    return None


def _single_thread_series(prefix_data):
    """Return {size: value} using the single-thread (threads=1) readings."""
    result = {}
    for size, threads_map in prefix_data.items():
        if size is None:
            continue
        if 1 in threads_map:
            result[size] = threads_map[1]
        else:
            # Fall back to the lowest thread count.
            t = min(threads_map)
            result[size] = threads_map[t]
    return result


# ---------------------------------------------------------------------------
# Chart 1: metric vs size
# ---------------------------------------------------------------------------

def plot_metric_vs_size(data, metric, title, ax):
    """Line chart: metric vs matrix size, one line per kernel variant."""
    cmap = plt.cm.tab10
    has_data = False

    for idx, (prefix, prefix_data) in enumerate(sorted(data.items())):
        series = _single_thread_series(prefix_data)
        if not series:
            continue
        sizes = sorted(s for s in series if s is not None)
        values = [series[s] for s in sizes]

        short = prefix.replace("BM_", "")
        ax.plot(sizes, values, marker="o", color=cmap(idx % 10), label=short, linewidth=1.8)
        has_data = True

    ax.set_xlabel("Matrix Size (N, for N×N)", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    if has_data:
        ax.legend(fontsize=8)
    return has_data


# ---------------------------------------------------------------------------
# Chart 2: speedup vs baseline
# ---------------------------------------------------------------------------

def plot_speedup(data, baseline_data, metric, title, ax):
    """Bar chart: speedup of each variant vs baseline, grouped by size.

    For time-based metrics, speedup = baseline / candidate (lower is better
    means faster). For throughput metrics (GFLOPS, GB/s), speedup =
    candidate / baseline (higher is better).
    """
    time_metrics = {"real_time", "cpu_time"}
    throughput = metric not in time_metrics

    # Collect all sizes across all variants and baseline.
    all_sizes = set()
    for prefix_data in data.values():
        all_sizes.update(s for s in prefix_data if s is not None)
    for prefix_data in baseline_data.values():
        all_sizes.update(s for s in prefix_data if s is not None)
    sizes = sorted(all_sizes)

    if not sizes:
        ax.set_visible(False)
        return False

    variants = sorted(data.keys())
    n_variants = len(variants)
    n_sizes = len(sizes)
    width = 0.8 / max(n_variants, 1)
    x = np.arange(n_sizes)
    cmap = plt.cm.tab10

    has_data = False
    for v_idx, prefix in enumerate(variants):
        candidate_series = _single_thread_series(data[prefix])
        # Find best-matching baseline prefix (exact match or fallback to first).
        if prefix in baseline_data:
            base_series = _single_thread_series(baseline_data[prefix])
        elif baseline_data:
            base_series = _single_thread_series(next(iter(baseline_data.values())))
        else:
            continue

        speedups = []
        for size in sizes:
            cand = candidate_series.get(size)
            base = base_series.get(size)
            if cand is not None and base is not None and base != 0.0:
                if throughput:
                    speedups.append(cand / base)
                else:
                    speedups.append(base / cand)
            else:
                speedups.append(0.0)

        offset = (v_idx - n_variants / 2.0 + 0.5) * width
        short = prefix.replace("BM_", "")
        ax.bar(
            x + offset,
            speedups,
            width=width * 0.9,
            label=short,
            color=cmap(v_idx % 10),
            alpha=0.8,
        )
        has_data = True

    ax.axhline(y=1.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Matrix Size (N)", fontsize=11)
    speedup_label = "Speedup (candidate / baseline)"
    ax.set_ylabel(speedup_label, fontsize=11)
    ax.set_title(f"{title} — Speedup vs baseline", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if has_data:
        ax.legend(fontsize=8)
    return has_data


# ---------------------------------------------------------------------------
# Chart 3: thread scaling
# ---------------------------------------------------------------------------

def _has_thread_scaling(data):
    """Return True if any prefix has multiple thread counts."""
    for prefix_data in data.values():
        for size, threads_map in prefix_data.items():
            if len(threads_map) > 1:
                return True
    return False


def plot_thread_scaling(data, metric, title, output_path):
    """Line chart: GFLOPS vs thread count for a representative size."""
    # Choose the largest common size.
    all_sizes = set()
    for prefix_data in data.values():
        all_sizes.update(s for s in prefix_data if s is not None)
    if not all_sizes:
        return

    rep_size = max(all_sizes)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.tab10
    has_data = False

    for idx, (prefix, prefix_data) in enumerate(sorted(data.items())):
        if rep_size not in prefix_data:
            continue
        threads_map = prefix_data[rep_size]
        if len(threads_map) <= 1:
            continue
        threads = sorted(threads_map.keys())
        values = [threads_map[t] for t in threads]
        short = prefix.replace("BM_", "")
        ax.plot(threads, values, marker="o", color=cmap(idx % 10), label=short, linewidth=1.8)
        has_data = True

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Thread Count", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{title} — Thread Scaling (size={rep_size})", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Thread-scaling plot saved to: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _thread_scaling_path(output_path):
    """Derive a sibling filename for the thread-scaling chart."""
    base, ext = os.path.splitext(output_path)
    return base + "_thread_scaling" + (ext if ext else ".png")


def run(args):
    data = load_bench_data(args.bench, args.metric)

    baseline_data = {}
    if args.baseline:
        baseline_data = load_bench_data([args.baseline], args.metric)
        if not baseline_data:
            print(
                f"WARNING: baseline file {args.baseline} produced no data for metric '{args.metric}'.",
                file=sys.stderr,
            )

    # Determine layout: 1 or 2 sub-charts in the main figure.
    has_baseline = bool(baseline_data)
    n_rows = 2 if has_baseline else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    title = args.title

    plot_metric_vs_size(data, args.metric, title, axes[0])

    if has_baseline:
        plot_speedup(data, baseline_data, args.metric, title, axes[1])

    fig.tight_layout(pad=3.0)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Performance chart saved to: {args.output}")
    plt.close(fig)

    # Thread-scaling chart — only if data exists.
    if _has_thread_scaling(data):
        ts_path = _thread_scaling_path(args.output)
        plot_thread_scaling(data, args.metric, title, ts_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ENG-203: benchmark visualization tool for inference-engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bench",
        nargs="+",
        required=True,
        metavar="PATTERN",
        help="One or more benchmark JSON files (globs accepted).",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        metavar="PATH",
        help="Baseline benchmark JSON for speedup comparison (optional).",
    )
    parser.add_argument(
        "--output",
        default="docs/assets/cpu_vs_gpu.png",
        metavar="PATH",
        help="Output PNG path (default: docs/assets/cpu_vs_gpu.png).",
    )
    parser.add_argument(
        "--metric",
        default="GFLOPS",
        metavar="METRIC",
        help="Metric to plot: GFLOPS, GB/s, real_time, cpu_time (default: GFLOPS).",
    )
    parser.add_argument(
        "--title",
        default="FP32 GEMM Performance",
        help="Chart title.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
