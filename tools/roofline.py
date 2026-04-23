#!/usr/bin/env python3
"""ENG-202: Roofline model plot for inference-engine benchmarks.

Reads one or more Google Benchmark JSON files and an optional hardware profile
JSON, then produces a roofline chart saved to --output (default:
docs/assets/roofline.png).

Roofline model recap
--------------------
  memory-bound ceiling : attainable_GFLOPS = peak_bw_GBs * AI
  compute ceiling       : attainable_GFLOPS = peak_gflops
  ridge point           : AI_ridge = peak_gflops / peak_bw_GBs  (FLOP/Byte)

where AI (arithmetic intensity) = GFLOPS / (GB/s).

For benchmarks that report GB/s == 0 the kernel is assumed compute-bound and
plotted at the right edge of the chart (AI >= AI_ridge) at its measured GFLOPS.

Usage
-----
    python3 tools/roofline.py \\
        --bench build/bench/results_*.json \\
        --peak-gflops 25.0 \\
        --peak-bw-gbs 40.0 \\
        --output docs/assets/roofline.png \\
        [--hardware build/bench/hardware.json] \\
        [--title "GEMM Roofline - WSL2"] \\
        [--show]

--peak-gflops and --peak-bw-gbs are required unless --hardware is supplied and
contains 'peak_gflops_fp32' and 'peak_dram_bandwidth_gb_s' keys.
"""

import argparse
import glob
import json
import os
import sys


# ---------------------------------------------------------------------------
# Dependency check — actionable error before anything else.
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
matplotlib.use("Agg")  # non-interactive backend; overridden by --show
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def _expand_paths(patterns):
    """Expand glob patterns into a sorted list of existing file paths."""
    paths = []
    for pat in patterns:
        expanded = sorted(glob.glob(pat))
        if expanded:
            paths.extend(expanded)
        elif os.path.isfile(pat):
            paths.append(pat)
    return paths


def load_bench_files(patterns):
    """Return list of (label, gflops, gbs) tuples from benchmark JSON files.

    - label  : benchmark name (string)
    - gflops : attained GFLOPS (float)
    - gbs    : GB/s counter value; 0.0 means compute-bound / not reported
    """
    points = []
    paths = _expand_paths(patterns)
    if not paths:
        print("WARNING: no benchmark JSON files matched the given patterns.", file=sys.stderr)
        return points

    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: skipping {path}: {exc}", file=sys.stderr)
            continue

        for bench in data.get("benchmarks", []):
            # Skip aggregate rows (mean/median/stddev entries).
            if bench.get("run_type") == "aggregate":
                continue
            name = bench.get("name", "unknown")
            gflops = float(bench.get("GFLOPS", 0.0))
            gbs = float(bench.get("GB/s", 0.0))
            if gflops <= 0.0:
                continue
            points.append((name, gflops, gbs))

    return points


def load_hardware_json(path):
    """Return (peak_gflops, peak_bw_gbs) from hardware.json, or (None, None)."""
    if path is None:
        return None, None
    if not os.path.isfile(path):
        print(f"WARNING: hardware.json not found at {path}; skipping roofline ceilings.", file=sys.stderr)
        return None, None
    try:
        with open(path) as f:
            hw = json.load(f)
        return (
            hw.get("peak_gflops_fp32"),
            hw.get("peak_dram_bandwidth_gb_s"),
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read hardware.json: {exc}", file=sys.stderr)
        return None, None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_roofline(points, peak_gflops, peak_bw_gbs, output_path, title, show):
    """Render and save the roofline chart.

    Parameters
    ----------
    points      : list of (label, gflops, gbs)
    peak_gflops : float or None  — compute ceiling
    peak_bw_gbs : float or None  — memory bandwidth (GB/s)
    output_path : str
    title       : str
    show        : bool — open interactive window
    """
    has_roof = (peak_gflops is not None) and (peak_bw_gbs is not None)

    # Determine AI range for the chart.
    if has_roof:
        ridge_ai = peak_gflops / peak_bw_gbs  # FLOP/Byte
        ai_min = ridge_ai / 100.0
        ai_max = ridge_ai * 100.0
    else:
        ai_min = 0.01
        ai_max = 1000.0

    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Roofline ceilings ---------------------------------------------------
    if has_roof:
        ridge_ai = peak_gflops / peak_bw_gbs
        mem_roof = np.minimum(peak_bw_gbs * ai_range, peak_gflops)
        ax.plot(
            ai_range,
            mem_roof,
            color="steelblue",
            linewidth=2,
            label=f"Roofline ceiling (peak BW={peak_bw_gbs:.1f} GB/s, peak={peak_gflops:.1f} GFLOPS)",
        )
        ax.axvline(
            x=ridge_ai,
            color="steelblue",
            linewidth=1,
            linestyle=":",
            alpha=0.7,
        )
        ax.text(
            ridge_ai * 1.05,
            peak_gflops * 0.85,
            f"Ridge\n{ridge_ai:.2f} FLOP/B",
            fontsize=8,
            color="steelblue",
            va="top",
        )

    # --- Scatter: benchmark data points --------------------------------------
    cmap = plt.cm.tab10
    label_seen = {}
    plotted_x = []
    plotted_y = []

    for idx, (name, gflops, gbs) in enumerate(points):
        color = cmap(idx % 10)

        if gbs > 0.0:
            ai = gflops / gbs  # FLOP/Byte
        else:
            # Compute-bound: place point at AI >= ridge (or ai_max/4 as fallback).
            ai = ridge_ai * 10.0 if has_roof else ai_max / 4.0

        # Shorten label for readability: keep last two slash-separated parts.
        parts = name.split("/")
        short = "/".join(parts[-2:]) if len(parts) >= 2 else name

        if short not in label_seen:
            label_seen[short] = color
            ax.scatter(ai, gflops, color=color, s=80, zorder=5, label=short)
        else:
            ax.scatter(ai, gflops, color=label_seen[short], s=80, zorder=5)

        ax.annotate(
            short,
            xy=(ai, gflops),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
        plotted_x.append(ai)
        plotted_y.append(gflops)

    # --- Axis formatting -----------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    ax.set_ylabel("Attained Performance (GFLOPS)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    # Axis limits: encompass all points with some margin.
    all_x = plotted_x + ([ai_min, ai_max] if has_roof else [ai_min, ai_max])
    all_y = plotted_y + ([0.1, peak_gflops * 1.5] if has_roof else [0.1, 1.0])
    ax.set_xlim(min(all_x) * 0.5, max(all_x) * 2.0)
    ax.set_ylim(min(all_y) * 0.5, max(all_y) * 2.0)

    if label_seen or has_roof:
        ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()

    # --- Save ----------------------------------------------------------------
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Roofline plot saved to: {output_path}")

    if show:
        matplotlib.use("TkAgg")  # switch to interactive backend
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ENG-202: roofline model plot for inference-engine benchmarks.",
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
        "--peak-gflops",
        type=float,
        default=None,
        metavar="N",
        help="Single-core SP peak GFLOPS (overrides --hardware).",
    )
    parser.add_argument(
        "--peak-bw-gbs",
        type=float,
        default=None,
        metavar="N",
        help="Peak DRAM bandwidth in GB/s (overrides --hardware).",
    )
    parser.add_argument(
        "--hardware",
        default=None,
        metavar="PATH",
        help="Path to hardware.json (build/bench/hardware.json). Used when "
             "--peak-gflops / --peak-bw-gbs are not supplied.",
    )
    parser.add_argument(
        "--output",
        default="docs/assets/roofline.png",
        metavar="PATH",
        help="Output PNG path (default: docs/assets/roofline.png).",
    )
    parser.add_argument(
        "--title",
        default="Roofline Model",
        help="Chart title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive plot window after saving.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Resolve peak numbers: CLI flags override hardware.json.
    peak_gflops = args.peak_gflops
    peak_bw_gbs = args.peak_bw_gbs

    if peak_gflops is None or peak_bw_gbs is None:
        hw_gflops, hw_bw = load_hardware_json(args.hardware)
        if peak_gflops is None:
            peak_gflops = hw_gflops
        if peak_bw_gbs is None:
            peak_bw_gbs = hw_bw

    if peak_gflops is None or peak_bw_gbs is None:
        print(
            "WARNING: peak GFLOPS or peak bandwidth not available. "
            "Roofline ceiling lines will be omitted. "
            "Supply --peak-gflops and --peak-bw-gbs or --hardware.",
            file=sys.stderr,
        )

    points = load_bench_files(args.bench)
    if not points:
        print("WARNING: no benchmark data points found; the plot will be empty.", file=sys.stderr)

    plot_roofline(
        points=points,
        peak_gflops=peak_gflops,
        peak_bw_gbs=peak_bw_gbs,
        output_path=args.output,
        title=args.title,
        show=args.show,
    )


if __name__ == "__main__":
    main()
