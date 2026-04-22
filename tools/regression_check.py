#!/usr/bin/env python3
"""Compare two Google Benchmark JSON files and gate on a regression threshold.

Exit 0  — no regressions above the threshold.
Exit 1  — one or more benchmarks regressed by more than --threshold (default 5%).

Usage:
    regression_check.py <current_json> --baseline <baseline_json> [--threshold 0.05]

Aggregates (run_type == "aggregate") are excluded; only individual iterations or
the per-repetition rows are used so the comparison is median-of-raw rather than
relying on the benchmark library's own aggregate.

PR comment format:
    The printed Markdown table can be pasted directly into a GitHub PR comment or
    posted via:
        gh pr comment <PR> --body "$(python3 tools/regression_check.py ...)"
"""

import json
import sys
import argparse


def load_benchmarks(path: str) -> dict[str, float]:
    """Return {benchmark_name: real_time_ns} for non-aggregate rows."""
    with open(path) as f:
        data = json.load(f)

    result: dict[str, float] = {}
    for b in data.get("benchmarks", []):
        if b.get("run_type") == "aggregate":
            continue
        name = b["name"]
        time = float(b["real_time"])
        unit = b.get("time_unit", "ns")
        # Normalise everything to nanoseconds for a uniform comparison unit.
        if unit == "us":
            time *= 1_000
        elif unit == "ms":
            time *= 1_000_000
        elif unit == "s":
            time *= 1_000_000_000
        # Accumulate all repetitions; we take the median later.
        result.setdefault(name, [])  # type: ignore[arg-type]
        result[name].append(time)  # type: ignore[index]

    # Reduce each list to its median.
    import statistics

    return {k: statistics.median(v) for k, v in result.items()}  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regression gate: compare benchmark JSON files."
    )
    parser.add_argument("current", help="Current (candidate) benchmark JSON path")
    parser.add_argument(
        "--baseline",
        help="Baseline benchmark JSON path. If omitted, comparison is skipped.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Maximum allowed regression fraction (default 0.05 = 5%%).",
    )
    args = parser.parse_args()

    current = load_benchmarks(args.current)

    if not args.baseline:
        print("No baseline provided; skipping regression check.")
        sys.exit(0)

    baseline = load_benchmarks(args.baseline)

    failures: list[tuple[str, float, float, float]] = []
    rows: list[tuple[str, float, float, float, str]] = []

    for name, cur_ns in sorted(current.items()):
        if name not in baseline:
            continue
        base_ns = baseline[name]
        delta = (cur_ns - base_ns) / base_ns
        status = "PASS"
        if delta > args.threshold:
            status = f"FAIL (+{delta * 100:.1f}%)"
            failures.append((name, base_ns, cur_ns, delta))
        rows.append((name, base_ns, cur_ns, delta, status))

    # Print Markdown table (GitHub-flavoured).
    header = "| Benchmark | Baseline (ns) | Candidate (ns) | Delta | Status |"
    sep = "|-----------|--------------|----------------|-------|--------|"
    print(header)
    print(sep)
    for name, base_ns, cur_ns, delta, status in rows:
        sign = "+" if delta >= 0 else ""
        print(
            f"| {name} | {base_ns:.1f} | {cur_ns:.1f} | "
            f"{sign}{delta * 100:.1f}% | {status} |"
        )

    if not rows:
        print("| (no matching benchmarks found) | — | — | — | — |")

    print()
    if failures:
        print(
            f"REGRESSION DETECTED: {len(failures)} benchmark(s) regressed "
            f">{args.threshold * 100:.0f}%."
        )
        for name, base_ns, cur_ns, delta in failures:
            print(f"  {name}: {base_ns:.1f} ns -> {cur_ns:.1f} ns (+{delta * 100:.1f}%)")
        sys.exit(1)
    else:
        print(f"No regressions above {args.threshold * 100:.0f}% threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()
