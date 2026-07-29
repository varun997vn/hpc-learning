#!/usr/bin/env python3
"""PTQ calibration tool — collects FP32 tensor statistics from numpy .npy files
and writes per-tensor quantization parameters to JSON."""
import argparse
import glob
import json
import os
import sys


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        print("ERROR: numpy not installed. Run: pip install numpy", file=sys.stderr)
        sys.exit(1)


def collect_stats(paths, np):
    stats = {}
    for p in paths:
        arr = np.load(p).astype(np.float32).ravel()
        if arr.size == 0:
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        mn, mx = float(arr.min()), float(arr.max())
        stats[name] = {
            "min": mn,
            "max": mx,
            "abs_max": float(max(abs(mn), abs(mx))),
        }
    return stats


def compute_symmetric(stats):
    out = {}
    for name, s in stats.items():
        scale = s["abs_max"] / 127.0 if s["abs_max"] > 0 else 1.0
        out[name] = {"scale": scale, "zero_point": 0, "method": "symmetric"}
    return out


def compute_asymmetric(stats):
    out = {}
    for name, s in stats.items():
        r = s["max"] - s["min"]
        scale = r / 255.0 if r > 0 else 1.0
        zp = int(round(-s["min"] / scale))
        zp = max(-128, min(127, zp))
        out[name] = {"scale": scale, "zero_point": zp, "method": "asymmetric"}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", metavar="FILE_OR_GLOB",
                   help=".npy files or glob patterns")
    p.add_argument("--method", choices=["symmetric", "asymmetric"],
                   default="symmetric")
    p.add_argument("--output", default="calibration.json",
                   help="Output JSON path (default: calibration.json)")
    args = p.parse_args()

    np = _require_numpy()

    paths = []
    for pat in args.inputs:
        expanded = sorted(glob.glob(pat))
        paths.extend(expanded if expanded else [pat])

    if not paths:
        print("ERROR: no input files found", file=sys.stderr)
        sys.exit(1)

    stats = collect_stats(paths, np)
    if not stats:
        print("ERROR: no valid tensors loaded", file=sys.stderr)
        sys.exit(1)

    params = compute_symmetric(stats) if args.method == "symmetric" \
        else compute_asymmetric(stats)

    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Calibration params written to: {args.output}")
    print(f"  {len(params)} tensor(s), method={args.method}")


if __name__ == "__main__":
    main()
