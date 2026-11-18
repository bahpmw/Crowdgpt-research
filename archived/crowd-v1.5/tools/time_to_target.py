#!/usr/bin/env python3
"""Read scale_test loss curves and report time- and tokens-to-target-loss.

The README asks how many draws DUST needs to match backprop "at fixed tokens,
measured as time-to-target-loss, not FLOPs". ``tools/scale_test.py`` writes one
``<arm>.jsonl`` per run; this reads several of them (any mix of DUST
populations and an Adam baseline on the same tokens), and for each target eval
loss reports the first logged point at or below it: tokens consumed and
wall-clock seconds. Ratios are against the run named by ``--baseline``.

    python tools/time_to_target.py adam=runs/ttt/adam/adam.jsonl \
        dust-128=runs/ttt/p128/dust.jsonl --targets 7.0 6.5 6.0 --baseline adam

Targets default to evenly spaced losses between the shared starting loss and
the worst run's best loss, so every target is reached by every run that
learned anything at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "crowd-v15-time-to-target-1"


def load_curve(path: Path) -> list[dict]:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("evalLoss") is not None:
            points.append({"tokens": int(record["tokens"]), "seconds": float(record["seconds"]),
                           "evalLoss": float(record["evalLoss"])})
    return points


def first_reach(points: list[dict], target: float) -> dict | None:
    for point in points:
        if point["evalLoss"] <= target:
            return point
    return None


def default_targets(curves: dict[str, list[dict]], count: int) -> list[float]:
    start = max(points[0]["evalLoss"] for points in curves.values() if points)
    floor = max(min(p["evalLoss"] for p in points) for points in curves.values() if points)
    if floor >= start:
        return [floor]
    step = (start - floor) / count
    return [round(start - step * (i + 1), 4) for i in range(count)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="label=path/to/arm.jsonl")
    parser.add_argument("--targets", type=float, nargs="*", default=None)
    parser.add_argument("--count", type=int, default=4, help="number of default targets")
    parser.add_argument("--baseline", default=None, help="label the ratios are relative to")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    curves: dict[str, list[dict]] = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected label=path, got {spec!r}")
        curves[label] = load_curve(Path(path))
        if not curves[label]:
            raise SystemExit(f"{path}: no evalLoss records")
    targets = args.targets or default_targets(curves, args.count)

    rows = []
    for target in targets:
        reached = {label: first_reach(points, target) for label, points in curves.items()}
        base = reached.get(args.baseline) if args.baseline else None
        for label, point in reached.items():
            row = {"target": target, "run": label,
                   "tokens": point["tokens"] if point else None,
                   "seconds": point["seconds"] if point else None,
                   "tokenRatio": None, "timeRatio": None}
            if point and base and base["tokens"] and base["seconds"]:
                row["tokenRatio"] = round(point["tokens"] / base["tokens"], 3)
                row["timeRatio"] = round(point["seconds"] / base["seconds"], 3)
            rows.append(row)

    print(f"{'target':>7} {'run':>14} {'tokens':>11} {'seconds':>9} {'x tokens':>9} {'x time':>8}")
    for row in rows:
        tokens = f"{row['tokens']:,}" if row["tokens"] is not None else "never"
        seconds = f"{row['seconds']:.1f}" if row["seconds"] is not None else "-"
        token_ratio = f"{row['tokenRatio']:.2f}" if row["tokenRatio"] is not None else "-"
        time_ratio = f"{row['timeRatio']:.2f}" if row["timeRatio"] is not None else "-"
        print(f"{row['target']:>7.3f} {row['run']:>14} {tokens:>11} {seconds:>9} "
              f"{token_ratio:>9} {time_ratio:>8}")

    best = {label: min(p["evalLoss"] for p in points) for label, points in curves.items()}
    result = {"schema": SCHEMA, "baseline": args.baseline, "targets": targets,
              "bestEvalLoss": best, "rows": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
