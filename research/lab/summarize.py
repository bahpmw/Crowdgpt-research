"""Round summary: loss at fixed wall time and fixed tokens, plus crowd metrics."""

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def at(curve, key, x):
    """Eval loss at curve[key] == x, linearly interpolated between logged
    points (None if the run never got that far)."""
    prev = None
    for row in curve:
        if row[key] >= x:
            if prev is None or row[key] == prev[key]:
                return row["evalLoss"]
            f = (x - prev[key]) / (row[key] - prev[key])
            return prev["evalLoss"] + f * (row["evalLoss"] - prev["evalLoss"])
        prev = row
    return None


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="*", default=[""])
    ap.add_argument("--seconds", type=float, nargs="*", default=[30, 60, 120])
    ap.add_argument("--tokens", type=float, nargs="*", default=[1e5, 5e5, 2e6])
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    runs = []
    for p in sorted(RESULTS.glob("*.json")):
        if not any(p.stem.startswith(x) for x in a.prefix) or p.stem.startswith("smoke"):
            continue
        runs.append(json.loads(p.read_text()))
    head = (["run", "steps", "tok/s"] + [f"L@{int(s)}s" for s in a.seconds]
            + [f"L@{x/1e6:g}Mtok" for x in a.tokens]
            + ["best", "up MB/step", "up B/token", "fwd-eq/token", "diverged"])
    rows = []
    for r in sorted(runs, key=lambda r: r["bestLoss"]):
        c, t = r["curve"], r["totals"]
        secs = c[-1]["seconds"] or 1
        rows.append([
            r["label"], str(r["steps"]), f"{t['tokens']/secs:,.0f}",
            *[fmt(at(c, "seconds", s)) for s in a.seconds],
            *[fmt(at(c, "tokens", x)) if t["tokens"] >= x else "-" for x in a.tokens],
            fmt(r["bestLoss"]), f"{r['upBytesPerStep']/1e6:.3f}",
            f"{t['up']/max(t['tokens'],1):,.1f}", f"{t['compute']/max(t['tokens'],1):.1f}",
            "yes" if r["diverged"] else "",
        ])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    text = "\n".join(lines)
    print(text)
    if a.md:
        a.md.write_text(text + "\n")


if __name__ == "__main__":
    main()
