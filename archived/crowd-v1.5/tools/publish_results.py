#!/usr/bin/env python3
"""Publish a training run's results to the Hugging Face repo.

The weights baton answers "what is the model"; a 24-hour run also has to answer
"is it learning, and how fast". This uploads the raw evidence next to the
weights so a reader can check the claim instead of trusting a screenshot:

    results/<machine>/<arm>.jsonl     every log line the trainer emitted
    results/<machine>/<arm>.summary   final step/tokens/bestEval
    results/<machine>/curve.csv       step,tokens,trainLoss,evalLoss (long form)
    results/<machine>/curve.svg       the same curve, dependency-free SVG
    results/<machine>/RESULTS.md      one table a human can read

The SVG is hand-rolled on purpose: the trainer must not need matplotlib, and a
plain <polyline> renders in the Hub's file view, in a browser and in a diff.

    python tools/publish_results.py --dir runs/24h --repo peva3/Crowd-v1.5-dust \\
        --machine 4060ti --loop-minutes 30
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

DEFAULT_REPO = "peva3/Crowd-v1.5-dust"
COLOURS = {"trainLossInstant": "#2563eb", "trainLoss": "#1d4ed8", "evalLoss": "#dc2626"}


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A trainer killed mid-write can leave one partial line; ignore it
            # rather than refusing to publish the rest of the evidence.
            continue
    return records


def collect(run_dir: Path) -> dict[str, list[dict]]:
    """Group every ``<arm>.jsonl`` in a directory by arm name."""
    arms: dict[str, list[dict]] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        records = read_jsonl(path)
        if records:
            arms[path.stem] = records
    return arms


def summarise(records: list[dict]) -> dict:
    evals = [r for r in records if r.get("evalLoss") is not None]
    last = records[-1] if records else {}
    best = min((r["evalLoss"] for r in evals), default=None)
    first_eval = evals[0]["evalLoss"] if evals else None
    return {
        "steps": last.get("step"),
        "tokens": last.get("tokens"),
        "seconds": last.get("seconds"),
        "tokensPerSecond": last.get("tokensPerSecond"),
        "peakVramGb": last.get("peakVramGb"),
        "trainLoss": last.get("trainLoss"),
        "firstEvalLoss": first_eval,
        "bestEvalLoss": best,
        "lastEvalLoss": evals[-1]["evalLoss"] if evals else None,
        "evaluations": len(evals),
    }


def curve_rows(records: Iterable[dict]) -> list[dict]:
    rows = []
    for r in records:
        if r.get("trainLoss") is None and r.get("evalLoss") is None:
            continue
        rows.append({
            "step": r.get("step"),
            "tokens": r.get("tokens"),
            "trainLoss": r.get("trainLoss"),
            "evalLoss": r.get("evalLoss"),
        })
    return rows


def to_csv(rows: list[dict]) -> str:
    out = ["step,tokens,trainLoss,evalLoss"]
    for r in rows:
        out.append(",".join("" if r[k] is None else str(r[k])
                            for k in ("step", "tokens", "trainLoss", "evalLoss")))
    return "\n".join(out) + "\n"


def _scale(values: list[float], lo: float, hi: float, size: float) -> list[float]:
    span = (hi - lo) or 1.0
    return [size - (v - lo) / span * size for v in values]


def to_svg(rows: list[dict], title: str, width: int = 720, height: int = 320,
           pad: int = 44) -> str:
    """Dependency-free loss-vs-tokens chart (x = tokens, y = loss)."""
    series: dict[str, list[tuple[float, float]]] = {}
    for key in ("trainLoss", "evalLoss"):
        pts = [(float(r["tokens"]), float(r[key]))
               for r in rows if r.get(key) is not None and r.get("tokens")]
        if pts:
            series[key] = pts
    if not series:
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">'
                '<text x="16" y="32" font-family="monospace" font-size="13">'
                'no data yet</text></svg>' % (width, height))

    xs = [x for pts in series.values() for x, _ in pts]
    ys = [y for pts in series.values() for _, y in pts]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    pad_y = (y_hi - y_lo) * 0.08 or 0.05
    y_lo, y_hi = y_lo - pad_y, y_hi + pad_y
    w, h = width - 2 * pad, height - 2 * pad

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad}" y="24" font-family="monospace" font-size="13" fill="#111">{title}</text>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + h}" stroke="#888"/>',
        f'<line x1="{pad}" y1="{pad + h}" x2="{pad + w}" y2="{pad + h}" stroke="#888"/>',
    ]
    for frac in (0.0, 0.5, 1.0):
        y = pad + h * frac
        value = y_hi - (y_hi - y_lo) * frac
        parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad + w}" y2="{y:.1f}" '
                     f'stroke="#eee"/>')
        parts.append(f'<text x="4" y="{y + 4:.1f}" font-family="monospace" '
                     f'font-size="10" fill="#555">{value:.2f}</text>')
    for frac in (0.0, 0.5, 1.0):
        x = pad + w * frac
        value = x_lo + (x_hi - x_lo) * frac
        label = f"{value / 1e6:.1f}M" if value >= 1e6 else f"{value:.0f}"
        parts.append(f'<text x="{x:.1f}" y="{pad + h + 16}" text-anchor="middle" '
                     f'font-family="monospace" font-size="10" fill="#555">{label}</text>')

    for key, pts in series.items():
        px = [pad + (x - x_lo) / ((x_hi - x_lo) or 1.0) * w for x, _ in pts]
        py = _scale([y for _, y in pts], y_lo, y_hi, h)
        py = [pad + v for v in py]
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))
        dash = ' stroke-dasharray="4 3"' if key == "evalLoss" else ""
        radius = 2 if key == "evalLoss" else 1
        parts.append(f'<polyline points="{points}" fill="none" '
                     f'stroke="{COLOURS[key]}" stroke-width="{radius}"{dash}/>')

    legend_x = pad
    for key in series:
        parts.append(f'<line x1="{legend_x}" y1="{height - 10}" x2="{legend_x + 18}" '
                     f'y2="{height - 10}" stroke="{COLOURS[key]}" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x + 22}" y="{height - 6}" '
                     f'font-family="monospace" font-size="10" fill="#333">{key}</text>')
        legend_x += 150
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def to_markdown(arm: str, stats: dict, machine: str) -> str:
    def fmt(value, digits=4):
        return "—" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))
    return (
        f"| arm | steps | tokens | tokens/s | peak GB | first eval | best eval | last eval |\n"
        f"|---|---|---|---|---|---|---|---|\n"
        f"| `{arm}` | {fmt(stats['steps'])} | {fmt(stats['tokens'])} | "
        f"{fmt(stats['tokensPerSecond'], 1)} | {fmt(stats['peakVramGb'], 2)} | "
        f"{fmt(stats['firstEvalLoss'])} | {fmt(stats['bestEvalLoss'])} | "
        f"{fmt(stats['lastEvalLoss'])} |\n"
    )


def build_uploads(run_dir: Path, machine: str) -> dict[str, bytes]:
    """Everything to publish for one machine, keyed by path inside the repo."""
    arms = collect(run_dir)
    if not arms:
        raise SystemExit(f"no *.jsonl with records under {run_dir}")
    uploads: dict[str, bytes] = {}
    tables = []
    for arm, records in arms.items():
        stats = summarise(records)
        rows = curve_rows(records)
        base = f"results/{machine}"
        uploads[f"{base}/{arm}.jsonl"] = (run_dir / f"{arm}.jsonl").read_bytes()
        uploads[f"{base}/{arm}.summary.json"] = (
            json.dumps(stats, indent=2, sort_keys=True) + "\n").encode()
        uploads[f"{base}/{arm}.curve.csv"] = to_csv(rows).encode()
        uploads[f"{base}/{arm}.curve.svg"] = to_svg(
            rows, f"{machine} / {arm} — loss vs tokens").encode()
        tables.append((arm, stats))

    header = ("# Training results\n\n"
              f"Machine: `{machine}` · updated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n"
              "Generated by `tools/publish_results.py`. The raw JSONL is the authority; the\n"
              "SVG/CSV are derived from it. A run that is still training grows in place.\n\n")
    body = header + "\n".join(to_markdown(arm, stats, machine) for arm, stats in tables)
    body += ("\n![curve](%s.curve.svg)\n" % tables[0][0]) if len(tables) == 1 else ""
    uploads[f"results/{machine}/RESULTS.md"] = body.encode()

    index = ("# Run index\n\n"
             "| machine | arm | tokens | tokens/s | best eval |\n|---|---|---|---|---|\n")
    for arm, stats in tables:
        index += (f"| `{machine}` | `{arm}` | {stats['tokens']} | "
                  f"{stats['tokensPerSecond']} | {stats['bestEvalLoss']} |\n")
    uploads[f"results/{machine}/index.md"] = index.encode()
    return uploads


def publish(uploads: dict[str, bytes], repo: str, token_env: str) -> list[str]:
    from huggingface_hub import CommitOperationAdd, HfApi

    token = os.environ.get(token_env) or ""
    if not token:
        raise SystemExit(f"{token_env} is not set; publishing needs a write token")
    api = HfApi(token=token)
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=blob)
                  for path, blob in sorted(uploads.items())]
    api.create_commit(repo_id=repo, repo_type="model", operations=operations,
                      commit_message=f"results: publish {len(uploads)} run artefacts")
    return sorted(uploads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", required=True, help="run output directory (scale_test --out)")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--machine", required=True, help="label, e.g. 4060ti or gfx1150")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--env-file", default="", help="parse KEY=VALUE lines from this file")
    parser.add_argument("--loop-minutes", type=float, default=0.0,
                        help="republish every N minutes instead of once")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the artefacts and print them without uploading")
    args = parser.parse_args(argv)

    if args.env_file and os.path.exists(args.env_file):
        for line in Path(args.env_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    while True:
        uploads = build_uploads(Path(args.dir), args.machine)
        if args.dry_run:
            for path in sorted(uploads):
                print(path, len(uploads[path]), "bytes")
        else:
            for path in publish(uploads, args.repo, args.token_env):
                print("uploaded", path, flush=True)
        if args.loop_minutes <= 0:
            return 0
        time.sleep(max(30.0, args.loop_minutes * 60.0))


if __name__ == "__main__":
    raise SystemExit(main())
