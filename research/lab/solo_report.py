"""SOLO scaling table: gap to the matching backprop (K=1) run, per-module cosine, worker share.

Usage: python solo_report.py s-10m s-100m ... [--tokens 1e6 2e6 4e6] [--md out.md]
A run's baseline is the K=1 ``solo`` run with the same preset, seq, batch and token budget
(for ``mosaic`` runs: the ``solo`` K=1 run with the same preset/seq/batch/budget).
"""

import argparse
import json
from pathlib import Path

import torch

from crowd_v15.model import GPT, ModelConfig
from summarize import RESULTS, at, fmt


def module_share(r):
    """Largest fraction of the model's parameters one module owns (what a worker must hold)."""
    cfg = r["cfg"]
    if r["method"] == "mosaic":
        return max((row.get("workerParamFrac", 0) for row in r["curve"]), default=None)
    K = int(cfg.get("K", 1))
    if K == 1:
        return 1.0
    with torch.device("meta"):
        m = GPT(ModelConfig.preset(r["preset"]))
    L = len(m.blocks)
    blk = sum(p.numel() for p in m.blocks[0].parameters())
    emb = m.tok_emb.weight.numel()
    total = r["params"] + emb  # untied head is an extra V x d matrix
    sizes = [len(c) for c in torch.arange(L).tensor_split(K)]
    shares = [sizes[0] * blk + emb] + [s * blk for s in sizes[1:-1]] + [sizes[-1] * blk + emb]
    return max(shares) / total


def key(r):
    return (r["preset"], r["seq"], r["batch"], r.get("budgetTokens", 0), r["seed"])


def cos_means(r):
    rows = [row for row in r["curve"] if any(k.startswith("cos") and k != "cosIt" for k in row)]
    rows = rows[len(rows) // 2:]
    ks = sorted({k for row in rows for k in row if k.startswith("cos") and k[3:].isdigit()})
    return [round(sum(row.get(k, 0) for row in rows) / max(len(rows), 1), 3) for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="+")
    ap.add_argument("--tokens", type=float, nargs="*", default=[1e6, 2e6, 4e6])
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))
            if any(p.stem.startswith(x) for x in a.prefix)]
    base = {key(r): r for r in runs if r["method"] == "solo" and int(r["cfg"].get("K", 1)) == 1}
    head = (["run", "params", "K", "tok/s"] + [f"L@{x/1e6:g}M" for x in a.tokens]
            + ["final", "BP final", "gap %", "cos/module", "worker share", "up MB/step"])
    rows = []
    for r in runs:
        b = base.get(key(r))
        cfg = r["cfg"]
        final = r["finalLoss"]
        gap = None if b is None or b is r else 100 * (final - b["finalLoss"]) / b["finalLoss"]
        secs = r.get("trainSeconds") or r["curve"][-1]["seconds"]
        rows.append([r["label"], f"{r['params']/1e6:.0f}M", str(cfg.get("K", 1)),
                     f"{r['totals']['tokens']/secs/1e3:.1f}k"]
                    + [fmt(at(r["curve"], "tokens", x)) for x in a.tokens]
                    + [fmt(final), fmt(b["finalLoss"]) if b else "-",
                       "-" if gap is None else f"{gap:+.2f}",
                       " ".join(f"{c:.2f}" for c in cos_means(r)) or "-",
                       fmt(module_share(r), 2), f"{r['upBytesPerStep']/1e6:.3f}"])
    widths = [max(len(h), *(len(x[i]) for x in rows)) for i, h in enumerate(head)]
    lines = ["| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += ["| " + " | ".join(c.ljust(w) for c, w in zip(x, widths)) + " |" for x in rows]
    print("\n".join(lines))
    if a.md:
        a.md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
