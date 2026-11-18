"""Estimator quality: cosine to the backprop gradient at a matched forward budget.

Any method exposing ``estimate(idx, tgt, it) -> (grads, forwards)`` can be
probed. ``grads`` is keyed like ``zo.trainable(model)``; the cosine covers the
2-D block weights only (the tied embedding/head gets an exact head gradient in
several arms, which would flatter them). Optionally starts from a trained
state_dict, because alignment at random init and mid-training differ.

  python probe.py "dust pop=32 chunk=16" "szo kind=szo1 n=8" --ckpt ckpt/adam30.pt
"""

from __future__ import annotations

import argparse
import importlib
import json
import shlex
import sys
import time
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

from actzo import backprop  # noqa: E402
from harness import ROOT, Data, build_model, parse_cfg, sync  # noqa: E402


def block_cosine(est: dict, ref: dict) -> float:
    keys = [k for k in est if k in ref and k.startswith("blocks.") and ref[k].dim() == 2]
    a = torch.cat([est[k].flatten().float().cpu().double() for k in keys])
    b = torch.cat([ref[k].flatten().float().cpu().double() for k in keys])
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help='"method k=v ..." strings')
    ap.add_argument("--preset", default="10m")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    dev = torch.device(a.device)
    rows = []
    for arm in a.arms:
        name, *pairs = shlex.split(arm)
        model = build_model(a.preset, 0, dev)
        if a.ckpt:
            model.load_state_dict(torch.load(a.ckpt, map_location=dev))
        data = Data(ROOT / "runs/data/tok", a.seq, a.batch, 7, dev)
        method = importlib.import_module(f"methods.{name}").Method(
            model, data, parse_cfg(pairs), dev)
        cos, fw, secs = [], [], []
        for r in range(a.repeats):
            idx, tgt = data.next()
            ref = backprop(model, idx, tgt)
            sync(dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                est, forwards = method.estimate(idx, tgt, r)
            sync(dev)
            secs.append(time.perf_counter() - t0)
            cos.append(block_cosine(est, ref))
            fw.append(forwards)
        row = {"arm": arm, "cosine": sum(cos) / len(cos), "cosines": cos,
               "forwards": sum(fw) / len(fw), "seconds": sum(secs) / len(secs),
               "ckpt": str(a.ckpt) if a.ckpt else None}
        rows.append(row)
        print(f"{arm:50s} cos {row['cosine']:+.4f}  fwd {row['forwards']:7.1f}  "
              f"{row['seconds']:.2f}s", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
