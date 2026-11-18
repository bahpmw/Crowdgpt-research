"""DIAGNOSTIC (uses backprop HVPs): Hessian trace two ways and the predicted ZO-GDM lr.

  python eos_check.py [--ckpt ckpt/adam30.pt] [--hutch 20] [--fwd 64] [--power 10]

Hutchinson (Rademacher v, v^T H v via double backward) vs the forward-only
estimator used by methods/eos.py ((f+ + f- - 2 f0)/eps^2, Gaussian z). Prints
the paper's (2604.14669, Thm 1-2) mean-square band for MeZO with one direction:
  upper 2(1-beta)/Tr H,  lower 2(1-beta)/(Tr H + 2 lambda_max/(1+beta)),
plus the extra condition lr*lambda_max < 1 - beta^2.

Caveat (measured on ckpt/adam30.pt, confirmed in float64): power iteration
finds lambda ~ -3358, about 92% on tok_emb, and Hutchinson's trace is
dominated by it (about -3200 +- 1700). The finite second difference along that
eigenvector is -3377 at h=1e-5 but about 0 at h>=1e-3. It is a micro-scale
non-quadratic feature, invisible at the ZO smoothing radius. The
forward-only number (smoothed curvature at eps) is what governs MeZO; prefer it.
At random init the two agree (about 830 vs 870).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

from harness import ROOT, Data, build_model, loss_fn  # noqa: E402
from zo import spsa_pair, trainable  # noqa: E402


def hvp(model, params, idx, tgt, v):
    loss = loss_fn(model, idx, tgt)
    g = torch.autograd.grad(loss, params, create_graph=True)
    return torch.autograd.grad(sum((a * b).sum() for a, b in zip(g, v)), params)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--preset", default="10m")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--hutch", type=int, default=20)
    ap.add_argument("--fwd", type=int, default=64)
    ap.add_argument("--power", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1e-3)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model = build_model(a.preset, 0, dev)
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    idx, tgt = Data(ROOT / "runs/data/tok", 128, 8, 7, dev).next()
    params = list(trainable(model).values())
    g = torch.Generator(device=dev).manual_seed(0)
    hs = []
    for _ in range(a.hutch):
        v = [torch.randint(0, 2, p.shape, generator=g, device=dev).float() * 2 - 1 for p in params]
        hs.append(float(sum((x * y).sum() for x, y in zip(v, hvp(model, params, idx, tgt, v)))))
    lam = 0.0
    if a.power:
        v = [torch.randn(p.shape, generator=g, device=dev) for p in params]
        for _ in range(a.power):
            nv = torch.sqrt(sum(x.pow(2).sum() for x in v))
            v = [x / nv for x in v]
            hv = hvp(model, params, idx, tgt, v)
            lam = float(sum((x * y).sum() for x, y in zip(v, hv)))
            v = [x.detach() for x in hv]
    with torch.no_grad():
        f0 = float(loss_fn(model, idx, tgt))
        named = trainable(model)
        fs = []
        for k in range(a.fwd):
            _, lp, lm = spsa_pair(model, named, idx, tgt, 99991 + k, a.eps)
            fs.append((lp + lm - 2 * f0) / a.eps ** 2)
    ht, ft = torch.tensor(hs), torch.tensor(fs)
    print(f"loss {f0:.4f}  ckpt {a.ckpt}")
    print(f"Tr H  Hutchinson (diagnostic, backprop) {ht.mean():10.2f} +- {ht.std() / len(hs) ** .5:.2f}"
          f"  ({len(hs)} probes)")
    print(f"Tr H  forward-only (f+ + f- - 2f0)/eps^2 {ft.mean():10.2f} +- {ft.std() / len(fs) ** .5:.2f}"
          f"  ({len(fs)} probes, eps {a.eps})")
    if a.power:
        print(f"lambda_max (power iteration, {a.power} it)  {lam:10.2f}")
    for name, tr in (("hutch", float(ht.mean())), ("fwd", float(ft.mean()))):
        for beta in (0.0, 0.9):
            up = 2 * (1 - beta) / tr
            lo = 2 * (1 - beta) / (tr + 2 * lam / (1 + beta))
            cap = (1 - beta ** 2) / lam if lam > 0 else float("inf")
            print(f"[{name}] n=1 beta={beta}: max stable MeZO lr in [{lo:.3g}, {up:.3g}]"
                  f"  (also lr < {cap:.3g} from lambda_max)")


if __name__ == "__main__":
    main()
