"""SZO-kappa (2611.08712): activation-space ZO estimators + SGD momentum.

``kind`` picks the estimator from ``actzo.estimate``: ``np`` (DUST-like node
perturbation, the control), ``ntwp`` (node-through-weight, per-sample credit)
or ``szo1`` (spectrally shaped SZO-1, per-sample credit, App. I). ``n`` probes
per layer, one-sided against one clean pass, iterative probing (one layer per
forward), ``aware`` = activation-aware credit on the SwiGLU up/gate outputs.

Optimizer: optional per-tensor adaptive clip (``adaclip``: norm capped at
``adaclip`` x the lower median of the previous 100 norms after 50 warm-up
steps; the paper uses 5) -> global-norm clip (``clip``, 0 disables) -> SGD
momentum. Noise is drawn from a generator seeded by the step, so a
coordinator could replay it.

Simplifications vs the paper: transformer instead of MLP/CNN; relative eps
(1e-2 of each layer's output RMS) instead of absolute 1e-4, because MPS fp32
loss differences at 1e-4 are near rounding; momentum 0.9 (DUST's outer
optimizer) instead of plain SGD; RMSNorm weights are not trained (no estimate),
as in DUST; the tied embedding/head gets the exact head gradient plus an NP
estimate of the embedding term. Upload is the dense bf16 estimate.
"""

from collections import deque

import torch

from actzo import estimate
from harness import Cost
from zo import gen, trainable


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data, self.device = model, data, device
        self.kw = dict(kind=cfg.get("kind", "szo1"), n=int(cfg.get("n", 4)),
                       eps=float(cfg.get("eps", 1e-2)), rel=bool(cfg.get("rel", True)),
                       aware=bool(cfg.get("aware", False)))
        self.lr = float(cfg.get("lr", 0.05))
        self.mom = float(cfg.get("momentum", 0.9))
        self.clip = float(cfg.get("clip", 1.0))
        self.adaclip = float(cfg.get("adaclip", 0.0))
        self.params = trainable(model)
        self.buf, self.hist = {}, {}
        self.n = model.n_params()

    def estimate(self, idx, tgt, it):
        grads, forwards, _ = estimate(self.model, idx, tgt, gen=gen(it, self.device), **self.kw)
        return grads, forwards

    @torch.no_grad()
    def step(self, it):
        idx, tgt = self.data.next()
        grads, forwards, st = estimate(self.model, idx, tgt, gen=gen(it, self.device), **self.kw)
        clipped = 0
        if self.adaclip:
            for name, g in grads.items():
                h = self.hist.setdefault(name, deque(maxlen=100))
                gn = float(g.norm())
                if len(h) >= 50:
                    cap = self.adaclip * sorted(h)[(len(h) - 1) // 2]
                    if gn > cap:
                        g.mul_(cap / gn)
                        clipped += 1
                h.append(gn)
        norm = float(torch.sqrt(sum(g.pow(2).sum() for g in grads.values())))
        scale = min(1.0, self.clip / (norm + 1e-12)) if self.clip else 1.0
        for name, g in grads.items():
            p = self.params[name]
            b = self.buf.setdefault(name, torch.zeros_like(p))
            b.mul_(self.mom).add_(g.to(p.dtype), alpha=scale)
            p.add_(b, alpha=-self.lr)
        n_tok = idx.numel()
        return Cost(tokens=n_tok, fwd=forwards * n_tok, up=2 * self.n,
                    extra={"gradNorm": round(norm, 4), "forwards": forwards,
                           "loss": round(st["loss"], 4), "adaClipped": clipped})
