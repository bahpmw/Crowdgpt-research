"""ZO at the Edge of Stability (2604.14669): MeZO-GDM with a trace-set learning rate.

Base: Gaussian two-point SPSA with n directions, Polyak momentum as in the
paper (m = beta*m + g; w -= lr*m, no (1-beta) dampening).
Per step we add one clean forward f0, so each direction also yields a
forward-only Hessian-trace sample  (f+ + f- - 2 f0)/eps^2 = z^T H z + O(eps^2),
whose mean over Gaussian z is Tr H. An EMA of it (beta_tr, bias-corrected,
clamped positive) sets
    lr = min(lr_max, c * 2 (1-beta) n / TrH_ema).
Theorem 2 bounds the ZO-GDM mean-square limit in
[2(1-beta)/(Tr + 2 lambda_max/(1+beta)), 2(1-beta)/Tr] and also needs
lr*lambda_max < 1 - beta^2 (checked in the PDF: beta squared, not (1-beta)^2).
lambda_max is unknown forward-only, so we use the upper bound 2(1-beta)/Tr and
rely on c < 1 for the gap. The factor n (averaged directions) is our
extrapolation from the paper's App. B n-query operator, valid until the FO
limit ~2(1+beta)/lambda_max. Crowd view: a worker uploads (seed, c_k, curv_k)
per direction plus f0; momentum is replayable by any party from the seeds.

Measured (smoke, 10m, batch 8x128): the paper's full-batch bound is far too
loose for mini-batch LM training. c=0.5 and c=0.2 diverge within 50 steps,
c=0.1 is stable but the loss rises (Tr falls, so lr rises: an EoS feedback
loop), and c=0.05 trains. Likely cause: data noise. The useful lr shrinks by
roughly the gradient SNR ||E g||^2 / E||g_batch||^2, which the theory leaves
out. The default stays 0.5 as specified; pass c=0.05 for real runs.
"""

import torch

from harness import Cost, loss_fn
from zo import perturb, spsa_pair, trainable


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        self.n = int(cfg.get("n", 4))
        self.eps = float(cfg.get("eps", 1e-3))
        self.beta = float(cfg.get("beta", 0.0))
        self.c = float(cfg.get("c", 0.5))
        self.lr_max = float(cfg.get("lr_max", 1e-2))
        self.beta_tr = float(cfg.get("beta_tr", 0.99))
        self.params = trainable(model)
        self.buf = None
        self.tr_ema, self.tr_w = 0.0, 0.0

    @torch.no_grad()
    def probe(self, idx, tgt, it):
        f0 = float(loss_fn(self.model, idx, tgt))
        cs, trs = [], []
        for k in range(self.n):
            seed = it * 1000 + k
            c, lp, lm = spsa_pair(self.model, self.params, idx, tgt, seed, self.eps)
            cs.append((seed, c))
            trs.append((lp + lm - 2 * f0) / self.eps ** 2)
        return cs, sum(trs) / len(trs), f0

    def estimate(self, idx, tgt, it):
        cs, _, _ = self.probe(idx, tgt, it)
        g = {n: torch.zeros_like(p) for n, p in self.params.items()}
        for seed, c in cs:
            perturb(g, seed, c / self.n)
        return g, 2 * self.n + 1

    @torch.no_grad()
    def step(self, it):
        idx, tgt = self.data.next()
        cs, tr, f0 = self.probe(idx, tgt, it)
        self.tr_ema = self.beta_tr * self.tr_ema + (1 - self.beta_tr) * tr
        self.tr_w = self.beta_tr * self.tr_w + (1 - self.beta_tr)
        tr_hat = max(self.tr_ema / self.tr_w, 1e-6)
        lr = min(self.lr_max, self.c * 2 * (1 - self.beta) * self.n / tr_hat)
        if self.beta == 0.0:
            for seed, c in cs:
                perturb(self.params, seed, -lr * c / self.n)
        else:
            if self.buf is None:
                self.buf = {n: torch.zeros_like(p) for n, p in self.params.items()}
            for b in self.buf.values():
                b.mul_(self.beta)
            for seed, c in cs:
                perturb(self.buf, seed, c / self.n)
            for n, p in self.params.items():
                p.add_(self.buf[n], alpha=-lr)
        n = idx.numel()
        return Cost(tokens=n, fwd=(2 * self.n + 1) * n, up=8 + 4 * (2 * self.n + 1),
                    extra={"trH": round(tr, 2), "trEma": round(tr_hat, 2),
                           "lr": float(f"{lr:.3g}"), "loss": round(f0, 4)})
