"""ZO-Muon (2602.17155): subspace RGE + Muon orthogonalisation of the r x n core.

For every 2-D matrix W (m x n, stored [out, in]; tok_emb uses its stored
[vocab, d] layout, i.e. the tied head's [out, in]) a seeded orthonormal
P (m x r, QR of a Gaussian) is redrawn every ``refresh`` steps. Each of ``nq``
probes perturbs ALL matrices at once by mu * P Psi_i (Psi_i r x n Gaussian),
one-sided against one shared clean loss f0:
  C = (1/nq) sum_i (f(W + mu P Psi_i) - f0)/mu * Psi_i          (r x n)
  ortho=true :  W -= eta * scale * P msign(C)    (Newton-Schulz quintic, 5 steps)
  ortho=false:  W -= eta * P C                   (plain Subspace-MeZO control)
``scale=muon`` multiplies by sqrt(max(1, m/n)) as Muon does; ``scale=none`` uses 1.
Simplifications: RMSNorm gains are frozen (the paper gives them a MeZO update
with no stated lr); no momentum (as in the paper).
Crowd view: upload is C per matrix in bf16 (r*n*2 bytes); P comes from a seed.
"""

import torch

from harness import Cost, loss_fn
from zo import gen, trainable


def msign(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        X = a * X + (b * A + c * A @ A) @ X
    return X.t() if tall else X


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data, self.device = model, data, device
        self.r = int(cfg.get("r", 64))
        self.nq = int(cfg.get("nq", 8))
        self.mu = float(cfg.get("mu", 1e-3))
        self.eta = float(cfg.get("eta", 1e-2))
        self.refresh = int(cfg.get("refresh", 100))
        self.ortho = bool(cfg.get("ortho", True))
        self.scale = cfg.get("scale", "muon")
        self.mats = {n: p for n, p in trainable(model).items() if p.dim() == 2}
        self.P, self.P_epoch = {}, None

    def _subspaces(self, it):
        epoch = it // self.refresh
        if epoch == self.P_epoch:
            return
        g = torch.Generator().manual_seed(7919 * epoch + 1)
        for name, p in self.mats.items():
            q, _ = torch.linalg.qr(torch.randn(p.shape[0], self.r, generator=g))
            self.P[name] = q.to(self.device)
        self.P_epoch = epoch

    @torch.no_grad()
    def core(self, idx, tgt, it):
        self._subspaces(it)
        f0 = float(loss_fn(self.model, idx, tgt))
        C = {n: torch.zeros(self.r, p.shape[1], device=p.device) for n, p in self.mats.items()}
        for i in range(self.nq):
            g = gen(it * 1000 + i, self.device)
            psi = {n: torch.randn(self.r, p.shape[1], generator=g, device=p.device)
                   for n, p in self.mats.items()}
            for n, p in self.mats.items():
                p.add_(self.P[n] @ psi[n], alpha=self.mu)
            s = (float(loss_fn(self.model, idx, tgt)) - f0) / self.mu
            for n, p in self.mats.items():
                p.sub_(self.P[n] @ psi[n], alpha=self.mu)
                C[n].add_(psi[n], alpha=s / self.nq)
        return C, f0

    def estimate(self, idx, tgt, it):
        C, _ = self.core(idx, tgt, it)
        return {n: self.P[n] @ C[n] for n in C}, self.nq + 1

    @torch.no_grad()
    def step(self, it):
        idx, tgt = self.data.next()
        C, f0 = self.core(idx, tgt, it)
        cnorm = 0.0
        for n, p in self.mats.items():
            cnorm += float(C[n].pow(2).sum())
            if self.ortho:
                m, k = p.shape
                sc = max(1.0, m / k) ** 0.5 if self.scale == "muon" else 1.0
                p.sub_(self.P[n] @ msign(C[n]), alpha=self.eta * sc)
            else:
                p.sub_(self.P[n] @ C[n], alpha=self.eta)
        n = idx.numel()
        up = sum(2 * self.r * p.shape[1] for p in self.mats.values())
        return Cost(tokens=n, fwd=(self.nq + 1) * n, up=up,
                    extra={"loss": round(f0, 4), "coreNorm": round(cnorm ** 0.5, 3)})
