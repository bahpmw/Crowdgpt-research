"""Baseline: MeZO / SPSA in weight space, seed-replayable (upload = K scalars).

Each step: K seeded directions z_k, projected grad c_k = (L(w+eps z)-L(w-eps z))/(2 eps),
update w -= lr/K * sum_k c_k z_k (regenerated from seed, no gradient stored).
Crowd view: a worker uploads (seed, c_k) pairs - a few bytes per direction.
"""

import torch

from harness import Cost
from zo import perturb, spsa_pair, trainable


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        self.k = int(cfg.get("k", 4))
        self.eps = float(cfg.get("eps", 1e-3))
        self.lr = float(cfg.get("lr", 1e-4))
        self.params = trainable(model)

    def estimate(self, idx, tgt, it):
        g = {n: torch.zeros_like(p) for n, p in self.params.items()}
        for k in range(self.k):
            seed = it * 1000 + k
            c, _, _ = spsa_pair(self.model, self.params, idx, tgt, seed, self.eps)
            perturb(g, seed, c / self.k)
        return g, 2 * self.k

    @torch.no_grad()
    def step(self, it):
        idx, tgt = self.data.next()
        cs = []
        for k in range(self.k):
            seed = it * 1000 + k
            c, _, _ = spsa_pair(self.model, self.params, idx, tgt, seed, self.eps)
            cs.append((seed, c))
        for seed, c in cs:
            perturb(self.params, seed, -self.lr * c / self.k)
        n = idx.numel()
        return Cost(tokens=n, fwd=2 * self.k * n, up=12 * self.k,
                    extra={"projGrad": round(cs[0][1], 4)})
