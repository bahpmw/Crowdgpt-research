"""LOO-ROLL (2609.10980): rank-1 weight-space ES with a leave-one-out baseline.

Each step draws N members. Member s perturbs every 2-D matrix by sigma*a b^T
(a, b ~ N(0, I), rank 1) and every 1-D param (RMSNorm gains) by dense
sigma*z (as EGGROLL does for vectors). One forward per member on a shared
batch; fitness F_s = loss (minimised).

  baseline=loo     g = 1/(N sigma) sum_s E_s (F_s - mean_{j!=s} F_j)       N members
  baseline=mirror  EGGROLL antithetic: N/2 pairs (+E_p, -E_p),
                   g = 1/(N/2) sum_p E_p (F+_p - F-_p) / (2 sigma)        same N forwards
``zscore=true`` divides the centred scores by the population std (EGGROLL's
standardisation; rescale only). Update: SGD momentum on g with global-norm
clip (the paper does not specify an optimizer; this is our choice).
Seeds per (step, member) regenerate (a, b, z). The update is A diag(w) B^T per
matrix, replayable from the seed and N fitness scalars: upload = 4N + 8 bytes.
"""

import torch

from harness import Cost, loss_fn
from zo import gen, trainable


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        self.n = int(cfg.get("n", 32))
        self.sigma = float(cfg.get("sigma", 1e-3))
        self.lr = float(cfg.get("lr", 0.5))
        self.mom = float(cfg.get("momentum", 0.9))
        self.clip = float(cfg.get("clip", 1.0))
        self.zscore = bool(cfg.get("zscore", True))
        self.baseline = cfg.get("baseline", "loo")
        if self.baseline == "mirror" and self.n % 2:
            raise ValueError("mirror needs an even population n")
        self.params = trainable(model)
        self.buf = {}

    def _draw(self, seed):
        g = gen(seed, next(iter(self.params.values())).device)
        out = {}
        for name, p in self.params.items():
            if p.dim() == 2:
                out[name] = (torch.randn(p.shape[0], generator=g, device=p.device),
                             torch.randn(p.shape[1], generator=g, device=p.device))
            else:
                out[name] = torch.randn(p.shape, generator=g, device=p.device)
        return out

    def _apply(self, d, scale):
        for name, p in self.params.items():
            if p.dim() == 2:
                p.addr_(d[name][0], d[name][1], alpha=scale)
            else:
                p.add_(d[name], alpha=scale)

    @torch.no_grad()
    def estimate(self, idx, tgt, it):
        grads, forwards, _ = self._estimate(idx, tgt, it)
        return grads, forwards

    def _estimate(self, idx, tgt, it):
        mirror = self.baseline == "mirror"
        draws = [self._draw(it * 100003 + s) for s in range(self.n // 2 if mirror else self.n)]
        signs = (1, -1) if mirror else (1,)
        F = []
        for d in draws:
            for sg in signs:
                self._apply(d, sg * self.sigma)
                F.append(float(loss_fn(self.model, idx, tgt)))
                self._apply(d, -sg * self.sigma)
        F = torch.tensor(F, dtype=torch.float64)
        sd = float(F.std()) + 1e-12 if self.zscore else 1.0
        if mirror:
            Fp, Fm = F[0::2], F[1::2]
            w = (Fp - Fm) / (2 * sd) / (self.sigma * len(draws))
        else:
            w = (F - F.mean()) * self.n / (self.n - 1) / sd / (self.sigma * self.n)
        grads = {}
        for name, p in self.params.items():
            if p.dim() == 2:
                A = torch.stack([d[name][0] for d in draws], 1)          # (m, N)
                B = torch.stack([d[name][1] for d in draws], 1)          # (n, N)
                grads[name] = (A * w.to(A).unsqueeze(0)) @ B.t()
            else:
                grads[name] = sum(float(w[s]) * d[name] for s, d in enumerate(draws))
        return grads, self.n, F

    @torch.no_grad()
    def step(self, it):
        idx, tgt = self.data.next()
        grads, forwards, F = self._estimate(idx, tgt, it)
        norm = float(torch.sqrt(sum(g.pow(2).sum() for g in grads.values())))
        scale = min(1.0, self.clip / (norm + 1e-12)) if self.clip else 1.0
        for name, g in grads.items():
            b = self.buf.setdefault(name, torch.zeros_like(g))
            b.mul_(self.mom).add_(g, alpha=scale)
            self.params[name].add_(b, alpha=-self.lr)
        n = idx.numel()
        return Cost(tokens=n, fwd=forwards * n, up=4 * self.n + 8,
                    extra={"gradNorm": round(norm, 3), "fitStd": round(float(F.std()), 5)})
