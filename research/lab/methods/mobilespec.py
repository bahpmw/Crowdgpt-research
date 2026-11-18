"""MobileSpec / FuRA-S (2611.04797): forward-only ES over block singular values.

Each 2-D block weight W (m x n, [out, in]) is split into K = n/b column blocks
(b = ``block``, default 64; the paper picks b ~ sqrt(n), i.e. 16/32 here) and
each block is factored once, W_k = U_k diag(s_k) V_k^T. Only the block singular
values train, relatively: s' = s * (1 + delta). Exact rewrite used for the
forward (paper Eq. 4-5): W' = W0 + sum_k (U_k diag(s_k * delta_k)) V_k^T, so
delta = 0 reproduces W0 bit-exactly. In delta coordinates D = mu_rel*|theta0|
becomes a plain mu_rel, so the antithetic ES update is
    delta += (alpha/N) mu_rel sum_j (f~+_j - f~-_j) z_j,   N = 2*pairs members,
with fitness f = -loss standardised over the whole population, all members on
one shared batch, no momentum. The tied embedding/head and RMSNorm gains are
frozen (no forward-only exact path for them; the paper adapts linears only).
Fine-tuning method: expected to stall from scratch; run with --init too.
Crowd view: upload = seed + (f+, f-) per pair = 8*pairs + 8 bytes.
"""

import torch

from harness import Cost, loss_fn
from zo import gen


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data, self.device = model, data, device
        self.pairs = int(cfg.get("pairs", 15))
        self.mu = float(cfg.get("mu_rel", 3e-2))
        self.alpha = float(cfg.get("alpha", 1.0))
        b = int(cfg.get("block", 64))
        self.w, self.W0, self.US, self.Vh = {}, {}, {}, {}
        for name, p in model.named_parameters():
            if not name.startswith("blocks.") or p.dim() != 2:
                continue
            m, n = p.shape
            blocks = p.detach().cpu().double().view(m, n // b, b).permute(1, 0, 2)  # (K, m, b)
            U, S, Vh = torch.linalg.svd(blocks, full_matrices=False)
            self.w[name], self.W0[name] = p, p.detach().clone()
            self.US[name] = (U * S.unsqueeze(1)).float().to(device)          # (K, m, b)
            self.Vh[name] = Vh.float().to(device)                            # (K, b, b)
        self.delta = {n: torch.zeros(us.shape[0], us.shape[2], device=device)
                      for n, us in self.US.items()}
        self.d = sum(t.numel() for t in self.delta.values())

    def _dW(self, name, dlt):
        m = self.US[name].shape[1]
        dw = torch.bmm(self.US[name] * dlt.unsqueeze(1), self.Vh[name])          # (K, m, b)
        return dw.permute(1, 0, 2).reshape(m, -1)

    def _load(self, delta):
        for n, p in self.w.items():
            p.copy_(self.W0[n] + self._dW(n, delta[n]))

    def _z(self, seed):
        g = gen(seed, self.device)
        return {n: torch.randn(t.shape, generator=g, device=self.device) for n, t in self.delta.items()}

    @torch.no_grad()
    def _search(self, idx, tgt, it):
        zs, F = [], []
        for j in range(self.pairs):
            z = self._z(it * 1000 + j)
            zs.append(z)
            for sg in (1, -1):
                self._load({n: self.delta[n] + sg * self.mu * z[n] for n in z})
                F.append(-float(loss_fn(self.model, idx, tgt)))
        self._load(self.delta)
        F = torch.tensor(F, dtype=torch.float64)
        Fs = (F - F.mean()) / (F.std() + 1e-12)
        w = (Fs[0::2] - Fs[1::2]).tolist()
        step = {n: sum(w[j] * zs[j][n] for j in range(self.pairs)) for n in self.delta}
        return step, F

    def estimate(self, idx, tgt, it):
        step, _ = self._search(idx, tgt, it)
        return {n: -self._dW(n, s) for n, s in step.items()}, 2 * self.pairs

    @torch.no_grad()
    def step(self, it):
        idx, tgt = self.data.next()
        center = -float(loss_fn(self.model, idx, tgt))      # diagnostic (paper Fig. 5)
        step, F = self._search(idx, tgt, it)
        N = 2 * self.pairs
        for n in self.delta:
            self.delta[n].add_(step[n], alpha=self.alpha * self.mu / N)
        self._load(self.delta)
        rms = torch.cat([t.flatten() for t in self.delta.values()]).pow(2).mean().sqrt()
        n = idx.numel()
        return Cost(tokens=n, fwd=(N + 1) * n, up=8 * self.pairs + 8,
                    extra={"loss": round(-center, 4), "fitStd": round(float(F.std()), 5),
                           "beatCenter": round(float((F > center).double().mean()), 3),
                           "deltaRms": round(float(rms), 5)})
