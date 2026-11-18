"""WIS-MeZO / WISD-MeZO (2611.16317): MeZO with Wanda-masked sparse perturbations.

For each 2-D weight W (out x in) the perturbation is restricted to the row-wise
top-r fraction of the Wanda score S_ij = |W_ij| * ||X_j||_2, where X_j is the
j-th input feature over a calibration batch. Direction U = Z*M/sqrt(r) (same
expected energy as dense MeZO); update W -= lr/k * sum_k c_k U_k, with
c_k = (L(w+eps U_k) - L(w-eps U_k)) / (2 eps). Mask refreshed every ``tau``
steps (WISD; tau=0 means compute once = WIS).

Choices / simplifications:
- Tied tok_emb/head is masked in its *head* view: rows = vocab, inputs = the
  final-norm hidden state, S_vj = |W_vj| * ||h_j||. ``emb=dense`` leaves it
  unmasked instead. 1-D params (RMSNorm gains) stay dense (unmasked Gaussian).
- Calibration = ``cal`` fresh training sequences (paper: 128 downstream
  examples); its tokens and forward are counted in the step that refreshes.
- k directions per step averaged (paper uses k=1); constant lr, no momentum.
- ``mask=random`` is the paper's Random-MeZO control (same r, random support).
Crowd view: mask is a deterministic function of (W, public calibration batch),
so a worker uploads only (seed, c_k) = 12 bytes per direction.
"""

import torch
from crowd_v15.model import Perturb

from harness import Cost
from zo import gen, perturb, spsa_pair, trainable


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        self.k = int(cfg.get("k", 4))
        self.eps = float(cfg.get("eps", 1e-3))
        self.lr = float(cfg.get("lr", 1e-4))
        self.r = float(cfg.get("r", 0.1))
        self.tau = int(cfg.get("tau", 100))
        self.cal = int(cfg.get("cal", 32))
        self.emb = cfg.get("emb", "head")
        self.kind = cfg.get("mask", "wanda")
        self.params = trainable(model)
        self.masks, self.refreshes = None, 0

    @torch.no_grad()
    def compute_masks(self):
        idx, _ = self.data.next(batch=self.cal)
        st = Perturb(record={})
        self.model(idx, st)
        masks, g = {}, gen(self.refreshes, idx.device)
        self.refreshes += 1
        for name, p in self.params.items():
            if p.dim() != 2:
                continue
            key = "__head__" if name == "tok_emb.weight" else name[:-len(".weight")]
            if key == "__head__" and self.emb == "dense":
                continue
            x = st.record[key]
            q = x.reshape(-1, x.shape[-1]).float().norm(dim=0)          # ||X_j||, (in,)
            s = p.abs() * q.unsqueeze(0) if self.kind == "wanda" else torch.rand(p.shape, generator=g, device=p.device)
            keep = max(1, int(p.shape[1] * self.r))
            top = s.topk(keep, dim=1).indices
            m = torch.zeros_like(p)
            m.scatter_(1, top, 1.0 / self.r ** 0.5)                      # M / sqrt(r)
            masks[name] = m
        self.masks = masks
        return idx.numel()

    def _refresh(self, it):
        if self.masks is None or (self.tau > 0 and it % self.tau == 0 and it > 0):
            return self.compute_masks()
        return 0

    def estimate(self, idx, tgt, it):
        self._refresh(it)
        g = {n: torch.zeros_like(p) for n, p in self.params.items()}
        for k in range(self.k):
            seed = it * 1000 + k
            c, _, _ = spsa_pair(self.model, self.params, idx, tgt, seed, self.eps, self.masks)
            perturb(g, seed, c / self.k, self.masks)
        return g, 2 * self.k

    @torch.no_grad()
    def step(self, it):
        cal_tokens = self._refresh(it)
        idx, tgt = self.data.next()
        cs = []
        for k in range(self.k):
            seed = it * 1000 + k
            c, _, _ = spsa_pair(self.model, self.params, idx, tgt, seed, self.eps, self.masks)
            cs.append((seed, c))
        for seed, c in cs:
            perturb(self.params, seed, -self.lr * c / self.k, self.masks)
        n = idx.numel()
        return Cost(tokens=n + cal_tokens, fwd=2 * self.k * n + cal_tokens, up=12 * self.k,
                    extra={"projGrad": round(cs[0][1], 4)})
