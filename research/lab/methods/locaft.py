"""LoCA faithful variant (2608.03020): frozen backbone + ridge-fitted residual adapters.

Run from a trained checkpoint (``--init ckpt/adam30.pt``): the "task" is a
small shift, continued LM on fresh tokens. Every block gets a parallel adapter
h_l = block_l(h_{l-1}) + B_l A_l h_{l-1} (A_l ~ N(0, 1/d) fixed, r x d; B_l = 0
at start), installed as a forward hook so ``model.forward`` (and eval) uses it.
Step 0 runs the one-shot calibration (``probe_batches`` backprop probes, F_l
ridge-fit to rank ``k`` as in methods/loca.py). Each step is one Jacobi outer
iteration over ``batches`` forward-only mini-batches:
  q_l = F_l e,  eta_l = eta * RMS(H_{l-1}) / RMS(Q_l),
  rho_l = B_l p_l - eta_l q_l  with p_l = A_l h_{l-1},
  G_l += P^T P,  C_l += rho^T P,   then  B_l = C_l (G_l + lam tr(G_l)/r I)^-1.
The new B is kept only if CE on a fixed held-out *training* batch improves
(greedy version of the paper's best-snapshot selection; the harness's eval
batch is never used for selection). ``solver=adam`` instead trains the same
adapters' B with backprop + AdamW (the paper's LoRA-style control).

Crowd view: G_l and C_l are additive across workers, so a worker uploads
L * (r^2 + d r) fp32 numbers per outer iteration, independent of tokens.
Simplifications: no outer-iteration cap (the harness runs on time), Jacobi
schedule only, eta/lam/r fixed by config instead of selected.
"""

import torch

from harness import Cost, loss_fn
from methods.loca import Method as Calib


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        self.r, self.eta = int(cfg.get("r", 32)), float(cfg.get("eta", 0.01))
        self.lam, self.nb = float(cfg.get("lam", 1e-3)), int(cfg.get("batches", 8))
        self.solver = cfg.get("solver", "ridge")
        L, d = len(model.blocks), model.cfg.d_model
        for p in model.parameters():
            p.requires_grad_(False)
        g = torch.Generator(device="cpu").manual_seed(0)
        self.A = (torch.randn(L, self.r, d, generator=g) / d ** 0.5).to(device)
        self.B = torch.zeros(L, d, self.r, device=device, requires_grad=self.solver == "adam")
        self.rec = {}
        for i, blk in enumerate(model.blocks):
            blk.register_forward_hook(self._hook(i))
        self.calib = Calib(model, data, {"rank": int(cfg.get("k", 8)),
                                         "probe_batches": int(cfg.get("probe_batches", 4))},
                           device) if self.solver == "ridge" else None
        if self.solver == "adam":
            self.opt = torch.optim.AdamW([self.B], lr=float(cfg.get("lr", 1e-3)), weight_decay=0.0)
        self.sel = data.next(batch=16)
        self.best = float(loss_fn(model, *self.sel))
        self.L, self.d = L, d

    def _hook(self, i):
        def hook(_mod, inp, out):
            p = inp[0] @ self.A[i].t()
            self.rec[i] = (inp[0].detach(), p.detach())
            return out + p @ self.B[i].t()
        return hook

    def _ridge_step(self, it):
        m, r, L, d = self.model, self.r, self.L, self.d
        if it == 0:
            with torch.enable_grad():
                for p in m.parameters():
                    p.requires_grad_(True)
                self.calib._calibrate(0)
                for p in m.parameters():
                    p.requires_grad_(False)
        G = torch.zeros(L, r, r, device=self.B.device)
        C = torch.zeros(L, d, r, device=self.B.device)
        for _ in range(self.nb):
            idx, tgt = self.data.next()
            logits = m(idx)
            pr = torch.softmax(logits.float(), -1).reshape(-1, logits.size(-1))
            pr[torch.arange(pr.shape[0], device=pr.device), tgt.reshape(-1)] -= 1.0
            e = pr @ m.head.weight.float()
            for i in range(L):
                h, p = (t.reshape(-1, t.shape[-1]) for t in self.rec[i])
                q = (e @ self.calib.V[i]) @ self.calib.U[i].t()
                qn = q.pow(2).mean().sqrt()
                eta = self.eta * h.pow(2).mean().sqrt() / qn if qn > 0 else 0.0
                rho = p @ self.B[i].t() - eta * q
                G[i] += p.t() @ p
                C[i] += rho.t() @ p
        lam = self.lam * G.diagonal(dim1=1, dim2=2).sum(-1) / r
        A = G + lam[:, None, None] * torch.eye(r, device=G.device)
        old = self.B.clone()
        self.B.copy_(torch.linalg.solve(A.cpu(), C.transpose(1, 2).cpu()).transpose(1, 2).to(G.device))
        ce = float(loss_fn(m, *self.sel))
        if ce < self.best:
            self.best = ce
        else:
            self.B.copy_(old)
        return ce

    def step(self, it):
        n = self.data.batch * self.data.seq
        if self.solver == "adam":
            idx, tgt = self.data.next()
            self.opt.zero_grad(set_to_none=True)
            loss = loss_fn(self.model, idx, tgt)
            loss.backward()
            self.opt.step()
            return Cost(tokens=n, fwd=n, bwd=n, up=4 * self.B.numel(),
                        extra={"loss": round(float(loss), 4)})
        with torch.no_grad():
            ce = self._ridge_step(it)
        probe = self.calib.probes * n if it == 0 else 0
        return Cost(tokens=self.nb * n + probe, fwd=self.nb * n + probe, bwd=probe,
                    up=4 * self.L * (self.r ** 2 + self.d * self.r),
                    extra={"selCE": round(ce, 4), "bestSelCE": round(self.best, 4),
                           "Bnorm": round(float(self.B.norm()), 4)})
