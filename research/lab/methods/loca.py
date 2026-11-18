"""LoCA (2608.03020) adapted to pretraining: calibrated low-rank feedback + local backward.

Every ``calib`` steps one backprop probe (``probe_batches`` batches) records the
top error e = W_u^T(softmax - onehot) at the final hidden (the norm_f output,
i.e. dL/dh per token for the summed loss) and the exact dL/dx_l at every block
output, then fits per block F_l = G E^T (E E^T + beta I)^-1 (beta =
``beta`` x tr(E E^T)/d) truncated by SVD to rank ``rank``, stored as U_l V_l^T.
Before refitting, the cosine of the old F_l e to the true dL/dx_l is logged
(``cosPre*``: how far alignment decayed over ``calib`` steps); ``cosFit`` is the
in-probe cosine of the new fit.

Every step (including calibration steps, with the fresh F) is local: block l
gets input x_l.detach(), receives the synthetic output gradient F_l e / N (e
from that step's own forward) and backprops through itself only. The head and
norm_f get exact gradients from the loss; the tied embedding gets the head
gradient plus block 0's local input gradient. AdamW on everything, global clip
1.0. ``calib=1`` refits on every step's own batch (in-probe upper bound).
``exact_last`` gives the last block the exact gradient through norm_f (it can
share the head's peer) instead of F e.

Crowd accounting (one block per peer, pipeline view): each step every block
ships its output forward (L * B*T*d bf16) and the head peer sends each block
its rank-r feedback code V_l^T e (L * B*T*r bf16), no backward chain and no
waiting on deeper blocks. Calibration steps add one real backward chain
(L * B*T*d) and the E broadcast for the per-block fit (L * B*T*d), per probe
batch. ``bwd`` counts the local backwards as one full backward.

Simplifications: pretraining from scratch, which the paper does not test
(it fine-tunes a frozen backbone with ridge-fitted adapters, no optimizer);
calibration batch = the step's training batch; feedback ignores cross-token
credit (a tokenwise linear map, as in the paper).
"""

import torch
import torch.nn.functional as F

from harness import Cost


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        self.calib = int(cfg.get("calib", 50))
        self.rank = int(cfg.get("rank", 8))
        self.beta = float(cfg.get("beta", 1e-3))
        self.probes = int(cfg.get("probe_batches", 1))
        self.exact_last = bool(cfg.get("exact_last", False))
        self.opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-3)),
                                     betas=(0.9, 0.95), weight_decay=0.0)
        self.L, self.d = len(model.blocks), model.cfg.d_model
        self.U = self.V = None          # (L,d,r): F_l = U_l V_l^T
        self.extra = {}

    def _blocks(self, idx, detach):
        T = idx.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)[None, None]
        x, outs = self.model.tok_emb(idx), []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x.detach() if detach and i else x, mask, None, f"blocks.{i}")
            outs.append(x)
        return outs

    def _calibrate(self, it):
        m, d = self.model, self.d
        EtE = torch.zeros(d, d, dtype=torch.float64)
        EtG = torch.zeros(self.L, d, d, dtype=torch.float64)
        pre, idx0 = [], None
        for _ in range(self.probes):
            idx, tgt = self.data.next()
            idx0 = (idx, tgt) if idx0 is None else idx0
            outs = self._blocks(idx, detach=False)
            h = m.norm_f(outs[-1])
            logits = h @ m.head.weight.t()
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                   reduction="sum")
            gs = torch.autograd.grad(loss, outs + [h])
            E = gs[-1].reshape(-1, d).cpu().double()
            G = torch.stack([g.reshape(-1, d) for g in gs[:-1]]).cpu().double()   # (L,N,d)
            if self.U is not None:
                pre.append(self._cos(E, G))
            EtE += E.t() @ E
            EtG += torch.einsum("nd,lne->lde", E, G)
        A = EtE + self.beta * EtE.trace() / d * torch.eye(d, dtype=torch.float64)
        Ft = torch.linalg.solve(A, EtG)                     # (L,d,d) = F_l^T
        U, S, Vh = torch.linalg.svd(Ft.transpose(1, 2))     # F_l = U S Vh
        r = self.rank
        dev = m.tok_emb.weight.device
        self.U = (U[:, :, :r] * S[:, None, :r]).float().to(dev)
        self.V = Vh[:, :r, :].transpose(1, 2).float().to(dev)
        fit = self._cos(E, G)
        ex = {"calibIt": it, "cosFit": round(sum(fit) / self.L, 4)}
        if pre:
            mean = [sum(c[i] for c in pre) / len(pre) for i in range(self.L)]
            ex["cosPre"] = round(sum(mean) / self.L, 4)
            ex.update({f"cosPre{i}": round(c, 4) for i, c in enumerate(mean)})
        self.extra = ex
        return idx0

    def _cos(self, E, G):
        U, V = self.U.cpu().double(), self.V.cpu().double()
        Q = torch.einsum("nd,ldr,ler->lne", E, V, U)
        return [float((q * g).sum() / (q.norm() * g.norm() + 1e-30)) for q, g in zip(Q, G)]

    def step(self, it):
        m = self.model
        calib = it % self.calib == 0
        idx, tgt = self._calibrate(it) if calib else self.data.next()
        self.opt.zero_grad(set_to_none=True)
        outs = self._blocks(idx, detach=True)
        last = outs[-1] if self.exact_last else outs[-1].detach()
        logits = m.norm_f(last) @ m.head.weight.t()
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))
        with torch.no_grad():
            p = torch.softmax(logits.float(), -1).reshape(-1, logits.size(-1))
            p[torch.arange(p.shape[0], device=p.device), tgt.reshape(-1)] -= 1.0
            e = p @ m.head.weight.float()                                   # (N,d)
            code = torch.einsum("nd,ldr->lnr", e, self.V)                   # V_l^T e
            syn = torch.einsum("lnr,ldr->lnd", code, self.U) / e.shape[0]   # F_l e / N
        nb = self.L - 1 if self.exact_last else self.L
        torch.autograd.backward([loss] + outs[:nb],
                                [None] + [syn[i].view_as(outs[i]) for i in range(nb)])
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        self.opt.step()
        n = idx.numel()
        act = self.L * n * self.d * 2
        up = act + self.L * n * self.rank * 2
        extra_fb = self.probes * n if calib else 0
        if calib:
            up += self.probes * 2 * act
        return Cost(tokens=n + (self.probes - 1) * n * calib, fwd=n + extra_fb, bwd=n + extra_fb,
                    up=up, extra={"loss": round(float(loss), 4), **self.extra})
