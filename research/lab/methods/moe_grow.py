"""X4: mixture-of-experts MLPs that grow mid-run by copying experts (Expert Upcycling, 2604.19835).

Every block's SwiGLU MLP is replaced in place by an MoE layer: E experts of hidden
``eh`` (default half the dense hidden, 512 on 10m) with top-``k`` (2) routing, so the
active MLP parameters equal the dense preset's. Router: softmax over E logits; the top-k
are picked on logits + a loss-free balancing bias (moved ``bal`` (1e-2) per step toward even
load, no auxiliary loss) and their probabilities are renormalised to sum to 1.

``grow="0.25:2,0.5:2"`` multiplies the expert count by m at that fraction of the step
budget (``tokens`` must equal --tokens). Growth copies every expert m times (uniform
copies; the paper's gradient-utility allocation is not implemented), copies router rows,
adds U(-delta, delta) to the copies' balancing biases, and resets AdamW (fresh state,
``rewarm`` warmup steps). ``offset`` > 0 also lowers every copy's bias by that much, so
the source experts keep winning top-k (growth is then exactly function-preserving) and
the balancer moves load onto the copies over ~offset/bal steps. ``E=1`` is a dense SwiGLU
of hidden ``eh`` (weight 1 always),
so ``E=1 grow=0.25:4`` is dense->MoE sparse upcycling. ``E=0`` keeps the stock dense MLP.

``doc=G`` is document routing: each sequence picks a group of G experts from the router
applied to the mean of its first 32 positions (causal: position t < 32 uses the mean of
0..t), and each token then routes top-k inside its group.

LR: linear warmup, constant, linear decay to 10% over the last ``decay`` fraction (WSD).
Cost counts tokens only. Extra: experts, active/total params, and per growth event i the
eval loss just before (g{i}pre), just after (g{i}post) and 20/100/300 steps later
(those evals run inside step() and are counted as training time; ~0.1 s each).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness import Cost, eval_loss, loss_fn

PREFIX = 32
CHECKS = (20, 100, 300)


def wsd_lr(lr, it, steps, warm, decay, since=0):
    """Warmup-stable-decay: warmup restarts at ``since`` (an optimizer reset), decay is global."""
    f = min(1.0, (it - since + 1) / warm) if warm else 1.0
    d0 = int(steps * (1 - decay))
    if decay and it >= d0:
        f *= 1 - 0.9 * min(1.0, (it - d0) / max(1, steps - d0))
    return lr * f


class MoE(nn.Module):
    def __init__(self, d, eh, n_exp, k, doc, bal):
        super().__init__()
        self.k, self.doc, self.bal = k, doc, bal
        self.router = nn.Parameter(torch.randn(n_exp, d) * 0.02)
        self.w_up = nn.Parameter(torch.randn(n_exp, d, eh) * 0.02)
        self.w_gate = nn.Parameter(torch.randn(n_exp, d, eh) * 0.02)
        self.w_down = nn.Parameter(torch.randn(n_exp, eh, d) * 0.02)
        self.register_buffer("bias", torch.zeros(n_exp))
        self.load = None

    @property
    def n_exp(self):
        return self.router.shape[0]

    def forward(self, x, st=None, prefix=""):
        b, t, d = x.shape
        k = min(self.k, self.n_exp)
        logits = x @ self.router.t()
        sel = logits.detach() + self.bias
        if self.doc and self.n_exp > self.doc:
            n = min(t, PREFIX)
            pre = x[:, :n].detach().cumsum(1) / torch.arange(1, n + 1, device=x.device)[:, None]
            doc_in = torch.cat([pre, pre[:, -1:].expand(b, t - n, d)], 1)
            grp = (doc_in @ self.router.detach().t() + self.bias).topk(self.doc, -1).indices
            allowed = torch.zeros_like(sel, dtype=torch.bool).scatter_(-1, grp, True)
            sel = sel.masked_fill(~allowed, float("-inf"))
        top = sel.topk(k, -1).indices
        probs = logits.softmax(-1).gather(-1, top)
        gates = torch.zeros_like(logits).scatter(-1, top, probs / probs.sum(-1, keepdim=True))
        if self.training:
            self.load = torch.bincount(top.reshape(-1), minlength=self.n_exp)
        # Dense dispatch: every expert runs on every token and the gates (zero off the top-k)
        # pick the outputs. Same values and gradients as sparse dispatch; much faster on MPS
        # at E <= 8 than per-expert gathers, but wall time no longer reflects sparse cost.
        h = F.silu(torch.einsum("btd,edh->ebth", x, self.w_gate)) * torch.einsum(
            "btd,edh->ebth", x, self.w_up)
        return torch.einsum("bte,ebtd->btd", gates, h @ self.w_down[:, None])

    @torch.no_grad()
    def grow(self, m, delta, offset=0.0):
        n = self.n_exp
        for name in ("router", "w_up", "w_gate", "w_down"):
            setattr(self, name, nn.Parameter(getattr(self, name).data.repeat_interleave(m, 0)))
        noise = (torch.rand(n * m, device=self.bias.device) * 2 - 1) * delta - offset
        noise[::m] = 0  # the first copy keeps the source expert's bias exactly
        self.bias = self.bias.repeat_interleave(m) + noise


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        c = model.cfg
        n_exp = int(cfg.get("E", 8))
        self.eh = int(cfg.get("eh", c.hidden // 2))
        self.k = int(cfg.get("k", 2))
        self.lr = float(cfg.get("lr", 3e-3))
        self.warm, self.rewarm = int(cfg.get("warm", 100)), int(cfg.get("rewarm", 50))
        self.decay = float(cfg.get("decay", 0.2))
        self.delta = float(cfg.get("delta", 1e-3))
        self.offset = float(cfg.get("offset", 0.0))
        self.steps = int(cfg["tokens"]) // (data.batch * data.seq)
        self.grows = sorted((int(float(f) * self.steps), int(m)) for f, m in
                            (g.split(":") for g in str(cfg.get("grow", "")).split(",") if g))
        if n_exp:
            for blk in model.blocks:
                blk.mlp = MoE(c.d_model, self.eh, n_exp, self.k, int(cfg.get("doc", 0)),
                              float(cfg.get("bal", 1e-2))).to(device)
        self.moes = [b.mlp for b in model.blocks if isinstance(b.mlp, MoE)]
        self.events, self.last_reset = [], 0
        self.info = {}
        self._reset_opt()

    def _reset_opt(self):
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95),
                                     weight_decay=0.0)
        total = self.model.n_params()
        idle = sum((m.n_exp - min(self.k, m.n_exp)) * 3 * m.w_up.shape[1] * self.eh
                   for m in self.moes)
        self.info.update(experts=self.moes[0].n_exp if self.moes else 0, totalParams=total,
                         activeParams=total - idle)

    def step(self, it):
        while self.grows and it == self.grows[0][0]:
            _, m = self.grows.pop(0)
            i = len(self.events) + 1
            self.info[f"g{i}pre"] = eval_loss(self.model, self.data)
            for moe in self.moes:
                moe.grow(m, self.delta, self.offset)
            self.info[f"g{i}post"] = eval_loss(self.model, self.data)
            self.events.append(it)
            self.last_reset = it
            self._reset_opt()
            print(f"grow x{m} at step {it}: eval {self.info[f'g{i}pre']:.4f} -> "
                  f"{self.info[f'g{i}post']:.4f}, experts {self.info['experts']}", flush=True)
        warm = self.rewarm if self.last_reset else self.warm
        lr = wsd_lr(self.lr, it, self.steps, warm, self.decay, self.last_reset)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        idx, tgt = self.data.next()
        loss = loss_fn(self.model, idx, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        with torch.no_grad():
            for moe in self.moes:
                if moe.n_exp > 1:
                    load = moe.load.float()
                    moe.bias += moe.bal * torch.sign(load.mean() - load)
        for i, at in enumerate(self.events, 1):
            if it - at in CHECKS:
                self.info[f"g{i}post{it - at}"] = eval_loss(self.model, self.data)
        if self.moes and self.moes[0].n_exp > 1:
            load = torch.stack([m.load.float() for m in self.moes])
            self.info["maxLoad"] = float((load.max(1).values / load.mean(1)).mean())
        n = idx.numel()
        return Cost(tokens=n, fwd=n, bwd=n,
                    extra=dict(self.info, train=float(loss.detach()), lr=lr))
