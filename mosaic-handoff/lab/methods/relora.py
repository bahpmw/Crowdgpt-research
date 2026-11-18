"""ReLoRA (Lialin et al. 2023, arXiv 2307.05695): high-rank training from repeated
low-rank updates.

Faithful: a full-rank warm start (first ``warm`` steps, default 10% of the run),
then every block linear (qkv, o, up, gate, down) gets a LoRA adapter
W = W0 + scale * B @ A with W0 frozen; every ``every`` steps the product is merged
into W0, A is re-drawn, B is zeroed, the adapters' AdamW state is pruned by
magnitude (``prune`` = 0.99 of entries zeroed) and the LR restarts with a short
linear re-warmup (``rwarm`` steps) under one global cosine ("jagged" schedule).
Embedding (tied head) and RMSNorm gains stay full-rank and trainable, as in the
paper.

Simplified: one worker, no 8-bit optimiser; LoRA is a torch parametrization, so
autograd still forms the dense dW and backward compute equals full rank (an
efficient kernel would skip it); A ~ N(0, 1/n_in), B = 0, scale = alpha/r with
alpha = 32 (s = 1 at r32; s = 1 at r128 was unstable here); no dropout.
``rank=0`` is the full-rank control with the same schedule (warmup ``wu``
steps, cosine to ``minlr`` x lr).

Accounting: up = 4 bytes x trainable parameters per step (a synchronous dense fp32
all-reduce of what is being trained); down = 0.
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from harness import Cost, loss_fn
from zo import gen

LINEARS = ("attn.qkv", "attn.o", "mlp.up", "mlp.gate", "mlp.down")


class LoRA(nn.Module):
    """Parametrization W = W0 + scale * B @ A; W0 lives frozen in ``original``."""

    def __init__(self, weight, rank, scale):
        super().__init__()
        m, n = weight.shape
        self.scale = scale
        self.A = nn.Parameter(weight.new_zeros(rank, n))
        self.B = nn.Parameter(weight.new_zeros(m, rank))

    def forward(self, w0):
        return w0 + self.scale * (self.B @ self.A)

    @torch.no_grad()
    def reset(self, seed):
        a = torch.randn(self.A.shape, generator=gen(seed, "cpu")) / math.sqrt(self.A.shape[1])
        self.A.copy_(a)
        self.B.zero_()


def add_lora(blocks, rank, scale):
    """Adapters on every LINEARS weight of each (prefix, Block); returns
    [(prefix.lin.weight, frozen W0, LoRA)] in a fixed order."""
    out = []
    for prefix, blk in blocks:
        for name in LINEARS:
            lin = blk.get_submodule(name)
            p = LoRA(lin.weight.detach(), rank, scale).to(lin.weight.device)
            parametrize.register_parametrization(lin, "weight", p)
            w0 = lin.parametrizations.weight.original
            w0.requires_grad_(False)
            out.append((f"{prefix}.{name}.weight", w0, p))
    return out


@torch.no_grad()
def prune_state(opt, params, frac):
    """Zero the smallest ``frac`` of each tensor's AdamW moments (mask from |m|)."""
    for p in params:
        st = opt.state.get(p)
        if not st or "exp_avg" not in st or frac <= 0:
            continue
        m = st["exp_avg"]
        if frac >= 1:
            m.zero_()
            st["exp_avg_sq"].zero_()
            continue
        keep = max(1, int(m.numel() * (1 - frac)))
        thr = m.abs().flatten().kthvalue(m.numel() - keep + 1).values
        mask = m.abs() >= thr
        m.mul_(mask)
        st["exp_avg_sq"].mul_(mask)


class Method:
    def __init__(self, model, data, cfg, device):
        self.model = model
        self.data = data
        self.rank = int(cfg.get("rank", 32))
        self.scale = float(cfg.get("scale", 32 / max(self.rank, 1)))
        self.lr = float(cfg.get("lr", 3e-3))
        self.total = int(cfg.get("total", 0)) or round(
            float(cfg.get("tokens", 1e7)) / (data.seq * data.batch))
        self.warm = int(cfg.get("warm", round(0.1 * self.total)))
        self.every = int(cfg.get("every", 500))
        self.rwarm = int(cfg.get("rwarm", 50))
        self.wu = int(cfg.get("wu", 100))
        self.minlr = float(cfg.get("minlr", 0.1))
        self.prune = float(cfg.get("prune", 0.99))
        self.lin = [p for n, p in model.named_parameters()
                    if n.startswith("blocks.") and any(f".{x}." in n for x in LINEARS)]
        lin_ids = {id(p) for p in self.lin}
        self.rest = [p for p in model.parameters() if id(p) not in lin_ids]
        self.opt_rest = torch.optim.AdamW(self.rest, lr=self.lr, betas=(0.9, 0.95),
                                          weight_decay=0.0)
        self.opt_lin = torch.optim.AdamW(self.lin, lr=self.lr, betas=(0.9, 0.95),
                                         weight_decay=0.0)
        self.adapters = []
        self.n_all = model.n_params()

    def _lr(self, it):
        cos = self.minlr + (1 - self.minlr) * 0.5 * (1 + math.cos(math.pi * min(it / self.total, 1)))
        ramp = min(1.0, (it + 1) / self.wu)
        if self.adapters:
            ramp = min(ramp, ((it - self.warm) % self.every + 1) / self.rwarm)
        return self.lr * cos * ramp

    def step(self, it):
        if self.rank and it == self.warm:
            self.adapters = add_lora([(f"blocks.{i}", b) for i, b in enumerate(self.model.blocks)],
                                     self.rank, self.scale)
            for j, (_, _, p) in enumerate(self.adapters):
                p.reset(it * 1009 + j)
            self.lin = [t for _, _, p in self.adapters for t in (p.A, p.B)]
            self.opt_lin = torch.optim.AdamW(self.lin, lr=self.lr, betas=(0.9, 0.95),
                                             weight_decay=0.0)
        elif self.adapters and (it - self.warm) % self.every == 0:
            with torch.no_grad():
                for j, (_, w0, p) in enumerate(self.adapters):
                    w0.add_(p.B @ p.A, alpha=p.scale)
                    p.reset(it * 1009 + j)
            prune_state(self.opt_lin, self.lin, self.prune)
        lr = self._lr(it)
        for o in (self.opt_rest, self.opt_lin):
            for g in o.param_groups:
                g["lr"] = lr
            o.zero_grad(set_to_none=True)
        idx, tgt = self.data.next()
        loss = loss_fn(self.model, idx, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rest + self.lin, 1.0)
        self.opt_rest.step()
        self.opt_lin.step()
        n = idx.numel()
        trainable = sum(p.numel() for p in self.rest + self.lin)
        return Cost(tokens=n, fwd=n, bwd=n, up=4 * trainable,
                    extra={"lr": round(lr, 6), "trainFrac": round(trainable / self.n_all, 4),
                           "lora": int(bool(self.adapters))})
