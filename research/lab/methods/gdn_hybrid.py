"""X5: Gated DeltaNet hybrid (Olmo Hybrid 2604.03444; GDN from Yang et al. 2412.06464), pure PyTorch.

``pattern`` has one letter per block: A keeps softmax attention, G swaps it for a GDN layer,
C is a control: the stock attention behind a causal depthwise conv(4) + SiLU on its input
(this GPT has no position encoding, so a short conv alone adds local order information).
``conv=0`` drops the GDN layer's own short conv (q, k, v then just pass through SiLU).
GDN layer (FLA GatedDeltaNet layout): q, k, v projections, causal depthwise conv (4) +
SiLU, L2-normalised q and k, value dim 2*dk, beta = sigmoid(W_b x) (x2 with ``neg=1``,
which allows negative eigenvalues as in Olmo Hybrid), decay
log alpha = -exp(A_log) * softplus(W_a x + dt_bias), per-head RMSNorm of the output gated by
SiLU(W_g x), output projection. With H*dk = d/2 its projections hold 8*d*H*dk = 4*d^2,
the same as the attention layer it replaces (plus ~3k conv/gate scalars).

Recurrence (row convention, S is dk x dv):
  S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t / sqrt(dk)
computed chunkwise (WY form, FLA's naive chunk reference). The in-chunk inverse
(I + M)^-1 of the strictly lower M uses torch.linalg.solve_triangular (works on MPS with
autograd). An earlier version used the nilpotent product (I - M)(I + M^2)(I + M^4)...;
it overflows when keys are correlated (NaN even in float64), so runs made with it are void.
This is a slow reference: wall-time numbers are not representative of a kernel.

Training is AdamW (as adam.py) with the moe_grow WSD schedule; ``tokens`` must equal
--tokens so the decay lands at the end.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from crowd_v15.model import GPT
from harness import Cost, loss_fn
from methods.moe_grow import wsd_lr


def chunk_gdn(q, k, v, g, beta, chunk):
    """q, k: (b, h, t, dk) L2-normalised; v: (b, h, t, dv); g = log alpha, beta: (b, h, t)."""
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    pad = (-t) % chunk
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = (t + pad) // chunk
    q = (q * dk ** -0.5).reshape(b, h, n, chunk, dk)
    k = k.reshape(b, h, n, chunk, dk)
    v = v.reshape(b, h, n, chunk, dv)
    beta = beta.reshape(b, h, n, chunk, 1)
    g = g.reshape(b, h, n, chunk).cumsum(-1)
    ones = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device)
    upper = ones.triu(1)
    decay = (g[..., :, None] - g[..., None, :]).masked_fill(upper, float("-inf")).exp()
    kb = k * beta
    m = (kb @ k.transpose(-1, -2) * decay).masked_fill(~ones.tril(-1), 0.0)
    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    inv = torch.linalg.solve_triangular(eye + m, eye.expand_as(m), upper=False, unitriangular=True)
    u = inv @ (v * beta)
    w = inv @ (kb * g.exp()[..., None])
    s = q.new_zeros(b, h, dk, dv)
    outs = []
    for i in range(n):
        qi, ki, gi = q[:, :, i], k[:, :, i], g[:, :, i]
        vnew = u[:, :, i] - w[:, :, i] @ s
        att = (qi @ ki.transpose(-1, -2)) * decay[:, :, i]
        outs.append((qi * gi.exp()[..., None]) @ s + att @ vnew)
        s = s * gi[..., -1, None, None].exp() + (
            ki * (gi[..., -1:] - gi).exp()[..., None]).transpose(-1, -2) @ vnew
    return torch.stack(outs, 2).reshape(b, h, n * chunk, dv)[:, :, :t]


class GDN(nn.Module):
    def __init__(self, d, heads, dk, neg, chunk, conv):
        super().__init__()
        self.h, self.dk, self.dv, self.neg, self.chunk = heads, dk, 2 * dk, neg, chunk
        self.q = nn.Linear(d, heads * dk, bias=False)
        self.k = nn.Linear(d, heads * dk, bias=False)
        self.v = nn.Linear(d, heads * self.dv, bias=False)
        self.g = nn.Linear(d, heads * self.dv, bias=False)
        self.o = nn.Linear(heads * self.dv, d, bias=False)
        self.a = nn.Linear(d, heads, bias=False)
        self.b = nn.Linear(d, heads, bias=False)
        self.apply(GPT._init_weights)
        self.convs = nn.ModuleList(nn.Conv1d(c, c, 4, groups=c, padding=3, bias=False)
                                   for c in (heads * dk, heads * dk, heads * self.dv)) if conv else None
        self.A_log = nn.Parameter(torch.empty(heads).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(heads) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.onorm = nn.Parameter(torch.ones(self.dv))

    def forward(self, x, mask=None, st=None, prefix=""):
        b, t, _ = x.shape
        q, k, v = (proj(x) for proj in (self.q, self.k, self.v))
        if self.convs is not None:
            q, k, v = (conv(z.transpose(1, 2))[..., :t].transpose(1, 2)
                       for conv, z in zip(self.convs, (q, k, v)))
        q, k, v = F.silu(q), F.silu(k), F.silu(v)
        q = F.normalize(q.view(b, t, self.h, self.dk), dim=-1).transpose(1, 2)
        k = F.normalize(k.view(b, t, self.h, self.dk), dim=-1).transpose(1, 2)
        v = v.view(b, t, self.h, self.dv).transpose(1, 2)
        beta = torch.sigmoid(self.b(x)).transpose(1, 2) * (2.0 if self.neg else 1.0)
        g = (-self.A_log.exp() * F.softplus(self.a(x) + self.dt_bias)).transpose(1, 2)
        o = chunk_gdn(q, k, v, g, beta, self.chunk)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5) * self.onorm
        o = o.transpose(1, 2).reshape(b, t, self.h * self.dv) * F.silu(self.g(x))
        return self.o(o)


class ConvAttention(nn.Module):
    """Control: the stock attention behind a causal depthwise conv(4), the GDN layer's short conv."""

    def __init__(self, attn, d):
        super().__init__()
        self.attn = attn
        self.conv = nn.Conv1d(d, d, 4, groups=d, padding=3, bias=False)

    def forward(self, x, mask, st=None, prefix=""):
        h = F.silu(self.conv(x.transpose(1, 2))[..., :x.shape[1]]).transpose(1, 2)
        return self.attn(h, mask, st, prefix)


class RopeAttention(nn.Module):
    """Control: the stock attention with rotary position embeddings (the v1.5 model has no
    position signal at all, so this separates 'any position signal' from the GDN layer itself)."""

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        hd = attn.head_dim
        self.register_buffer("inv", 10000.0 ** (-torch.arange(0, hd, 2).float() / hd), persistent=False)

    def forward(self, x, mask, st=None, prefix=""):
        a = self.attn
        b, t, _ = x.shape
        q, k, v = (z.view(b, t, a.n_heads, a.head_dim).transpose(1, 2)
                   for z in a.qkv(x).split(a.d, dim=-1))
        ang = torch.arange(t, device=x.device).float()[:, None] * self.inv
        cos, sin = ang.cos(), ang.sin()

        def rot(z):
            z1, z2 = z[..., 0::2], z[..., 1::2]
            return torch.stack((z1 * cos - z2 * sin, z1 * sin + z2 * cos), -1).flatten(-2)

        att = (rot(q) @ rot(k).transpose(-2, -1)) / math.sqrt(a.head_dim)
        y = torch.softmax(att.masked_fill(mask, float("-inf")), dim=-1) @ v
        return a.o(y.transpose(1, 2).reshape(b, t, a.d))


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        c = model.cfg
        pattern = str(cfg.get("pattern", "GGAGGA"))
        if len(pattern) != c.n_layers or set(pattern) - {"A", "C", "G", "R"}:
            raise ValueError(f"pattern {pattern!r} needs {c.n_layers} letters from A/C/G/R")
        heads = int(cfg.get("heads", 2))
        dk = int(cfg.get("dk", c.d_model // (2 * heads)))
        for blk, kind in zip(model.blocks, pattern):
            if kind == "G":
                blk.attn = GDN(c.d_model, heads, dk, int(cfg.get("neg", 0)),
                               int(cfg.get("chunk", 64)), int(cfg.get("conv", 1))).to(device)
            elif kind == "C":
                blk.attn = ConvAttention(blk.attn, c.d_model).to(device)
            elif kind == "R":
                blk.attn = RopeAttention(blk.attn).to(device)
        self.lr = float(cfg.get("lr", 3e-3))
        self.warm = int(cfg.get("warm", 100))
        self.decay = float(cfg.get("decay", 0.2))
        self.steps = int(cfg["tokens"]) // (data.batch * data.seq)
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.lr, betas=(0.9, 0.95),
                                     weight_decay=0.0)

    def step(self, it):
        lr = wsd_lr(self.lr, it, self.steps, self.warm, self.decay)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        idx, tgt = self.data.next()
        loss = loss_fn(self.model, idx, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        n = idx.numel()
        return Cost(tokens=n, fwd=n, bwd=n, extra={"train": float(loss.detach()), "lr": lr})
