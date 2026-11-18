"""SOLO (2609.35440): gradient-isolated modules with a shared stop-grad readout.

The 6 blocks are split into ``K`` contiguous modules (K=2: blocks 0-2, 3-5).
The head is untied in __init__ (``model.head.weight`` becomes its own
parameter, cloned from tok_emb), so eval's ``model.forward`` uses the untied
head. Module k < K: h_k = f_k(sg(h_{k-1})), local loss
CE(tau_k * phi_k(h_k) @ W_bar^T, y) where phi_k = ``depth`` x Block + RMSNorm,
tau_k a learned scalar (init 1) and W_bar a frozen copy of the head refreshed
every ``S`` steps. Module 1 owns tok_emb and updates it from its local loss
only; only the last module's true loss (norm_f + head) updates W. Each module
has its own AdamW and its own clip at 1.0. ``heads=private`` gives every local
head its own trainable readout (the paper's PRIV ablation). ``K=1`` is plain
backprop + AdamW on the same untied model (the fair BP control).

``stale=s`` simulates pipeline peers: each boundary is a FIFO, and module k
trains on the activation (and targets) module k-1 produced s steps earlier,
so module k's effective data is (k*s) steps old and its upstream weights are
s steps stale; downstream modules idle for the first s steps.

``cos_every=N`` (diagnostic, not counted in cost; only with stale=0): every N
steps compares each module's local gradient on its block weights with the
full-backprop gradient of the same untied model on the same batch.

Accounting: fwd/bwd = tokens x FLOP factor for the extra heads (block cost
16d^2+2Td MACs/token for this SwiGLU GPT, readout Vd; a shared readout needs
no weight gradient). up = activations crossing module boundaries,
(K-1)*B*T*d*2 bytes (bf16). down = the W_bar broadcast, (K-1)*V*d*2 bytes on
refresh steps (shared heads only).

Simplifications: paper uses 24-layer Transformer++ with 2-block heads, V/d >
26 (ours 16), and S=1; constant LR, no warmup/cosine; no head bias.
"""

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from crowd_v15.model import GPT, Block, RMSNorm
from harness import Cost


class LocalHead(nn.Module):
    def __init__(self, cfg, depth, private):
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(depth)])
        self.norm = RMSNorm(cfg.d_model)
        self.tau = nn.Parameter(torch.ones(()))
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) if private else None
        self.apply(GPT._init_weights)

    def forward(self, h, mask, wbar):
        for i, blk in enumerate(self.blocks):
            h = blk(h, mask, None, f"head.{i}")
        h = self.norm(h)
        return self.tau * (self.out(h) if self.out is not None else h @ wbar.t())


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        model.head.weight = nn.Parameter(model.tok_emb.weight.detach().clone())
        self.K, self.S = int(cfg.get("K", 2)), int(cfg.get("S", 20))
        depth = int(cfg.get("depth", 1))
        self.private = cfg.get("heads", "shared") == "private"
        self.stale = int(cfg.get("stale", 0))
        self.cos_every = int(cfg.get("cos_every", 0))
        lr = float(cfg.get("lr", 3e-3))
        mc, L = model.cfg, len(model.blocks)
        self.split = [list(map(int, c)) for c in torch.arange(L).tensor_split(self.K)]
        self.heads = nn.ModuleList([LocalHead(mc, depth, self.private)
                                    for _ in range(self.K - 1)]).to(device)
        self.wbar = model.head.weight.detach().clone()
        self.groups = []
        for k, ids in enumerate(self.split):
            ps = [p for i in ids for p in model.blocks[i].parameters()]
            ps += list(model.tok_emb.parameters()) if k == 0 else []
            ps += (list(self.heads[k].parameters()) if k < self.K - 1
                   else [model.norm_f.weight, model.head.weight])
            self.groups.append(ps)
        self.opts = [torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
                     for ps in self.groups]
        self.queues = [deque() for _ in range(self.K - 1)]
        self.cos = {}
        d, T, V = mc.d_model, data.seq, mc.vocab_size
        b, r, extra = 16 * d * d + 2 * T * d, V * d, (self.K - 1) * depth * (16 * d * d + 2 * T * d)
        full = L * b + r
        self.fwd_f = (full + extra + (self.K - 1) * r) / full
        self.bwd_f = (full + extra + (self.K - 1) * r * (1.0 if self.private else 0.5)) / full

    def _losses(self, idx, tgt, queue):
        m = self.model
        T = idx.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)[None, None]
        h, y, losses = m.tok_emb(idx), tgt, []
        for k, ids in enumerate(self.split):
            if k:
                if not queue:
                    h = h.detach()
                else:
                    self.queues[k - 1].append((h.detach(), y))
                    if len(self.queues[k - 1]) <= self.stale:
                        break
                    h, y = self.queues[k - 1].popleft()
            for i in ids:
                h = m.blocks[i](h, mask, None, f"blocks.{i}")
            logits = (self.heads[k](h, mask, self.wbar) if k < self.K - 1
                      else m.norm_f(h) @ m.head.weight.t())
            losses.append(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                          y.reshape(-1)))
        return losses

    def _block_keys(self, k):
        return [f"blocks.{i}.{n}" for i in self.split[k]
                for n, p in self.model.blocks[i].named_parameters() if p.dim() == 2]

    def estimate(self, idx, tgt, it):
        """Local (gradient-isolated) grads of every module, no update; forwards=1."""
        names = dict(self.model.named_parameters())
        with torch.enable_grad():
            losses = self._losses(idx, tgt, queue=False)
            ps = list(names.values())
            gs = torch.autograd.grad(sum(losses), ps, allow_unused=True)
        return {n: g for n, g in zip(names, gs) if g is not None}, 1

    def _bp_ref(self, idx, tgt):
        named = dict(self.model.named_parameters())
        keys = [k for k in named if k.startswith("blocks.") and named[k].dim() == 2]
        logits = self.model(idx)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return dict(zip(keys, torch.autograd.grad(loss, [named[k] for k in keys])))

    def step(self, it):
        m = self.model
        refresh = it % self.S == 0
        if refresh:
            self.wbar.copy_(m.head.weight.detach())
        idx, tgt = self.data.next()
        diag = self.cos_every and self.stale == 0 and it % self.cos_every == 0
        ref = self._bp_ref(idx, tgt) if diag else None
        for o in self.opts:
            o.zero_grad(set_to_none=True)
        losses = self._losses(idx, tgt, queue=True)
        sum(losses).backward()
        extra = {f"loss{k}": round(float(v), 4) for k, v in enumerate(losses)}
        extra |= {f"tau{k}": round(float(h.tau), 3) for k, h in enumerate(self.heads)}
        if diag:
            named = dict(m.named_parameters())
            for k in range(self.K):
                keys = self._block_keys(k)
                a = torch.cat([named[n].grad.flatten() for n in keys]).cpu().double()
                b = torch.cat([ref[n].flatten() for n in keys]).cpu().double()
                self.cos[f"cos{k}"] = round(float(a @ b / (a.norm() * b.norm() + 1e-30)), 4)
            self.cos["cosIt"] = it
        extra |= self.cos
        for ps, o in zip(self.groups[:len(losses)], self.opts):
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            o.step()
        n = idx.numel()
        d, V = m.cfg.d_model, m.cfg.vocab_size
        down = (self.K - 1) * V * d * 2 if refresh and not self.private else 0
        return Cost(tokens=n, fwd=n * self.fwd_f, bwd=n * self.bwd_f,
                    up=(self.K - 1) * n * d * 2, down=down, extra=extra)
