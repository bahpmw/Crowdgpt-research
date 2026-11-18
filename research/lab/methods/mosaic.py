"""MOSAIC (round-2 hybrid): MOdule-Sharded local learning with a Sparse, Auditable,
Infrequent Communication outer loop.

Built from the round-1 survivors:
- SOLO (2609.35440): the GPT's blocks are cut into ``K`` modules; module k < K
  learns from a local loss through a small local head and a frozen copy
  ``W_bar`` of the output matrix, so no gradient crosses a module boundary.
- SparseLoCo / Covenant-72B (2603.08163): every worker runs ``H`` local AdamW
  steps, then uploads a chunk-Top-k + 2-bit pseudo-gradient with error feedback.
- Decoupled DiLoCo (2604.21428): workers are not lock-stepped. In ``het=budget``
  mode a round is a wall-clock budget: each worker does as many inner steps as
  its speed allows and updates are merged token-weighted, so a slow GPU never
  stalls the round. ``drop=p`` loses each upload with probability p (churn).
- Throttle (2609.39357) / Gauntlet: per module, uploads are clipped to the
  (lower) median upload norm (no peer can push harder than the typical one)
  and ``gauntlet=1`` keeps only uploads that lower the loss on a fresh batch.

What is new: the crowd is sharded by *module*, not by replica. Worker w owns
module k = w mod K. It runs the earlier modules forward-only (frozen at the
round's global weights, no peer-to-peer activation streaming, unlike SOLO's
pipeline), trains only module k (+ its local head) and uploads only module k's
compressed delta. W_bar is refreshed for free every round because the head is
broadcast with the weights anyway. Per worker this cuts backward compute and
optimizer memory to ~1/K and upload to ~1/K of SparseLoCo's, and the outer
loop is the same robust, churn-tolerant one. ``K=1`` is exactly SparseLoCo
(with the same robust outer loop): the built-in control.

Accounting: fwd/bwd use the block FLOP model of methods/solo.py (block
16d^2+2Td MACs/token, readout Vd, a frozen readout needs input-grad only).
up = delivered upload bytes; down = W x min(all uploads, bf16 weights).
simTime: sync rounds cost H/min(speed); budget rounds cost H/max(speed).
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from crowd import corrupt, norm
from harness import Cost, loss_fn
from methods.solo import LocalHead
from methods.sparseloco import CHUNK, compress
from zo import gen


class Net(nn.Module):
    def __init__(self, model, heads):
        super().__init__()
        self.model, self.heads = model, heads


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data, self.device = model, data, device
        model.head.weight = nn.Parameter(model.tok_emb.weight.detach().clone())
        self.K = int(cfg.get("K", 2))
        self.W = int(cfg.get("workers", 4))
        self.H = int(cfg.get("H", 30))
        self.k = int(cfg.get("topk", 64))
        self.beta = float(cfg.get("beta", 0.95))
        self.alpha = float(cfg.get("alpha", 1.0))
        self.outer = cfg.get("outer", "sgd")
        self.olr, self.mom = float(cfg.get("olr", 0.7)), float(cfg.get("mom", 0.9))
        self.clipmed = bool(cfg.get("clipmed", 1))
        self.gauntlet = bool(cfg.get("gauntlet", 0))
        self.vbatch = int(cfg.get("vbatch", 8))
        self.noniid = bool(cfg.get("noniid", 0))
        self.byz, self.attack = int(cfg.get("byz", 0)), cfg.get("attack", "flip")
        self.c = float(cfg.get("c", 5.0 if self.attack == "flip" else 1.0))
        self.drop = float(cfg.get("drop", 0.0))
        self.het = cfg.get("het", "sync")
        self.speeds = [float(s) for s in cfg.get("speeds", [1.0] * self.W)]
        depth = int(cfg.get("depth", 1))
        lr = float(cfg.get("lr", 3e-3))
        mc, L = model.cfg, len(model.blocks)
        self.split = [list(map(int, c)) for c in torch.arange(L).tensor_split(self.K)]
        heads = nn.ModuleList([LocalHead(mc, depth, private=False)
                               for _ in range(self.K - 1)]).to(device)
        self.net = Net(model, heads)
        self.G = dict(self.net.named_parameters())
        self.names = [self._module_names(k) for k in range(self.K)]
        self.wbar = model.head.weight.detach().clone()
        self.reps = [copy.deepcopy(self.net) for _ in range(self.W)]
        self.rp = [dict(r.named_parameters()) for r in self.reps]
        self.mod = [w % self.K for w in range(self.W)]
        self.opts = [torch.optim.AdamW([self.rp[w][n] for n in self.names[self.mod[w]]], lr=lr,
                                       betas=(0.9, 0.95), weight_decay=0.0)
                     for w in range(self.W)]
        self.ef = [{n: torch.zeros_like(self.G[n]) for n in self.names[self.mod[w]]
                    if self.G[n].dim() == 2} for w in range(self.W)] if self.k else None
        self.buf = {n: torch.zeros_like(p) for n, p in self.G.items()}
        self.msg_bytes = [self._bytes(k) for k in range(self.K)]
        self.n_params = sum(p.numel() for p in self.G.values())
        self.mem_frac = [sum(self.G[n].numel() for n in self.names[k]) / self.n_params
                         for k in range(self.K)]
        d, T, V = mc.d_model, data.seq, mc.vocab_size
        b, r = 16 * d * d + 2 * T * d, V * d
        full = L * b + r
        last = self.K - 1
        self.fwd_f = [((self.split[k][-1] + 1) * b + (depth * b + r if k < last else r)) / full
                      for k in range(self.K)]
        self.bwd_f = [(len(self.split[k]) * b + (depth * b + 0.5 * r if k < last else r)) / full
                      for k in range(self.K)]
        self.sim = 0.0
        self.rng = gen(int(cfg.get("seed", 0)) + 7, "cpu")

    def _module_names(self, k):
        pre = tuple(f"model.blocks.{i}." for i in self.split[k])
        keep = [n for n in self.G if n.startswith(pre)]
        keep += ["model.tok_emb.weight"] if k == 0 else []
        keep += ([n for n in self.G if n.startswith(f"heads.{k}.")] if k < self.K - 1
                 else ["model.norm_f.weight", "model.head.weight"])
        return keep

    def _bytes(self, k):
        n2 = sum(self.G[n].numel() for n in self.names[k] if self.G[n].dim() == 2)
        n1 = sum(self.G[n].numel() for n in self.names[k] if self.G[n].dim() != 2)
        if not self.k:
            return 4 * (n1 + n2)
        return n2 // (CHUNK * CHUNK) * (self.k * 14 + 32) / 8 + 2 * n1

    def _local_loss(self, net, k, idx, tgt):
        m = net.model
        T = idx.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)[None, None]
        with torch.set_grad_enabled(k == 0):
            h = m.tok_emb(idx)
            for i in range(self.split[k][0]):
                h = m.blocks[i](h, mask, None, f"blocks.{i}")
        for i in self.split[k]:
            h = m.blocks[i](h, mask, None, f"blocks.{i}")
        logits = (net.heads[k](h, mask, self.wbar) if k < self.K - 1
                  else m.norm_f(h) @ m.head.weight.t())
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))

    def _inner(self, w, steps):
        k, rep, opt = self.mod[w], self.reps[w], self.opts[w]
        params = [self.rp[w][n] for n in self.names[k]]
        loss = None
        for _ in range(steps):
            idx, tgt = self.data.next(worker=w, workers=self.W, noniid=self.noniid)
            loss = self._local_loss(rep, k, idx, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        return steps * idx.numel(), float(loss.detach())

    @torch.no_grad()
    def _message(self, w):
        msg = {}
        for n in self.names[self.mod[w]]:
            d = self.G[n] - self.rp[w][n]
            if self.k and d.dim() == 2:
                acc = self.ef[w][n].mul_(self.beta).add_(d)
                msg[n] = compress(acc, self.k)
                acc.sub_(msg[n])
            else:
                msg[n] = d.bfloat16().float() if self.k else d
        return msg

    def step(self, it):
        fast, slow = max(self.speeds), min(self.speeds)
        steps = ([max(1, round(self.H * s / fast)) for s in self.speeds] if self.het == "budget"
                 else [self.H] * self.W)
        self.wbar.copy_(self.G["model.head.weight"])
        tokens, fwd, bwd, up = 0, 0.0, 0.0, 0.0
        inner = [[] for _ in range(self.K)]
        arrived = [[] for _ in range(self.K)]
        for w in range(self.W):
            k = self.mod[w]
            with torch.no_grad():
                for n, p in self.rp[w].items():
                    p.copy_(self.G[n])
            t, loss = self._inner(w, steps[w])
            tokens += t
            fwd += t * self.fwd_f[k]
            bwd += t * self.bwd_f[k]
            inner[k].append(loss)
            msg = self._message(w)
            if w < self.byz:
                msg = corrupt(msg, self.attack, self.c, gen(it * 1000 + w, self.device))
            if self.drop and float(torch.rand((), generator=self.rng)) < self.drop:
                continue
            up += self.msg_bytes[k]
            arrived[k].append((t, msg))
        kept = 0
        with torch.no_grad():
            if self.gauntlet:
                idx, tgt = self.data.next(batch=self.vbatch)
                base = float(loss_fn(self.model, idx, tgt))
                for k in range(self.K):
                    good, tot = [], sum(t for t, _ in arrived[k])
                    for t, msg in arrived[k]:
                        # Score each upload at its share of the merge: a lone full-strength
                        # upload overshoots and honest peers would be dropped.
                        share = self.alpha * t / tot
                        for n, v in msg.items():
                            self.G[n].sub_(v, alpha=share)
                        if base - float(loss_fn(self.model, idx, tgt)) > 0:
                            good.append((t, msg))
                        for n, v in msg.items():
                            self.G[n].add_(v, alpha=share)
                        fwd += idx.numel()
                    arrived[k] = good
                fwd += idx.numel()
            norms = []
            for k in range(self.K):
                if not arrived[k]:
                    continue
                ns = [float(norm(m)) for _, m in arrived[k]]
                norms += ns
                med = sorted(ns)[(len(ns) - 1) // 2]
                scale = [min(1.0, med / (x + 1e-12)) if self.clipmed else 1.0 for x in ns]
                tot = sum(t for t, _ in arrived[k])
                kept += len(arrived[k])
                for n in self.names[k]:
                    avg = sum(s * t / tot * m[n] for s, (t, m) in zip(scale, arrived[k]))
                    if self.outer == "nesterov":
                        b = self.buf[n].mul_(self.mom).add_(avg)
                        self.G[n].sub_(avg + self.mom * b, alpha=self.olr)
                    else:
                        self.G[n].sub_(avg, alpha=self.alpha)
        self.sim += self.H / (fast if self.het == "budget" else slow)
        down = self.W * min(up, 2 * self.n_params)
        extra = {"simTime": round(self.sim, 2), "kept": kept,
                 "msgNorm": round(sum(norms) / max(len(norms), 1), 4),
                 "workerParamFrac": round(max(self.mem_frac), 3),
                 "upPerWorkerMB": round(up / max(kept, 1) / 1e6, 3)}
        extra |= {f"inner{k}": round(sum(v) / len(v), 4) for k, v in enumerate(inner) if v}
        return Cost(tokens=tokens, fwd=fwd, bwd=bwd, up=up, down=down, extra=extra)
