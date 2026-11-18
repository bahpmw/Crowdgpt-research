"""X3 sync shapes for CrowdGPT v2: flat averaging vs groups-of-16 + sparse inter-group
gossip vs pure 2-peer gossip, plus merge rules, work units and the inner beta2 rule.

Every replica runs H local AdamW steps (lr 3e-3, betas (0.9, beta2), clip 1) from its
node's weights. Nodes are:
  topo=flat    one node; all W replicas merge (SparseLoCo outer SGD, alpha=1).
  topo=hier    W/group nodes ("groups", default 16 replicas). Inside a group the
               replicas send SparseLoCo messages (chunk top-k of 64x64 blocks, 2-bit,
               error feedback beta 0.95) and the group applies the merged update,
               exactly as flat does. Between groups: sparse value gossip (below).
  topo=gossip  every replica is its own node and keeps its full local update; nodes
               only mix through sparse value gossip with ``partners`` peers.

Sparse value gossip (SPICE 2611.23053 style, form assumed): each node keeps an
accumulator a = 0.95 a + (its own change this round); it picks the chunk top-``gk``
coordinates of |a| in every 64x64 block, sends its *current values* there (bf16 +
12-bit index) and zeroes a on them. A receiver averages those coordinates 50/50 with
its own values. 1-D tensors are averaged dense. Partners come from a seeded "beacon"
random perfect matching that changes every round; ``partners=2`` uses two matchings.

Merge rules (per node): ``merge=mean`` equal (or token) weighted mean; ``merge=mdir``
clips every message to the lower-median message norm, takes the (weighted) mean, then
rescales the non-embedding part of that mean to the median norm (Decoupled DiLoCo
App. D.2 RDA with Covenant's median; the tied embedding keeps the clipped mean).

Adversaries (global worker index w): ``byz=8`` makes every 8th worker send -c*msg
(c=5); ``lazy=4`` makes every 4th worker (w % 4 == 1) send 0.01*msg.

Heterogeneous speed (flat only): ``slow=n`` workers with w % 4 == 3 (first n of them)
run at 1/``slowx``. ``het=sync`` everyone does H, the round waits for the slowest;
``het=budget`` steps = round(H*speed); ``het=units`` work is cut into equal-token
units of ``unit`` steps, each started from the round's weights (fast cards take more
units). ``weight=equal|tokens``.

Accounting: Cost.up/down are summed over replicas. Object-store model: a replica
uploads its message once and downloads every other message of its node. A group's
gossip message is uploaded once (relay) and downloaded by every member of the
partner group. extra upKB/downKB are per replica per round.
"""

import copy

import torch

from crowd import assign, corrupt, local_adamw, norm
from harness import Cost, eval_loss
from methods.sparseloco import CHUNK, compress
from zo import gen, trainable

EMB = "tok_emb.weight"


def chunk_mask(acc: torch.Tensor, k: int) -> torch.Tensor:
    r, c = acc.shape
    blocks = acc.reshape(r // CHUNK, CHUNK, c // CHUNK, CHUNK).permute(0, 2, 1, 3)
    blocks = blocks.reshape(-1, CHUNK * CHUNK)
    idx = blocks.abs().topk(k, dim=1).indices
    m = torch.zeros(blocks.shape, dtype=torch.bool, device=acc.device).scatter_(1, idx, True)
    return m.reshape(r // CHUNK, c // CHUNK, CHUNK, CHUNK).permute(0, 2, 1, 3).reshape(r, c)


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data, self.device = model, data, device
        self.W = int(cfg.get("workers", 16))
        self.H = int(cfg.get("H", 30))
        self.k = int(cfg.get("topk", 64))
        self.gk = int(cfg.get("gk", 64))
        self.beta = float(cfg.get("beta", 0.95))
        self.alpha = float(cfg.get("alpha", 1.0))
        self.topo = cfg.get("topo", "flat")
        self.partners = int(cfg.get("partners", 1 if self.topo != "flat" else 0))
        self.merge = cfg.get("merge", "mean")
        self.byz, self.lazy = int(cfg.get("byz", 0)), int(cfg.get("lazy", 0))
        self.c = float(cfg.get("c", 5.0))
        self.het = cfg.get("het", "sync")
        self.weight = cfg.get("weight", "equal")
        self.unit = int(cfg.get("unit", 8))
        slow, slowx = int(cfg.get("slow", 0)), float(cfg.get("slowx", 4.0))
        slow_ids = [w for w in range(self.W) if w % 4 == 3][:slow]
        self.speeds = [1.0 / slowx if w in slow_ids else 1.0 for w in range(self.W)]
        self.seed = int(cfg.get("seed", 0))
        lr, b2 = float(cfg.get("lr", 3e-3)), float(cfg.get("beta2", 0.95))
        self.G = trainable(model)
        self.reps = [copy.deepcopy(model) for _ in range(self.W)]
        self.rp = [trainable(r) for r in self.reps]
        self.opts = [torch.optim.AdamW(r.parameters(), lr=lr, betas=(0.9, b2), weight_decay=0.0)
                     for r in self.reps]
        self.ef = [{n: torch.zeros_like(p) for n, p in self.G.items() if p.dim() == 2}
                   for _ in range(self.W)]
        if self.topo == "flat":
            self.groups, self.X = [list(range(self.W))], [self.G]
        elif self.topo == "hier":
            gs = int(cfg.get("group", 16))
            self.groups = [list(range(i, i + gs)) for i in range(0, self.W, gs)]
            self.X = [{n: p.detach().clone() for n, p in self.G.items()} for _ in self.groups]
        elif self.topo == "gossip":
            self.groups, self.X = [[w] for w in range(self.W)], self.rp
        else:
            raise ValueError(f"unknown topo {self.topo!r}")
        self.N = len(self.groups)
        self.partners = min(self.partners, self.N - 1)
        self.acc = [{n: torch.zeros_like(p) for n, p in self.G.items() if p.dim() == 2}
                    for _ in range(self.N)] if self.partners else None
        n2 = sum(p.numel() for p in self.G.values() if p.dim() == 2)
        n1 = sum(p.numel() for p in self.G.values() if p.dim() == 1)
        chunks = n2 // (CHUNK * CHUNK)
        self.msg_bytes = chunks * (self.k * 14 + 32) / 8 + 2 * n1
        self.gmsg_bytes = chunks * self.gk * (12 + 16) / 8 + 2 * n1
        self.sim, self.round = 0.0, 0

    @torch.no_grad()
    def _message(self, w: int, start: dict) -> dict:
        msg = {}
        for n in self.G:
            d = start[n] - self.rp[w][n]
            if d.dim() == 2:
                acc = self.ef[w][n].mul_(self.beta).add_(d)
                msg[n] = compress(acc, self.k)
                acc.sub_(msg[n])
            else:
                msg[n] = d.bfloat16().float()
        if self.byz and w % self.byz == 0:
            msg = corrupt(msg, "flip", self.c, gen(self.round * 1000 + w, self.device))
        elif self.lazy and w % self.lazy == 1:
            msg = {n: 0.01 * v for n, v in msg.items()}
        return msg

    @torch.no_grad()
    def _merge(self, msgs: list, weights: list) -> tuple[dict, list]:
        ws = sum(weights)
        norms = [float(norm(m)) for m in msgs]
        if self.merge == "mean":
            return {n: sum(w * m[n] for w, m in zip(weights, msgs)) / ws for n in self.G}, norms
        if self.merge != "mdir":
            raise ValueError(f"unknown merge {self.merge!r}")
        med = sorted(norms)[(len(norms) - 1) // 2]
        sc = [w * min(1.0, med / (x + 1e-12)) for w, x in zip(weights, norms)]
        u = {n: sum(s * m[n] for s, m in zip(sc, msgs)) / ws for n in self.G}
        body = float(norm({n: v for n, v in u.items() if n != EMB}))
        r = med / (body + 1e-12)
        return {n: v if n == EMB else v * r for n, v in u.items()}, norms

    def _work(self, w: int) -> list[int]:
        """Inner-step counts of the work items worker ``w`` does this round."""
        s = self.speeds[w] / max(self.speeds)
        if self.het == "budget":
            return [max(1, round(self.H * s))]
        if self.het == "units":
            return [self.unit] * max(1, round(self.H * s / self.unit))
        return [self.H]

    @torch.no_grad()
    def _gossip(self, before: list) -> tuple[float, float]:
        sent = []
        for g in range(self.N):
            masks, vals = {}, {}
            for n, x in self.X[g].items():
                if x.dim() == 2:
                    a = self.acc[g][n].mul_(self.beta).add_(x - before[g][n])
                    masks[n] = chunk_mask(a, self.gk)
                    a.masked_fill_(masks[n], 0.0)
                vals[n] = x.bfloat16().float()
            sent.append((masks, vals))
        rng = gen(self.seed * 7919 + self.round, "cpu")
        for _ in range(self.partners):
            perm = torch.randperm(self.N, generator=rng).tolist()
            for i, j in zip(perm[0::2], perm[1::2]):
                for src, dst in ((i, j), (j, i)):
                    masks, vals = sent[src]
                    for n, x in self.X[dst].items():
                        mixed = 0.5 * (x + vals[n])
                        x.copy_(torch.where(masks[n], mixed, x) if n in masks else mixed)
        gs = len(self.groups[0])
        return self.partners * self.gmsg_bytes / gs, self.partners * self.gmsg_bytes

    def step(self, it):
        self.round = it
        tokens, inner, up, down, all_norms = 0, [], 0.0, 0.0, []
        before = ([{n: x.detach().clone() for n, x in X.items()} for X in self.X]
                  if self.partners else None)
        for g, members in enumerate(self.groups):
            if self.topo == "gossip":
                w = members[0]
                t, loss = local_adamw(self.reps[w], self.opts[w], self.data, self.H, clip=1.0,
                                      worker=w, workers=self.W, noniid=False)
                tokens += t
                inner.append(loss)
                continue
            msgs, weights, owner = [], [], []
            for w in members:
                for steps in self._work(w):
                    assign(self.rp[w], self.X[g])
                    t, loss = local_adamw(self.reps[w], self.opts[w], self.data, steps,
                                          clip=1.0, worker=w, workers=self.W, noniid=False)
                    tokens += t
                    inner.append(loss)
                    msgs.append(self._message(w, self.X[g]))
                    weights.append(float(t) if self.weight == "tokens" else 1.0)
                    owner.append(w)
            merged, norms = self._merge(msgs, weights)
            all_norms += norms
            with torch.no_grad():
                for n, x in self.X[g].items():
                    x.sub_(merged[n], alpha=self.alpha)
            up += len(msgs) * self.msg_bytes
            down += sum(len(msgs) - owner.count(w) for w in members) * self.msg_bytes
        spread, node0 = 0.0, None
        if self.partners:
            gup, gdown = self._gossip(before)
            up += gup * self.W
            down += gdown * self.W
        if self.N > 1:
            with torch.no_grad():
                for n, p in self.G.items():
                    p.copy_(sum(X[n] for X in self.X) / self.N)
                gnorm = float(norm(self.G))
                spread = sum(float(norm({n: X[n] - self.G[n] for n in self.G}))
                             for X in self.X) / self.N / gnorm
                if self.topo == "hier":
                    assign(self.rp[0], self.X[0])
            node0 = eval_loss(self.reps[0], self.data)
        if self.het == "sync":
            self.sim += self.H / min(self.speeds)
        else:
            self.sim += self.H
        extra = {"simTime": self.sim, "innerLoss": round(sum(inner) / len(inner), 4),
                 "upKB": round(up / self.W / 1e3, 1), "downKB": round(down / self.W / 1e3, 1),
                 "msgNorm": round(sum(all_norms) / max(len(all_norms), 1), 4)}
        if node0 is not None:
            extra |= {"node0Loss": round(node0, 4), "spread": round(spread, 5)}
        return Cost(tokens=tokens, fwd=tokens, bwd=tokens, up=up, down=down, extra=extra)
