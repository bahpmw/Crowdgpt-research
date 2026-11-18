"""Decoupled DiLoCo (arXiv 2604.21428) MVP: asynchronous learners with
heterogeneous speeds, a syncer with quorum K plus a grace window, token-weighted
RDA merging and outer SGD-Nesterov, simulated on an event-driven crowd clock.

A learner pulls the global weights, runs H local AdamW steps (sim cost H/speed),
and delivers Delta = theta_pulled - theta_local. The syncer takes the next
arrival(s): once K learners have delivered it keeps collecting for up to
``grace`` sim-time units (it stops early when every learner is in), merges, takes
one outer step, and the delivering learners pull the new weights and restart
(staleness = outer steps applied since a learner's pull). Learners that arrived
early wait for the merge; that idle time is what ``goodput`` measures.
One harness ``step()`` = one merge event. ``model`` holds the syncer's weights.
Local training is computed lazily when the learner's event fires, which is
equivalent because a learner never receives anything mid-run.

Faithful: K=1 quorum + grace, Weight = c_tokens * (c_tokens / c_steps), RDA
(mean norm times the normalised mean direction) on non-embedding parameters and
a weighted average on the tied embedding, outer SGD-Nesterov, int4 option
without error feedback. ``async=0`` -> K = W, i.e. synchronous DiLoCo that waits
for the slowest learner every round (the control).

Simplified: no streaming fragments (P=1: the whole model syncs at once, RDA over
all non-embedding tensors as one fragment); learners block during the grace
window instead of continuing; outer lr 0.7 / momentum 0.9 come from the original
DiLoCo (this paper gives none); inner lr constant 3e-3, wd 0. Wire format: Delta
in bf16 (2 B/param) or int4 with one fp16 absmax scale per 4096 values (``quant=4``);
each delivering learner downloads the fp32 global (4 B/param).
"""

import copy
import heapq

import torch

from crowd import assign, local_adamw, norm
from harness import Cost
from zo import trainable

EMB = "tok_emb.weight"
QCHUNK = 4096


@torch.no_grad()
def int4(d: torch.Tensor) -> torch.Tensor:
    flat = d.reshape(-1)
    pad = (-flat.numel()) % QCHUNK
    x = torch.cat([flat, flat.new_zeros(pad)]).view(-1, QCHUNK)
    s = x.abs().amax(1, keepdim=True).clamp_min(1e-12) / 7
    return ((x / s).round().clamp(-7, 7) * s).reshape(-1)[:flat.numel()].view_as(d)


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        self.speeds = [float(s) for s in cfg.get("speeds", [1, 1, 0.5, 0.25])]
        self.W = len(self.speeds)
        self.H = int(cfg.get("H", 30))
        self.K = int(cfg.get("quorum", 1)) if int(cfg.get("async", 1)) else self.W
        self.grace = float(cfg.get("grace", 3.0)) if int(cfg.get("async", 1)) else 0.0
        self.merge = cfg.get("merge", "rda")
        self.olr = float(cfg.get("olr", 0.7))
        self.mom = float(cfg.get("mom", 0.9))
        self.quant = int(cfg.get("quant", 16))
        self.noniid = bool(cfg.get("noniid", 0))
        self.clip = float(cfg.get("clip", 1.0))
        self.G = trainable(model)
        self.reps = [copy.deepcopy(model) for _ in range(self.W)]
        self.rp = [trainable(r) for r in self.reps]
        self.opts = [torch.optim.AdamW(r.parameters(), lr=float(cfg.get("lr", 3e-3)),
                                       betas=(0.9, 0.95), weight_decay=float(cfg.get("wd", 0.0)))
                     for r in self.reps]
        self.buf = {n: torch.zeros_like(p) for n, p in self.G.items()}
        self.n_params = sum(p.numel() for p in self.G.values())
        self.msg_bytes = (self.n_params / 2 + 2 * -(-self.n_params // QCHUNK)
                          if self.quant == 4 else 2 * self.n_params)
        self.version, self.wait, self.now = 0, 0.0, 0.0
        self.snap, self.pulled, self.heap = [None] * self.W, [0] * self.W, []
        for m in range(self.W):
            self._pull(m, 0.0)

    @torch.no_grad()
    def _pull(self, m: int, t: float) -> None:
        assign(self.rp[m], self.G)
        self.snap[m] = {n: p.detach().clone() for n, p in self.G.items()}
        self.pulled[m] = self.version
        heapq.heappush(self.heap, (t + self.H / self.speeds[m], m))

    def step(self, it):
        group = [heapq.heappop(self.heap)]
        while self.heap and (len(group) < self.K or self.heap[0][0] <= group[self.K - 1][0] + self.grace):
            group.append(heapq.heappop(self.heap))
        t_merge = group[-1][0] if len(group) == self.W else max(group[-1][0],
                                                                  group[self.K - 1][0] + self.grace)
        deltas, weights, tokens, inner = [], [], 0, 0.0
        for t_arr, m in group:
            t, loss = local_adamw(self.reps[m], self.opts[m], self.data, self.H, clip=self.clip,
                                  worker=m, workers=self.W, noniid=self.noniid)
            tokens += t
            inner += loss / len(group)
            with torch.no_grad():
                d = {n: self.snap[m][n] - self.rp[m][n] for n in self.G}
                deltas.append({n: int4(v) for n, v in d.items()} if self.quant == 4
                              else {n: v.bfloat16().float() for n, v in d.items()})
            weights.append(t * (t / self.H))
            self.wait += t_merge - t_arr
        stale = sum(self.version - self.pulled[m] for _, m in group) / len(group)
        with torch.no_grad():
            ws = sum(weights)
            body = [n for n in self.G if n != EMB]
            if self.merge == "rda":
                norms = [float(norm({n: d[n] for n in body})) for d in deltas]
                rbar = sum(w * r for w, r in zip(weights, norms)) / ws
                u = {n: sum(w * d[n] / (r + 1e-12) for w, d, r in zip(weights, deltas, norms)) / ws
                     for n in body}
                scale = rbar / (float(norm(u)) + 1e-12)
                merged = {n: u[n] * scale for n in body}
                merged[EMB] = sum(w * d[EMB] for w, d in zip(weights, deltas)) / ws
            else:
                merged = {n: sum(w * d[n] for w, d in zip(weights, deltas)) / ws for n in self.G}
            for n, p in self.G.items():
                b = self.buf[n].mul_(self.mom).add_(merged[n])
                p.sub_(merged[n] + self.mom * b, alpha=self.olr)
        self.version += 1
        self.now = t_merge
        for _, m in group:
            self._pull(m, t_merge)
        goodput = 1.0 - self.wait / (self.W * self.now)
        extra = {"simTime": round(self.now, 2), "merged": len(group), "staleness": round(stale, 3),
                 "goodput": round(goodput, 4), "innerLoss": round(inner, 4)}
        return Cost(tokens=tokens, fwd=tokens, bwd=tokens, up=len(group) * self.msg_bytes,
                    down=len(group) * 4 * self.n_params, extra=extra)
