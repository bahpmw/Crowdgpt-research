"""THROTTLE (arXiv 2609.39357) MVP: asynchronous robust SGD with soft throttling,
simulated on an event-driven crowd clock with heterogeneous and Byzantine workers.

Each worker holds the model snapshot it was handed, computes one backprop
minibatch gradient on it (sim cost 1/speed x (1 + jitter*U)), and sends it. The
server applies every arrival immediately with a per-arrival weight:

  rule=throttle  first arrival of a client in the round: weight 1/n and
                 g = a + clip_tau(v - a) (centered clipping around anchor a =
                 mean of the previous round's clipped first arrivals);
                 c-th repeat: weight q^-c and g = clip_lam(v). The round ends
                 when all n clients delivered; the server then hard-restarts:
                 every worker gets the current model and in-flight work is
                 discarded (counted in fwd/bwd by the fraction already done).
  rule=mean      plain async SGD: every arrival weight 1, global-norm clip
                 ``clip`` (default 1, as in the adam baseline; 0 = none).

One harness ``step()`` = one arrival. ``model`` holds the server weights.

Attacks (``byz=f`` makes workers 0..f-1 Byzantine):
  flip   send -c * g (c=10, the paper's NG attack)
  noise  send Gaussian noise with norm c * ||g||
  fast   flip, and the Byzantine workers run ``fastx`` (4) times faster than the
         fastest honest worker, so they flood the server with repeats.

Simplified / assumed: the server optimizer is Adam (default, ``opt=adam``,
lr 1e-3) or SGD with momentum (``opt=sgd``, mom 0.9) fed weight * g as the
gradient - the paper uses plain SGD on tiny problems; Adam keeps a 120 s GPT run
informative but also caps per-step movement, so ``opt=sgd`` is the purer test.
Clipping radii are fixed, tau=2 (first arrivals, around the anchor) and lam=1
(repeats), on the scale of the adam baseline's clip 1 - radii adapted to the
median norm fed back on themselves and blew up when the loss rose; the anchor is
not projected onto a G-ball. Upload = ``bytes`` (4; 2 rounds the gradient to
bf16) B/param per arrival; download = 4 B/param per hand-out (all n at a restart).
Discarded in-flight work counts in fwd/bwd by the fraction already done.
"""

import copy
import heapq

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from crowd import corrupt
from harness import Cost, loss_fn
from zo import gen


def clip(u: torch.Tensor, r: float) -> torch.Tensor:
    return u * min(1.0, r / (float(u.norm()) + 1e-12))


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data, self.device = model, data, device
        self.speeds = [float(s) for s in cfg.get("speeds", [1, 1, 1, 1, 0.5, 0.5, 0.25, 0.25])]
        self.n = len(self.speeds)
        self.byz = int(cfg.get("byz", 0))
        self.attack = cfg.get("attack", "flip")
        self.c = float(cfg.get("c", 10.0))
        if self.attack == "fast":
            fast = float(cfg.get("fastx", 4.0)) * max(self.speeds[self.byz:])
            self.speeds[:self.byz] = [fast] * self.byz
        self.rule = cfg.get("rule", "throttle")
        self.q = float(cfg.get("q", 1.3))
        self.mean_clip = float(cfg.get("clip", 1.0))
        self.tau, self.lam = float(cfg.get("tau", 2.0)), float(cfg.get("lam", 1.0))
        self.jitter = float(cfg.get("jitter", 0.2))
        self.bytes = float(cfg.get("bytes", 4))
        self.noniid = bool(cfg.get("noniid", 0))
        lr = float(cfg.get("lr", 1e-3))
        self.params = list(model.parameters())
        self.opt = (torch.optim.SGD(self.params, lr=lr, momentum=float(cfg.get("mom", 0.9)))
                    if cfg.get("opt", "adam") == "sgd" else torch.optim.Adam(self.params, lr=lr))
        self.rep = copy.deepcopy(model)
        self.rparams = list(self.rep.parameters())
        self.P = sum(p.numel() for p in self.params)
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.version, self.snaps = 0, {}
        self.ver, self.start, self.heap = [0] * self.n, [0.0] * self.n, []
        self.count = [0] * self.n
        self.anchor = torch.zeros(self.P, device=device)
        self.first_sum = torch.zeros_like(self.anchor)
        self.rounds, self.done, self.discarded = 0, 0.0, 0.0
        self.infl_byz, self.infl = 0.0, 0.0
        for i in range(self.n):
            self._hand(i, 0.0)

    def _hand(self, i: int, now: float) -> None:
        if self.version not in self.snaps:
            self.snaps[self.version] = parameters_to_vector(self.params).detach().clone()
        self.ver[i], self.start[i] = self.version, now
        dur = (1.0 + self.jitter * self.rng.random()) / self.speeds[i]
        heapq.heappush(self.heap, (now + dur, i))

    def step(self, it):
        now, i = heapq.heappop(self.heap)
        vector_to_parameters(self.snaps[self.ver[i]], self.rparams)
        idx, tgt = self.data.next(worker=i, workers=self.n, noniid=self.noniid)
        self.rep.zero_grad(set_to_none=True)
        loss = loss_fn(self.rep, idx, tgt)
        loss.backward()
        tokens = idx.numel()
        with torch.no_grad():
            v = torch.cat([p.grad.reshape(-1) for p in self.rparams])
            raw = float(v.norm())
            if self.bytes == 2:
                v = v.bfloat16().float()
            if i < self.byz:
                kind = "flip" if self.attack == "fast" else self.attack
                v = corrupt({"v": v}, kind, self.c, gen(it * 1000 + i, self.device))["v"]
            if self.rule == "mean":
                w, g = 1.0, (clip(v, self.mean_clip) if self.mean_clip else v)
            elif self.count[i] == 0:
                w, g = 1.0 / self.n, self.anchor + clip(v - self.anchor, self.tau)
                self.first_sum += g
            else:
                w, g = self.q ** -self.count[i], clip(v, self.lam)
            self.count[i] += 1
        for p, gp in zip(self.params, torch.split(w * g, [p.numel() for p in self.params])):
            p.grad = gp.view_as(p)
        self.opt.step()
        self.version += 1
        gn = w * float(g.norm())
        self.infl += gn
        self.infl_byz += gn if i < self.byz else 0.0
        self.done += 1
        down = 4 * self.P
        self._hand(i, now)
        fwd_waste = 0.0
        if self.rule != "mean" and min(self.count) >= 1:
            self.rounds += 1
            self.anchor = self.first_sum / self.n
            self.first_sum.zero_()
            self.count = [0] * self.n
            for due, j in self.heap:
                frac = (now - self.start[j]) / (due - self.start[j])
                self.discarded += frac
                fwd_waste += frac * tokens
            self.heap = []
            for j in range(self.n):
                self._hand(j, now)
            down = 4 * self.P * self.n
        live = set(self.ver)
        self.snaps = {k: s for k, s in self.snaps.items() if k in live}
        extra = {"simTime": round(now, 3), "rounds": self.rounds, "w": round(w, 4),
                 "gradNorm": round(raw, 4),
                 "loss": round(float(loss.detach()), 4), "byzShare": round(self.infl_byz / self.infl, 4),
                 "wasteFrac": round(self.discarded / (self.done + self.discarded), 4)}
        return Cost(tokens=tokens, fwd=tokens + fwd_waste, bwd=tokens + fwd_waste,
                    up=self.bytes * self.P, down=down, extra=extra)
