"""SOMA (arXiv 2609.37899) MVP: N independent GPT experts, each trained by SPSA on
its own k-means data cluster with its own loss and optimizer; merged only at
inference by routing on a prefix. Control: ``n=1``, one full-size model with the
same SPSA recipe and the same parameter-forward compute per step.

!! Read ``extra['ensembleLoss']``, not the harness evalLoss. For n>1 the
harness ``model`` is untouched (its evalLoss stays at the untrained value); the
ensemble is evaluated here on 256 fixed val windows: route each window by the
nearest centroid of its first 32 tokens' bag-of-tokens (top ``route_k`` experts,
probabilities averaged) and score positions 33..128. For n=1 the single expert
IS the harness ``model`` and ensembleLoss is that model on the same windows.
The ensemble eval runs inside ``step()`` every ``ens_secs`` (5) seconds and on
every step near the ``seconds`` (120) budget, so the last logged row is fresh.

Faithful: per-expert SPSA with n_pert (8) 50%-sparse Rademacher probes (0 w.p.
1/2, +-1 w.p. 1/4, no rescaling; RMSNorm gains dense +-1), all probes and both
signs on one batch, g = mean_j (L+ - L-)/(2 eps) z_j; Adam (0.9, 0.999) with
coupled weight decay 1e-4, lr = eps = 1e-3; independent losses and no
communication between experts; tied embedding/head never perturbed.

Simplified: (1) head: the tied tok_emb/head gets the exact decoder-path gradient
(softmax(Eh) - onehot) h^T every step from a clean pass, as repo DUST does, with
Adam lr ``head_lr`` (3e-3); ``head_freeze=K`` stops it after K steps. There is
no shared seed run - experts start from different random inits and each has its
own embedding (a shared frozen embedding would need equal widths). (2) Clusters:
tf-idf-lite bag-of-tokens (log1p counts x idf over the 4096 ids, L2-normalised),
spherical k-means (10 iters, unbalanced) fit on 20k windows of 128 tokens, then
a ``pool`` (60k) of windows is assigned and each expert samples its own
windows. (3) No eps/lr plateau decay. (4) Sizes: by default experts are 6 layers
with width chosen so N experts ~ the 10m model's parameters (n=4 -> d=128,
2.10M each, 8.39M total = 1.14x); ``width=256`` gives same-size experts.

Accounting: tokens = N x batch x seq per step. fwd is in *full-model-equivalent*
token passes (raw passes x expert params / 10m params) so compute compares
across arms. simTime per step = (2 n_pert + 1) x expert/full params for n>1
(experts in parallel, one GPU each, no communication -> up = down = 0); the n=1
control is run as DDPP over ``ddpp`` (4) workers: simTime (2 n_pert / ddpp + 1),
and per step every worker exchanges the dense fp32 head gradient plus 12 B/probe.
"""

import time

import numpy as np
import torch

from actzo import head_grad, per_seq_loss
from crowd_v15.model import GPT, ModelConfig, Perturb, param_report
from harness import Cost, sync
from zo import gen, trainable

EMB = "tok_emb.weight"
PREFIX = 32


@torch.no_grad()
def spsa(model, body: dict, idx, tgt, seed: int, n_pert: int, eps: float, head: bool):
    """SOMA probe estimate for ``body`` (+ exact head grad); returns (grads, clean loss, forwards)."""
    g = {n: torch.zeros_like(p) for n, p in body.items()}
    out, forwards, loss = {}, 0, None
    if head:
        st = Perturb(record={})
        loss = per_seq_loss(model(idx, st), tgt).mean()
        out[EMB] = head_grad(model, st.record["__head__"], tgt)
        forwards += 1
    for j in range(n_pert):
        gn = gen(seed * 1000 + j, idx.device)
        z = {}
        for n, p in body.items():
            r = torch.randint(0, 4, p.shape, generator=gn, device=p.device)
            z[n] = (r == 2).float() - (r == 3).float() if p.dim() > 1 else (r % 2).float() * 2 - 1
            p.add_(z[n], alpha=eps)
        lp = per_seq_loss(model(idx), tgt).mean()
        for n, p in body.items():
            p.sub_(z[n], alpha=2 * eps)
        lm = per_seq_loss(model(idx), tgt).mean()
        coef = (lp - lm) / (2 * eps * n_pert)
        for n, p in body.items():
            p.add_(z[n], alpha=eps)
            g[n].add_(coef * z[n])
        forwards += 2
    out.update(g)
    return out, (loss if loss is not None else (lp + lm) / 2), forwards


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data, self.device = model, data, device
        self.N = int(cfg.get("n", 4))
        self.n_pert = int(cfg.get("n_pert", 8))
        self.eps = float(cfg.get("eps", 1e-3))
        self.head_freeze = int(cfg.get("head_freeze", 0))
        self.route_k = int(cfg.get("route_k", 1))
        self.ens_secs = float(cfg.get("ens_secs", 5.0))
        self.budget = float(cfg.get("seconds", 120.0))
        self.ddpp = int(cfg.get("ddpp", 4)) if self.N == 1 else 1
        seed = int(cfg.get("seed", 0))
        full = model.n_params()
        if self.N == 1:
            self.experts = [model]
        else:
            width = int(cfg.get("width", min(range(32, 257, 32), key=lambda w: abs(
                self.N * param_report(ModelConfig(n_layers=6, d_model=w))["total"] - full))))
            ecfg = ModelConfig(vocab_size=4096, n_layers=6, d_model=width, n_heads=width // 32,
                               context_length=512)
            self.experts = []
            for e in range(self.N):
                torch.manual_seed(seed + 1 + e)
                self.experts.append(GPT(ecfg).to(device))
        self.frac = self.experts[0].n_params() / full
        self.bodies, self.opts = [], []
        for m in self.experts:
            tp = trainable(m)
            body = {n: p for n, p in tp.items() if n != EMB}
            self.bodies.append(body)
            self.opts.append(torch.optim.Adam(
                [{"params": list(body.values()), "lr": float(cfg.get("lr", 1e-3)),
                  "weight_decay": float(cfg.get("wd", 1e-4))},
                 {"params": [tp[EMB]], "lr": float(cfg.get("head_lr", 3e-3)), "weight_decay": 0.0}],
                betas=(0.9, 0.999), eps=1e-8))
        self._cluster(data, int(cfg.get("pool", 60000)), int(cfg.get("fit", 20000)), seed)
        vrng = np.random.default_rng(4321)
        starts = vrng.integers(0, len(data.val) - data.seq - 1, size=int(cfg.get("nval", 256)))
        v = torch.from_numpy(np.asarray(data.val[starts[:, None] + np.arange(data.seq + 1)])
                             .astype(np.int64)).to(device)
        self.val = v
        self.route = (self._feats(v[:, :PREFIX]) @ self.cent.t()).topk(self.route_k, dim=1).indices
        self.sim, self.t, self.dt, self.last_ens, self.ens = 0.0, 0.0, 0.0, -1e9, {}

    def _feats(self, tok: torch.Tensor) -> torch.Tensor:
        counts = torch.zeros(tok.size(0), 4096, device=self.device)
        counts.scatter_add_(1, tok, torch.ones_like(tok, dtype=torch.float))
        x = torch.log1p(counts) * self.idf
        return x / (x.norm(dim=1, keepdim=True) + 1e-12)

    @torch.no_grad()
    def _cluster(self, data, pool: int, fit: int, seed: int) -> None:
        rng = np.random.default_rng(seed + 99)
        starts = rng.integers(0, len(data.train) - data.seq - 1, size=pool)
        win = np.asarray(data.train[starts[:, None] + np.arange(data.seq + 1)]).astype(np.int64)
        self.pool = torch.from_numpy(win).to(self.device)
        if self.N == 1:
            self.idf = torch.ones(4096, device=self.device)
            self.cent = torch.ones(1, 4096, device=self.device)
            self.members = [torch.arange(pool, device=self.device)]
            return
        sample = self.pool[:fit, :-1]
        df = torch.zeros(4096, device=self.device)
        for c in sample.split(5000):
            cnt = torch.zeros(c.size(0), 4096, device=self.device)
            cnt.scatter_add_(1, c, torch.ones_like(c, dtype=torch.float))
            df += (cnt > 0).float().sum(0)
        self.idf = torch.log((fit + 1) / (df + 1)) + 1
        x = self._feats(sample)
        g = torch.Generator().manual_seed(seed)
        cent = x[torch.randperm(fit, generator=g)[:self.N].to(self.device)]
        for _ in range(10):
            a = (x @ cent.t()).argmax(1)
            new = torch.zeros_like(cent).index_add_(0, a, x)
            cent = new / (new.norm(dim=1, keepdim=True) + 1e-12)
        self.cent = cent
        assign = torch.cat([(self._feats(c[:, :-1]) @ cent.t()).argmax(1) for c in self.pool.split(5000)])
        self.members = [torch.nonzero(assign == e).squeeze(1) for e in range(self.N)]
        print(f"[soma] cluster sizes {[int(m.numel()) for m in self.members]}", flush=True)

    @torch.no_grad()
    def _ensemble(self) -> dict:
        idx, tgt = self.val[:, :-1], self.val[:, 1:]
        lp = []
        for m in self.experts:
            m.eval()
            rows = []
            for i, t in zip(idx.split(64), tgt.split(64)):
                logp = torch.log_softmax(m(i).float(), -1).gather(-1, t.unsqueeze(-1)).squeeze(-1)
                rows.append(logp[:, PREFIX:])
            m.train()
            lp.append(torch.cat(rows))
        lp = torch.stack(lp)                                     # (N, nval, tail)
        sel = lp.gather(0, self.route.t().unsqueeze(-1).expand(-1, -1, lp.size(2)))
        mix = torch.logsumexp(sel, 0) - np.log(self.route_k)
        per = -lp.mean(2)                                        # (N, nval)
        out = {"ensembleLoss": round(float(-mix.mean()), 4),
               "oracleLoss": round(float(per.min(0).values.mean()), 4)}
        top1 = self.route[:, 0]
        for e in range(self.N):
            hit = top1 == e
            if hit.any():
                out[f"exp{e}Own"] = round(float(per[e][hit].mean()), 4)
        return out

    def estimate(self, idx, tgt, it):
        body = {n: p for n, p in trainable(self.model).items() if n != EMB}
        grads, _, forwards = spsa(self.model, body, idx, tgt, it, self.n_pert, self.eps, True)
        # probe.py calls .double() before .cpu(), which MPS cannot do
        return {n: g.cpu() for n, g in grads.items()}, forwards

    def step(self, it):
        t0 = time.perf_counter()
        head = not (self.head_freeze and it >= self.head_freeze)
        losses, forwards, B = [], 0, self.data.batch
        for e, m in enumerate(self.experts):
            mem = self.members[e]
            pick = mem[torch.randint(0, mem.numel(), (B,), device=self.device)]
            w = self.pool[pick]
            grads, loss, f = spsa(m, self.bodies[e], w[:, :-1], w[:, 1:], it * 100 + e,
                                  self.n_pert, self.eps, head)
            forwards += f
            losses.append(loss)
            for n, p in trainable(m).items():
                p.grad = grads.get(n)
            self.opts[e].step()
        T = self.data.seq
        tokens = self.N * B * T
        self.sim += (2 * self.n_pert / self.ddpp + head) * (self.frac if self.N > 1 else 1.0)
        if self.t - self.last_ens >= self.ens_secs or self.t + 2 * self.dt >= self.budget:
            self.ens = self._ensemble()
            self.last_ens = self.t
        sync(self.device)
        self.dt = time.perf_counter() - t0
        self.t += self.dt
        if self.N == 1:
            v, d = self.experts[0].cfg.vocab_size, self.experts[0].cfg.d_model
            up = self.ddpp * 4 * v * d + 12 * self.n_pert
        else:
            up = 0
        extra = {"simTime": round(self.sim, 3),
                 "trainLoss": round(float(torch.stack(losses).mean()), 4), **self.ens}
        return Cost(tokens=tokens, fwd=forwards * B * T * self.frac, bwd=0, up=up, down=up,
                    extra=extra)
