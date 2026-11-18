"""MOSAIC + ReLoRA with DUST workers: swap local backprop for DUST, or add DUST
workers next to backprop workers in the same round.

Workers 0..workers-1 train exactly as methods/mosaic_relora.py (local backprop
through the SOLO-style local head). ``dworkers`` extra workers (appended after
them, module = w mod K as usual) estimate their module's gradient with this
repo's DUST estimator (crowd_v15.dust, forward passes only) instead.
``donly=1`` turns every worker into a DUST worker (the "swap" arm).

A DUST worker runs the whole model forward and scores the TRUE next-token loss,
perturbing only its own module's layers (``layers=`` of dust_gradients; with
``mode=all`` every layer is perturbed at once and only its module's estimates
are used), so it needs no local head. Each inner step copies the replica's effective weights
(W0 + s*B@A under LoRA) into a plain GPT copy, estimates dL/dW for its module,
and maps it onto what it trains: full weights (rank=0), or LoRA factors
(dB = s*G@A^T, dA = s*B^T@G) plus the full-rank embedding/head. RMSNorm gains
and the local-head blocks get no DUST estimate and stay unchanged (zero delta),
so in mixed rounds they are diluted by the DUST workers' token share.
Optimiser: SGD with momentum (``dlr``, ``dmom``), as methods/dust.py; under
LoRA the momentum restarts each round because the adapters are re-drawn.

Accounting: fwd counts every DUST forward (forwards x draw_chunk x tokens);
DUST workers add no backward. Upload bytes follow the base method.
"""

import copy

import torch

from crowd_v15.dust import DustConfig, _param_map, dust_gradients
from methods import mosaic_relora


class Method(mosaic_relora.Method):
    def __init__(self, model, data, cfg, device):
        self.n_bp = int(cfg.get("workers", 8))
        nd = int(cfg.get("dworkers", 0))
        if int(cfg.get("donly", 0)):
            nd, self.n_bp = self.n_bp, 0
        cfg = dict(cfg, workers=self.n_bp + nd)
        super().__init__(model, data, cfg, device)
        self.dcfg = DustConfig(
            population=int(cfg.get("pop", 8)), draw_chunk=int(cfg.get("chunk", 8)),
            sigma=float(cfg.get("sigma", 1e-2)), sigma_mode=cfg.get("sigma_mode", "relative"),
            sigma_rel=float(cfg.get("sigma_rel", 0.05)), gamma=float(cfg.get("gamma", 0.9)),
            mode=cfg.get("mode", "layer"), noise_device="device")
        self.dlr = float(cfg.get("dlr", 0.05))
        self.dmom = float(cfg.get("dmom", 0.9))
        self.dm = copy.deepcopy(model)
        self.dmp = _param_map(self.dm)
        shapes = self.dm.noise_shapes()
        self.layers = [[n for n in shapes
                        if any(n.startswith(f"blocks.{i}.") for i in self.split[k])
                        or (n == "tok_emb" and k == 0)] for k in range(self.K)]
        # mode=all perturbs every layer in one pass (cheap, diluted); estimates
        # for other modules' layers are then simply not used.
        self.layer_arg = self.layers if self.dcfg.mode == "layer" else [None] * self.K
        self.dopt = {}
        self.dforwards = 0

    def _dust_params(self, w):
        """(param, grad-key or None) pairs a DUST worker trains this round."""
        k, rp = self.mod[w], self.rp[w]
        ad = self.adapters[w] if self.adapters is not None else {}
        out = []
        for n in self.names[k]:
            if not n.startswith("model."):
                continue
            key = n[len("model."):]
            if n in ad:
                out += [(ad[n][1].A, ("A", key, n)), (ad[n][1].B, ("B", key, n))]
            elif key in self.dmp:
                out.append((rp[n], ("W", key, n)))
        return out

    def _inner(self, w, steps):
        if w < self.n_bp:
            return super()._inner(w, steps)
        k, rep = self.mod[w], self.reps[w]
        ad = self.adapters[w] if self.adapters is not None else {}
        for j, n in enumerate(sorted(ad)):
            ad[n][1].reset(self.round * 100003 + self.names[k].index(n) * 7 + j)
        pairs = self._dust_params(w)
        if ad or w not in self.dopt:
            self.dopt[w] = torch.optim.SGD([p for p, _ in pairs], lr=self.dlr, momentum=self.dmom)
        opt = self.dopt[w]
        idx = None
        for j in range(steps):
            idx, tgt = self.data.next(worker=w, workers=self.W, noniid=self.noniid)
            with torch.no_grad():
                for name, p in self.dm.named_parameters():
                    mod, attr = name.rsplit(".", 1)
                    p.copy_(getattr(rep.model.get_submodule(mod), attr))
                grads, stats = dust_gradients(self.dm, self.dcfg, idx, tgt, base_seed=w,
                                              step=self.round * 1000 + j, layers=self.layer_arg[k])
            self.dforwards += stats["forwards"] * self.dcfg.draw_chunk
            for p, (kind, key, n) in pairs:
                g = grads[key].to(p.dtype)
                if kind == "A":
                    lo = ad[n][1]
                    p.grad = lo.scale * (lo.B.detach().t() @ g)
                elif kind == "B":
                    lo = ad[n][1]
                    p.grad = lo.scale * (g @ lo.A.detach().t())
                else:
                    p.grad = g.clone()
            torch.nn.utils.clip_grad_norm_([p for p, _ in pairs], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            loss = float(torch.nn.functional.cross_entropy(
                self.dm(idx, None).float().reshape(-1, self.dm.cfg.vocab_size), tgt.reshape(-1)))
        return steps * idx.numel(), loss

    def step(self, it):
        self.dforwards = 0
        c = super().step(it)
        n_tok = self.data.seq * self.data.batch
        nd = self.W - self.n_bp
        # base accounting charged DUST workers as backprop workers; replace it
        if nd:
            per_w = c.tokens / self.W
            c.fwd += self.dforwards * n_tok - nd * per_w * sum(self.fwd_f) / self.K
            c.bwd -= nd * per_w * sum(self.bwd_f) / self.K
        c.extra["dustWorkers"] = nd
        c.extra["dustForwards"] = self.dforwards
        return c
