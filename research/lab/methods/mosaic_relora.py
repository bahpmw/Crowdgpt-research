"""MOSAIC + ReLoRA: each MOSAIC worker trains LoRA adapters on its module's block
linears instead of the full weights, after ``warm`` full-rank MOSAIC rounds.

Per round (after warm-up) worker w with module k: the replica's frozen W0 is the
broadcast round weights (mosaic.step copies them in), every adapter is reset
(A re-drawn from a seed shared by all workers of the round, B = 0), the adapters'
AdamW state is pruned by magnitude (``prune``) and the inner LR re-warms linearly
over ``rwarm`` steps (ReLoRA's jagged schedule with reset interval = H). After H
steps the local weight is W0 + s*B@A, so the module delta is the rank-r product
-s*B@A. Merge-and-reset therefore happens every round, on the server.

Upload modes (``up``):
- ``product`` (default): the dense rank-r product goes through the same
  SparseLoCo chunk-Top-k + 2-bit + error-feedback compressor as MOSAIC, so the
  bytes equal MOSAIC's; the server averages products.
- ``factors``: bf16 A and B are uploaded, r*(m+n)*2 bytes per matrix; the server
  forms each worker's product and averages PRODUCTS (exact up to bf16).
- ``factoravg``: the ablation the literature warns about (arXiv 2601.00566):
  the server averages A and B separately and applies s * mean(B) @ mean(A).
  Same bytes as ``factors``. Median clipping then sees only non-LoRA tensors.
Embedding, head, RMSNorm gains and local-head tau stay full-rank in every mode
and are sent exactly as MOSAIC sends them. ``rank=0`` is plain MOSAIC. LoRA
scale s = 32/r unless ``scale`` is given (see methods/relora.py).
"""

import torch

from methods import mosaic
from methods.relora import add_lora, prune_state
from methods.sparseloco import CHUNK, compress


class Method(mosaic.Method):
    def __init__(self, model, data, cfg, device):
        super().__init__(model, data, cfg, device)
        self.rank = int(cfg.get("rank", 32))
        self.scale = float(cfg.get("scale", 32 / max(self.rank, 1)))
        self.warm = int(cfg.get("warm", 4))
        self.up_mode = cfg.get("up", "product")
        if self.up_mode not in ("product", "factors", "factoravg"):
            raise ValueError(f"unknown up mode {self.up_mode!r}")
        self.rwarm = int(cfg.get("rwarm", 5))
        self.prune = float(cfg.get("prune", 0.99))
        self.lr = float(cfg.get("lr", 3e-3))
        self.adapters = None
        self.fac = {}

    def _to_lora(self):
        self.adapters = []
        for w in range(self.W):
            k, rep = self.mod[w], self.reps[w]
            blocks = [(f"model.blocks.{i}", rep.model.blocks[i]) for i in self.split[k]]
            if k < self.K - 1:
                blocks += [(f"heads.{k}.blocks.{j}", b) for j, b in enumerate(rep.heads[k].blocks)]
            ad = {n: (w0, p) for n, w0, p in add_lora(blocks, self.rank, self.scale)}
            self.adapters.append(ad)
            rp = dict(rep.named_parameters())
            self.rp[w] = {n: ad[n][0] if n in ad else rp[n] for n in self.G}
            full = [self.rp[w][n] for n in self.names[k] if n not in ad]
            lora = [t for _, p in ad.values() for t in (p.A, p.B)]
            self.opts[w] = torch.optim.AdamW([{"params": full}, {"params": lora}], lr=self.lr,
                                             betas=(0.9, 0.95), weight_decay=0.0)
        if self.up_mode != "product":
            for k in range(self.K):
                w = self.mod.index(k)
                ad = self.adapters[w]
                fac = sum(2 * self.rank * sum(self.G[n].shape) for n in ad)
                rest = [n for n in self.names[k] if n not in ad]
                n2 = sum(self.G[n].numel() for n in rest if self.G[n].dim() == 2)
                n1 = sum(self.G[n].numel() for n in rest if self.G[n].dim() != 2)
                self.msg_bytes[k] = fac + (n2 // (CHUNK * CHUNK) * (self.k * 14 + 32) / 8 + 2 * n1
                                           if self.k else 4 * (n1 + n2))

    def _inner(self, w, steps):
        if self.adapters is None:
            return super()._inner(w, steps)
        k, rep, opt, ad = self.mod[w], self.reps[w], self.opts[w], self.adapters[w]
        order = sorted(ad)
        for j, n in enumerate(order):
            ad[n][1].reset(self.round * 100003 + self.names[k].index(n) * 7 + j)
        lora = opt.param_groups[1]["params"]
        prune_state(opt, lora, self.prune)
        params = opt.param_groups[0]["params"] + lora
        loss, idx = None, None
        for j in range(steps):
            for g in opt.param_groups:
                g["lr"] = self.lr * min(1.0, (j + 1) / self.rwarm)
            idx, tgt = self.data.next(worker=w, workers=self.W, noniid=self.noniid)
            loss = self._local_loss(rep, k, idx, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        return steps * idx.numel(), float(loss.detach())

    @torch.no_grad()
    def _message(self, w):
        if self.adapters is None:
            return super()._message(w)
        ad, msg = self.adapters[w], {}
        for n in self.names[self.mod[w]]:
            if n in ad:
                p = ad[n][1]
                if self.up_mode == "product":
                    d = -p.scale * (p.B @ p.A)
                    if self.k:
                        acc = self.ef[w][n].mul_(self.beta).add_(d)
                        msg[n] = compress(acc, self.k)
                        acc.sub_(msg[n])
                    else:
                        msg[n] = d
                    continue
                b, a = p.B.bfloat16().float(), p.A.bfloat16().float()
                if self.up_mode == "factors":
                    msg[n] = -p.scale * (b @ a)
                else:
                    self.fac.setdefault(n, []).append((b, a, p.scale))
                    msg[n] = torch.zeros_like(self.G[n])
                continue
            d = self.G[n] - self.rp[w][n]
            if self.k and d.dim() == 2:
                acc = self.ef[w][n].mul_(self.beta).add_(d)
                msg[n] = compress(acc, self.k)
                acc.sub_(msg[n])
            else:
                msg[n] = d.bfloat16().float() if self.k else d
        return msg

    def step(self, it):
        self.round = it
        if self.rank and it == self.warm:
            self._to_lora()
        self.fac = {}
        c = super().step(it)
        with torch.no_grad():
            for n, fs in self.fac.items():
                b = sum(f[0] for f in fs) / len(fs)
                a = sum(f[1] for f in fs) / len(fs)
                self.G[n].add_(b @ a, alpha=fs[0][2] * self.alpha)
        c.extra["lora"] = int(self.adapters is not None)
        c.extra["upMode"] = self.up_mode
        return c
