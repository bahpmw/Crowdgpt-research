"""Baseline: this repo's DUST estimator (crowd_v15.dust) + SGD momentum.

Data tokens per step are batch*seq; compute is every forward DUST runs.
Upload is the dense bf16 estimate (2 bytes/param); NGD1R would be ~8x less.
"""

import torch

from crowd_v15.dust import DustConfig, _param_map, dust_gradients
from harness import Cost


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        self.dcfg = DustConfig(
            population=int(cfg.get("pop", 32)), draw_chunk=int(cfg.get("chunk", 16)),
            sigma=float(cfg.get("sigma", 1e-2)), sigma_mode=cfg.get("sigma_mode", "relative"),
            sigma_rel=float(cfg.get("sigma_rel", 0.05)), gamma=float(cfg.get("gamma", 0.9)),
            mode=cfg.get("mode", "layer"), noise_device="device")
        self.lr = float(cfg.get("lr", 0.05))
        self.mom = float(cfg.get("momentum", 0.9))
        self.clip = float(cfg.get("clip", 1.0))
        self.params = _param_map(model)
        self.buf = {}
        self.n = model.n_params()

    def estimate(self, idx, tgt, it):
        grads, stats = dust_gradients(self.model, self.dcfg, idx, tgt, base_seed=0, step=it)
        return grads, stats["forwards"] * self.dcfg.draw_chunk

    @torch.no_grad()
    def step(self, it):
        idx, tgt = self.data.next()
        grads, stats = dust_gradients(self.model, self.dcfg, idx, tgt, base_seed=0, step=it)
        norm = torch.sqrt(sum(g.pow(2).sum() for g in grads.values()))
        scale = min(1.0, self.clip / (float(norm) + 1e-12)) if self.clip else 1.0
        for name, g in grads.items():
            p = self.params[name]
            b = self.buf.setdefault(name, torch.zeros_like(p))
            b.mul_(self.mom).add_(g.to(p.dtype), alpha=scale)
            p.add_(b, alpha=-self.lr)
        n_tok = idx.numel()
        # each DUST forward runs draw_chunk perturbed copies of the batch
        fwd = stats["forwards"] * self.dcfg.draw_chunk * n_tok
        return Cost(tokens=n_tok, fwd=fwd, up=2 * self.n,
                    extra={"gradNorm": round(float(norm), 4), "forwards": stats["forwards"]})
