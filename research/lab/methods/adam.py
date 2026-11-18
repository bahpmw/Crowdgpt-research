"""Baseline: backprop + AdamW on one worker (the quality ceiling).

``workers=W`` simulates synchronous data parallel: W micro-batches per step and
a dense fp32 gradient all-reduce, i.e. 4 bytes/param up per worker per step.
"""

import torch

from harness import Cost, loss_fn


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data = model, data
        self.workers = int(cfg.get("workers", 1))
        self.opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-3)),
                                     betas=(0.9, 0.95), weight_decay=0.0)
        self.params = model.n_params()

    def step(self, it):
        self.opt.zero_grad(set_to_none=True)
        tokens = 0
        for w in range(self.workers):
            idx, tgt = self.data.next(worker=w, workers=self.workers)
            (loss_fn(self.model, idx, tgt) / self.workers).backward()
            tokens += idx.numel()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return Cost(tokens=tokens, fwd=tokens, bwd=tokens, up=4 * self.params * self.workers)
