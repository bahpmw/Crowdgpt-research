"""Shared crowd-simulation helpers for the distributed lab methods.

Used by methods/sparseloco.py, methods/ddiloco.py and methods/throttle.py.
Worker replicas are extra GPT copies (``copy.deepcopy(model)``, which keeps the
tied embedding/head tied); updates travel as dicts keyed like ``zo.trainable``.
"""

from __future__ import annotations

import torch

from harness import loss_fn


@torch.no_grad()
def assign(dst: dict, src: dict) -> None:
    for n, p in dst.items():
        p.copy_(src[n])


def norm(update: dict) -> torch.Tensor:
    return torch.sqrt(sum(v.float().pow(2).sum() for v in update.values()))


def local_adamw(rep, opt, data, steps: int, *, clip: float, worker: int, workers: int,
                noniid: bool) -> tuple[int, float]:
    """``steps`` backprop AdamW steps on a worker replica; returns (tokens, last loss)."""
    tokens, loss = 0, None
    for _ in range(steps):
        idx, tgt = data.next(worker=worker, workers=workers, noniid=noniid)
        loss = loss_fn(rep, idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(rep.parameters(), clip)
        opt.step()
        tokens += idx.numel()
    return tokens, float(loss.detach())


@torch.no_grad()
def corrupt(update: dict, kind: str, c: float, gen: torch.Generator) -> dict:
    """A Byzantine message built from an honest one.

    flip:  -c * update
    noise: Gaussian values on the update's support (so a sparse message stays in
           the same wire format), rescaled to c * ||update||.
    """
    if kind == "flip":
        return {n: -c * v for n, v in update.items()}
    if kind == "noise":
        z = {n: torch.randn(v.shape, generator=gen, device=v.device) * (v != 0)
             for n, v in update.items()}
        s = c * norm(update) / (norm(z) + 1e-12)
        return {n: s * v for n, v in z.items()}
    raise ValueError(f"unknown attack {kind!r}")
