"""Shared zeroth-order building blocks (weight-space perturbation, seeded)."""

from __future__ import annotations

import torch

from harness import loss_fn


def trainable(model) -> dict[str, torch.Tensor]:
    seen, out = set(), {}
    for n, p in model.named_parameters(remove_duplicate=False):
        if id(p) not in seen:
            seen.add(id(p))
            out[n] = p
    return out


def gen(seed: int, device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFF)
    return g


@torch.no_grad()
def perturb(params: dict, seed: int, scale: float, masks: dict | None = None,
            directions: dict | None = None):
    """p += scale * z(seed) (optionally masked or with explicit directions)."""
    g = None
    for n, p in params.items():
        if directions is not None:
            z = directions[n]
        else:
            if g is None:
                g = gen(seed, p.device)
            z = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)
        if masks is not None and n in masks:
            z = z * masks[n]
        p.add_(z, alpha=scale)


@torch.no_grad()
def spsa_pair(model, params, idx, tgt, seed, eps, masks=None, directions=None):
    """Two-point loss difference along z(seed): returns (proj_grad, loss+, loss-)."""
    perturb(params, seed, eps, masks, directions)
    lp = float(loss_fn(model, idx, tgt))
    perturb(params, seed, -2 * eps, masks, directions)
    lm = float(loss_fn(model, idx, tgt))
    perturb(params, seed, eps, masks, directions)
    return (lp - lm) / (2 * eps), lp, lm
