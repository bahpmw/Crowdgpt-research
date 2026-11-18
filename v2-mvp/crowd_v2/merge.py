"""Robust, bit-deterministic merge and the model gate.

Per module: messages in sorted payload-hash order, each clipped to the lower
median norm, converted to int64 fixed point (``rint(x * 2**24)``), summed as
integers and divided once. Everything runs in numpy float64/int64 on cpu, whose
pairwise sums do not depend on thread count, so every orchestrator gets the
same weight hash. Outer step: SGD with ``alpha`` (no outer state).
"""

from __future__ import annotations

import numpy as np
import torch

FIX = float(2 ** 24)


def merge_module(msgs: dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], dict]:
    """Mean of median-clipped messages keyed by payload hash; returns (avg, stats)."""
    order = sorted(msgs)
    arrays = [{n: msgs[h][n].to(torch.float64).numpy() for n in sorted(msgs[h])} for h in order]
    norms = [float(np.sqrt(sum(np.square(a).sum() for a in arr.values()))) for arr in arrays]
    med = sorted(norms)[(len(norms) - 1) // 2]
    scales = [min(1.0, med / n) if n > 0 else 1.0 for n in norms]
    avg = {}
    for name in arrays[0]:
        acc = np.zeros(arrays[0][name].shape, dtype=np.int64)
        for arr, s in zip(arrays, scales):
            acc += np.rint(arr[name] * s * FIX).astype(np.int64)
        avg[name] = torch.from_numpy(np.asarray(acc.astype(np.float64) / (FIX * len(arrays)), dtype=np.float32))
    return avg, {"n": len(order), "median_norm": round(med, 6),
                 "clipped": sum(1 for s in scales if s < 1.0)}


def outer_step(weights: dict[str, torch.Tensor], avg: dict[str, torch.Tensor],
               alpha: float) -> dict[str, torch.Tensor]:
    out = dict(weights)
    for n, v in avg.items():
        out[n] = (weights[n].double() - alpha * v.double()).float()
    return out


def gate(prev_loss: float | None, new_loss: float, tol: float) -> bool:
    """True when the candidate must be rolled back (non-finite or loss up by more than tol)."""
    if not np.isfinite(new_loss):
        return True
    return prev_loss is not None and new_loss > prev_loss + tol
