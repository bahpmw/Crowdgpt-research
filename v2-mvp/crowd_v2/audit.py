"""Checking layers 0, 2 (replay comparison + calibration) and 3.

Layer 1 (commit-then-upload) lives in the orchestrator; this module holds the
pure checks so the proposer and every verifier compute identical verdicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .cell import run_cell
from .nets import Hyper, Tokens, init_weights

NORM_MULT = 4.0       # layer 3: reject norms above 4x the module median
NEG_COS = -0.3        # layer 3: reject strongly negative cosine to the others' mean


def flat(msg: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([msg[n].reshape(-1).to(torch.float64) for n in sorted(msg)])


def layer0(msg: dict[str, torch.Tensor], shapes: dict[str, tuple[int, ...]]) -> str | None:
    if set(msg) != set(shapes):
        return "layer0:names"
    for n, t in msg.items():
        if tuple(t.shape) != tuple(shapes[n]):
            return "layer0:shape"
        if not bool(torch.isfinite(t.float()).all()):
            return "layer0:nonfinite"
    return None


def compare(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> tuple[float, float]:
    """Cosine and norm ratio (a over b) of two decompressed messages."""
    x, y = flat(a), flat(b)
    nx, ny = float(x.norm()), float(y.norm())
    if ny == 0.0:
        return (1.0, 1.0) if nx == 0.0 else (0.0, 1e9)
    if nx == 0.0:
        return 0.0, 0.0
    return round(float(x @ y) / (nx * ny), 6), round(nx / ny, 6)


@dataclass(frozen=True)
class Thresholds:
    cos_min: float = 0.95
    ratio_lo: float = 0.9
    ratio_hi: float = 1.1

    def passes(self, cos: float, ratio: float) -> bool:
        return cos >= self.cos_min and self.ratio_lo <= ratio <= self.ratio_hi


def layer3(msgs: dict[str, dict[str, torch.Tensor]], trusted: set[str]) -> dict[str, str]:
    """Cheap unsampled checks within one module.

    Norm: reject above 4x the module median. Direction: an unaudited message
    must not point against the mean of the messages that passed replay
    (a leave-one-out mean over everyone lets one strong attacker make honest
    cells look opposite, so only replay-verified cells define the direction).
    """
    if len(msgs) < 2:
        return {}
    vecs = {kid: flat(m) for kid, m in msgs.items()}
    norms = {kid: float(v.norm()) for kid, v in vecs.items()}
    ordered = sorted(norms.values())
    med = ordered[(len(ordered) - 1) // 2]
    bad = {kid: "layer3:norm" for kid in sorted(vecs) if med > 0 and norms[kid] > NORM_MULT * med}
    ref = [vecs[k] for k in sorted(trusted) if k in vecs and k not in bad]
    if not ref:
        return bad
    mean = sum(ref)
    for kid, v in sorted(vecs.items()):
        denom = float(mean.norm()) * norms[kid]
        if kid in bad or kid in trusted or denom == 0:
            continue
        if float(mean @ v) / denom < NEG_COS:
            bad[kid] = "layer3:direction"
    return bad


def calibrate(hp: Hyper, tokens: Tokens, devices: list[str]) -> dict:
    """Measure honest replay error and derive thresholds.

    Pairs measured: the same cell twice on cpu (same thread count), cpu with a
    different thread count, and cpu vs each extra device in ``devices``.
    """
    weights = init_weights(hp)
    threads = torch.get_num_threads()
    pairs = []
    for k in range(hp.K):
        base = run_cell(hp, weights, k, 1000 + k, {}, tokens).msg
        runs = {"cpu-same": run_cell(hp, weights, k, 1000 + k, {}, tokens).msg}
        torch.set_num_threads(max(1, threads // 2) if threads > 1 else 2)
        runs["cpu-threads"] = run_cell(hp, weights, k, 1000 + k, {}, tokens).msg
        torch.set_num_threads(threads)
        for dev in devices:
            runs[f"cpu-{dev}"] = run_cell(hp, weights, k, 1000 + k, {}, tokens, device=dev).msg
        for name, msg in runs.items():
            cos, ratio = compare(msg, base)
            pairs.append({"module": k, "pair": name, "cos": cos, "ratio": ratio})
    worst_cos = min(p["cos"] for p in pairs)
    worst_dev = max(abs(p["ratio"] - 1.0) for p in pairs)
    th = Thresholds(cos_min=round(max(0.5, min(0.98, worst_cos - 0.02)), 4),
                    ratio_lo=round(max(0.5, 1.0 - worst_dev - 0.05), 4),
                    ratio_hi=round(min(2.0, 1.0 + worst_dev + 0.05), 4))
    return {"pairs": pairs, "thresholds": asdict(th)}
