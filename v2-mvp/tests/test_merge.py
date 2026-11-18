import random

import torch

from crowd_v2.crypto import hash_tensors
from crowd_v2.merge import gate, merge_module, outer_step


def _msgs(n: int, seed: int) -> dict[str, dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return {f"{i:064x}": {"a": torch.randn(64, 64, generator=g).bfloat16(),
                          "b": torch.randn(64, generator=g).bfloat16(), "t": torch.randn((), generator=g)}
            for i in range(n)}


def test_merge_is_bit_identical_under_shuffle():
    msgs = _msgs(6, 0)
    keys = list(msgs)
    w = {"a": torch.zeros(64, 64), "b": torch.ones(64), "t": torch.ones(())}
    hashes = set()
    for seed in range(4):
        random.Random(seed).shuffle(keys)
        avg, _ = merge_module({k: msgs[k] for k in keys})
        hashes.add(hash_tensors(outer_step(w, avg, 1.0)))
    assert len(hashes) == 1


def test_median_clipping_bounds_a_huge_update():
    msgs = _msgs(5, 1)
    honest_avg, _ = merge_module(msgs)
    attacker = next(iter(msgs))
    msgs[attacker] = {n: v * 1e4 for n, v in msgs[attacker].items()}
    avg, stats = merge_module(msgs)
    norm = lambda d: float(torch.sqrt(sum(v.double().pow(2).sum() for v in d.values())))  # noqa: E731
    med = stats["median_norm"]
    assert stats["clipped"] >= 1
    assert norm(avg) <= med + 1e-6
    assert norm(avg) < 3 * norm(honest_avg) + med


def test_gate_rolls_back_on_rise_or_nonfinite():
    assert gate(5.0, 5.3, 0.25)
    assert not gate(5.0, 5.2, 0.25)
    assert not gate(None, 9.0, 0.25)
    assert gate(5.0, float("nan"), 0.25)
