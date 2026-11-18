"""Noise-source tests: CPU generation is device-independent, device generation matches it on CPU."""

from __future__ import annotations

import torch

from crowd_v15.dust import DustConfig, derive_seed, draw_noise
from crowd_v15.model import GPT, ModelConfig, Perturb


def _shapes():
    model = GPT(ModelConfig.preset("tiny")).eval()
    return model.noise_shapes()


def test_cpu_and_device_modes_agree_on_cpu():
    shapes = _shapes()
    cpu = draw_noise(shapes, rows=2, seq_len=4, base_seed=7, step=1, draws=[3],
                     device=torch.device("cpu"), dtype=torch.float32, noise_device="cpu")
    dev = draw_noise(shapes, rows=2, seq_len=4, base_seed=7, step=1, draws=[3],
                     device=torch.device("cpu"), dtype=torch.float32, noise_device="device")
    assert set(cpu) == set(dev)
    for name in cpu:
        assert torch.equal(cpu[name], dev[name]), name


def test_cpu_noise_is_reproducible_and_draw_dependent():
    shapes = _shapes()
    a = draw_noise(shapes, rows=1, seq_len=3, base_seed=5, step=0, draws=[0],
                   device=torch.device("cpu"), dtype=torch.float32)
    b = draw_noise(shapes, rows=1, seq_len=3, base_seed=5, step=0, draws=[0],
                   device=torch.device("cpu"), dtype=torch.float32)
    c = draw_noise(shapes, rows=1, seq_len=3, base_seed=5, step=0, draws=[1],
                   device=torch.device("cpu"), dtype=torch.float32)
    for name in a:
        assert torch.equal(a[name], b[name])
        assert not torch.equal(a[name], c[name]), name


def test_bad_noise_device_is_rejected():
    try:
        DustConfig(noise_device="gpu")
    except ValueError as exc:
        assert "noise_device" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_seed_derivation_is_stable():
    assert derive_seed(1, 2, 3, "blocks.0.attn.qkv") == derive_seed(1, 2, 3, "blocks.0.attn.qkv")
    assert derive_seed(1, 2, 3, "a") != derive_seed(1, 2, 4, "a")
    assert derive_seed(1, 2, 3, "a") < 2**63


def test_perturb_scale_lookup_accepts_a_mapping():
    st = Perturb(sigma={"x": 0.5, "y": 0.25})
    from crowd_v15.model import layer_sigma

    assert layer_sigma(st, "x") == 0.5
    assert layer_sigma(st, "y") == 0.25
    assert layer_sigma(st, "missing") == 0.0
    assert layer_sigma(Perturb(sigma=0.125), "any") == 0.125
