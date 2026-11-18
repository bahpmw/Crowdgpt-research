"""Apple-silicon (MPS) parity: the estimator on MPS is the estimator on CPU.

CPU-generated noise is device-independent, so in float32 an MPS step must give
the CPU step's gradient up to float reassociation. Skipped where MPS is absent.
"""

from __future__ import annotations

import pytest
import torch

from crowd_v15 import accel
from crowd_v15.dust import DustConfig, draw_noise, dust_gradients
from crowd_v15.model import GPT, ModelConfig
from crowd_v15.tasks import induction_batch

pytestmark = pytest.mark.skipif(not accel.mps_available(), reason="MPS not available")


def _flat_grads(device: str, cfg: DustConfig) -> torch.Tensor:
    model_cfg = ModelConfig.preset("tiny")
    torch.manual_seed(0)
    model = GPT(model_cfg).to(device)
    idx, tgt = induction_batch(2, 16, model_cfg.vocab_size, torch.device(device),
                               torch.Generator().manual_seed(0))
    grads, stats = dust_gradients(model, cfg, idx, tgt, base_seed=1, step=0)
    assert stats["tokens"] == cfg.population * 2 * 16
    return torch.cat([g.reshape(-1).cpu() for g in grads.values()])


def test_cpu_noise_reaches_mps_unchanged():
    shapes = GPT(ModelConfig.preset("tiny")).noise_shapes()
    kwargs = dict(rows=2, seq_len=4, base_seed=7, step=1, draws=[0, 3], dtype=torch.float32)
    cpu = draw_noise(shapes, device=torch.device("cpu"), **kwargs)
    mps = draw_noise(shapes, device=torch.device("mps"), **kwargs)
    for name in cpu:
        assert mps[name].device.type == "mps"
        assert torch.equal(cpu[name], mps[name].cpu()), name


@pytest.mark.parametrize("sigma_mode", ["absolute", "relative"])
def test_mps_estimate_matches_cpu_in_float32(sigma_mode):
    cfg = DustConfig(population=8, draw_chunk=4, sigma_mode=sigma_mode)
    cpu = _flat_grads("cpu", cfg)
    mps = _flat_grads("mps", cfg)
    assert torch.isfinite(mps).all()
    cosine = float(torch.nn.functional.cosine_similarity(cpu, mps, dim=0))
    assert cosine > 0.999, cosine
    assert torch.allclose(cpu.norm(), mps.norm(), rtol=1e-3)


def test_device_noise_runs_on_mps():
    cfg = DustConfig(population=4, draw_chunk=2, noise_device="device")
    grads = _flat_grads("mps", cfg)
    assert torch.isfinite(grads).all()
    assert float(grads.norm()) > 0.0


def test_auto_device_prefers_an_accelerator():
    assert accel.resolve_device("auto").type in ("cuda", "mps")
    assert accel.resolve_device("cpu").type == "cpu"
    assert accel.peak_memory_gb(torch.device("cpu")) is None
    assert accel.peak_memory_gb(torch.device("mps")) is not None
