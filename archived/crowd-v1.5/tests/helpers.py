"""Shared fixtures: a tiny model, its safetensors bytes and a submitted body."""

from __future__ import annotations

import io

import numpy as np
import torch

from crowd_v15 import protocol
from crowd_v15.model import GPT, ModelConfig, canonical_params


def tiny_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(ModelConfig.preset("tiny"))


def model_tensors(model: GPT | None = None) -> dict[str, np.ndarray]:
    model = model or tiny_model()
    return {
        name: param.detach().cpu().float().numpy().copy()
        for name, param in canonical_params(model).items()
    }


def safetensors_bytes(tensors: dict[str, np.ndarray]) -> bytes:
    from safetensors.numpy import save

    return save({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()})


def gradient_vector(tensors: dict[str, np.ndarray], seed: int = 1, scale: float = 1e-3) -> tuple[list[str], list[int], np.ndarray]:
    names = sorted(tensors)
    sizes = [int(np.asarray(tensors[n]).size) for n in names]
    rng = np.random.default_rng(seed)
    flat = (rng.standard_normal(sum(sizes)) * scale).astype(np.float32)
    return names, sizes, flat


def submit_body(
    names: list[str],
    sizes: list[int],
    vector: np.ndarray,
    *,
    route: str = "dust",
    codec_name: str = "ngd1r",
    client_id: str = "client-a",
    round_id: int = 0,
    base_sha256: str = "",
    tokens: int = 256,
    population: int = 8,
    sigma: float = 1e-3,
    gamma: float = 0.98,
) -> bytes:
    header = protocol.build_header(
        route=route,
        client_id=client_id,
        round_id=round_id,
        base_sha256=base_sha256,
        names=names,
        sizes=sizes,
        codec_name=codec_name,
        tokens=tokens,
        population=population,
        sigma=sigma,
        gamma=gamma,
    )
    return protocol.encode_body(header, vector)


def load_state_dict_roundtrip(model: GPT):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    return buf
