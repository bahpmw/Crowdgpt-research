"""Tensor dicts on the wire: safetensors bytes, base64 in JSON, content hashes."""

from __future__ import annotations

import base64

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from .crypto import hash_tensors

MAX_BLOB = 64 * 1024 * 1024


def to_bytes(tensors: dict[str, torch.Tensor]) -> bytes:
    return st_save({n: t.detach().to("cpu").contiguous() for n, t in tensors.items()})


def from_bytes(data: bytes) -> dict[str, torch.Tensor]:
    if len(data) > MAX_BLOB:
        raise ValueError("blob too large")
    return dict(st_load(data))


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def blob_id(data: bytes) -> str:
    """Content address of a tensor blob: ``hash_tensors`` of what it decodes to."""
    return hash_tensors(from_bytes(data))
