"""Content-addressed blob store on disk (weights per module and payloads)."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .wire import blob_id, from_bytes, to_bytes


class BlobStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, h: str) -> Path:
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError("bad blob id")
        return self.root / h

    def put_bytes(self, data: bytes) -> str:
        h = blob_id(data)
        path = self._path(h)
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return h

    def put(self, tensors: dict[str, torch.Tensor]) -> str:
        return self.put_bytes(to_bytes(tensors))

    def get_bytes(self, h: str) -> bytes | None:
        path = self._path(h)
        return path.read_bytes() if path.exists() else None

    def get(self, h: str) -> dict[str, torch.Tensor] | None:
        data = self.get_bytes(h)
        return from_bytes(data) if data is not None else None
