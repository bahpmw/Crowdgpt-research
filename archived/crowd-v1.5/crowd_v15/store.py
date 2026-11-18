"""On-disk state for a v1.5 coordinator.

Weights live in one safetensors file plus a ``status.json`` that a browser can
read without touching the binary.  Both are written atomically so a crash mid
round leaves the previous consistent pair in place -- a coordinator that serves
half-updated weights teaches every client a different model.

The digest function is the interesting one.  ``baseSha256`` in the upload header
binds a gradient to the weights it was measured from, so the client and the
coordinator must compute the *same* digest from the *same* tensors on different
machines.  It is therefore defined over a canonical serialisation: names sorted,
float32 rounded to bfloat16 (the dtype the model is trained in), concatenated.
A digest over the safetensors file itself would differ for any container
change -- a re-save, a compression switch, a different writer.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

WEIGHTS = "weights.safetensors"
STATUS = "status.json"
# Download pieces for large models. Each shard is a byte range of the
# safetensors file cut at tensor boundaries, so a client can fetch, verify and
# retry them one at a time instead of holding one multi-GB response open.
SHARD_BYTES = 256 << 20
SUBMISSIONS = "submissions.jsonl"


def weights_digest(tensors: dict[str, Any]) -> str:
    from .codec import _f32_to_bf16

    digest = hashlib.sha256()
    for name in sorted(tensors):
        arr = np.ascontiguousarray(np.asarray(tensors[name], dtype=np.float32).reshape(-1))
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_f32_to_bf16(arr).tobytes())
    return digest.hexdigest()


class StoreError(RuntimeError):
    """The on-disk state is missing or inconsistent."""


class ModelStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.root / WEIGHTS
        self.status_path = self.root / STATUS
        self.log_path = self.root / SUBMISSIONS
        self.shard_bytes = SHARD_BYTES

    # ---- weights ---------------------------------------------------------

    def exists(self) -> bool:
        return self.weights_path.is_file() and self.status_path.is_file()

    def load_weights(self) -> dict[str, np.ndarray]:
        if not self.weights_path.is_file():
            raise StoreError(f"no weights at {self.weights_path}")
        from safetensors.numpy import load_file

        return {k: v.astype(np.float32, copy=False) for k, v in load_file(str(self.weights_path)).items()}

    def save_weights(self, tensors: dict[str, np.ndarray]) -> str:
        """Atomically replace the weights; returns the new digest."""
        from safetensors.numpy import save_file

        clean = {
            k: np.ascontiguousarray(np.asarray(v, dtype=np.float32))
            for k, v in tensors.items()
        }
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".weights-", suffix=".safetensors")
        os.close(fd)
        try:
            save_file(clean, tmp)
            os.replace(tmp, self.weights_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return weights_digest(clean)

    def weights_bytes(self) -> bytes:
        if not self.weights_path.is_file():
            raise StoreError(f"no weights at {self.weights_path}")
        return self.weights_path.read_bytes()

    def weights_size(self) -> int:
        if not self.weights_path.is_file():
            raise StoreError(f"no weights at {self.weights_path}")
        return int(self.weights_path.stat().st_size)

    def weights_range(self, start: int, end: int) -> bytes:
        """Bytes ``[start, end)`` of the weight file."""
        size = self.weights_size()
        if not (0 <= start < end <= size):
            raise StoreError(f"range {start}-{end} outside 0-{size}")
        with open(self.weights_path, "rb") as fh:
            fh.seek(start)
            return fh.read(end - start)

    def tensor_layout(self) -> dict[str, dict[str, Any]]:
        """Name -> dtype, shape and absolute ``[start, end)`` byte offsets in the file.

        Read from the safetensors header alone, so it costs a few KB of I/O
        whatever the model size.
        """
        size = self.weights_size()
        with open(self.weights_path, "rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            if header_len <= 0 or 8 + header_len > size:
                raise StoreError("weight file has a corrupt safetensors header")
            header = json.loads(fh.read(header_len))
        base = 8 + header_len
        return {
            name: {"dtype": info["dtype"], "shape": list(info["shape"]),
                   "start": base + int(info["data_offsets"][0]),
                   "end": base + int(info["data_offsets"][1])}
            for name, info in header.items() if name != "__metadata__"
        }

    def shards(self, max_bytes: int | None = None) -> list[dict[str, Any]]:
        """Contiguous byte ranges covering the file, cut at tensor boundaries.

        Shard 0 starts at byte 0 and carries the safetensors header, so the
        shards concatenated in order are the file. Each has its own sha256.
        """
        stat = self.weights_path.stat() if self.weights_path.is_file() else None
        if stat is None:
            raise StoreError(f"no weights at {self.weights_path}")
        max_bytes = int(max_bytes or self.shard_bytes)
        key = (stat.st_mtime_ns, stat.st_size, max_bytes)
        cached = getattr(self, "_shard_cache", None)
        if cached and cached[0] == key:
            return cached[1]
        layout = sorted(self.tensor_layout().items(), key=lambda kv: kv[1]["start"])
        cuts: list[tuple[int, int, list[str]]] = []
        start, names = 0, []
        for name, info in layout:
            if names and info["end"] - start > max_bytes:
                cuts.append((start, info["start"], names))
                start, names = info["start"], []
            names.append(name)
        cuts.append((start, int(stat.st_size), names))
        shards = []
        with open(self.weights_path, "rb") as fh:
            for index, (lo, hi, members) in enumerate(cuts):
                digest = hashlib.sha256()
                fh.seek(lo)
                left = hi - lo
                while left:
                    chunk = fh.read(min(left, 8 << 20))
                    digest.update(chunk)
                    left -= len(chunk)
                shards.append({"index": index, "start": lo, "end": hi, "bytes": hi - lo,
                               "names": members, "sha256": digest.hexdigest()})
        self._shard_cache = (key, shards)
        return shards

    # ---- status ----------------------------------------------------------

    def load_status(self) -> dict[str, Any]:
        if not self.status_path.is_file():
            return {}
        with open(self.status_path, encoding="utf-8") as fh:
            return json.load(fh)

    def save_status(self, status: dict[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".status-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(status, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, self.status_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def manifest(self, model: dict[str, Any]) -> dict[str, Any]:
        """What a client needs before it can train: shapes, dtype and the base digest."""
        tensors = self.load_weights()
        return {
            "schema": "crowd-v15-model-1",
            "name": model.get("name", "crowd-v1.5-dust"),
            "route": model.get("route", "dust"),
            "dtype": "float32",
            "params": int(sum(int(np.asarray(t).size) for t in tensors.values())),
            "baseSha256": weights_digest(tensors),
            "bytes": int(self.weights_path.stat().st_size),
            "names": sorted(tensors),
            "sizes": [int(np.asarray(tensors[n]).size) for n in sorted(tensors)],
            "shapes": {n: list(np.asarray(tensors[n]).shape) for n in sorted(tensors)},
            # Ranged download plan: fetch ``Range: bytes=start-(end-1)`` of
            # /model/weights per shard, check its sha256, concatenate in order.
            "shards": [{k: v for k, v in shard.items() if k != "names"}
                       for shard in self.shards()],
        }

    # ---- audit trail -----------------------------------------------------

    def append_submission(self, record: dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["SHARD_BYTES", "STATUS", "SUBMISSIONS", "WEIGHTS", "ModelStore", "StoreError",
           "weights_digest"]
