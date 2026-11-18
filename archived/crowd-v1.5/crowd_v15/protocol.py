"""Wire format shared by the v1.5 client and coordinator.

Every upload is one body::

    <u32 header length><header JSON><container>

The header is JSON because it is the part a human debugs, and the container is
the NGD family from :mod:`crowd_v15.codec` because it is the part that costs
bandwidth.  A ``dense`` body skips the container entirely and is raw bf16.

The v1 protocol carried the same shape (4-byte length, JSON, then frames), so a
coordinator can keep one parser and one set of size guards.

Why a header at all
-------------------
A v1 delta was "the weights now".  A v1.5 gradient is "the direction I measured
from the weights you gave me", which is only meaningful together with:

``baseSha256``
    the digest of the weights the client trained from.  Without it a submission
    is unbound: the coordinator would apply a direction measured at one point in
    parameter space as if it were measured at another.
``route``
    ``dust`` or ``backprop``.  They are not interchangeable -- a backprop
    gradient is exact and a DUST estimate is noisy, so the coordinator weights
    and clamps them differently.
``tokens``/``population``/``sigma``/``gamma``
    the estimator's own record of how much evidence produced the number.  The
    coordinator rejects reports that cannot have produced a gradient (zero
    tokens, zero population, non-positive sigma).
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from . import codec

SCHEMA = "crowd-v15-submit-1"
ROUTES = ("dust", "backprop")
CLIENT_VERSION = "1.5.0"

HEADER_PREFIX = 4
MAX_HEADER_BYTES = 64 * 1024
FRAME_PIECE = 4_000_000  # values per frame, same as the v1 writer


class ProtocolError(ValueError):
    """A body that cannot be trusted enough to decode."""


def flatten(grads: dict[str, Any]) -> tuple[list[str], list[int], np.ndarray]:
    """Flatten a name -> tensor mapping into one canonical vector.

    Order is sorted by name so two clients that hold the same parameters produce
    the same layout without negotiating it.
    """
    import torch

    names = sorted(grads)
    sizes: list[int] = []
    pieces: list[np.ndarray] = []
    for name in names:
        tensor = grads[name].detach().reshape(-1).to("cpu", torch.float32)
        sizes.append(int(tensor.numel()))
        pieces.append(tensor.numpy())
    flat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return names, sizes, flat


def unflatten(vector: np.ndarray, names: list[str], sizes: list[int]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    pos = 0
    for name, size in zip(names, sizes):
        out[name] = np.asarray(vector[pos:pos + size], dtype=np.float32)
        pos += size
    if pos != int(np.asarray(vector).size):
        raise ProtocolError(f"vector has {np.asarray(vector).size} values, layout wants {pos}")
    return out


def build_header(
    *,
    route: str,
    client_id: str,
    round_id: int,
    base_sha256: str,
    names: list[str],
    sizes: list[int],
    codec_name: str,
    **extra: Any,
) -> dict[str, Any]:
    if route not in ROUTES:
        raise ProtocolError(f"unknown route {route!r}")
    if codec_name not in codec.CODECS:
        raise ProtocolError(f"unknown codec {codec_name!r}")
    header = {
        "schema": SCHEMA,
        "route": route,
        "clientId": str(client_id),
        "roundId": int(round_id),
        "baseSha256": str(base_sha256),
        "codec": codec_name,
        "names": list(names),
        "sizes": [int(s) for s in sizes],
        "params": int(sum(sizes)),
        "clientVersion": CLIENT_VERSION,
    }
    header.update({k: v for k, v in extra.items() if v is not None})
    return header


def encode_body(header: dict[str, Any], vector: np.ndarray, codec_name: str | None = None) -> bytes:
    name = codec_name or header["codec"]
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_HEADER_BYTES:
        raise ProtocolError(f"header is {len(raw)} bytes, limit {MAX_HEADER_BYTES}")
    if name == "dense":
        container = codec.encode_dense(vector)
    else:
        pieces = [
            vector[i:i + FRAME_PIECE] for i in range(0, int(vector.size), FRAME_PIECE)
        ] or [np.zeros(0, dtype=np.float32)]
        container = codec.pack_frames(pieces, residual=(name == "ngd1r"))
    return struct.pack("<I", len(raw)) + raw + container


def decode_body(
    body: bytes,
    *,
    max_params: int,
    max_body_bytes: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Parse one upload. Raises :class:`ProtocolError` on anything suspicious."""
    if max_body_bytes is not None and len(body) > max_body_bytes:
        raise ProtocolError(f"body is {len(body)} bytes, limit {max_body_bytes}")
    if len(body) < HEADER_PREFIX:
        raise ProtocolError("body too short for a header length")
    (header_len,) = struct.unpack_from("<I", body, 0)
    if header_len == 0 or header_len > MAX_HEADER_BYTES:
        raise ProtocolError(f"implausible header length {header_len}")
    if len(body) < HEADER_PREFIX + header_len:
        raise ProtocolError("body truncated inside the header")
    try:
        header = json.loads(body[HEADER_PREFIX:HEADER_PREFIX + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ProtocolError("header is not an object")
    _validate_header(header, max_params=max_params)

    container = body[HEADER_PREFIX + header_len:]
    name = header["codec"]
    total = int(header["params"])
    if name == "dense":
        try:
            vector = codec.decode_dense(container, total)
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc
    else:
        try:
            frames = list(
                codec.iter_frames(
                    container,
                    max_total_params=total,
                    expect_residual=(name == "ngd1r"),
                )
            )
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc
        vector = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        if int(vector.size) != total:
            raise ProtocolError(f"decoded {vector.size} values, header says {total}")
    if not np.all(np.isfinite(vector)):
        raise ProtocolError("gradient contains NaN or Inf")
    return header, vector


def _validate_header(header: dict[str, Any], *, max_params: int) -> None:
    if header.get("schema") != SCHEMA:
        raise ProtocolError(f"unexpected schema {header.get('schema')!r}")
    if header.get("route") not in ROUTES:
        raise ProtocolError(f"unknown route {header.get('route')!r}")
    if header.get("codec") not in codec.CODECS:
        raise ProtocolError(f"unknown codec {header.get('codec')!r}")
    names, sizes = header.get("names"), header.get("sizes")
    if not isinstance(names, list) or not isinstance(sizes, list) or len(names) != len(sizes):
        raise ProtocolError("names and sizes must be equal-length lists")
    if not all(isinstance(n, str) and n for n in names):
        raise ProtocolError("names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ProtocolError("duplicate parameter names")
    if not all(isinstance(s, int) and s > 0 for s in sizes):
        raise ProtocolError("sizes must be positive integers")
    total = sum(sizes)
    if total != int(header.get("params", -1)):
        raise ProtocolError(f"params {header.get('params')} != sum(sizes) {total}")
    if total > max_params:
        raise ProtocolError(f"{total} parameters exceeds the model ({max_params})")
    base = header.get("baseSha256")
    if not isinstance(base, str) or len(base) != 64:
        raise ProtocolError("baseSha256 must be a 64-character digest")
    if not isinstance(header.get("roundId"), int) or header["roundId"] < 0:
        raise ProtocolError("roundId must be a non-negative integer")
    if not isinstance(header.get("clientId"), str) or not header["clientId"]:
        raise ProtocolError("clientId is required")


def estimator_fields(header: dict[str, Any]) -> dict[str, float]:
    """Read the estimator record defensively; missing values become 0/NaN-free defaults."""
    def num(key: str, default: float) -> float:
        try:
            value = float(header.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    return {
        "tokens": num("tokens", 0.0),
        "population": num("population", 0.0),
        "sigma": num("sigma", 0.0),
        "gamma": num("gamma", 0.0),
    }


__all__ = [
    "CLIENT_VERSION", "FRAME_PIECE", "HEADER_PREFIX", "MAX_HEADER_BYTES", "ProtocolError",
    "ROUTES", "SCHEMA", "build_header", "decode_body", "encode_body", "estimator_fields",
    "flatten", "unflatten",
]
