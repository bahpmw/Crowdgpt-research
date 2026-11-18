"""Gradient containers for the Crowd-v1.5 wire.

A DUST client does not ship weights; it ships an *estimate* of the descent
direction, ``G_W = (1/sigma) sum_t (r_t a_t) x_t^T``.  That estimate is dense in
parameter space, so it needs a container.  Two are supported:

``ngd1``
    one sign plane plus an fp16 mean-absolute scale per 1024 values.  Preserves
    the sign of every gradient entry, which is the part the update actually uses,
    and keeps one magnitude per block.
``ngd1r``
    the same with a second plane over the leftover, so magnitude error inside a
    block is halved at twice the bytes.
``dense``
    raw little-endian bf16, for parity checks and for clients whose accelerator
    cannot afford the encode.

Layouts (all little-endian, all zlib-compressed per frame)::

    NGD1   magic | u64 count | u64 block(1024) | fp16 scale * nb | sign bits
    NGD1R  magic | u64 count | u64 block(1024) | fp16 scale * nb
                                             | fp16 residual scale * nb
                                             | residual sign bits | sign bits
    NGD2   "NGD2" | u64 total | u64 frames | (u32 length | frame)*
    NGD3   "NGD3" | u64 total | u64 frames | (u32 length | frame)*

``NGD3`` carries ``NGD1R`` frames.  The decoder sniffs the magic, so one
function reads every combination.

This is deliberately the same layout that the v1 server already decodes
(``orchestrator/compression.py``).  Reusing it keeps the quantisation error and
the abuse surface already understood, and lets a v1.5 coordinator accept a v1
client's delta with no new code path.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

BLOCK = 1024
MAGIC = b"NGD1"
MAGIC_RESIDUAL = b"NGD1R"
FRAME_MAGIC = b"NGD2"
FRAME_MAGIC_RESIDUAL = b"NGD3"

_DELTA_HEADER = len(MAGIC) + 12
_DELTA_HEADER_RESIDUAL = len(MAGIC_RESIDUAL) + 12

CODECS = ("dense", "ngd1", "ngd1r")


def _plane_bytes(nb: int, block: int) -> int:
    return (nb * block + 7) // 8


def _expected_raw_size(n: int, block: int, residual: bool = False) -> int:
    nb = (n + block - 1) // block
    if residual:
        return _DELTA_HEADER_RESIDUAL + 4 * nb + 2 * _plane_bytes(nb, block)
    return _DELTA_HEADER + 2 * nb + _plane_bytes(nb, block)


def _block_stats(matrix: np.ndarray, counts: np.ndarray):
    scales = (np.abs(matrix).sum(axis=1) / counts).astype(np.float16)
    bits = (matrix >= 0).astype(np.uint8)
    packed = np.packbits(bits.reshape(-1), bitorder="little").tobytes()
    return scales, bits, packed


def compress_frame(values, *, block: int = BLOCK, level: int = 6, residual: bool = False) -> bytes:
    """Encode one frame. ``residual`` adds the second bit plane."""
    if block <= 0:
        raise ValueError("block must be positive")
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(arr.size)
    nb = (n + block - 1) // block
    padded = np.zeros(nb * block, dtype=np.float32)
    padded[:n] = arr
    matrix = padded.reshape(nb, block) if nb else np.zeros((0, block), dtype=np.float32)
    if nb:
        # The final block is zero padded; divide by its real length so a partial
        # block keeps the right mean-absolute scale.
        counts = np.full(nb, block, dtype=np.float64)
        counts[-1] = n - (nb - 1) * block
        scales, bits, packed = _block_stats(matrix, counts)
    else:
        counts = np.zeros(0, dtype=np.float64)
        scales, bits, packed = np.zeros(0, dtype=np.float16), np.zeros((0, block), np.uint8), b""
    if not residual:
        raw = MAGIC + struct.pack("<QI", n, block) + scales.tobytes() + packed
        return zlib.compress(raw, level)
    if nb:
        recon = np.repeat(scales.astype(np.float32)[:, None], block, axis=1)
        recon *= np.where(bits == 1, np.float32(1.0), np.float32(-1.0))
        leftover = matrix - recon
        del recon
        if n < nb * block:
            # Padding reconstructs to +/-scale, not to zero, so it would swamp
            # the residual scale of the last block.
            leftover = leftover.reshape(-1)
            leftover[n:] = 0.0
            leftover = leftover.reshape(nb, block)
        rscales, _, rpacked = _block_stats(leftover, counts)
        del leftover
    else:
        rscales, rpacked = np.zeros(0, dtype=np.float16), b""
    raw = (MAGIC_RESIDUAL + struct.pack("<QI", n, block) + scales.tobytes()
           + rscales.tobytes() + rpacked + packed)
    return zlib.compress(raw, level)


def decompress_frame(blob: bytes, *, max_raw_bytes: int | None = None) -> np.ndarray:
    """Decode one frame, plain or residual; the magic selects the layout."""
    try:
        inflater = zlib.decompressobj()
        head = inflater.decompress(blob, _DELTA_HEADER_RESIDUAL)
        if len(head) >= len(MAGIC_RESIDUAL) and head.startswith(MAGIC_RESIDUAL):
            residual, magic_len = True, len(MAGIC_RESIDUAL)
        elif len(head) >= len(MAGIC) and head.startswith(MAGIC):
            residual, magic_len = False, len(MAGIC)
        elif len(head) < len(MAGIC):
            raise ValueError("truncated frame header")
        else:
            raise ValueError("bad frame magic")
        header = magic_len + 12
        if len(head) < header:
            raise ValueError("truncated frame header")
        n, block = struct.unpack_from("<QI", head, magic_len)
        if block <= 0:
            raise ValueError(f"invalid block size {block}")
        expected = _expected_raw_size(n, block, residual)
        if max_raw_bytes is not None and expected > max_raw_bytes:
            raise ValueError(f"frame too large: {expected} > {max_raw_bytes} bytes")
        raw = head
        if len(raw) < expected:
            raw += inflater.decompress(inflater.unconsumed_tail, expected - len(raw) + 1)
        if len(raw) != expected:
            raise ValueError("truncated or oversized frame")
        if inflater.unconsumed_tail:
            raise ValueError("oversized frame")
        if inflater.unused_data:
            raise ValueError("trailing bytes in frame")
        if not inflater.eof:
            raise ValueError("truncated frame stream")
    except (zlib.error, struct.error) as exc:
        raise ValueError(f"bad frame stream: {exc}") from exc

    nb = (n + block - 1) // block
    pos = header
    scales = np.frombuffer(raw, dtype=np.float16, count=nb, offset=pos)
    pos += nb * 2
    if not np.all(np.isfinite(scales)):
        raise ValueError("non-finite frame scale")
    if not residual:
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8, offset=pos), bitorder="little")[: nb * block]
        out = np.where(bits == 1, scales.repeat(block), -scales.repeat(block))[:n]
        return out.astype(np.float32)

    plane = _plane_bytes(nb, block)
    rscales = np.frombuffer(raw, dtype=np.float16, count=nb, offset=pos)
    pos += nb * 2
    if not np.all(np.isfinite(rscales)):
        raise ValueError("non-finite residual scale")
    rbits = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8, count=plane, offset=pos), bitorder="little"
    )[: nb * block]
    pos += plane
    bits = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8, count=plane, offset=pos), bitorder="little"
    )[: nb * block]
    first = np.where(bits == 1, scales.repeat(block), -scales.repeat(block))
    second = np.where(rbits == 1, rscales.repeat(block), -rscales.repeat(block))
    return (first + second)[:n].astype(np.float32)


def pack_frames(blocks, *, residual: bool = False) -> bytes:
    frames = [compress_frame(block, residual=residual) for block in blocks]
    total = sum(int(np.asarray(block).size) for block in blocks)
    parts = [FRAME_MAGIC_RESIDUAL if residual else FRAME_MAGIC, struct.pack("<QI", total, len(frames))]
    for frame in frames:
        parts.append(struct.pack("<I", len(frame)))
        parts.append(frame)
    return b"".join(parts)


def container_is_residual(blob: bytes) -> bool:
    if blob.startswith(FRAME_MAGIC_RESIDUAL):
        return True
    if blob.startswith(FRAME_MAGIC):
        return False
    raise ValueError("bad container magic")


def iter_frames(blob: bytes, *, max_total_params: int | None = None, expect_residual: bool | None = None):
    """Yield decoded blocks; raises on a truncated, oversized or trailing container."""
    residual = container_is_residual(blob)
    if expect_residual is not None and residual != expect_residual:
        raise ValueError(
            f"container is {'NGD3' if residual else 'NGD2'}, "
            f"expected {'NGD3' if expect_residual else 'NGD2'}"
        )
    pos = len(FRAME_MAGIC)
    if len(blob) < pos + 12:
        raise ValueError("truncated container header")
    total, count = struct.unpack_from("<QI", blob, pos)
    pos += 12
    if max_total_params is not None and total > max_total_params:
        raise ValueError(f"container total {total} exceeds limit {max_total_params}")
    if count > (len(blob) - pos) // 4:
        raise ValueError("truncated container header")
    seen = 0
    for _ in range(count):
        if len(blob) < pos + 4:
            raise ValueError("truncated frame header")
        (size,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        if size <= 0:
            raise ValueError("empty frame")
        frame = blob[pos:pos + size]
        if len(frame) != size:
            raise ValueError("truncated frame")
        limit = None
        if max_total_params is not None:
            limit = _expected_raw_size(max(0, max_total_params - seen), BLOCK, True)
        block = decompress_frame(frame, max_raw_bytes=limit)
        seen += int(block.size)
        if max_total_params is not None and seen > max_total_params:
            raise ValueError(f"container total exceeds limit {max_total_params}")
        yield block
        pos += size
    if pos != len(blob):
        raise ValueError("trailing bytes after frames")
    if seen != total:
        raise ValueError(f"container total mismatch: {seen} != {total}")


def compression_ratio(original_bytes: int, compressed: bytes) -> float:
    return float(original_bytes) / max(1, len(compressed))


def _f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (round-to-nearest-even) without torch.

    The arithmetic stays in uint32. Upcasting to uint64 (the obvious way to
    avoid the carry out of the low half) allocates four 8-byte temporaries per
    element, which at 306M parameters is ~10 GB of traffic per call and made a
    coordinator digest take minutes. The carry into the exponent is exactly
    what rounding should do, so uint32 is both correct and ~6x cheaper.
    """
    u = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    carry = (u >> np.uint32(16)) & np.uint32(1)
    rounded = (u + np.uint32(0x8000) + carry) >> np.uint32(16)
    return (rounded & np.uint32(0xFFFF)).astype("<u2")


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def encode_dense(values) -> bytes:
    """Raw little-endian bf16, no framing (the ``dense`` codec)."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return _f32_to_bf16(arr).tobytes()


def decode_dense(blob: bytes, count: int) -> np.ndarray:
    if len(blob) != int(count) * 2:
        raise ValueError(f"dense body is {len(blob)} bytes, expected {int(count) * 2}")
    return _bf16_to_f32(np.frombuffer(blob, dtype="<u2", count=int(count))).astype(np.float32)


__all__ = [
    "BLOCK", "CODECS", "FRAME_MAGIC", "FRAME_MAGIC_RESIDUAL", "MAGIC", "MAGIC_RESIDUAL",
    "compress_frame", "compression_ratio", "container_is_residual", "decode_dense",
    "decompress_frame", "encode_dense", "iter_frames", "pack_frames",
]
