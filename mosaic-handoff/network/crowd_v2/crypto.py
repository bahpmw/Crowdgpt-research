"""Ed25519 identities, canonical JSON, signed envelopes and tensor hashes.

Identity always comes from a verified signature (``key_id(pub)``), never from a
payload field.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ZERO_HASH = "0" * 64


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8") if isinstance(p, str) else p)
    return h.hexdigest()


def key_id(pub_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()[:16]


class Identity:
    def __init__(self, private_key: Ed25519PrivateKey):
        self._key = private_key
        self.pub = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
        self.kid = key_id(self.pub)

    @classmethod
    def generate(cls) -> "Identity":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path: Path) -> "Identity":
        if path.exists():
            raw = bytes.fromhex(path.read_text().strip())
            return cls(Ed25519PrivateKey.from_private_bytes(raw))
        ident = cls.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ident._key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                       serialization.NoEncryption())
        path.write_text(raw.hex())
        path.chmod(0o600)
        return ident

    def sign(self, data: bytes) -> str:
        return self._key.sign(data).hex()

    def envelope(self, body: dict) -> dict:
        """A signed message: the signer is whoever owns ``pub``."""
        return {"v": 1, "pub": self.pub, "body": body, "sig": self.sign(canonical(body))}


def verify(pub_hex: str, data: bytes, sig_hex: str) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex)).verify(bytes.fromhex(sig_hex), data)
    except (InvalidSignature, ValueError):
        return False
    return True


def open_envelope(env: dict) -> tuple[str, dict]:
    """Return ``(key_id, body)`` of a valid envelope or raise ``ValueError``."""
    pub, body, sig = env.get("pub"), env.get("body"), env.get("sig")
    if not isinstance(pub, str) or not isinstance(body, dict) or not isinstance(sig, str):
        raise ValueError("malformed envelope")
    if not verify(pub, canonical(body), sig):
        raise ValueError("bad signature")
    return key_id(pub), body


def hash_tensors(tensors: dict[str, torch.Tensor]) -> str:
    """sha256 over exact fp32 little-endian bytes in sorted-name order (names included)."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().to("cpu", torch.float32).contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().astype("<f4", copy=False).tobytes())
    return h.hexdigest()


def seed_int(*parts: str) -> int:
    """A 63-bit integer derived from a hash, usable as a numpy seed."""
    return int(sha256_hex(*parts), 16) % (2 ** 63)


def unit_float(*parts: str) -> float:
    return int(sha256_hex(*parts), 16) / float(2 ** 256)


def merkle_root(leaves: list[str]) -> str:
    level = sorted(leaves) or [ZERO_HASH]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha256_hex(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]
