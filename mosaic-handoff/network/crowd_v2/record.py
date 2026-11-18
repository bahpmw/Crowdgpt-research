"""Hash-chained, co-signed, append-only record (JSONL).

Line: ``{"n","type","round","prev","body","hash","sigs"}`` with
``hash = sha256(canonical({n,type,round,prev,body}))``. Signatures are over the
hash and are not part of it, so late co-signatures can be added without
changing the chain.

mvp: adding a late co-signature (or dropping a provisional fork suffix during
catch-up) rewrites the file atomically; entries themselves are never edited.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .crypto import ZERO_HASH, canonical, sha256_hex, verify

log = logging.getLogger("crowd_v2.record")

ENTRY_TYPES = {"genesis", "register", "round", "verdict", "rollback",
               "orchestrator_down", "orchestrator_up"}


def entry_hash(e: dict) -> str:
    return sha256_hex(canonical({k: e[k] for k in ("n", "type", "round", "prev", "body")}))


def chain_entries(prev: str, start_n: int, items: list[tuple[str, int, dict]]) -> list[dict]:
    """Link ``(type, round, body)`` items onto ``prev``; returns unsigned entries."""
    out = []
    for i, (typ, rnd, body) in enumerate(items):
        if typ not in ENTRY_TYPES:
            raise ValueError(f"unknown entry type {typ}")
        e = {"n": start_n + i, "type": typ, "round": rnd, "prev": prev, "body": body}
        e["hash"] = entry_hash(e)
        e["sigs"] = {}
        prev = e["hash"]
        out.append(e)
    return out


def valid_sigs(e: dict, keys: dict[str, str]) -> set[str]:
    """Key ids (from ``keys``: kid -> pub) whose signature on the entry hash verifies."""
    return {kid for kid, sig in e.get("sigs", {}).items()
            if kid in keys and isinstance(sig, str) and verify(keys[kid], e["hash"].encode(), sig)}


class Record:
    def __init__(self, path: Path, keys: dict[str, str], quorum: int):
        self.path, self.keys, self.quorum = path, keys, quorum
        self.entries: list[dict] = []
        if path.exists():
            self.entries = [json.loads(line) for line in path.read_text().splitlines() if line]

    @property
    def tail_hash(self) -> str:
        return self.entries[-1]["hash"] if self.entries else ZERO_HASH

    @property
    def next_n(self) -> int:
        return len(self.entries)

    def is_final(self, e: dict) -> bool:
        return len(valid_sigs(e, self.keys)) >= self.quorum

    def last_final(self) -> dict | None:
        return next((e for e in reversed(self.entries) if self.is_final(e)), None)

    def append(self, e: dict) -> None:
        if e["n"] != self.next_n or e["prev"] != self.tail_hash or entry_hash(e) != e["hash"]:
            raise ValueError(f"entry {e.get('n')} does not extend the chain")
        if not valid_sigs(e, self.keys):
            raise ValueError(f"entry {e['n']} carries no valid orchestrator signature")
        self.entries.append(e)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(e, sort_keys=True) + "\n")

    def add_sigs(self, n: int, sigs: dict[str, str]) -> bool:
        """Merge valid signatures into entry ``n``; True if anything was added."""
        e = self.entries[n]
        new = {k: v for k, v in sigs.items() if k not in e["sigs"]}
        good = valid_sigs({"hash": e["hash"], "sigs": new}, self.keys)
        if not good:
            return False
        e["sigs"].update({k: new[k] for k in good})
        self._rewrite()
        return True

    def truncate(self, n: int) -> None:
        log.warning(json.dumps({"msg": "record truncated", "from": n}))
        self.entries = self.entries[:n]
        self._rewrite()

    def _rewrite(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in self.entries))
        os.replace(tmp, self.path)


def verify_chain(path: Path, genesis_keys: dict[str, str]) -> dict:
    """Check hashes, links and signatures; count final vs provisional entries."""
    lines = [line for line in path.read_text().splitlines() if line]
    prev, final, provisional, errors = ZERO_HASH, 0, 0, []
    quorum = None
    for i, line in enumerate(lines):
        e = json.loads(line)
        if e.get("n") != i:
            errors.append(f"entry {i}: wrong index {e.get('n')}")
        if e.get("prev") != prev:
            errors.append(f"entry {i}: broken link")
        if entry_hash(e) != e.get("hash"):
            errors.append(f"entry {i}: hash mismatch")
        if i == 0:
            if e.get("type") != "genesis" or e["body"].get("orchestrators") != genesis_keys:
                errors.append("entry 0: not the expected genesis")
            quorum = int(e["body"].get("quorum", len(genesis_keys)))
        sigs = valid_sigs(e, genesis_keys)
        if len(sigs) != len(e.get("sigs", {})):
            errors.append(f"entry {i}: invalid signature present")
        if not sigs:
            errors.append(f"entry {i}: unsigned")
        if quorum is not None and len(sigs) >= quorum:
            final += 1
        else:
            provisional += 1
        prev = e.get("hash")
    return {"ok": not errors, "entries": len(lines), "final": final,
            "provisional": provisional, "errors": errors[:20]}
