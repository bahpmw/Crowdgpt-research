"""Public randomness: beacon, cell assignment, data seeds, audit sampling, leader order.

``beacon_r = sha256("beacon" || hash of the last final entry || r)``.
Deviation from the spec: the round number is mixed in so that a lone
orchestrator writing only provisional entries does not repeat the same data
every round.

mvp: the beacon is record-derived. The orchestrator whose entry becomes the
last final one can grind (try many proposals) to bias the next beacon and so
which data and audits come next. A VRF or drand-style external beacon closes
this in v2.
"""

from __future__ import annotations

from .crypto import seed_int, sha256_hex, unit_float


def beacon(src_hash: str, rnd: int) -> str:
    return sha256_hex("beacon", src_hash, str(rnd))


def assign(beacon_hex: str, key_ids: list[str], k_modules: int) -> dict[str, int]:
    """Deal clients round-robin onto modules after a beacon-keyed shuffle."""
    order = sorted(set(key_ids), key=lambda kid: sha256_hex(beacon_hex, kid))
    return {kid: i % k_modules for i, kid in enumerate(order)}


def data_seed(beacon_hex: str, kid: str) -> int:
    return seed_int(beacon_hex, kid, "data")


def audit_seed(beacon_hex: str, commit_root: str) -> str:
    return sha256_hex(beacon_hex, commit_root)


def sampled(seed_hex: str, kid: str, p: float) -> bool:
    return unit_float(seed_hex, kid, "audit") < p


def auditor_order(seed_hex: str, kid: str, orchestrators: list[str], receiver: str) -> list[str]:
    """Orchestrators other than the receiver, in audit-seed order."""
    others = [o for o in orchestrators if o != receiver]
    return sorted(others, key=lambda o: sha256_hex(seed_hex, kid, o))


def leader_order(beacon_hex: str, orchestrators: list[str]) -> list[str]:
    return sorted(orchestrators, key=lambda o: sha256_hex(beacon_hex, o))
