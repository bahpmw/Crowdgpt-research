"""Chain-derived state and the deterministic round close.

``close_round`` is the single code path that turns a round's public inputs
(signed registrations, signed commits, payloads, signed audits) into the batch
of record entries and the new weights. The proposer runs it to build a
proposal; every peer (and every catching-up orchestrator) runs it again on the
same inputs and co-signs only if the entries hash identically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from . import audit, beacon
from .crypto import canonical, hash_tensors, merkle_root, open_envelope, sha256_hex
from .merge import gate, merge_module, outer_step
from .nets import Hyper, Tokens, eval_loss

PROBATION_ROUNDS = 3
MAX_STRIKES = 3
# Not provably the contributor's fault (views can differ after a failover): excluded, no strike.
NO_STRIKE = {"stale_beacon", "mismatch:assignment", "missing_payload", "missing_audit"}


@dataclass
class Params:
    """Network parameters fixed in the genesis entry."""
    hp: Hyper
    orchestrators: dict[str, str]
    quorum: int
    p_audit: float
    thresholds: audit.Thresholds
    shapes: list[dict[str, tuple[int, ...]]]

    @classmethod
    def from_genesis(cls, body: dict) -> "Params":
        cfg = body["config"]
        return cls(hp=Hyper(**cfg["hyper"]), orchestrators=body["orchestrators"],
                   quorum=int(body["quorum"]), p_audit=float(cfg["p_audit"]),
                   thresholds=audit.Thresholds(**cfg["thresholds"]),
                   shapes=[{n: tuple(s) for n, s in m.items()} for m in cfg["shapes"]])

    @property
    def tokens_per_cell(self) -> int:
        return self.hp.H * self.hp.batch * self.hp.seq


@dataclass
class ChainState:
    registered: dict[str, str] = field(default_factory=dict)
    strikes: dict[str, int] = field(default_factory=dict)
    banned: set[str] = field(default_factory=set)
    cells_seen: dict[str, int] = field(default_factory=dict)
    payloads: set[str] = field(default_factory=set)
    weights: dict[int, str] = field(default_factory=dict)
    last_loss: float | None = None
    round: int = 0
    down: set[str] = field(default_factory=set)

    @classmethod
    def from_entries(cls, entries: list[dict]) -> "ChainState":
        st = cls()
        for e in entries:
            st.apply(e)
        return st

    def apply(self, e: dict) -> None:
        b = e["body"]
        if e["type"] == "genesis":
            self.weights = {int(k): v for k, v in b["weights"].items()}
            self.last_loss = b["eval_loss"]
        elif e["type"] == "register":
            self.registered[b["key"]] = b["pub"]
        elif e["type"] == "verdict":
            self.strikes[b["key"]] = b["strikes"]
            if b["banned"]:
                self.banned.add(b["key"])
        elif e["type"] == "orchestrator_down":
            self.down.add(b["orchestrator"])
        elif e["type"] == "orchestrator_up":
            self.down.discard(b["orchestrator"])
        elif e["type"] == "round":
            for c in b["cells"]:
                self.cells_seen[c["key"]] = self.cells_seen.get(c["key"], 0) + 1
                self.payloads.add(c["payload_hash"])
            self.weights = {int(k): v for k, v in b["weights"].items()}
            self.last_loss = b["eval_loss"]
            self.round = e["round"]


@dataclass
class Closed:
    items: list[tuple[str, int, dict]]
    weights: dict[str, torch.Tensor]
    changed: list[int]


def _opened(env: dict, expect: str) -> tuple[str, dict] | None:
    try:
        kid, body = open_envelope(env)
    except ValueError:
        return None
    return (kid, body) if body.get("type") == expect else None


def close_round(p: Params, st: ChainState, weights: dict[str, torch.Tensor], inputs: dict,
                get_payload: Callable[[str], dict[str, torch.Tensor] | None],
                tokens: Tokens) -> Closed:
    r, bcn = inputs["round"], inputs["beacon"]
    items: list[tuple[str, int, dict]] = []
    registered = dict(st.registered)
    for env in inputs["registrations"]:
        got = _opened(env, "register")
        if got and got[0] not in registered:
            registered[got[0]] = env["pub"]
            items.append(("register", r, {"key": got[0], "pub": env["pub"], "sig": env["sig"],
                                          "signed": got[1]}))
    items += [("orchestrator_down", r, {"orchestrator": o}) for o in sorted(inputs["down"])]
    items += [("orchestrator_up", r, {"orchestrator": o}) for o in sorted(inputs["up"])]
    down = (st.down | set(inputs["down"])) - set(inputs["up"])
    lone = len(set(p.orchestrators) - down) <= 1

    people = sorted(k for k in set(inputs["participants"]) if k in registered and k not in st.banned)
    assignment = beacon.assign(bcn, people, p.hp.K)
    commits: dict[str, dict] = {}
    status: dict[str, str] = {}
    for kid, env in sorted(inputs["commits"].items()):
        got = _opened(env, "commit")
        if not got or got[0] != kid or kid not in assignment or got[1].get("round") != r:
            continue
        commits[kid] = got[1]
        b = got[1]
        if b.get("beacon") != bcn:
            status[kid] = "stale_beacon"
        elif b.get("module") != assignment[kid] or b.get("data_seed") != beacon.data_seed(bcn, kid):
            status[kid] = "mismatch:assignment"
        elif b.get("tokens") != p.tokens_per_cell:
            status[kid] = "layer0:tokens"
    root = merkle_root([sha256_hex(canonical(inputs["commits"][k])) for k in commits])
    aseed = beacon.audit_seed(bcn, root)

    msgs: dict[str, dict[str, torch.Tensor]] = {}
    seen_payloads: set[str] = set()
    for kid, b in commits.items():
        if kid in status:
            continue
        ph = b.get("payload_hash", "")
        if ph in st.payloads or ph in seen_payloads:
            status[kid] = "layer0:replayed_payload"
            continue
        seen_payloads.add(ph)
        msg = get_payload(ph)
        if msg is None:
            status[kid] = "missing_payload"
            continue
        bad = audit.layer0(msg, p.shapes[b["module"]])
        if bad:
            status[kid] = bad
        else:
            msgs[kid] = msg

    sampled, audits = [], []
    for kid in sorted(msgs):
        prob = 1.0 if st.cells_seen.get(kid, 0) < PROBATION_ROUNDS else p.p_audit
        if not beacon.sampled(aseed, kid, prob):
            continue
        sampled.append(kid)
        env = inputs["audits"].get(kid)
        got = _opened(env, "audit") if env else None
        a = got[1] if got else {}
        valid = (got is not None and got[0] in p.orchestrators
                 and (got[0] != inputs["via"].get(kid) or lone)
                 and a.get("round") == r and a.get("key") == kid
                 and a.get("payload_hash") == commits[kid]["payload_hash"])
        if not valid:
            status[kid] = "missing_audit"
            continue
        ok = p.thresholds.passes(float(a["cos"]), float(a["ratio"]))
        audits.append({"key": kid, "auditor": got[0], "cos": a["cos"], "ratio": a["ratio"],
                       "ok": ok, "sig": env["sig"]})
        if not ok:
            status[kid] = "audit:replay"

    for k in range(p.hp.K):
        mod = {kid: m for kid, m in msgs.items() if kid not in status and commits[kid]["module"] == k}
        status.update(audit.layer3(mod, {x["key"] for x in audits if x["ok"]}))

    new_weights, changed, merge_stats = dict(weights), [], {}
    for k in range(p.hp.K):
        acc = {commits[kid]["payload_hash"]: m for kid, m in msgs.items()
               if kid not in status and commits[kid]["module"] == k}
        if acc:
            avg, merge_stats[str(k)] = merge_module(acc)
            new_weights = outer_step(new_weights, avg, p.hp.alpha)
            changed.append(k)
    loss = round(eval_loss(p.hp, new_weights, tokens), 6) if changed else st.last_loss
    rolled = bool(changed) and gate(st.last_loss, loss, p.hp.gate_tol)
    hashes = {str(k): st.weights[k] for k in range(p.hp.K)}
    if rolled:
        items.append(("rollback", r, {"rejected_loss": loss, "kept_loss": st.last_loss}))
        new_weights, changed, loss = weights, [], st.last_loss
    for k in changed:
        hashes[str(k)] = hash_tensors({n: new_weights[n] for n in p.shapes[k]})

    for kid in sorted(status):
        if status[kid] in NO_STRIKE:
            continue
        strikes = st.strikes.get(kid, 0) + 1
        items.append(("verdict", r, {"key": kid, "reason": status[kid], "strikes": strikes,
                                     "banned": strikes >= MAX_STRIKES}))
    cells = [{"key": kid, "module": b.get("module"), "payload_hash": b.get("payload_hash"),
              "via": inputs["via"].get(kid), "status": status.get(kid, "accepted")}
             for kid, b in sorted(commits.items())]
    accepted = sum(1 for c in cells if c["status"] == "accepted")
    items.append(("round", r, {
        "inputs": inputs, "leader": inputs["leader"], "beacon": bcn, "assignment": assignment,
        "commit_root": root, "audit_seed": aseed, "sampled": sampled, "audits": audits,
        "cells": cells, "weights": hashes, "eval_loss": loss, "prev_loss": st.last_loss,
        "rolled_back": rolled, "tokens": 0 if rolled else accepted * p.tokens_per_cell,
        "merge": merge_stats}))
    return Closed(items=items, weights=new_weights, changed=changed)
