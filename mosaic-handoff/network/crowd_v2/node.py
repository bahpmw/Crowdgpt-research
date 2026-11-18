"""Orchestrator node: client endpoints, round phases, audits and events.

Round phases on every orchestrator (each on its own clock; deadlines are
absolute epoch seconds so clients on the same machine can read them):
open (collect work requests) -> commit (signed commits; closes early when every
assigned cell committed) -> upload/audit (payloads, EF reveals for sampled
cells, replays by a non-receiving orchestrator) -> propose (leader in beacon
order; the next one after ``propose_timeout`` if the leader is silent).

mvp: deadlines assume one machine's clock; a real deployment needs clock-skew
margins. Reveals travel to all orchestrators, not only the assigned auditor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch
from pydantic import BaseModel, ValidationError

from . import beacon, status
from .audit import compare
from .blobs import BlobStore
from .cell import run_cell
from .chain import Chain
from .crypto import Identity, canonical, hash_tensors, key_id, merkle_root, open_envelope, sha256_hex
from .nets import Tokens
from .peers import Peers
from .record import Record, valid_sigs
from .rounds import PROBATION_ROUNDS, Params
from .schemas import (AuditBody, CommitBody, ForwardBody, RegisterBody, RevealBody, UploadBody,
                      WorkBody)
from .wire import blob_id, from_bytes, unb64

log = logging.getLogger("crowd_v2.node")
AUDIT_GRACE = 3.0     # seconds of peer heartbeats (1 s apart) before a receiver audits its own cell


class NodeConfig(BaseModel):
    port: int
    host: str = "127.0.0.1"
    key: Path
    genesis: Path
    init: Path
    data: Path
    peers: list[dict[str, str]] = []
    device: str = "cpu"


class Reject(Exception):
    """A client or peer request that fails validation; carries the HTTP status."""

    def __init__(self, code: int, reason: str):
        super().__init__(reason)
        self.code, self.reason = code, reason


@dataclass
class RoundState:
    r: int
    beacon: str
    beacon_src: int
    t_open: float
    deadlines: dict[str, float]
    weights: dict[str, torch.Tensor]
    work: set[str] = field(default_factory=set)
    assignment: dict[str, int] | None = None
    commits: dict[str, dict] = field(default_factory=dict)
    via: dict[str, str] = field(default_factory=dict)
    payloads: dict[str, str] = field(default_factory=dict)
    reveals: dict[str, bytes] = field(default_factory=dict)
    audits: dict[str, dict] = field(default_factory=dict)
    auditing: set[str] = field(default_factory=set)
    aseed: str | None = None
    sample: set[str] | None = None
    sample_t: float = 0.0
    phase: str = "open"


def parse(model: type[BaseModel], body: dict) -> BaseModel:
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        raise Reject(422, f"invalid {model.__name__}: {exc.errors()[0]['msg']}") from exc


class Node:
    def __init__(self, cfg: NodeConfig):
        self.cfg = cfg
        self.ident = Identity.load_or_create(cfg.key)
        self.kid = self.ident.kid
        genesis = json.loads(cfg.genesis.read_text())
        self.params = Params.from_genesis(genesis["body"])
        if self.kid not in self.params.orchestrators:
            raise RuntimeError("this key is not a genesis orchestrator")
        self.timing = genesis["body"]["config"]["timing"]
        self.genesis = genesis
        cfg.data.mkdir(parents=True, exist_ok=True)
        record = Record(cfg.data / "record.jsonl", self.params.orchestrators, self.params.quorum)
        blobs = BlobStore(cfg.data / "blobs")
        if not record.entries:
            self._bootstrap(record, blobs)
        self.events: deque[dict] = deque(maxlen=300)
        self.subscribers: set[asyncio.Queue] = set()
        urls = {key_id(p["pub"]): p["url"] for p in cfg.peers if key_id(p["pub"]) != self.kid}
        self.peers = Peers(self.ident, urls, self._peer_changed)
        self.tokens = Tokens(self.params.hp.vocab, self.params.hp.seq)
        self.chain = Chain(self.ident, self.params, record, blobs, self.tokens, self.peers, self.emit)
        self.pending_regs: dict[str, dict] = {}
        self.live: dict[str, dict] = {}
        self.rs: RoundState | None = None
        self.early: list[tuple[str, str, dict]] = []
        self.audit_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    def _bootstrap(self, record: Record, blobs: BlobStore) -> None:
        weights = from_bytes(self.cfg.init.read_bytes())
        for k, names in enumerate(self.params.shapes):
            h = blobs.put({n: weights[n] for n in names})
            if h != self.genesis["body"]["weights"][str(k)]:
                raise RuntimeError(f"initial weights do not match genesis for module {k}")
        if len(valid_sigs(self.genesis, self.params.orchestrators)) < len(self.params.orchestrators):
            raise RuntimeError("genesis is not signed by every orchestrator")
        record.append(self.genesis)

    # ---- events -------------------------------------------------------------
    def emit(self, typ: str, data: dict) -> None:
        ev = {"type": typ, "t": round(time.time(), 3), "node": self.kid, **data}
        self.events.append(ev)
        for q in list(self.subscribers):
            if q.qsize() < 500:
                q.put_nowait(ev)

    def _peer_changed(self, kid: str, alive: bool) -> None:
        self.emit("orchestrator_up" if alive else "orchestrator_down", {"orchestrator": kid})

    def spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ---- client-facing operations (also used for forwarded messages) -------
    def _client(self, env: dict) -> tuple[str, dict]:
        try:
            return open_envelope(env)
        except ValueError as exc:
            raise Reject(401, str(exc)) from exc

    def _forward(self, kind: str, env: dict, data: str | None) -> None:
        if self.peers.urls:
            self.peers.broadcast("/v2/peer/submission", {"type": "forward", "kind": kind,
                                                         "via": self.kid, "env": env, "data": data})

    def _seen(self, kid: str, **kw: object) -> None:
        self.live.setdefault(kid, {}).update(t=round(time.time(), 3), **kw)

    def register(self, env: dict, via: str | None = None) -> dict:
        kid, body = self._client(env)
        parse(RegisterBody, body)
        known = kid in self.chain.state.registered or kid in self.pending_regs
        if not known:
            self.pending_regs[kid] = env
            if via is None:
                self._forward("register", env, None)
            self.emit("register", {"key": kid})
        self._seen(kid)
        return {"v": 1, "key": kid, "banned": kid in self.chain.state.banned}

    def round_info(self) -> dict:
        rs = self.rs
        order = beacon.leader_order(rs.beacon, sorted(self.params.orchestrators)) if rs else []
        return {"v": 1, "round": rs.r if rs else 0, "phase": rs.phase if rs else "starting",
                "beacon": rs.beacon if rs else None, "leader": order[0] if order else None,
                "deadlines": rs.deadlines if rs else {}, "orchestrator": self.kid,
                "weights": {str(k): h for k, h in self.chain.state.weights.items()},
                "hyper": self.params.hp.to_dict(), "now": time.time()}

    def _current(self, r: int) -> RoundState:
        if self.rs is None or self.rs.r != r:
            raise Reject(409, f"round {r} is not open here (current {self.rs.r if self.rs else 0})")
        return self.rs

    def work(self, env: dict, via: str | None = None) -> dict:
        kid, body = self._client(env)
        w = parse(WorkBody, body)
        st = self.chain.state
        if kid not in st.registered and kid not in self.pending_regs:
            raise Reject(403, "not registered")
        if kid in st.banned:
            raise Reject(403, "banned")
        if via is not None and self.rs is not None and w.round == self.rs.r + 1:
            self.early.append(("work", via, env))
            return {"v": 1}
        rs = self._current(w.round)
        if rs.assignment is None:
            if kid not in rs.work:
                rs.work.add(kid)
                if via is None:
                    self._forward("work", env, None)
            self._seen(kid, round=rs.r)
            return {"v": 1, "status": "pending", "round": rs.r, "deadlines": rs.deadlines}
        if kid not in rs.assignment:
            return {"v": 1, "status": "not_assigned", "round": rs.r, "deadlines": rs.deadlines}
        k = rs.assignment[kid]
        self._seen(kid, round=rs.r, module=k)
        return {"v": 1, "status": "assigned", "round": rs.r, "module": k, "beacon": rs.beacon,
                "data_seed": beacon.data_seed(rs.beacon, kid), "deadlines": rs.deadlines,
                "weights": {str(m): h for m, h in self.chain.state.weights.items()},
                "tokens": self.params.tokens_per_cell}

    def commit(self, env: dict, via: str | None = None) -> dict:
        kid, body = self._client(env)
        c = parse(CommitBody, body)
        if via is not None and self.rs is not None and c.round == self.rs.r + 1:
            self.early.append(("commit", via, env))
            return {"v": 1}
        rs = self._current(c.round)
        if rs.assignment is None or kid not in rs.assignment:
            raise Reject(409, "no assignment for this key in this round")
        if rs.sample is not None and kid not in rs.commits:
            raise Reject(409, "commit phase closed")
        if kid in rs.commits:
            if rs.commits[kid]["body"] != body:
                raise Reject(409, "a different commit is already recorded for this round")
            return {"v": 1, "status": "committed"}
        rs.commits[kid] = env
        rs.via[kid] = via or self.kid
        if via is None:
            self._forward("commit", env, None)
        self.emit("commit", {"round": rs.r, "key": kid, "module": c.module})
        return {"v": 1, "status": "committed"}

    def upload(self, env: dict, data: str, via: str | None = None) -> dict:
        kid, body = self._client(env)
        u = parse(UploadBody, body)
        rs = self._current(u.round)
        if kid not in rs.commits or rs.commits[kid]["body"]["payload_hash"] != u.payload_hash:
            raise Reject(409, "upload does not match a commit")
        if time.time() > rs.deadlines["upload"]:
            raise Reject(409, "upload deadline passed")
        raw = unb64(data)
        try:
            ok = blob_id(raw) == u.payload_hash
        except (ValueError, RuntimeError) as exc:
            raise Reject(422, f"payload is not a tensor blob: {exc}") from exc
        if not ok:
            raise Reject(422, "payload hash mismatch")
        self.chain.blobs.put_bytes(raw)
        rs.payloads[kid] = u.payload_hash
        self._seen(kid, round=rs.r, claimed=u.claimed)
        if via is None:
            self._forward("upload", env, data)
        self.emit("upload", {"round": rs.r, "key": kid, "bytes": len(raw)})
        return {"v": 1, "status": "uploaded"}

    def audit_requests(self, kid: str) -> dict:
        rs = self.rs
        if rs is None or rs.sample is None:
            return {"v": 1, "round": rs.r if rs else 0, "status": "pending"}
        return {"v": 1, "round": rs.r, "status": "ready", "reveal": kid in rs.sample}

    def reveal(self, env: dict, data: str, via: str | None = None) -> dict:
        kid, body = self._client(env)
        rv = parse(RevealBody, body)
        rs = self._current(rv.round)
        if kid not in rs.commits or rs.commits[kid]["body"]["ef_hash"] != rv.ef_hash:
            raise Reject(409, "reveal does not match the commit")
        raw = unb64(data)
        try:
            ok = hash_tensors(from_bytes(raw)) == rv.ef_hash
        except (ValueError, RuntimeError) as exc:
            raise Reject(422, f"EF is not a tensor blob: {exc}") from exc
        if not ok:
            raise Reject(422, "EF hash mismatch")
        rs.reveals[kid] = raw
        if via is None:
            self._forward("reveal", env, data)
        return {"v": 1, "status": "revealed"}

    # ---- peer-facing --------------------------------------------------------
    def peer_auth(self, env: dict) -> tuple[str, dict]:
        kid, body = self._client(env)
        if kid not in self.params.orchestrators or kid == self.kid:
            raise Reject(403, "not a peer orchestrator")
        self.peers.mark(kid, True)
        return kid, body

    def on_forward(self, env: dict) -> dict:
        kid, body = self.peer_auth(env)
        f = parse(ForwardBody, body)
        if f.via != kid:
            raise Reject(403, "forwarder must be the receiver")
        if f.kind in ("upload", "reveal") and f.data is None:
            raise Reject(422, "data required")
        if f.kind == "register":
            return self.register(f.env, via=kid)
        if f.kind == "work":
            return self.work(f.env, via=kid)
        if f.kind == "commit":
            return self.commit(f.env, via=kid)
        if f.kind == "upload":
            return self.upload(f.env, f.data or "", via=kid)
        return self.reveal(f.env, f.data or "", via=kid)

    def on_verdict(self, env: dict) -> dict:
        kid, body = self.peer_auth(env)
        a = parse(AuditBody, body)
        rs = self._current(a.round)
        held = rs.audits.get(a.key)
        if held is None or (open_envelope(held)[0] == rs.via.get(a.key) and kid != rs.via.get(a.key)):
            # A receiver's own replay (its fallback when no peer held the commit) is replaced by a peer's.
            rs.audits[a.key] = env
        return {"v": 1}

    # ---- round loop ---------------------------------------------------------
    async def run(self) -> None:
        self.spawn(self._heartbeat())
        await self.peers.heartbeat()
        await self.chain.sync()
        while True:
            rs = self._open_round()
            await self._until(rs, lambda: False, rs.deadlines["open"])
            while not rs.work and self.chain.state.round < rs.r:
                # Nobody asked for work: keep the round open instead of recording empty rounds.
                rs.deadlines = self._deadlines(time.time())
                await self._until(rs, lambda: False, rs.deadlines["open"])
            if self.chain.state.round < rs.r:
                self._assign(rs)
                await self._until(rs, lambda: set(rs.commits) >= set(rs.assignment or {}),
                                  rs.deadlines["commit"])
            if self.chain.state.round < rs.r:
                self._close_commits(rs)
                await self._until(rs, lambda: self._ready(rs), rs.deadlines["audit"])
            if self.chain.state.round < rs.r:
                await self._finish(rs)

    async def _heartbeat(self) -> None:
        while True:
            await self.peers.heartbeat()
            if any(r > self.chain.state.round for r in self.peers.rounds.values()):
                self.spawn(self.chain.sync())
            await asyncio.sleep(1.0)

    def _open_round(self) -> RoundState:
        src = self.chain.record.last_final()
        r = self.chain.state.round + 1
        t = time.time()
        dl = self._deadlines(t)
        self.rs = RoundState(r=r, beacon=beacon.beacon(src["hash"], r), beacon_src=src["n"],
                             t_open=t, deadlines=dl, weights=self.chain.weights)
        self.emit("round_open", {"round": r, "beacon": self.rs.beacon[:16],
                                 "deadlines": {k: round(v, 3) for k, v in dl.items()}})
        early, self.early = self.early, []
        for kind, via, env in early:
            try:
                self.work(env, via=via) if kind == "work" else self.commit(env, via=via)
            except Reject as exc:
                log.info("early message dropped", extra={"f": {"round": r, "reason": exc.reason}})
        return self.rs

    def _deadlines(self, t: float) -> dict[str, float]:
        T = self.timing
        dl = {"open": t + T["open"]}
        dl["commit"] = dl["open"] + T["commit"]
        dl["upload"] = dl["commit"] + T["upload"]
        dl["audit"] = dl["upload"] + T["audit"]
        return dl

    async def _until(self, rs: RoundState, cond, deadline: float) -> None:
        while time.time() < deadline and self.chain.state.round < rs.r and not cond():
            self._maybe_audit(rs)
            await asyncio.sleep(0.1)

    def _assign(self, rs: RoundState) -> None:
        st = self.chain.state
        eligible = [k for k in rs.work if k not in st.banned]
        rs.assignment = beacon.assign(rs.beacon, eligible, self.params.hp.K)
        rs.phase = "commit"
        self.emit("assigned", {"round": rs.r, "cells": len(rs.assignment)})

    def _close_commits(self, rs: RoundState) -> None:
        root = merkle_root([sha256_hex(canonical(rs.commits[k])) for k in rs.commits])
        rs.aseed = beacon.audit_seed(rs.beacon, root)
        seen = self.chain.state.cells_seen
        rs.sample_t = time.time()
        rs.sample = {kid for kid in rs.commits if beacon.sampled(
            rs.aseed, kid, 1.0 if seen.get(kid, 0) < PROBATION_ROUNDS else self.params.p_audit)}
        rs.phase = "audit"
        self.emit("audit_sample", {"round": rs.r, "commits": len(rs.commits), "sampled": len(rs.sample)})

    def _ready(self, rs: RoundState) -> bool:
        return (all(k in rs.payloads for k in rs.commits)
                and all(k in rs.audits for k in rs.sample or ()))

    def _auditor(self, rs: RoundState, kid: str) -> str | None:
        """First orchestrator in audit-seed order (never the receiver) that holds this commit.

        A peer holds a commit only if it was in this round, with the same beacon, when the commit
        was forwarded. A peer that restarted mid-round, or caught up and opened the round from a
        newer final entry, refuses forwarded commits, so it is skipped. If no peer holds the
        commit ``AUDIT_GRACE`` seconds after the sample closes, the receiver replays the cell
        itself so the round does not wait for the audit deadline. ``close_round`` then excludes
        that cell as ``missing_audit`` (no strike) unless the receiver is the lone live
        orchestrator. ``None`` means "nobody yet".
        """
        via = rs.via.get(kid, self.kid)
        order = beacon.auditor_order(rs.aseed or "", kid, sorted(self.params.orchestrators), via)
        live = [o for o in order if o == self.kid or self._holds_commit(rs, o, kid)]
        if live:
            return live[0]
        return via if time.time() >= rs.sample_t + AUDIT_GRACE else None

    def _holds_commit(self, rs: RoundState, peer: str, kid: str) -> bool:
        op = self.peers.open.get(peer)
        return (bool(self.peers.up.get(peer)) and op is not None and op.get("round") == rs.r
                and op.get("beacon") == rs.beacon and kid in op.get("commits", ()))

    def _maybe_audit(self, rs: RoundState) -> None:
        for kid in sorted(rs.sample or ()):
            if (kid in rs.audits or kid in rs.auditing or kid not in rs.payloads
                    or kid not in rs.reveals or self._auditor(rs, kid) != self.kid):
                continue
            rs.auditing.add(kid)
            self.spawn(self._audit(rs, kid))

    async def _audit(self, rs: RoundState, kid: str) -> None:
        c = rs.commits[kid]["body"]
        msg = self.chain.blobs.get(c["payload_hash"])
        ef = from_bytes(rs.reveals[kid])
        async with self.audit_lock:
            t0 = time.time()
            replay = await asyncio.to_thread(run_cell, self.params.hp, rs.weights, c["module"],
                                             c["data_seed"], ef, self.tokens, self.cfg.device)
        cos, ratio = compare(msg, replay.msg) if msg is not None else (0.0, 0.0)
        env = self.ident.envelope({"type": "audit", "round": rs.r, "key": kid,
                                   "payload_hash": c["payload_hash"], "cos": cos, "ratio": ratio})
        rs.audits.setdefault(kid, env)
        ok = self.params.thresholds.passes(cos, ratio)
        self.emit("audit", {"round": rs.r, "key": kid, "cos": cos, "ratio": ratio, "ok": ok,
                            "seconds": round(time.time() - t0, 2)})
        # Ed25519 is deterministic: the peer envelope over the same body carries the same signature.
        self.peers.broadcast("/v2/peer/verdict", env["body"])

    async def _finish(self, rs: RoundState) -> None:
        rs.phase = "propose"
        order = beacon.leader_order(rs.beacon, sorted(self.params.orchestrators))
        rank = order.index(self.kid)
        t0, proposed, last_sync = time.time(), False, time.time()
        T = self.timing
        while self.chain.state.round < rs.r:
            now = time.time()
            if not proposed and now >= t0 + rank * T["propose_timeout"] and rs.r not in self.chain.held:
                proposed = True
                await self.chain.propose(self._inputs(rs))
            held = self.chain.held.get(rs.r)
            if held and held.proposer != self.kid and now > held.t + T["final_timeout"]:
                await self.chain.adopt_held(rs.r)
            if now - last_sync > T["final_timeout"]:
                last_sync = now
                await self.chain.sync()
            self._maybe_audit(rs)
            await asyncio.sleep(0.1)

    def _inputs(self, rs: RoundState) -> dict:
        st = self.chain.state
        live = set(self.peers.live()) | {self.kid}
        return {"round": rs.r, "beacon": rs.beacon, "beacon_src": rs.beacon_src, "leader": self.kid,
                "t_open": round(rs.t_open, 3), "t_close": round(time.time(), 3),
                "participants": sorted(rs.work),
                "registrations": [env for k, env in sorted(self.pending_regs.items())
                                  if k not in st.registered],
                "commits": dict(sorted(rs.commits.items())), "via": dict(sorted(rs.via.items())),
                "audits": dict(sorted(rs.audits.items())),
                "down": sorted(o for o in self.params.orchestrators if o not in live and o not in st.down),
                "up": sorted(o for o in live if o in st.down)}

    # ---- views --------------------------------------------------------------
    def status(self) -> dict:
        rec, st = self.chain.record, self.chain.state
        cur = self.rs.r if self.rs else st.round
        orchs = [{"key": k, "self": k == self.kid, "url": self.peers.urls.get(k),
                  "up": k == self.kid or bool(self.peers.up.get(k)),
                  "recorded_down": k in st.down, "round": self.peers.rounds.get(k, st.round)}
                 for k in sorted(self.params.orchestrators)]
        return {"v": 1, "orchestrator": self.kid, "now": time.time(), "round": self.round_info(),
                "chain": status.finality(rec) | {"head_round": st.round, "quorum": self.params.quorum},
                "loss": status.loss_curve(rec), "orchestrators": orchs,
                "contributors": status.contributors(rec, self.live, cur, st.strikes, st.banned),
                "record_tail": status.tail(rec),
                "thresholds": self.genesis["body"]["config"]["thresholds"],
                "hyper": self.params.hp.to_dict()}

    def metrics(self) -> str:
        rec, st = self.chain.record, self.chain.state
        fin = status.finality(rec)
        lines = [f"crowd_round {st.round}", f"crowd_entries_final {fin['final']}",
                 f"crowd_entries_provisional {fin['provisional']}",
                 f"crowd_contributors_registered {len(st.registered)}",
                 f"crowd_contributors_banned {len(st.banned)}",
                 f"crowd_eval_loss {st.last_loss if st.last_loss is not None else 'NaN'}"]
        lines += [f'crowd_peer_up{{peer="{k}"}} {int(v)}' for k, v in sorted(self.peers.up.items())]
        return "\n".join(lines) + "\n"

    def blob(self, h: str) -> bytes | None:
        try:
            return self.chain.blobs.get_bytes(h)
        except ValueError as exc:
            raise Reject(400, str(exc)) from exc

