"""The local copy of the shared record: propose, verify, co-sign, commit, catch up.

Every batch of entries for a round (registers, orchestrator up/down, verdicts,
rollback, round) is recomputed from its public inputs by ``rounds.close_round``
before this orchestrator signs or appends it. One batch per round is signed;
a conflicting proposal for the same round is refused with the held batch.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .beacon import beacon
from .blobs import BlobStore
from .crypto import Identity
from .nets import Tokens
from .peers import Peers
from .record import Record, chain_entries, entry_hash, valid_sigs
from .rounds import ChainState, Closed, Params, close_round
from .wire import blob_id

log = logging.getLogger("crowd_v2.chain")


@dataclass
class Held:
    entries: list[dict]
    closed: Closed
    t: float
    proposer: str


def _hashes(entries: list[dict]) -> list[str]:
    return [e["hash"] for e in entries]


class Chain:
    def __init__(self, ident: Identity, params: Params, record: Record, blobs: BlobStore,
                 tokens: Tokens, peers: Peers, emit: Callable[[str, dict], None]):
        self.ident, self.params, self.record, self.blobs = ident, params, record, blobs
        self.tokens, self.peers, self.emit = tokens, peers, emit
        self.state = ChainState.from_entries(record.entries)
        self.weights = self.load_weights()
        self.held: dict[int, Held] = {}
        self.lock = asyncio.Lock()
        self.sync_lock = asyncio.Lock()

    def load_weights(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, h in sorted(self.state.weights.items()):
            part = self.blobs.get(h)
            if part is None:
                raise RuntimeError(f"weights blob for module {k} missing: {h}")
            out.update(part)
        return out

    def _sign(self, e: dict) -> dict:
        e = dict(e, sigs=dict(e["sigs"]))
        e["sigs"].setdefault(self.ident.kid, self.ident.sign(e["hash"].encode()))
        return e

    async def _fetch_payloads(self, inputs: dict, source: str) -> None:
        for env in inputs["commits"].values():
            h = env.get("body", {}).get("payload_hash")
            if not isinstance(h, str) or len(h) != 64 or self.blobs.get_bytes(h) is not None:
                continue
            resp = await self.peers.get(source, f"/v2/blob/{h}") if source in self.peers.urls else None
            if resp is not None and resp.status_code == 200 and blob_id(resp.content) == h:
                self.blobs.put_bytes(resp.content)

    async def verify_batch(self, entries: list[dict], source: str) -> Closed:
        """Recompute a proposed batch from its inputs; raise ValueError if anything differs."""
        if not entries or entries[-1]["type"] != "round":
            raise ValueError("batch must end with a round entry")
        r = entries[-1]["round"]
        if r != self.state.round + 1:
            raise ValueError(f"batch for round {r}, expected {self.state.round + 1}")
        if entries[0]["n"] != self.record.next_n or entries[0]["prev"] != self.record.tail_hash:
            raise ValueError("batch does not extend my chain")
        for e in entries:
            if entry_hash(e) != e["hash"] or e["round"] != r or not valid_sigs(e, self.params.orchestrators):
                raise ValueError(f"entry {e['n']} malformed or unsigned")
        inputs = entries[-1]["body"]["inputs"]
        src = inputs["beacon_src"]
        if not 0 <= src < self.record.next_n or beacon(self.record.entries[src]["hash"], r) != inputs["beacon"]:
            raise ValueError("beacon does not match the record")
        await self._fetch_payloads(inputs, source)
        closed = await asyncio.to_thread(close_round, self.params, self.state, self.weights, inputs,
                                         self.blobs.get, self.tokens)
        expected = chain_entries(self.record.tail_hash, self.record.next_n, closed.items)
        if _hashes(expected) != _hashes(entries):
            raise ValueError("recomputed batch differs from the proposal")
        return closed

    def commit_batch(self, entries: list[dict], closed: Closed) -> list[dict]:
        """Append a verified batch (adding my signature). Caller holds ``lock``."""
        keys = self.params.orchestrators
        out = []
        for e in entries:
            e = self._sign(e)
            e["sigs"] = {k: e["sigs"][k] for k in valid_sigs(e, keys)}
            self.record.append(e)
            self.state.apply(e)
            out.append(e)
        self.weights = closed.weights
        for k in closed.changed:
            self.blobs.put({n: self.weights[n] for n in self.params.shapes[k]})
        rnd = out[-1]
        self.held.pop(rnd["round"], None)
        for e in out:
            if e["type"] in ("verdict", "rollback", "orchestrator_down", "orchestrator_up"):
                self.emit(e["type"], {"round": e["round"], **{k: v for k, v in e["body"].items()
                                                              if k in ("key", "reason", "strikes", "banned",
                                                                       "orchestrator", "rejected_loss")}})
        self.emit("round_final" if self.record.is_final(rnd) else "round_provisional",
                  {"round": rnd["round"], "n": rnd["n"], "eval_loss": rnd["body"]["eval_loss"],
                   "sigs": len(rnd["sigs"]), "leader": rnd["body"]["leader"]})
        return out

    async def propose(self, inputs: dict) -> None:
        r = inputs["round"]
        async with self.lock:
            if self.state.round >= r or r in self.held:
                return
            closed = await asyncio.to_thread(close_round, self.params, self.state, self.weights,
                                             inputs, self.blobs.get, self.tokens)
            entries = [self._sign(e) for e in
                       chain_entries(self.record.tail_hash, self.record.next_n, closed.items)]
            self.held[r] = Held(entries, closed, time.time(), self.ident.kid)
        replies = await asyncio.gather(*(self.peers.post(kid, "/v2/peer/propose",
                                                         {"type": "propose", "entries": entries})
                                         for kid in self.peers.urls))
        refused = False
        for resp in replies:
            if resp is None:
                continue
            if resp.status_code == 200:
                for e, sigs in zip(entries, resp.json()["sigs"]):
                    e["sigs"].update(sigs)
            else:
                refused = True
                log.info("proposal refused", extra={"f": {"round": r, "code": resp.status_code,
                                                          "reason": resp.text[:200]}})
        async with self.lock:
            if self.state.round >= r:
                return
            have = min(len(valid_sigs(e, self.params.orchestrators)) for e in entries)
            if have < self.params.quorum and refused:
                return
            done = self.commit_batch(entries, closed)
        self.peers.broadcast("/v2/peer/cosign", {"type": "final", "entries": done})

    async def on_propose(self, entries: list[dict], proposer: str) -> tuple[int, dict]:
        async with self.lock:
            r = entries[-1]["round"] if entries else 0
            if r <= self.state.round:
                mine = [self.record.entries[e["n"]] for e in entries if e["n"] < self.record.next_n]
                if _hashes(mine) == _hashes(entries):
                    return 200, {"sigs": [{self.ident.kid: m["sigs"][self.ident.kid]} for m in mine]}
                return 409, {"reason": "round already recorded differently"}
            if r > self.state.round + 1 or entries[0]["n"] != self.record.next_n:
                asyncio.get_running_loop().create_task(self.sync())
                return 409, {"reason": "behind"}
            held = self.held.get(r)
            if held and _hashes(held.entries) != _hashes(entries):
                return 409, {"reason": "holding another batch", "held": held.entries}
            if not held:
                try:
                    closed = await self.verify_batch(entries, proposer)
                except ValueError as exc:
                    log.warning("proposal rejected", extra={"f": {"round": r, "err": str(exc)}})
                    return 400, {"reason": str(exc)}
                held = Held([self._sign(e) for e in entries], closed, time.time(), proposer)
                self.held[r] = held
            return 200, {"sigs": [{self.ident.kid: e["sigs"][self.ident.kid]} for e in held.entries]}

    async def on_final(self, entries: list[dict], source: str) -> tuple[int, dict]:
        async with self.lock:
            for e in entries:
                n = e["n"]
                if n < self.record.next_n and self.record.entries[n]["hash"] == e["hash"]:
                    was = self.record.is_final(self.record.entries[n])
                    if self.record.add_sigs(n, e["sigs"]) and not was and self.record.is_final(self.record.entries[n]):
                        self.emit("entry_final", {"n": n, "round": e["round"], "type": e["type"]})
            new = [e for e in entries if e["n"] >= self.record.next_n]
            if not new:
                return 200, {"added": 0}
            r = new[-1]["round"]
            held = self.held.get(r)
            if held and _hashes(held.entries) == _hashes(new):
                merged = [dict(h, sigs={**h["sigs"], **e["sigs"]}) for h, e in zip(held.entries, new)]
                done = self.commit_batch(merged, held.closed)
            else:
                try:
                    closed = await self.verify_batch(new, source)
                except ValueError as exc:
                    log.warning("final batch rejected", extra={"f": {"round": r, "err": str(exc)}})
                    asyncio.get_running_loop().create_task(self.sync())
                    return 409, {"reason": str(exc)}
                done = self.commit_batch(new, closed)
        self.peers.broadcast("/v2/peer/cosign", {"type": "final", "entries": done})
        return 200, {"added": len(done)}

    async def adopt_held(self, r: int) -> None:
        """A co-signed proposal whose final broadcast never came: it has quorum, append it."""
        async with self.lock:
            held = self.held.get(r)
            if not held or held.proposer == self.ident.kid or self.state.round != r - 1:
                return
            done = self.commit_batch(held.entries, held.closed)
        self.peers.broadcast("/v2/peer/cosign", {"type": "final", "entries": done})

    async def sync(self) -> None:
        """Catch up from peers: verify and re-run every missing round, then co-sign it."""
        if self.sync_lock.locked():
            return
        async with self.sync_lock:
            for kid in list(self.peers.urls):
                resp = await self.peers.get(kid, "/v2/record?since=0", timeout=60.0)
                if resp is None or resp.status_code != 200:
                    continue
                async with self.lock:
                    done = await self._adopt(resp.json()["entries"], kid)
                if done:
                    self.peers.broadcast("/v2/peer/cosign", {"type": "final", "entries": done})

    async def _adopt(self, remote: list[dict], source: str) -> list[dict]:
        mine = self.record.entries
        i = 0
        while i < min(len(mine), len(remote)) and mine[i]["hash"] == remote[i]["hash"]:
            if set(remote[i]["sigs"]) - set(mine[i]["sigs"]):
                self.record.add_sigs(i, remote[i]["sigs"])
            i += 1
        if i < len(mine):
            keys, q = self.params.orchestrators, self.params.quorum
            if any(self.record.is_final(e) for e in mine[i:]) or not any(
                    len(valid_sigs(e, keys)) >= q for e in remote[i:]):
                log.warning("fork not adopted", extra={"f": {"at": i, "peer": source}})
                return []
            self.record.truncate(i)
            self.state = ChainState.from_entries(self.record.entries)
            self.weights = self.load_weights()
            self.held.clear()
        done, batch = [], []
        for e in remote[self.record.next_n:]:
            batch.append(e)
            if e["type"] != "round":
                continue
            try:
                closed = await self.verify_batch(batch, source)
            except ValueError as exc:
                log.warning("catch-up stopped", extra={"f": {"round": e["round"], "err": str(exc)}})
                break
            done += self.commit_batch(batch, closed)
            batch = []
        if done:
            log.info("caught up", extra={"f": {"peer": source, "entries": len(done),
                                                "round": self.state.round}})
        return done
