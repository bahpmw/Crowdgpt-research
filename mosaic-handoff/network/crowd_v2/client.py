"""Contributor client: ``python -m crowd_v2.client --urls http://127.0.0.1:8701,...``.

Loop: register -> get round -> request work -> download module blobs (hash
checked) -> train cell -> signed commit -> upload -> answer reveal request ->
wait for the round to close. Fails over to the next orchestrator URL on
connection errors.

Behaviours other than ``honest`` exist only to test the checks (sim/tests):
``flip`` (negated message), ``scale`` (x8), ``noise`` (random on the same
support, same norm), ``zero`` (x1e-3, lazy), ``copy`` (re-sends another
client's accepted payload from the last round), ``wrongdata`` (trains on data
it chose), ``stale`` (trains from the previous round's weights). Attackers
report their EF honestly; the replay then shows their message is not the one
the public inputs produce.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import httpx
import torch

from . import logs
from .audit import flat
from .cell import run_cell, wire_bytes
from .crypto import Identity, hash_tensors
from .nets import Hyper, Tokens
from .wire import b64, from_bytes, to_bytes

log = logging.getLogger("crowd_v2.client")
BEHAVIOURS = ("honest", "flip", "scale", "noise", "zero", "copy", "wrongdata", "stale")


class RoundAbandoned(Exception):
    """The current round cannot be completed by this client (late, closed, refused)."""


def corrupt(msg: dict[str, torch.Tensor], kind: str, seed: int) -> dict[str, torch.Tensor]:
    if kind == "flip":
        return {n: -v for n, v in msg.items()}
    if kind == "scale":
        return {n: v * 8 for n, v in msg.items()}
    if kind == "zero":
        return {n: v * 1e-3 for n, v in msg.items()}
    if kind == "noise":
        g = torch.Generator().manual_seed(seed)
        z = {n: torch.randn(v.shape, generator=g) * (v != 0) for n, v in msg.items()}
        s = flat(msg).norm() / (flat(z).norm() + 1e-12)
        return {n: (v * float(s)).bfloat16() for n, v in z.items()}
    return msg


class Client:
    def __init__(self, urls: list[str], ident: Identity, behaviour: str = "honest",
                 device: str = "cpu", http: httpx.Client | None = None, log_path: Path | None = None):
        if behaviour not in BEHAVIOURS:
            raise ValueError(f"unknown behaviour {behaviour}")
        self.urls, self.ident, self.behaviour, self.device = urls, ident, behaviour, device
        self.http = http or httpx.Client(timeout=httpx.Timeout(120.0, connect=2.0))
        self.log_path = log_path
        self.cur = 0
        self.ef: dict[str, torch.Tensor] = {}
        self.ef_module: int | None = None
        self.cache: dict[str, dict[str, torch.Tensor]] = {}
        self.prev_weights: dict[str, torch.Tensor] | None = None
        self.tokens: Tokens | None = None
        self.up = self.down = 0

    # ---- transport ----------------------------------------------------------
    def call(self, method: str, path: str, body: dict | None = None, patience: float = 30.0) -> httpx.Response:
        deadline = time.time() + patience
        while True:
            url = self.urls[self.cur]
            try:
                resp = self.http.request(method, url + path, json=body)
            except httpx.TransportError as exc:
                log.info("orchestrator unreachable, failing over",
                         extra={"f": {"url_index": self.cur, "err": type(exc).__name__}})
                self.cur = (self.cur + 1) % len(self.urls)
                if time.time() > deadline:
                    raise
                time.sleep(0.3)
                continue
            self.up += len(resp.request.content or b"")
            self.down += len(resp.content)
            return resp

    def signed(self, path: str, body: dict, data: bytes | None = None) -> dict:
        env = self.ident.envelope(body)
        if data is not None:
            env["data"] = b64(data)
        resp = self.call("POST", path, env)
        if resp.status_code != 200:
            raise RoundAbandoned(f"{path} -> {resp.status_code} {resp.text[:200]}")
        return resp.json()

    def round_info(self) -> dict:
        return self.call("GET", "/v2/round").json()

    # ---- one round ----------------------------------------------------------
    def wait_open_round(self, after: int, patience: float) -> dict:
        """Poll until a round newer than ``after`` is in its open phase."""
        deadline = time.time() + patience
        while time.time() < deadline:
            info = self.round_info()
            if info["round"] > after and info["phase"] == "open" and time.time() < info["deadlines"]["open"] - 0.3:
                return info
            time.sleep(0.2)
        raise RoundAbandoned("no open round")

    def weights(self, hashes: dict[str, str]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, h in sorted(hashes.items()):
            if h not in self.cache:
                resp = self.call("GET", f"/v2/blob/{h}")
                part = from_bytes(resp.content) if resp.status_code == 200 else {}
                if hash_tensors(part) != h:
                    raise RoundAbandoned(f"module {k} weights failed the hash check")
                self.cache[h] = part
            out.update(self.cache[h])
        self.cache = {h: self.cache[h] for h in hashes.values()}
        return out

    def copied_payload(self, k: int) -> dict[str, torch.Tensor] | None:
        entries = self.call("GET", "/v2/record?since=0").json()["entries"]
        for e in reversed(entries):
            if e["type"] != "round":
                continue
            for c in e["body"]["cells"]:
                if c["module"] == k and c["status"] == "accepted" and c["key"] != self.ident.kid:
                    resp = self.call("GET", f"/v2/blob/{c['payload_hash']}")
                    if resp.status_code == 200:
                        return from_bytes(resp.content)
            return None
        return None

    def train(self, hp: Hyper, tokens: Tokens, w: dict, k: int, seed: int,
              rnd: int) -> tuple[dict, dict, float]:
        """Returns (message, EF that went in, train loss)."""
        if self.ef_module != k:
            # A new module means the old EF buffer no longer matches anything we send: drop it.
            self.ef, self.ef_module = {}, k
        ef_in = self.ef
        if self.behaviour == "copy":
            copied = self.copied_payload(k)
            if copied is not None:
                return copied, ef_in, float("nan")
        start = self.prev_weights if self.behaviour == "stale" and self.prev_weights else w
        data_seed = seed + 1 if self.behaviour == "wrongdata" else seed
        out = run_cell(hp, start, k, data_seed, ef_in, tokens, self.device)
        self.ef = out.ef
        return corrupt(out.msg, self.behaviour, rnd), ef_in, out.loss

    def do_round(self, after: int, patience: float) -> dict:
        self.up = self.down = 0
        info = self.wait_open_round(after, patience)
        r = info["round"]
        hp = Hyper(**info["hyper"])
        if self.tokens is None:
            self.tokens = Tokens(hp.vocab, hp.seq)
        row = {"round": r, "key": self.ident.kid, "behaviour": self.behaviour, "device": self.device}
        while True:
            a = self.signed("/v2/work", {"type": "work", "round": r})
            if a["status"] != "pending":
                break
            time.sleep(0.2)
        if a["status"] != "assigned":
            return row | {"status": a["status"], "bytes_up": self.up, "bytes_down": self.down}
        k = a["module"]
        w = self.weights(a["weights"])
        t0 = time.time()
        msg, ef_in, loss = self.train(hp, self.tokens, w, k, a["data_seed"], r)
        self.prev_weights = w
        train_s = time.time() - t0
        payload = to_bytes(msg)
        ph, eh = hash_tensors(msg), hash_tensors(ef_in)
        self.signed("/v2/commit", {"type": "commit", "round": r, "beacon": a["beacon"], "module": k,
                                   "data_seed": a["data_seed"], "payload_hash": ph,
                                   "ef_hash": eh, "tokens": a["tokens"]})
        claimed = {"train_loss": loss if loss == loss else -1.0, "train_seconds": round(train_s, 3)}
        self.signed("/v2/upload", {"type": "upload", "round": r, "payload_hash": ph, "claimed": claimed},
                    data=payload)
        sampled = self.answer_audit(r, ef_in, eh, a["deadlines"]["audit"])
        return row | {"status": "submitted", "module": k, "payload_hash": ph, "sampled": sampled,
                      "train_loss": claimed["train_loss"], "train_s": round(train_s, 3),
                      "bytes_up": self.up, "bytes_down": self.down,
                      "wire_bytes": wire_bytes({n: tuple(v.shape) for n, v in msg.items()}, hp.topk)}

    def answer_audit(self, r: int, ef_in: dict, ef_hash: str, deadline: float) -> bool:
        while time.time() < deadline:
            q = self.call("GET", f"/v2/audit/{self.ident.kid}").json()
            if q["round"] != r:
                return False
            if q["status"] == "ready":
                if q["reveal"]:
                    self.signed("/v2/reveal", {"type": "reveal", "round": r, "ef_hash": ef_hash},
                                data=to_bytes(ef_in))
                return bool(q["reveal"])
            time.sleep(0.2)
        return False

    def run(self, rounds: int, leave_after: int | None, patience: float) -> None:
        self.signed("/v2/register", {"type": "register", "name": self.behaviour})
        done, last = 0, 0
        while last < rounds and (leave_after is None or done < leave_after):
            try:
                row = self.do_round(last, patience)
            except RoundAbandoned as exc:
                log.info("round abandoned", extra={"f": {"after": last, "reason": str(exc)}})
                last = max(last, self.round_info()["round"])
                continue
            last = row["round"]
            done += row["status"] == "submitted"
            self._log(row)

    def _log(self, row: dict) -> None:
        log.info("round done", extra={"f": row})
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", required=True, help="comma-separated orchestrator URLs (failover order)")
    ap.add_argument("--key", type=Path, required=True, help="key file (created if missing)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--behaviour", default="honest", choices=BEHAVIOURS)
    ap.add_argument("--rounds", type=int, default=10, help="stop once the network passes this round")
    ap.add_argument("--leave-after", type=int, help="churn: leave after this many submitted rounds")
    ap.add_argument("--patience", type=float, default=600.0, help="seconds to wait for an open round")
    ap.add_argument("--log", type=Path, help="JSONL per-round metrics")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ident = Identity.load_or_create(a.key)
    logs.setup(f"client-{ident.kid}")
    Client(a.urls.split(","), ident, a.behaviour, a.device, log_path=a.log).run(
        a.rounds, a.leave_after, a.patience)


if __name__ == "__main__":
    main()
