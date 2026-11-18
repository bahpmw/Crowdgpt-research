"""close_round: the deterministic path every orchestrator runs on the same inputs."""

import dataclasses
import json

import pytest

from conftest import TINY
from crowd_v2 import beacon
from crowd_v2.audit import compare
from crowd_v2.cell import run_cell
from crowd_v2.client import corrupt
from crowd_v2.crypto import Identity, hash_tensors
from crowd_v2.genesis import TIMING, make_genesis
from crowd_v2.rounds import ChainState, Params, close_round
from crowd_v2.wire import from_bytes


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("g")
    g = make_genesis(d, 3, TINY, 0.5, TIMING, [])
    orchs = [Identity.load_or_create(d / f"orch{i}.key") for i in range(3)]
    return g, Params.from_genesis(g["body"]), orchs, from_bytes((d / "init.safetensors").read_bytes())


def _inputs(world, tokens, behaviours: dict[str, str], rnd=1):
    g, p, orchs, w = world
    st = ChainState.from_entries([g])
    bcn = beacon.beacon(g["hash"], rnd)
    clients = {Identity.generate(): b for b in behaviours.values()}
    regs = [c.envelope({"type": "register", "name": b}) for c, b in clients.items()]
    assign = beacon.assign(bcn, [c.kid for c in clients], p.hp.K)
    commits, payloads, audits = {}, {}, {}
    for c, b in clients.items():
        k, seed = assign[c.kid], beacon.data_seed(bcn, c.kid)
        out = run_cell(p.hp, w, k, seed, {}, tokens)
        msg = corrupt(out.msg, b, 0)
        ph = hash_tensors(msg)
        payloads[ph] = msg
        commits[c.kid] = c.envelope({"type": "commit", "round": rnd, "beacon": bcn, "module": k,
                                     "data_seed": seed, "payload_hash": ph,
                                     "ef_hash": hash_tensors({}), "tokens": p.tokens_per_cell})
        cos, ratio = compare(msg, out.msg)
        audits[c.kid] = orchs[1].envelope({"type": "audit", "round": rnd, "key": c.kid,
                                           "payload_hash": ph, "cos": cos, "ratio": ratio})
    inputs = {"round": rnd, "beacon": bcn, "beacon_src": 0, "leader": orchs[0].kid, "t_open": 0.0,
              "t_close": 1.0, "participants": sorted(commits), "registrations": regs,
              "commits": commits, "via": {k: orchs[0].kid for k in commits}, "audits": audits,
              "down": [], "up": []}
    return st, inputs, payloads, {c.kid: b for c, b in clients.items()}


def test_close_round_deterministic_and_rejects_attacker(world, tokens):
    g, p, _, w = world
    st, inputs, payloads, who = _inputs(world, tokens, {"a": "honest", "b": "honest", "c": "honest",
                                                        "d": "flip"})
    one = close_round(p, st, w, inputs, payloads.get, tokens)
    two = close_round(p, st, w, json.loads(json.dumps(inputs)), payloads.get, tokens)
    assert json.dumps(one.items, sort_keys=True) == json.dumps(two.items, sort_keys=True)
    rnd = one.items[-1][2]
    status = {c["key"]: c["status"] for c in rnd["cells"]}
    assert all(status[k] == "accepted" for k, b in who.items() if b == "honest")
    assert [status[k] for k, b in who.items() if b == "flip"] == ["audit:replay"]
    verdicts = [b for t, _, b in one.items if t == "verdict"]
    assert len(verdicts) == 1 and verdicts[0]["strikes"] == 1
    assert sum(t == "register" for t, _, _ in one.items) == 4
    assert rnd["eval_loss"] < g["body"]["eval_loss"]


def test_model_gate_rolls_back(world, tokens):
    g, p, orchs, w = world
    strict = dataclasses.replace(p, hp=dataclasses.replace(p.hp, gate_tol=-10.0))
    st, inputs, payloads, _ = _inputs(world, tokens, {"a": "honest", "b": "honest"})
    out = close_round(strict, st, w, inputs, payloads.get, tokens)
    types = [t for t, _, _ in out.items]
    assert "rollback" in types
    rnd = out.items[-1][2]
    assert rnd["rolled_back"] and rnd["eval_loss"] == g["body"]["eval_loss"]
    assert rnd["weights"] == g["body"]["weights"] and out.weights is w


def test_unsigned_audit_and_missing_payload_are_not_accepted(world, tokens):
    g, p, orchs, w = world
    st, inputs, payloads, who = _inputs(world, tokens, {"a": "honest", "b": "honest"})
    victim, other = sorted(who)
    inputs["audits"][victim]["body"]["cos"] = 1.0
    inputs["audits"][victim]["body"]["ratio"] = 1.0
    inputs["audits"][victim]["sig"] = "00" * 64
    missing = inputs["commits"][other]["body"]["payload_hash"]
    out = close_round(p, st, w, inputs, lambda h: None if h == missing else payloads.get(h), tokens)
    status = {c["key"]: c["status"] for c in out.items[-1][2]["cells"]}
    assert status == {victim: "missing_audit", other: "missing_payload"}
    assert not any(t == "verdict" for t, _, _ in out.items)


def test_three_strikes_ban(world, tokens):
    g, p, orchs, w = world
    st, inputs, payloads, who = _inputs(world, tokens, {"a": "honest", "b": "scale"})
    bad = next(k for k, b in who.items() if b == "scale")
    st.registered = {k: "" for k in who}
    st.strikes[bad] = 2
    inputs["registrations"] = []
    out = close_round(p, st, w, inputs, payloads.get, tokens)
    verdict = next(b for t, _, b in out.items if t == "verdict")
    assert verdict == {"key": bad, "reason": verdict["reason"], "strikes": 3, "banned": True}
    st.apply({"type": "verdict", "body": verdict})
    assert bad in st.banned
