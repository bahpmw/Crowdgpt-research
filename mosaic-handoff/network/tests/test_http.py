"""HTTP contract tests against one in-process orchestrator (FastAPI TestClient)."""

import time

import pytest
from fastapi.testclient import TestClient

from conftest import TINY
from crowd_v2.api import create_app
from crowd_v2.client import Client
from crowd_v2.crypto import Identity, hash_tensors
from crowd_v2.genesis import make_genesis
from crowd_v2.node import Node, NodeConfig
from crowd_v2.record import verify_chain
from crowd_v2.wire import from_bytes

FAST = {"open": 1.0, "commit": 30.0, "upload": 10.0, "audit": 30.0,
        "propose_timeout": 2.0, "final_timeout": 4.0}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    d = tmp_path_factory.mktemp("net")
    g = make_genesis(d / "g", 1, TINY, 1.0, FAST, [])
    cfg = NodeConfig(port=0, key=d / "g" / "orch0.key", genesis=d / "g" / "genesis.json",
                     init=d / "g" / "init.safetensors", data=d / "o")
    node = Node(cfg)
    with TestClient(create_app(node)) as http:
        yield node, http, g


def poll(fn, timeout: float = 60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = fn()
        if got:
            return got
        time.sleep(0.1)
    raise TimeoutError("condition not reached")


def test_validation_and_auth(served):
    node, http, _ = served
    stranger = Identity.generate()
    assert http.post("/v2/register", json={"v": 1, "pub": "zz", "body": {}, "sig": "00"}).status_code == 422
    env = stranger.envelope({"type": "register"})
    env["sig"] = "11" * 64
    assert http.post("/v2/register", json=env).status_code == 401
    env = stranger.envelope({"type": "register", "evil": 1})
    assert http.post("/v2/register", json=env).status_code == 422
    assert http.post("/v2/work", json=stranger.envelope({"type": "work", "round": 1})).status_code == 403
    peer_env = stranger.envelope({"type": "propose", "entries": []})
    assert http.post("/v2/peer/propose", json=peer_env).status_code == 403
    assert http.get("/v2/blob/" + "0" * 64).status_code == 404
    assert http.get("/v2/blob/not-a-hash").status_code == 422
    assert http.get("/v2/audit/xyz").status_code == 422


def test_read_endpoints(served):
    node, http, g = served
    h = http.get("/health").json()
    assert h["status"] == "ok" and h["orchestrator"] == node.kid
    info = http.get("/v2/round").json()
    assert info["v"] == 1 and info["weights"] == g["body"]["weights"]
    blob = http.get(f"/v2/blob/{info['weights']['0']}")
    assert hash_tensors(from_bytes(blob.content)) == info["weights"]["0"]
    assert "crowd_round" in http.get("/metrics").text
    assert http.get("/v2/record?since=0").json()["entries"][0]["type"] == "genesis"
    assert http.get("/dashboard").status_code == 200


def test_full_round_reaches_final_entry(served):
    node, http, g = served
    client = Client(["http://testserver"], Identity.generate(), http=http)
    client.signed("/v2/register", {"type": "register", "name": "t"})
    row = client.do_round(0, patience=30)
    assert row["status"] == "submitted" and row["sampled"]
    assert row["bytes_up"] > row["wire_bytes"] > 0
    poll(lambda: http.get("/health").json()["round"] >= row["round"])
    entries = http.get("/v2/record?since=0").json()["entries"]
    rnd = [e for e in entries if e["type"] == "round"][-1]
    assert [c["status"] for c in rnd["body"]["cells"]] == ["accepted"]
    assert rnd["body"]["audits"][0]["cos"] == 1.0
    rep = verify_chain(node.chain.record.path, g["body"]["orchestrators"])
    assert rep["ok"] and rep["provisional"] == 0
    st = http.get("/v2/status").json()
    row_c = next(c for c in st["contributors"] if c["key"] == client.ident.kid)
    assert row_c["accepted"] == 1 and row_c["self_reported"]["train_loss"] > 0
    late = client.ident.envelope({"type": "commit", "round": row["round"], "beacon": "0" * 64,
                                  "module": 0, "data_seed": 1, "payload_hash": "0" * 64,
                                  "ef_hash": "0" * 64, "tokens": 1})
    assert http.post("/v2/commit", json=late).status_code == 409
