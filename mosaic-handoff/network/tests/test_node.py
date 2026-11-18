"""Auditor choice when a peer's view of the open round differs (restart / catch-up split)."""

import time

import pytest

from conftest import TINY
from crowd_v2 import beacon
from crowd_v2.crypto import Identity
from crowd_v2.genesis import TIMING, make_genesis
from crowd_v2.node import AUDIT_GRACE, Node, NodeConfig

KID = "c" * 16


@pytest.fixture
def node(tmp_path):
    make_genesis(tmp_path / "g", 3, TINY, 1.0, TIMING, [])
    pubs = [Identity.load_or_create(tmp_path / "g" / f"orch{i}.key").pub for i in range(3)]
    cfg = NodeConfig(port=0, key=tmp_path / "g" / "orch0.key", genesis=tmp_path / "g" / "genesis.json",
                     init=tmp_path / "g" / "init.safetensors", data=tmp_path / "o",
                     peers=[{"url": f"http://127.0.0.1:{9000 + i}", "pub": pubs[i]} for i in (1, 2)])
    n = Node(cfg)
    rs = n._open_round()
    rs.aseed = "ab" * 32
    rs.via[KID] = n.kid
    rs.sample_t = time.time()
    n.test_peer = Identity.load_or_create(tmp_path / "g" / "orch1.key")
    return n


def _order(n: Node) -> list[str]:
    return beacon.auditor_order(n.rs.aseed, KID, sorted(n.params.orchestrators), n.kid)


def _view(n: Node, peer: str, *, up: bool = True, beacon_hex: str | None = None,
          commits: tuple[str, ...] = (KID,)) -> None:
    n.peers.up[peer] = up
    n.peers.open[peer] = {"round": n.rs.r, "beacon": beacon_hex or n.rs.beacon, "commits": list(commits)}


def test_first_peer_holding_the_commit_audits(node):
    first, second = _order(node)
    _view(node, first)
    _view(node, second)
    assert node._auditor(node.rs, KID) == first


def test_peer_with_other_beacon_or_without_the_commit_is_skipped(node):
    first, second = _order(node)
    _view(node, first, beacon_hex="f" * 64)
    _view(node, second)
    assert node._auditor(node.rs, KID) == second
    _view(node, second, commits=())
    assert node._auditor(node.rs, KID) is None


def test_receiver_audits_itself_only_after_grace(node):
    first, second = _order(node)
    _view(node, first, up=False)
    _view(node, second, commits=())
    assert node._auditor(node.rs, KID) is None
    node.rs.sample_t -= AUDIT_GRACE + 0.1
    assert node._auditor(node.rs, KID) == node.kid


def test_peer_audit_replaces_receiver_self_audit(node):
    body = {"type": "audit", "round": node.rs.r, "key": KID, "payload_hash": "0" * 64, "cos": 1.0, "ratio": 1.0}
    node.rs.audits[KID] = node.ident.envelope(body)
    peer_env = node.test_peer.envelope(body)
    node.on_verdict(peer_env)
    assert node.rs.audits[KID] == peer_env
    node.on_verdict(node.test_peer.envelope(dict(body, cos=0.5)))
    assert node.rs.audits[KID] == peer_env
