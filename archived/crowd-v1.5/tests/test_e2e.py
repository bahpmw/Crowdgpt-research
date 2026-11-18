"""End to end: a real HTTP coordinator, two real DUST clients, one real step.

The unit tests fake one side or the other.  This one runs uvicorn in a thread and
uses the actual client, so the codec, the header guards, the HTTP body limits and
the round protocol are all exercised together -- which is where mismatches
between the two halves actually hide.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
import uvicorn

from crowd_v15.client import ClientConfig, ClientError, DustClient
from crowd_v15.coordinator import CoordConfig
from crowd_v15.server import app as app_module
from crowd_v15.store import weights_digest

from .helpers import model_tensors


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
def live_server(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ADMIN_SECRET", "secret")
    app = app_module.create_app(
        root=tmp_path / "state",
        config=CoordConfig(lr=1e-2, min_submissions=2),
        model_meta={"name": "e2e"},
    )
    app.state.coordinator.bootstrap(model_tensors())
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    try:
        yield f"http://127.0.0.1:{port}", app
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _config(url: str, client_id: str, **overrides) -> ClientConfig:
    base = dict(
        server=url,
        client_id=client_id,
        preset="tiny",
        task="identity",
        batch=2,
        seq_len=8,
        population=4,
        draw_chunk=4,
        local_steps=1,
    )
    base.update(overrides)
    return ClientConfig(**base)


def test_two_clients_produce_one_round_and_the_weights_move(live_server):
    url, app = live_server
    coord = app.state.coordinator
    before_digest = coord.base_sha256
    before = app.state.store.load_weights()

    clients = [DustClient(_config(url, f"c{i}", seed=i)) for i in range(2)]
    receipts = []
    for client in clients:
        manifest = client.sync()
        assert manifest["baseSha256"] == before_digest
        grads, stats = client.gradient()
        receipts.append(client.submit(grads, stats))
        client.round_id += 1

    assert all(r["accepted"] for r in receipts), receipts
    assert receipts[0]["acceptedThisRound"] == 1
    assert receipts[0]["neededThisRound"] == 1
    assert receipts[1]["finalizedRound"] == 0

    after = app.state.store.load_weights()
    assert coord.round_id == 1
    assert coord.base_sha256 != before_digest
    assert coord.base_sha256 == weights_digest(after)
    changed = [name for name in before if not np.array_equal(before[name], after[name])]
    assert changed, "no parameter moved"
    record = coord.status()["lastRound"]
    assert record["route"] == "dust"
    assert record["clients"] == 2
    assert record["tokens"] > 0

    # The next round serves the new weights and the client verifies the digest.
    next_manifest = clients[0].sync()
    assert next_manifest["baseSha256"] == coord.base_sha256
    assert next_manifest["round"] == 1


def test_client_sees_a_rejection_as_an_error(live_server):
    url, app = live_server
    coord = app.state.coordinator
    coord.config.min_submissions = 1
    client = DustClient(_config(url, "solo"))
    client.sync()
    grads, stats = client.gradient()
    client.submit(grads, stats)
    # The round moved; a second submission for the old round must be refused.
    with pytest.raises(ClientError) as exc:
        client.submit(grads, stats)
    assert "409" in str(exc.value)
    assert "stale_round" in str(exc.value)


def test_dry_run_never_touches_the_server(live_server):
    url, app = live_server
    coord = app.state.coordinator
    before = coord.base_sha256
    client = DustClient(_config(url, "dry", dry_run=True))
    summary = client.run(rounds=1, log=lambda *a: None)
    assert summary["dryRun"] is True
    assert summary["rounds"][0]["receipt"]["dryRun"] is True
    assert coord.base_sha256 == before
    assert coord.round_id == 0


def test_boundary_client_uploads_far_less_than_dense(live_server):
    url, _ = live_server
    client = DustClient(_config(url, "bytes", codec="ngd1r"))
    client.sync()
    grads, stats = client.gradient()
    receipt = client.submit(grads, stats)
    assert receipt["uploadBytes"] < receipt["denseBytes"] / 3
