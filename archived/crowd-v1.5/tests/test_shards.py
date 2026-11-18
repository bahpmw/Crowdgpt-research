"""Sharded, ranged weight serving for models too large for one response."""

from __future__ import annotations

import hashlib
import itertools

import pytest
from fastapi.testclient import TestClient

from crowd_v15.client import ClientConfig, DustClient
from crowd_v15.coordinator import CoordConfig, Coordinator
from crowd_v15.server import app as app_module
from crowd_v15.store import ModelStore

from .helpers import model_tensors


@pytest.fixture()
def store(tmp_path):
    store = ModelStore(tmp_path / "state")
    Coordinator(store, CoordConfig(min_submissions=1)).bootstrap(model_tensors())
    return store


def test_shards_cover_the_file_in_order_at_tensor_boundaries(store):
    blob = store.weights_bytes()
    shards = store.shards(max_bytes=64 << 10)
    assert len(shards) > 3
    assert shards[0]["start"] == 0 and shards[-1]["end"] == len(blob)
    for left, right in itertools.pairwise(shards):
        assert left["end"] == right["start"]
    assert b"".join(blob[s["start"]:s["end"]] for s in shards) == blob
    for shard in shards:
        assert hashlib.sha256(blob[shard["start"]:shard["end"]]).hexdigest() == shard["sha256"]
    layout = store.tensor_layout()
    starts = {s["start"] for s in shards[1:]}
    assert starts <= {info["start"] for info in layout.values()}
    assert sorted(n for s in shards for n in s["names"]) == sorted(layout)


def test_tensor_layout_matches_the_tensors(store):
    tensors = store.load_weights()
    blob = store.weights_bytes()
    for name, info in store.tensor_layout().items():
        assert info["dtype"] == "F32"
        assert info["shape"] == list(tensors[name].shape)
        assert blob[info["start"]:info["end"]] == tensors[name].tobytes()


def test_manifest_lists_the_shards(store):
    store.shard_bytes = 64 << 10
    manifest = store.manifest({})
    assert len(manifest["shards"]) == len(store.shards())
    assert sum(s["bytes"] for s in manifest["shards"]) == manifest["bytes"]


@pytest.fixture()
def http(tmp_path, monkeypatch):
    app = app_module.create_app(root=tmp_path / "state", config=CoordConfig(min_submissions=1))
    app.state.coordinator.bootstrap(model_tensors())
    with TestClient(app) as client:
        yield client, app


def test_weights_route_serves_byte_ranges(http):
    client, app = http
    blob = app.state.store.weights_bytes()
    reply = client.get("/model/weights", headers={"Range": "bytes=10-109"})
    assert reply.status_code == 206
    assert reply.content == blob[10:110]
    assert reply.headers["content-range"] == f"bytes 10-109/{len(blob)}"
    tail = client.get("/model/weights", headers={"Range": "bytes=-16"})
    assert tail.status_code == 206 and tail.content == blob[-16:]
    open_end = client.get("/model/weights", headers={"Range": f"bytes={len(blob) - 5}-"})
    assert open_end.content == blob[-5:]
    full = client.get("/model/weights")
    assert full.status_code == 200 and full.content == blob
    assert full.headers["accept-ranges"] == "bytes"


@pytest.mark.parametrize("bad", ["bytes=999999999-", "bytes=0-1,5-6", "items=0-1", "bytes=a-b"])
def test_unsatisfiable_ranges_are_416(http, bad):
    client, _ = http
    assert client.get("/model/weights", headers={"Range": bad}).status_code == 416


def test_client_downloads_shard_by_shard(http, monkeypatch):
    client, app = http
    app.state.store.shard_bytes = 64 << 10
    calls = []

    def fake_http(url, *, method="GET", body=None, headers=None, timeout=0):
        path = url.split("127.0.0.1", 1)[-1]
        calls.append(headers or {})
        reply = client.request(method, path, content=body, headers=headers or {})
        assert reply.status_code < 400, reply.text
        return reply.content

    from crowd_v15 import client as client_module

    monkeypatch.setattr(client_module, "_http", fake_http)
    dust = DustClient(ClientConfig(server="http://127.0.0.1", client_id="sharded"))
    manifest = dust.sync()
    assert dust.base_sha256 == app.state.coordinator.base_sha256
    assert sum("Range" in h for h in calls) == len(manifest["shards"]) > 3
