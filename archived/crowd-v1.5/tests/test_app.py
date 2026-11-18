"""HTTP surface: read-only endpoints, both submit routes, and the admin guards."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from crowd_v15 import protocol
from crowd_v15.coordinator import CoordConfig
from crowd_v15.server import app as app_module

from .helpers import gradient_vector, model_tensors, safetensors_bytes, submit_body

SECRET = "test-secret"
ADMIN = {"X-Admin-Secret": SECRET}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ADMIN_SECRET", SECRET)
    app = app_module.create_app(root=tmp_path / "state", config=CoordConfig(lr=1e-2, min_submissions=1),
                               model_meta={"name": "unit-test"})
    with TestClient(app) as test_client:
        yield test_client, app


@pytest.fixture()
def booted(client):
    test_client, app = client
    tensors = model_tensors()
    reply = test_client.post("/admin/bootstrap", content=safetensors_bytes(tensors), headers=ADMIN)
    assert reply.status_code == 200, reply.text
    return test_client, app, tensors


def test_index_lists_the_routes(client):
    test_client, _ = client
    body = test_client.get("/").json()
    assert body["service"] == "crowd-v1.5"
    assert body["submit"] == {"dust": "/submit/dust", "backprop": "/submit/backprop"}


def test_status_before_bootstrap_says_so(client):
    test_client, _ = client
    body = test_client.get("/status").json()
    assert body["round"] == 0
    assert body["model"].get("bootstrap") == "no weights installed yet"


def test_manifest_and_weights_after_bootstrap(booted):
    test_client, _, tensors = booted
    manifest = test_client.get("/model/manifest").json()
    assert manifest["params"] == sum(int(np.asarray(t).size) for t in tensors.values())
    weights = test_client.get("/model/weights")
    assert weights.status_code == 200
    assert weights.headers["content-type"] == "application/octet-stream"
    assert weights.headers["x-model-sha256"] == manifest["baseSha256"]
    assert int(manifest["bytes"]) == len(weights.content)


def test_dust_submission_is_accepted_and_finalizes(booted):
    test_client, app, tensors = booted
    coord = app.state.coordinator
    names, sizes, vector = gradient_vector(tensors, seed=11)
    body = submit_body(names, sizes, vector, base_sha256=coord.base_sha256)
    reply = test_client.post("/submit/dust", content=body)
    assert reply.status_code == 200, reply.text
    receipt = reply.json()
    assert receipt["accepted"] is True
    assert receipt["finalizedRound"] == 0
    assert test_client.get("/status").json()["round"] == 1


def test_backprop_submission_uses_its_own_route(booted):
    test_client, app, tensors = booted
    coord = app.state.coordinator
    names, sizes, vector = gradient_vector(tensors, seed=12)
    body = submit_body(names, sizes, vector, route="backprop", tokens=0,
                       base_sha256=coord.base_sha256)
    reply = test_client.post("/submit/backprop", content=body)
    assert reply.status_code == 200, reply.text
    assert reply.json()["route"] == "backprop"


def test_route_must_match_the_url(booted):
    test_client, app, tensors = booted
    coord = app.state.coordinator
    names, sizes, vector = gradient_vector(tensors, seed=13)
    body = submit_body(names, sizes, vector, route="backprop", base_sha256=coord.base_sha256)
    reply = test_client.post("/submit/dust", content=body)
    assert reply.status_code == 400
    assert reply.json()["detail"]["reason"] == "route_mismatch"


def test_rejections_carry_a_reason(booted):
    test_client, app, tensors = booted
    coord = app.state.coordinator
    names, sizes, vector = gradient_vector(tensors, seed=14)
    body = submit_body(names, sizes, vector, base_sha256=coord.base_sha256, tokens=0)
    reply = test_client.post("/submit/dust", content=body)
    assert reply.status_code == 409
    assert reply.json()["detail"]["reason"] == "insufficient_evidence"


def test_malformed_body_is_a_400(booted):
    test_client, _, _ = booted
    reply = test_client.post("/submit/dust", content=b"\x00\x00")
    assert reply.status_code == 400
    assert reply.json()["detail"]["reason"] == "malformed_body"


def test_empty_body_is_refused(booted):
    test_client, _, _ = booted
    assert test_client.post("/submit/dust", content=b"").status_code == 400


def test_oversized_body_is_refused(booted):
    test_client, app, tensors = booted
    app.state.coordinator.config.max_body_bytes = 128
    names, sizes, vector = gradient_vector(tensors, seed=15)
    body = submit_body(names, sizes, vector, base_sha256=app.state.coordinator.base_sha256)
    reply = test_client.post("/submit/dust", content=body)
    assert reply.status_code == 413


def test_admin_needs_the_header(booted):
    test_client, _, _ = booted
    assert test_client.get("/admin/round").status_code == 403
    assert test_client.get("/admin/round", headers={"X-Admin-Secret": "wrong"}).status_code == 403
    assert test_client.get("/admin/round", headers={"Authorization": f"Bearer {SECRET}"}).status_code == 200


def test_admin_fails_closed_without_configuration(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(app_module, "ADMIN_SECRET", "")
    reply = test_client.get("/admin/round", headers={"X-Admin-Secret": ""})
    assert reply.status_code == 503


def test_finalize_endpoint_reports_the_round(booted):
    test_client, app, tensors = booted
    coord = app.state.coordinator
    coord.config.min_submissions = 2
    names, sizes, vector = gradient_vector(tensors, seed=16)
    test_client.post("/submit/dust", content=submit_body(names, sizes, vector, client_id="a",
                                                        base_sha256=coord.base_sha256))
    assert test_client.post("/admin/finalize", headers=ADMIN).json()["finalized"] is False
    reply = test_client.post("/admin/finalize?force=true", headers=ADMIN).json()
    assert reply["finalized"] is True
    assert reply["round"]["clients"] == 1


def test_routes_can_be_closed_at_runtime(booted):
    test_client, app, tensors = booted
    coord = app.state.coordinator
    assert test_client.post("/admin/routes?dust=false", headers=ADMIN).json()["routes"] == ["backprop"]
    names, sizes, vector = gradient_vector(tensors, seed=17)
    reply = test_client.post("/submit/dust", content=submit_body(names, sizes, vector,
                                                               base_sha256=coord.base_sha256))
    assert reply.status_code == 409
    assert reply.json()["detail"]["reason"] == "route_disabled"


def test_bootstrap_rejects_non_safetensors(booted):
    test_client, _, _ = booted
    reply = test_client.post("/admin/bootstrap", content=b"not a safetensors file", headers=ADMIN)
    assert reply.status_code == 400


def test_bootstrap_refuses_a_checkpoint_that_is_not_a_plain_transformer(client):
    test_client, _ = client
    tensors = model_tensors()
    tensors["engram.table.weight"] = np.zeros((4, 4), dtype=np.float32)
    reply = test_client.post("/admin/bootstrap", content=safetensors_bytes(tensors), headers=ADMIN)
    assert reply.status_code == 400
    assert "plain transformer" in reply.json()["detail"]
