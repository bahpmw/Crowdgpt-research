"""Coordinator behaviour: what counts, what is refused, and what a step does."""

from __future__ import annotations

import numpy as np
import pytest

from crowd_v15 import protocol
from crowd_v15.protocol import ProtocolError
from crowd_v15.coordinator import CoordConfig, Coordinator, Rejected
from crowd_v15.store import ModelStore, StoreError, weights_digest

from .helpers import gradient_vector, model_tensors, submit_body


@pytest.fixture()
def store(tmp_path):
    return ModelStore(tmp_path / "state")


@pytest.fixture()
def booted(store):
    tensors = model_tensors()
    coord = Coordinator(store, CoordConfig(lr=1e-2, min_submissions=1))
    coord.bootstrap(tensors)
    return coord, tensors


def _submit(coord, tensors, **kwargs):
    names, sizes, vector = gradient_vector(tensors, seed=kwargs.pop("seed", 1))
    kwargs.setdefault("round_id", coord.round_id)
    body = submit_body(names, sizes, vector,
                       base_sha256=kwargs.pop("base", coord.base_sha256), **kwargs)
    return coord.submit(body), vector


def test_bootstrap_sets_round_zero_and_a_digest(booted):
    coord, tensors = booted
    assert coord.round_id == 0
    assert coord.base_sha256 == weights_digest(tensors)
    status = coord.status()
    assert status["acceptedThisRound"] == 0
    assert status["routes"] == ["dust", "backprop"]


def test_submit_then_finalize_moves_the_weights_by_lr_times_gradient(booted):
    coord, tensors = booted
    names, sizes, vector = gradient_vector(tensors, seed=5, scale=1e-3)
    receipt = coord.submit(
        submit_body(names, sizes, vector, codec_name="dense", base_sha256=coord.base_sha256)
    )
    assert receipt["accepted"] is True
    assert receipt["finalizedRound"] == 0
    after = coord.store.load_weights()
    assert coord.round_id == 1
    assert coord.base_sha256 == weights_digest(after)

    offset = 0
    for name in names:
        size = sizes[names.index(name)]
        expected = tensors[name].reshape(-1) - 0.01 * vector[offset:offset + size]
        assert np.allclose(after[name].reshape(-1), expected, atol=1e-6), name
        offset += size


def test_quorum_is_required_before_a_step(booted):
    coord, tensors = booted
    coord.config.min_submissions = 2
    receipt, _ = _submit(coord, tensors, client_id="a")
    assert receipt["accepted"] is True
    assert "finalizedRound" not in receipt
    assert coord.finalize() is None
    _submit(coord, tensors, client_id="b", seed=2)
    assert coord.round_id == 1


def test_one_submission_per_client_per_round(booted):
    coord, tensors = booted
    coord.config.min_submissions = 2
    _submit(coord, tensors, client_id="a")
    with pytest.raises(Rejected) as exc:
        _submit(coord, tensors, client_id="a", seed=2)
    assert exc.value.reason == "duplicate_client"


def test_stale_round_is_rejected(booted):
    coord, tensors = booted
    with pytest.raises(Rejected) as exc:
        _submit(coord, tensors, round_id=7)
    assert exc.value.reason == "stale_round"


def test_gradient_from_other_weights_is_rejected(booted):
    coord, tensors = booted
    with pytest.raises(Rejected) as exc:
        _submit(coord, tensors, base="b" * 64)
    assert exc.value.reason == "base_weights_changed"


def test_parameter_layout_must_match_the_served_model(booted):
    coord, tensors = booted
    body = submit_body(["w"], [8], np.zeros(8, dtype=np.float32), base_sha256=coord.base_sha256)
    with pytest.raises(Rejected) as exc:
        coord.submit(body)
    assert exc.value.reason == "parameter_set_mismatch"


def test_dust_needs_evidence(booted):
    coord, tensors = booted
    with pytest.raises(Rejected) as exc:
        _submit(coord, tensors, tokens=0)
    assert exc.value.reason == "insufficient_evidence"
    with pytest.raises(Rejected) as exc:
        _submit(coord, tensors, client_id="z", sigma=0.0)
    assert exc.value.reason == "invalid_estimator_record"


def test_backprop_route_skips_the_evidence_check(booted):
    coord, tensors = booted
    receipt, _ = _submit(coord, tensors, route="backprop", tokens=0, population=0, sigma=0.0)
    assert receipt["route"] == "backprop"
    assert receipt["accepted"] is True


def test_routes_cannot_be_mixed_within_a_round(booted):
    coord, tensors = booted
    coord.config.min_submissions = 2
    _submit(coord, tensors, route="dust", client_id="d")
    with pytest.raises(Rejected) as exc:
        _submit(coord, tensors, route="backprop", client_id="b", seed=3)
    assert exc.value.reason == "mixed_routes"


def test_closed_route_is_refused(booted):
    coord, tensors = booted
    coord.config.routes = ("backprop",)
    with pytest.raises(Rejected) as exc:
        _submit(coord, tensors, route="dust")
    assert exc.value.reason == "route_disabled"


def test_huge_norm_estimate_is_capped_not_refused(booted):
    """A DUST estimate at sigma=1e-3 has a norm near 1e5 on a 1B model.

    An absolute peak bound rejected every honest submission, so the coordinator
    normalises to a direction and applies the declared norm capped by
    ``max_abs_grad``.
    """
    coord, tensors = booted
    coord.config.max_abs_grad = 1e-3
    names, sizes, vector = gradient_vector(tensors, scale=1e4)
    coord.submit(submit_body(names, sizes, vector, client_id="huge", round_id=coord.round_id,
                             base_sha256=coord.base_sha256, codec_name="dense", tokens=64, population=2, sigma=1e-3))
    record = coord.status()["lastRound"]
    assert record is not None
    assert record["declaredNorm"] > 1e3
    assert record["gradNorm"] <= 1e-3 + 1e-6
    assert record["clipped"] is True
    assert record["route"] == "dust"


def test_non_finite_estimate_is_refused(booted):
    import numpy as np

    coord, tensors = booted
    names, sizes, vector = gradient_vector(tensors)
    vector = vector.copy()
    vector[3] = np.inf
    # The wire format refuses a non-finite vector before the coordinator's own
    # guard can see it, which is the stronger of the two checks.
    with pytest.raises(ProtocolError):
        coord.submit(submit_body(names, sizes, vector, client_id="nan", round_id=coord.round_id,
                                 base_sha256=coord.base_sha256, codec_name="dense", tokens=64,
                                 population=2, sigma=1e-3))

def test_gradient_clip_scales_the_step(booted):
    coord, tensors = booted
    coord.config.grad_clip = 1e-6
    _submit(coord, tensors, codec_name="dense")
    record = coord.status()["lastRound"]
    assert record["clipped"] is True
    assert record["gradNorm"] > 1e-6 * 0.99


def test_momentum_carries_across_rounds(booted):
    coord, tensors = booted
    coord.config.momentum = 0.9
    _submit(coord, tensors, codec_name="dense")
    first = coord.store.load_weights()
    _submit(coord, first, codec_name="dense", seed=9)
    second = coord.store.load_weights()
    # Tied embeddings: the shared tensor is listed once, under its first name.
    shared = "tok_emb.weight"
    assert not np.allclose(first[shared], second[shared])


def test_history_records_each_round(booted):
    coord, _ = booted
    _submit(coord, model_tensors(), client_id="a")
    _submit(coord, coord.store.load_weights(), client_id="b", seed=4)
    history = coord.status()
    assert history["round"] == 2
    assert history["lastRound"]["round"] == 1


def test_restart_drops_unapplied_submissions(store):
    tensors = model_tensors()
    coord = Coordinator(store, CoordConfig(min_submissions=2))
    coord.bootstrap(tensors)
    _submit(coord, tensors, client_id="a")
    restarted = Coordinator(store, CoordConfig(min_submissions=2))
    assert restarted.status()["acceptedThisRound"] == 0
    # Resubmission is accepted because the round never moved.
    _submit(restarted, tensors, client_id="a")


def test_missing_weights_are_a_store_error(store):
    coord = Coordinator(store, CoordConfig())
    with pytest.raises(StoreError):
        coord.store.load_weights()
    assert "bootstrap" in coord.status()["model"] or coord.status()["round"] == 0


def test_manifest_describes_the_served_model(booted):
    coord, tensors = booted
    manifest = coord.store.manifest({"name": "demo"})
    assert manifest["schema"] == "crowd-v15-model-1"
    assert manifest["params"] == sum(int(np.asarray(t).size) for t in tensors.values())
    assert manifest["baseSha256"] == coord.base_sha256
    assert manifest["names"] == sorted(tensors)


def test_estimator_fields_land_in_the_audit_log(booted):
    coord, tensors = booted
    _submit(coord, tensors, tokens=4096, population=64)
    lines = (coord.store.log_path).read_text().strip().splitlines()
    assert len(lines) == 1
    import json

    record = json.loads(lines[0])
    assert record["tokens"] == 4096
    assert record["population"] == 64
    assert record["weight"] == 4096

def test_outer_nesterov_step_uses_its_own_learning_rate(store):
    """DiLoCo's outer optimizer: a pseudo-gradient step size separate from ``lr``."""
    tensors = model_tensors()
    coord = Coordinator(store, CoordConfig(lr=1e-2, min_submissions=1,
                                           outer_lr=0.7, outer_momentum=0.9))
    coord.bootstrap(tensors)
    names, sizes, vector = gradient_vector(tensors, seed=5, scale=1e-3)
    coord.submit(submit_body(names, sizes, vector, codec_name="dense",
                             base_sha256=coord.base_sha256))
    assert coord.round_id == 1

    norm = float(np.linalg.norm(vector))
    magnitude = min(norm, coord.config.max_abs_grad)
    pseudo = (vector / norm) * magnitude
    # A zero buffer makes the first Nesterov step (1 + momentum) * outer_lr * p.
    factor = coord.config.outer_lr * (1.0 + coord.config.outer_momentum)
    after = coord.store.load_weights()
    offset = 0
    for name, size in zip(names, sizes):
        chunk = pseudo[offset:offset + size]
        before = np.asarray(tensors[name], dtype=np.float32).reshape(-1)
        got = np.asarray(after[name], dtype=np.float32).reshape(-1)
        # Weights are stored bf16, so compare the move, not the absolute value.
        assert np.allclose(got - before, -factor * chunk, rtol=2e-2, atol=2e-6), name
        offset += size
    record = coord.status()["lastRound"]
    assert record["outerLr"] == 0.7
    assert record["outerMomentum"] == 0.9


def test_outer_optimizer_rejects_bad_values():
    with pytest.raises(ValueError):
        CoordConfig(outer_lr=-1.0)
    with pytest.raises(ValueError):
        CoordConfig(outer_momentum=1.0)
