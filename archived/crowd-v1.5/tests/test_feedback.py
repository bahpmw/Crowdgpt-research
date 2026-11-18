"""Error feedback: the carried residual, checked against the real codec.

The 1-bit containers are biased, so the claim under test is not "the upload is
small" (that is ``test_protocol``) but "carrying the quantization error forward
makes an equal number of rounds land closer to the optimum".  These tests use the
same ``quantize`` the client uses on the wire, so a passing result cannot come
from a friendlier encoder than the coordinator actually decodes.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn

from crowd_v15 import codec, feedback, protocol
from crowd_v15.client import ClientConfig, DustClient
from crowd_v15.coordinator import CoordConfig
from crowd_v15.server import app as app_module

from .helpers import gradient_vector, model_tensors


def _values(count: int = 4096, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(count).astype(np.float32)


# --------------------------------------------------------------------------- #
# The quantizer mirrors the wire
# --------------------------------------------------------------------------- #


def test_quantize_matches_what_the_coordinator_decodes():
    names, sizes, vector = gradient_vector(model_tensors(), seed=2)
    total = int(sum(sizes))
    for codec_name in codec.CODECS:
        header = protocol.build_header(
            route="dust",
            client_id="client-a",
            round_id=0,
            base_sha256="0" * 64,
            names=names,
            sizes=sizes,
            codec_name=codec_name,
            tokens=256,
            population=8,
            sigma=1e-3,
            gamma=0.98,
        )
        body = protocol.encode_body(header, vector)
        _, decoded = protocol.decode_body(body, max_params=total)
        sent = feedback.quantize(vector, codec_name)
        assert sent.shape == vector.shape
        # Byte-for-byte the same quantizer, so not "close": equal.
        assert np.array_equal(sent, decoded.astype(np.float32, copy=False)), codec_name


def test_quantize_refuses_an_unknown_codec():
    with pytest.raises(ValueError):
        feedback.quantize(_values(16), "ngd9")


# --------------------------------------------------------------------------- #
# The residual is exactly the error, and it is carried
# --------------------------------------------------------------------------- #


def test_residual_is_the_quantization_error():
    engine = feedback.ErrorFeedback("ngd1")
    vector = _values()
    sent, stats = engine.encode(vector)
    assert np.array_equal(sent, feedback.quantize(vector, "ngd1"))
    np.testing.assert_allclose(engine.residual, vector - sent, atol=1e-6)
    assert stats["rounds"] == 1
    assert stats["residualNorm"] > 0
    assert stats["correctionNorm"] == 0.0  # nothing carried yet


def test_second_round_sends_the_gradient_plus_the_previous_error():
    engine = feedback.ErrorFeedback("ngd1")
    first, _ = engine.encode(_values(seed=1))
    carried = engine.residual.copy()
    assert np.linalg.norm(carried) > 0

    second_vector = _values(seed=2)
    second, stats = engine.encode(second_vector)
    np.testing.assert_allclose(
        second, feedback.quantize(second_vector + carried, "ngd1"), atol=1e-6
    )
    assert stats["correctionNorm"] > 0
    np.testing.assert_allclose(
        engine.residual, (second_vector + carried) - second, atol=1e-6
    )


def test_residual_resets_when_the_parameter_count_changes():
    engine = feedback.ErrorFeedback("ngd1")
    engine.encode(_values(2048))
    assert engine.residual.size == 2048
    engine.encode(_values(4096, seed=5))
    assert engine.residual.size == 4096
    assert engine.rounds == 2


def test_reset_clears_the_carried_error():
    engine = feedback.ErrorFeedback("ngd1r")
    engine.encode(_values())
    assert np.linalg.norm(engine.residual) > 0
    engine.reset()
    assert engine.residual is None
    assert engine.rounds == 0


# --------------------------------------------------------------------------- #
# Equal rounds, lower loss
# --------------------------------------------------------------------------- #


def _quadratic(codec_name: str, carry: bool, *, steps: int = 1000, lr: float = 0.05,
               count: int = 4096, seed: int = 3) -> float:
    """Minimize 0.5*||w||^2 through the wire quantizer; return the final norm."""

    weights = np.random.default_rng(seed).standard_normal(count).astype(np.float32)
    engine = feedback.ErrorFeedback(codec_name) if carry else None
    for _ in range(steps):
        gradient = weights.copy()
        sent = engine.encode(gradient)[0] if engine is not None else feedback.quantize(
            gradient, codec_name
        )
        weights = weights - lr * sent
    return float(np.linalg.norm(weights))


@pytest.mark.parametrize("codec_name", ["ngd1", "ngd1r"])
def test_carrying_the_error_reaches_a_lower_loss_at_equal_rounds(codec_name):
    plain = _quadratic(codec_name, False)
    carried = _quadratic(codec_name, True)
    # Measured: ngd1 4.23 -> 0.48 (8.9x) and ngd1r 1.80 -> 0.10 (18.2x) at these
    # settings. The bound is deliberately loose; the direction is what matters.
    assert carried < plain / 4.0, (codec_name, plain, carried)


def test_error_feedback_is_a_no_op_for_the_dense_codec():
    # dense is bf16, which is also biased, but the per-value error is tiny; what
    # is asserted here is that the engine does not corrupt an unbiased upload.
    engine = feedback.ErrorFeedback("dense")
    vector = _values(1024)
    sent, _ = engine.encode(vector)
    np.testing.assert_allclose(sent, vector, rtol=1e-2, atol=1e-3)


# --------------------------------------------------------------------------- #
# End to end through the real server
# --------------------------------------------------------------------------- #


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
        model_meta={"name": "feedback"},
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
        yield f"http://127.0.0.1:{port}", app, tmp_path / "state"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_a_feedback_client_completes_a_round_and_says_so_in_the_audit_log(live_server):
    url, app, root = live_server
    coord = app.state.coordinator
    config = dict(
        server=url,
        preset="tiny",
        task="identity",
        batch=2,
        seq_len=8,
        population=4,
        draw_chunk=4,
        codec="ngd1",
        error_feedback=True,
    )
    clients = [DustClient(ClientConfig(client_id=f"fb{i}", seed=i, **config)) for i in range(2)]
    receipts = []
    for client in clients:
        client.sync()
        grads, stats = client.gradient()
        receipt = client.submit(grads, stats)
        assert receipt["uploadBytes"] < receipt["denseBytes"] / 4
        receipts.append(receipt)
        client.round_id += 1

    assert all(r["accepted"] for r in receipts), receipts
    assert coord.round_id == 1
    record = coord.status()["lastRound"]
    assert record["route"] == "dust"
    assert record["clients"] == 2

    lines = [
        json.loads(line)
        for line in (root / "submissions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    assert all(line["errorFeedback"] is True for line in lines), lines
    assert all(line["sigma"] > 0 and line["tokens"] > 0 for line in lines)


def test_a_client_without_feedback_reports_false(live_server):
    url, app, root = live_server
    client = DustClient(
        ClientConfig(
            server=url,
            client_id="plain",
            preset="tiny",
            task="identity",
            batch=2,
            seq_len=8,
            population=4,
            draw_chunk=4,
            codec="ngd1",
        )
    )
    client.sync()
    grads, stats = client.gradient()
    client.submit(grads, stats)
    lines = [
        json.loads(line)
        for line in (root / "submissions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert lines[-1]["errorFeedback"] is False
