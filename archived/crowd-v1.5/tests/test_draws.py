"""Averaging k draws at fixed weights: same payload, less Monte-Carlo noise."""

from __future__ import annotations

import numpy as np
import pytest

from crowd_v15.client import ClientConfig, DustClient


def _client(tmp_path, **kwargs):
    cfg = ClientConfig(
        server="http://127.0.0.1:9",
        client_id="draws-test",
        preset="tiny",
        population=4,
        draw_chunk=4,
        batch=1,
        seq_len=8,
        offline=True,
        dry_run=True,
        seed=3,
        device="cpu",
        **kwargs,
    )
    return DustClient(cfg)


def test_draws_must_be_positive():
    with pytest.raises(ValueError):
        ClientConfig(server="http://x", client_id="c", draws=0).check()


def test_run_reports_the_averaged_draw_count(tmp_path):
    summary = _client(tmp_path, draws=2).run(1, log=lambda *_: None)
    stats = summary["rounds"][0]["stats"]
    assert stats["averagedDraws"] == 2
    assert stats["tokens"] > 0


def test_averaging_two_draws_changes_the_estimate(tmp_path):
    one = _client(tmp_path / "one", draws=1).run(1, log=lambda *_: None)
    two = _client(tmp_path / "two", draws=2).run(1, log=lambda *_: None)
    single = float(one["rounds"][0]["gradNorm"])
    averaged = float(two["rounds"][0]["gradNorm"])
    assert single > 0 and averaged > 0
    # Averaging independent zero-mean contributions shrinks the norm; identical
    # values would mean the second draw was not actually independent.
    assert not np.isclose(single, averaged, rtol=1e-6)
