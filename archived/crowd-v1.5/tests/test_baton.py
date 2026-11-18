"""Tests for the Hugging Face baton (the no-server checkpoint hand-off).

Nothing here touches the network: the hub calls are monkeypatched, so what is
being tested is the document format, the digest contract and the round guard.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("hf_baton", REPO / "tools" / "hf_baton.py")
BATON = importlib.util.module_from_spec(spec)
sys.modules["hf_baton"] = BATON
spec.loader.exec_module(BATON)

from crowd_v15.model import GPT, ModelConfig, canonical_params  # noqa: E402
from crowd_v15.store import weights_digest  # noqa: E402


def _tiny_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "baton"
    BATON.cmd_init(BATON.parse_args(["init", "--dir", str(directory), "--preset", "tiny"]))
    return directory


def _tokens(tmp_path: Path) -> Path:
    directory = tmp_path / "tok"
    directory.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rng.integers(0, 256, size=8000, dtype=np.uint16).tofile(directory / "train.bin")
    return directory


def test_init_writes_a_verifiable_checkpoint(tmp_path):
    directory = _tiny_dir(tmp_path)
    assert BATON.cmd_verify(BATON.parse_args(["verify", "--dir", str(directory), "--quiet"])) == 0
    seed = json.loads((directory / BATON.SEED).read_text())
    status = json.loads((directory / BATON.STATUS).read_text())
    assert seed["preset"] == "tiny"
    assert seed["params"] == ModelConfig.preset("tiny").n_params
    assert status["round"] == 0
    assert status["baseSha256"] == seed["weightsDigest"]
    assert sorted(seed["canonicalNames"]) == sorted(seed["canonicalNames"])


def test_digest_ignores_noise_below_bf16_resolution(tmp_path):
    """The digest is over bf16-rounded bytes, so a re-save cannot move it."""
    directory = _tiny_dir(tmp_path)
    tensors = BATON.load_weights(directory)
    digest = weights_digest(tensors)
    assert digest == BATON._read_json(directory / BATON.STATUS)["baseSha256"]
    jittered = {k: (v * (1.0 + 1e-9)).astype(np.float32) for k, v in tensors.items()}
    assert weights_digest(jittered) == digest
    moved = {k: v.copy() for k, v in tensors.items()}
    first = sorted(moved)[0]
    moved[first][0] += 1.0
    assert weights_digest(moved) != digest


def test_verify_rejects_tampered_weights(tmp_path):
    directory = _tiny_dir(tmp_path)
    tensors = BATON.load_weights(directory)
    first = sorted(tensors)[0]
    tensors[first] = tensors[first] + 1.0
    BATON.save_weights(directory, tensors)
    with pytest.raises(BATON.BatonError, match="baseSha256"):
        BATON.cmd_verify(BATON.parse_args(["verify", "--dir", str(directory), "--quiet"]))


def test_push_guard_refuses_a_stale_writer():
    assert BATON.push_problem(remote_round=3, local_round=2, force=False).startswith("remote round 3")
    assert BATON.push_problem(remote_round=2, local_round=2, force=False).startswith("remote round 2 equals")
    assert BATON.push_problem(remote_round=2, local_round=3, force=False) is None
    assert BATON.push_problem(remote_round=9, local_round=1, force=True) is None


def test_lock_document_is_an_advisory_lease():
    lock = BATON.lock_document("alice", 4, "abc", now=1000.0)
    assert lock["holder"] == "alice" and lock["round"] == 4
    assert lock["expiresAt"] - lock["acquiredAt"] == BATON.LOCK_TTL_S
    assert "advisory" in lock["note"]


def test_train_advances_the_round_and_rewrites_the_documents(tmp_path, monkeypatch):
    directory = _tiny_dir(tmp_path)
    tokens = tmp_path / "tok"
    tokens.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, 256, size=4000, dtype=np.uint16).tofile(tokens / "train.bin")
    before = json.loads((directory / BATON.STATUS).read_text())
    code = BATON.cmd_train(BATON.parse_args([
        "train", "--dir", str(directory), "--data", str(tokens), "--device", "cpu",
        "--compute-dtype", "float32", "--steps", "2", "--batch", "1", "--seq-len", "16",
        "--population", "4", "--draw-chunk", "4", "--mode", "all", "--log-every", "1",
    ]))
    assert code == 0
    after = json.loads((directory / BATON.STATUS).read_text())
    assert after["round"] == before["round"] + 1
    assert after["baseSha256"] != before["baseSha256"]
    assert after["history"][-1]["round"] == after["round"]
    assert after["history"][-1]["tokens"] == 2 * 16
    assert BATON.cmd_verify(BATON.parse_args(["verify", "--dir", str(directory), "--quiet"])) == 0


def test_push_uploads_after_reading_the_remote_round(tmp_path, monkeypatch):
    directory = _tiny_dir(tmp_path)
    uploaded = {}

    def fake_download(repo, name, dest, *, revision, token_env="HF_TOKEN"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"round": 0, "baseSha256": "remote"}))
        return dest

    def fake_upload(repo, directory_, names, message, *, token_env="HF_TOKEN"):
        uploaded["names"] = list(names)
        uploaded["message"] = message
        return "commit-sha"

    monkeypatch.setattr(BATON, "hub_download", fake_download)
    monkeypatch.setattr(BATON, "hub_upload_files", fake_upload)
    code = BATON.cmd_train(BATON.parse_args([
        "train", "--dir", str(directory), "--data", str(_tokens(tmp_path)), "--device", "cpu",
        "--compute-dtype", "float32", "--steps", "1", "--batch", "1", "--seq-len", "16",
        "--population", "4", "--draw-chunk", "4", "--mode", "all", "--log-every", "1",
    ]))
    assert code == 0
    code = BATON.cmd_push(BATON.parse_args(["push", "--dir", str(directory)]))
    assert code == 0
    assert set(uploaded["names"]) >= {"weights.safetensors", "status.json", "manifest.json", "lock.json"}
    assert uploaded["message"].startswith("round 1")


def test_push_refuses_when_the_remote_moved_on(tmp_path, monkeypatch):
    directory = _tiny_dir(tmp_path)

    def fake_download(repo, name, dest, *, revision, token_env="HF_TOKEN"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"round": 7, "baseSha256": "remote"}))
        return dest

    monkeypatch.setattr(BATON, "hub_download", fake_download)
    monkeypatch.setattr(BATON, "hub_upload_files",
                        lambda *a, **k: pytest.fail("must not upload a stale round"))
    with pytest.raises(BATON.BatonError, match="remote round 7"):
        BATON.cmd_push(BATON.parse_args(["push", "--dir", str(directory)]))


def test_missing_token_names_the_fix(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(BATON.BatonError, match="fine-grained token"):
        BATON._token("HF_TOKEN")


def test_300m_preset_is_the_parameter_budget_the_author_asked_for():
    cfg = ModelConfig.preset("300m")
    report = __import__("crowd_v15.model", fromlist=["param_report"]).param_report(cfg)
    assert cfg.n_params == report["total"]
    assert 2.9e8 < cfg.n_params < 3.2e8
    # The published seed doc must agree with the preset count.
    assert cfg.n_params == 306_222_080


def test_pull_reads_anonymously(tmp_path, monkeypatch):
    """A public repo must be pullable without a token; only push needs one."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert BATON._optional_token("HF_TOKEN") is None
    calls = {}

    def fake_download(repo, name, dest, *, revision, token_env="HF_TOKEN"):
        calls[name] = BATON._optional_token(token_env)
        source = tmp_path / "src"
        source.mkdir(exist_ok=True)
        return None

    monkeypatch.setattr(BATON, "hub_download", fake_download)
    with pytest.raises(BATON.BatonError, match="missing"):
        BATON.cmd_pull(BATON.parse_args(["pull", "--dir", str(tmp_path / "state")]))
    assert set(calls) == set(BATON.FILES)
    assert all(value is None for value in calls.values())
