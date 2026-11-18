"""The audit primitive: a coordinator re-scores a DUST submission."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from crowd_v15 import audit, protocol
from crowd_v15.client import ClientConfig, DustClient
from crowd_v15.coordinator import CoordConfig, Coordinator, Rejected
from crowd_v15.dust import DustConfig, backprop_gradients, dust_gradients
from crowd_v15.store import ModelStore

from .helpers import model_tensors, tiny_model


def _client(**overrides) -> DustClient:
    config = {"client_id": "honest", "offline": True, "task": "induction", "batch": 2,
              "seq_len": 16, "population": 8, "draw_chunk": 4, "sigma": 1e-2,
              "codec": "dense", "seed": 3}
    config.update(overrides)
    torch.manual_seed(0)
    client = DustClient(ClientConfig(**config))
    client.sync()
    return client


def _weights(client: DustClient) -> dict[str, np.ndarray]:
    return model_tensors(client.model)


def _body(client: DustClient, grads=None, stats=None) -> tuple[dict, np.ndarray]:
    if grads is None:
        grads, stats = client.gradient(step=0)
    body, _ = client.encode(grads, stats)
    return protocol.decode_body(body, max_params=10**8)


def test_layer_subset_rows_are_bit_identical_to_the_full_estimate():
    model = tiny_model()
    cfg = DustConfig(population=8, draw_chunk=4, sigma=1e-2)
    idx = torch.randint(0, model.cfg.vocab_size, (2, 16), generator=torch.Generator().manual_seed(0))
    tgt = torch.roll(idx, -1, dims=1)
    full, _ = dust_gradients(model, cfg, idx, tgt, base_seed=5, step=0)
    pick = ["blocks.1.attn.o", "blocks.3.mlp.down"]
    part, _ = dust_gradients(model, cfg, idx, tgt, base_seed=5, step=0, layers=pick)
    for name in pick:
        assert torch.equal(full[f"{name}.weight"], part[f"{name}.weight"])


def test_honest_submission_passes_a_full_rescore():
    client = _client()
    header, vector = _body(client)
    result = audit.rescore(_weights(client), header, vector)
    assert result["verdict"] == "pass"
    assert result["cosine"] > 0.999 and result["headCosine"] > 0.999
    assert result["full"] is True


def test_honest_submission_passes_a_partial_rescore_through_a_lossy_codec():
    client = _client(codec="ngd1r")
    header, vector = _body(client)
    result = audit.rescore(_weights(client), header, vector, layers=3, salt="s")
    assert result["verdict"] == "pass", result
    assert result["layersChecked"] == 3 and result["full"] is False
    assert result["forwards"] < 2 * 3 * 8 + 8


def test_pick_layers_depends_on_the_salt():
    shapes = [f"layer{i}" for i in range(20)]
    a = audit.pick_layers(shapes, 4, salt="one", client_id="c", round_id=0)
    b = audit.pick_layers(shapes, 4, salt="two", client_id="c", round_id=0)
    assert len(a) == 4 and a != b
    assert a == sorted(a, key=shapes.index)


@pytest.mark.parametrize("cheat", ["random", "other_seed", "backprop"])
def test_fabricated_estimates_fail(cheat):
    client = _client()
    grads, stats = client.gradient(step=0)
    if cheat == "random":
        gen = torch.Generator().manual_seed(9)
        grads = {k: torch.randn(v.shape, generator=gen) for k, v in grads.items()}
    elif cheat == "other_seed":
        idx, tgt = client.batch()
        grads, _ = dust_gradients(client.model, client.dust_cfg, idx, tgt,
                                  base_seed=stats["baseSeed"] + 1, step=0)
    else:
        idx, tgt = client.batch()
        grads = backprop_gradients(client.model, idx, tgt)
    header, vector = _body(client, grads, stats)
    result = audit.rescore(_weights(client), header, vector)
    assert result["verdict"] == "fail", result


def test_unreplayable_submissions_are_skipped():
    client = _client()
    header, vector = _body(client)
    header["replay"]["noiseDevice"] = "device"
    assert audit.rescore(_weights(client), header, vector)["verdict"] == "skipped"
    del header["replay"]
    assert audit.rescore(_weights(client), header, vector)["verdict"] == "skipped"


def test_wrong_preset_fails():
    client = _client()
    header, vector = _body(client)
    header["replay"]["preset"] = "10m"
    result = audit.rescore(_weights(client), header, vector)
    assert result["verdict"] == "fail"


def _coordinator(tmp_path, client, **config):
    coord = Coordinator(ModelStore(tmp_path / "state"),
                        CoordConfig(min_submissions=1, audit_layers=4, **config))
    coord.bootstrap(_weights(client))
    assert coord.base_sha256 == client.base_sha256
    return coord


def test_coordinator_accepts_an_honest_client_and_records_the_audit(tmp_path):
    client = _client()
    coord = _coordinator(tmp_path, client)
    grads, stats = client.gradient(step=0)
    body, _ = client.encode(grads, stats)
    receipt = coord.submit(body)
    assert receipt["accepted"] and receipt["audit"]["verdict"] == "pass"


def test_coordinator_rejects_a_fabricated_estimate(tmp_path):
    client = _client()
    coord = _coordinator(tmp_path, client)
    grads, stats = client.gradient(step=0)
    gen = torch.Generator().manual_seed(1)
    fake = {k: torch.randn(v.shape, generator=gen) for k, v in grads.items()}
    body, _ = client.encode(fake, stats)
    with pytest.raises(Rejected) as err:
        coord.submit(body)
    assert err.value.reason == "audit_failed"


def test_audit_require_refuses_what_cannot_be_rescored(tmp_path):
    client = _client(local_steps=2)
    coord = _coordinator(tmp_path, client, audit_require=True)
    grads, stats = client.gradient(step=0)
    body, _ = client.encode(grads, stats)
    with pytest.raises(Rejected) as err:
        coord.submit(body)
    assert err.value.reason == "unauditable"


def test_bf16_from_another_device_is_not_rescored_on_this_one():
    client = _client()
    header, vector = _body(client)
    header["replay"].update(computeDtype="bfloat16", device="mps")
    result = audit.rescore(_weights(client), header, vector)
    assert result["verdict"] == "skipped" and "mps" in result["reason"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_float32_mps_client_is_rescored_exactly_on_cpu():
    client = _client(device="mps")
    header, vector = _body(client)
    result = audit.rescore(_weights(client), header, vector, layers=3, salt="x")
    assert result["verdict"] == "pass" and result["cosine"] > 0.999
