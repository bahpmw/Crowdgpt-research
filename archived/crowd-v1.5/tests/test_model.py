"""The model is a plain transformer, and its parameter arithmetic is honest.

The author's constraint is explicit: no engrams, because a sparse table with its
own update rule cannot be trained by a zeroth-order estimate.  These tests pin
that as a property of the code rather than a promise in a README.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from crowd_v15.model import (
    GPT,
    ModelConfig,
    assert_plain,
    canonical_params,
    check_plain_parameter_names,
    param_report,
)

from .helpers import model_tensors, safetensors_bytes


def test_every_preset_has_matching_parameter_arithmetic():
    for name in ("tiny", "10m", "100m", "1b", "1b-deep", "2b"):
        cfg = ModelConfig.preset(name)
        assert cfg.n_params == param_report(cfg)["total"], name


def test_built_models_agree_with_the_analytic_count():
    for name in ("tiny", "10m"):
        cfg = ModelConfig.preset(name)
        torch.manual_seed(0)
        assert GPT(cfg).n_params() == cfg.n_params, name


def test_the_one_billion_preset_is_actually_a_billion():
    cfg = ModelConfig.preset("1b")
    # 15 layers x 2048 wide, tied embedding: 1,015,085,056 parameters.
    assert cfg.n_params == 1_015_085_056
    assert abs(cfg.n_params - 1e9) / 1e9 < 0.02
    assert cfg.n_layers == 15 and cfg.d_model == 2048 and cfg.n_heads == 16


def test_the_maintainer_sizing_is_kept_for_comparison():
    cfg = ModelConfig.preset("1b-deep")
    assert cfg.n_layers == 20 and cfg.d_model == 2048
    assert cfg.n_params == 1_350_649_856


def test_plain_module_tree_and_breakdown():
    cfg = ModelConfig.preset("tiny")
    torch.manual_seed(0)
    model = GPT(cfg)
    breakdown = assert_plain(model)
    assert breakdown["total"] == model.n_params()
    assert breakdown["embedding"] + breakdown["attention"] + breakdown["mlp"] + breakdown["norms"] == breakdown["total"]
    # Nothing but the plain stack: no table, no gate network, no adapter.
    names = set(dict(model.named_modules()))
    assert not [n for n in names if any(m in n.lower() for m in ("engram", "moe", "expert", "lora"))]


def test_tied_embedding_is_counted_once():
    cfg = ModelConfig.preset("tiny")
    torch.manual_seed(0)
    model = GPT(cfg)
    assert model.head.weight is model.tok_emb.weight
    canonical = canonical_params(model)
    # One entry for the shared storage, listed under the first registered name.
    assert "head.weight" not in canonical
    assert canonical["tok_emb.weight"] is model.head.weight
    assert len(canonical) == len(dict(model.named_parameters()))


@pytest.mark.parametrize(
    "names",
    [
        ["engram.table.weight", "tok_emb.weight"],
        ["blocks.0.mlp.experts.0.w1", "tok_emb.weight"],
        ["blocks.0.attn.qkv.lora_A", "tok_emb.weight"],
        ["router.gate", "tok_emb.weight"],
        ["ENGRAM.table.weight"],
    ],
)
def test_non_plain_parameter_names_are_refused(names):
    with pytest.raises(ValueError, match="plain transformer"):
        check_plain_parameter_names(names)


@pytest.mark.parametrize(
    "names",
    [
        ["tok_emb.weight", "blocks.0.attn.qkv.weight"],
        ["norm_f.weight"],
        ["blocks.3.mlp.up.weight", "blocks.3.mlp.down.weight"],
    ],
)
def test_plain_names_are_accepted(names):
    check_plain_parameter_names(names)


def test_rejection_lists_the_offending_tables():
    with pytest.raises(ValueError) as exc:
        check_plain_parameter_names(["engram.table.weight", "moe.gate.weight"])
    message = str(exc.value)
    assert "engram.table.weight" in message
    assert "moe.gate.weight" in message


def test_bootstrap_refuses_a_checkpoint_with_an_engram(tmp_path):
    from crowd_v15.coordinator import CoordConfig, Coordinator
    from crowd_v15.store import ModelStore

    tensors = model_tensors()
    tensors["engram.table.weight"] = np.zeros((8, 4), dtype=np.float32)
    coord = Coordinator(ModelStore(tmp_path / "s"), CoordConfig())
    with pytest.raises(ValueError, match="plain transformer"):
        coord.bootstrap(tensors)
    assert not (tmp_path / "s" / "weights.safetensors").exists()
