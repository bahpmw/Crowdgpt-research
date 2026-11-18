import pytest
import torch

from conftest import TINY
from crowd_v2.audit import Thresholds, compare, layer0, layer3
from crowd_v2.cell import run_cell, wire_bytes
from crowd_v2.client import corrupt
from crowd_v2.nets import init_weights, module_names


@pytest.fixture(scope="module")
def honest(tokens):
    w = init_weights(TINY)
    return w, {k: run_cell(TINY, w, k, 77, {}, tokens) for k in range(TINY.K)}


def test_replay_equals_original_on_same_device(honest, tokens):
    w, outs = honest
    for k, out in outs.items():
        again = run_cell(TINY, w, k, 77, {}, tokens)
        assert compare(again.msg, out.msg) == (1.0, 1.0)
        assert all(torch.equal(again.msg[n], out.msg[n]) for n in out.msg)


def test_replay_with_error_feedback_is_exact(honest, tokens):
    w, outs = honest
    ef = outs[0].ef
    a = run_cell(TINY, w, 0, 78, ef, tokens)
    b = run_cell(TINY, w, 0, 78, ef, tokens)
    assert compare(a.msg, b.msg) == (1.0, 1.0)
    assert compare(a.msg, run_cell(TINY, w, 0, 78, {}, tokens).msg)[0] < 0.999


@pytest.mark.parametrize("attack", ["flip", "scale", "noise", "zero"])
def test_each_message_attack_fails_replay(honest, attack):
    _, outs = honest
    th = Thresholds()
    for out in outs.values():
        cos, ratio = compare(corrupt(out.msg, attack, 1), out.msg)
        assert not th.passes(cos, ratio), (attack, cos, ratio)


def test_wrongdata_stale_and_copy_fail_replay(honest, tokens):
    w, outs = honest
    th = Thresholds()
    wrong = run_cell(TINY, w, 0, 999, {}, tokens).msg
    stale_w = {n: v + 0.01 * torch.randn_like(v) for n, v in w.items()}
    stale = run_cell(TINY, stale_w, 0, 77, {}, tokens).msg
    copied = run_cell(TINY, w, 0, 31337, {}, tokens).msg
    for name, msg in (("wrongdata", wrong), ("stale", stale), ("copy", copied)):
        cos, ratio = compare(msg, outs[0].msg)
        assert not th.passes(cos, ratio), (name, cos, ratio)


def test_message_shape_and_wire_size(honest):
    w, outs = honest
    names = module_names(TINY)
    for k, out in outs.items():
        shapes = {n: tuple(w[n].shape) for n in names[k]}
        assert layer0(out.msg, shapes) is None
        assert all(v.dtype == torch.bfloat16 for v in out.msg.values())
        dense = sum(v.numel() for v in out.msg.values()) * 2
        assert 0 < wire_bytes(shapes, TINY.topk) < dense / 10
        bad = dict(out.msg)
        bad[names[k][0]] = torch.full_like(bad[names[k][0]], float("nan"))
        assert layer0(bad, shapes) == "layer0:nonfinite"
        assert layer0({n: v for n, v in list(out.msg.items())[1:]}, shapes) == "layer0:names"


def test_layer3_flags_oversized_and_opposite(honest, tokens):
    w, outs = honest
    msgs = {f"h{i}": run_cell(TINY, w, 1, 200 + i, {}, tokens).msg for i in range(3)}
    msgs["big"] = corrupt(msgs["h0"], "scale", 0)
    msgs["neg"] = {n: -(msgs["h1"][n] + msgs["h2"][n]) for n in msgs["h1"]}
    msgs["h3"] = run_cell(TINY, w, 1, 300, {}, tokens).msg
    bad = layer3(msgs, trusted={"h0", "h1", "h2"})
    assert bad == {"big": "layer3:norm", "neg": "layer3:direction"}
    assert layer3(msgs, trusted=set()) == {"big": "layer3:norm"}
