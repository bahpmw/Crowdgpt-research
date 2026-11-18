"""Wire-format tests: the container, the header guards and the round trip."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from crowd_v15 import codec, protocol


def _header(**overrides):
    base = dict(
        route="dust",
        client_id="c1",
        round_id=0,
        base_sha256="a" * 64,
        names=["w"],
        sizes=[5000],
        codec_name="ngd1",
        tokens=256,
        population=8,
    )
    base.update(overrides)
    return protocol.build_header(**base)


@pytest.mark.parametrize("name", ["dense", "ngd1", "ngd1r"])
def test_round_trip_preserves_the_direction(name):
    values = (np.random.default_rng(7).standard_normal(5000) * 0.01).astype(np.float32)
    header = _header(codec_name=name)
    decoded_header, decoded = protocol.decode_body(
        protocol.encode_body(header, values), max_params=10_000
    )
    assert decoded_header["codec"] == name
    assert decoded.shape == values.shape
    # A gradient is used for its sign and block magnitude, so compare that way.
    agree = float(np.mean(np.sign(decoded) == np.sign(values)))
    assert agree > 0.98, agree
    if name == "dense":
        # bf16 keeps ~8 mantissa bits, so compare relatively; the point of the
        # dense codec is that no quantisation beyond the wire dtype happens.
        assert np.allclose(decoded, values, rtol=1e-2, atol=0.0)
    else:
        # The 1-bit codec must at least keep the block magnitude in the ballpark.
        assert 0.3 < float(np.abs(decoded).mean() / np.abs(values).mean()) < 3.0


def test_residual_codec_is_more_accurate_and_bigger():
    values = (np.random.default_rng(3).standard_normal(4096) * 0.02).astype(np.float32)
    plain = codec.decompress_frame(codec.compress_frame(values))
    residual = codec.decompress_frame(codec.compress_frame(values, residual=True))
    plain_mae = float(np.abs(plain - values).mean())
    residual_mae = float(np.abs(residual - values).mean())
    assert residual_mae < plain_mae / 1.4
    assert len(codec.compress_frame(values, residual=True)) > 1.5 * len(codec.compress_frame(values))


def test_container_magic_follows_the_codec():
    values = np.zeros(2048, dtype=np.float32)
    assert codec.pack_frames([values])[:4] == codec.FRAME_MAGIC
    assert codec.pack_frames([values], residual=True)[:4] == codec.FRAME_MAGIC_RESIDUAL


def test_container_mismatch_is_rejected():
    values = np.zeros(2048, dtype=np.float32)
    with pytest.raises(ValueError):
        list(codec.iter_frames(codec.pack_frames([values]), expect_residual=True))
    with pytest.raises(ValueError):
        list(codec.iter_frames(codec.pack_frames([values], residual=True), expect_residual=False))


def test_trailing_bytes_are_rejected():
    values = np.zeros(2048, dtype=np.float32)
    blob = codec.pack_frames([values]) + b"\x00"
    with pytest.raises(ValueError):
        list(codec.iter_frames(blob))


def test_dense_body_must_be_the_declared_size():
    header = _header(codec_name="dense")
    body = protocol.encode_body(header, np.zeros(5000, dtype=np.float32))
    broken = body[:-3]
    with pytest.raises(protocol.ProtocolError, match="dense body"):
        protocol.decode_body(broken, max_params=10_000)


def test_bit_flip_in_a_scale_is_caught():
    # A corrupted scale previously decoded to NaN without complaint.
    raw = bytearray(codec.compress_frame(np.zeros(2048, dtype=np.float32)))
    import zlib

    plain = bytearray(zlib.decompress(bytes(raw)))
    # n=2048 -> 2 blocks -> scales occupy bytes 16:20 of the inflated frame.
    plain[16:18] = b"\xff\xff"
    with pytest.raises(ValueError):
        codec.decompress_frame(zlib.compress(bytes(plain)))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"names": ["w", "w"], "sizes": [2500, 2500]}, "duplicate"),
        ({"names": ["w"], "sizes": [0]}, "positive"),
        ({"route": "weights"}, "unknown route"),
        ({"codec": "gz"}, "unknown codec"),
    ],
)
def test_header_guards(overrides, message):
    """Guards must hold on decode even if a peer never ran build_header."""
    header = dict(_header())
    header.update(overrides)
    raw = json.dumps(header).encode()
    body = struct.pack("<I", len(raw)) + raw + b""
    with pytest.raises(protocol.ProtocolError, match=message):
        protocol.decode_body(body, max_params=10_000)


def test_build_header_refuses_unknown_values():
    with pytest.raises(protocol.ProtocolError, match="unknown route"):
        _header(route="weights")
    with pytest.raises(protocol.ProtocolError, match="unknown codec"):
        _header(codec_name="gz")


def test_params_must_match_the_size_list():
    header = _header()
    header["params"] = 4999
    body = struct.pack("<I", len(json.dumps(header).encode())) + json.dumps(header).encode()
    with pytest.raises(protocol.ProtocolError, match="sum\\(sizes\\)"):
        protocol.decode_body(body, max_params=10_000)


def test_parameter_budget_is_enforced():
    header = _header(sizes=[50_000])
    body = protocol.encode_body(header, np.zeros(50_000, dtype=np.float32))
    with pytest.raises(protocol.ProtocolError, match="exceeds the model"):
        protocol.decode_body(body, max_params=1000)


def test_header_must_be_json():
    with pytest.raises(protocol.ProtocolError, match="header is not JSON"):
        protocol.decode_body(struct.pack("<I", 5) + b"{{{{{", max_params=100)


def test_implausible_header_length():
    with pytest.raises(protocol.ProtocolError, match="implausible header length"):
        protocol.decode_body(struct.pack("<I", 1 << 20) + b"x", max_params=100)


def test_oversized_body_is_refused_before_parsing():
    header = _header()
    body = protocol.encode_body(header, np.zeros(5000, dtype=np.float32))
    with pytest.raises(protocol.ProtocolError, match="limit"):
        protocol.decode_body(body, max_params=10_000, max_body_bytes=64)


def test_flatten_unflatten_round_trip():
    import torch

    grads = {"b": torch.ones(3), "a": torch.arange(4, dtype=torch.float32)}
    names, sizes, flat = protocol.flatten(grads)
    assert names == ["a", "b"]
    back = protocol.unflatten(flat, names, sizes)
    assert np.allclose(back["a"], np.arange(4))
    assert np.allclose(back["b"], np.ones(3))


def test_unflatten_rejects_a_short_vector():
    with pytest.raises(protocol.ProtocolError):
        protocol.unflatten(np.zeros(2, dtype=np.float32), ["a"], [3])


def test_estimator_fields_are_defensive():
    assert protocol.estimator_fields({"tokens": "abc", "sigma": float("nan")}) == {
        "tokens": 0.0, "population": 0.0, "sigma": 0.0, "gamma": 0.0,
    }
    assert protocol.estimator_fields({"tokens": 10, "sigma": 1e-3})["tokens"] == 10.0
