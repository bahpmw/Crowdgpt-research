import torch

from crowd_v2.crypto import Identity, canonical, hash_tensors, key_id, merkle_root, open_envelope, verify


def test_sign_verify_and_tamper():
    ident = Identity.generate()
    env = ident.envelope({"type": "work", "round": 3})
    kid, body = open_envelope(env)
    assert kid == ident.kid == key_id(ident.pub)
    assert body == {"type": "work", "round": 3}
    env["body"]["round"] = 4
    try:
        open_envelope(env)
        raise AssertionError("tampered body accepted")
    except ValueError as exc:
        assert "signature" in str(exc)


def test_signature_from_other_key_rejected():
    a, b = Identity.generate(), Identity.generate()
    sig = a.sign(canonical({"x": 1}))
    assert verify(a.pub, canonical({"x": 1}), sig)
    assert not verify(b.pub, canonical({"x": 1}), sig)
    assert not verify(a.pub, canonical({"x": 1}), "00" * 64)


def test_key_file_roundtrip(tmp_path):
    a = Identity.load_or_create(tmp_path / "k")
    b = Identity.load_or_create(tmp_path / "k")
    assert a.kid == b.kid and len(a.kid) == 16


def test_hash_tensors_order_independent_and_sensitive():
    x, y = torch.arange(6.0).reshape(2, 3), torch.ones(4)
    assert hash_tensors({"a": x, "b": y}) == hash_tensors({"b": y, "a": x})
    assert hash_tensors({"a": x, "b": y}) == hash_tensors({"a": x.bfloat16().float(), "b": y})
    assert hash_tensors({"a": x}) != hash_tensors({"a": x.reshape(3, 2)})
    z = x.clone()
    z[0, 0] = 1e-7
    assert hash_tensors({"a": x}) != hash_tensors({"a": z})


def test_merkle_root_is_order_free():
    assert merkle_root(["b" * 64, "a" * 64, "c" * 64]) == merkle_root(["c" * 64, "a" * 64, "b" * 64])
    assert merkle_root(["a" * 64]) != merkle_root(["b" * 64])
