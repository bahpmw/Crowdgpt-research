import json

import pytest

from crowd_v2.crypto import Identity
from crowd_v2.record import Record, chain_entries, verify_chain


@pytest.fixture()
def net(tmp_path):
    ids = [Identity.generate() for _ in range(3)]
    keys = {i.kid: i.pub for i in ids}
    rec = Record(tmp_path / "record.jsonl", keys, quorum=2)
    g = chain_entries("0" * 64, 0, [("genesis", 0, {"orchestrators": keys, "quorum": 2})])[0]
    g["sigs"] = {i.kid: i.sign(g["hash"].encode()) for i in ids}
    rec.append(g)
    return ids, keys, rec


def _add(rec, ids, signers, rnd):
    e = chain_entries(rec.tail_hash, rec.next_n, [("round", rnd, {"eval_loss": 5.0, "cells": []})])[0]
    e["sigs"] = {ids[i].kid: ids[i].sign(e["hash"].encode()) for i in signers}
    rec.append(e)
    return e


def test_chain_verifies_and_counts_finality(net):
    ids, keys, rec = net
    _add(rec, ids, [0, 1], 1)
    _add(rec, ids, [2], 2)
    rep = verify_chain(rec.path, keys)
    assert rep == {"ok": True, "entries": 3, "final": 2, "provisional": 1, "errors": []}
    assert rec.is_final(rec.entries[1]) and not rec.is_final(rec.entries[2])
    assert rec.last_final()["n"] == 1


def test_late_cosign_makes_entry_final(net):
    ids, keys, rec = net
    e = _add(rec, ids, [0], 1)
    assert not rec.is_final(rec.entries[1])
    assert rec.add_sigs(1, {ids[1].kid: ids[1].sign(e["hash"].encode())})
    assert verify_chain(rec.path, keys)["final"] == 2
    assert not rec.add_sigs(1, {ids[2].kid: "ab" * 64})


def test_tampered_body_is_detected(net):
    ids, keys, rec = net
    _add(rec, ids, [0, 1], 1)
    _add(rec, ids, [0, 1], 2)
    lines = rec.path.read_text().splitlines()
    e = json.loads(lines[1])
    e["body"]["eval_loss"] = 1.0
    lines[1] = json.dumps(e)
    rec.path.write_text("\n".join(lines) + "\n")
    rep = verify_chain(rec.path, keys)
    assert not rep["ok"]
    assert any("hash mismatch" in x for x in rep["errors"])


def test_forged_signature_and_reordering_detected(net):
    ids, keys, rec = net
    _add(rec, ids, [0, 1], 1)
    _add(rec, ids, [0, 1], 2)
    lines = rec.path.read_text().splitlines()
    e = json.loads(lines[2])
    e["sigs"][ids[2].kid] = "cd" * 64
    lines[2] = json.dumps(e)
    rec.path.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n")
    rep = verify_chain(rec.path, keys)
    assert any("invalid signature" in x for x in rep["errors"])
    assert any("broken link" in x for x in rep["errors"])


def test_append_refuses_unlinked_or_unsigned(net):
    ids, keys, rec = net
    e = chain_entries("f" * 64, rec.next_n, [("round", 1, {})])[0]
    e["sigs"] = {ids[0].kid: ids[0].sign(e["hash"].encode())}
    with pytest.raises(ValueError):
        rec.append(e)
    e = chain_entries(rec.tail_hash, rec.next_n, [("round", 1, {})])[0]
    with pytest.raises(ValueError):
        rec.append(e)


def test_record_reloads_from_disk(net):
    ids, keys, rec = net
    _add(rec, ids, [0, 1], 1)
    again = Record(rec.path, keys, 2)
    assert again.tail_hash == rec.tail_hash and again.next_n == 2
