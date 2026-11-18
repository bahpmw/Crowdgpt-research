import torch

from crowd_v2 import beacon
from crowd_v2.crypto import sha256_hex


def test_assignment_deterministic_and_covers_every_module():
    b = beacon.beacon("a" * 64, 3)
    keys = [sha256_hex(str(i))[:16] for i in range(7)]
    one = beacon.assign(b, keys, 3)
    two = beacon.assign(b, list(reversed(keys)), 3)
    assert one == two
    counts = [sum(1 for m in one.values() if m == k) for k in range(3)]
    assert set(one) == set(keys) and min(counts) >= 2 and max(counts) - min(counts) <= 1
    assert beacon.assign(beacon.beacon("a" * 64, 4), keys, 3) != one


def test_beacon_changes_with_round_and_source():
    assert beacon.beacon("a" * 64, 1) != beacon.beacon("a" * 64, 2)
    assert beacon.beacon("a" * 64, 1) != beacon.beacon("b" * 64, 1)


def test_data_seed_reproducible_and_per_client(tokens):
    b = beacon.beacon("c" * 64, 1)
    s1, s2 = beacon.data_seed(b, "k1"), beacon.data_seed(b, "k2")
    assert s1 == beacon.data_seed(b, "k1") and s1 != s2
    x = tokens.batches(s1, 2, 4)
    y = tokens.batches(s1, 2, 4)
    z = tokens.batches(s2, 2, 4)
    assert all(torch.equal(a[0], c[0]) for a, c in zip(x, y))
    assert not torch.equal(x[0][0], z[0][0])


def test_audit_sampling_rate_and_auditor_excludes_receiver():
    seed = beacon.audit_seed("d" * 64, "e" * 64)
    keys = [sha256_hex(str(i))[:16] for i in range(400)]
    rate = sum(beacon.sampled(seed, k, 0.5) for k in keys) / len(keys)
    assert 0.4 < rate < 0.6
    assert all(beacon.sampled(seed, k, 1.0) for k in keys)
    order = beacon.auditor_order(seed, keys[0], ["o1", "o2", "o3"], "o2")
    assert sorted(order) == ["o1", "o3"]
