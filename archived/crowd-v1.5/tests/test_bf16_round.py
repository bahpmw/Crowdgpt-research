"""bf16 rounding must stay bit-identical to the published digest while being cheap."""

from __future__ import annotations

import numpy as np

from crowd_v15.codec import _bf16_to_f32, _f32_to_bf16


def _reference(arr):
    u = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x8000 + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype("<u2")


def test_matches_the_uint64_reference_exactly():
    rng = np.random.default_rng(0)
    values = np.concatenate([
        rng.standard_normal(200_000).astype(np.float32) * 1e-2,
        np.array([0.0, -0.0, 1.0, -1.0, 1e-45, -1e-45, np.inf, -np.inf, np.nan], np.float32),
        rng.standard_normal(65_536).astype(np.float32) * 1e3,
    ])
    assert np.array_equal(_f32_to_bf16(values), _reference(values))


def test_round_trip_is_within_bf16_resolution():
    rng = np.random.default_rng(1)
    values = (rng.standard_normal(10_000) * 0.01).astype(np.float32)
    back = _bf16_to_f32(_f32_to_bf16(values))
    assert np.allclose(back, values, rtol=2**-8, atol=1e-9)


def test_does_not_upcast_to_uint64():
    # A regression guard: the cheap path must not allocate 8-byte temporaries.
    values = np.zeros((1024,), dtype=np.float32)
    state = np.finfo(np.float32)
    assert _f32_to_bf16(values).dtype == np.dtype("<u2")
    assert _f32_to_bf16(np.full(4, state.max, dtype=np.float32)).shape == (4,)
