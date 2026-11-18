"""Tests for the DUST estimator: correctness, reproducibility, and learning.

Small models on CPU, so the whole file runs in seconds. The decisive tests are
``test_one_step_matches_backprop_descent`` (the estimate must move the loss the
way the true gradient would) and the per-layer isolation comparison.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch

# Tiny tensors are dominated by thread hand-off, not arithmetic.
torch.set_num_threads(1)

from crowd_v15.dust import (
    DustConfig,
    DustOptimizer,
    backprop_gradients,
    derive_seed,
    dust_gradients,
    dust_step,
    probe_cosine,
    reverse_discounted_cumsum,
)
from crowd_v15.model import GPT, ModelConfig, Perturb
from crowd_v15.tasks import (
    identity_batch,
    induction_batch,
    load_tokens,
    make_source,
    token_batch,
)

TINY = dict(vocab_size=32, n_layers=2, d_model=64, n_heads=4, context_length=32)
TOY = dict(vocab_size=4, n_layers=1, d_model=32, n_heads=4, context_length=16)


def build(config: dict = TINY, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(ModelConfig(**config))


def config(**overrides) -> DustConfig:
    base = dict(population=8, draw_chunk=8, sigma=1e-3, gamma=0.9, lr=1e-2)
    base.update(overrides)
    return DustConfig(**base)


def batch(seq_len: int = 16, n: int = 2, seed: int = 3, vocab: int = 32):
    return induction_batch(n, seq_len, vocab, torch.device("cpu"),
                           torch.Generator().manual_seed(seed), period=2)


def loss_on(model: GPT, idx: torch.Tensor, tgt: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(idx, None)
        return float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)))


class NoiseTests(unittest.TestCase):
    def test_seed_is_stable_and_depends_on_every_input(self):
        base = derive_seed(7, 0, 0, "blocks.0.attn.qkv")
        self.assertEqual(base, derive_seed(7, 0, 0, "blocks.0.attn.qkv"))
        for other in (derive_seed(7, 1, 0, "blocks.0.attn.qkv"),
                      derive_seed(7, 0, 1, "blocks.0.attn.qkv"),
                      derive_seed(8, 0, 0, "blocks.0.attn.qkv"),
                      derive_seed(7, 0, 0, "blocks.0.mlp.up")):
            self.assertNotEqual(base, other)

    def test_noise_shapes_cover_every_linear_plus_the_embedding(self):
        model = build()
        shapes = model.noise_shapes()
        self.assertEqual(shapes["tok_emb"], TINY["d_model"])
        self.assertEqual(shapes["blocks.0.attn.qkv"], 3 * TINY["d_model"])
        self.assertEqual(shapes["blocks.1.mlp.down"], TINY["d_model"])
        self.assertEqual(len(shapes), 5 * TINY["n_layers"] + 1)

    def test_zero_noise_leaves_logits_untouched_and_real_noise_moves_them(self):
        model = build()
        idx, _tgt = batch()
        zeros = {n: torch.zeros((idx.shape[0], idx.shape[1], w))
                 for n, w in model.noise_shapes().items()}
        ones = {n: torch.ones_like(t) for n, t in zeros.items()}
        with torch.no_grad():
            clean = model(idx, None)
            same = model(idx, Perturb(enabled=True, sigma=1e-3, sign=1, noise=zeros))
            moved = model(idx, Perturb(enabled=True, sigma=1e-3, sign=1, noise=ones))
        self.assertTrue(torch.allclose(clean, same, atol=1e-6))
        self.assertFalse(torch.allclose(clean, moved, atol=1e-6))

    def test_only_limits_the_injected_layers(self):
        model = build()
        idx, _tgt = batch()
        noise = {"blocks.0.attn.o": torch.ones((idx.shape[0], idx.shape[1], TINY["d_model"]))}
        with torch.no_grad():
            clean = model(idx, None)
            one_layer = model(idx, Perturb(enabled=True, sigma=1e-3, sign=1, noise=noise,
                                           only="blocks.0.attn.o"))
        self.assertFalse(torch.allclose(clean, one_layer, atol=1e-6))
        # Perturbing a layer that is not in the dict must be inert.
        with torch.no_grad():
            other = model(idx, Perturb(enabled=True, sigma=1e-3, sign=1, noise=noise,
                                       only="blocks.1.mlp.up"))
        self.assertTrue(torch.allclose(clean, other, atol=1e-6))

    def test_records_clean_activations_and_token_ids(self):
        model = build()
        idx, _tgt = batch()
        record: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            model(idx, Perturb(record=record))
        self.assertTrue(torch.equal(record["tok_emb"], idx))
        self.assertIn("__head__", record)
        self.assertEqual(record["blocks.0.mlp.up"].shape,
                         (idx.shape[0], idx.shape[1], TINY["d_model"]))


class EstimatorTests(unittest.TestCase):
    def test_chunking_cannot_change_the_estimate(self):
        model = build()
        idx, tgt = batch()
        whole, _ = dust_gradients(model, config(population=8, draw_chunk=8),
                                  idx, tgt, base_seed=0, step=0)
        split, _ = dust_gradients(model, config(population=8, draw_chunk=2),
                                  idx, tgt, base_seed=0, step=0)
        for name, grad in whole.items():
            # Same draws, only the accumulation order differs. Normalise by the
            # largest entry so near-zero entries cannot fail on absolute noise;
            # measured worst case is ~1e-5 of the tensor scale.
            scale = float(grad.abs().max())
            if scale == 0.0:
                self.assertTrue(torch.equal(grad, split[name]), name)
                continue
            drift = float((grad - split[name]).abs().max())
            self.assertLess(drift, 1e-3 * scale, f"{name}: {drift:.3e} vs {scale:.3e}")

    def test_estimate_is_reproducible_for_a_seed(self):
        model = build()
        idx, tgt = batch()
        first, _ = dust_gradients(model, config(), idx, tgt, base_seed=5, step=0)
        again, _ = dust_gradients(model, config(), idx, tgt, base_seed=5, step=0)
        other, _ = dust_gradients(model, config(), idx, tgt, base_seed=5, step=1)
        for name in first:
            self.assertTrue(torch.equal(first[name], again[name]), name)
        self.assertFalse(torch.equal(first["tok_emb.weight"], other["tok_emb.weight"]))

    def test_no_autograd_graph_is_built(self):
        model = build()
        idx, tgt = batch()
        grads, _ = dust_gradients(model, config(), idx, tgt, base_seed=0, step=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(not g.requires_grad for g in grads.values()))

    def test_untied_head_gradient_is_exactly_backprop(self):
        # With separate matrices the head only receives the analytic term, so it
        # must match the true gradient rather than merely correlate with it.
        model = build(dict(TINY, tie_embeddings=False))
        idx, tgt = batch()
        report = probe_cosine(model, config(population=4, draw_chunk=4), idx, tgt,
                              base_seed=0, step=0)
        self.assertGreater(report["perLayer"]["head.weight"], 0.999)

    def test_estimate_aligns_with_backprop(self):
        model = build()
        idx, tgt = batch()
        report = probe_cosine(model, config(population=32, draw_chunk=8), idx, tgt,
                              base_seed=0, step=0)
        self.assertIsNotNone(report["overall"])
        self.assertGreater(report["overall"], 0.25)
        self.assertGreater(report["byType"]["attn"], 0.15)
        self.assertGreater(report["byType"]["mlp"], 0.15)

    def test_cosine_probe_buckets_the_tied_embedding_separately(self):
        # The head shares ``tok_emb.weight``, so a name == "head.weight" test
        # silently folded the embedding into the mlp bucket.
        model = build()
        gen = torch.Generator().manual_seed(5)
        idx, tgt = induction_batch(2, 8, model.tok_emb.num_embeddings, torch.device("cpu"), gen)
        report = probe_cosine(model, config(population=4, draw_chunk=4), idx, tgt,
                              base_seed=3, step=0)
        self.assertIn("embedding", report["byType"])
        self.assertNotIn("head", report["byType"])
        self.assertIn("tok_emb.weight", report["perLayer"])

    def test_isolating_layers_beats_perturbing_them_together(self):
        # The paper's interference control: jittering every layer at once sums
        # their effects into one reward, so each layer's own signal is diluted.
        model = build()
        idx, tgt = batch()
        together = probe_cosine(model, config(population=32, draw_chunk=8, mode="all"),
                                idx, tgt, base_seed=0, step=0)
        isolated = probe_cosine(model, config(population=32, draw_chunk=8, mode="layer"),
                                idx, tgt, base_seed=0, step=0)
        self.assertGreater(isolated["overall"], together["overall"] * 3)

    def test_one_step_matches_backprop_descent(self):
        model = build(TOY, seed=1)
        idx, tgt = batch(seq_len=8, n=8, vocab=TOY["vocab_size"])
        before = loss_on(model, idx, tgt)

        reference = build(TOY, seed=1)
        grads = backprop_gradients(reference, idx, tgt)
        lr = 1e-3
        with torch.no_grad():
            for name, param in reference.named_parameters():
                if name in grads:
                    param -= lr * grads[name]
        backprop_after = loss_on(reference, idx, tgt)

        estimate = build(TOY, seed=1)
        dust, _stats = dust_gradients(estimate, config(population=64, draw_chunk=16),
                                      idx, tgt, base_seed=0, step=0)
        with torch.no_grad():
            for name, param in estimate.named_parameters():
                if name in dust:
                    param -= lr * dust[name]
        dust_after = loss_on(estimate, idx, tgt)

        self.assertLess(backprop_after, before)
        self.assertLess(dust_after, before)
        # Within a few percent of the true step, not merely the same direction.
        self.assertAlmostEqual(dust_after, backprop_after, delta=0.02 * abs(before))

    def test_reverse_discounted_cumsum_matches_the_definition(self):
        losses = torch.randn(3, 5)
        got = reverse_discounted_cumsum(losses, 0.9)
        expected = torch.zeros_like(losses)
        for t in range(5):
            for s in range(t, 5):
                expected[:, t] += 0.9 ** (s - t) * losses[:, s]
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_gamma_zero_uses_only_the_current_token(self):
        losses = torch.randn(2, 4)
        self.assertTrue(torch.equal(reverse_discounted_cumsum(losses, 0.0), losses))

    def test_stats_describe_the_work_done(self):
        model = build()
        idx, tgt = batch()
        cfg = config(population=8, draw_chunk=4)
        _grads, stats = dust_gradients(model, cfg, idx, tgt, base_seed=0, step=0)
        passes = len(model.noise_shapes())
        self.assertEqual(stats["tokens"], 8 * idx.shape[0] * idx.shape[1])
        self.assertEqual(stats["forwards"], 2 * (1 + 2 * passes))
        self.assertGreater(stats["gradNorm"], 0.0)

    def test_config_rejects_nonsense(self):
        for kwargs in ({"population": 0}, {"sigma": 0.0}, {"gamma": 1.5},
                       {"draw_chunk": 0}, {"mode": "everything"}):
            with self.assertRaises(ValueError):
                DustConfig(**kwargs)


class TrainingTests(unittest.TestCase):
    def test_optimizer_moves_parameters_and_reports_state(self):
        model = build()
        idx, tgt = batch()
        before = model.blocks[0].mlp.up.weight.detach().clone()
        opt = DustOptimizer(model, lr=0.1)
        stats = dust_step(model, opt, config(), idx, tgt, base_seed=0, step=0)
        after = model.blocks[0].mlp.up.weight.detach()
        self.assertFalse(torch.equal(before, after))
        self.assertEqual(stats["applied"], stats["parameters"])
        self.assertFalse(torch.isnan(after).any())

    def test_momentum_keeps_one_buffer_per_parameter(self):
        model = build()
        idx, tgt = batch()
        opt = DustOptimizer(model, lr=0.01, momentum=0.9)
        dust_step(model, opt, config(), idx, tgt, base_seed=0, step=0)
        self.assertEqual(len(opt.buffers), len(opt.params))
        self.assertTrue(all(torch.isfinite(b).all() for b in opt.buffers.values()))

    def test_gradient_clipping_bounds_the_step(self):
        model = build()
        idx, tgt = batch()
        plain = DustOptimizer(build(), lr=1.0)
        clipped = DustOptimizer(build(seed=0), lr=1.0, grad_clip=0.01)
        grads, _ = dust_gradients(model, config(), idx, tgt, base_seed=0, step=0)
        clipped.step(grads)
        self.assertLessEqual(max(float(b.abs().max()) for b in clipped.buffers.values())
                             if clipped.buffers else 0.0, 0.01 + 1e-9)
        self.assertEqual(plain.step(grads)["applied"], len(plain.params))

    def test_learning_rate_warmup(self):
        opt = DustOptimizer(build(), lr=1.0)
        self.assertAlmostEqual(opt.lr_at(0, 10, 1.0), 0.1)
        self.assertAlmostEqual(opt.lr_at(4, 10, 1.0), 0.5)
        self.assertAlmostEqual(opt.lr_at(50, 10, 1.0), 1.0)
        self.assertAlmostEqual(opt.lr_at(0, 0, 0.7), 0.7)

    def test_identity_task_descends(self):
        # The reference scripts default to random targets, where the loss cannot
        # fall. This is the guard that our default regime actually trains.
        model = build(dict(TOY, vocab_size=32), seed=1)
        cfg = config(population=32, draw_chunk=16, lr=1e-2)
        opt = DustOptimizer(model, lr=cfg.lr)
        gen = torch.Generator().manual_seed(0)
        eval_idx, eval_tgt = identity_batch(8, 8, 32, torch.device("cpu"),
                                            torch.Generator().manual_seed(999))
        first = loss_on(model, eval_idx, eval_tgt)
        for step in range(40):
            idx, tgt = identity_batch(8, 8, 32, torch.device("cpu"), gen)
            dust_step(model, opt, cfg, idx, tgt, base_seed=0, step=step)
        final = loss_on(model, eval_idx, eval_tgt)
        self.assertLess(final, first * 0.95, (first, final))


class TaskTests(unittest.TestCase):
    def test_induction_batch_repeats_with_the_requested_period(self):
        for seq_len in (15, 16):
            idx, tgt = induction_batch(2, seq_len, 32, torch.device("cpu"),
                                       torch.Generator().manual_seed(0), period=2)
            self.assertEqual(idx.shape, (2, seq_len))
            self.assertEqual(tgt.shape, (2, seq_len))
            row = idx[0]
            for t in range(seq_len - 2):
                self.assertEqual(int(row[t]), int(row[t + 2]))
            self.assertTrue(torch.equal(tgt[:, :-1], idx[:, 1:]))

    def test_induction_rejects_a_period_longer_than_the_window(self):
        with self.assertRaises(ValueError):
            induction_batch(1, 4, 8, torch.device("cpu"), torch.Generator(), period=9)

    def test_identity_targets_are_the_inputs(self):
        idx, tgt = identity_batch(2, 6, 16, torch.device("cpu"), torch.Generator())
        self.assertTrue(torch.equal(idx, tgt))

    def test_token_file_roundtrip_uses_the_requested_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokens.bin")
            values = np.arange(64, dtype=np.uint32) + 5
            values.tofile(path)
            loaded = load_tokens(path, "uint32")
            self.assertEqual(loaded.tolist(), values.astype(np.int64).tolist())
            # A uint16 read of a uint32 file must not look correct.
            self.assertNotEqual(load_tokens(path, "uint16").tolist()[:8],
                                loaded.tolist()[:8])

    def test_unknown_dtype_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokens.bin")
            np.arange(8, dtype=np.uint32).tofile(path)
            with self.assertRaises(ValueError):
                load_tokens(path, "uint64")

    def test_token_batch_wraps_and_returns_targets(self):
        tokens = torch.arange(10)
        idx, tgt, cursor = token_batch(tokens, 2, 4, 0, torch.device("cpu"))
        self.assertEqual(idx[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(tgt[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(cursor, 5)
        idx, _tgt, cursor = token_batch(tokens, 2, 4, 7, torch.device("cpu"))
        self.assertEqual(idx[0].tolist(), [0, 1, 2, 3])  # 7 + 5 > 10, so it wraps
        self.assertEqual(cursor, 5)

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(ValueError):
            make_source("nope", batch=1, seq_len=4, vocab=8,
                        device=torch.device("cpu"), generator=torch.Generator())

    def test_copy_task_needs_a_token_file(self):
        with self.assertRaises(ValueError):
            make_source("copy", batch=1, seq_len=4, vocab=8,
                        device=torch.device("cpu"), generator=torch.Generator())


if __name__ == "__main__":
    unittest.main()
