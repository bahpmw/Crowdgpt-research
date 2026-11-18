# solo — Shared-Output Local Learning (2609.35440)

File: `methods/solo.py`.

## Faithful vs simplified
- Faithful: K contiguous gradient-isolated modules (`sg(h_{k-1})` inputs);
  untied embedding/head (the head is cloned from tok_emb in `__init__`, so
  eval's `model.forward` uses the untied head); local head φ_k = `depth` ×
  crowd_v15 Block + RMSNorm; logits τ_k·φ_k(h_k)·W̄ᵀ with learned scalar τ
  (init 1) and W̄ a stop-grad copy of the head refreshed every S steps; only
  the last module's true loss updates W; one AdamW per module; PRIV ablation
  (`heads=private`).
- Simplified: 6-layer model, V/d = 16 (paper's V/d is 26–85, so the gain from
  sharing should be smaller); default depth 1 (paper 2) and S=20 (paper 1,
  but S of 10–50 is harmless there); constant LR 3e-3, no warmup or cosine;
  no head bias.
- Crowd simulation: with `stale=s`, each boundary is a FIFO and module k
  trains on activations and targets module k−1 produced s steps earlier.
- Accounting: fwd/bwd include the head FLOP factor. up = (K−1)·B·T·d·2 bytes
  of boundary activations; down = the W̄ broadcast on refresh steps.
- `K=1` is plain untied backprop + AdamW, the fair BP control. The repo
  `adam` is tied.
- `cos_every=N` is a diagnostic not counted in cost. It logs per-module cosine
  of the local block-weight gradient to full backprop on the same batch.

## Smoke results (GPU shared)
- `solo` (K=2, S=20, depth 1), 15 s: 8.366 → 5.993, 829 steps. Module-0
  cosine to BP was 0.94 (0.937 at step 650). Module 1 is exact (1.0).
- `solo K=3`, 12 s: 6.03. cos0 0.91, cos1 0.97.
- `solo heads=private`, 12 s: 5.95, cos0 0.89. Too short to separate it from
  shared.
- `solo stale=4`, 12 s: 5.99.
- `solo K=1`, 6 s: 6.14 (BP control).
- `szo_probe.py "solo"` from the checkpoint (untrained local heads) gave a
  block cosine of 0.09. That number is not meaningful: use the in-training
  `cos*`.

## Round-1 arms (120 s each)
```
../../.venv/bin/python harness.py solo K=1 --label r1-solo-bp                       # untied BP control
../../.venv/bin/python harness.py adam lr=0.003 --label r1-solo-adam                # = base-adam (tied)
../../.venv/bin/python harness.py solo K=2 cos_every=100 --label r1-solo-k2
../../.venv/bin/python harness.py solo K=3 cos_every=100 --label r1-solo-k3
../../.venv/bin/python harness.py solo K=2 heads=private cos_every=100 --label r1-solo-k2-priv
../../.venv/bin/python harness.py solo K=2 stale=4 --label r1-solo-k2-stale4
../../.venv/bin/python harness.py solo K=2 depth=2 cos_every=100 --label r1-solo-k2-d2
```
The DUST comparison is base-dust8 / base-dust32.

## Proves out if
1. SOLO K=2 is within ~5% of the K=1 BP eval loss at matched tokens, and
   beats PRIV.
2. The per-module cosine (cos0) stays ≥ 0.5.
3. SOLO is ≥10× more token-efficient than DUST at reaching DUST's final loss.
4. stale=4 costs < 2% loss vs stale=0.

Points 3 and 4 decide crowd viability. Upload per step is 0.5 MB per boundary
at batch 8×128, versus 14.7 MB for a dense bf16 DUST estimate.
