# loca — LoCA calibrated local credit (2608.03020)

Files: `methods/loca.py` (pretraining adaptation, main), `methods/locaft.py`
(faithful frozen-backbone adapter variant + LoRA-style control).

## Faithful vs simplified
- Faithful: top error e = W_u^T(softmax − onehot) at the norm_f output
  (closed form); one backprop probe records the exact dL/dx_l at each block
  output; ridge fit F = G Eᵀ(EEᵀ + βI)⁻¹ (β = 1e-3·tr(EEᵀ)/d), SVD truncated to
  rank 8, applied as U(Vᵀe). locaft: parallel adapters h_l = h̄_l + B_l A_l h_{l-1}
  (r=32, A ~ N(0,1/d)), Jacobi outer iteration with relative step
  η̄ = η·RMS(H)/RMS(Q), streaming G/C statistics, closed-form ridge solve,
  held-out selection.
- Adapted (not in the paper): `loca` trains the full model from scratch. Every
  `calib` steps it refits F. Each block does a local backward from the
  synthetic gradient F_l e/N on a detached input. The head and norm_f get exact
  gradients, and the embedding gets the head gradient plus block 0's local
  input gradient. AdamW lr 3e-3, clip 1. The calibration batch is that step's
  training batch. `exact_last=true` gives the last block the exact gradient
  through norm_f, since it can live on the head's peer.
- locaft simplifications: greedy accept/revert on a fixed held-out training
  batch instead of best-snapshot selection; no 40-iteration cap; η, λ fixed.
- Logged at each calibration (persisted in `extra`): `cosPre`/`cosPre{l}` is the
  cosine of the *old* F_l e to the true dL/dx_l before refitting (alignment
  after `calib` steps of drift). `cosFit` is the in-probe cosine of the new fit.

## Smoke results (GPU shared, so step counts are noisy)
- `loca` (defaults, calib=50), 15 s: 8.366 → 6.983 (best 6.975), 1093 steps.
- `loca calib=10`, 15 s: 8.366 → 6.816 (best 6.777), 1027 steps. cosFit
  0.43–0.57 (the paper reports 0.3–0.5). After 10 steps, cosPre had decayed to
  0.15–0.35, with early blocks lowest (block 0 0.01–0.2, block 5 0.27–0.59).
  Loss plateaus near 6.8. Adam was at 5.86 by 10 s.
- `loca calib=1`, 12 s: 7.00 (204 steps; the per-step CPU solve makes it
  slow). cosPre at age 1 was 0.43 mean.
- `loca calib=10 exact_last=true`, 12 s: 6.47. cosPre held at 0.44–0.58 in
  every block, so exact credit at the top keeps the feedback map aligned.
- `loca calib=10 probe_batches=4`, 12 s: 6.91. A bigger probe set did not help.
- `locaft --init ckpt/adam30.pt`, 15 s: 5.598 → 5.590 (195 outer iterations).
  The Adam-trained adapter (`solver=adam`) reached 5.547 in 10 s, so ridge
  LoCA recovers about 15% of the LoRA-style gain. eta=0.03 and eta=0.1/lam=0.1
  gave no improvement.

## Round-1 arms (120 s each)
```
../../.venv/bin/python harness.py adam lr=0.003 --label r1-loca-adam            # = base-adam
../../.venv/bin/python harness.py loca calib=1   --label r1-loca-c1             # refit every step (upper bound)
../../.venv/bin/python harness.py loca calib=10  --label r1-loca-c10
../../.venv/bin/python harness.py loca calib=50  --label r1-loca-c50
../../.venv/bin/python harness.py loca calib=200 --label r1-loca-c200
../../.venv/bin/python harness.py loca calib=50 exact_last=true --label r1-loca-c50-xl
# faithful small-shift fine-tuning from the 30 s Adam checkpoint
../../.venv/bin/python harness.py locaft --init ckpt/adam30.pt --label r1-locaft
../../.venv/bin/python harness.py locaft solver=adam --init ckpt/adam30.pt --label r1-locaft-lora
../../.venv/bin/python harness.py adam --init ckpt/adam30.pt --label r1-locaft-fullft
```

## Proves out if
- The pretraining variant at calib=50 keeps cosPre ≥ 0.2 in most blocks, and its
  eval loss at matched tokens is within ~0.3 nats of Adam while beating DUST by
  a wide margin. The crowd value: no backward chain and r=8 feedback codes per
  token instead of d=256 gradients.
- calib=1 vs calib=50 shows how much of any gap comes from drift and how much
  from the rank-8 tokenwise linear map itself.
- locaft recovers ≥0.8 of the LoRA-style (`solver=adam`) CE gain (the paper's
  criterion). The smoke run is far below that (~0.15).
