# loo: LOO-ROLL vs EGGROLL antithetic (2609.10980)

## Faithful
- Rank-1 Gaussian perturbation E_s = a_s b_s^T on every 2-D matrix, including tok_emb.
  1-D params (RMSNorm gains) get dense Gaussian noise, as EGGROLL does for vectors.
- `baseline=loo`: N members, one forward each, g = 1/(N sigma) sum_s E_s (F_s - mean_{j!=s} F_j).
- `baseline=mirror`: EGGROLL antithetic, N/2 (+-) pairs at the same N forwards.
- `zscore=true`: EGGROLL fitness standardisation (rescale only). All members see one shared batch.
- Seeds per (step, member). The update is A diag(w) B^T, replayable from seed + N fitness scalars.
  Upload = 4N+8 bytes.

## Simplified / chosen
- Optimizer: SGD momentum 0.9 with global-norm clip 1.0 (the paper leaves it unspecified).
  Because of the clip, `lr` is the step length.
- Defaults are sigma=1e-3 and lr=0.5. At 256-wide matrices, sigma=1e-2 rank-1 has spectral norm
  about 4x the weights.
- Members run sequentially with an in-place rank-1 add/subtract (not the paper's batched
  population forward). This costs about 2-4x MeZO wall time per forward, so compare on
  forwards/tokens as well as wall time.

## Measured in smoke (10 s each, shared GPU)
| arm | result |
|---|---|
| sigma=1e-2, lr=0.05 | 8.366 -> 8.365 (gradNorm about 5e4, so the clip makes steps tiny) |
| sigma=1e-3, lr=0.5 | -> 8.321 (35 steps) |
| sigma=1e-3, lr=0.2 | -> 8.323 (61 steps) |
| sigma=1e-3, lr=2 | overshoots (8.43) |
| mirror, sigma=1e-3, lr=0.5 | -> 8.337 (49 steps) |

probe on ckpt (CPU, N=32, 3 repeats): loo 3.5e-4, mirror 3.4e-4, mezo k=16 3.5e-4. All arms are
at the noise floor. On the checkpoint the embedding/head gradient is 3.4x the block gradient,
which dilutes every weight-space ZO block cosine.

## Round-1 arms (120 s each)
```
P=../../.venv/bin/python
$P harness.py loo n=32 sigma=0.001 lr=0.5                    --label r1-loo-n32
$P harness.py loo n=32 sigma=0.001 lr=0.5 baseline=mirror    --label r1-mirror-n32
$P harness.py loo n=32 sigma=0.001 lr=0.25                   --label r1-loo-n32-lr025
$P harness.py loo n=32 sigma=0.001 lr=0.25 baseline=mirror   --label r1-mirror-n32-lr025
$P harness.py mezo k=16 eps=0.001 lr=0.0001                  --label r1-mezo-k16
```
`r1-mezo-k16` is a reference at the same 32 forwards per step.

## Proves out if
- At matched lr, LOO reaches lower loss than mirror at equal forwards/tokens. The paper predicts
  about half the estimator MSE. The bar is LOO's final loss being at or below mirror's at both
  lrs, with a clear margin (>0.01) at one or more.
- For the crowd: LOO beats `mezo k=16` per token at 136 B/step upload. The notes expect ES from
  scratch to lose to DUST overall; that is not the question here.
