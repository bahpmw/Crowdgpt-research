# soma — Sharded Optimization Mixture of Assemblies (2609.37899)

File: `methods/soma.py`.

## READ `extra.ensembleLoss`, NOT `evalLoss`
For n>1 the harness `model` is never touched, so its `evalLoss` stays at the
untrained 8.366. That is expected, not a bug. The ensemble is evaluated inside
`step()` on 256 fixed val windows (seeded 4321):
- Each window is routed by its first 32 tokens: nearest k-means centroid of
  the tf-idf bag-of-tokens, top `route_k` experts (default 1), with their
  probabilities averaged.
- Scoring covers positions 33–128 only.

For the `n=1` control, the single expert *is* the harness `model`, so both
numbers are real. Compare arms on `ensembleLoss` only (same windows, same
tail). Also logged:
- `oracleLoss`: the best single expert per window.
- `exp{e}Own`: expert e on the windows routed to it.
- `trainLoss`, `simTime`.

The ensemble eval runs every `ens_secs`=5 s, and on every step within two
steps of `seconds`=120, so the final row is fresh. Pass `seconds=` equal to
the harness `--seconds` budget.

## Faithful vs simplified
- Faithful:
  - N independent experts (default 4), each trained only on its own cluster
    with its own loss and its own Adam. Experts exchange no communication.
  - SPSA with n_pert (8) 50%-sparse Rademacher probes: 0 w.p. ½, ±1 w.p. ¼,
    no rescaling. RMSNorm gains use dense ±1.
  - All probes and both signs share one batch.
  - Adam (0.9, 0.999, eps 1e-8), coupled weight decay 1e-4, lr = ε = 1e-3.
  - The tied embedding/head is never perturbed.
- Simplified:
  - The head gets the exact decoder-path gradient every step, as repo DUST
    does. It uses Adam with `head_lr` 3e-3. `head_freeze=K` stops head
    training after K steps.
  - There is no shared seed run. Experts start from different random inits
    and each owns its embedding.
  - Clustering: tf-idf-lite on 4096-id bags (log1p counts × idf,
    L2-normalized). Spherical k-means (10 iterations, unbalanced) is fit on
    20k windows of 128, then a 60k-window pool is assigned. Smoke sizes were
    [14600, 20146, 17390, 7864].
  - No plateau decay.
- Sizes: experts have 6 layers, with width chosen so N experts ≈ 10m params.
  n=4 gives d=128: 2.10M each, 8.39M total (1.14× the 10m model). `width=256`
  gives same-size experts.
- Accounting:
  - tokens = N·8·128 per step.
  - fwd is in full-model-equivalent token passes: raw passes × expert/10m
    params. Compute is therefore comparable across n.
  - simTime/step = (2·n_pert+1)·expert/full for n>1 (experts run in
    parallel), and (2·n_pert/ddpp + 1) for the n=1 control run as DDPP over
    `ddpp`=4 workers. These come out nearly equal (4.86 vs 5.0).
  - up/down: 0 for n>1. For n=1, each worker exchanges the fp32 head gradient
    plus 12 B per probe.
- `estimate()` = the same SPSA plus head gradient on the harness model (the
  full-size monolith).

## Smoke results (GPU shared with other runs)
- `soma seconds=15`, n=4, 15 s, 119 steps: ensembleLoss 8.304 → 7.003,
  oracleLoss 6.900, exp*Own 6.99–7.06. evalLoss stays at 8.366 (untouched
  model).
- `soma n=1 seconds=15`, 15 s, 167 steps: evalLoss 8.366 → 7.110,
  ensembleLoss 7.099. At an equal simTime of about 400, n=4 had 7.08 and n=1
  had about 7.25, but that comparison is noisy.
- `probe.py "soma n=1" --repeats 1 --device cpu`: block cosine +0.0009 at 17
  forwards and +0.0029 at n_pert=64 (129 forwards). This matches the SPSA
  expectation of about √(n/d).
- **probe.py fails on MPS for every method.** It calls `.double()` before
  `.cpu()` on the backprop reference. Run it with `--device cpu`.
  `estimate()` returns CPU tensors.

## Caveat for interpreting
Most of the loss drop in 120 s comes from the exact head gradient (unigram and
bigram statistics through the tied embedding). SPSA body learning at a cosine
of about 1e-3 is slow. The n=4 vs n=1 gap therefore mixes the paper's
variance effect with head capacity (4 heads of width 128 vs 1 of width 256).
The `head_freeze` arms isolate the body.

## Round-1 arms (120 s each)
```
P=../../.venv/bin/python
$P harness.py soma n=1 seconds=120 --label r1-soma-n1            # control: monolith, same SPSA, same param-forwards
$P harness.py soma n=4 seconds=120 --label r1-soma-n4
$P harness.py soma n=4 route_k=2 seconds=120 --label r1-soma-n4-top2
$P harness.py soma n=2 seconds=120 --label r1-soma-n2
$P harness.py soma n=1 head_freeze=200 seconds=120 --label r1-soma-n1-hf200   # body-only after 200 head steps
$P harness.py soma n=4 head_freeze=200 seconds=120 --label r1-soma-n4-hf200
```
References: `base-mezo` (monolithic SPSA, k=1) and `base-adam`. Those use the
harness evalLoss over all 128 positions, so compare them to `r1-soma-n1`'s
evalLoss, not to `ensembleLoss`.

## Proves out if
1. `r1-soma-n4` ensembleLoss < `r1-soma-n1` at equal simTime and equal compute
   (the fwd column is already full-model-equivalent).
2. The gap persists in the `head_freeze` pair, which shows the body-variance
   effect rather than head capacity.
3. `oracleLoss` − `ensembleLoss` is small, which shows the prefix router works.
   Expect backprop (`base-adam`) to dominate everything, as in the paper.
