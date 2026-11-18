# sparseloco — SparseLoCo / Covenant-72B (2603.08163)

File: `methods/sparseloco.py` (shared helpers: `crowd.py`).

## Faithful vs simplified
- Faithful: W peers (default 4), each runs H=30 local AdamW steps (lr 3e-3,
  betas 0.9/0.95, clip 1) from the global weights. Δ = θ_global − θ_local.
  Error feedback: acc = βe + Δ, Δ̂ = Q(TopK(acc)), e ← acc − Δ̂, with β=0.95.
  Top-k keeps 64 of every 64×64 block of every 2-D tensor, including the tied
  embedding/head. Outer step θ −= α·mean(Δ̂), α=1, no momentum.
- Assumed: the 2-bit quantizer is not specified in the paper. Ours uses a
  per-chunk scale s = mean|kept|; the code is sign + (|v|>s), and it
  dequantizes to ±0.5s or ±1.5s.
- Simplified:
  - 1-D tensors (RMSNorm gains) are sent dense in bf16 with no EF.
  - Inner AdamW state persists per peer.
  - wd 0 and a constant LR (matches `adam`).
  - Peers are deep-copied GPTs and run sequentially.
- Options:
  - `topk=0`: dense fp32 Δ with no EF or quantization (DiLoCo with outer SGD α).
  - `outer=nesterov olr=0.7 mom=0.9`: DiLoCo outer Nesterov.
  - `noniid=1`: per-peer stream slices.
  - `speeds=[...]`: sets the simTime of synchronous rounds.
- Gauntlet-lite (`gauntlet=1`): LossScore_r = L(θ) − L(θ − gscale·α·Δ̂_r) on one
  fresh batch of `vbatch` (8) training windows. Peers with a score ≤ 0 are
  dropped. There is no OpenSkill history and no assigned-vs-unassigned check.
  `mednorm=1` adds median-norm rescaling of the kept messages.
- `byz=f`: peers 0..f−1 send −c·Δ̂ (`attack=flip`, c=5) or random values on the
  same support with norm c·‖Δ̂‖ (`attack=noise`, c=1).
- Accounting:
  - One harness step = one outer round (W·H inner steps).
  - up = W × (per 4096-chunk: 64·(12+2) bits + a 32-bit scale, plus 1-D dense
    bf16) = 0.234 bits/param, 0.86 MB/round for 4 peers, vs 117 MB dense fp32.
  - down = every peer fetches every message.
  - fwd includes the validator's (W+1) forwards.
  - `simTime` = H/min(speed) per round.

## Smoke results (GPU shared with other runs, so wall times are noisy)
- `sparseloco` (default), 20 s, 11 rounds: 8.366 → 6.195. There is an early
  wobble: round 3 jumped 7.50 → 7.99 before recovering.
- `sparseloco topk=0` (dense DiLoCo α=1), 20 s, 13 rounds: 8.366 → 5.827. At
  equal rounds (11) it reached 5.889 vs 6.195 sparse, with 137× fewer bytes
  for the sparse arm.
- `byz=1 attack=flip`, plain mean: diverging, 9.65 after 6 rounds.
  `byz=1 attack=flip gauntlet=1`: 6.85 after 6 rounds. `byz=1 attack=noise
  gauntlet=1`: 6.88.
- Gauntlet with α=1 also drops some *honest* peers (kept 0–3 of 4 in clean
  runs). A single peer's Δ̂ applied at full α overshoots. `gscale=0.25` (score
  each peer at its 1/W share) kept 4/4 when clean.

## Round-1 arms (120 s each)
```
P=../../.venv/bin/python
$P harness.py adam lr=0.003 workers=4 --label r1-sl-adam4                 # DDP-equivalent backprop
$P harness.py sparseloco topk=0 --label r1-sl-diloco-sgd                  # dense DiLoCo, outer SGD a=1
$P harness.py sparseloco topk=0 outer=nesterov --label r1-sl-diloco-nest  # dense DiLoCo, Nesterov .7/.9
$P harness.py sparseloco --label r1-sl-sparseloco                         # the method
$P harness.py sparseloco beta=0 --label r1-sl-noef                        # EF ablation (beta=0 = no memory)
$P harness.py sparseloco byz=1 attack=flip --label r1-sl-byz-mean
$P harness.py sparseloco byz=1 attack=flip gauntlet=1 gscale=0.25 --label r1-sl-byz-gauntlet
$P harness.py sparseloco byz=1 attack=noise gauntlet=1 gscale=0.25 --label r1-sl-byznoise-gauntlet
$P harness.py sparseloco gauntlet=1 gscale=0.25 --label r1-sl-clean-gauntlet   # cost of the filter when honest
```
`beta=0` keeps the residual for one round only (e ← acc − Δ̂ with no decayed
memory). That is the closest "EF off" in this code. Compare all arms on
evalLoss vs tokens, vs `up`, and vs `simTime`.

## Proves out if
1. `r1-sl-sparseloco` is within about 2–5% of `r1-sl-diloco-sgd` evalLoss at equal
   tokens, with about 137× fewer upload bytes.
2. The EF ablation is clearly worse, which shows that EF is what makes 0.23
   bits/param work.
3. With one flip attacker, the plain mean stalls or diverges. Gauntlet-lite
   stays within about 1–2% of `r1-sl-clean-gauntlet`.
4. DiLoCo-family arms land within reach of `r1-sl-adam4` at equal tokens. They
   cannot match it at equal Mac wall time, because peers run serially.
