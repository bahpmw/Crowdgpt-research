# ddiloco — Decoupled DiLoCo (2604.21428)

File: `methods/ddiloco.py` (shared helpers: `crowd.py`).

## Faithful vs simplified
- Faithful:
  - Asynchronous learners with heterogeneous speeds. The default is
    `speeds=[1,1,0.5,0.25]`.
  - Each learner runs H=30 local AdamW steps (lr 3e-3, betas 0.9/0.95, clip
    1), at a sim cost of H/speed.
  - The syncer has a quorum K=1 plus a grace window: after the K-th arrival
    it waits up to `grace` sim units (default 3, about 10% of H) for more.
    It stops early once all learners are in.
  - Weight = c_tokens·(c_tokens/c_steps), as written in the paper.
  - RDA merges the non-embedding parameters: mean norm × normalized mean
    direction. The tied embedding gets a weighted plain average.
  - Outer step is SGD-Nesterov (olr 0.7, mom 0.9; these come from the
    original DiLoCo, because this paper gives none).
  - `quant=4` sends int4 with one absmax scale per 4096 values and no EF.
- Control: `async=0` sets K=W, which is synchronous DiLoCo that waits for the
  slowest learner every round. Same merge and same outer optimizer.
- Simplified:
  - No streaming fragments (P=1): the whole model syncs at once, and RDA
    treats all non-embedding tensors as one fragment.
  - Δ is the learner's own pseudo-gradient (θ_pulled − θ_local). The paper
    compares against Θ^(t−H) because every learner is overwritten by every
    broadcast.
  - Learners block during the grace window instead of continuing to step.
  - Local training runs lazily when the learner's event fires. That is
    equivalent, because a learner receives nothing mid-run.
- Logged: `simTime`, `merged` (learners per merge), `staleness` (outer steps
  applied since the merged learners pulled), `goodput` (1 − idle
  learner-time / (W·simTime)), and `innerLoss`.
- Accounting:
  - One harness step = one merge event.
  - up = learners merged × 2 B/param (bf16 Δ), or ≈0.5 B/param for int4.
  - down = 4 B/param per learner that pulls.

## Smoke results (GPU shared with other runs)
| arm | 15 s result | simTime | tokens |
|---|---|---|---|
| `async=0` (sync) | 6.187 | 1080 | 1.11M; goodput 0.50 |
| default async (Nesterov .7/.9) | oscillates 6.77 ↔ 7.72, ends 7.30 | 369 | 0.95M |
| async `mom=0 olr=1.0` | 6.033 | 615 | 1.57M; goodput 0.95 |
| async `mom=0.5` | 6.030 | 588 | 1.51M |
| async `quant=4 mom=0.5` (8 s) | 6.41 | 309 | — |

**Surprise:** with the paper defaults, the async variant is unstable. Every
arrival triggers an outer Nesterov step on one learner's full-norm Δ (RDA
preserves norm). Outer steps per sim-time are therefore about 4–11× the
synchronous rate, and momentum 0.9 amplifies that. The paper avoids this with
a fixed per-fragment cadence. Damping the outer momentum (0.5 or 0) fixes it,
and then async beats sync on simTime.

## Round-1 arms (120 s each)
```
P=../../.venv/bin/python
$P harness.py ddiloco async=0 --label r1-dd-sync                     # control: sync DiLoCo, waits for slowest
$P harness.py ddiloco async=0 mom=0.5 --label r1-dd-sync-m5          # control matched to the damped async arm
$P harness.py ddiloco --label r1-dd-async                            # paper defaults (K=1, grace 3, Nesterov .7/.9)
$P harness.py ddiloco mom=0.5 --label r1-dd-async-m5                 # damped outer momentum
$P harness.py ddiloco mom=0.5 grace=0 --label r1-dd-async-m5-nograce # Table 5 direction: no grace
$P harness.py ddiloco mom=0.5 merge=avg --label r1-dd-async-m5-avg   # RDA vs plain weighted average
$P harness.py ddiloco mom=0.5 quant=4 --label r1-dd-async-m5-int4    # int4 outer gradients
$P harness.py adam lr=0.003 workers=4 --label r1-dd-adam4            # DP reference (shared with sparseloco)
```
Read evalLoss against **simTime** (crowd wall time) and against tokens. Mac
seconds are about equal tokens for all arms here, because every learner step
runs serially.

## Proves out if
1. An async arm reaches `r1-dd-sync`'s final loss in clearly less simTime
   (goodput is about 0.95 vs 0.50), and is within about 1–2% of it at equal tokens.
2. `nograce` is measurably worse than grace (the paper's Table 5 direction).
3. RDA ≥ avg at W=4 (the paper says the difference is small at M≤8).
4. int4 costs ≤ about 1% loss at 4× fewer bytes than bf16.
