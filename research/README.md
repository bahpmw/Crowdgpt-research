# Research sweep — what else composes with DUST in v1.5

Three parallel literature/ecosystem sweeps, each in its own file, plus what we have
already measured. The question they answer is not "what is interesting" but "what can
we build and *judge* this week, at 1B, on hardware volunteers actually have".

| file | scope |
|---|---|
| [`zo-es-lineage.md`](zo-es-lineage.md) | zeroth-order / evolution-strategy lineage: variance reduction, forward-only training, MeZO/SPSA/DeepZero, DUST follow-ups, activation- vs weight-space theory |
| [`communication-efficiency.md`](communication-efficiency.md) | error feedback (EF21, signSGD, 1-bit Adam), low-rank (PowerSGD), local-SGD (DiLoCo family), async buffers (FedBuff), aggregation defences (Krum/FedNova), server-side amortisation (FedOpt) |
| [`open-source-ecosystem.md`](open-source-ecosystem.md) | who else is doing this: `qlabs-eng/dust`, `huskydoge/dust`, `qlabs-eng/noise`, OpenDiLoCo, hivemind, petals, DisTrO, licences |
| [`training-data/`](training-data/README.md) | DRAFT: twelve candidate pretraining datasets, fixes to the current mix, and a phased training order for 1B → 27B MoE |

Each file flags which of its sources assume **unbiased** gradients. That matters here:
a DUST estimate is unbiased *in expectation* but very high variance, and the upload is
a **biased** 1-bit quantisation of it, and the coordinator turns submissions into a
unit direction with a declared magnitude. Several classical results do not transfer.

## What is already built and measured

| idea | status | measured |
|---|---|---|
| Biased 1-bit upload (`ngd1`, `ngd1r`) | shipped | 16.4× / 8.1× smaller than dense bf16 |
| Error feedback (Seide 2014; Karimireddy 2019) | shipped, `2bf5142` | recovers **~71 %** of the accuracy the 1-bit codec gives up: final loss 3.7947 dense / 3.8863 `ngd1+ef` / 4.1121 plain `ngd1` at the same payload |
| Structural (low-rank) upload | measured, `8cbe1af`, not on the wire | `r=8` truncated ≈ **12×** smaller and *beats* `ngd1+ef` (3.8681 vs 3.8863); `r = T` is **exact** (3.7963 vs 3.7947 dense) at 6× |
| DiLoCo-style outer optimizer | shipped this round | `outer_lr=0.7`, Nesterov 0.9: 4.1121 → **0.5031** (−3.61 nats) on identity, tiny preset |

The low-rank result needs one clarification because the comms sweep ranks PowerSGD
"low-to-negative" for us: PowerSGD approximates a *gradient* by power iteration over
random bases. Our case is different and stronger — a DUST update is
`G_W = Σ_t (r_t a_t) x_tᵀ`, so it is **rank ≤ T by construction**, exactly, with no
power iteration and no approximation at `r = T`. PowerSGD's objection (rank truncation
keeps the noisiest coordinates) still applies once `r < T`, which is why the practical
route is to choose tokens rather than SVD directions.

## Ranked: what to build next

Ordered by (payoff at 1B) ÷ (implementation cost), and by whether it can be judged on a
16 GB card this week.

| # | idea | why it composes | cost | how we would judge it |
|---|---|---|---|---|
| 1 | **Token-selected low-rank upload** (send `r` token rows of `a_t`, `x_t` instead of the dense matrix) | the update is rank ≤ T by construction; `r=8` already beats 1-bit dense at 12× on the tiny preset | 2–3 days: new codec name on the wire + coordinator reassembly | same bench, sweep `r` at the 10m and 100m presets; at 1B it is arithmetic (≈10 MB/round vs 127 MB for `ngd1`) |
| 2 | **Per-layer `sigma_l = rel · rms(y_l)`** (G25) | the alignment collapse we measured is sigma ≪ activation scale, not population | 1 day: measure `rms(y_l)` once, scale each layer's noise | the same cosine probe at 10m/100m: it must clear the knee, then the 1B numbers become estimator quality instead of mechanics |
| 3 | **Split-half SNR tuning** (from the ZO sweep) | DUST's own tuning recipe needs a backprop gradient; correlating two halves of draws needs none | 1 day, client-side only | per-layer SNR printed per round; compare cosine with the backprop probe on the small presets |
| 4 | **Head jitter on cached logits over a vocabulary slab** (`qlabs-eng/dust`, MIT) | removes the exact-softmax shortcut so the head is zeroth-order too, and slabbing is what makes 1B fit 16 GB | 2 days | short 10m run: does loss still fall, how many bytes/round now |
| 5 | **FedBuff-style buffer** (K≈8–16, over-issue 2K) | removes the per-round latency floor; the paper measures 3.3–3.8× fewer client-trips | 2–3 days, coordinator only | simulated clients with staggered arrival; rounds-to-target and wall clock, not loss |
| 6 | **Rank-normalised reward** (Salimans 2017; standard in EGGROLL, absent from `dust.py`) | free variance reduction on heavy-tailed per-token deltas | half a day | cosine probe + short identity run, A/B |
| 7 | **Warm-up rounds uncompressed** (1-bit Adam's own precondition) | our variance is highest early; the first rounds are exactly where 1 bit hurts | half a day, one flag | the existing codec bench with an unquantised first 10 % |
| 8 | **Cross-round control variate** (MeZO-SVRG) | up to 2× fewer steps in the paper | 2–3 days, and it breaks the no-state promise unless the snapshot is compressed | identity/induction A/B at equal upload bytes, not equal rounds |
| 9 | **Median norm cap instead of mean** | our `declaredNorm` is a client claim; the mean is the wrong statistic for outliers | hours | adversarial fixture: one client claiming 100× the norm |
| 10 | **Token-count weighting** | cheap and correct *if* estimates were independent and unbiased — ours share data, so it only fixes evidence accounting, not bias | hours | audit-only; no expected loss win |

Explicitly **not** recommended: PowerSGD as published (wrong tool, see above);
Krum/trimmed-mean upgrades (the `σ ≪ ‖g‖` precondition is violated — our unit vectors are
nearly orthogonal, so distances concentrate); gossip/consensus SGD (needs a topology we do
not have, and its `1/(1−β)` penalty is worst under churn); DisTrO code (no licence — read
the paper, copy nothing).

## Scale arithmetic at 1B (1,015,085,056 params, plain transformer)

| payload | bytes/round | note |
|---|---|---|
| dense bf16 | ≈ 2.03 GB | what a backprop client ships today |
| `ngd1` (1 bit + fp16 scales) | ≈ 127 MB | measured 16.4× on the tiny preset |
| `ngd1r` (2 bits) | ≈ 254 MB | 8.1× |
| low-rank `r=8`, fp16 factors | ≈ 10 MB | arithmetic over 15 blocks; ≈128× per 2048² matrix, ≈205× per 2048×8192 MLP matrix |

So the ordering of the *upload* problem is: low-rank ≫ 1-bit + error feedback > 1-bit.
The open question is not bytes but whether `r = 8` carries enough of a 1B estimate — a
question one 10m/100m run can answer, which is why it ranks first.

## The caveat that governs all of this

No zeroth-order method has been trained from scratch past **243M parameters / 20M
tokens** (DUST's own 1B "result" is a gradient-alignment measurement on *backprop*
checkpoints). At 1B we are ~4× beyond the largest model and far beyond the token budget
any of these papers ran. Everything above is a mechanism check at our scale, and the
honest deliverable is a curve with its error bars, not a claim that DUST scales.
