DRAFT — requires human review

# v2 test day (2026-11-18): what the roadmap ideas did when we actually ran them

Status: in progress. Sections marked *pending* are still running on this machine.
All numbers are from `research/lab/results/*.json` and `v2-mvp/runs/*/report.json`.
One Apple laptop GPU (MPS), small models (about 7M parameters unless stated), usually one
seed, short runs (4M–10M tokens). Treat differences under about 0.03 in loss as noise.

## 1. Short answer

| Question | Answer from today's runs |
|---|---|
| Does a v2 network (several orchestrators, signed record, checks) actually work end to end? | Yes, at toy size. 3 orchestrators, 8 volunteers: loss fell every round, every round was co-signed by all 3, 0 honest volunteers wrongly rejected. See §2. |
| Do the checks catch cheaters? | All 7 cheating styles we built (flip, scale, noise, zero, copy, wrong data, stale) were caught and banned within 3–4 rounds, with 0 false alarms. See §2. |
| Does it survive orchestrators going down? | Yes, after one fix. Two of three orchestrators were killed and restarted; training kept going on the survivor, the others caught up and co-signed, all 10 rounds ended final. See §2. |
| Author's worry: "SOLO needs more downloads (activations)" | Correct for SOLO crews passing activations: about as many bytes per round as a whole dense model. MOSAIC (each volunteer recomputes the earlier slices itself) cuts download about 20x. See §3. |
| Author's worry: "does it work with ReLoRA?" | Yes, inside MOSAIC, if volunteers send the low-rank factors and the merge averages the full products. It beat dense full-update syncing at about 1/17 of the upload. See §4. |
| Does pipeline delay (staleness) break split models? | No, not up to 8 steps of delay at 10M tokens (cost about 0.01–0.02 loss, inside noise). The earlier bad result was a too-short run. See §5. |
| How should volunteer groups sync? | Groups of 16 that fully average, plus one partner between groups, matched flat full averaging. Pure 2-partner gossip was much worse. See §6. |
| Is "median length × mean direction" merging worth it? | Yes: slightly better with no attack, and it shrugged off a 2-of-16 sign-flip attack that hurt plain averaging badly. See §6. |
| Gated DeltaNet hybrid layers? | Huge gain (about 0.8 loss) but mostly because our v1.5 model has no position information at all. Fair re-test with position encoding (RoPE) is running. See §7. |

## 2. MVP network (`v2-mvp/`)

Built from `v2-mvp/SPEC.md`: Ed25519-signed identities and record entries, a hash-chained
record co-signed by orchestrators (final at 2 of 3), beacon-assigned work and data, MOSAIC
cells (each volunteer trains one of 2 slices of a 10M model for 30 steps), commit-then-upload,
sampled replay audits by a different orchestrator, SparseLoCo-compressed uploads, integer
fixed-point merging, a held-out loss gate with rollback, and a static dashboard.
Tests: 39 passed.

**Baseline** (3 orchestrators, 8 honest volunteers, 7 on CPU and 1 on MPS, 10 rounds):

- Held-out loss 8.37 → 7.93 (round 1) → 7.31 (round 6) → 7.01 (round 10); no rollbacks.
- 10 of 10 rounds final with 3 of 3 signatures; chain verified on every orchestrator.
- 80 honest cells, 80 accepted.
- Replay checks: CPU volunteer vs CPU checker matched exactly (cosine 1.0). MPS volunteer vs
  CPU checker: cosine 0.985–1.0 (mean 0.996), length ratio within 0.05 %. Threshold set from
  genesis calibration: cosine ≥ 0.98. So cross-device checking worked here, but with only
  about 2 % margin on the worst audit; this needs more data before trusting it.
- Per volunteer per round: about 20 MB up (most of it is the error-feedback buffer, sent only
  when audited; the compressed update itself is 122–153 KB) and about 38 MB down.
- Round time 31–66 s (mean 43 s).

**Attacks** (2 honest, 7 attackers, 8 rounds): every attacker was rejected in every round its
attack was active and banned after 3 strikes (rounds 3–4). The two "misses" were round 1 for
`stale` and `copy`, where there was nothing old to replay or copy, so they trained honestly.
0 honest cells rejected. Round 5 was rolled back by the loss gate even though both cells were
honest: with only 2 honest volunteers there was no averaging, so one noisy cell moved the
model too far. That is a real small-network risk, not a bug.

**Chaos** (orchestrator kills, restarts, volunteer churn; 10 rounds, 310 s total):

- Held-out loss 8.37 → 6.95; no rollbacks.
- 10 of 10 rounds final with 3 of 3 signatures; chain check passed on all 3 orchestrators
  (21 entries, 0 errors).
- Rounds 6–7 ran on one orchestrator alone. After restart, the two others caught up (6 and 5
  entries), re-ran each merge themselves and co-signed.
- 52 of 53 honest cells accepted. One (round 8) was left out as "missing audit", with no strike.
- Round time 16–40 s (mean 27 s).

The first chaos run found a real bug: a restarted orchestrator co-signed the survivor's entries,
then opened the current round with a different beacon, refused forwarded commits, but was still
picked as auditor. Rounds stalled until the audit deadline (320 s and 312 s; 921 s whole run).
Fix: `/health` lists the commits each orchestrator holds, auditors are only picked from peers
holding the commit, and after 3 s with no such peer the receiving orchestrator replays the cell
itself (cell left out, no strike). 4 new tests in `v2-mvp/tests/test_node.py`; 39 tests pass.
Earlier reports kept in `v2-mvp/runs/chaos-before-fix/` and `v2-mvp/runs/chaos-run2/`.

Still open: the different-beacon split itself. The fix stops the stall; the proper fix is for
a restarted orchestrator to join the round its peers already have open.

## 3. Downloads: SOLO crews vs MOSAIC (the author's first worry)

Computed by `research/lab/bandwidth.py` (details in `plans/relora.md`). Per volunteer per
round, 30 local steps, sequence 2048 × batch 8, 8 peers, MB = 10^6 bytes.

| 1B model, 2 slices | Down | Up |
|---|---|---|
| Whole-model DiLoCo, dense bf16 | 2,030 | 2,030 |
| SparseLoCo (whole model) | 231 | 29 |
| SOLO crew, raw activations | 2,083 | 2,031 |
| SOLO crew, activations compressed 64x (ResBM-style) | 101 | 49 |
| MOSAIC, compressed weight changes | 97–124 | 16–17 |
| MOSAIC + ReLoRA rank 32 factors | (same) | 22–25 |

With 4 slices, MOSAIC drops to 47–70 MB down and 9–10 MB up. A volunteer who rejoins after a
long absence still pays one full bf16 download (about 2 GB at 1B).

Why: in a SOLO crew, activations cross every boundary on every step, so 30 steps of
activations cost about as much as the whole model. MOSAIC sends no activations: each volunteer
re-runs the earlier slices forward from weights it already has, and only small compressed
weight changes travel. The price is extra forward compute on the volunteer.

## 4. ReLoRA inside MOSAIC (the author's second worry)

10M model, 10M tokens, MOSAIC with 2 slices and 8 volunteers, 41 rounds of 30 steps.

| Setup | Loss | Upload per round |
|---|---|---|
| Single machine, full AdamW (reference) | 4.93 | — |
| Single machine, ReLoRA rank 32 | 5.11 | — |
| MOSAIC, dense full updates | 5.28 | 151 MB |
| MOSAIC, SparseLoCo compressed | 5.46 | 1.1 MB |
| MOSAIC, ReLoRA r32, send factors, average products | 5.15 | 9.0 MB |
| MOSAIC, ReLoRA r128, send factors, average products | 5.12 | 35 MB |
| MOSAIC, ReLoRA r32, average the two factors separately | 5.23 | 9.0 MB |
| MOSAIC, ReLoRA product then SparseLoCo compressed | 5.61–5.64 | 1.1 MB |

Takeaways: ReLoRA works with MOSAIC. Send the factors, merge by averaging the products
(averaging the factors separately is worse, and checks must look at the product, not the
factors). Squeezing the product through SparseLoCo throws away the benefit.
Still running: rank 128 single machine and the 100M-size pair.

## 5. Pipeline delay (staleness) in SOLO

10M model, 10M tokens.

| Setup | Loss |
|---|---|
| Ordinary backprop (K=1) | 4.965 |
| 2 slices, delay 0 / 1 / 2 / 4 / 8 steps | 4.961 / 4.970 / 4.971 / 4.954 / 4.979 |
| 3 slices, delay 4 | 4.977 |

Second seed at 4M tokens: K=1 5.30 (first seed 5.26), K=2 5.22 (5.21), K=3 5.22 (5.25).
So 2–3 slices match or slightly beat ordinary backprop here, and delays up to 8 steps cost
nothing measurable. The earlier "delay 4 costs 0.26" result came from a 2-minute run.

## 6. Syncing groups of volunteers (X3)

10M model, about 9.8M tokens in total, 30 local AdamW steps per round.

| Layout | Loss |
|---|---|
| 16 replicas, everyone averages | 5.88 |
| 32 replicas, everyone averages | 6.11 |
| 32 replicas, groups of 16 + 1 partner between groups | 6.10 |
| 32 replicas, gossip with 2 random partners | 7.03 |
| 64 replicas, everyone averages / groups of 16 + 1 partner | 6.61 / 6.62 |
| Reference: one machine, 16-way data-parallel every step | 4.70 |

More replicas at a fixed token budget means fewer rounds, so loss rises with replica count;
that part is expected. The useful finding is that groups-plus-one-partner costs nothing over
full averaging, while plain gossip falls far behind.

Merging rule, 16 replicas:

| | Plain mean | Median length × mean direction |
|---|---|---|
| No attack | 5.88 | 5.78 |
| 2 of 16 send a −5x update | 6.36 | 5.84 |
| 4 of 16 send a near-zero update | 5.93 | 5.83 |

*Pending*: equal-token work units vs token weighting, and the Adam beta2 rule.

## 7. Gated DeltaNet hybrid layers (X5)

| Layers (A = attention, G = Gated DeltaNet) | Loss (10M tokens) |
|---|---|
| AAAAAA (our v1.5 model) | 4.67, 4.66 |
| GGAGGA | 3.90–3.92 |
| GGGGGA / GGGGGG | 3.87–3.89 |

A 0.8 gap is far bigger than the paper's about 1.3x token saving, so we checked it.
`leak_check.py` showed every variant is properly causal (no peeking at future tokens). The
real cause: the v1.5 attention has no position information (no RoPE, no learned positions),
and Gated DeltaNet supplies order for free. *Pending*: the same comparison with RoPE attention
(`arms/x5-rope.txt`). Separately, this means v1.5 itself is missing position encoding, which
should be fixed regardless.

## 8. Still running

- MOSAIC crowd sweep (`arms/solo-crowd.txt`: worker counts, slices, dense vs compressed,
  uneven speeds, Byzantine with Gauntlet, 25 % dropout).
- SOLO at 40M tokens and at 300M with 2 and 4 slices.
- Expert growth by copying (X4, 15 runs).
- Remaining X3 and ReLoRA arms; RoPE re-test.
