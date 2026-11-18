DRAFT — requires human review

# Synthesis: training CrowdGPT on millions of consumer GPUs

Date: 2026-11-18. Scope: all 36 papers in `research/papers/` (12 from round 1, 24 from round 2; see `INDEX.md`).
Each paper has its full text (`<id>.txt`) and notes (`notes/<id>.md`) here. This document reads them together
and asks one question: **what would let CrowdGPT v2 train real models across up to a million volunteer GPUs
(graphics cards), most of them 16 GB consumer cards on home internet?**

How to read the numbers:
- **Reported** numbers come from the papers. Each paper's caveats are in its notes file.
- **Our arithmetic** marks numbers we worked out from those reports.
- **Guess** marks assumptions we have not measured, such as 6 TFLOPS (trillion operations per second) of
  useful work per consumer card.
- Nobody has run anything close to a million GPUs over the internet. The largest internet runs in this set are
  Agora (8.6B parameters, at most 60 machines online at once) and Covenant-72B (72B parameters, about 25 active
  peers, each with 8 datacenter GPUs). Everything beyond that is extrapolation.

---

## 1. Bottom line

1. **Stop trying to make one copy of the model bigger, and make many small, verifiable pieces instead.** At a
   million GPUs, plain "everyone trains a full copy and we average" breaks on every count: card memory, upload
   size, batch size, and the number of copies averaged. No paper shows averaging working past about 128 copies.
   The workable shape **multiplies several modest axes**: separate model *paths*, model *modules* (slices), expert
   *groups*, *replicas* (copies of the same piece), and *redundancy* (duplicate jobs). Each axis stays near
   a size someone has tested.
2. **The engine should be local learning per module (SOLO), not DUST.** SOLO (2609.35440) cuts the model into
   2–4 modules that each train against their own local copy of the output layer. Only forward activations cross
   between modules. Quality comes within about 3.5% of full backprop at 2B parameters, and the gap shrinks as
   models grow. Add learned compression at the module boundary (ResBM, 2604.11947, 100–128×), and a volunteer
   needs about **1 Mbit/s** of boundary traffic (our arithmetic). Every forward-only method in round 1, DUST
   included, is either fine-tuning-only or proven only at small scale. Our own lab runs agree (Section 3.8):
   SOLO matched backprop at 10M–120M parameters with up to 6 modules, while every forward-only method,
   DUST included, stalled near loss 7.4–8.2 against about 5.1 for backprop. DUST moves to a side role for cards
   that can only run inference.
3. **Gradient isolation makes each piece of work a checkable "share".** A SOLO module's update depends only on
   its signed input activations, its assigned data, and its starting weights. That makes it the CrowdGPT
   equivalent of a mining-pool share: small, self-contained, and replayable on another card without the rest
   of the model. This is what makes zero-trust checking affordable at scale, and it is the most original idea
   here. No paper combines local learning with verification.
4. **One redundancy budget can do three jobs.** Assigning each piece of work to two volunteers does three things.
   It covers people leaving mid-job (SPARe, 2603.00357). It catches cards that are silently wrong (LLM-PRISM,
   2604.10390). It catches cheaters, because the two results are compared. That costs about 2× compute, and it
   replaces three separate mechanisms.
5. **Make growth a scheduled, rule-driven event, like Bitcoin's halving.** Two 2026 results give firm rules.
   *Optimal Splitting* (2603.19149) gives a scaling law for when to fork a shared trunk into independent
   specialists. *Expert Upcycling* (2604.19835) shows that doubling the number of experts mid-run works.
   Trigger both from verified counts in the public record, such as tokens trained or active contributors.
6. **At a million GPUs the limit is data, not compute.** A million cards could chew through all of FineWeb
   (about 15T tokens, from memory) in under a week on a 27B model (our arithmetic). Some of the network has to
   do data work: filtering, deduplication, synthetic data, and evaluation.

---

## 2. Five walls at million scale

| Wall | Our arithmetic | What follows |
|---|---|---|
| **Compute is not the problem** | 1M cards × 6 TFLOPS (guess) ≈ 6×10¹⁸ operations/s ≈ 1.9×10²⁶ per year. That trains ~1.3T parameters on ~25T tokens at compute-optimal ratios, or ~630B on ~12.5T at 25% efficiency (guess). | Frontier-class compute *if* the system works. Target large sparse models (most parameters idle per token), not huge dense ones. |
| **One model cannot absorb 1M cards by averaging** | 1M cards × 16k tokens each = 16B tokens per step. The largest batch shown to work is ~100M tokens (SOAP/Muon, 2607.20548). The most copies averaged in any test is 128 (SPICE, 2611.23053). | Need several axes of parallelism, not just more copies (Section 4). |
| **People constantly leave** | With ~3-hour sessions (guess), 1M volunteers ≈ 90 departures per second. SPARe reports datacenter mean time between failures of ~5 minutes at 600k GPUs. | Treat departures as normal. Never wait for "everyone". Lease work with time limits and pre-place spare copies. |
| **Some cards are silently wrong** | LLM-PRISM's fault rates scale to ~7 silent corruption events per day at 1M GPUs (our arithmetic), and probably more on consumer cards. Some faults leave loss looking normal while weights drift. | Checking loss alone is not enough. Compare update vectors, not just loss. |
| **Data runs out** | A 27B dense model at full rate ≈ 3.7×10⁷ tokens/s ≈ 3.2T tokens/day. | Data supply becomes the bottleneck. Assign volunteers to data roles and favour more parameters per token (sparse models). |

---

## 3. What the 36 papers give us, by problem

### 3.1 Fit big models on small cards

| Paper | Main point | Use in v2 | Evidence ceiling |
|---|---|---|---|
| **SOLO** 2609.35440 | Cuts the model into K modules. Each trains with its own small head that predicts through a frozen copy of the shared output layer, refreshed every 10–50 steps. Only activations travel forward between modules. | **Main engine.** Each module is a separate job. No backward traffic across machines. | 340M–2B params, 15B tokens, K = 2–4, one machine, honest workers only. K=8 costs +89% compute. |
| **ResBM** 2604.11947 | Learned autoencoder at each pipeline boundary plus a low-rank skip path. Compresses boundary data 100–128× with no quality loss at 2B. | Compress module boundaries to ~128 bytes per token (our arithmetic). That is cheap to send, hash, store and replay. | 2B, 8 blocks, unequal optimizer comparison. |
| **Async pipeline** 2606.30634 | A one-step-stale pipeline works with Muon: +0.012 loss vs +0.278 with AdamW. 10B MoE matches synchronous training. | Within a module split across 2–4 cards, accept one step of delay. Use Muon-family optimizers. | Delay simulated, fixed lockstep. |
| **Agora** 2607.13332 | Real internet run: 8.6B params, 500B tokens, 40 days, 330 nodes, 669 joins / 607 departures, replicated pipeline stages, ~100× boundary compression. | The closest existing system to v2. Lessons: per-hop *latency* limits speed more than bandwidth, and coordination struggled at only 60 concurrent machines. | Datacenter-class volunteers (median 504 GB RAM, 703 Mbit/s); operator ran key stages. |
| **MegaTrain** 2604.05091 | Host RAM holds the model and the GPU streams one layer at a time. Trains 14B on a 24 GB card with ~100 GB RAM. | Role for volunteers or orchestrators with lots of RAM: trunk, head/tail, or checkpoint verification. | Fine-tuning only. Needs large RAM. |
| **ECO** 2601.22101 | Drops the FP32 (32-bit) master copy of weights, keeps FP8 (8-bit) weights, and folds the rounding error into momentum. ~9 bytes per parameter instead of 12. | Lets more parameters fit on 16 GB. Halves download size for an FP8 canonical model. | 800M params. The ~3 bytes/param combined with 8-bit Adam is untested. |

### 3.2 Cut communication between replicas

| Paper | Main point | Use in v2 | Evidence ceiling |
|---|---|---|---|
| **Covenant-72B / SparseLoCo** 2603.08163 | 30 local steps, then send the top 64 of every 4,096 values, 2-bit, with error feedback (>146× smaller). Gauntlet scores each peer by how much its update lowers loss. | Replica sync format (~29 MB per round for a 1B-parameter module, our arithmetic). Loss-based scoring as one trust signal. | ~25 peers, each 8×B200 datacenter GPUs. Central object storage. |
| **Decoupled DiLoCo** 2604.21428 | Learners never wait for each other. The syncer merges whatever has arrived. 88% goodput with 1.2M simulated chips failing. | Rounds close on a time budget, never "wait for all". | 2–9B, simulated failures, central syncer. |
| **GASLoC** 2606.11081 | DiLoCo where each worker talks to 1–2 random peers instead of everyone. Slow links do fewer local steps. | Gossip inside small neighbourhoods, so there is no global all-to-all. | ≤32 replicas, 134M. |
| **SPICE** 2611.23053 | Sparse gossip with peers picked from a shared seed and changed every round. Scales to 128 replicas, but only if each worker talks to half of them. | Pick neighbourhoods from the public randomness beacon (prevents collusion). Keep neighbourhoods small and add a hierarchy above them. | 134M simulated, H=1 (sync every step). |
| **LACCO** 2611.17991 | Two parameter copies hide one round of communication behind the next local steps with no staleness. | Lets slow uploads overlap with compute. | ≤0.5B, ≤16 replicas, +12 GB RAM at 1B. |
| **Replica scaling** 2610.19064 | Most of the quality lost to averaging many replicas comes from the AdamW optimizer. At 64 replicas the loss penalty is AdamW +11%, SOAP +4.8%, SPlus +3.1%. A simple rule sets beta2. Plain equal-weight averaging is near-optimal. | Use SOAP/SPlus/Muon for local steps. Use **equal weights**, so self-reported token counts never set influence. | ≤512M, ≤96 replicas. |
| **THROTTLE** 2609.39357 | Async SGD where each client's repeat submissions get shrinking step sizes, so flooding cannot dominate. | Limits per-contributor influence per round. | Small models. |

### 3.3 Grow and split the model over time

| Paper | Main point | Use in v2 | Evidence ceiling |
|---|---|---|---|
| **Optimal Splitting** 2603.19149 | Train a shared trunk, then fork K copies that never talk again, routed by nearest data centroid. A scaling law predicts when to fork. 16 × 2.7B specialists beat a 6.7B dense model. | **Forest of paths:** the strongest pool-like axis. Fork timing is deterministic from the record. | K ≤ 64, ≤2.7B per path. Memory is K × model size. |
| **Expert Upcycling** 2604.19835 | Double the experts mid-run (copy busy experts, add router noise). Matches from-scratch quality with 24–32% fewer GPU-hours. MoE→MoE beats dense→MoE. | **Halving-style growth event:** double experts when verified participation doubles. Go MoE early instead of staying dense to 9B. | 7B→13B, 1B active. |
| **SPES** 2602.11543 | Each node owns separate experts and trains a shared trunk, uploading only what it trained. Matches centralized training at 1B. | **Expert-owner groups** inside a module. Uploads shrink to owned experts plus trunk. | ≤16 nodes, 17 Gbps links, needs warm-up. |
| **SOMA** 2609.37899 | Independent experts trained separately on data clusters cut gradient-estimate noise roughly in proportion to their number. | Supports paths. Damage from a bad actor stays inside one expert. | 8M-parameter LSTMs. |

### 3.4 Survive churn and faults

| Paper | Main point | Use in v2 |
|---|---|---|
| **SPARe** 2603.00357 | Pre-place r copies of each data shard with a fixed spacing scheme. Aggregate as soon as every shard is covered once. Masks ~426 failures for ~2.8× compute vs 20× for naive replication. | Duplicate assignment that also serves verification. Simulation only. |
| **Cross-Facility / DARL** 2610.03457 | Elastic outer step works whether 0 or many sites report. Data is leased with heartbeat time limits, with work stealing and exactly-once processing. | Lease model for data shards and jobs. (Its single coordinator is what v2 removes.) |
| **LLM-PRISM** 2604.10390 | 7,664 fault-injection runs. 57–76% of faults are harmless. Gradual weight drift passes loss checks. | Assume some contributors are silently wrong at all times. Compare updates. |

### 3.5 Trust and verification

| Paper | Main point | Use in v2 |
|---|---|---|
| **OPEN-1B** 2609.17380 | Bit-exact replay of any step of a 1.6B run across NVIDIA, CPU and Apple Silicon, at 5–7× compute cost. | Use only for **disputes** and small exact computations (aggregation, boundary hashes), not routine checks. The method (RepOps) is proprietary. |
| **Chorus** 2609.37367 | Peers judge each other's updates by behaviour, pool the evidence, and vote. Attack success drops from 48–63% to 0–2%. | Behaviour screening of path-level updates by several beacon-picked verifiers. Corrects our earlier claim that no validated defence exists (fine-tuning only so far). |
| **Giskard** 2606.19129 | Private robust median among 10⁶ parties via committee trees drawn from a public seed. Costs 146 GB–3 PB per party *per number*. | Private aggregation is infeasible for LLM updates. Reuse the **public-seed committee tree** idea in the clear. |
| **Covenant Gauntlet** 2603.08163 | Scores each update by the loss drop it produces on a small batch. | Cheap usefulness score alongside replay. |

### 3.6 More learning per GPU-hour

| Paper | Main point | Verdict for v2 |
|---|---|---|
| **Olmo Hybrid** 2604.03444 | Three Gated DeltaNet (a linear-attention layer type) layers per attention layer. Reaches the same loss with 35% fewer tokens at 7B. Projected gain ~1.3× at 1B, growing with size. Much smaller inference state. | **Adopt at genesis.** It is the largest per-token gain in the set, and it eases the data wall. Check consumer-card and Apple kernel support and determinism first. |
| **SOAP/Muon at scale** 2607.20548 | Beat AdamW up to 72B MoE, and tolerate 2–4× larger batches. | **Adopt.** Also wins on replica scaling and async delay (above). |
| **NVFP4** 2610.24343 | 4-bit matrix maths within ~1% of BF16 (16-bit brain float) over 1T tokens if the last ~15% of layers stay BF16. | Blackwell-only. Keep it out of the shared canonical maths, because mixed hardware would break replay checks. Optional for unchecked roles such as data scoring. |
| **FlashAttention-4** 2603.05451 | 1.1–1.3× faster attention, but only on datacenter Blackwell. | Not for consumer cards (they stay on FlashAttention-2-class kernels). Its scheduling idea transfers. |
| **Soft-MTP** 2610.38331 / **ECHO** 2610.24766 | Multi-token prediction (MTP) does **not** improve next-token loss. Soft-MTP explains why. ECHO gains 0.5–1.1 points downstream at 10B. | Not a pretraining lever. Maybe later for a speculative-decoding drafter. |

### 3.7 Forward-only (zeroth-order) methods from round 1

SZO-κ (2611.08712), WIS-MeZO (2611.16317), MobileSpec (2611.04797), EGGROLL LOO-ROLL (2609.10980),
ZO-Muon (2602.17155), ZO edge of stability (2604.14669) and LoCA (2608.03020) all improve forward-only
(zeroth-order, ZO) training. Each one is either **fine-tuning** (WIS-MeZO up to 32B, ZO-Muon 8–13B,
MobileSpec at 4-bit on-device, LoCA up to 14B) or **small-scale** (SZO-κ is an MLP/CNN study). None
pretrains a language model at scale.
- **Use:** a role for the long tail of devices that can only run inference (Apple laptops, older cards, possibly phones).
  They could do post-training of specialist adapters (MobileSpec, WIS-MeZO), evaluation, and data scoring.
- **Not a use:** the main pretraining engine.

### 3.8 Our own lab results (`research/lab`)

The lab harness in `research/lab/` has run most of the round-1 methods plus SOLO and SparseLoCo. The write-up
in `research/lab/REPORT.md` has empty results sections, so these numbers come straight from `research/lab/results/*.json`
and `research/lab/logs/`. REPORT.md does already describe a design called MOSAIC; see `SYNTHESIS-round2-addendum.md`.
All runs: one Apple M5 Max (MPS), sequence 256, batch 16, one seed. Lower held-out loss is better. Some runs were
still in progress when this was written.

**Time-budget runs, 10M preset (~7–8M parameters, about 120 s each):**

| Method | Final held-out loss | Note |
|---|---|---|
| Adam backprop (baseline) | 5.09 | 4.75 with 4× batch |
| SOLO, 1 module (= backprop) / K=2 / K=3 / K=2 private heads | 5.06 / 5.11 / 5.08 / 5.07 | Local learning costs almost nothing here |
| SOLO K=2, activations 4 steps stale | 5.37 | Pipeline delay hurts; see below |
| LoCA (forward-only with local credit) | 6.61–6.75 | Best forward-only method |
| DUST population 8 / 32 | 7.43 / 8.15 | Bigger population did worse in a fixed time |
| MeZO, SZO, LOO-ROLL, ZO-Muon, ZO edge-of-stability, WIS-MeZO | 7.46–7.91 | All stall near the unigram level |
| MobileSpec from scratch | 8.22 | Fine-tuning method; not meant for pretraining |
| Fine-tuning from an Adam checkpoint: MeZO / WIS-MeZO / MobileSpec / LoCA vs Adam | 5.60 / 5.60 / 5.60 / 5.50–5.59 vs 5.10 | Forward-only also trails on fine-tuning |
| SparseLoCo (top-k, 2-bit, error feedback) vs dense DiLoCo | 5.07 at 858 KB/round vs 4.88 at 117 MB/round | ~137× less upload, 0.23 bits per parameter |
| SparseLoCo without error feedback | 5.66 | Error feedback is essential |
| SparseLoCo with a Byzantine sender (1 of 4 replicas flips its update), plain mean | 10.52 (diverged) | One bad sender wrecks a plain average |
| Same, with Gauntlet-style held-out filtering | 5.97 at 2.1M tokens; run stopped there (no result file) | Filtering kept training alive while it ran |

**Fixed 4M-token SOLO runs (K=1 is ordinary backprop):**

| Size | K=1 | K=2 | K=3 | K=4 | K=6 |
|---|---|---|---|---|---|
| 10M preset | 5.26 | 5.21 (private heads 5.20) | 5.25 | — | 5.22 |
| 100M preset (119.6M) | 5.17 | 5.14 | — | 5.14 | 5.12 |
| 300M preset (310M) | 5.12 (5.45 at 2M tokens) | 5.44 at 2M tokens, then stopped | — | not run | — |

What this adds to the papers:

- **SOLO held up to K=6 at 120M parameters**, beyond the paper's tested K≤4. At these short runs it even edged
  out backprop. That is within one-seed noise, so read it as "no visible penalty", not "better".
- **Pipeline delay is the weak point.** Four steps of staleness at module boundaries cost about 0.26 in loss.
  A lag-tolerance study (`stale=1,2,4,8` at 10M tokens) is defined in `research/lab/arms/` but has not run. Crews should run in
  near-lockstep, or need the error-feedback fix from Async PP (2606.30634).
- **Robust aggregation is not optional.** A single bad sender made a plain mean diverge. This is direct
  evidence for the trust layers in Section 4.5.
- **Forward-only methods lost by a wide margin**, both from scratch and when fine-tuning. This is the measured
  basis for demoting DUST.
- **SOLO boundary traffic is large without compression:** 6.3 MB per step at K=2 and 18.9 MB at K=4 for the 100M
  preset. That is why ResBM-style boundary compression (100–128×) is part of the recommendation.

Limits: tiny models, at most 4M tokens, one seed, one machine, and no real network.

---

## 4. Recommended shape: the cell mesh

### 4.1 The cell

A **cell** is one unit of assigned work: *(path p, module m, expert group e, replica r, round t)*.
It has:

- **Inputs (all hashed in the record):** starting weights hash; assigned data shard (drawn from the beacon,
  not chosen by the contributor); hash of the signed input activations from the upstream module (ResBM-compressed);
  hash of the frozen output-layer copy.
- **Work:** H local steps of SOLO training on that module (or on its owned experts), with a SOAP/SPlus/Muon-family
  optimizer and FP8 state (ECO).
- **Outputs:** commitment to the compressed update (SparseLoCo format); hash of the output activations sent downstream;
  loss figures.

Because SOLO cuts gradient flow between modules, everything needed to **replay** a cell is small: a ~1B-parameter
slice, some compressed activations, and a data-shard ID. Any 16 GB card can recheck any cell. That is the
mining-pool "share", and it is what makes zero trust affordable.

### 4.2 Six axes, each near a tested size

| Axis | What it is | Tested up to | Source | Example |
|---|---|---|---|---|
| Paths | Independent specialists forked from a shared trunk | 64 | Optimal Splitting | 64 |
| Modules | SOLO slices of one path | 4 | SOLO | 4 |
| Cards per module | Short pipeline inside a module (one-step delay) | 8 stages | Agora, Async PP | 2 |
| Expert groups | Disjoint expert ownership inside a module | 16 nodes | SPES | 8 |
| Replicas | Copies of the same piece, synced by gossip plus compressed outer steps | 96–128 | Replica scaling, SPICE | 128 |
| Redundancy | Duplicate cells for churn, faults and cheating | r = 2–3 | SPARe | 2 |

64 × 4 × 2 × 8 × 128 × 2 = **1,048,576 cards.** Each axis is near something measured. **No one has
combined them.** The combination is the research risk.

What the example implies (all our arithmetic, labelled guesses):
- **Per card:** ~1B parameters of training state, which fits 16 GB with FP8 weights and compact optimizer state.
- **Per path:** 4 modules × 2 cards × 8 expert groups ≈ 64B total parameters, sparse (only some experts run per token).
- **Forest:** 64 paths ≈ 4T total parameters, routed by topic at inference. That is the "1T-class model" without
  any card ever holding more than ~1B.

### 4.3 Bandwidth check for a 16 GB card on a home link (our arithmetic)

- **Compute rate:** at 6 TFLOPS (guess), training a ~1B-parameter slice runs about 1,000 tokens/s (6 × params operations per token).
- **Module boundary traffic:** ResBM ~128 bytes/token gives ~128 KB/s ≈ **1 Mbit/s**. Uncompressed, at width 4096
  in BF16 (~8 KB/token), it would be ~64 Mbit/s, more than most home uploads. **Compression at the boundary is what
  makes slicing work at home.**
- **Replica sync:** SparseLoCo ~0.234 bits/param gives ~29 MB per outer round for a 1B slice, ~12 s at 20 Mbit/s.
  LACCO-style overlap hides it.
- **Joining:** download ~1 GB (one FP8 slice), not the whole model.

### 4.4 Sync hierarchy (no global all-to-all, no single coordinator)

1. **Neighbourhood** (8–16 replicas): gossip with 1–2 peers per round (GASLoC/SPICE). The peer schedule is drawn
   from the public beacon, so colluders cannot pick each other. Slow cards do fewer local steps.
2. **Shard committee**, one per (path, module): a small committee drawn from the beacon (Giskard-style tree, in
   the clear). It collects neighbourhood updates and applies clipping to the lower-median norm (Covenant/THROTTLE)
   with equal-weight averaging. It closes the round when shards are *covered*, never when everyone has reported
   (SPARe/Decoupled DiLoCo). It writes a shard checkpoint entry.
3. **Root record:** orchestrators commit only shard checkpoint hashes and verdicts. This keeps the shared record
   small at any scale and avoids the coordination collapse Agora hit at 60 machines.

Crews (the cards inside one module's pipeline) are formed **by latency region**, randomised within the region.
Agora showed per-hop latency is the real speed limit.

### 4.5 Trust in layers

1. **Duplicate cells** (r = 2 for new contributors; for established ones a sampled 5–10%, a guess) are compared
   by update direction and size within a tolerance. The tolerance is set from the cross-GPU measurement in Section 6.
   This also catches silent hardware faults (PRISM).
2. A **disagreement** triggers a third replay on a different hardware vendor.
3. **Disputes** go to bit-exact deterministic replay (OPEN-1B-style) of just that cell. That costs 5–7×, but on a
   tiny fraction of cells.
4. **Usefulness score:** the loss drop on a held-out batch (Covenant Gauntlet).
5. **Behaviour screening** (Chorus-style) of path-level checkpoints by several beacon-picked verifiers, before
   they reach the root record.
6. **Influence caps:** equal weights, norm clipping, and per-round limits (THROTTLE). Self-reported numbers never set influence.

### 4.6 Growth events (the halving analogue)

All triggers are computed from the public record. Nobody decides them by hand:

| Event | Trigger (proposal) | Basis |
|---|---|---|
| Add modules (K = 2 → 4) | Model reaches the size where SOLO K=4 is within the tested gap (~2B) | SOLO |
| Become MoE | Early, around 1B, rather than at 27B | Upcycling: MoE→MoE beats dense→MoE |
| **Expert doubling** | Verified active contributors double since the last doubling | Expert Upcycling |
| **Forest split** | Verified trunk tokens reach the split point the scaling law predicts for the current size and K | Optimal Splitting |
| Path doubling | Each path saturates its replica limit (~128) | Replica scaling, SPICE |

This replaces the dense 1B → 4B → 9B → 27B MoE ladder in `archived/crowdgpt-v2/05-scale-roadmap.md`.

### 4.7 Jobs for every kind of device

| Device | Role |
|---|---|
| 16 GB+ consumer GPU | Module or expert cell trainer, cell replayer |
| Large-RAM machine (≥128 GB) | Trunk, head/tail, or large-slice roles via MegaTrain-style streaming; checkpoint verifier |
| Inference-only (Apple, older cards) | Evaluation, Chorus-style behaviour screening, data filtering and scoring, forward-only adapter post-training (MobileSpec / WIS-MeZO) |
| Datacenter Blackwell (if any volunteer) | Same roles, but no FP4 in replay-checked work |

**Data work as a first-class job.** Volunteers filter, deduplicate, score, and generate candidate data. Each data
job is small and replayable, so it is checked the same way as training cells.

---

## 5. What to drop or demote

- **DUST as main engine:** move to a forward-only probe role. Our lab runs put DUST at 7.4–8.2 held-out loss
  against about 5.1 for backprop and SOLO at equal time (Section 3.8).
- **Dense growth beyond ~1–4B:** switch to MoE early and grow by expert doubling and path forking.
- **Multi-token prediction for pretraining loss:** no gain shown.
- **FlashAttention-4 and NVFP4 in canonical maths:** wrong hardware, and they break cross-vendor replay.
- **Private (MPC, multi-party computation) robust aggregation:** about 10⁵–10⁸× too expensive (Giskard).
- **Token-weighted averaging:** equal weights are near-optimal (Replica scaling), and the counts can be faked.
- **Eager DiLoCo-style overlap:** degrades with more replicas (LACCO). Use LACCO's two-copy method instead.

---

## 6. Experiments to run first (Phase 0 additions)

Effort is in agent-hours. All numbers are **guesses**; the only comparison is the v1.5 build (23 commits over
about 5–6 active hours). GPU run time and human review are not included.

| # | Experiment | Answers | Guess |
|---|---|---|---|
| X1 | SOLO is already shown at 10M–120M (Section 3.8). Still to do: 300M with K>1, more seeds and tokens, a ResBM boundary, and the staleness sweep | Does the main engine hold at 300M+ with compressed boundaries and real lag? | 3–6 h |
| X2 | Cell replay across Apple MPS vs CUDA: update cosine and size gap for identical inputs, plus PRISM-style injected faults | Sets the duplicate-cell tolerance; tells us if gradual drift is caught. | 3–6 h |
| X3 | 16–64 simulated replicas: gossip with 1–2 beacon-drawn peers plus SparseLoCo outer step, H>1; SPlus/SOAP vs AdamW; beta2 rule | Can replica sync scale without all-to-all? (H>1 gossip is untested; SparseLoCo itself already works in our lab at 4 simulated replicas.) | 4–8 h |
| X4 | Small MoE at 100M: expert doubling mid-run, then a forest split at the predicted point | Do growth events work as scheduled? | 4–8 h |
| X5 | Gated DeltaNet 3:1 hybrid vs plain at 100M, on CUDA and MPS, including determinism | Adopt at genesis? | 2–4 h |
| X6 | ECO FP8-no-master memory and quality on a 16 GB card | How many parameters per card? | 2–4 h |
| X7 | Churn simulation: leases, SPARe placement, 3-hour sessions, coverage-based round close | Goodput under realistic departures. | 2–4 h |

Total about 20–40 agent-hours. That is about 4–8× the v1.5 build, because it covers seven separate studies.
These replace or merge TODO items 0.5, 0.6, 0.11–0.18 in `archived/crowdgpt-v2/TODO.md`.

---

## 7. Gaps and honest limits

- **Nobody has combined these methods.** The cell mesh is a design built on separate results.
- Internet evidence stops at 8.6B parameters and 60 concurrent machines (Agora), or ~25 datacenter peers (Covenant-72B).
- SOLO is proven only for K ≤ 4 on one machine with honest workers. Untested: modules arriving out of step, mixed hardware, attackers.
- Gossip with more than one local step between syncs is untested past 32 replicas.
- Byzantine (malicious) defences are validated only for fine-tuning (Chorus) or small models.
- Gated DeltaNet kernels on consumer and Apple hardware, and their determinism, are unverified.
- The data wall is real, and synthetic or filtered data at this scale needs its own trust design.
- Some round-2 notes depend on how papers state their own results. Read each `notes/<id>.md` Caveats section before relying on a number.

---

## 8. What this changes in the archived v2 plan

| v2 document | Change |
|---|---|
| `adr/0004` (training lanes) | Make `local_modules` (SOLO) the primary lane. DUST becomes a forward-only side lane. |
| `adr/0005` (model sharding) | Replace "pipeline crews / independent paths" with the six-axis cell mesh. Add ResBM boundaries and latency-region crews. |
| `05-scale-roadmap.md` | Switch to MoE early. Growth by expert doubling and forest split, triggered by scaling-law rules. Add data roles. |
| `02-trust-and-verification.md` | Cell as the replay unit. Duplicate cells covering churn, faults and cheating. Deterministic replay for disputes only. Chorus screening. PRISM threat. |
| `01-architecture.md` / `03-consensus-and-record.md` | Three-level sync (neighbourhood, shard committee, root record). Root records shard checkpoints only. |
| `TODO.md` | Add X1–X7. |

---

## 9. Papers used

Round 1: 2611.08712 SZO-κ · 2611.16317 WIS-MeZO · 2611.04797 MobileSpec · 2609.10980 EGGROLL Unrolled ·
2609.37899 SOMA · 2602.17155 ZO-Muon · 2604.14669 ZO Edge of Stability · 2608.03020 LoCA · 2609.35440 SOLO ·
2603.08163 Covenant-72B · 2604.21428 Decoupled DiLoCo · 2609.39357 THROTTLE.

Round 2: 2607.13332 Agora · 2604.11947 ResBM · 2606.30634 Async pipeline · 2604.05091 MegaTrain · 2602.11543 SPES ·
2604.19835 Expert Upcycling · 2603.19149 Optimal Splitting · 2611.23053 SPICE · 2606.11081 GASLoC ·
2611.17991 LACCO · 2610.19064 Replica scaling · 2610.03457 Cross-Facility/DARL · 2603.00357 SPARe ·
2604.10390 LLM-PRISM · 2609.17380 OPEN-1B · 2609.37367 Chorus · 2606.19129 Giskard · 2601.22101 ECO ·
2610.24343 NVFP4 · 2607.20548 SOAP/Muon at scale · 2604.03444 Olmo Hybrid · 2603.05451 FlashAttention-4 ·
2610.38331 Soft-MTP · 2610.24766 ECHO.
