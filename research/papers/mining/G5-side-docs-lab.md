DRAFT — requires human review

# G5 mining: side research docs, archived v2 research, and the lab

Date: 2026-11-18. Baseline: `research/papers/SYNTHESIS-round2.md` (called "the synthesis").
Sources read: `research/README.md`, `research/communication-efficiency.md`, `research/open-source-ecosystem.md`,
`research/zo-es-lineage.md`, `archived/crowdgpt-v2/research/02-v1-baseline.md` and `03-prior-art.md`, and
`research/lab/` (REPORT, SPEC, plans, arms, logs, results, scripts). None of these files has a classification or
personal-data marking on line 1. No lab job was running when I checked (`ps`), so nothing was disturbed. The only
thing I ran was `crowd_scale.py`, a few-second script that reads the JSON results files.

Labels: **Source says** = stated in the file cited. **Inference** = my reasoning. **Guess** = not measured.
Acronyms: SOLO = shared-output local learning (each model slice trains on its own local loss); EF = error feedback;
DiLoCo = distributed low-communication training (many local steps, then an "outer" merge step); H = local steps per
merge; M = number of replicas; PoL = proof of learning; BFT = Byzantine fault tolerant (consensus that survives
malicious nodes).

---

## 1. Top 5 unused ideas or results, ranked by impact

### 1. MOSAIC: each worker recomputes the earlier slices itself instead of receiving activations from peers
- **Source says:** `research/lab/REPORT.md:64-109` and `research/lab/methods/mosaic.py:18-31` define MOSAIC. Worker w
  owns module k. It runs modules 0..k-1 *forward-only at the round's published weights*, trains only module k for H
  local steps, and uploads module k's SparseLoCo-compressed change. There is "no peer-to-peer activation streaming,
  unlike SOLO's pipeline". The code supports churn (`drop`), budget rounds (`het=budget`), lower-median clipping,
  Gauntlet filtering, Byzantine attackers and non-IID data (`mosaic.py:57-75`). A 15 s smoke test reached loss 5.78
  with K=2 versus 5.89 with K=1, at 0.55 versus 0.98 MB of upload per step (`results/smoke-mosaic.json`,
  `smoke-mosaic-k1.json`). The full arms in `arms/round2.txt`, `arms/round2b.txt` and `arms/solo-crowd.txt` were
  never run: there are no `r2-*` or `s-mo-*` result files.
- **How it applies (inference):** the synthesis's cell mesh relies on pipeline crews, ResBM compression at module
  boundaries, latency-region crew formation, and signed upstream activations (Sections 4.1, 4.3 and 4.4). MOSAIC
  removes all four. It also removes the main weak point the synthesis names, which is staleness at module boundaries
  (synthesis line 177). Trust improves too: a cell's inputs can be rebuilt from public weights plus a data-shard ID,
  so a cheating upstream volunteer cannot poison the inputs to downstream modules. The cost is redundant forward
  compute for the earlier modules, plus downloading their weights. A forward pass is roughly one third of a training
  step, and weights can be held in FP8 or 4-bit (**guess**).
- **Confidence:** medium-low. This is a 15-second smoke test on one machine, and upstream weights stay frozen within
  a round, which is a kind of staleness the lab has not measured.
- **Cheap next step:** run `arms/solo-crowd.txt` (`s-mo-k2w8`, `s-mo-k1w8`, `s-bp-128x8`, plus the `-drop`, `-het`
  and `-byz-g` arms) at 10M tokens. This is a few hours on the M5 Max (**guess**).

### 2. The lab's own network model says boundary compression or recomputation is mandatory, and a flat hub is impossible
- **Source says:** `research/lab/crowd_scale.py:1-9,60-63` projects lab runs onto consumer cards. Its defaults are a
  1.6× step-time factor (a measured RTX 4060 Ti versus M5 Max ratio for DUST at 1B), 20 Mbit/s up and 200 Mbit/s down.
  My run of it gave:
  - SOLO at 100M parameters spends **79% (K=2), 91% (K=4) and 93% (K=6)** of each worker step on the network.
    Worker throughput falls from 6,644 tokens/s (K=1) to 1,279, 489 and 302 tokens/s.
  - SparseLoCo (H=30) spends 19% on the network. Hub ingress is **27.6 Gbit/s at 10,000 workers** and
    **2.76 Tbit/s at 1,000,000 workers**.
- **Inference:** the synthesis treats ResBM as "part of the recommendation" (line 185). These numbers make boundary
  compression, or MOSAIC-style recomputation, a hard requirement: without it, SOLO loses about 5–20× in throughput on
  home links. The ingress numbers give the synthesis's three-level hierarchy (Section 4.4) a concrete basis. Even
  with ~146× compression, no single receiver can take a million uploads, so committee fan-in needs sizing (**guess:**
  fewer than about 1,000 uploads per committee per round). One caveat: for SOLO, the "ingress" column counts
  peer-to-peer boundary traffic, not traffic to a coordinator (`plans/solo.md:20-22`, "up = boundary activations").
- **Confidence:** medium. The model is simple: it ignores latency and large-batch efficiency.
- **Cheap next step:** add a `--compress` factor (ResBM 128×) and a committee fan-in column to `crowd_scale.py`, then
  rerun.

### 3. Outer-loop rules the synthesis leaves unset: merge learning rate, number of local steps, momentum damping
- **Source says:**
  - DiLoCo scaling laws (arXiv 2503.09799): the best outer learning rate does not depend on model size above 335M
    parameters, rises with M and H, and has fitted formulas (`communication-efficiency.md` §2.3, lines 142-150).
  - OpenDiLoCo at 1.1B parameters: H=125 trails data-parallel training by 0.24 in loss, while H=500 trails by 0.62.
    The usable number of local steps shrinks as models grow (§2.2, lines 130-140).
  - Streaming DiLoCo: syncing one fragment at a time cuts peak bandwidth 8×, and overlap should stay at or below
    5 inner steps (§2.4, lines 152-160).
  - Radial-directional averaging (RDA) "drastically improves" results at M=16, but should not be used on embeddings
    (§2.6, lines 189-202).
  - A delay-adaptive learning rate: dropping the most-delayed half of updates costs at most 2× (§3.2, lines 220-237).
  - Lab: asynchronous Decoupled DiLoCo with the paper's default Nesterov momentum (0.7/0.9) **oscillated between 6.77
    and 7.72**. Damping momentum to 0.5 fixed it, and async goodput was 0.95 versus 0.50 for synchronous
    (`research/lab/plans/ddiloco.md:39-52`; 15 s smoke tests on a shared GPU).
- **Inference:** the synthesis specifies equal-weight averaging but no outer optimizer, no rule for H, no fragment
  schedule and no embedding exception. At a million cards with time-budget rounds, merge events are frequent, which
  is exactly the setting where the lab saw oscillation.
- **Confidence:** medium-high for the papers (4B and 10B parameters validated), low for the lab smoke tests.
- **Cheap next step:** fold an (M, H) → outer-learning-rate rule and an H-versus-size cap into experiment X3, and
  rerun the `r1-dd-*` arms in `arms/round1.txt`, which never produced results.

### 4. The duplicate-cell check needs PoL-attack and collusion defences
- **Source says:**
  - PoL was broken by the "infinitesimal-update" attack (learning rate ≈ 0 passes any replay tolerance) and the
    "blindfold top-Q" attack. Lessons: any tolerance band is attack surface; the verifier must choose what to check
    using randomness revealed after commitment; and update size must be bounded by protocol
    (`03-prior-art.md` §2.1, lines 75-86).
  - "A Little Is Enough" (arXiv 1902.06156): small coordinated shifts inside the honest-variance band defeat median
    and Krum (§2.5, lines 121-131).
  - In v1, the 3-client pool cross-check passes 3 colluding clients, and bf16 differences across hardware flag honest
    users (`02-v1-baseline.md` §3, line 99; §5, line 154).
  - Lab THROTTLE smoke test: attackers always sit at the clip radius while honest small-batch norms (0.5–0.7) sit
    below it, so attackers get *more* influence per message. Fast senders hold about 1/(q−1) of the round's weight
    (`plans/throttle.md:44-66`).
  - Gauntlet scoring at α=1 dropped honest peers. Scoring each peer at its 1/W share fixed this
    (`plans/sparseloco.md:45-49`).
  - Gauntlet also compares improvement on a peer's *assigned* data with improvement on random data, as a proof of
    computation (`03-prior-art.md` §2.4, lines 109-114).
- **Inference:** the synthesis's layer 1 (compare duplicate cells "within a tolerance", Section 4.5) is exactly the
  PoL tolerance band. Pair assignment must come from a beacon after commitment. Cells need a minimum-progress floor,
  so near-zero updates earn nothing. Clipping should use the honest norm distribution, not a fixed radius. The
  assigned-versus-random Gauntlet test is a cheap extra layer the synthesis does not use.
- **Confidence:** high for the literature, low for the lab smoke tests.
- **Cheap next step:** add an infinitesimal-update attack and a colluding-pair attack to experiment X2.

### 5. Unreported SOLO lab data: 300M parameters at K=2, slice alignment by depth, and compute overhead
- **Source says:**
  - **300M preset, K=2: 5.4355 at 2.003M tokens, versus 5.4525 for K=1** (`logs/s-300m-k2.log`, last line;
    `logs/s-300m-k1.log`, iteration 489). The queue was then killed (`logs/solo-a.log`: "Killed: 9"). The synthesis
    says "not yet run" (line 171).
  - Per-module cosine of the local gradient to the backprop gradient falls toward the input end of the model:
    100M K=4 gives cos0 = 0.64, cos1 = 0.92, cos2 = 0.95; 100M K=6 gives 0.72 up to 0.96; 10M K=3 gives 0.77 and 0.93
    (last lines of `logs/s-100m-k4.log`, `s-100m-k6.log`, `s-10m-k3.log`). At 300M with K=2, cos0 rose from 0.57 to
    about 0.97–0.99 during training (`logs/s-300m-k2.log`).
  - Wall time relative to K=1, with all modules run serially on one machine, so this is total compute including the
    local heads. At 10M: 1.25× (K=2), 1.54× (K=3), 2.33× (K=6). At 100M: 1.09× (K=2), 1.27× (K=4), 1.47× (K=6).
    Source lines: the `it 977` lines of the `s-*` logs. The cosine diagnostic ran in every arm (`arms/solo-scale.txt`).
- **Inference:** the extra compute shrinks as models grow, which supports more modules at larger sizes. The module
  nearest the input gets the weakest signal, so slices should not be equal: make the first module shallower, or give
  it more replicas (**guess**). The 300M K=2 run took 2.8× longer than K=1 per token before it was killed, which looks
  like memory pressure rather than a property of the method (**guess**).
- **Confidence:** medium (one seed, short runs).
- **Cheap next step:** resume `s-300m-k2` and `s-300m-k6` with lower memory use, and add per-module cosine to the
  X1 report.

---

## 2. Findings per source

### `research/README.md`
- Measured on the DUST tiny preset: the DiLoCo outer Nesterov step took loss from 4.1121 to 0.5031 on the identity
  task (lines 23-26). Error feedback recovers about 71% of the 1-bit codec's loss. Source says.
- It explicitly recommends against Krum, trimmed-mean and gossip upgrades, because of churn and near-orthogonal
  updates (lines 54-58). See Section 3 of this file.
- FedBuff buffered asynchronous aggregation: 3.3–3.8× fewer client trips. Over-issuing 2K jobs and applying the first
  K bounds staleness by construction (line 47; `communication-efficiency.md` §3.1, lines 206-218). Not in the
  synthesis.
- Warm-up rounds without compression (line 49; `communication-efficiency.md` §1.3). Not in the synthesis.

### `research/communication-efficiency.md`
- EF21-SGDM: at batch size 1, error feedback alone does not converge, and momentum has to live on the server
  (§1.2, lines 37-45).
- Instrumentation: log the distribution of pairwise cosines between updates every round (§4.1, line 274). Inference:
  this is cheap, and it shows whether distance-based filters and RDA work at all.

### `research/open-source-ecosystem.md`
- 1-bit compression cannot represent exact zero, so embedding rows that never get gradient build up phantom error
  forever. The fix is a per-block "known zero" mask (§3.2, lines 158-161). Inference: check whether this also applies
  to the SparseLoCo 2-bit plus error-feedback path with a large vocabulary.
- safetensors `mmap` doubles peak memory on unified-memory machines (Apple, APUs). Stream instead (§4.2,
  lines 192-194). This is relevant to Apple and other inference-only roles.
- Xet-style content-defined chunk dedupe for near-identical checkpoints (§4.3). `hivemind.load_state_from_peers` lets
  joiners fetch from a random peer instead of the origin (§4.1). Petals keeps retry state on the *client* (§4.4).
  The synthesis says only "download ~1 GB".
- NoLoCo (arXiv 2506.10911) is listed as unverified: "gossip instead of all-reduce" (§7, lines 267-269).

### `research/zo-es-lineage.md`
- Hiratani 2022: node perturbation is always unstable when the model cannot fit the target, showing up as weight-norm
  growth. The suggested alarm is per-layer weight-norm and update-ratio logging (§1.2, lines 36-42). Inference: this
  is a cheap health check for any role, and it complements PRISM-style drift detection.
- Ren et al. 2023 (arXiv 2210.03310) and Fournier et al. 2023 are the local-loss lineage behind SOLO. Using local-loss
  gradients as the guessed direction "drastically improves" forward gradients (§2.2-2.3, lines 79-93). Inference:
  this could give inference-only devices a better probe direction from SOLO's local heads.
- DeComFL (arXiv 2405.15861): one scalar per upload (§6.1, lines 269-275). Combined with SeedFlood (below), this fits
  the inference-only role in synthesis Section 4.7.
- Lab measurement of SZO: at equal forward passes, SZO-1 has 4–5× the gradient cosine of DUST at initialisation and
  about 1.5–2.5× from a trained checkpoint, but all cosines are at or below 0.07 (`research/lab/plans/szo.md:24-44`).
  SOMA's cosine is about 1e-3 (`plans/soma.md:67-78`). This quantifies how little gradient signal the forward-only
  roles produce.

### `archived/crowdgpt-v2/research/03-prior-art.md`
- **Orchestrator consensus, which the synthesis does not cover:**
  - Use CometBFT with no token. It halts if more than 1/3 of voting power goes offline (the "liveness cliff"). The
    recovery options are an inactivity leak, epoch-wise ejection of inactive validators, and a scripted re-genesis
    (§3, lines 164-189).
  - Mirror the record in a Tessera/tlog-tiles log and anchor it with OpenTimestamps (§8.1, line 324).
  - Inference: "Orchestrators commit" in synthesis Section 4.4 needs this design.
- **Growth activation:**
  - BIP9-style readiness signalling (90% or more) plus a hard-height fallback, with a deterministic transform that
    every orchestrator hashes and votes on (§6, lines 284-289).
  - Depth and width growth operators: G_stack (arXiv 2405.15319; 54.6% speedup at 7B), HyperCloning (2409.12903),
    and MoE orthogonal growth (2510.08008) (§6, lines 255-267).
  - Inference: "Add modules K=2→4" in the synthesis needs a depth-growth operator, and that is missing.
- **Failure lessons:** add an automatic rollback rule (Consilience). Decentralized compute is about 1000× below the
  frontier and growing about 20× per year (Epoch) (§1.3, lines 62-69).
- **SeedFlood** (arXiv 2602.18181): flooding (seed, scalar) pairs over gossip used 400 KB *in total* for 16
  OPT-1.3B replicas. Seeds must come from a beacon after commitment. Bounded-influence argument (§2.6,
  lines 141-149).
- **Sarmenta credibility-based spot-checking:** error rate falls exponentially in the number of checks passed
  (§2.4, lines 116-119).

### `archived/crowdgpt-v2/research/02-v1-baseline.md`
- **Real consumer numbers:**
  - About 450–500 tokens/s on a Radeon 890M integrated GPU with the 658M dense model (§7, line 190). Inference:
    6 × 0.66B × 475 ≈ 1.9 TFLOPS of useful work on an integrated GPU, which calibrates the synthesis's 6 TFLOPS
    guess for a discrete card.
  - 92 accounts, 18 active GPUs, top-10 accounts holding 88.7% of tokens (§7, lines 184-188).
  - A "0.1% download-to-contribution funnel" (§5, line 153).
- **Unwired assets:**
  - SimHash activation fingerprints at 258 bytes per 32 tokens. Exact-bit comparison breaks across hardware.
  - Signed `StepReceipt`s, never sent by the client.
  - Evaluator receipts with unpredictable k-of-2N spot recomputation. Good design, but HMAC means third parties
    cannot verify them.
  - `NodeRecord` roles {leader, witness, validator, scheduler, storage}.
  - Sources: §3, lines 97-104; §8, lines 219-227.
  - Inference: fingerprints with tolerance comparison could be the cheap "hash of output activations" in synthesis
    Section 4.1.
- **Record defects to avoid in v2:** the base and aggregate weights hash the same tensor; deltas and data are not
  committed; and the Merkle tree has no domain separation (§4, lines 127-134).
- **The v1 aggregate was not a descent step** (cosine ≈ 0), and a flat initialisation gave 36× less held-out
  improvement per unit of weight movement (§2, line 69). Inference: add a "cosine of the aggregate to the held-out
  gradient" health metric.

### `research/lab/` (beyond Top 5)
- `REPORT.md:54-62,113`: the result placeholders (`RESULTS_ROUND1`, `RESULTS_ROUND2` and others) are still unfilled.
- `arms/round1.txt`: the Decoupled DiLoCo (`r1-dd-*`), THROTTLE (`r1-th-*`) and SOMA (`r1-soma-*`) arms have no
  result files. `probe.py` fails on Apple's MPS backend; use `szo_probe.py` or `--device cpu` (`plans/soma.md:70`).

---

## 3. Contradictions with the synthesis

1. **Stale status.**
   - Synthesis line 163 says the Gauntlet Byzantine run is "still running". Its log stops at 2.089M tokens
     (`logs/r1-sl-byz-gauntlet.log`), there is no result JSON, and the queues were killed (`logs/solo-a.log`).
   - Synthesis line 171 says 300M K=2 is "not yet run". It actually reached 2.0M tokens (Top 5, item 5).
   - Synthesis line 178 says the staleness study is "queued". No queue process was running.
2. **MOSAIC is missing.** The synthesis says REPORT.md "is not filled in yet" (line 144), but REPORT.md already holds
   a full MOSAIC design (lines 64-109) that differs from the cell mesh on activation transport.
3. **Gossip.** The synthesis recommends gossip with 1–2 peers (Section 4.4). `research/README.md:56-57` and
   `communication-efficiency.md` §3.3 rank gossip low under churn, because its spectral-gap penalty is worst on
   sparse, changing topologies. SPICE and GASLoC were tested without churn and with at most 128 replicas.
   Unresolved.
4. **Asynchronous merging.** The synthesis adopts time-budget rounds (Decoupled DiLoCo). The lab saw paper-default
   momentum oscillate under event-driven merges (`plans/ddiloco.md:47-52`).
5. **THROTTLE as an influence cap.** The synthesis presents it as a cap (Sections 3.2 and 4.5). The lab smoke test
   says it "reduces flooding damage a lot … but it does not neutralize it" (`plans/throttle.md:58-66`).
6. **Weighting.** The synthesis bans token-weighted averaging. The lab's MOSAIC weights by tokens in `het=budget`
   mode (`mosaic.py:10-12`). The two designs need reconciling.
7. **Contributor shape.** The synthesis implicitly assumes a million roughly equal volunteers. v1 had 18 active GPUs
   and top-10 accounts holding 88.7% of tokens (`02-v1-baseline.md` lines 184-188). Per-identity caps and
   anti-collusion pairing matter more than the synthesis assumes.

---

## 4. Top 10 papers not yet downloaded

Every ID below appears in repo files. None was downloaded.

| # | arXiv | What it is | Why it matters at million scale | Repo source |
|---|---|---|---|---|
| 1 | 2503.09799 | DiLoCo scaling laws | Fitted rules for outer learning rate, batch size and loss versus M; validated at 4B and 10B. Sets the outer step without sweeps. | comm-eff §2.3 |
| 2 | 2501.18512 | Streaming DiLoCo | Fragment-by-fragment sync gives 8× lower peak bandwidth and 400× fewer bits; overlap capped at about 5 steps. Fits 20 Mbit/s links. | comm-eff §2.4 |
| 3 | 2506.10911 | NoLoCo (Gensyn) | Gossip instead of all-reduce, with no coordinator. Directly about removing the hub. | open-source §7 |
| 4 | 2502.19405 | Verde (refereed delegation) | Bisection narrows a dispute to **one operator** for recomputation. A cheaper dispute path than replaying a whole cell OPEN-1B-style. | prior-art §2.2 |
| 5 | 2403.09603 | Optimistic verifiable training (Srivastava, Arora, Boneh) | An open method for cross-GPU replication through logged rounding decisions. The RepOps method used by OPEN-1B is proprietary. | prior-art §2.2 |
| 6 | 2208.03567 | Fang et al., PoL attacks | Infinitesimal-update and blindfold attacks against tolerance-based replay; needed to design duplicate-cell checks. | prior-art §2.1 |
| 7 | 2505.21684 | Gauntlet (templar) | Assigned-versus-random proof of computation, and OpenSkill ranking to smooth noisy scores. | prior-art §2.4 |
| 8 | 2603.03592 | SENTINEL | Notes say it covers part of Agora's unchecked stage activations and gradients. Relevant to boundary trust. | `notes/2607.13332.md:106` |
| 9 | 2510.08008 | MoE orthogonal growth (with G_stack, 2405.15319) | Adds layers or experts to a converged MoE. Needed for the "K=2→4" depth growth event. | prior-art §6 |
| 10 | 1902.06156 | A Little Is Enough | Attacks that hide inside honest variance; the stress test for norm clipping and duplicate-cell tolerances. | prior-art §2.5 |

**Also worth a quick look:**
- The round-1 verification papers 2609.33915, 2611.09889 and 2606.21045. The repo records only their IDs
  (`INDEX.md:29`), so their relevance is unknown.
- FedBuff (2106.06639) and Koloskova's delay-adaptive SGD (2206.08307), for bounding staleness.
