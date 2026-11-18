DRAFT — requires human review

# B1 runners-up and takeaways: low-communication data-parallel training

This covers bucket B1 of the round-2 survey (low-communication data-parallel training over the internet).

**Downloaded and noted in round 2:** 2611.23053 SPICE, 2610.19064 Replica scaling, 2606.11081 GASLoC, 2611.17991
LACCO.

**How to read the rest.** Everything below was **found but not downloaded**.
- For each runner-up I read only the arXiv abstract and search-engine excerpts, never the full paper.
- Dates are arXiv v1 dates from arXiv's own listing (export API or abstract page), except those marked "excerpt
  date", which I saw only in a search result.
- Each claim is labelled: **Paper says** (from the abstract or an excerpt), **Inference** (ours) or **Guess**
  (unmeasured).

**Acronyms:**

| Short | Meaning |
|---|---|
| DiLoCo | Distributed Low-Communication training |
| EF | error feedback |
| DP | data parallel |
| EMA | exponential moving average |
| H | local steps between syncs |
| R | number of replicas |
| SYNTHESIS | `research/papers/SYNTHESIS-round2.md` |

## Runners-up, ranked (abstract or excerpts read)

1. **2601.22442** (2026-01-30) — *AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism*
   (Ajanthan, Ramasinghe, Avraham et al., Pluralis).
   - **Why at ~1M GPUs:** there are no sync rounds. Each replica averages a small random slice of the parameters
     (5% by default) in the background and corrects the delay with an EMA of staleness, so a slow uplink never
     blocks compute.
   - **Inference:** the replicas agree on the slice in advance, so per-coordinate indices need not be sent.
   - **Paper says:** matches fully synchronous training at up to 1B parameters. It still works with 1% slices and
     with up to 50 steps of delay, and the gap shrinks as replicas increase. Pluralis's docs say Agora/Pluralis-8B
     uses this averaging ("async SPARTA").
   - **Caveat:** mostly 2–4 replicas. Also listed by bucket B2, which covers its pipeline side.
2. **2601.02360** (2026-01-05) — *Heterogeneous Low-Bandwidth Pre-Training of LLMs* (Obeidi, Sarfi, Lidin,
   Janson, Belilovsky; Templar).
   - **Why:** a single SparseLoCo outer loop serves two kinds of replica. Large peers each hold a full replica;
     small peers band together into one pipeline replica with compressed stage boundaries. That is a recipe for
     mixing datacenter peers with 8–16 GB home cards.
   - **Paper says (excerpt Tables 1–2), 512M:**
     - With 87.5% boundary compression, loss is 2.80–2.84 against 2.73 uncompressed.
     - Making half the replicas pipelines and half full beats compressing every replica: at 99.9% compression
       loss rises 9.8% instead of 12.4%.
     - A simulation shows over 97% compute utilization at 0.1–1 Gbit/s.
   - **Caveat:** 178M–1B only. Covenant-72B already cites it, and bucket B7 also lists it.
3. **2609.36662** (2026-09-29) — *AutoLoCo: Communication Efficient Distributed LLM Training via Adaptive
   Synchronization* (He, Zhang, Li et al.).
   - **Why:** if H varies between rounds or workers (as with GASLoC's per-worker H_i), the outer optimizer needs a
     matching correction. AutoLoCo rescales outer momentum and learning rate by the accumulated inner learning
     rate.
   - **Paper says:**
     - 215M pretraining: 73 syncs instead of DiLoCo's 100 (−27%), with lower validation loss (2.7842 vs 2.8020).
     - Llama-3.1-8B fine-tuning on 16 real hosts (1.26–1.46 Gbit/s): 24.6% less payload.
   - **Caveat:** small pretraining run; modest gains.
4. **2610.22623** (2026-10-15) — *Understanding Stability of Outer Nesterov Momentum on Stochastic
   Pseudo-Gradients* (Sharma, Gomezjurado Gonzalez, Talluri et al.).
   - **Why:** gives a stability rule for the outer step, roughly outer learning rate × noise /
     ((1 − momentum) × workers) < 1. More workers or larger local batches allow more outer momentum; unstable
     settings need less. Crowd must retune exactly this knob as replica counts and card batch sizes change.
   - **Caveat:** theory on quadratic models with SGD inner steps, plus probes at checkpoints of trained models (an
     MLP, ResNet-20, and Pythia up to 1B).
5. **2512.17131** (2025-12-18) — *Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs* (Defazio,
   Mishchenko, Raman, Shi, Xiao; Meta).
   - **Why:** gets single-replica DiLoCo-style averaging at every step without DiLoCo's extra parameter copy.
     That makes it a memory-cheap inner optimizer for 8–16 GB cards.
   - **Paper says:** reaches AdamW's final loss in 8.71% / 10.13% / 9.58% fewer steps at 160M / 1B / 8B. Code is in
     facebookresearch/optimizers (gpa).
   - **Caveat:** never tested inside multi-replica training. GASLoC cites it.
6. **2607.01678** (excerpt date 2026-07-02) — *SCAPE: Accurate and Efficient LLM Training with Extreme Sparse
   Communication*.
   - **Why:** synchronizes data-parallel updates every step at 90–99% sparsity. Workers share one Top-k mask, built
     from the optimizer's first moment and refreshed one step late. That fits inside a LAN or cluster "island",
     and the shared mask keeps sparse messages easy to aggregate and check.
   - **Paper says:** Llama-500M on 32 GH200: 1.55–1.76× faster end to end, validation loss 2.30–2.31 vs 2.30 for
     AdamW.
   - **Caveat:** datacenter setting, tested with the AdamS optimizer only.
7. **2605.20866** (2026-05-20) — *LOSCAR-SGD: Local SGD with Communication-Computation Overlap and Delay-Corrected
   Sparse Model Averaging* (Maziane, Mahran, Maranjyan, Richtárik).
   - **Why:** the first theory for the exact mix Crowd needs: unequal local steps per worker, random sparse
     averaging, and computing on while messages are in flight, with a delay-corrected merge. Its time bound depends
     on the harmonic mean of worker speeds, so slow cards cost little.
   - **Caveat:** experiments are logistic regression only.
8. **2605.09126** (2026-05-09) — *Cosine-Gated Adam-Decay: Drop-In Staleness-Aware Outer Optimization for
   Decoupled DiLoCo* (Shah, Sun; FLock.io).
   - **Why:** an outer optimizer that down-weights stale pseudo-gradients and drops any older than a cutoff. That
     is a simple, publishable age rule for asynchronous rounds.
   - **Caveat:** weak evidence. Reported 1B losses are 7–10 nats (chance is about 10.8), on a 30M-token slice with
     2 workers.
9. **2607.02544** (2026-06-24 in arXiv's listing) — *Not Every Sync Is Safe: Calibrated DiLoCo Scheduling for
   Shared AI Infrastructure*.
   - **Why:** a method lesson for any adaptive "when to sync" rule: test it against random deferral with the same
     sync budget. Most clever policies only tie it.
   - **Caveat:** SmolLM2-135M on one node. The goal is meeting service-level objectives (SLOs) on shared
     infrastructure, not volunteers.
10. **2604.24463** (2026-04-27) — *Heterogeneous-Horizon Exact-Weight Local SGD* (Pasechnyuk-Vilensky, Takáč).
    - **Why:** gives principled aggregation weights for workers that ran different numbers of local steps.
    - **Caveat:** convex and linear models only. The weights also come from worker-reported statistics, which
      conflicts with "equal weights, no self-reported influence" (SYNTHESIS §4.5).
11. **2602.03839** (2026-02-03) — *Understanding and Exploiting Weight Update Sparsity for Communication-Efficient
    Distributed RL* (Miahi, Belilovsky).
    - **Why:** in a later reinforcement-learning (RL) phase, broadcasting new weights to many rollout volunteers
      becomes the bottleneck. The paper exploits sparse weight updates and sparsifies pseudo-gradient sync with EF.
    - **Caveat:** seen in excerpts only; covers RL post-training, not pretraining.

## Seen in arXiv listings, not assessed (title only; relevance guessed from the title)

- 2605.27739 (2026-05-26) *Worker Disagreement Reveals Sharp Directions in Local SGD*: disagreement between
  replicas as a free curvature or health signal.
- 2605.28585 (2026-05-27) *Outer-Momentum Restarting in High-Dimensional Two-Phase Optimization*: when to reset
  outer momentum, for example after churn or growth events.
- 2603.26954 (2026-03-27) *High dimensional theory of two-phase optimizers* (Agarwala): theory behind DiLoCo-style
  outer steps; 2610.22623 cites it.
- 2604.09970 (2026-04-11) *LoDAdaC: a unified local training-based decentralized framework with adaptive gradients
  and compressed communication*: SPICE-like; worth checking.
- 2607.29674 (2026-07-31) *Sign compression for Muon: SignMuon, MuonSign, and the Limits of Error Feedback*:
  1-bit sync for Muon-family optimizers.
- 2609.18388 (2026-09-16) *GeoMesh: Workload-Balanced and Sign-Compressed Geo-Distributed LLM Training*: overlaps
  bucket B7.
- Theory only:
  - 2601.01493 (2026-01-04) *Accelerating Decentralized Optimization via Overlapping Local Steps*.
  - 2607.14731 (2026-07-16) *What's in a Smoothness Constant? Tighter Rates for Local SGD with Bounded Second-order
    Heterogeneity*.
  - 2606.01128 (2026-05-31) *Local MixVR*.

## Pointers for other buckets, found while searching B1

- **B2:**
  - 2609.14339 (excerpt date 2026-09-13) *Communication-Efficient LLM Adaptation over Decentralized GPU Meshes*.
  - 2606.16384 (excerpt date 2026-06-15) *Mixtures of Subspaces for Bandwidth Efficient Context Parallel Training*.
  - 2610.14277 (excerpt date 2026-10-09) *Echo: Error-Compensated 4-Bit Tensor-Parallel Communication*.
- **B3:** 2606.19025 (2026-06-17) *FoMoE: Breaking the Full-Replica Barrier with a Federation of MoEs*.
- **B6:** 2607.08651 (2026-07-09) *Secure Decentralized Federated Learning via Gossip and Virtual Voting*, a
  possible start on Byzantine robustness for gossip (title only).
- **Bandwidth data:** 2605.29359 (excerpt date 2026-05-28) *Does Distributed Training Undermine Compute
  Governance?* quotes Ookla's April 2026 averages: US 310 Mbit/s down / 57 up; China 207 / 47. Seen in an excerpt
  only, unverified.
- **Outside the date window but useful:** HALoS 2506.04531 (June 2025), hierarchical asynchronous local SGD with
  regional parameter servers. Its Table 3 shows that groups of unequal size (2–6 workers per regional server)
  diverged until the groups were made equal.
- **Seen but excluded by the "already mentioned under research/" rule:** Pier 2511.17849, LoRDO 2602.04396,
  nanochat-meets-DiLoCo 2511.13761.

## Ranked takeaways (B1 view)

1. **Replicas have two ceilings, and both are low: copies averaged and tokens per inner step.**
   - **Paper says:**
     - No study in our set averages more than 128 replicas.
     - SPICE's 128 needs each replica to talk to R/2 = 64 peers: 2 model-sizes of traffic per round, every step.
       With 4 peers, loss rises 10.3% from 8 to 128 replicas (2611.23053, Table 4).
     - All four B1 studies fix the global batch and split it across replicas. That is about 2M tokens per inner
       step in SPICE, GASLoC and the 512M replica-scaling runs, 524K in the 178M runs, and 131–262K in LACCO.
   - **Inference:**
     - Crowd's regime is different. Each card's batch is set by its memory, so the global batch grows with R.
     - At about 16K tokens per card per step (a guess, from SYNTHESIS §2), 128 cards already reach the tested 2M
       tokens per step. Covenant-72B ran up to about 7.9M (20 peers × 192 sequences × 2048 tokens; our arithmetic).
     - Size each replica set by tokens per inner step (target 2–8M at most) and add capacity through paths and
       experts, not more copies.
2. **Inside a neighbourhood, average fully; gossip only between neighbourhoods.** This refines SYNTHESIS §4.4
   step 1.
   - **Paper says:** at H = 30, 1–2-peer gossip falls behind full averaging as replicas grow: 3.45 vs 3.40 at 16
     replicas and 3.62 vs 3.53 at 32 (134M); 2.72 vs 2.67 at 16 (551M) (2606.11081, Tables 1b and 3).
   - **Inference, cost for a group of 16 (our arithmetic):**
     - Each member uploads one SparseLoCo-compressed update (about 29 MB per 1B slice, about 12 s at 20 Mbit/s;
       SYNTHESIS §4.3) once, to a relay or to seeding peers.
     - Each member then fetches the other 15: about 435 MB down, roughly 11 s at 310 Mbit/s or 35 s at 100 Mbit/s.
   - **Inference, design:**
     - That recovers full-averaging quality at home-link cost.
     - Between neighbourhoods, where all-to-all is impossible, use SPICE-style gossip with partners drawn from the
       public randomness beacon.
     - Keep groups equal-sized: HALoS diverged with unequal groups.
3. **On the replica axis, SPlus > SOAP >> Muon ≈ AdamW; scale β2 with the local batch.** This refines SYNTHESIS
   §3.2 and §4.1, which treat SOAP, SPlus and Muon as interchangeable.
   - **Paper says:**
     - At 64 replicas and 178M, loss rises against each optimizer's own DP baseline by: SPlus +3.1%, SOAP +4.8%,
       Muon +8.5%, AdamW +11%.
     - With a single local step, Muon is worse than AdamW (3.417 vs 3.157) (2610.19064, Tables 1 and 3).
     - Changing only β2 recovers about 1.2% at 64 replicas (Fig. 5).
   - **Inference:**
     - Muon remains a good choice for the one-step-stale pipeline axis (2606.30634), but not for averaging many
       replicas.
     - The open cost is SOAP/SPlus memory on 16 GB cards, because they store factor matrices. A rough estimate
       (our arithmetic, unverified) is up to about 5d² fp32 values per d×d matrix without blocking.
     - Add an SPlus-vs-Muon arm at 64+ replicas with small local batches to experiment X3.
4. **Make "what to send" checkable: derive it from public randomness, or commit to it.**
   - **Paper says:**
     - SPICE and SparseLoCo choose coordinates by Top-k on a private EF buffer, which no verifier can recompute.
     - AsyncMesh averages agreed random slices and still matches synchronous training up to 1B (abstract).
     - Earlier work found random dropping worse than Top-k with EF: Streaming DiLoCo 2501.18512 (Fig. 11) and
       SparseLoCo 2508.15706.
   - **Inference:** choose one of two options, and measure the quality gap in X3 first:
     - (a) Put the EF-buffer hash in the cell record so a replay can recompute the Top-k mask.
     - (b) Use beacon-seeded random-k for the publicly verifiable channel and accept a quality cost.
5. **Stale updates: hide one round without staleness, cap age with a public rule, avoid per-replica corrections.**
   - **Paper says:**
     - LACCO hides a whole round with no staleness and beats both Delayed DiLoCo (stale updates) and Eager DiLoCo
       (local fresh correction).
     - Eager DiLoCo degrades fastest as replicas grow: 3.445 vs 3.154 at K = 16 (2611.17991, Table 6).
     - HeLoCo (2606.00271) and Cosine-Gated Adam-Decay (2605.09126) both report that very stale pseudo-gradients
       hurt, and suggest dropping them.
     - Cross-Facility (2610.03457) rejects updates built on an outdated global model.
   - **Inference:**
     - Accept a pseudo-gradient only if its base checkpoint is the current or previous round, with LACCO-style
       overlap covering the one-round lag. Drop anything older.
     - The rule is simple and can be checked against the public record.
     - AsyncMesh's EMA delay correction is the one correction worth testing at scale. So far it is shown only with
       2–4 replicas.
