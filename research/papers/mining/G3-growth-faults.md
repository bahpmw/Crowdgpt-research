DRAFT — requires human review

# G3 mining: growth, modularity, faults, multi-token prediction

Papers: 2603.19149 Optimal Splitting · 2604.19835 Expert Upcycling · 2602.11543 SPES · 2609.37899 SOMA ·
2603.00357 SPARe · 2604.10390 LLM-PRISM · 2610.38331 Soft-MTP · 2610.24766 ECHO. Full text and notes read;
checked against `SYNTHESIS-round2.md`. **Paper** means reported in the paper. **Inference** means our reasoning.
**Guess** means not measured. Acronyms: MTP = multi-token prediction; CPT = continued pretraining (training after
a growth step); SDC = silent data corruption; ppl = perplexity.

## 1. Top 5 unused ideas (ranked by impact at ~1M GPUs)

**1. Split the redundancy budget. Use lazy spares for churn and eager duplicates for checking, and aim the checks at module-boundary activations.**
- *Source:* SPARe §3.1, Thm 4.2, §5.2.2. PRISM Obsv. 4, Obsv. 2.
- *Applies how:* SPARe gets its ~2.8× overhead (instead of 20×) by **not computing** spare copies unless a failure forces it. A copy that never runs checks nothing, so the synthesis's "one ~2× budget does three jobs" does not hold. PRISM shows that weight-gradient faults get "diluted … by a factor of 1/N" when updates are averaged. Faults in forward outputs, by contrast, "propagate without attenuation". In a SOLO mesh the only data that crosses machines without being averaged is the boundary activations. So eager duplicates should go mainly to cells that produce activations. Replica updates that are equal-weight averaged need less checking. PRISM finds fault rates up to 0.01 are harmless and damage starts at 0.05, so sampled checks only need to catch cards that are wrong on more than ~1–5% of steps.
- *Novelty:* new combination. *Confidence:* medium. PRISM is GPT-2 scale. SPARe is simulation only.
- *Cheap test:* lab SOLO K=2 at 10M with 4 replicas. Inject PRISM-style bit-pattern faults at rates 0.01–0.1 either (a) into module-1 output activations or (b) into one replica's update. Compare downstream held-out loss and weight L2 drift. About 1–2 agent-hours.

**2. Pair checkers on a Golomb ruler with a beacon-rotated offset.**
- *Source:* SPARe App. B.1, Lemma B.2: with cyclic Golomb-ruler placement, "any two distinct shard types … share at most one host."
- *Applies how:* use the same placement rule for volunteers in duplicate cells. Any two volunteers are then co-assigned at most once per round. Each volunteer is compared against r−1 different partners. Rotating the ruler offset from the public beacon each round stops a colluding pair from co-signing repeatedly. The construction is deterministic and cheap, so any orchestrator can recompute and audit it.
- *Novelty:* new (SPARe placement + Giskard/Chorus beacon committees). *Confidence:* medium-high for the combinatorics. The anti-collusion value is untested.
- *Cheap test:* pure-Python simulation, N=1,000, r=3, colluder fraction 1–20%. Measure P(a bad cell is co-signed only by colluders) and the colluder pair-repeat rate under random pairing vs Golomb pairing. Under 1 agent-hour.

**3. Growth-event rules the synthesis is missing.**
- *Source:* Optimal Splitting §5.3, App. G.2, App. K, App. L. Upcycling Table 2, §3.3, §6. PRISM Obsv. 3.
- *Rules:*
  - **(a) Commit the horizon.** The optimal fork point "must be aware of the training horizon". A greedy token-count trigger is wrong, so publish the total tokens D_T in the record.
  - **(b) Space the doublings.** Gap closure is 35% at 10% CPT and 98% at 50% CPT, so the next doubling waits until tokens since the last one ≥ tokens before it.
  - **(c) Fork recursively.** Fork 1→4→16→64. Four domains win early, and 16/64 only catch up later.
  - **(d) Make growth reproducible.** Plain copy-paste is as good as 20 fancier tricks, so a growth event can be rebuilt from the parent hash, 10 beacon-chosen batches and beacon-seeded noise.
  - **(e) Expect a loss bump.** Loss jumps after every fork or doubling and optimizer state is reset, and PRISM finds early faults do the most damage. Re-baseline the anomaly detectors and raise the check rate for a window after each event.
- *Novelty:* (a)/(b)/(e) are new uses of reported results. (c) is a new combination. *Confidence:* medium. Repeated doubling is untested ("natural extension", §6).
- *Cheap test:* extend synthesis X4 (100M MoE). Compare doubling at τ/T=0.25 against two doublings at 0.25 and 0.4 (breaking the spacing rule), and one 1→16 fork against 1→4→16. Log the size and duration of each loss bump. About 4–6 agent-hours.

**4. Make expert ownership actually fit 16 GB, and use expert merging to repair churn.**
- *Source:* SPES §3.3 ("each node stores a full copy of the model"), Fig. 3, Table A5, App. A.4, Table 4.
- *Applies how:*
  - **Memory problem.** SPES saves only optimizer and gradient memory: 34 GB vs 54.5 GB at 2B. Every owner still holds every expert, which breaks the synthesis's "~1B per card, 64B per path" arithmetic.
  - **Fix (guess).** Store the frozen, non-owned experts in 4-bit, since they never get gradients, and keep owned experts at full precision.
  - **Churn repair.** When all owners of an expert leave, rebuild it with SPES's warm-up merge: a weighted average with its K=4 nearest experts by input-projection cosine (α=0.1).
  - **Trust.** SPES assigns experts "via direct assignment" from a single owner, so every expert group needs at least 2 owners whose updates are compared before acceptance.
- *Novelty:* new combination. *Confidence:* low-medium.
- *Cheap test:* 8-expert MoE at 50–100M with 8 simulated owners. Compare (i) non-owned experts fake-quantized to int4 against bf16 on held-out loss, and (ii) after dropping one owner, merge-repair vs freezing vs random re-init. About 3–4 agent-hours.

**5. Reconsider MTP as a data-wall lever. Use the ECHO variant in the last SOLO module only.**
- *Source:* Soft-MTP §7, Tables 3, 4 and 5. ECHO §3.
- *Applies how:* the synthesis drops MTP because it "does not improve next-token loss". Soft-MTP says MTP "improves sample efficiency" in data-constrained settings (Slowrun: "infinite compute, fixed data"). That is CrowdGPT's regime once the data wall hits. At 1.2B, Soft-MTP beats plain next-token training on WikiText ppl (17.49 vs 17.50), LAMBADA ppl (14.07 vs 14.16), zero-shot accuracy (57.31 vs 56.75) and HumanEval pass@64 (43.9 vs 39.0). It is still worse on in-distribution ppl. ECHO reuses the final layer and adds almost no parameters, at E[D]=1.5 extra unrolls, so it fits a 16 GB card.
- *Novelty:* known technique, new rationale. *Confidence:* low. The Slowrun claim is second-hand and GitHub-only.
- *Cheap test:* 100M model, fixed unique-token budget repeated for 4–8 epochs. Compare next-token-only against ECHO-1.5(3) on held-out loss at equal unique tokens. About 2–3 agent-hours.

## 2. Per-paper findings

### Optimal Splitting (2603.19149)
- Greedy splitting is suboptimal; the optimal fork point depends on the total budget (§5.3, Fig. 4).
- The optimum is flat: allocations "within 0.005 of the optimum" form a wide band (Fig. 12), so rounding forks to epoch boundaries is safe (inference).
- The optimal pretraining fraction fits a power law f_s(N)=a·N^b from 100M–1.3B runs and extrapolates to 2.7B (App. G.2). The growth rule can be fitted once in the lab.
- Domain count: 4 is best at low pretraining; 16 or 64 only win later; the authors pick 16 as the tradeoff (§6.4.2, App. L).
- Loss shows "a sharp increase for the first few thousand steps of each split model", with optimizer state reset (App. K, App. A). Inference: fork downloads need weights only.
- The router is nearest-centroid on MiniLM-L6 embeddings. A 32-token prefix gives R@1 0.88; 64 tokens give 0.97 (App. E, Table 5). Inference: any verifier can recompute a document's cluster, which blocks off-cluster (poisoned) data being slipped into a path.
- Splitting a 3.8B/0.7B-active MoE gains 2.2% ppl in all 16 clusters (App. N), so forest paths and expert groups stack.
- Routing to only 2 of 16 models costs ~1–3% (Table 2).
- 1.3B split models lose to a dense 6.7B; 2.7B split models win by only 0.6% (§6.3.2).
- 47% of compute goes to the synchronous trunk; the rest runs "on non-specialized hardware" (App. B).

### Expert Upcycling (2604.19835)
- CPT budget is "the binding constraint": efficiency 34.7% / 64% / 84% / 98% at 10 / 20 / 33 / 50% CPT (Table 2b). Very early upcycling (τ/T=0.05) reaches only 81% (Table 2a).
- 20 diversity heuristics gain ≤10⁻³ over copy-paste (App. D). Utility = squared gradient norm over 10 batches, duplicate cap 3, δ=10⁻³ (§4).
- Initial loss after growth predicts final loss: Spearman ρ=0.80/0.86 over 65 runs (§5.2.2). Candidate growth plans can be ranked by a short trial (inference).
- The datacenter step cost rises 1.9× at doubling (§3.1). Under expert ownership, per-volunteer cost stays flat (inference).
- The sunk-cost case saves 50–67% (§3.1). Inference: genesis could upcycle an open-weight MoE (licence and provenance tradeoff).
- Loss-free load balancing is a prerequisite (§3.4). Inference: its router bias can also steer tokens away from absent owners.
- A 200K vocabulary takes 411M of 1,263M parameters (Table 5b). Inference: keep SOLO's frozen output layer small.

### SPES (2602.11543)
- Full model resident on every node; memory formula 4|ψ|+|Φ|+3|Φᵢ| (§3.3).
- Upload is 2.8 GB/round at 2B and 9.8 GB at 7B (Fig. 3). Our arithmetic: 2.8 GB takes ~19 min at 20 Mbit/s, so SparseLoCo-style compression is mandatory (untested with ownership).
- H=50 is best; quality drops at H=200 and 400 (Fig. A2).
- Expert noise does not average: the shared block gets σ²/N but experts are "owner-only" (App. A.4). Replicas of expert groups are what reduce it (inference).
- Score falls from 50.6 to 49.5 going from 2 to 8 nodes (Table A5).
- After 70% of tokens, halve the per-node batch and set H=50 (App. B).
- Experiments ran on 13–17 Gbit/s links and 720 GB+ RAM hosts (§4.1). Not home-class.

### SOMA (2609.37899)
- The 1/N variance gain is specific to forward-only (zeroth-order) training and fades: −0.035 nats/byte early, 0.0003 at 4M updates (App. J). Backprop beats SOMA at every budget (Table 5). It does not support forest paths under backprop.
- The head is learned once in a 1,000-update seed run and then frozen in all experts (§4). Inference (new): a permanently frozen SOLO head would remove head-refresh sync and give every path one token space. Risk: SOMA says its frozen decoder "does not establish the feasibility of a larger token vocabulary" (App. O).
- Averaging the top-4 routed experts "captures most of the gain" (Fig. 16). Inference: averaging 2–4 paths at serving time recovers subset-routing loss and dilutes one bad path.
- Router outputs were identical on 4,096 prefixes (App. N), so a fixed router is a reproducible public function.
- Budget-or-deadline rule: s_max=min(C/u, D/τ) (Eq. 6). "A deadline can favor more shards" (§7).
- N=256 means ~700 passes per shard (§9). Forest width multiplies the data wall.

### SPARe (2603.00357)
- SPARe does worse than plain replication at r=2 under Weibull k<1 (§5.2.2). The optimum is r*≈⌊log₂N+0.833⌋, i.e. 8–9 (Thm 4.3, Table 2). That assumes a 60-minute global restart; ours is ~0, so our optimum is unknown (inference).
- Failures come early (Weibull k=0.78; Table 1). Inference: give singleton or critical cells to long-uptime volunteers and redundant cells to new joiners.
- "Making the shard placement independent of physical failure domains" (§4.1). Inference: duplicates should span regions, ISPs and time zones even if pipeline crews are regional.
- Detecting failures before aggregation reaches the overhead lower bound (Thm 4.2). Lease heartbeats help here.
- Checkpoint period T*c=Ts+√(Ts²+2Ts(Tf+Tr)) (Eq. 1) can set the committee checkpoint cadence.

### LLM-PRISM (2604.10390)
- Forward and input-gradient faults have ~60% crash rates; weight-gradient faults are "largely self-correcting" (RQ4, Obsv. 4).
- FP8 runs show no silent degradation or gradual drift ("binary"); BF16 shows silent ppl loss up to ~400% (RQ1, RQ4). Inference (guess): FP8 boundary data limits how large a lie can be.
- Larger models degrade silently more often: 13.8% vs 11.0% (RQ2).
- Fault-free repeats vary by 1% ppl (RQ2). Loss-based scores cannot see faults below that.
- Timing-marginal defects appear "at specific voltage or frequency operating points" (§II-A). Inference: run a known-answer GEMM canary at the card's working clocks at the start of each lease (overclocked consumer cards).
- Runs that recover still lose downstream accuracy (Fig. 7).
- Scale: GPT-2 124M/355M, one faulty GPU at a time.

### Soft-MTP (2610.38331)
- Cost: +26% FLOPs, −21% throughput, +11.5 GB memory at micro-batch 19 (Table 5). The gain over NTP grows with size: ~0.1 points at ≤332M, ~0.5 at ≥689M (§6.2).
- MTP must sit on the output-head stage (§7). In SOLO that means the last module, with no extra traffic.
- The one-hot residual has 4.1–4.6× lower SNR yet carries ~80% of the trunk gradient (§6.1). Inference (speculative): SOLO early-module heads could distil toward the last module through the shared frozen head using the KL gradient-recovery trick. Cost: back-traffic of last-module hidden states.
- Readout effective rank drops 496→131 under standard MTP and stays at 437 under Soft-MTP (Fig. 6). Guess: usable as a cheap checkpoint health scalar.
- DeepSeek-V4.1-Flash dropped MTP for a post-trained drafter (§7).

### ECHO (2610.24766)
- Run-to-run SD is 0.08 on reasoning/code and 0.38 on QA/multiple choice (Fig. 4c). Behavioural screening should use reasoning/code probes.
- "D is a deterministic function of the seed and the step index" (App. A.1). Every stochastic choice in a cell should derive from (beacon, cell ID, step) so it can be replayed.
- ECHO-2(7) trails after pretraining but wins after SFT (§5.2). Rank growth candidates by loss, not benchmarks.
- Gains are concentrated: +1.59 on reasoning/code vs +0.27 on QA (Fig. 1a).

## 3. Contradictions with the synthesis

1. **§1.4 / §4.5, "one ~2× redundancy budget does three jobs":** SPARe's low overhead comes from skipping spare computation, which leaves nothing to compare (SPARe §3.1).
2. **§4.2, r=2:** SPARe at r=2 is worse than plain replication; its tuned r is 8–9 (SPARe §5.2.2, Thm 4.3).
3. **§4.2, "~1B per card, 64B per path":** under SPES as written, every owner holds all experts (SPES §3.3).
4. **§4.6, "double experts when contributors double":** this ignores the ≥50%-CPT spacing (Upcycling Table 2b). Repeated doubling is untested (§6).
5. **§4.6, forest-split trigger on tokens reached:** greedy; it must be horizon-aware (Splitting §5.3).
6. **§3.3, SOMA "supports paths":** the variance gain is zeroth-order-only and fades late (SOMA App. J, O).
7. **§3.6 / §5, MTP "not a pretraining lever":** at 1.2B Soft-MTP beats next-token training off-distribution and downstream, and is claimed to help when data is scarce (Soft-MTP Tables 3–4, §7).
8. **§4.4, "crews by latency region":** duplicates of a cell should span failure domains (SPARe §4.1).
9. **§4.5, emphasis on comparing update vectors:** PRISM ranks unaveraged forward activations as the larger risk (Obsv. 4).

## 4. Promising cited papers not in our folder (2025-11 to 2026-11)

| arXiv | Why |
|---|---|
| 2602.00277 | FT-HSDP: fault-tolerant training on 100,000 GPUs (cited by SPARe); real-scale fault data |
| 2601.16956 | DataStates-LLM: composable checkpointing; useful for shard-committee checkpoints |
| 2602.02472 | SPARKLING: width-progressive MoE growth; a second growth axis |
| 2511.04981 | Deep progressive training: depth growth theory behind Upcycling's bound |
| 2602.07145 | Convex dominance: loss/learning-rate scaling law; could set the schedule after growth |
| 2511.05963 | Next-Latent Prediction: KL to hidden states with a frozen head (fits SOLO) |
| 2602.06019 | MTP via self-distillation |
| 2607.05147 / 2602.06036 / 2609.09338 | DSpark / DFlash / Osprey drafters for volunteer inference |
| 2512.20856 | Nemotron 3 latent MoE (ECHO's backbone) |
| (GitHub) qlabs-eng/slowrun | "Infinite compute, fixed data": the data-wall regime |

Just outside the window but relevant: 2510.06548 (bootstrapped-pretraining scaling law), 2510.02375 (hierarchical
memories), 2408.15664 (loss-free load balancing).
