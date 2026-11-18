DRAFT — requires human review

# G4 mining: trust/verification and forward-only (zeroth-order, ZO) papers

Scope: 2609.17380 OPEN-1B, 2609.37367 Chorus, 2606.19129 Giskard, 2611.08712 SZO-κ, 2611.16317 WIS-MeZO,
2611.04797 MobileSpec, 2609.10980 LOO-ROLL, 2602.17155 ZO-Muon, 2604.14669 ZO edge of stability, 2608.03020 LoCA.
The full text and notes of each were read against SYNTHESIS-round2.md.
Labels: **Paper** = reported; **Inference** = ours, not tested; **Guess** = assumption.

## 1. Top 5 unused ideas (ranked by impact at million-GPU scale)

**1. Forward-only cell audits run by inference-only devices.**
- *Idea:* check a contributor's committed cell update Δ with a few forward passes, not a replay. Three checks:
  - Descent: [L(θ₀+εΔ) − L(θ₀)]/ε on the assigned batch must be clearly negative.
  - Data binding: Δ must descend faster on its assigned shard than on a beacon-drawn other shard. This catches copied or substitute-data work.
  - Gradient spot check: compare a committed step gradient g with a ZO estimate of g·ψ, where the direction ψ is drawn from the beacon *after* the commit. This is a Freivalds-style randomized check.
- *Source:* a forward difference "measures the directional derivative" (2611.08712 §2.2). Activation-shaped probes give larger responses (2611.08712 Fig. 1). Many rank-1 probes run "at near-inference throughput" in one batched pass (2609.10980 §1). A phone matched a GPU server's loss to 0.002 nats and trained an 8B model while streaming layers from flash with 2.1 GB resident (2611.04797 §6.3).
- *Use:* the synthesis's checks (§4.5) need replay on 16 GB+ cards, about 1× cell compute or 5–12× for bit-exact. These checks cost a few forward passes over a slice of the shard (Inference: ≪1× cell compute). Phones and old laptops become auditors, and the pool of verifiers grows by an order of magnitude (Guess).
- *Limit:* backdoor directions do not change clean loss, so these checks catch lazy, fake or copied work, not poison (Inference from G = V·Aᵀ, 2611.08712 §3.1).
- *Novelty:* new combination. *Confidence:* medium-low.
- *Cheap test:* 10M preset, SOLO cells. Compare honest Δ against four fakes: random Δ of the same norm, Δ trained on another shard, a 1-step "lazy" Δ, and a stale Δ. Run the checks on MPS and on CPU. Pass if AUC ≥ 0.95 with ≤ 8 forward passes and the cross-device gap stays below the separation. About 2–4 agent-hours (Guess; comparable to X2).

**2. Integer or fixed-point arithmetic for everything that must agree bit-for-bit.**
- *Idea:* do aggregation sums, boundary hashes and verifier scoring in integer or fixed-point form. Their sums give the same result in any order on any hardware.
- *Source:* reproducible int8 is 1.6–1.8× faster than reproducible BF16. GEMMs (matrix multiplies) "accumulate in int32", and the query gradient uses "64-bit fixed point for bitwise reproducibility" (2609.17380 §3.2, §4.3, Table 10). Giskard's robustness rests on validated integer counts (2606.19129 §IV-A).
- *Use:* independent orchestrators get identical aggregates whatever order gossip delivers updates in. That removes TreeFold's canonical-order requirement and any "sort before summing" step. SparseLoCo 2-bit values (synthesis §3.2) are integers already.
- *Novelty:* known technique, new placement. *Confidence:* high for sums, medium for kernels.
- *Cheap test:* sum fixed-point-encoded 2-bit updates from 16 simulated replicas in random orders on CPU and MPS. Require identical bits and loss equal to the float mean. About 1 agent-hour (Guess).

**3. Contributions whose format bounds their influence.**
- *Idea:* define the update format so that a validity check also bounds the update's influence.
  - *Values:* 2-bit values or orthogonalized (msign) matrices whose singular values are all 1.
  - *Positions:* a public coordinate mask recomputed by everyone from the weights and a beacon calibration batch, so indices are never sent.
  - *Aggregation:* per-coordinate votes or quantiles.
- *Source:* Giskard G2, "each contribution only admits a fixed admissible input" (2606.19129 §IV-D). msign fixes the step's spectral norm (2602.17155 Eq. 9–10). The Wanda mask keeps >40% of gradient energy at 5% density (2611.16317 Fig. 2).
- *Use:* our arithmetic: SparseLoCo spends 12 index bits per kept 2-bit value, so indices are ~86% of its upload (~79% entropy-coded). A public mask cuts uploads ~7×, to ~0.03 bits per parameter. A poisoner can no longer choose which coordinates to concentrate on, and no clipping threshold needs tuning.
- *Novelty:* new combination. *Confidence:* low-medium. Top-k after msign breaks orthogonality, and error feedback may pile up outside the mask.
- *Cheap test:* SparseLoCo at 4–16 replicas: top-k vs a public Wanda mask refreshed every N rounds, with and without one Byzantine sender. Measure loss, bytes and divergence. About 3–5 agent-hours (Guess).

**4. Memorization screening for base-model poisoning, run on inference-only devices.**
- *Idea:* Chorus's movability test needs forward scoring only. Harvest suspicious strings from a module, then measure how much showing each string in context raises its probability. A model that has memorized a string "barely moves". Use per-verifier private canaries, majority votes, and evidence pooled rather than verdicts.
- *Source:* attacker ratios ≤ 0.024 vs honest ≥ 0.110 at one threshold c = 0.1. The movability test catches what the probe misses as the backdoor settles. Harvesting is 73% of the cost; scoring "generates no text" (2609.37367 App. A.2, E.4, §4.4).
- *Use:* the synthesis puts "Chorus-style screening" on path checkpoints (§4.5), but Stage 1a only detects refusal backdoors in instruction-tuned models. Movability is the part that may transfer to pretraining payloads.
- *Novelty:* new application. *Confidence:* low (untested in pretraining).
- *Cheap test:* plant trigger→payload text in one replica's shard at the 10M/100M presets. Measure the movability ratio of poisoned vs clean modules, and whether the Gauntlet loss score notices. About 3–4 agent-hours (Guess).

**5. LoCA "sufficient-statistics" jobs instead of DUST for inference-only devices.**
- *Idea:* a GPU node runs one calibration backward pass. Forward-only devices then compute additive statistics G (r×r) and C (d×r). The adapter update is a public, closed-form ridge solve (regularized least squares) that is "deterministic when the probe data, projection, data order, and numerical implementation are fixed".
- *Source:* 2608.03020 §3.3, §5. It was the best forward-only method in our lab, at 6.6 vs 7.4–8.2 (synthesis §3.8).
- *Use:* each job's output can be checked by recomputing it on a sample (forward only). Aggregation is a sum, which suits fixed point (idea 2). The update needs no learning-rate tuning.
- *Novelty:* new combination. *Confidence:* medium for post-training, none for pretraining.
- *Cheap test:* LoCA fine-tune at 10M. Compute G and C on MPS and CPU, measure the gap, and test whether a fabricated C is caught by recomputing 5% of positions. About 2–3 agent-hours (Guess).

## 2. Per-paper findings

**OPEN-1B (2609.17380)**
- Reproducible BF16 runs 12.2× slower than PyTorch on one node (3.3% vs 40.5% MFU, model FLOPs utilization). Int8 brings the gap to 5.2–6.8× (Table 10). The int8 run's loss drifts ≈0.2 nats from BF16 by 400B tokens (§4.3).
- Counter-based random numbers are mandatory: one seed gives "three different sequences" on CPU, NVIDIA and Apple (§2.1). Any seed-only ZO scheme, stochastic rounding or router noise must use Philox-style value = f(seed, index).
- Failure mode: learned quantizer step sizes drifted until the gradient norm passed 300. Recomputing them from the weights costs 2% vs 70% for dropping quantization (§4.2). Failure mode: a zero query gradient appeared only as slow "gradient creep" (§4.1). Per-group norm traces would catch both.
- Clean-run anomaly baselines: pre-clip gradient norm median 0.128, 99th percentile 0.383; clipping fired on 0.16% of steps (Table 9).
- A fixed held-out probe set makes the loss curve predictable (R² 0.93 vs 0.71 for training loss, §3.4.1). Inference: publish a predicted-loss band per path, so anyone can flag an off-trend checkpoint for almost no compute.
- The host-side state hash costs a constant ~6 s/step, up to 21% of wall time (App. B). The deterministic all-reduce premium is 1.13× at 2 nodes but 3.54× at 6 (Table 11). Use power-of-two group sizes.

**Chorus (2609.37367)**
- Relative outlier filters have a false-positive floor: AlignIns rejected 47.6–59.5% of honest adapters in round 1 for populations of 4–16. The paper's reason: "the value farthest from the median lies at least one standard deviation from it" (App. B.1, Table 3).
- A weight-space classifier caught the merging attacker 0/22 times; "it detects the absence of aggregation, not the backdoor" (App. B.2).
- Backdoors stay dormant ~10–12 rounds, then jump above 50% attack success (Fig. 2).
- Use a majority vote, not a mean: "a single fabricated score can move a mean but not a strict majority" (App. A.1).
- Cheap settings: 5 harvest prompts, beam width 16 (87% of beam 64's recovery at ¼ the cost), 404 bytes of extra communication. Trust-state settings were robust across wide ranges (App. A.3, C).

**Giskard (2606.19129)**
- The committee topology is fixed once from one seed and assumes a static adversary (§II-E, §III-B). Our per-round redraw is an extension, and its failure bound grows with the number of draws over a run (Inference).
- Interactive median search needs ~10 tree passes, which takes minutes to hours at a 1 s round trip (Fig. 5). Prefer one-shot rules.
- Median rules collapse under non-IID data (data not independent and identically distributed): MNIST under FoE (Fall of Empires attack) at α = 0.1 scored 0.18 vs 0.97 for CAF. Near-IID CIFAR still scored 0.57 vs 0.70 (Table III).

**SZO-κ (2611.08712)**
- At a fixed compute per step, use "a few probes over fewer examples": 49.9% at (16 probes, batch 14) vs 45.7% at (1 probe, batch 128). At an equal total forward budget, more probes hurt (App. M).
- Simultaneous all-layer probing cost ~11 points (App. N). DUST probes this way (Guess: part of our lab stall).
- SOAP lifted every estimator except weight perturbation (App. L). Our lab never paired node perturbation with SOAP.
- Per-tensor clipping rule: cap each tensor's norm at 5× the lower median of its last 100 norms (App. O.2).

**WIS-MeZO (2611.16317)**
- The Wanda mask needs 128 calibration examples and takes 0.5–1.3 s. Refreshing it (WISD) helped in 4 of 4 model–benchmark pairs. r = 0.1 was usually best (Tables 7–9).
- Energy capture was measured on AdamW gradients, not along the ZO path (§5.1). In our lab it tied MeZO when fine-tuning.

**MobileSpec (2611.04797)**
- Population z-scoring bounds any one member's weight (§3.2). The fraction of members beating the center (70.8% at the base model, 0–8% later, Fig. 5) is a free stall diagnostic.
- Rademacher noise cost ~10 points vs Gaussian, unexplained (§5.4). Check which distribution DUST uses.

**LOO-ROLL (2609.10980)**
- A mean over 8,192 rank-1 directions had 0.98 cosine with the backprop block gradient (Fig. 3). Inference: inference-only verifiers can rebuild the gradient direction of small pieces (router, bottleneck, norm gains).
- σ from 1e-3 to 4e-3 cut reward 70–100% (§6). Verifier probes face the same radius trade-off.
- Stability rule: 1 − 2η + η²(1 + κ_r/N) < 1, where η is the step size, κ_r the variance factor and N the number of directions (§6). Gains came from more updates, not better steps (Table 2).

**ZO-Muon (2602.17155)**
- Our arithmetic from Table 2: forward-only used ~2.7–3.3× more GPU-time than backprop (237 vs 88 GPU-minutes) for 94.3% vs 96.0% accuracy. It buys memory, not speed.
- Whitening a noisy full-space estimate made it worse than MeZO, and rank 256 went unstable (Fig. 5–6). Methods tuned for LLMs collapsed on vision (FZOO 1.7%) "even after extensive hyperparameter tuning" (Table 1).

**ZO edge of stability (2604.14669)**
- The stable step size is ~2(1−β)/Tr(H), where β is the momentum coefficient and Tr(H) the Hessian trace. Momentum β = 0.9 cuts it 10×; averaging n queries raises it toward 2/λ_max, where λ_max is the largest Hessian eigenvalue (Table 1, App. B).
- Inference: [f(x+µu) + f(x−µu) − 2f(x)]/µ² averages to Tr(H), so antithetic probes plus a clean pass give a free curvature sample. Inference-only devices could publish a sharpness trace per checkpoint as an early warning of spikes. Untested; FP32 is needed.

**LoCA (2608.03020)**
- A step size scaled to the residual-stream RMS transferred from 0.5B to 14B and across model families; absolute η gave recovery ≈0 on SmolLM2 (Table 3). Inference: express update-size limits and hyperparameters relative to residual RMS so they survive growth events.
- LoRA lowered cross-entropy while BoolQ accuracy fell from 0.852 to 0.364 (§4.2). A loss-only usefulness score can reward harmful updates.
- One GPU node calibrates; "a device without backward support can receive the calibrated maps" (§5). Alignment is 0.3–0.5 in-probe only, with single seeds.

## 3. Contradictions with the synthesis

1. **Replay cost (§3.5, §4.5).** "5–7×" holds only for int8 QAT (quantization-aware training) with a 0.2-nat quality loss. Bit-exact BF16 is 12.2× at one node (2609.17380 Table 10). Dispute replay of our BF16/FP8 cells will cost more than stated unless cells run int8.
2. **DUST as the inference-only engine (§1.2, §5).** LoCA beat DUST in our lab. The forward-only papers say simultaneous node perturbation is the weakest regime (2611.08712 App. N) and that momentum hurts (2604.14669). Prefer LoCA-style jobs (idea 5).
3. **Chorus screening of pretraining checkpoints (§4.5 layer 5).** Stage 1a detects refusal imitation in instruction-tuned models only. Only Stage 1b plausibly transfers, and it is untested on pretraining.
4. **Giskard committee trees (§4.4).** Giskard draws its committees once from one seed and assumes a static adversary. Per-round beacon redraws under churn are our extension, with a larger union bound.
5. **Usefulness by loss drop (§4.5 layer 4).** Poisoned updates still lower clean loss (Chorus), and lower CE can coincide with collapsed accuracy (LoCA BoolQ). Loss-drop scores need behavioural checks beside them, not just replay.
6. **Comparing updates (§4.5 layer 1).** Tolerances must be absolute and calibrated (X2). Any check that standardizes against the few peers in one round carries a ≥34% false-positive floor (Chorus App. B.1).

## 4. Promising cited work not in our folder

- **2511.16652** EGGROLL, "Evolution Strategies at the Hyperscale" (ICML 2026). From memory, unverified: it may include integer-only language-model pretraining with ES. If so, it combines with idea 2.
- **2601.17261** AGZO, activation-guided ZO for LLMs. This is the LLM-scale version of SZO shaping (cited in 2611.16317).
- **2602.03120** QES, quantized evolution strategies (cited in 2611.04797 and 2609.10980).
- **2602.15322** masking updates in adaptive optimizers (cited in 2611.16317). Relevant to public-mask uploads (idea 3).
- **2605.19969** "Your Neighbors Know" (NeurIPS 2026), the decentralized backdoor detection that Chorus builds on.
- **2602.15195** and **2602.03085**: weight-space backdoor detection and trigger reconstruction. Both are negative results in Chorus; read them before relying on either.
- **2606.05433**: same-machine determinism at ~2–8% overhead (cited in the OPEN-1B notes). This is the cheap middle tier between float replay and bit-exact replay.
- Round-1 verification papers set aside in INDEX.md (**2609.33915, 2611.09889, 2606.21045**) deserve a reread now that verification is central.
- No arXiv ID was given for: Levi et al., TMLR 2026, on the scalability of forward gradients and ES; Dalm et al. 2026, on node perturbation training multilayer networks; or Hoy et al., COLM 2026, ES vs GRPO. All three bear on why our forward-only runs stalled.
