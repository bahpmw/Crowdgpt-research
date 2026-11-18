DRAFT — requires human review.

# B2 runners-up and ranked takeaways (round 2 paper survey)

**Bucket B2:** training a model in chunks without holding all of it, and model/pipeline parallelism over slow or
unreliable links.

**Downloaded and fully noted:** 2607.13332 (Agora), 2604.11947 (ResBM), 2606.30634 (one-step-delay asynchronous
pipeline parallelism), and 2604.05091 (MegaTrain). Notes are in `research/papers/notes/`.

**Runners-up (below):** found but not downloaded or read in full.
- Each line comes from the abstract or search-result excerpts, so the figures are unverified.
- "abs" means the v1 date was checked on the arxiv.org/abs page; "meta" means it comes from search metadata.
- No runner-up PDF exists in `research/papers/` as of this writing, and none is in the known-ID exclusion list.

## Runners-up, ranked

| # | arXiv | v1 date | Title | Why it could matter at ~1M consumer GPUs |
|---|---|---|---|---|
| 1 | [2608.13277](https://arxiv.org/abs/2608.13277) | 2026-08-13 (abs) | Mixture of Training: Recombining Small-Scale Scaffolded Pretraining Runs into a Larger Language Model | Depth slices train as independent jobs, each inside a small frozen "aligner" model, then are recombined. No traffic crosses slices during training, so each slice could be its own swarm. But at 1.3B, parity cost more total compute (285.0 vs 268.4 EFLOPs), and quality falls with more slices (cold perplexity 19.3 at K=2 and 24.8 at K=4, vs 15.0). Workshop paper. |
| 2 | [2611.11238](https://arxiv.org/abs/2611.11238) | 2026-11-08 (abs) | FRESCO: A Novel Consistency Control for Asynchronous Pipeline Parallel Training | Small cards force deep pipelines. At depth 32 it reaches the best baseline's loss with 43% fewer iterations and 30% less wall-clock, at 8-921x less memory overhead. This covers variable delays, which a churning swarm cannot avoid. NeurIPS 2026. |
| 3 | [2601.22442](https://arxiv.org/abs/2601.22442) | 2026-01-30 (PDF header) | AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism | Agora's algorithmic basis, with convergence analysis. Asynchronous pipelines use Nesterov look-ahead, and asynchronous averaging of 5% of weights uses an EMA correction. Matches fully synchronous training at 1B (perplexity 20.10 vs 19.92 at compute-optimal). Pluralis; NeurIPS 2026 per the Pluralis site. |
| 4 | [2602.03515](https://arxiv.org/abs/2602.03515) | 2026-02-03 (meta) | Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation | Explains why Adam fails under pipeline staleness (Hessian-basis misalignment) and fixes it by rotating the basis. At 1B and 24 stages it reaches the same loss in 76.8% fewer iterations than the best asynchronous baseline. At 32 stages it slows down only 1.27x, against 4.24x. ICML. |
| 5 | [2512.15306](https://arxiv.org/abs/2512.15306) | 2025-12-17 (meta) | LLMQ: Efficient Lower-Precision Pretraining for Consumer GPUs | A per-node engine for exactly our hardware. It pretrains 7B on a 16 GB RTX 5060 Ti at about 70% MFU (FP8 plus offloading), and 32B on 4x RTX 4090 at 51% MFU. Code from IST-DASLab. Overlaps B5. |
| 6 | [2609.14339](https://arxiv.org/abs/2609.14339) | Sep 2026 (from ID; day unverified) | Communication-Efficient LLM Adaptation over Decentralized GPU … (title truncated in search) | Fast masked pipeline traffic and compressed data-parallel sync, plus a slow "anchor" path of full passes and a spectral-correction optimizer. Up to 9x throughput from pipeline compression and over 40x combined at about 200 Mbps, matching dense quality on continual pretraining. Evidence is adaptation, not pretraining from scratch. |
| 7 | [2606.16384](https://arxiv.org/abs/2606.16384) | 2026-06-15 (meta) | Mixtures of Subspaces for Bandwidth Efficient Context Parallel Training | Long context over home links. Context parallelism at 300 Mbps with over 95% compression: an 800M model at 132K context matches 100 Gbps centralized perplexity at 20x the uncompressed throughput (55K vs 2.7K tokens/s). Pluralis. |
| 8 | [2606.07881](https://arxiv.org/abs/2606.07881) | 2026-06-05 (meta) | Breaking the Bubble: Asynchronous Pipeline Parallel Training with Bounded Weight Inconsistency (PACI) | Bounds version drift with gradient accumulation, needing no weight stashing or extra copies. Matches 1F1B-flush perplexity, up to 1.69x faster to target accuracy. Code ItayElam/PACI. A cheap alternative to PipeDream-2BW. |
| 9 | [2611.02200](https://arxiv.org/abs/2611.02200) | 2026-11-01 (meta) | ZeroGraph: Compositional Blockwise Training via Materialized Boundary Supervision | Blocks train independently against a frozen teacher's cached boundary activations, which act as a versioned dataset. Blocks resume bit-identically, so chunk jobs can be scheduled like data shards. Works for conversion and distillation (Qwen2.5 1.5B and 3.1B); unanchored local objectives fail badly (GPT-2 Large cross-entropy 3.63 -> 10.48). |
| 10 | [2604.27085](https://arxiv.org/abs/2604.27085) | 2026-04-29 (meta) | Efficient Training on Multiple Consumer GPUs with RoundPipe | GPUs act as stateless workers, and stages are dispatched round-robin with CPU offload: the same "stateless worker" idea as Agora, at the node level. Near-zero pipeline bubble on 8x RTX 4090, 1.48-2.16x over baselines when fine-tuning 1.7-32B. |
| 11 | [2608.11919](https://arxiv.org/abs/2608.11919) | 2026-08-12 (meta) | LazyTrain: Limited-resource Allocation toward Zero-waste Yield Optimization in Large Language Model Training | A scheduling (MILP) layer over MegaTrain-style layer streaming, plus 8-bit optimizer states. About 1.24x sustained TFLOPS up to a 27B model; fine-tuning only. A node-engine refinement. |
| 12 | [2603.16428](https://arxiv.org/abs/2603.16428) | Mar 2026 per ID (meta showed 2026-06-30; v1 day unverified) | An Efficient Heterogeneous Co-Design for Fine-Tuning on a Single GPU (SlideFormer) | Adds a GPUDirect-Storage NVMe tier, fine-tuning 123B+ on one RTX 4090 at 1.40-6.27x the throughput of baselines. Shows SSD as a third tier for volunteers short on RAM. |
| 13 | [2608.07974](https://arxiv.org/abs/2608.07974) | 2026-08-08 (meta) | ZeroLock: Concurrent Memory-Efficient LLM Training via Modular Update Decoupling | Chunk-local objectives (frozen readout plus a consistency term) remove backward traffic between stages. Prototype includes failure recovery. Edge LoRA fine-tuning only; small gains (-26.5% memory, +4.9% throughput). |
| 14 | [2610.26178](https://arxiv.org/abs/2610.26178) | 2026-10-19 (meta) | GeoTrain: Energy-Aware Automatic Parallelism for Foundation Model Training on Heterogeneous Geo-Distributed Resources | Chooses GPUs, layer partition, and pipeline layout together for heterogeneous geo-distributed fleets: up to 61.4% faster iterations and 11.7% less energy. A planner idea for stage placement. Overlaps B7. |
| 15 | [2605.29664](https://arxiv.org/abs/2605.29664) | 2026-05-28 (meta) | AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training | Runs several concurrent pipelines with bounded mismatch: another way to bound asynchronous staleness. No numbers seen. |
| 16 | [2610.05105](https://arxiv.org/abs/2610.05105) | 2026-10-04 (meta) | CommuteProp: Decoupled Training for Communication Bound Split LLM Fine-Tuning | Overlaps the cross-block pass with in-block updates, so cycle time follows the slowest stage instead of the sum of stage latencies. Split-learning fine-tuning only. |

**Seen but outside the date window (not eligible):**
- 2509.21221, Go With The Flow (GWTF): churn-tolerant routing by min-cost flow; up to 45% faster than SWARM at 7B.
- 2505.01099: Nesterov method for asynchronous pipeline parallel optimization (Pluralis AsyncPP).

## Ranked takeaways (B2 view)

1. **For pipeline swarms, the scaling wall is latency and coordination, not bandwidth or memory.**
   - Agora (2607.13332) trained 8.6B on 500B tokens using consumer-class nodes, none holding the full model.
   - Node throughput was flat across 200-1000 Mbit/s but fell about 40% from 0-20 ms to 60-80 ms RTT (Fig. 36).
   - Coordination degraded first, at only 60 concurrent contributors: DHT latency rose 9-10x and all-reduce time up
     to 8x (§5.9).
   - For a million GPUs: minimize WAN hops per token (fewer, deeper stages; latency-aware, geo-local placement).
     Replace the flat per-stage DHT and all-reduce matchmaking with hierarchical cells, and keep every barrier off the
     critical path.
   - Forward-only boundaries would remove the backward round trip: SOLO (2609.35440) and the chunked-training
     runners-up #1, #9, #13. Their quality evidence is weak beyond 2-4 chunks.

2. **Trust is the unsolved blocker for permissionless model parallelism.**
   - Agora's operator ran the head, the tail, and every reducer; an Authorizer gated who could join.
   - Averaging weights each replica by its self-reported sample count, and stage steps fire on counts published to the
     DHT. Both are spoofable.
   - None of the four B2 papers verifies stage computation.
   - Compressed boundaries are tiny, which makes some defenses cheap in bandwidth (my design idea, untested):
     - send a random fraction of microbatches through two workers of a stage and compare outputs within a tolerance
       (replicas drift between averaging rounds);
     - commit to boundary tensors;
     - average robustly within each stage, weighting by verified rather than claimed work (cf. Soft Throttling
       2609.39357, SENTINEL 2603.03592).
   - The cost is duplicated compute; that budget needs to be measured.

3. **Standardize on Muon-family optimizers and constant-delay asynchronous updates.**
   - Evidence (2606.30634):
     - One-step delay costs Muon +0.012 loss but AdamW +0.278 at 360M.
     - A 10B MoE trained on 200B tokens matches synchronous loss with error feedback (1.906 vs 1.906).
     - Variable-delay schedules degrade by 16 stages.
   - ResBM (2604.11947) needs Muon to keep its 32-dimensional bottleneck high-rank (layer-1 effective rank 19.95 vs
     10.26 under AdamW).
   - Agora, Covenant-72B, and Decoupled DiLoCo all use AdamW, likely leaving quality on the table once asynchronous.
   - Costs: Muon is slower per step (6,795 vs 7,837 tokens/s in ResBM Table 3), and Muon-trained models are less
     compressible by fixed-subspace schemes like SSN. Interactions with SSN, ResBM, and DiLoCo-style outer steps are
     untested.

4. **Bake about 100x boundary compression into the architecture from step 0, and pick the scheme in the lab first.**
   - Results so far: ResBM at 128x matches the AdamW baseline at 2B / 26B tokens (21.77 vs 21.75); Agora's
     reparameterized SSN at 100x trained 8.6B.
   - Back-of-envelope (mine): at h = 32 in bf16, a stage uploads about 128 bytes per token. A 20 Mbit/s uplink then
     carries about 19.5k tokens/s, more than an 8-24 GB card computes through a 1B+ stage, so home links are adequate.
   - These schemes change the model and cannot be retrofitted.
   - Before 1B -> 27B, run an ablation at about 1B: uncompressed vs ResBM vs reparameterized SSN, all with Muon, at
     matched tokens, with downstream evaluations. No paper has done this.

5. **Size stages by host RAM, not VRAM, and treat chunked or decoupled training as a research track.**
   - MegaTrain (2604.05091) trained 14B on a 24 GB RTX 3090 with about 104 GB of host RAM, using about 7.4-8 GB RAM per
     billion parameters. That conflicts with its own 12 bytes-per-parameter floor, so it is unverified.
   - LLMQ (#5 above) pretrains 7B on a 16 GB card.
   - Back-of-envelope: if a volunteer with 32-64 GB of RAM holds about 4-8B parameters, a 27B model needs roughly 4-7
     stages instead of dozens of VRAM-sized stages. That compounds with takeaway 1 (fewer hops).
   - Admission should score RAM and latency, not just VRAM.
   - Open question: streaming engines need large batches to hide PCIe transfers, while pipeline workers process small,
     latency-sensitive microbatches. This needs a lab test.
