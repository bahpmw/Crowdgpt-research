DRAFT — requires human review

# B7 runners-up and takeaways: extreme-scale and heterogeneous systems

**Scope.** B7 covers failures at 100k+ GPUs, fault tolerance, silent data corruption (SDC), geo-distributed and
mixed-vendor training, and reports from real decentralized runs.

**Downloaded with notes in `research/papers/notes/`:**
- 2610.03457: Cross-Facility LLM pretraining with DARL data leasing.
- 2604.10390: LLM-PRISM, SDC from permanent GPU faults.
- 2603.00357: SPARe, fault tolerance at 100k+ GPUs.

**How to read the list below.** Nothing below was downloaded.
- Dates are arXiv v1 submission dates, checked on the abstract page. Dates marked "(API)" come from the arXiv API's
  `published` field instead.
- One-liners come from abstracts, except Guard and GeoMesh, whose HTML full text was skimmed.
- Numbers are as stated by the authors and have not been checked against full text. Numbers marked "snippet" come
  from search-result excerpts only.

## Runners-up, found but not downloaded

### Tier A: would add the most to a design for ~1M consumer GPUs

| arXiv | v1 date | Title | Why it could matter for ~1M consumer GPUs |
|---|---|---|---|
| 2602.00277 | 2026-01-30 | Training LLMs with Fault Tolerant HSDP on 100,000 GPUs (Meta) | **The only real failure data at around 100k GPUs (O(100K)) we found in the window.** A data-parallel replica is the unit of failure, and recovering replicas catch up without blocking the others. Effective training time rises from 44% to 80%, and recovery stalls drop from 10 to 3 min. That is close to how volunteers drop out and rejoin. The failure table (SDC about 5.5% of interruptions; 2.3 interruptions per 1,000 servers per day) is snippet only. **Skipped only by the grep rule**: it is cited in 2604.21428.txt, 2603.00357.txt and mining/G2. It should be included. |
| 2605.11215 | 2026-05-11 | ReCoVer: Resilient LLM Pre-Training System via Fault-Tolerant Collective and Versatile Workload | Keeps the number of microbatches per step constant by moving surviving GPUs' work quotas around. The training trajectory held even though 256 GPUs were lost over a 512-GPU run, and effective throughput was 2.23× that of checkpoint-and-restart. For Crowd this is the rule "the global batch is fixed no matter who is online". |
| 2609.18388 | 2026-09-16 | GeoMesh: Workload-Balanced and Sign-Compressed Geo-Distributed LLM Training | Sets each GPU's batch size and number of inner steps by its speed, and sends 1-bit sign pseudo-gradients plus magnitude and token count (about 32× less traffic). Hardware was an H100, A100, L40S and RTX 6000 Ada, with an emulated Azure wide-area network (WAN) of 580–940 Mbit/s. Time-to-target fell 70.2%, and straggler and WAN idle fell 8.0× and 5.6×. It reports async baselines 11.7% worse in perplexity at equal tokens, which is evidence for synchronous rounds with speed-based quotas. |
| 2605.17879 | 2026-05-18 | Guard: Scalable Straggler Detection and Node Health Management for Large-Scale Training (Amazon, MLSys 2026) | "Grey nodes" pass health checks but run slow. Guard compares each node's signals to its peers online, then quarantines suspects and re-tests them with 1–2 h sweeps. Results: mean time to failure (MTTF) went from 6.6 to 16.7 h, step-time variance from 20% to 1%, and step time from 17 to 10 s. False positives were 12.4% and false negatives 7.8%, and three strikes in a week ends a node. This is a template for volunteer probation and peer-relative speed scoring. |
| 2608.30769 | 2026-08-31 | TrainSDC: Characterizing and Mitigating Silent Data Corruption in Large Language Model Training | Shows where SDC hurts transformer training. Faults on the attention Q/K path cause lasting deviation, while backward-pass damage depends on gradient exponents. Its cheap client-side guards cost 1.65–6.76% overhead on Llama 3.2-1B and Qwen3-0.6B. Volunteers could run them to protect themselves. |
| 2604.00726 | 2026-04-01 | Exploring Silent Data Corruption as a Reliability Challenge in LLM Training | A lightweight detector of harmful parameter updates, plus recomputing the last step when it fires (LLaMA 60M–1.3B). This maps onto an orchestrator-side filter for corrupted submissions and complements LLM-PRISM. |
| 2601.02360 | 2026-01-05 | Heterogeneous Low-Bandwidth Pre-Training of LLMs (Templar) | Weak participants jointly form one SparseLoCo replica through compressed pipeline parallelism, while strong ones hold full replicas. Compressing only some replicas beats compressing all of them (178M–1B). This is how 8–12 GB cards could join a 9B–27B run. Excluded by the grep rule (cited in 2603.08163.txt and notes/2607.13332.md), and it overlaps B2. |

### Tier B: useful but narrower, or datacenter-specific

| arXiv | v1 date | Title | Why it could matter for ~1M consumer GPUs |
|---|---|---|---|
| 2610.00687 | 2026-09-30 | Leto: Fast In-Place Recovery for LLM Training on Surviving Hardware | Keeps state warm on the same hardware. Recovery is 3.6–6.5× faster than the best checkpointing, with +13.7 points of productive time on 6 and 72 A100s, and over 95% productive time simulated at 131,072 GPUs. The idea fits a volunteer who blips (sleep, Wi-Fi) and resumes without re-downloading. |
| 2607.01646 | 2026-07-02 (API) | PHOENIX: Resilient LLM Training with Hot-Swapping via Zero-Overhead Checkpoint | In-memory checkpoints with no overhead, plus hot-swapped spare nodes; recovery takes under 40 s at up to 512 A100s and 65B parameters. Supports keeping spare volunteers on standby (as Covenant does) and having peers hold state copies. |
| 2605.09370 | 2026-05-10 | From Detection to Recovery: Operational Analysis on LLM Pre-training with 504 GPUs (Lablup) | Rare public operations data: 63 B200 nodes, 224 sessions, 751 metrics. No single metric predicts every failure type, and checkpoint reloads reach only 21.5% of peak read bandwidth. Snippet only: mean time between failures (MTBF) 56.2 h, and 10 of 10 failures detected with 0.84 false positives per day. Useful baseline rates to extrapolate from. |
| 2606.20374 | 2026-06-18 | ARGUS: Production-Scale Tracing and Performance Diagnosis for over 10,000-GPU Clusters | Always-on tracing at under 2% overhead; kernel events compressed about 3,700× (10 MB to 2.7 KB per rank per step); six months in production. Gives a telemetry budget per participant per step for finding stragglers. |
| 2607.15115 | 2026-07-16 | Don't Predict, Prioritize: Rethinking GPU Reliability Assessment (HeaRank) | Double-bit errors and "GPU lost" events are too random to predict in time. Ranking nodes by relative risk instead catches 64% of future failures in the top 5% of nodes (vs 21% today), with AUC 0.83. Suggests volunteer reputation should be a ranking, not a prediction. |
| 2610.07688 | 2026-10-06 | FailBench: Evaluating Fault Tolerance Across Distributed Training Architectures | Tests 142 combinations on 8×V100. No fault-tolerance mechanism wins everywhere. Gossip throughput rose 17.9% after a worker was lost, yet loss progress did not improve. A caution against throughput-only claims for decentralized designs. |
| 2512.20953 | 2025-12-24 | Diving into 3D Parallelism with Heterogeneous Spot Instance GPUs: Design and Implications (AutoHet) | Uneven 3D parallelism across mixed GPUs. When a spot instance is preempted, recovery pulls state from local nodes first and downloads only the missing checkpoint pieces. Points to peer-first state recovery for rejoining volunteers. |
| 2602.18007 | 2026-02-20 | Joint Training on AMD and NVIDIA GPUs | Mixed-vendor training at 98% of all-NVIDIA throughput (LLaMA-8B, Qwen2-7B). Evidence that mixed-vendor fleets work, though only over datacenter interconnects. |
| 2601.22585 | 2026-01-30 (API) | HetCCL: Efficient LLM Training on Heterogeneous Vendor GPUs | A collective-communication library that unifies vendor backends over RDMA (direct remote memory access between machines). Mixed-vendor collectives, but RDMA does not apply to home links. |
| 2605.31000 | 2026-05-29 (API) | HetCCL: Enabling Collective Communication For Mixed-Vendor Heterogeneous Clusters | Same theme; title only, not read. |
| 2602.22760 | 2026-02-26 | Distributed LLM Pretraining During Renewable Curtailment Windows: A Feasibility Study | A real three-cluster prototype (561M parameters, Flower) that switches between local-only and federated sync as sites come and go; emissions fell to 5–12% of single-site. Treats intermittent participation as a schedulable resource. Overlaps 2610.03457. |
| 2610.26178 | 2026-10-19 | GeoTrain: Energy-Aware Automatic Parallelism for Foundation Model Training on Heterogeneous Geo-Distributed Resources | A memetic (evolutionary) search picks which GPUs to use, how to split layers and how to configure the pipeline, under compute, memory, power and network limits. Its point that you should not always use every GPU applies to mixed pools. Results not read. |
| 2602.11456 | 2026-02-12 | AuroraRL: Fast, Fault-Tolerant, and Cost-Efficient Reinforcement Learning over Decentralized Network | About 1% of weights change per RL step. That enables bit-exact sparse delta syncs, scheduling aware of throughput and bandwidth, and lease-based fault tolerance over commodity WAN (Qwen3 4B–14B). Relevant if Crowd adds RL post-training. Snippet only: +35.5% and +26.4% from balancing. |
| 2608.11034 | 2026-08-11 (API) | SCOUT: Symmetric Consensus Outlier Detection for Failure Localization in LLM Pre-Training | Peer-consensus outlier detection to locate failing ranks may carry over to scoring volunteers against each other. Title only, not read. |
| 2605.04213 | 2026-05-05 (API) | The Anatomy of Silent Data Corruption: GPU Error Pattern Study and Modeling Guidance | Gate-level stuck-at fault injection on a production GPU, over 3M simulator hours (snippet). A source of error patterns for realistic SDC simulation of volunteer GPUs. |
| 2605.02125 | 2026 (date not checked) | FedQueue: Queue-aware federated learning for cross-facility HPC training (cited by 2610.03457) | Buffers late arrivals and down-weights stale updates inside aggregation. An alternative to rejecting stale work outright. |

### Tier C: low relevance to home GPUs (titles or abstracts only)

These are about datacenter or multi-datacenter infrastructure, checkpoint storage tiers, or very small scale:

| arXiv | v1 date (API) | Title or topic |
|---|---|---|
| 2606.12963 | 2026-06-11 | ScaleAcross |
| 2605.19169 | 2026-05-18 | Fiber-latency modelling |
| 2605.25375 | 2026-05-25 | Bandwidth-aware geo pipeline scheduling |
| 2608.23840 | 2026-08-24 | ShardMeter |
| 2606.10415 | 2026-06-09 | RATrain |
| 2605.07569 | 2026-05-08 | HexiSeq |
| 2605.17821 | 2026-05-18 | TierCheck |
| 2601.16956 | 2026-01-23 | DataStates-LLM |
| 2604.00785 | 2026-04-01 | MoE pretraining on the Aurora supercomputer |
| 2606.24722 | 2026-06-23 (abstract page) | BlockTrain: byte-level WikiText only, tiny scale |

### Already in research/papers (downloaded by other agents)

- **2607.13332, Agora / Pluralis-8B.** A real 40-day run with 330 nodes on mostly consumer GPUs; the closest analogue to Crowd.
- **2609.17380, OPEN-1B (Gensyn).** Training made bitwise reproducible across different hardware.

## Ranked takeaways (mine)

1. **Nothing may wait for everyone.**
   - The evidence: at ~1M consumer GPUs, membership changes constantly. Our illustrative arithmetic: 1M volunteers with
     ~3-hour sessions is about 90 departures per second. Datacenters already hit this wall. SPARe projects a failure
     every ~5 min at 600k GPUs, and checkpoint-only training barely progressed in its simulation. FT-HSDP reports only
     44% effective time for synchronous training at O(100K) GPUs.
   - What to do: every aggregation should close on coverage, quorum or deadline, never on full membership. These are
     the rules of SPARe, Decoupled DiLoCo 2604.21428 and Covenant 2603.08163 respectively. The unit of failure should
     be one small work unit, never a global restart.

2. **Each orchestrator's core primitive should be a leased work ledger.**
   - The evidence: DARL (2610.03457) kept exactly-once data coverage through four site departures. It uses TTL leases,
     work stealing, grants proportional to speed, and checkpoint-before-commit.
   - What to do: a block is committed when the orchestrator accepts the submission. A lease is about one inner phase,
     sized well below the median volunteer session. A deterministic global block permutation is split across
     independent orchestrators, so there is no single coordinator. Following ReCoVer and SPARe, the global batch per
     step stays fixed by moving quotas between survivors.

3. **Assume a steady stream of silently wrong contributors, separate from cheaters.**
   - The evidence: LLM-PRISM (2604.10390) shows a defective GPU can leave the loss curve normal while the weights
     drift. Silent damage reaches about 400% in BF16, and FP16 is worst. Our rough linear extrapolation of Meta's
     Llama 3 figure is about 7 SDC-type events per day at 1M devices; consumer cards are likely worse (unverified).
   - What to do: NaN and norm checks on every submission. Bounded-influence aggregation. Random replay audits that
     compare the update vector, not just the loss. Per-device reputation, since a hardware fault repeats every round;
     ranking risk works better than predicting failures (HeaRank). No FP16 on clients. Heavier audits early in
     training, which is the most fragile phase.

4. **Redundancy is one dial that buys both availability and verification.**
   - The evidence: in SPARe, pre-placing r copies of each work unit with a Golomb-ruler layout masks about 426
     failures for about 2.8× compute (N=600, r=20); full replication would cost 20×.
   - Our inference: the 2× floor comes from coarse work units, so many small units should push the cost toward
     N/(N−k). Duplicate results can be cross-checked for SDC or cheating. In a volunteer network a "wipe-out" only
     costs a re-lease, so r = 2–3 plus re-leasing probably suffices. SPARe's closed-form formulas can size r once
     churn is measured.

5. **Mixed hardware is cheap to absorb; the quality gap is not free.**
   - Easy wins: speed-proportional batches cut idle time from 75% to 8% (2610.03457), and GeoMesh reports 70% faster
     time-to-target. CUDA and ROCm sites merged fine through bf16 weights, although bitwise replay across vendors needs
     OPEN-1B-style determinism.
   - Real costs: at 0.6B, cross-site DiLoCo lost to centralised training at matched steps (perplexity 34.7 vs 28.2).
     Dense bf16 weights for a 1B model take ~160–800 s on 20–100 Mbit/s uplinks (our arithmetic). Compression is
     therefore mandatory, and the quality gap has to be measured, not assumed away.
