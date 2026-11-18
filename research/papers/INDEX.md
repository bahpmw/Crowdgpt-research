# Paper index (round 1 survey)

Local only (git-ignored). Each paper: `<id>.pdf` (full paper from arxiv.org/pdf/<id>),
`<id>.txt` (full text extracted with pypdf), `notes/<id>.md` (method + MVP recipe).

Selection: 2026 papers only, biased to the newest (6 of 12 from Sep-Nov 2026), split
between ZO-estimator complements that attack DUST's variance/cost, forward-only or
local-learning replacements, and backprop-based low-communication / trustless
replacements that set the bar DUST has to beat for a volunteer crowd.

| # | arXiv | Date | Title | Authors | Role | Why it matters for Crowd |
|---|-------|------|-------|---------|------|--------------------------|
| 1 | [2611.08712](https://arxiv.org/abs/2611.08712) | 2026-11-05 | An Educated Guess: Deriving Statistically Aligned Gradient Estimates for Zeroth-Order Optimisation | Huebotter, Thill, van Gerven, Ahmad | Complement | Better-aligned ZO estimator family (SZO-κ): same forwards, higher cosine - attacks DUST's core weakness (variance) |
| 2 | [2611.16317](https://arxiv.org/abs/2611.16317) | 2026-11-12 | Wanda-Guided Sparse Perturbations for Efficient Zeroth-Order Optimization (WIS-MeZO) | Wen, Song, Wang, Luo, Zeng, Li et al. | Complement | Perturb only salient weights: fewer effective dimensions -> lower variance per draw |
| 3 | [2611.04797](https://arxiv.org/abs/2611.04797) | 2026-11-03 | Training Specialist LLMs on Your Phone: Forward-Only Optimization via Block-Spectral Coordinates (MobileSpec) | Tan, Zhao, Liu, Su, Zhang, Zhang | Complement / replacement | Forward-only training on weak devices; perturb in spectral coordinates (tiny search space) |
| 4 | [2609.10980](https://arxiv.org/abs/2609.10980) | 2026-09-10 | EGGROLL, Unrolled: Understanding and Improving Low-Rank Evolution Strategies at Scale | Kaya, Hashemi | Complement | Low-rank ES at scale: cheap batched perturbations, seed-only communication |
| 5 | [2609.37899](https://arxiv.org/abs/2609.37899) | 2026-09-29 | Scaling Zero-Order Pretraining through Model Sharding (SOMA) | Chaubard, Kochenderfer, Ré | Replacement | ZO pretraining that shards the model into independently trained experts - no gradient exchange |
| 6 | [2602.17155](https://arxiv.org/abs/2602.17155) | 2026-02-19 | Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization | Lang, Wang, Zhang, Hong, Zhang, Yin et al. | Complement | Subspace ZO + Muon-style orthogonalised updates: conditioning fix for noisy ZO directions |
| 7 | [2604.14669](https://arxiv.org/abs/2604.14669) | 2026-04-16 | Zeroth-Order Optimization at the Edge of Stability | Song, Zhang, Li, He, Muehlebach, Oh | Complement | Stability limit for ZO step sizes - explains and prevents the divergence seen in our DUST runs |
| 8 | [2608.03020](https://arxiv.org/abs/2608.03020) | 2026-08-04 | LoCA: Forward-Only LLM Tuning after One-Shot Calibration with Local Credit Assignment | Xia, Liu, Zhang, Wang, Shen, Zhu | Complement | Forward-only updates with block-local credit from a one-shot calibration: removes global-loss variance |
| 9 | [2609.35440](https://arxiv.org/abs/2609.35440) | 2026-09-28 | SOLO: Pretraining Billion-Parameter Language Models with Shared-Output Local Learning | Yin, Wang, Pan, Li | Replacement | Local learning with a shared output head: block-parallel pretraining, no end-to-end backprop |
| 10 | [2603.08163](https://arxiv.org/abs/2603.08163) | 2026-03-09 | Covenant-72B: Pre-Training a 72B LLM with Trustless Peers Over-the-Internet | Lidin, Sarfi, Miahi, Anthony, Chauhan, Pappas et al. | Replacement | Permissionless internet pretraining at 72B: SparseLoCo compression + Gauntlet peer scoring |
| 11 | [2604.21428](https://arxiv.org/abs/2604.21428) | 2026-04-23 | Decoupled DiLoCo for Resilient Distributed Pre-training | Douillard, Rush, Donchev, Charles, Fallen, Dubey et al. | Replacement | Async learners + quorum aggregation: stragglers and dropouts, the normal state of a volunteer fleet |
| 12 | [2609.39357](https://arxiv.org/abs/2609.39357) | 2026-09-30 | Robustifying Asynchronous SGD via Soft Throttling | Otsuka, Meyer, Takezawa, Yamada, Koloskova | Complement | Byzantine-robust async aggregation: untrusted peers without synchrony |

Considered and not selected (overlap or not MVP-able on one Mac): HeLoCo 2606.00271
(overlaps #11), SparseLoCo 2508.15706 (inside #10), Greedy Local Learning 2610.04867
(overlaps #9), Infer-ZO 2605.28760 (needs an inference engine), SENTINEL 2603.03592,
verification papers 2609.33915 / 2611.09889 / 2606.21045, Factored Gossip DiLoCo
2606.22768, model-merging DiLoCo 2607.03011, Subspace Networks 2506.01260, MAPL 2606.05484.

---

# Round 2 (2026-11-18): scaling toward millions of GPUs

24 more papers, all first posted between 2025-11-18 and 2026-11-18. Full text was downloaded and read for each one.
Files follow the same pattern: `<id>.pdf`, `<id>.txt` (extracted with pypdf), `notes/<id>.md` (sections:
Citation, Summary, Method, Key reported results, Relevance to Crowd, Caveats).
Synthesis: [SYNTHESIS-round2.md](SYNTHESIS-round2.md) and the deeper second pass
[SYNTHESIS-round2-addendum.md](SYNTHESIS-round2-addendum.md), with raw findings in [mining/](mining/)
(DRAFT — requires human review).

Roles: **Replacement** (could replace a core piece of the training method), **Complement** (adds to it),
**Building block** (a part we would reuse), **Systems** (scale and fault-tolerance engineering), **Trust** (verification and poisoning).

| # | arXiv | Date | Title | Authors | Role | Why it matters for Crowd |
|---|---|---|---|---|---|---|
| 13 | [2607.13332](https://arxiv.org/abs/2607.13332) | Jul 2026 | Agora: Collective and Permissionless Internet-Scale Pretraining of LLMs | Avraham, Shevchenko, Dolatabadi, Pajak et al. (Pluralis) | Replacement | 8.6B model trained on 330 consumer-class GPUs over the internet, each holding one pipeline stage; ~100x boundary compression; its coordination failed first, at ~60 concurrent nodes |
| 14 | [2604.11947](https://arxiv.org/abs/2604.11947) | 13 Apr 2026 | ResBM: Residual Bottleneck Models for Low-Bandwidth Pipeline Parallelism | Aboudib, Lopez Portillo, Brady, Cruz (Macrocosmos) | Building block | Small learned encoder/decoder at each stage boundary gives 100-128x compression with an ordinary optimizer; ~128 bytes per token per boundary |
| 15 | [2606.30634](https://arxiv.org/abs/2606.30634) | 29 Jun 2026 | One-Step Gradient Delay is Not a Barrier for Large-Scale Asynchronous Pipeline Parallel LLM Pretraining | Zmushko, Petrov, Abdullaev, Khrushchev, Horváth | Complement | Muon plus error feedback matches synchronous loss with a one-step-stale pipeline at 10B MoE / 200B tokens; AdamW does not |
| 16 | [2604.05091](https://arxiv.org/abs/2604.05091) | 6 Apr 2026 | MegaTrain: Full Precision Training of 100B+ Parameter LLMs on a Single GPU | Yuan, Sun, Sun, Ye | Building block | Weights and optimizer live in host RAM and stream through the GPU one layer at a time; 14B on a 24 GB card, at the cost of large host RAM |
| 17 | [2602.11543](https://arxiv.org/abs/2602.11543) | 12 Feb 2026 | Pretraining an LLM using Distributed GPUs: A Memory-Efficient Decentralized Paradigm (SPES) | Zhang, Xiao, Wu, Zhang, Zhang | Replacement | Each node owns a few MoE experts plus the shared trunk; uploads only what it trained; 2B-9B MoE on 4-16 nodes |
| 18 | [2604.19835](https://arxiv.org/abs/2604.19835) | 21 Apr 2026 | Expert Upcycling: Shifting the Compute-Efficient Frontier of MoE | Dwivedi, Huang, Gupta, Jayarao, Varshney, Yin (Amazon) | Building block | Doubling experts at fixed top-K mid-run matches a from-scratch 64-expert model with 24-32% less compute: a growth operator for epoch pivots |
| 19 | [2603.19149](https://arxiv.org/abs/2603.19149) | 19 Mar 2026 | Optimal Splitting of Language Models from Mixtures to Specialized Domains | Seto, Ablin, Filippova, Ye, Bethune, Katharopoulos, Grangier (Apple) | Replacement | Train a shared trunk, then fork into K specialists that never talk again; a fitted scaling law says when to fork |
| 20 | [2611.23053](https://arxiv.org/abs/2611.23053) | 17 Nov 2026 | SPICE: Sparse Parameter Gossip for LLM Pretraining | Cagnasso, Kopyshevskiy, Oyallon, Belilovsky | Replacement | Hub-free sparse gossip holds quality from 8 to 128 replicas (+1.3% loss), but only if peers per round grow with replicas |
| 21 | [2606.11081](https://arxiv.org/abs/2606.11081) | 9 Jun 2026 | Unifying Local Communications and Local Updates for LLM Pretraining (GASLoC) | Cagnasso, Belilovsky, Oyallon | Complement | DiLoCo with 1-2 random peers instead of an all-reduce; each worker picks its own number of local steps, so slow links do fewer |
| 22 | [2611.17991](https://arxiv.org/abs/2611.17991) | 13 Nov 2026 | LACCO: Local Accumulation with Computation-Communication Overlap | Kopyshevskiy, Thérien, Pinto, Oyallon | Complement | Hides one full round of communication behind the next round's compute without staleness; "use my own update while waiting" tricks get worse with more replicas |
| 23 | [2610.19064](https://arxiv.org/abs/2610.19064) | 13 Oct 2026 | How Inner Optimizers Shape Replica Scaling | Belilovsky, Horoi, Oyallon, Thérien | Complement | Loss penalty from adding replicas is mostly the inner optimizer; SOAP/SPlus cut it to +2-5% at 64-96 replicas vs +11-12% for AdamW; β2 rule for small local batches |
| 24 | [2610.03457](https://arxiv.org/abs/2610.03457) | 2 Oct 2026 | Cross-Facility LLM Pre-training on HPC: Elastic Aggregation, Data Leasing, Queue-Aware Placement | Palanciyan, van Osch, van der Wal, Kok, Kotevska | Systems | Data leases with heartbeats and time-to-live: no sample trained twice or lost through crashes and late joins; honest about the quality loss |
| 25 | [2603.00357](https://arxiv.org/abs/2603.00357) | Mar 2026 | SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant Pretraining with 100k+ GPUs | Lee, Chen, He, Lu, Underwood, Nicolae, Cappello, Di, Zhang | Systems | At 100k+ GPUs failures outpace restarts; pre-placed redundant work units mask ~426 failures for ~2.8x compute (simulation only) |
| 26 | [2604.10390](https://arxiv.org/abs/2604.10390) | Apr 2026 | LLM-PRISM: Silent Data Corruption from Permanent GPU Faults in LLM Training | Tyagi, Zhu, Hukerikar, Saxena, Huang, Shirvani, Tung | Trust | Faulty GPUs silently corrupt training; one failure mode looks normal on loss while weights drift; loss-only checks are not enough |
| 27 | [2609.17380](https://arxiv.org/abs/2609.17380) | 15 Sep 2026 | OPEN-1B: A Fully Auditable Training Run | Donaghy, Wilcox, Ersoy et al. (Gensyn) | Trust | Any step of a 1.6B / 400B-token run replays bit-for-bit on NVIDIA, CPU or Apple Silicon; costs ~5-7x throughput |
| 28 | [2609.37367](https://arxiv.org/abs/2609.37367) | 29 Sep 2026 | Backdoor Mitigation in Decentralized LLM Fine-Tuning (Chorus) | Biswas, Garcia Bourrée, Guerraoui, Jacovella, Kermarrec, de Vos, Vujasinovic, Peechara (EPFL) | Trust | Several receivers judge each update by behavior and vote; cuts backdoor success from ~50-60% to 0-2% (fine-tuning only) |
| 29 | [2606.19129](https://arxiv.org/abs/2606.19129) | 17 Jun 2026 | Giskard: Byzantine Robust and Private Aggregation for Large-Scale Decentralized Learning | Touat, Sabater, Ben Mokhtar, Maouche | Trust | Only robust-aggregation protocol costed at a million parties; private version too costly for model-sized updates, but its beacon-drawn committee trees carry over |
| 30 | [2601.22101](https://arxiv.org/abs/2601.22101) | 29 Jan 2026 | ECO: Quantized Training without Full-Precision Master Weights | Nikdan, Zandieh, Alistarh, Mirrokni | Complement | Drops the FP32 weight copy by folding rounding error into momentum; cuts memory per parameter on small cards |
| 31 | [2610.24343](https://arxiv.org/abs/2610.24343) | Oct 2026 | What Matters for NVFP4 Training? | Agrusa, Panferov, Wei, Wyss et al. (NVIDIA, ISTA) | Complement | 4-bit matrix multiplies within ~1% of BF16 at 8B dense and 30B MoE over 1T tokens; Blackwell-only, compute not memory |
| 32 | [2607.20548](https://arxiv.org/abs/2607.20548) | Jul 2026 | SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales | Khona, Vavre, Wang et al. (NVIDIA) | Complement | Muon/SOAP beat AdamW at 8B-72B over 1-3T tokens and tolerate 2-4x larger batches: more contributors per step |
| 33 | [2604.03444](https://arxiv.org/abs/2604.03444) | Apr 2026 | Olmo Hybrid: From Theory to Practice and Back | Merrill, Li, Romero et al. (Ai2) | Complement | 3:1 Gated DeltaNet / attention hybrid reaches Olmo 3 7B loss with 35% fewer tokens; projected ~1.3x at 1B rising to ~1.7-1.9x at 7-70B |
| 34 | [2603.05451](https://arxiv.org/abs/2603.05451) | Mar 2026 | FlashAttention-4 | Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao | Building block | Fastest exact attention on datacenter Blackwell; main path does not run on consumer cards; deterministic backward at up to 75% speed |
| 35 | [2610.38331](https://arxiv.org/abs/2610.38331) | 28 Oct 2026 | Multi-Token Prediction Done Right (Soft-MTP) | Teoh (MIT) | Complement | Standard MTP gradient is mostly noise that raises next-token loss; Soft-MTP removes it; still no perplexity win up to 1.2B, +26% FLOPs |
| 36 | [2610.24766](https://arxiv.org/abs/2610.24766) | Oct 2026 | ECHO: Efficient Causal Self-Recursion Beyond Auxiliary MTP | Zhou, Shang, Wang et al. (Baidu) | Complement | Multi-token prediction by reusing the model's last layer recursively; small downstream gains on a 10B MoE, no loss gain |
