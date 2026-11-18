DRAFT — requires human review

# B5 runners-up and takeaways: low precision and optimizers

This is round-2 bucket B5: FP8/FP4 training, low-bit optimizer states, Muon at scale and distributed, and
memory-efficient optimizers. The three chosen papers have full notes in `research/papers/notes/`:

- 2610.24343: What Matters for NVFP4 Training? (Mosaic NVFP4)
- 2607.20548: SOAP, Muon, and Beyond (Muon/SOAP at scale)
- 2601.22101: ECO, quantized training without full-precision master weights

Nothing below was downloaded.

**How dates were checked:**
- "abs": the v1 date comes from the "Submitted on" line of the arXiv abstract page.
- "API": the v1 date is the `published` field of an arXiv API listing.
- "html": the v1 date comes from the arXiv HTML header.
- "snippet": the date comes from search-result metadata only.

**How claims were checked:**
- "abstract": the claim was read in the abstract or the HTML.
- "title only": only the title was seen, so the "why" is a guess to verify.
- "Inference": our own reasoning, not the paper's.

**Acronyms:**
- FP4/FP8: 4-/8-bit floating point. NVFP4: NVIDIA's FP4 block format. MXFP4/MXFP8: OCP microscaling formats.
- BF16: bfloat16. SR: stochastic rounding. EF: error feedback.
- NS: Newton–Schulz (Muon's orthogonalization). MoE: mixture of experts.
- QAT: quantization-aware training. ULP: unit in the last place.

## Runners-up

### FP4/FP8 pretraining recipes

| arXiv | v1 date | Title | Why it could matter for ~1M consumer GPUs |
|---|---|---|---|
| 2607.04422 | 2026-07-05 (html) | Full-Stack FP4: Stable LLM Pretraining with Quantized Projections, Optimizers, and Attention | The only FP4 paper measured on a consumer card. It runs NVFP4 linear layers with ~5-bit AdamW states, BF16 weights and no FP32 master, plus FP4 Muon NS and attention. At 3B / 64B tokens, loss is 2.267 (BF16) vs 2.286 (FP4). On one RTX 5090 it measures 37.9–42.5% lower AdamW peak memory and 2.50–2.83× faster Muon iterations (abstract). The loss gap was revised from 1.47% (v1) to 0.84% (v2), and the horizon is short |
| 2601.22813 | 2026-01, exact day unverified (v2 2026-06-01) | Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation | MS-EDEN is unbiased with more than 2× lower error than SR. Fully-FP4 training at 1.9B / 38B tokens is +1.44% validation bits-per-byte vs BF16 (html table). It ships RTX 5090 and B200 kernels, up to 4.2× over BF16, with code (abstract). This is a ready-made fast path for Blackwell volunteers |
| 2512.02010 | 2025-12-01 (API) | Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling | Adaptive per-block scale choice. It is already an ingredient of Mosaic NVFP4, and its pretraining runs used Nemotron 3 Nano (abstract) |
| 2610.32229 | 2026-10-23 (API) | Training an MoE LLM in MXFP4: How We Closed the Gap to BF16 | Title only. MXFP4 is the cross-vendor OCP format. A parity recipe for MoE would bear directly on the ~27B MoE stage |
| 2605.09825 | 2026-05-11 (API) | Pretraining large language models with MXFP4 on Native FP4 Hardware | Title only. Gives real-hardware MXFP4 pretraining numbers. Which hardware was used is not yet checked |
| 2603.02731 | 2026-03-03 (API) | Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs | Title only. FP4 on GPUs without FP4 tensor cores implies FP4 for memory/traffic rather than compute, which is the case for Crowd's non-Blackwell majority |
| 2611.12832 | 2026-11-09 (API) | What the Loss Curve Misses: FP4 Pretraining Slows the Acquisition of Rare Facts | Title only. If FP4 matches loss but learns rare facts more slowly, accepting a precision recipe would need knowledge probes, not loss parity alone |
| 2602.02047 | 2026-02-02 (API) | Dissecting Outlier Dynamics in LLM NVFP4 Pretraining | Title only. Growing outliers are a likely cause of late FP4 divergence (cf. 2610.24343 Fig. 2) |
| 2606.20381 | 2026-06-18 (API) | Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe | Title only. Another FP4 bias fix and a candidate recipe ingredient |
| 2609.02846 | 2026-09-02 (API) | UE5M3 FP4 Block Scaling for Stable Language Model Pretraining | Title only. An alternative scale format, relevant if non-NVIDIA accelerators join |

Also seen, title only:
- 2603.10444 The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training (2026-03-11, API)
- 2601.19026 Is Finer Better? The Limits of Microscaling Formats in Large Language Models (2026-01-26, API)
- 2607.24953 Stable FP4 Training via Transposition-Invariant Block Quantization (2026-07-27, API)
- 2610.00053 Format-Aware Fusion for Fast FP4 Pretraining (2026-09-04, API)
- 2608.01847 FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity Scaling (2026-08-03, API)

### Muon family: low-bit, compressed, distributed

| arXiv | v1 date | Title | Why it could matter for ~1M consumer GPUs |
|---|---|---|---|
| 2605.11396 | 2026-05-12 (abs) | MuonQ: Enhancing Low-Bit Muon Quantization via Directional Fidelity Optimization | 4-bit Muon states cut optimizer-state memory by up to 7.3× and recover most of full-precision Muon's loss and accuracy (abstract; COLM 2026; code). Muon has a single buffer, so per-parameter optimizer state could drop to about 0.5 byte |
| 2608.11612 | 2026-08-12 (abs) | Dion3: Full-Stack Orthogonal Updates | Orthogonalizes only a selected fraction of momentum rows, using Gram NS and fast kernels. It matches or beats Muon's loss with up to 6× less optimizer-step time (abstract). This lowers Muon's NS cost on small cards. Inference: row selection with EF is a structured-sparse update that is cheap to index on the wire |
| 2512.16928 | 2025-12, day unverified | Dion2: A Simple Method to Shrink Matrix in Muon | Dion3's predecessor: orthogonalize the top rows by norm, then decay them (EF). A row fraction of 0.25–0.5 is near full Muon (Microsoft Research talk transcript) |
| 2605.16341 | 2026-05-07 (abs) | Orth-Dion: Eliminating Geometric Mismatch in Distributed Low-Rank Spectral Optimization | A rank-r Muon-style update at Dion's communication cost that closes the convergence gap to Muon in LM pretraining (abstract). Low-rank factors are a compact payload for 20–100 Mbit/s uplinks |
| 2603.03597 | 2026-03-04 (abs) | NuMuon: Nuclear-Norm-Constrained Muon for Compressible LLM Training | Muon-trained weights turn out low-rank, and a nuclear-norm constraint makes them more compressible at billion scale (abstract). The authors appear to be the Pluralis decentralized-training group (unverified). This is a possible route to compressible updates and smaller weight downloads |
| 2605.16311 | 2026-05-04 (abs) | SignMuon: Communication-Efficient Distributed Muon Optimization | 1-bit majority-vote Muon, 32× smaller than FP32. Tested only small (CIFAR, nanoGPT, up to 16 GPUs). Inference: majority vote is also a classic Byzantine-robust aggregator |
| 2607.29674 | 2026-07-31 (abs) | Sign compression for Muon: SignMuon, MuonSign, and the Limits of Error Feedback | Sign-after-Muon can provably diverge, and EF applied to Muon's output can fail, yet that heuristic was the best compressor in experiments; EF applied to the gradient works (abstract). Placement of compression around Muon matters |
| 2608.20818 | 2026-08-21 (snippet) | Scaling Muon for Diffusion Transformers | Periodic Row-wise Muon runs full NS only every K steps. On 1.3B–15B diffusion transformers it cuts optimizer time 46.9–54.3% and logical communication 66.7% (abstract). Not tested on LMs |
| 2606.27153 | 2026-06-25 (abs) | DMuon: Efficient Distributed Muon Training with Near-Adam Overhead | Owner-based distributed NS gives 1.48–3.01× faster end-to-end steps (abstract). Datacenter only. The owner idea fits if models are ever sharded across volunteers |
| 2608.25990 | 2026-08-26 (snippet) | Spectral Allocation: Why Muon Outperforms Adam, and How to Improve Muon | SAMuon needs 13.3–24.0% fewer tokens than Muon at 124M–1B with no extra persistent state (abstract). Free token efficiency, if it holds at scale |
| 2611.03633 | 2026-11-02 (html) | Pretraining Data Scale Reverses the Muon–AdamW Fine-Tuning Gap | Caution: past a large pretraining budget, AdamW-pretrained models fine-tune better than Muon-pretrained ones. Larger models cross over at smaller token-to-parameter ratios. Only 30M–600M models were tested (abstract) |
| 2609.06073 | 2026-09-05 (html) | FedSubMuon: Communication-Efficient Federated LLM Fine-Tuning via Structured Subspace Muon | Seed-generated subspace bases mean clients upload only compact coefficient matrices (fine-tuning at 1B–8B). Inference: seed-based subspaces are a reusable bandwidth pattern for pretraining (untested) |
| 2604.10689 | 2026-04-12 (abs) | Communication-Efficient Gluon in Federated Learning | Convergence theory for compressed Muon-family updates with variance reduction. Experiments are small |
| 2605.09126 | 2026-05-09 (abs) | Cosine-Gated Adam-Decay: Drop-In Staleness-Aware Outer Optimization for Decoupled DiLoCo | An age-aware outer optimizer for stale pseudo-gradients, tested at 25M/1B/7B. The published Nesterov recipe was the least stable (abstract). Overlaps B1, and stale rounds are the normal case for volunteers |

### Memory-efficient optimizers and master-weight-free training

| arXiv | v1 date | Title | Why it could matter for ~1M consumer GPUs |
|---|---|---|---|
| 2607.10611 | 2026-07-12 (API) | M+Adam: Low-Precision Training via Additive–Multiplicative Optimization | A second master-weight-free method alongside ECO: training with BF16, FP8 or FP4 stored weights at 60M–1B and 1–8× Chinchilla (abstract; ICML 2026). The two should be compared before choosing |
| 2603.16731 | 2026-03-17 (html) | Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets | Low-bit EMA states stall, and scheduled resets recover the loss (abstract). This applies to any 8- or 4-bit momentum and to ECO-style error injection into a quantized buffer |
| 2610.22794 | 2026-10-16 (abs) | Ori: Drift-Aware Streaming k-PCA for Very Low-Rank LLM Training | A GaLore-style optimizer that tracks the drifting gradient subspace. It closes up to 69.3% of the gap to full rank on 1B LLaMA pretraining (abstract). Very low-rank optimizer state would suit 8 GB cards |
| 2602.24283 | 2026-02-27 (abs) | Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation (LoRA-Pre) | Low-rank momentum for both Adam and Muon (ICLR 2026 oral). At 1B, full-rank Muon still wins: perplexity 13.41 vs 13.53 (LoRA-Pre Adam) and 13.92 (LoRA-Pre Muon), Table 1 per the HTML |
| 2602.01233 | 2026-02-01 (html) | Lotus: Efficient LLM Training by Randomized Low-Rank Gradient Projection with Adaptive Subspace Switching | 30% less training time and 40% less gradient-plus-optimizer memory than GaLore-class methods, up to 1B (abstract) |
| 2611.08924 | 2026-11-06 (abs) | ResQAdam: Residual-Preserving Adaptive Optimization for Memory-Efficient Quantization Aware Training | One BF16 momentum, Adam in a top-r subspace, and the sign of the residual: 71–74% less optimizer-state memory with under 2% throughput loss (abstract). Tested on QAT, not plain pretraining |
| 2604.15416 | 2026-04-16 (API) | StoSignSGD: Unbiased Structural Stochasticity Fixes SignSGD for Training Large Language Models | Title only. Sign methods need no second moment and send about 1 bit per parameter. Scale not yet checked |

### Cautions and verification leads

| arXiv | v1 date | Title | Why it could matter for ~1M consumer GPUs |
|---|---|---|---|
| 2607.09800 | 2026-07-09 (html) | The Silent Freeze: Predicting When Low-Precision Training Stops Learning | Without master weights, updates smaller than half a ULP freeze the weights. A 124M GPT-2 on the FP8 grid froze at initialization; SR removes the freeze, and the freeze is predictable in advance (abstract). Usable as a pre-flight check for any FP8 canonical model |
| 2610.31859 | 2026-10-22 (html) | Full-Weight VLA Fine-Tuning Without an fp32 Master Copy Silently Erases Most Updates | Field evidence: training in plain BF16 without a master copy erased at least 79% of updates in the authors' runs (abstract). Volunteers "saving memory" this way would silently waste work |
| 2611.00486 | 2026-10-31 (API) | Integer-Exact Microscaling: Running MXFP4 Language Models on INT8 Matrix Units | Title only; an inference paper. Inference: integer arithmetic is exact, so integer-exact low-precision matmuls could make results bit-reproducible across GPU generations. RTX 3090s have INT8 tensor cores. A possible answer to the cross-GPU audit problem |
| 2611.15671 | 2026-11-11 (API) | The Noise Floor of a Quantization Comparison: Engine, restart, batch and kernel drift, measured against 8-bit and 4-bit effects in two decision models | Title only. Measures kernel and engine drift against quantization effects, which is useful for calibrating audit tolerances |

### Seen in passing, belongs to other buckets

- 2611.04323 FP4Quant: 4-bit communication compression for model-parallel training (2026-11-03, API, title only; B2)
- 2610.14277 Echo: error-compensated 4-bit tensor-parallel communication (2026-10-09, API, title only; B2)
- 2601.02360 Heterogeneous Low-Bandwidth Pre-Training of LLMs (2026-01-05, abs; B2/B7; already flagged in
  G2-comms-sync.md)

### Out of window, context only

- 2509.23106: 8-bit Muon matches full-precision Muon up to 2.7B.
- 2509.25149: NVIDIA's NVFP4 run reaches 12B parameters over 10T tokens.
- 2506.08027: MXFP8 is within about 0.5% perplexity of BF16 at 8B over 15T tokens (snippet).
- 2505.23725: MuLoCo, Muon as DiLoCo's inner optimizer, already cited in `research/`.
- Two ICLR 2026 papers whose arXiv IDs were not checked: 4-bit-Muon-GRASP (1.1B) and EF21-Muon (error
  feedback for Muon).

## Ranked takeaways (our synthesis)

1. **Storage precision is the lever for memory and bandwidth. Compute precision only buys speed, and only on
   some cards.** Make the canonical model's storage format low-precision for everyone, and let each device
   class choose its compute precision.
   - Paper says: dropping the FP32 master gave 12 → 9 bytes per parameter with Adam, near-lossless with SR up
     to 2.1B (2601.22101). Muon keeps one state buffer instead of two (2607.20548). 4-bit Muon states cut
     state memory up to 7.3× (2605.11396). FP4 matmuls alone do not shrink persistent memory (2610.24343).
   - Our arithmetic: FP8 weights + 8-bit Muon momentum + per-layer gradient release is about 2–4 bytes per
     parameter. That is roughly a 5–8B replica on 24 GB and 3–5B on 16 GB, before activations.
   - Inference: Crowd could stay full-replica data-parallel through 4B, and maybe 9B on 24–32 GB cards. That
     avoids pipeline stages over the internet, and whole-model pseudo-gradients are the easiest thing to
     score.
   - Untested: ECO combined with Muon (Muon is nonlinear in momentum) and ECO with quantized momentum
     (staleness, 2603.16731).
   - First experiment: at 300M–1B, compare BF16+FP32-master AdamW, BF16 Muon, and FP8 storage via ECO with SR
     plus 8-bit Muon momentum. Measure loss gap and bytes per parameter.
2. **No optimizer lets one synchronous step span a million GPUs. Plan for hierarchy plus MoE, with Muon on
   the dense parts.**
   - Paper says: Muon/SOAP hold quality to about 100M-token batches while AdamW degrades at 50M. Expert
     weights see only k/N of each batch (2607.20548).
   - Our arithmetic: a million GPUs at one 2k-token sequence each is about 2B tokens per step, roughly 20×
     beyond anything tested.
   - So: groups run Muon inner steps (MuLoCo-style) and merge by DiLoCo/SparseLoCo outer steps, and a
     fine-grained MoE spreads fleet growth over expert parameters.
   - Caveat: Muon-pretrained models may fine-tune worse after very long pretraining (2611.03633), so check
     downstream before committing.
3. **Every B5 technique adds hidden randomness or hidden state, so redesign the audit before adopting them.**
   - SR, Hadamard sign vectors and MS-EDEN rotations are random. ECO and SparseLoCo keep residuals in
     optimizer state. BF16 NS and FP4 tensor-core results differ by GPU class.
   - The current audit only works for reproducible estimates (DEEPDIVE.md).
   - Needed:
     - Public counter-based seeds keyed by round, worker, step and tensor.
     - A per-round hash commitment of optimizer and EF state, or a state reset each round.
     - Audits replayed on the same device class, or tolerance-based. Mosaic's run-to-run spread of 0.03–0.06%
       loss and ECO's noise-floor bound give a starting scale.
   - Lead to check: integer-exact microscaling (2611.00486) for bit-reproducible low-precision math.
   - First experiment: replay one worker step on RTX 3090 / 4090 / 5090 / Apple silicon, with BF16 compute
     and FP8 storage. Record bit agreement and cosine similarity.
4. **Make FP4 compute an optional Blackwell fast path, never the network default, and never validate it with
   short runs.**
   - Paper says: Mosaic NVFP4 is within 1.2% / 0.81% relative loss at 8B / 30B-A3B over 1T tokens, but it
     needs a BF16 tail. All-FP4 diverged after 300B tokens, and recipe rankings flip until well past 150B
     tokens (2610.24343).
   - Only RTX 50-series cards get the speedup (Quartet II kernels, up to 4.2×). Rare-fact learning may suffer
     even when loss matches (2611.12832, title only).
   - Inference: use BF16 compute as the common baseline, with FP8 storage. Allow FP4 lanes only after a long
     pilot, with a BF16 canary replica and class-matched auditors.
5. **Speculative: one residual buffer for all compression error.**
   - Weight quantization (ECO), pseudo-gradient sparsification (SparseLoCo's model-sized EF buffer, which
     Covenant-72B offloads), and Muon compression (Dion2/3 row selection, EF21-Muon) all carry a residual
     forward.
   - Inference: folding these into one per-volunteer buffer, or into momentum as ECO does, could save one
     model-sized buffer per card. The same idea applies to keeping the global model in FP8 with the rounding
     error folded into outer momentum.
   - Risk: SparseLoCo reported that EF plus Nesterov outer momentum hurt badly at high sparsity (loss 2.70 →
     3.39, 2508.15706 Table 6, snippet). Error buffers and momentum can interfere, so this needs a small lab
     test before any design commitment.
