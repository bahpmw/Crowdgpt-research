DRAFT — requires human review

# B4 runners-up and takeaways: multi-token prediction and attention/kernel efficiency

This is bucket B4 of round 2. The four papers we chose have full notes in `research/papers/notes/`:

- 2610.38331 Soft-MTP
- 2610.24766 ECHO
- 2603.05451 FlashAttention-4 (FA4)
- 2604.03444 Olmo Hybrid

Nothing below was downloaded.

**How dates were checked:**
- "abs" means the v1 date was read from the "Submitted on" line of the arXiv abstract page.
- "API" means the v1 date is the `published` field from an arXiv API listing.
- "unverified" means the paper was seen only in web-search results. The ID prefix gives roughly the month, but
  IDs early in a month can carry the previous month's date. Numbers marked "snippet" come from search-result text
  only.

**Acronyms:**
- MTP: multi-token prediction. NTP: next-token prediction.
- GDN: Gated DeltaNet. MoE: mixture of experts.
- FA2/FA3/FA4: FlashAttention-2/3/4. KV: key-value (cache).

## Runners-up

### Multi-token prediction and related training objectives

| arXiv | v1 date | Title | Why it could matter for ~1M consumer GPUs |
|---|---|---|---|
| 2610.06031 | 2026-10-05 (API) | LightMTP: Lightweight Latent Multi-Token Prediction | Latent MTP with at most 1% extra parameters and no extra vocabulary projections. It gives future-aware supervision without MTP's memory cost on 8–24 GB cards. At nanochat scale (a d12 model of about 286M total parameters, plus 24-layer models) it matches NTP perplexity and helps planning (Countdown). |
| 2608.05806 | 2026-08-06 (API) | Hierarchical Latent Prediction for Language Models (HiLP) | 1B models, 100B tokens, 8×B200. It measures the real local cost of MTP-style objectives: training throughput relative to NTP is 83% for NextLat, 65% for HiLP and 49% for 4-head MTP. We saw no code link. |
| 2609.07129 | 2026-09-07 (API) | Line-Coupled Language Model | Keeps the plain next-token loss but advances several text lines per forward pass. At 881M it produces 2.94 tokens per pass at loss 2.44, against 2.39 for the baseline. That is faster serving of a crowd-trained model without a drafter. |
| 2610.28500 | 2026-10-20 (API) | BitMTP: When Multi-Token Prediction Meets Low-Bit Large Language Models | MTP on ultra-low-bit backbones, with measured RTX 3090 results (up to 2.0× mean throughput) and released code. It is the most consumer-GPU-grounded MTP paper we found, but it covers inference and serving. |
| 2608.03624 | 2026-08-04 (API) | LoopMTP: A looped transformer guided by latent multi-token prediction | Weight-shared (looped) depth with MTP-style latent targets: up to +8.1% relative accuracy, stable to 15 loops. More compute per parameter could mean fewer bytes to sync per unit of depth (our inference). |
| 2603.23911 | 2026-03-25 (API) | Self-Distillation for Multi-Token Prediction (MTP-D) | Self-distillation raises MTP-head acceptance by 7.5%, and a "looped extension" adds heads cheaply. It is a post-pretraining complement to Soft-MTP. |
| 2602.06019 | 2026-02-05 (API) | Multi-Token Prediction via Self-Distillation | Turns a finished NTP model into a standalone multi-token decoder: more than 3× faster at under 5% GSM8K drop, with no new inference code. This is the fallback if we skip MTP in pretraining. |
| 2608.00434 | 2026-08-01 (API) | AdaMTP: An Adaptive Training Paradigm for Multi-Token Prediction | Masks MTP loss across high-entropy boundaries so noisy gradients do not reach the backbone. Tested only on existing 7–12B backbones. |

**Seen only in references (not checked):**
- NanoGPT Slowrun (github.com/qlabs-eng/slowrun; not on arXiv). Soft-MTP cites it as evidence that MTP improves
  sample efficiency when data is fixed. That matters if a million-GPU crowd runs short of good data.
- DeepSeek-V4.1-Flash (2609.19969). Soft-MTP (Sec 7) says it dropped MTP in favor of a post-trained drafter,
  DSpark (2607.05147).
- NVIDIA Nemotron 3 (2512.20856, date unverified). It is the source of the latent-MoE backbone used by ECHO.

**Outside the window:** Future Summary Prediction (2510.14751), Token Order Prediction (2508.19228) and NextLat
(2511.05963, which by its ID predates 2025-11-18).

### Attention kernels

| arXiv | v1 date | Title | Why it could matter for ~1M consumer GPUs |
|---|---|---|---|
| 2601.21824 | 2026-01-29 (abs) | DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training | Puts a price on bit-reproducible replay audits. The FA3 deterministic backward pass costs up to 37.9% throughput, and DASH's schedules recover up to 1.28× on H800. ICLR 2026, code on GitHub. Hopper only. |
| 2603.02170 | 2026-03-02 (abs) | SageBwd: A Trainable Low-bit Attention | INT8 attention that quantizes 6 of 7 matmuls and matches full precision in pretraining, given QK-norm and fewer tokens per step. The SageAttention family targets RTX-class GPUs, so this is the most direct consumer-GPU attention speedup for training. Overlaps B5. |
| 2603.00040 | unverified | Attn-QAT: 4-Bit Attention With Quantization-Aware Training | Snippet: FP4 attention trained with quantization-aware training, 1.5× over SageAttention3 on RTX 5090. |
| 2611.03344 | unverified | FLAMIA: FLAsh MIcroscaling Attention for Fast and Causally Safe Training on Blackwell | Snippet: MXFP8 and mixed NVFP4/FP8 attention training paths that fix forward/backward numerical inconsistency. Consumer-Blackwell coverage unknown. |
| 2609.04105 | unverified | Hardware-Aware FP4 FlashAttention-4 | Snippet: FP4 forward and backward on top of FA4, up to 2.13× the BF16 forward on GB200. Datacenter only. |
| 2609.15810 | unverified | Value Smoothing and Softmax Casting for Low-bit Attention | Snippet: low-bit attention implemented for RTX 5090, RTX PRO 6000 and datacenter parts. Inference on video models. |
| 2604.23798 | unverified | ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers | Snippet: exact attention with an FP32 path that runs identically on A100s and edge devices. Interesting for cross-device reproducibility; vision only. |
| 2610.31479 | unverified | Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon | Snippet: asks whether KV-cache, speculative-decoding and FlashAttention-style optimizations carry over to Apple silicon, which matters if Macs volunteer. |

### Sparse, linear and hybrid attention architectures

| arXiv | v1 date | Title | Why it could matter for ~1M consumer GPUs |
|---|---|---|---|
| 2606.20936 | 2026-06-18 (abs) | Comparing Transformers and Hybrid Models at the Token Level | Compares Olmo 3 and Olmo Hybrid token by token. Hybrid gains concentrate on content words and entity/state tracking, while attention wins on copying and bracket matching. This shows which data a hybrid learns faster, which helps decide data mixes and eval probes. |
| 2606.13392 | 2026-06-11 (abs) | MiniMax Sparse Attention | Blockwise top-k sparse attention on grouped-query attention. On a 109B model it matches dense GQA while cutting per-token attention compute 28.4× at 1M context, with 14.2× faster prefill on H800. Long-context phases only. |
| 2610.42147 | 2026-10-30 (abs) | Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention | Hierarchical clustering of keys. The native variant matches full attention with 1.6× training speedup per layer at 128k. Relevant only if we pretrain at very long context. |
| 2602.11761 | 2026-02-12 (abs) | MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling | Converts a pretrained transformer into a 9B sparse+linear (1:3) hybrid by continual training at about 75% less cost than training from scratch. It reaches 1M context on one A6000D-class GPU. A route to retrofit a crowd-trained transformer later. |
| 2607.07953 | unverified | Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing | Snippet: DeltaNet runs at 1.3B and 3B plus hybrid-versus-pure comparisons. A second opinion on Olmo Hybrid's choices. |
| 2603.14360 | unverified | Non-Linear RNNs with Matrix-Valued States for Scalable Language Modeling | Snippet: hybrid M2RNN beats equivalent Gated DeltaNet hybrids. A possible successor to the Olmo Hybrid recipe. |
| 2602.02108 | unverified | Out of the Memory Barrier: A Highly Memory Efficient Training System for LLMs with Million-Token Contexts | Snippet: chunk-recurrent training with recomputation keeps memory constant in sequence length. This is B2's domain (memory), so we hand it over. |

## Takeaways, ranked

1. **Choose the backbone before tuning the objective. A 3:1 Gated DeltaNet/attention hybrid is the largest lever in B4.**
   - Olmo Hybrid reaches the same MMLU with 49% fewer tokens at 7B. Its projected token savings are 1.32× at 1B,
     1.57× at 3B and about 1.7–1.8× at 7B–30B, at matched training throughput.
   - In a crowd, every token saved also saves outer rounds and their bytes, under any B1 outer loop.
   - Inference state shrinks from 16–512 MiB to 1.05 MiB per layer, so rollouts and evals fit on 8–24 GB cards.
   - Validate at small scale first: the projected gain is smallest at 1B, and the confidence intervals are wide.
   - Risks: numerical fragility of the recurrent state, possibly no Apple Metal kernels (unverified), and
     post-training regressions.

2. **Treat kernel determinism as part of the trust design, not only a speed question.**
   - Determinism costs real throughput. FA4's deterministic backward reaches about 75% of the non-deterministic
     speed. DASH puts the FA3 penalty at up to 37.9%.
   - Honest kernels also differ in low bits (FA4's exp emulation), and recurrent layers amplify such differences
     (Olmo Hybrid).
   - Across ~1M heterogeneous GPUs, bit-exact replay between device classes is not realistic. Our proposal: the
     work spec pins kernel, version and determinism flag per device class. Same-class audits can be exact;
     cross-class audits use calibrated tolerances.
   - Open decision: if audits are tolerance-based anyway, workers could run the faster non-deterministic kernels.

3. **Adopt MTP only for downstream reasoning and free drafting. If adopted, use Soft-MTP, and consider ECHO-style
   sharing.**
   - Up to 1.2B, plain NTP keeps the best perplexity. MTP costs +26% FLOPs, −21% throughput and +11.5 GB at
     micro-batch 19.
   - Its value is downstream accuracy from about 689M up, and a lossless self-drafter (about 1.7×) for
     volunteer-served inference and RL rollouts. A poisoned head can slow decoding but cannot change outputs.
   - Soft-MTP is free. ECHO keeps added parameters near zero (less to sync) and halves average overhead, but the
     deepest sampled depth sets peak memory. The combination is untested.

4. **Do not expect datacenter kernel advances to reach volunteers. Design for kernel diversity.**
   - FA4 needs SM100-only features. Consumer cards run FA2-class, Triton or Metal kernels, and the same GPU varies
     2–3× by kernel.
   - So credit by verified tokens, not by time or nominal FLOPs, and prefer asynchronous outer loops (Decoupled
     DiLoCo) so slow peers never stall others.
   - Pretrain at short context and move long context to a later phase. By our estimate, attention scores are about
     8% of a d=2048 layer's FLOPs at 2k and 17% at 4k. Hybrids and sparse attention pay off in the later phase.
   - Portable kernel ideas to reuse: longest-processing-time-first scheduling (+4–14% on H200) and conditional
     softmax rescaling.

5. **Cheapest next experiments, cheapest first. Ideas, not yet run.**
   - (a) Soft-MTP as a one-line change in the `research/lab` harness, to see whether MTP helps at all at 10M–100M
     under our settings.
   - (b) A GDN hybrid against a transformer at 10M–100M on identical data, with DiLoCo-style sync and compression
     switched on. Olmo Hybrid's ablations show hybrid gains already at 60M–190M (Table 5).
   - (c) A cross-device numerics test: the same batch on CUDA and on Apple MPS, attention-only vs hybrid, to
     calibrate audit tolerances before choosing an architecture.
