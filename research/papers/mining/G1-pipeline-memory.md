DRAFT — requires human review

# G1 mining: fit, pipeline, memory, kernels and architecture

Date: 2026-11-18. Papers: SOLO 2609.35440, ResBM 2604.11947, Async PP 2606.30634, Agora 2607.13332, MegaTrain
2604.05091, ECO 2601.22101, NVFP4 2610.24343, FlashAttention-4 (FA4) 2603.05451, Olmo Hybrid 2604.03444. I read the
full texts and the notes. Baseline: `SYNTHESIS-round2.md`. Labels: **[P]** means the paper says it; **[O]** means our
inference; **guess** means unmeasured.

## 1. Top 5 unused ideas (ranked by impact at a million GPUs)

**1. Constant-lag protocol: fix every delay to a published constant, never "use whatever arrived".**
- Source: Async PP §6, Fig. 6 [P]. With variable delays (original PipeDream), loss degrades as depth grows; with a
  fixed one-step delay it does not. At 16 stages, even the best variable-delay setup loses more than 0.03. SOLO
  App. F [P]: a stale readout copy is harmless up to ~3% drift of its norm (about 50 steps).
- Use [O]: upstream activations, the SOLO readout copy and outer-sync inputs all use version t−τ, with τ fixed and
  written to the record. Late volunteers wait or are skipped. Their arrival time never changes the math. Because every
  cell's input version is then deterministic, replay checks stay valid under churn. Readout refresh is triggered by
  drift (refresh when ‖W−W̄‖/‖W̄‖ > 2–3%), not by a fixed period.
- Novelty: new combination (async-pipeline result + verification requirement). Confidence: medium.
- Cheap test: lab arm, SOLO K=4 at 100M. Compare constant staleness τ∈{1,2,4} with random staleness of the same
  mean, using Adam vs Muon µ=0.99 ± weight-space error feedback (EF). Error feedback means feeding the rounding or
  delay error back into the next update.

**2. Stateless, unbiased compression for replica sync, which survives churn.**
- Source: NVFP4 §4.1, §4.3 [P]. Rotation + corrective rescaling (MS-EDEN, from Quartet II 2601.22813) gives unbiased
  estimates "at lower quantization-error cost" than stochastic rounding. Using a different random rotation per replica
  helped further.
- Use [O]: SparseLoCo's top-k needs a per-volunteer error-feedback residual (synthesis §3.8: without it, loss is 5.66
  instead of 5.07). That residual is lost when the volunteer leaves; with ~3-hour sessions, that is constantly. An
  unbiased rotated quantizer carries no state, and its error averages down as 1/√R over R replicas. Rotation seeds
  come from the beacon, so the result is replayable.
- Novelty: new combination. Confidence: low-medium (bits per parameter may be higher than top-k).
- Cheap test: add an X3 arm with 16–64 simulated replicas: SparseLoCo with EF buffers reset by simulated churn, vs
  Hadamard-rotated 2-bit unbiased quantization with no EF, at equal bytes.

**3. Re-do the per-card arithmetic: host RAM sets slice size, and real consumer throughput is ~5× the guess.**
- Source: MegaTrain Table 9 [P]. An RTX 3090 (24 GB, PCIe Gen3) trains 3B at 1,792 tokens/s (33 TFLOPS, 25 GB
  RAM), 7B at 768 tokens/s (35 TFLOPS, 57 GB RAM), and 14B with 104 GB RAM. GPU memory stays flat as depth grows
  (§4.4).
- Use [O]: a 16 GB card in a PC with 32–64 GB RAM can hold a ~3–7B module slice, not ~1B. That means fewer
  modules or expert groups per path, fewer boundaries, and fewer replicas for the same total parameter count. With
  10–20 TFLOPS of useful work per card (guess) instead of 6, the data wall in synthesis §2 arrives 2–3× sooner.
- Novelty: known technique, new placement (volunteer slice role). Confidence: medium; MegaTrain was tested only on
  fine-tuning at sequence length 8,192.
- Cheap test: stream one SOLO module (3B) from 64 GB RAM on a 16 GB card. Measure tokens/s against an in-VRAM 1B
  slice.

**4. A cheap verification kit built from by-products of the training method.**
- Sources [P]: SOLO Table 10: every module decodes through the shared readout. ResBM App. A.3: a 32-number code per
  token crosses each boundary. FA4 Table 2 / §3.2.4: legitimate kernels differ by about 1 BF16 ULP (unit in the last
  place), and the nondeterminism comes from atomic adds. NVFP4 §5: weaker precision recipes look better early.
- Use [O]:
  - (a) Score a module-k cell by its own exit loss through the signed readout copy, with no deeper modules needed.
  - (b) Duplicate cells compare per-token boundary codes during the job, as a forward checksum.
  - (c) Derive the duplicate tolerance from the ~1-ULP-per-op floor, measured in X2.
  - (d) Add "silent precision downgrade" to the X2 fault list, because loss-based scores will miss it.
  - (e) ECO without master weights leaves no hidden FP32 state, so checkpoint hashes cover the full state.
- Novelty: new combination. Confidence: medium.
- Cheap test: run duplicate SOLO cells on CUDA vs MPS. Record code cosine, exit-loss gap and update cosine. Then
  inject an FP8 downgrade and see which signal flags it first.

**5. Deep-narrow modules with one-block heads make K=8 modules per path affordable.**
- Source: SOLO Prop. 2, Eq. 6, Fig. 13c, Table 13 [P]. Head overhead is ((K−1)H/L + 2γ/3)/(1+γ), where K is the
  number of modules, H the head depth in blocks, L the number of layers, and γ the readout's cost relative to the
  blocks. Measured cost per boundary is 0.010 + 0.040·H. A 96-layer, 1.2B model at K=8 costs ρ=1.10 (+10%), not
  the +89% quoted in the synthesis, which is for a 24-layer model with H=2.
- Use [O]: the modules axis (synthesis §4.2) can double to 8 for deep paths, halving the replicas needed. Replicas
  are the axis with the worst scaling evidence (Agora §5.4 refused extra replicas over divergence).
- Novelty: known formula, unused. Confidence: medium on cost; quality at K=8 is untested beyond our K=6 lab run.
- Cheap test: lab, 100M with 48 thin layers vs 12 wide, K=8, H=1. Measure loss and throughput.

## 2. Per-paper findings

**SOLO 2609.35440**
- [P] Readout moves ~0.1%/step, 1%/10, 3%/50, 5%/100 steps. Cost 1.9% at a 5% lag. Tolerance "should grow" over long runs (App. F, Fig. 11).
- [P] Refresh is cheap to send (2.2 MiB/step at S=50); the cost is the forced alignment: lost throughput ≈ 0.24(p−1)/(M·S) (App. H).
- [P] A mature frozen SOLO readout beats the live one early (Table 4). The authors suggest an exponential moving average (EMA) of the readout (§5). [O] At fork or module-add events, seed new heads with the parent's mature readout.
- [P] A shared basis matters more than content: rotating the readout per module costs 13 ppl, and a shared random readout beats per-head random ones by 12 (Table 9).
- [P] The gap forms in the first ~2B tokens, then stays flat (App. C, Fig. 6). [O] SOLO health can be judged early.
- [P] Off NVLink, DDP loses 72% and SOLO 8% (Table 12). At 0.5 Gb/s SOLO also halves (Table 18).
- [P] Positive alignment with backprop holds when the head's prediction is close to the final module's (App. B, Eq. 5). [O] Use KL(exit‖terminal) on probe batches as a watchdog.
- [P] Not tested: MoE, GQA, long context. Untied embeddings (§5, App. C).

**ResBM 2604.11947**
- [P] Trained end to end; both activations and their gradients are compressed (App. A.3). See C1.
- [P] 64 KiB per 1,024-token micro-batch per boundary in BF16 = 64 B/token forward (App. A.3).
- [P] Muon is required for quality under compression, but Muon models are "less compressible" (§3.4, Fig. 1).
- [P] The 32-dim bottleneck uses an effective rank of ~14; layer 1 matters most. A footnote suggests giving early layers more dimensions (App. B.1).
- [P] Gains appear only below ~800 Mbit/s: 12.8× at 80 Mbit/s (App. A.2). Tested with only 8 blocks, batch 32×1,024 (§4.1).

**Async PP 2606.30634**
- [P] Higher momentum reduces the delay penalty across optimizers (§2.2, Fig. 4). AdamW cannot use it: β1=0.99 already hurts synchronous runs (App. A.8). Its first moment is the culprit (Table 8).
- [P] Batch size is the strongest lever: large batches push the gap above 0.1 even for robust optimizers (App. A.2).
- [P] Error feedback must act on updates, not gradients (gradient-level EF diverges, App. A.10). Best λ ≈ 1.5–2 for Muon, SOAP and AdamW (App. A.4).
- [P] Switching from synchronous to async causes spikes, worse with β2=0.999. A synchronous cooldown barely helps. Taylor correction (DC-ASGD) is useless (§3.1, A.3, A.9, A.11). [O] Keep one regime across growth events.
- [P] WPipe (half the model delayed) is about as good as synchronous training (App. B.2).
- [P] Stash + EF = two extra parameter-sized buffers (App. F.1). See C7.
- [P] Seed noise in final loss ≈ 4×10⁻⁴ (App. A.7). The 10B async run was already a Gated DeltaNet hybrid (§5.2).
- [P] Batch scaled as B ∝ D^0.58 (§5.1). [O] Publish this as a record-driven per-step token quota.

**Agora 2607.13332**
- [P] No extra tail replicas were added, because of divergence risk (§5.4). Replica variance grows linearly with time since the last averaging and sits above consensus movement, so averaging more often would help (§5.7.2, Fig. 27).
- [O] Self-tuning cadence: shard committees publish variance and movement; the averaging interval shrinks when variance > movement. The rule is deterministic and auditable.
- [P] Weight-sync phase = one full averaging cycle (400 steps). Optimizer-sync phase = 5/(1−β2) steps (§5.6). Probation also measures reliability "before it starts contributing". [O] Zero-weight probation is a cheaper first trust gate than r=2 duplicates for every newcomer.
- [P] Clipping per stage at 1/√S with no cross-stage reduction, tail ×2 (§4.3, Eq. 9). [O] Apply per SOLO module.
- [P] Trainer round trips grow linearly with stage count (§3.4.2). Unpinned backward routing costs 0.4 nats (§6.2). [O] SOLO removes both problems.
- [P] Usable rank grows slower than width: 40 at d=2048, ~51 at d=5120 (§3.4.5).
- [P] Clients who never got in gave up after a median of 8 minutes (§5.3.3). [O] Give queued volunteers immediate small jobs.

**MegaTrain 2604.05091**
- [P] Consumer numbers and flat GPU memory as depth grows (Table 9, §4.4); see Top 5 #3.
- [P] Double buffering is worth 31% throughput. Checkpoint interval 1 cuts the batch from 96 to 32 (Table 4).
- [P] The optimizer runs on the CPU (AVX-512 Adam) (§3.1). [O] Matrix optimizers (Muon/SOAP) fit this badly; see C6.
- [O] Overlap condition on PCIe Gen3: each pass needs roughly 600–1,000 tokens or more (our arithmetic). SOLO's freed activation memory supplies this.
- [O] Stateless layer templates make **single-layer spot replay** cheap for verifiers.

**ECO 2601.22101**
- [P] ECO with stochastic rounding (SR) is +0.4% loss at 800M; with round-to-nearest (RTN) +2.9%; RTN without master weights diverges (Table 1).
- [P] The noise floor is ∝ 1/(1−β²) with SR and ∝ 1/(1−β)² with RTN (§3.3). Derived only for SGD-momentum and Adam (Algs. 2–3).
- [P] Embedding and output layers stay high precision (§4.2).
- [O] Seed SR from the beacon with a counter-based random generator: unbiased and replayable.
- [O] Without master weights, the published state is the whole state (verification benefit).

**NVFP4 2610.24343**
- [P] Recipe rankings do not settle until after 150B tokens; short runs only catch catastrophic failures (§5). [O] Changes to canonical maths need long-horizon proxy runs.
- [P] All-layer FP4 diverges after 300B tokens; the last 15% must stay BF16 (§4.2).
- [P] Bias correction matters most for gradient tensors (§4.3). Separate rotation seeds per replica help (§4.1).
- [O] Agora's fleet was ~46% Blackwell-generation cards (Agora §5.3: RTX 5090, RTX PRO 6000). From memory, unverified: RTX 50-series has FP4 cores. An optional FP4 lane is worth more than the synthesis implies.

**FlashAttention-4 2603.05451**
- [P] The deterministic backward runs at up to 75% speed, with semaphore-ordered reductions (§3.2.4).
- [P] Emulated exp matches hardware within 1 BF16 ULP on 99% of inputs. Conditional rescaling uses τ=8 (§3.1.3–3.1.4).
- [P] Longest-processing-time-first (LPT) ordering gains 4–14% on any GPU (§3.3). [O] Assign the longest cells first to the most reliable volunteers (classic makespan rule). Use a fixed shortest-processing-time-first (SPT) order for deterministic aggregation.
- [O] JIT compiles in 2.5 s (Table 4). Pin kernel binaries by hash for replay-checked cells.

**Olmo Hybrid 2604.03444**
- [P] Mid-training ran two independent 100B-token branches and merged them (App. A.2). [O] Fork-train-merge as a late-stage axis with no communication during the fork.
- [P] WSD-S schedule is token-agnostic: a 5% decay yields a checkpoint at any budget (§4.1). [O] Lets an open-ended volunteer run cut decayed releases as side jobs.
- [P] Spike score = % of gradient norms ≥6σ from a rolling 128-step mean; flat for the hybrid (App. A.1, Fig. 13). [O] A deterministic health rule for committees.
- [P] Fits at ≤1B predicted 32B within 1.69% (Table 4). [O] Supports record-driven growth triggers fitted on small proxies.
- [P] 7:1 is best at ≤190M and 3:1 at ≥600M; interleaving beats middle placement (Table 5). [O] Start genesis at 7:1. Make module length a multiple of 4 so every module has attention.
- [P] Recurrent-state numerics compound; an FP32 GDN state fixes it (§5.3). RL rollout GPUs: 16 vs 48 at 4K (Table 8).

## 3. Contradictions with or corrections to the synthesis

- **C1. ResBM and SOLO do not compose as written** (synthesis §1.2, §4.3). ResBM's encoder learns from gradients
  coming back from the next stage (ResBM §3.2), which SOLO removes. [O] A possible fix, untested: put the encoder and
  decoder on the sender, train them through the sender's local head, and transmit only the code.
- **C2. Boundary bytes are 64 B/token, not ~128** (ResBM App. A.3). This is favourable: ~0.5 Mbit/s at 1,000 tok/s.
- **C3. "K=8 costs +89%" is shape-specific.** It is about +10% on deep models (SOLO Table 13, App. H).
- **C4. The 6 TFLOPS/card guess looks pessimistic.** MegaTrain sustained 30–35 TFLOPS on a 3090 (Table 9).
- **C5. "Absorb contributors with larger batches" (synthesis §2) conflicts with async delay.** The delay penalty grows
  with batch size (Async PP App. A.2). Scale out with replicas or paths instead.
- **C6. The ECO + Muon + high-momentum + delay stack is untested and may conflict.** ECO's floor rises as momentum
  approaches 1 (ECO §3.3), while Async PP wants µ≈0.99. ECO is defined only for SGD-momentum and Adam. CPU-side
  optimizers (MegaTrain, Agora) suit Adam, not Muon.
- **C7. The per-card memory budget (synthesis §4.2) omits** the weight stash + EF buffer: +2 parameter-sized states
  (Async PP App. F.1).
- **C8. Muon raises activation rank** (ResBM §3.4). [O] Its updates may compress worse under top-k (SparseLoCo). Check
  this before pairing them by default.
- **C9. The lab conclusion "crews must run near lockstep" (synthesis §3.8) may be an Adam artefact.** Muon with high
  momentum, constant delay and EF closes the gap at 10B (Async PP Table 3).
- **C10. Agora nuance.** Matchmaking was "extraordinarily stable" (§5.9.2). What degraded was DHT latency (9–10×) and
  all-reduce timeouts. "Latency limits speed" is confounded with GPU class (§5.10, Fig. 37).

## 4. Promising cited papers not in our folder (2025-11 to 2026-11)

- **AsyncMesh 2601.22442** (Agora): fully asynchronous data + pipeline parallel; the delta-rule generalization.
- **Ringmaster LMO 2605.18174** (Async PP): asynchronous Muon-type optimizer with *variable* worker delays (delay
  thresholding). This is the regime volunteers actually create.
- **LOSCAR-SGD 2605.20866** (Agora): convergence proof for delay-corrected sparse averaging with overlap.
- **Jung et al. 2602.03515** (Async PP): basis rotation to fix AdamW staleness.
- **Quartet II 2601.22813** (NVFP4): MS-EDEN unbiased estimator (Top 5 #2).
- **Four Over Six 2512.02010** (NVFP4): adaptive FP4 block scales.
- **Breaking chains with trees 2606.21497** (SOLO): model-parallel training in O(log N) time.
- **2605.04913** (SOLO): local learning for LLM post-training.
- **2607.16612** (SOLO): backprop-free trunk training via split forward gradients.
- **mHC 2512.24880**, **Universal weight subspace 2512.05117** (ResBM): safe dimension changes on the residual path,
  and shared low-rank structure across models.
- **DroPE 2512.12167**, **2603.03612** ("Why are linear RNNs more parallelizable?"), **Olmo 3 2512.13961** (Olmo
  Hybrid).
- Older than the window but central: Nesterov async PP 2505.01099; EF for Muon 2510.00643; critical batch size
  2505.23971.
