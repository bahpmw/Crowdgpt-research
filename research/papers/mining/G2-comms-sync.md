DRAFT — requires human review

# G2 mining: communication, sync, optimizers, data leases

Papers: 2603.08163 Covenant-72B/SparseLoCo, 2604.21428 Decoupled DiLoCo, 2606.11081 GASLoC, 2611.23053 SPICE,
2611.17991 LACCO, 2610.19064 Replica scaling, 2609.39357 THROTTLE, 2607.20548 SOAP/Muon at scale, 2610.03457 DARL.
Baseline: `SYNTHESIS-round2.md`. **Paper says** = cited section/table. **Inference** = ours. **Guess** = unmeasured.
Acronyms: EF = error feedback; DP = data-parallel; RDA = Radial-Directional Averaging; LR = learning rate;
TTL = time-to-live.

## 1. Top 5 unused ideas (ranked by impact at million-GPU scale)

**1. Equal-token cells, duplicates counted once, "exactly-r" leases.**
- *Source:* DARL §3.2 ("a window trained twice … tilting the merged model toward whichever site duplicated it");
  DARL Fig. 2a (per-site batch sizing cut idle 75%→8%); 2610.19064 §4.2 (uniform averaging is optimal when replica
  covariances match); 2604.21428 §3.2 (token weighting needed when learners differ in speed).
- *Apply:* every cell has the same token count; fast cards take more cells, not bigger ones. Equal tokens → matched
  noise → equal weights are optimal with no self-reported counts. The lease ledger hands each block to exactly r
  beacon-drawn cards (SPARe spacing). The r duplicates are checked, then enter aggregation **as one** contribution.
  Today the synthesis would double-weight any shard it duplicates.
- *Novelty:* new combination. *Confidence:* medium.
- *Cheap test:* lab, 100M, 16 simulated replicas at 1×–5× speeds. Arms: equal-step cells + equal weights;
  equal-step + token weights; equal-token cells + equal weights; duplicates counted twice vs once. Loss at equal tokens. ~2–3 h.

**2. Churn-proof, robust merge: median norm × mean direction.**
- *Source:* 2604.21428 App. D.2 (outer gradients are "almost pairwise orthogonal", so the mean's norm ≈ R/√M and
  the outer LR must be retuned per M; RDA fixes it; Table 9: M=16 text 52.7→54.5, vision 28.2→33.6);
  2603.08163 §2.2 (median-norm rescaling); 2609.39357 Prop. 2 (screening needs an anchor).
- *Apply:* contributor counts swing every round under churn, so a plain mean changes the step size ~1/√M without
  anyone noticing. Use direction = mean of unit vectors (each sender moves it ≤1/M); size = **median** of norms (not
  RDA's mean, which a liar can inflate). Embeddings stay on plain averaging (App. D.2).
- *Novelty:* new combination (RDA + Covenant median). *Confidence:* medium.
- *Cheap test:* SparseLoCo lab, participants per round drawn from 2–16. Arms: mean / RDA / median-RDA, with and
  without one −10Δ or 10×-norm sender (extends the Section 3.8 Byzantine run). ~2 h.

**3. Constant-peer sparse gossip at fixed local batch, with beacon-chosen coordinates.**
- *Source:* 2611.23053 Fig. 7 / App. A.1 (loss tracks the *peer batch* q·B/R; the q=R/2 rule comes from a fixed
  global batch); Eq. 12 (indices cost 1.5 bytes per value); §4.4.2 (deadline = 2× healthy p99).
- *Apply:* our local batch is fixed by card memory, so the peer batch is q × b_local and does not shrink as R grows.
  A *constant* q (4–8) may suffice. Separately: pick the coordinate mask from the public beacon instead of Top-k.
  That removes index bytes (86% of the payload at 2-bit values, our arithmetic) and aligns every message on the same
  coordinates, enabling per-coordinate trimmed mean around the receiver's own model (THROTTLE Prop. 2 anchor) and spot replay.
- *Novelty:* reinterpretation + new combination. *Confidence:* low–medium (random-k with EF is usually weaker than Top-k).
- *Cheap test:* 32–64 simulated replicas, fixed local batch, H=30, q ∈ {2,4,8} vs q=R/2; Top-k vs beacon mask at equal bytes. ~4 h.

**4. Put the whitening on the aggregator, keep cheap Adam on the card (IsoLoCo + β2 rule).**
- *Source:* 2610.19064 §2, §5.2, Fig. 5: IsoLoCo (2607.03011) = AdamW inner + per-matrix SVD of the averaged
  pseudo-gradient with all singular values set to their mean; with scaled β2 it beats MuLoCo (512M, R=64: 2.702 vs
  Muon 2.738; SOAP 2.649) "while only doing the spectral decomposition during the outer step". β2 rule
  β2(R)=β2(R0)^(R0/R) (Eq. 20) alone gained ~1.2%.
- *Apply:* SOAP/SPlus state (~5d² FP32 per matrix) does not fit next to a 1B slice on 16 GB. IsoLoCo moves the
  spectral work to shard committees or large-RAM orchestrators. Set β2 from local tokens: 16k-token cards vs a 393k
  reference at 0.95 → β2 ≈ 0.998 (our arithmetic).
- *Novelty:* known method, new placement. *Confidence:* medium.
- *Cheap test:* 16–64 replicas at 10M/100M: AdamW (fixed β2) / AdamW + β2 rule / + Iso-C outer / SPlus inner; log loss and peak card memory. ~3 h.

**5. Admit replicas by measured critical batch; overflow to experts and data roles.**
- *Source:* 2607.20548 §5.3 (batch ramp from 2M tokens, LR ∝ √B; "critical batch size typically depends on the
  training loss"); §3.1 Eq. 2 (experts see B·k/N of the batch); 2604.21428 Table 4 (diurnal extra learners: 0.62×
  train time at iso-FLOPs).
- *Apply:* a path's useful replica count grows as its loss falls. Inference: the record already holds many
  same-start, different-data replica updates, so their spread vs. mean estimates gradient noise scale (McCandlish et
  al. 2018, from memory). That gives a deterministic, auditable rule: admit replicas up to the measured critical
  batch. Overflow volunteers go to expert-owner groups (small effective batch), other paths, or data jobs. This
  replaces the fixed "~128 replicas" trigger in synthesis §4.6 and absorbs day/night swings.
- *Novelty:* new combination. *Confidence:* low–medium.
- *Cheap test:* at 100M, estimate noise scale from 8 replica pseudo-gradients every N rounds; compare fixed R vs
  noise-scale-admitted R at equal tokens. ~3 h.

## 2. Per-paper findings

**2603.08163 Covenant-72B**
- EF buffer is a full model-sized buffer, offloaded during compute (§3, Fig. 1). The synthesis' 16 GB budget omits it (1B × BF16 = 2 GB).
- Assigned-vs-unassigned check: an update that helps random data more than assigned data scores negative (§2.2). Inference: cheap proof-of-assigned-data per cell.
- 24.4 active vs 16.9 selected peers per round (App. A): ~31% of submitted work unused.
- Index bound 7.36 vs 12 bits sent (§2.1) → ~33% smaller uploads possible (our arithmetic), at encoder cost.
- Upload once to object store, everyone downloads (§3). Download = R × update (128 × 29 MB ≈ 3.7 GB per round, our arithmetic).
- LR flattened by hand for low participation; outer α cut 1.0→0.65 on a plateau (§4.1). Schedules must be keyed to verified tokens, with pre-committed rules.
- 14.2B-token anneal (~1.3% of tokens) raised MMLU 62.5→67.1 (Table 3): a natural record-triggered event.

**2604.21428 Decoupled DiLoCo**
- RDA and embedding exception (App. D.2, Tables 8–9) → Top 5 #2. Overwrite local fragments (α=0) for M ≥ 8 (App. D.1).
- K=1 without a grace window dropped vision 37.6→25.7; grace restored 37.0 (Table 5). Grace ≤ γ(τ·ξ_step − ξ_quorum − ξ_sync) (Eq. 3).
- Without EF: int4 +0.09% loss; 2-bit +32%; 1-bit +83% (Table 10).
- Joiners copy params **and** optimizer state (≈3× weights) from a peer, keyed on a vector clock (App. E.3).
- Event tapes (vector clocks, participant sets, weights) make an async run bitwise-replayable (App. E.1). Inference: store inclusion sets per merge in the record → aggregation is auditable too.
- Chandy–Lamport checkpoints without a global barrier; failed learners omitted (App. E.2).
- Async trails DP at 26B tokens, leads at 1.3T (Table 12).

**2606.11081 GASLoC**
- At H=1, 2-Peer gossip beats complete averaging (3.19 vs 3.24, 8 replicas, Table 5); sparse beats DiLoCo at H ≤ 5 (Fig. 4).
- A fixed ring degrades sharply (32 replicas: 3.81 vs 3.53, Table 1b). Peers must rotate every round.
- Outer Nesterov = gossip acceleration (χ→√χ, Prop. 2); Nesterov > SGDM > SGD (Fig. 7). Covenant's momentum-free α=1 should not be copied.
- At 10% bandwidth a straggler keeps H_i=15 on 1-Peer but only 1 on 2-Peer (Fig. 3). Slow uplinks should pick k=1.
- Exchanges are symmetric; "directed asymmetric protocols" not modelled (§3.2). Home NAT is untested.

**2611.23053 SPICE**
- Peer-batch result and index cost → Top 5 #3. More peers × sparser beats fewer × denser at equal bytes (Fig. 3b).
- Absolute values, not deltas (Eq. 6–7): any receiver can read a message without history, so per-coordinate robust statistics are possible.
- Outer momentum only 0.1–0.3, outer LR 0.8–0.9 (Table 7); β2 0.95→0.997 for R=8→128.
- Chunks cross tensor boundaries; workers need "a consistent parameter ordering and shard layout" (§3.2). That ordering belongs in the protocol spec.
- All quality runs are H=1 (§4.1).

**2611.17991 LACCO**
- Matches DiLoCo at H=10–15 but trails at H=30 (3.001 vs 2.961) and H=75 (3.103 vs 3.016) (Table 5).
- Full overlap needs T_inner ≥ T_comm + T_outer; offload moves 5 FP32 buffers, so H ≥ 19 vs 9 (App. D, Table 9).
- Gain shrinks with bandwidth: 1.87× at 0.5 Gbit/s, 1.07× at 5 Gbit/s (Table 3).
- Averaging both AdamW states beats shared/rewind (2.988/3.008/3.029, Table 10).
- ‖θ−θ̃‖ rises then collapses to ≈0 (App. E.1). Inference (untested): a known-shape drift curve could serve as a cheap per-card health signal.

**2610.19064 Replica scaling**
- R=64 penalty: SPlus +3.1%, SOAP +4.8%, Muon +8.5%, AdamW +11% (Table 1). At H=1, Muon is worse than AdamW (3.417 vs 3.157, Table 3).
- Anisotropy: AdamW 35 vs SPlus 0.43 (Table 4); mean mismatch only 7.5–19% of noise (Table 5).
- Inference: whitened honest updates are near-isotropic, so spikes stand out. The paper's "64 reruns from one checkpoint, vary only data" probe (§5.3) is a ready recipe for synthesis X2 tolerances.
- SPlus beats its own DP at R=8, possibly by "majority vote" behaviour (§5.1). Sign-vote aggregation is a robustness candidate.

**2609.39357 THROTTLE**
- Per-round cap is 1/n + 1/(q−1): at q=1.1 repeats can total ~10 units vs 1/n for a first arrival (Eq. 2).
- The asynchrony error term G²|B|²/(q−1)² persists even with full gradients and is not removed by momentum (§4). Sybils hurt quadratically.
- No anchor → no robust per-input screening (Prop. 2). Centered Truncated Mean drops inputs beyond V/√δ (§3.4).
- Fails on the Empire attack (59.6–61.5% vs 94–96%, §6.1). A round needs *all* clients, and restarts discard in-flight work (Alg. 1).
- Inference: geometric decay η/q^c could price repeat *leases* per contributor, to stop data hoarding.

**2607.20548 SOAP/Muon at scale**
- Stale preconditioner → "slingshot"; 8B diverged; fix = per-step QR with the current gradient + KL factors (§5.4). Rejoining cards need a basis refresh.
- Critical batch and MoE expert batch → Top 5 #5.
- Update RMS predictable (SOAP = AdamW; Muon factor ≈ 0.2, §5.2) → free per-matrix sanity check.
- Newton–Schulz in BF16 leaves a noise floor in small singular values (§7). Cross-vendor Muon replay needs looser or truncated comparisons.
- KL-SOAP best "when memory footprint is not a limiting factor" (§7); it is limiting for us.

**2610.03457 DARL**
- Duplicate bias and exactly-once → Top 5 #1. No leasing → 1.73× fewer unique blocks (§4).
- Failure unit = one lease = one inner phase; TTL = 2.5 × step time + β·RTT (§3.2). H must fit session lengths.
- Solo contributor takes the full step; stale generations rejected (§3.1). This is the opposite of Decoupled's grace window.
- Overhead flat at ~110 s/round across H (Table 1); HTTP blob 8.1% faster than gRPC; a proxy killed long gRPC streams (Tables 2, 5).
- Matched-steps quality: PPL 34.7 cross-site vs 28.2 centralised (§4).

## 3. Contradictions with the synthesis

- **C1 Optimizers.** Synthesis §3.2/§4.1 treats "SOAP/SPlus/Muon" as one family. On the replica axis Muon is much
  worse (Table 1), and at H=1 it is worse than AdamW (2610.19064 Table 3). Meanwhile the pipeline axis prefers Muon
  (2606.30634). Each axis needs its own choice, or a test of SPlus under pipeline delay.
- **C2 Bandwidth and memory arithmetic (§4.3).**
  - Download is missing: an all-gather of 128 replicas ≈ 3.7 GB per round, about 5 min at 100 Mbit/s.
  - Gossip to k peers costs k uploads unless relayed. Home links are asymmetric, so "upload once to a relay" (Covenant §3, DARL Table 2) may beat gossip.
  - Joining needs optimizer state (≈3×) (2604.21428 App. E.3).
  - The EF buffer needs ~2 GB more per 1B slice (2603.08163 §3).
- **C3 SPICE ceiling.** "Scales to 128 only if each worker talks to half" holds under a fixed global batch
  (2611.23053 Fig. 7). In our fixed-local-batch regime a constant q may work, or the critical batch may bind
  first. Untested either way.
- **C4 LACCO "hides it".** LACCO degrades at the large H that slow links push us toward (Table 5) and needs H ≥ 19 for
  PCIe offload (Table 9). At our numbers the 12 s upload already fits inside the compute window; the download is what needs hiding.
- **C5 THROTTLE as influence cap.** The cap is loose (≈10 units at q=1.1) and needs every client per round; it is not
  a per-contributor limit in the sense synthesis §4.5 implies.
- **C6 Equal weights.** Equal weights are optimal only when replica covariances match (2610.19064 §4.2). With
  heterogeneous speeds, Decoupled DiLoCo needed token weighting. Resolved by equal-token cells (Top 5 #1), not by assertion.
- **C7 Efficiency guess.** Matched-step cross-site DiLoCo lost clearly at 0.6B (DARL §4). Covenant's outer LR was
  hand-cut (§4.1). The "25% efficiency" guess (§2) carries real downside risk at small scale.

## 4. Promising cited papers not in the folder (2025-11 to 2026-11)

| arXiv | What | Why |
|---|---|---|
| 2607.03011 | IsoLoCo: model merging for DiLoCo aggregation | Top 5 #4; previously "considered, not selected" |
| 2601.22442 | AsyncMesh: fully async data + pipeline parallel (Pluralis) | Async on both axes we combine |
| 2601.02360 | Heterogeneous low-bandwidth pre-training of LLMs (Covenant/Mila) | Heterogeneous peers, same stack |
| 2606.22768 | Factored Gossip DiLoCo | Gossip overlap; previously skipped |
| 2511.13761 | What happens when nanochat meets DiLoCo? | Negative result (DiLoCo < DP) |
| 2512.17131 | Smoothing DiLoCo with primal averaging | Low-H DiLoCo failure mode |
| 2602.04396 | LoRDO: low-rank optimization, infrequent comms | Smaller syncs |
| 2602.02016 | DASH: block Shampoo | Memory-light whitening for cards |
| 2602.12429 | Stable native low-rank pretraining | Smaller slices and updates |
| 2605.02125 | FedQueue: queue-aware FL, stale down-weighting | Lease/admission policy |
| 2605.20866 | LoSCAR-SGD: delay-corrected sparse averaging | Overlap alternative |
| 2602.00277 | Fault-tolerant HSDP on 100k GPUs | Churn at scale |
