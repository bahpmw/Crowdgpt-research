DRAFT — requires human review

# Addendum: what a second, deeper read of the papers changed

Date: 2026-11-18. Companion to `SYNTHESIS-round2.md`. Local research note, not committed.

The first synthesis was written from the short notes for each paper. This addendum comes from a second pass that
read the full texts of all 36 papers, the round-1 lab code and results in `research/lab/`, the older research notes in
`research/*.md`, and the "runners-up" each paper-finder found but did not download. The raw findings are in
`research/papers/mining/` (G1–G5 by topic, runners-up-B1–B7 by search bucket, about 27,000 words in total). This file
keeps only what changes the plan.

Terms used below: a **cell** is one small, checkable piece of work (one slice of the model, one block of data, one
round). A **module** is a slice of the model's layers trained on its own (the SOLO method). A **replica** is a copy of a
module trained on different data and averaged with other copies. **Error feedback** keeps the part of an update that
was not sent so it can be sent later. **Beacon** is the shared, unpredictable random number that orchestrators publish
each round. All numbers marked "guess" are not measured.

---

## 1. Bottom line: seven things that changed

1. **There is a better way to split the model than pipelines: recompute instead of pass along.** The lab already has
   a design called MOSAIC (`research/lab/REPORT.md` lines 64–109, `research/lab/methods/mosaic.py`). Each volunteer
   who trains module *k* downloads the earlier modules and runs them forward itself, at the round's frozen weights,
   instead of waiting for another volunteer to send activations. That removes four problems at once: no pipeline delay,
   no need to compress activations between machines, no "latency crews", and no way for a dishonest upstream volunteer
   to feed bad inputs downstream. The price is extra forward compute and holding earlier modules in memory (inference
   precision is enough). It has only had a 15-second smoke test (K=2: 5.78 versus 5.89 for one module). This is now the
   first experiment to run.

2. **The poisoning threat is worse than the first synthesis said.** A November 2026 paper by Carlini, Jagielski and
   others (arXiv 2610.37741, not yet downloaded) reports that poisoning 0.1% of pretraining data gives about 92% attack
   success at 4B parameters, rising with model size and surviving later fine-tuning. Among a million equal volunteers
   that is about 1,000 identities; if a few volunteers can choose their own data it may be fewer. A second paper
   (2604.02372) shows that one dishonest middle pipeline stage can plant a backdoor that survives safety training. A
   third (2606.04819) found a large live "useful work" network that did no useful AI work because its checker accepted
   random matrices. So: the network must assign the data, must check which tokens a volunteer actually trained on, and
   must cap how much of the data any one identity touches.

3. **One redundancy dial cannot do three jobs.** SPARe's low cost works only because its spare copies never run unless
   something fails, so there is nothing to compare; at r=2 it does worse than plain duplication and it needs r≈8–9 to
   shine. Split the budget: churn is handled by **leases** (time-limited work grants that are handed out again if a
   volunteer vanishes, as in the DARL system), and wrong or dishonest work is handled by **sampled duplicates** that
   the beacon picks after volunteers commit.

4. **A consumer card can hold far more than the 1B parameters we assumed.** MegaTrain trained 3–7B models on one 24 GB
   RTX 3090 at 30–35 TFLOPS by keeping weights in system RAM (25–57 GB). A runner-up paper, LLMQ (2512.15306), reports
   pretraining a 7B model on a 16 GB RTX 5060 Ti. Storing weights in FP8 with 8-bit optimizer state brings memory to
   about 2–4 bytes per parameter. "Fatter" cells mean fewer messages and fewer cells to check, and also mean the data
   wall arrives sooner. These are single-machine numbers without a network or checking overhead; our planning figure
   stays at 6 TFLOPS per card until our own runs measure it.

5. **Delay and coordination are the wall, not raw bandwidth.** In Agora (8.6B model, 330 machines), throughput stayed
   flat between 200 and 1,000 Mbit/s but fell 40% when round-trip time went from 0–20 ms to 60–80 ms, and the lookup
   and averaging services slowed 8–10 times with only 60 machines online. The lab's own network model
   (`research/lab/crowd_scale.py`) shows that 1M volunteers sending compressed updates to one receiver would need
   about 2.76 Tbit/s into it. A hierarchy of small groups is not optional.

6. **"Use SOAP, SPlus or Muon" was too simple; the right optimizer depends on the axis.** When many replicas are
   averaged, SPlus and SOAP lose least (64 replicas at 178M: SPlus +3.1%, SOAP +4.8%, Muon +8.5%, AdamW +11%). When
   modules run with a fixed delay, Muon holds up best. SOAP and SPlus state does not fit on a 16 GB card next to a
   large slice. Practical answer: plain AdamW (with the β2 rule) on volunteers, and the more expensive optimizer maths
   applied once at the group level when updates are merged (the IsoLoCo idea, 2607.03011).

7. **Growth should be sparse, early, in doubling steps, and planned against a published horizon.** Turning a dense
   model into a mixture of experts (MoE) later did worse than training a half-size MoE from scratch at matched tokens
   (Expert Upcycling, Table 4). Doubling is the most reliable growth step (Warmstarting, 2605.13405). After doubling
   experts, train at least as many tokens again before the next doubling. The best point to split one model into many
   specialists depends on the total training horizon, so that horizon must be fixed in the genesis epoch table.

---

## 2. Corrections to `SYNTHESIS-round2.md`

| First synthesis said | Correction | Source |
|---|---|---|
| ResBM boundary ≈ 128 bytes per token (~1 Mbit/s) | ≈ 64 bytes per token, ~0.5 Mbit/s at 1,000 tokens/s | G1 C2 |
| SOLO modules plus ResBM boundaries compose | Not directly: the ResBM encoder learns from gradients that cross the boundary, which SOLO removes. Needs its own local loss, or MOSAIC avoids the boundary entirely | G1 C1 |
| SOLO with K=8 costs +89% compute | Shape-specific. Deep, narrow modules with one-block heads cost about +10% (SOLO Eq. 6, Table 13) | G1 #5 |
| About 1B parameters per card | 3–7B per card is possible with RAM offload and FP8 storage | G1 #3, B2, B5 |
| SOAP/SPlus/Muon as one recommendation | Split per axis (see §1 point 6). Muon is worse than AdamW with one local step (3.417 vs 3.157) | G2 C1 |
| Redundancy r=2 does triple duty | No. Leases for churn, sampled duplicates for checks | G3 #1 |
| Deterministic replay costs 5–7× (OPEN-1B) | Only in int8 with about 0.2 nats worse loss; bit-exact BF16 costs about 12× | G4 C1 |
| Equal weights near-optimal when merging | Only when every replica's noise is equal. Make work units equal in tokens and count duplicates once | G2 #1, C6 |
| Bandwidth check covered uploads | Downloads were missing: fetching 128 replicas' updates is ~3.7 GB per round; error feedback adds ~2 GB per 1B slice; a joining volunteer needs optimizer state ~3× the weights | G2 C2, G1 C7 |
| Behaviour screening (Chorus) is a validated defence | Only tested on refusal backdoors in instruction-tuned models; comparing peers within one round has a 34–60% false-positive floor | G4 C4, C6 |
| Score usefulness by loss drop | Loss can fall while the model gets worse (LoCA: BoolQ 0.852 → 0.364). Score loss drop on assigned data versus other data, and never as the only check | G4 C5 |
| Multi-token prediction (MTP) is not worth it | Not for pretraining loss, but it may help when data is the limit; keep as a low-confidence option | G3 #5 |
| Gossip between 1–2 random peers | Falls behind full averaging as replicas grow (3.62 vs 3.53 at 32); older notes in `research/README.md` ranked gossip low under churn. Use full averaging inside small groups, gossip only between groups | B1, G5 |
| Lab results say crews must run near lockstep | Possibly an AdamW artefact; Muon plus error feedback tolerated a one-step delay at 10B | G1 C9 |
| Token-weighted averaging is banned | The lab's MOSAIC uses token weights in its "budget" mode; equal-token work units make the two agree | G5 |

---

## 3. Ranked new ideas

Confidence: **High** = directly measured in a paper on a similar setup; **Medium** = measured on a different setup;
**Low** = our own inference.

| # | Idea | Source | Why it helps at large scale | Confidence | Cheap test |
|---|---|---|---|---|---|
| 1 | **MOSAIC: recompute earlier modules instead of receiving them** | lab REPORT.md, mosaic.py | No pipeline delay, no boundary traffic, each cell checkable alone, no upstream cheating | Medium (smoke test only) | Run `research/lab/arms/solo-crowd.txt` at 100M/300M |
| 2 | **One fixed, published delay τ for everything** (inputs, frozen output layer, merge inputs at version t−τ) | Async PP §6; SOLO App. F | Fixed delays train fine; variable delays degrade. Late volunteers are skipped, so every cell stays replayable | Medium | Staleness sweep already in `arms/` with fixed vs random delay |
| 3 | **Equal-token work units, each data block sent to exactly r volunteers, duplicates counted once** | 2610.19064 §4.2; DARL §3.2 | Makes equal-weight merging correct with mixed hardware; fast cards just take more units | Medium | Mixed-speed simulation in lab harness |
| 4 | **Leases with re-leasing; fixed global batch by moving quotas** | DARL; ReCoVer 2605.11215; GeoMesh 2609.18388 | Rounds close on coverage, not on everyone; batch size stays constant regardless of who is online | High (datacenter) | Churn simulator (X7) |
| 5 | **Sampled duplicate checks, paired by a Golomb ruler and rotated by the beacon** | SPARe App. B.1; Backdoor Absorption 2607.06643 | Any two volunteers share at most one cell per round, so colluders cannot keep vouching for each other; 10% step audits cut attack success from 99.5% to 7.2% at 2.8% overhead (small model) | Medium-low (pairing untested against collusion) | Simulated collusion in X8 |
| 6 | **Check activations, not just updates** | PRISM Obs. 2 and 4 | Errors in averaged updates get diluted; errors in forward activations pass through undiluted | Medium | Fault injection in X2 |
| 7 | **Integer or fixed-point maths wherever two parties must agree exactly** (merge sums, hashes of boundaries, verifier scores) | OPEN-1B; Giskard | Exact agreement without bit-exact training kernels; int8 replay was faster than BF16 | High for sums | Unit test: same sum on MPS, CUDA, CPU |
| 8 | **Fat cells: FP8 weight storage, 8-bit optimizer state, weights in system RAM** | ECO; MegaTrain; LLMQ; 2605.11396 | 3–5B slices on 16 GB cards, 5–8B on 24 GB; fewer cells and messages | Medium | X6 on a CUDA card |
| 9 | **Groups of ~16 replicas average fully through a relay; gossip only between groups** | B1 (SPICE, GASLoC, HALoS); research/README.md | Each volunteer uploads once (~29 MB, ~12 s at 20 Mbit/s) and downloads 15 others; equal-size groups (unequal ones diverged) | Medium | X3 |
| 10 | **Merge = median length × mean direction** | Decoupled DiLoCo App. D.2; Covenant | Averaging near-orthogonal updates shrinks the step by about 1/√M; the median length also caps any one volunteer's pull | Medium | X3 arm |
| 11 | **Admit volunteers to a model path only up to its critical batch size; send overflow to experts, other paths, or data jobs** | SOAP/Muon at scale §5.3 | Extra volunteers add value instead of noise; the limit can be estimated from the spread of updates already in the record | Low-medium | Estimate from X3 logs |
| 12 | **Make "what to send" checkable**: hash error-feedback state into the cell record, or send coordinates chosen by the beacon (no index bytes); or use stateless compression (random rotation then rescale) | B1; G1 #2; G4 #3 | Indices are ~86% of a 2-bit compressed message; stateless compression survives volunteer departure | Low-medium (random choice usually compresses worse) | X3 arm |
| 13 | **Grow sparse early, in doubling steps, with spacing and a published horizon; route whole documents** | Expert Upcycling; Warmstarting; Optimal Splitting §5.3; EMO 2605.06663; FoMoE 2606.19025 | Growth events become predictable, reproducible "halving-style" events; document routing lets a volunteer hold only its experts | Medium | X4 |
| 14 | **Cheap forward-only checks on inference-only devices** (does the update lower loss on assigned data more than on other data; does it match a random direction drawn after commit) | 2611.08712; 2609.10980; G4 #1 | Gives phones and small cards a real job; catches copying and lazy work, not poisoning | Medium-low | X9 |
| 15 | **Per-device reputation and quarantine for silently faulty hardware** | Guard 2605.17879; TrainSDC 2608.30769 | Faults repeat on the same device; quarantine raised time between failures from 6.6 to 16.7 h | High (datacenter) | Fault injection in X2 |

Long shot worth watching: a zero-knowledge proof of a checkpoint's held-out loss (zkLLMPoT, 2610.08258) takes 41–59 s
to prove and under 0.5 s to check at 1.1–1.5B. That would let the network prove model quality without publishing the
test set.

---

## 4. Revised cell mesh (changes to §4 of `SYNTHESIS-round2.md`)

### 4.1 Cell definition

A cell is now: (path, module, replica group, round), with these inputs written into the record before work starts:
weights version at round t−τ, data block number from a beacon-shuffled order, device class (vendor, kernel version,
precision). Output: update commitment, hash of the module's output activations on a fixed probe batch, loss on the
assigned data.

### 4.2 Choosing how to split the model

| Option | Use when | Main cost |
|---|---|---|
| Full replica (no split) | Model slice fits the card (up to ~3–5B on 16 GB with fat-cell tricks) | None beyond memory |
| MOSAIC (recompute earlier modules) | Bigger model, K ≤ 4 | Extra forward compute: about (K−1)/2 module forwards on average, or roughly a sixth to a half of a step (guess); earlier modules held in FP8 |
| SOLO crews (pass activations) | MOSAIC compute cost too high and volunteers are close together | Fixed delay τ, boundary compression, upstream trust, latency-region grouping |
| Independent paths / experts | Beyond what one path can absorb | Needs a shared trunk first; routing by document |

### 4.3 Sizing, redone with fatter cells

With 3–5B per card, one path of K=4 modules is 12–20B parameters. Replicas per path are capped by the critical batch
size (the first synthesis used 128). That makes one path absorb only a few hundred cards, so at a million cards almost
all of the scale must come from experts and independent paths, and the rest from verification, evaluation and data
jobs. Example (guess): 256 paths × 4 modules × 8 expert groups × 128 replicas ≈ 1.05M cards. The honest reading is the
same as before: a million GPUs means a very large sparse forest, and data runs out before compute does.

### 4.4 Syncing, three levels (revised)

1. **Replica group** of ~16 equal-speed-class volunteers on the same (path, module): each uploads once to a relay
   (orchestrator or volunteer seeder), downloads the others, and merges with median length × mean direction. Error
   feedback state is hashed into the cell record or reset each round.
2. **Shard committee** per (path, module), drawn by the beacon: merges the group results, applies the heavier
   optimizer maths, checks coverage, signs a shard checkpoint. Groups gossip with beacon-chosen partner groups.
3. **Root record**: commits only shard checkpoints and growth events.

Rounds close on coverage, quorum, or deadline, whichever comes first; nothing waits for everyone.

### 4.5 Optimizer per axis

| Axis | Choice | Why |
|---|---|---|
| Inside one volunteer | AdamW, β2 set from tokens per step | Fits memory; heavier states do not fit next to a big slice |
| Merging replicas | SPlus- or SOAP-style step at the committee | Least loss from many replicas |
| Modules with delay | Test Muon versus AdamW under fixed τ | Muon tolerated delay best in Async PP |

### 4.6 Growth rules (add to the epoch table)

- Publish the total token horizon at genesis; the split point depends on it.
- Grow in ×2 steps; after each, train at least as many tokens again before the next.
- Become a mixture of experts early (around 1B), not after 9B.
- Each expert has at least two owners; a lost expert is rebuilt from its nearest neighbours.
- Every growth step is exactly reproducible from the record, followed by a quality gate (loss close to before
  growth, not near random) and a temporarily higher check rate, because loss jumps hide faults.

---

## 5. Threats and defences (update to the trust doc)

| Threat | Evidence | Defence |
|---|---|---|
| Backdoor through ~0.1% poisoned data | 2610.37741 | Network assigns data from pinned shards; sampled replay proves which tokens were trained; cap each identity's share of tokens |
| One middle stage plants a backdoor | 2604.02372 | MOSAIC recompute, or duplicate checks on boundary activations |
| Thin poison spread over many clients | 2602.15671 (>85% success) | Identity caps alone do not stop it; assigned data plus behaviour screening on a sample |
| "Useful work" that is not useful | 2606.04819 | Verifiers get known-answer jobs; replay real work; reject updates that do not lower loss on assigned data |
| Near-zero or lazy updates pass a tolerance check | Proof-of-learning attacks; v1 three-client check | Minimum-progress floor; gain on assigned data must exceed gain on other data |
| Poison hidden in low-rank factors | 2601.00566 | Check the applied product, not the factors |
| Copying another volunteer's update | 2604.04611 | Commit before the beacon reveals checks; assigned-versus-other-data test |
| Silent hardware faults | PRISM; TrainSDC; Guard | Device reputation, quarantine, NaN and norm guards, no FP16 on clients |
| Colluding duplicates | v1 let three colluders through | Golomb-ruler pairing redrawn by the beacon after commit; pair across failure domains |
| Gamed contribution scores | 2605.15520 (Shapley score inflated 3×) | Credit only verified, assigned tokens |
| Committee capture | Giskard; B6 arithmetic | With 10% fake identities, ~67 members for honest majority, ~353 for under a quarter corrupted; identity cost is the real lever |

---

## 6. Revised experiment list

These replace X1–X7 in `SYNTHESIS-round2.md` §6.

| # | Experiment | Agent-hours (guess) |
|---|---|---|
| X1 | MOSAIC versus SOLO crews versus plain backprop at 100M and 300M, K=2 and 4; fixed-versus-random delay sweep | 4–8 |
| X2 | Same-class exact replay versus cross-class tolerance (MPS, CUDA, CPU), tolerance calibrated on honest replays (OPFL 2609.37011), fault injection, near-zero-update attack | 3–6 |
| X3 | Groups of 16: full average versus gossip, equal-token units, median-length merge, AdamW versus SPlus at committee, critical batch estimate | 4–8 |
| X4 | Expert doubling with spacing rule, ×2 versus ×4 growth, document routing | 4–8 |
| X5 | Gated DeltaNet hybrid versus plain at 10–100M (check Apple GPU kernel support first) | 2–4 |
| X6 | Largest trainable slice on a 16 GB CUDA card with FP8 storage, 8-bit states, RAM offload; tokens per second | 2–4 (needs CUDA card, decision D9) |
| X7 | Churn simulator: leases and re-leasing versus pre-placed spares | 2–4 |
| X8 | Small red team: one identity poisons 0.1% of its data; does assignment plus sampled replay catch it; colluding pairs | 3–6 |
| X9 | Forward-only checks on an inference-only device catch copied and lazy updates | 2–4 |

Total 26–52 agent-hours. Basis: a guess. The closest comparison in this repository is the v1.5 build (23 commits over
about 5–6 active hours), but each of these is a research experiment with tuning and repeat runs, so expect several
times that per experiment. Human wait (hardware access, review) is separate.

---

## 7. Papers to download next

IDs come from the helpers' reports and have not yet been checked by downloading.

| arXiv | Short name | Why |
|---|---|---|
| 2610.37741 | Agentic backdoors via pretraining poisoning | Sets the poisoning threat level |
| 2604.02372 | Backdoors in decentralised post-training | Malicious pipeline stage |
| 2609.37011 | OPFL | Tolerance calibrated on honest mixed hardware |
| 2602.00277 | FT-HSDP (100k GPUs) | Replica as failure unit, catch-up without blocking |
| 2601.22442 | AsyncMesh | Async partial averaging, used by Agora |
| 2601.02360 | Heterogeneous low-bandwidth pretraining | Small cards form a pipeline replica beside big-card replicas |
| 2512.15306 | LLMQ | 7B pretraining on a 16 GB card |
| 2605.06663 | EMO | Document-level expert pools |
| 2605.13405 | Warmstarting | Doubling as the reliable growth step |
| 2602.03515 | Basis Rotation | Why Adam fails under delay |
| 2605.11215 | ReCoVer | Fixed batch while machines drop out |
| 2609.18388 | GeoMesh | Speed quotas with synchronous rounds |

---

## 8. Open conflicts the owner should know about

- **MOSAIC versus crews.** MOSAIC is simpler and safer but costs extra compute; only X1 can settle it.
- **Dense first epoch versus sparse early.** The archived v2 roadmap trains a dense 1B first. The growth evidence says
  become a mixture of experts early. This touches decision D8 (epoch table).
- **Exact replay versus tolerance.** Exact replay across vendors costs about 12× in BF16. Plan: exact within a device
  class, tolerance across classes, exact integer maths for merges.
- **Error feedback.** It is needed for compression to work (lab: 5.07 with it, 5.66 without), but its state is hidden
  and lost when a volunteer leaves. Either record its hash or test stateless compression.
- **Fixed global batch versus batch that grows with volunteers.** Papers favour a fixed batch; growth then goes into
  more paths and experts.
- **Gossip.** Older notes ranked it low under churn; the new papers agree it falls behind full averaging. Keep it only
  between groups.

---

## 9. Further changes to the archived v2 plan

Extends the table in `SYNTHESIS-round2.md` §8. Nothing has been edited in `archived/crowdgpt-v2/` yet.

| File | Change |
|---|---|
| `adr/0005-model-sharding.md` | Add MOSAIC as the preferred split, crews as fallback; fixed delay τ |
| `adr/0004-two-training-lanes.md` | Local-module lane becomes the main lane; DUST becomes an optional probe |
| `05-scale-roadmap.md` | Sparse from ~1B; ×2 growth steps with spacing; published token horizon; hardware floor with fat cells |
| `02-trust-and-verification.md` | Threat table from §5; leases versus sampled duplicates; activation checks; device-class replay |
| `03-consensus-and-record.md` | Cell record fields (τ, device class, error-feedback hash); coverage-based round close; committee size arithmetic |
| `01-architecture.md` | Three-level sync with relays; per-axis optimizer |
| `TODO.md` | Phase 0 rows replaced by X1–X9 |
