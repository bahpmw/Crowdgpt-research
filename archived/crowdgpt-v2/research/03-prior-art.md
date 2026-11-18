DRAFT — requires human review

# CrowdGPT v2: prior art in decentralized, zero-trust LLM pretraining

- **Prepared:** 2026-11-17, from web research done that day. Local-only scoping document; do not commit.
- **Scope:** prior art for the five v2 requirements: (a) many user-run orchestrators that keep the network alive and scale out, (b) zero trust, (c) autonomous agreement with immutable records, (d) live public metrics, (e) a built-in scale schedule. It applies this to v1.5's DUST (activation-perturbation, zeroth-order) training and its backprop route.
- **Evidence labels:** **[P]** = primary source (paper, official docs or repo, author blog). **[S]** = secondary source (press, analysts, wikis). **[U]** = unverified, or my own analysis with no source. Dates are publication dates where known.
- **Known source caveat:** one search result (a "Lightbulb Partners" summary of INTELLECT-3) describes DiLoCo and federated features that do not appear in the primary report. It was ignored. INTELLECT-3 was trained on a co-located 512×H200 cluster ([arXiv 2512.16144](https://arxiv.org/abs/2512.16144)).

---

## 0. Executive summary

1. **Internet-scale pretraining works for data-parallel (DP) and pipeline-parallel (PP) training with backprop.** The largest completed permissionless run is **Covenant-72B**: 72B parameters, about 1.1T tokens, Sep 2025 to Mar 2026 ([arXiv 2603.08163](https://arxiv.org/abs/2603.08163)) [P]. No project has shown zeroth-order (ZO) or evolution-strategies (ES) *pretraining* at the billion-parameter scale. DUST's own write-up measures models up to 243M parameters and runs up to 20M tokens, and extrapolates beyond that ([Q Labs, Oct 2026](https://qlabs.sh/research/dust); [critique](https://posthypetech.com/posts/dust-training-without-backprop)) [P/S]. **The training algorithm, not the network, is CrowdGPT's dominant risk.**
2. **No production system is fully decentralized at the coordination layer.**
   - Psyche's coordinator is a Solana program, but it runs on **devnet**, runs are **permissioned**, and single keys can upgrade the programs [S, Oct 2026].
   - Templar relied on one validator process plus Bittensor, and its builder left Bittensor over governance in Apr 2026 [P/S].
   - Prime Intellect's protocol ran its orchestrator in Kubernetes ([INTELLECT-2 §2.4](https://arxiv.org/html/2505.07291v1)) [P].
   - CrowdGPT's "Bitcoin-like" orchestrator requirement is therefore **novel**, not off-the-shelf.
3. **Every live permissionless pretraining network verifies work by scoring, not by proof.** They measure loss before and after a contribution on assigned versus unassigned data (templar's Gauntlet), spot-check replays, or witness availability (Psyche Bloom filters). Cryptographic or bisection verification (Verde, TOPLOC) is used for inference and RL rollouts, not for pretraining gradients.
4. **Seeded ZO/ES has a structural verification advantage that nobody has exploited for pretraining yet.**
   - With shared seeds, a weight-space ES update is fully determined by (seeds, scalars). Salimans 2017, FedKSeed, MeZO, EGGROLL and SeedFlood (Feb 2026) all rely on this.
   - Verifiers can therefore recompute the aggregation exactly and recheck a sample of scalars with forward passes only.
   - DUST perturbs activations, not weights, so its updates are seed+data reproducible, not seed+scalar reconstructible. Section 2.6 sets out what this means for verification.
5. **Recommended direction:**
   - A BFT-replicated, append-only ledger run by orchestrators (CometBFT-class). Orchestrators admit validators by verified useful work, with no token.
   - Content-addressed data plane: iroh blobs/gossip plus object-store and Hugging Face mirrors.
   - Optimistic, sampled, seed-replay verification with reputation and clipped robust aggregation.
   - A deterministic milestone table in genesis that triggers model growth (G_stack / HyperCloning, then sparse upcycling to MoE), activated BIP9-style with a hard-height fallback.
   - Details and the top 10 risks are in Section 8.

---

## 1. Decentralized training systems

### 1.1 Summary table

| System | Topology | Trust / consensus | Verification | Compression | Scale achieved | Status (late 2026) |
|---|---|---|---|---|---|---|
| **Prime Intellect:** OpenDiLoCo / INTELLECT-1 | DP islands (DiLoCo), ElasticDeviceMesh, int8 ring all-reduce | Whitelisted compute | Trusted | H=100 inner steps plus int8 (~400× vs DP) | 10B, 1T tokens, up to 14 concurrent nodes, 30 contributors, 83–96% utilization ([arXiv 2412.01152](https://arxiv.org/abs/2412.01152)) [P] | Done (Nov 2024) |
| **Prime:** INTELLECT-2 | Async RL. Trusted FSDP trainers, permissionless inference swarm | Ledger on Base Sepolia testnet (Galaxy [S]). Orchestrator in k8s | **TOPLOC** on rollouts | SHARDCAST HTTP relay tree for weights | 32B RL (May 2025) ([arXiv 2505.07291](https://arxiv.org/abs/2505.07291)) [P] | Done |
| **Prime:** INTELLECT-3 | **Centralized** 512×H200 | n/a | n/a | n/a | 106B MoE RL ([arXiv 2512.16144](https://arxiv.org/abs/2512.16144)) [P] | Prime pivoted to hosted RL ("Lab", Feb 2026) [S] |
| **Nous Psyche** | DP, DisTrO/DeMo, overlapped steps. P2P via iroh | Solana coordinator program. Epochs, witness quorum | Witnesses plus Bloom filters (availability). Gradient-recompute verification listed as future work ([Lambda Class](https://lambdaclass.github.io/lambdaclass_blog/posts/lambdas-new-strategic-partnership-with-nous-research-decentralized-artificial-intelligence)) | DCT top-k momentum, sign-quantized | Consilience 40B dense (MLA), ~1.2T of a 20T target, paused [S]. Hermes 4.3 36B post-train on 24 nodes, 144k tok/s (Dec 2025) [S] | Devnet, permissioned runs [S] |
| **Gensyn** | RL Swarm (gossip, hivemind-based). Verde for jobs | Custom Ethereum rollup. Mainnet OP Stack L2, Apr 2026 [S] | **Verde** + RepOps (refereed delegation) | n/a | RL Swarm: 0.5–1.5B local models | Official swarms paused since 5 Jan 2026 ([docs](https://docs.gensyn.ai/testnet/rl-swarm)) [P]. Focus moved to Delphi prediction market |
| **templar / Covenant** (Bittensor SN3) | DP, SparseLoCo, peers post to own R2 buckets | Bittensor weights from a validator. Permissionless | **Gauntlet** loss scoring, OpenSkill, assigned vs unassigned data | Top-k + 2-bit + error feedback, >146× | **Covenant-72B**: 72B, ~1.1T tokens, ≥70 unique peers, ≤20 per round, 8×B200 per peer, 94.5% utilization [P] | Done Mar 2026. Covenant left Bittensor 9 Apr 2026. "Crucible" on own pool (Aug 2026) [P] |
| **Macrocosmos IOTA** (SN9) | PP + DP swarm. Orchestrator relays activations through S3 | Bittensor | Validator spot-checks, CLASP attribution | Bottleneck blocks, up to 128× activation compression | Orion-100B: 48 A100s, 16 stages, only 1.1B tokens, 30.8% MFU (Jun 2026) [P] | Orchestrator is centralized ([README](https://github.com/macrocosm-os/IOTA)) |
| **Pluralis** Node0 / Agora | PP + DP over consumer GPUs (16 GB+). Hivemind DHT matchmaking | Permissionless. "Protocol models" (weights unextractable) | Admission plus (per preprint) poisoning defenses [U] | Subspace networks (~100× on activations), async SPARTA | Node0-7.5B: 36B tokens, 3 weeks, >300 participants, 1.6k GPUs (Sep 2025). Pluralis-8B: 532B tokens (May 2026). Pluralis-13B systems test (Sep 2026) ([docs](https://pluralis.ai/docs/)) [P] | Active |
| **Hivemind / DeDLOC / Petals / SWARM** | DHT, volunteer DP (DeDLOC), PP inference (Petals), stochastic-wiring PP (SWARM) | Mostly trusted volunteers | Minimal | Various | sahajBERT, BLOOM-176B inference ([arXiv 2106.10207](https://arxiv.org/abs/2106.10207), [2209.01188](https://arxiv.org/abs/2209.01188), [2301.11913](https://arxiv.org/abs/2301.11913)) [P] | Library is the substrate for Pluralis and Gensyn |

### 1.2 Algorithms that set the bandwidth budget

- **DiLoCo** ([arXiv 2311.08105](https://arxiv.org/abs/2311.08105)):
  - Inner AdamW for H steps, then an outer Nesterov step on the "pseudo-gradient".
  - **DiLoCo scaling laws** (Charles et al., NeurIPS 2025, [arXiv 2503.09799](https://arxiv.org/abs/2503.09799)): DiLoCo with M≤4 replicas matches or beats DP at 4B and 10B with predictable hyperparameters.
  - Epoch reads tables 4 and 11 as ~1.5× compute penalty going from 1 to 8 replicas ([Epoch, Dec 2025](https://epoch.ai/gradient-updates/how-far-can-decentralized-training-over-the-internet-scale)) [S].
- **Streaming DiLoCo** ([arXiv 2501.18512](https://arxiv.org/abs/2501.18512), Jan 2025): syncs fragments in turn, overlaps communication with compute, and sends FP4 outer gradients. Result: ~400× fewer bits than DP at the same 1B quality. The 100B results are simulation only.
- **DeMo / DisTrO** ([arXiv 2411.19870](https://arxiv.org/abs/2411.19870)): decoupled momentum. Only the fast DCT components of momentum are sent. Psyche sends signs plus indices.
- **SparseLoCo** ([arXiv 2508.15706](https://arxiv.org/abs/2508.15706)): 1–3% top-k with 2-bit quantization and error feedback. Beats dense DiLoCo, and is what enabled Covenant-72B.
- **SPARTA** (EXO, ICLR 2025 workshop, [OpenReview](https://openreview.net/forum?id=stFPf3gzq1)): averages a random 0.05–0.5% of parameters each step, asynchronously. Pluralis uses an async variant (5% slice every 5 steps).
- **Subspace networks / protocol models** ([arXiv 2506.01260](https://arxiv.org/abs/2506.01260), NeurIPS 2025): compress pipeline activations to a low-rank subspace, about 100×.

### 1.3 Failure lessons (consolidated)

1. **Whitelists were not a real security model.** INTELLECT-1 and Consilience were whitelisted. Covenant is the first credible permissionless pretraining run, and it needed a validator that scored every round ([arXiv 2603.08163](https://arxiv.org/html/2603.08163)) [P].
2. **Loss spikes and rollbacks are normal.** Consilience rolled back and added OLMo-style skip-step guards ([Galaxy, Sep 2025](https://www.galaxy.com/insights/research/decentralized-ai-training)) [S]. CrowdGPT needs an *automatic* rollback rule in the protocol, not a human decision.
3. **Project scope drifts toward revenue.** Prime moved to hosted RL. Gensyn paused RL Swarm and pivoted to prediction markets. Nous now leads with Hermes Agent [S]. A volunteer network has to survive its founders losing interest. That is the point of requirement (a).
4. **Governance capture breaks token networks.** Covenant left Bittensor, alleging that one founder controlled the multisig "triumvirate". The founder denied this ([The Block, 9–10 Apr 2026](https://www.theblock.co/news/ecosystems/2026-04-09-covenant-ai-exits-bittensor-tao-396959)) [S].
5. **Minimum hardware bars defeat "crowd" claims.** Covenant needed 8×B200 per peer. Node0 accepted 16 GB GPUs, but only by using model parallelism.
6. **Compute supply is the ceiling.** Epoch estimates the largest decentralized pretraining runs at 6e22–6e23 FLOP, about 1000× below the frontier. Templar's network ran at ~9e17 FLOP/s. Decentralized compute has grown ~20×/year since 2020 ([Epoch](https://epoch.ai/gradient-updates/how-far-can-decentralized-training-over-the-internet-scale)) [S/P].

---

## 2. Verifying untrusted ML work

### 2.1 Proof-of-Learning (PoL) is broken for adversarial settings

- **The scheme:** Jia et al. (IEEE S&P 2021, [arXiv 2103.05633](https://arxiv.org/abs/2103.05633)) log checkpoints and data, and the verifier replays the top-Q largest updates within a tolerance δ.
- **Zhang et al. (S&P 2022)** forged proofs with adversarial examples.
- **Fang et al. (EuroS&P 2023, [arXiv 2208.03567](https://arxiv.org/abs/2208.03567)), largely the original authors, broke it more thoroughly:**
  - **Infinitesimal-update attack:** interpolate checkpoints with learning rate ≈ 0, so every replay passes δ at almost no cost.
  - **Blindfold top-Q attack:** place exactly Q genuine large updates where the verifier looks.
  - Their conclusion: provable robustness reduces to open problems in learning theory.
- **Lessons for CrowdGPT:**
  - Any tolerance band (needed for GPU non-determinism) is attack surface.
  - The verifier, not the prover, must pick what gets checked, using randomness revealed *after* commitment.
  - Update magnitude must be bounded by protocol, so tiny-norm forgeries contribute nothing.

### 2.2 Refereed delegation, bisection and determinism

- **Verde** (Gensyn, [arXiv 2502.19405](https://arxiv.org/abs/2502.19405), Feb 2025) [P]:
  - The job is given to ≥2 providers. They commit to Merkle trees over (iteration → operator inputs/outputs).
  - On disagreement, a two-level bisection (iteration, then operator) narrows it down, and the referee recomputes **one operator**.
  - Guarantee: correct if at least one provider is honest.
  - Requires **RepOps**, which fixes floating-point reduction order. Overhead: ~30–60% on matmuls above 2^10, ~60% on Llama-3.1-1B training on an A100, ~200% on small matmuls.
  - RepOps ships as closed binaries, and no independent evaluation exists as of Sep 2026 ([consortiumforverifiedcomputing atlas](https://consortiumforverifiedcomputing.org/atlas/gensyn-verde-repops)) [S].
  - Deployed for evaluation (Judge, Aug 2025) and inference, not for pretraining. A secondary source reports bit-identical results across consumer NVIDIA GPUs, x86/ARM CPUs and Apple M-series (Sep 2026) [S/U].
- **Optimistic verifiable training** (Srivastava, Arora, Boneh, NeurIPS 2024, [arXiv 2403.09603](https://arxiv.org/abs/2403.09603)) [P]: rounds to a coarser precision and logs rounding decisions, achieving cross-GPU replication (Titan XP, 2080 Ti, A40) inside a Teutsch–Reitwießner verification game. It notes that `torch.randn` parallelism breaks seed reproducibility, which matters directly for seeded ES.
- **Fraud proofs and bisection** in rollups (Arbitrum/Optimism) are the same pattern with a chain as referee. They need **(i)** deterministic execution, **(ii)** a challenge window, **(iii)** bonded or slashable identity. Without money, (iii) becomes reputation.

### 2.3 TOPLOC (inference-only)

- **What it is** ([arXiv 2501.16007](https://arxiv.org/abs/2501.16007), ICML 2025) [P]: the prover commits to top-k last-layer activations, 258 bytes per 32 tokens. The verifier does one batched prefill and checks within a tolerance.
- **Results reported:** 100% detection of model, prompt or precision changes in the authors' tests. Validation is up to 100× faster than generation.
- **Limits:** no independent audit as of Sep 2026. Known blind spots include speculative decoding with a cheaper model [S].
- **Use for CrowdGPT:** pattern only. "Commit to a locality-sensitive fingerprint of activations, verify with a forward pass" fits DUST's forward-only work (Section 2.6).

### 2.4 Scoring-based acceptance (what live pretraining networks actually use)

- **Gauntlet** (templar, [arXiv 2505.21684](https://arxiv.org/abs/2505.21684), NeurIPS 2025 workshop) [P]:
  - LossScore = L(θ, D) − L(θ − β·Δ_p, D), with β < learning rate.
  - **Proof of computation:** compare the improvement on the peer's *assigned* data (from a seed) with improvement on *random* data. Real training improves assigned data more, which catches copying and duplicate identities.
  - Fast checks for liveness and sync. OpenSkill (Plackett–Luce) ranking smooths noisy per-round scores.
  - Only the top-G peers are aggregated, with magnitude normalization and sign aggregation.
  - Known weakness, admitted by the authors: scores are inconsistent between adjacent rounds.
- **Psyche:** witnesses check that commitments (SHA-256 of results) were *received*, via Bloom filters with a 1% false-positive rate ([glossary](https://docs.psyche.network/explain/glossary.html)) [P]. The roles Trainer / Verifier / TieBreaker exist, but numeric-tolerance verification was still being explored (Jaccard, Hamming and other similarity metrics) [P/S].
- **Volunteer-computing prior art** [P, from memory, not re-fetched]:
  - BOINC's redundant computing with adaptive replication.
  - Sarmenta's credibility-based fault tolerance and spot-checking (*FGCS* 2002) formalized "spot-check plus reputation lowers the error rate exponentially in the number of checks passed".
  - This is the closest model to a no-money crowd. Recommended reading.

### 2.5 Byzantine-robust aggregation and its limits

- **Rules:** Krum ([arXiv 1703.02757](https://arxiv.org/abs/1703.02757)), coordinate-wise median and trimmed mean ([arXiv 1803.01498](https://arxiv.org/abs/1803.01498)), Bulyan, centered clipping ([arXiv 2012.10333](https://arxiv.org/abs/2012.10333)), bucketing for non-IID data ([arXiv 2006.09365](https://arxiv.org/abs/2006.09365)) [P, ids from memory].
- **Known breaks:**
  - "A Little Is Enough" ([arXiv 1902.06156](https://arxiv.org/abs/1902.06156)): small coordinated shifts inside the variance band defeat median and Krum.
  - "Fall of Empires" ([arXiv 1903.03936](https://arxiv.org/abs/1903.03936)): inner-product manipulation.
  - Under non-IID data, honest variance grows and the attacker's hiding room grows with it.
- **Implications:**
  - Robust aggregation bounds damage per round. It does **not** detect slow drift or backdoors.
  - Loss-on-held-out scoring does not detect backdoors either. A Nov 2026 survey found "no validated defense for decentralized LLM training" against data poisoning ([Turbin3](https://turbin3.org/blog/verifying-decentralized-training)) [S].
  - CrowdGPT therefore needs layered defenses: identity cost, verification of computation, clipping/robust aggregation, and post-hoc evaluation of the global model with automatic rollback.

### 2.6 Verifying zeroth-order / ES updates (key section)

**Weight-space ES with shared seeds.** These are all [P]:

- Salimans et al. 2017 ([arXiv 1703.03864](https://arxiv.org/abs/1703.03864)): workers share only scalar fitness values and rebuild every perturbation from seeds.
- MeZO ([arXiv 2305.17333](https://arxiv.org/abs/2305.17333)): in-place seeded SPSA.
- FedKSeed ([arXiv 2312.06353](https://arxiv.org/abs/2312.06353)): a pool of K seeds, so the whole update history is K scalars.
- **ES at Scale** (Qiu et al., ICML 2026, [arXiv 2509.24372](https://arxiv.org/abs/2509.24372)): full-parameter ES fine-tuning of models up to 72B, population ≈ 30, noise rebuilt from seeds.
- **EGGROLL** (Sarkar et al., ICML 2026, [arXiv 2511.16652](https://arxiv.org/abs/2511.16652)): rank-r perturbations AB^T from a counter-based RNG. Workers share only scalar fitness. Throughput reaches 91% of batch inference. Pretrains an int8 RNN language model with populations up to 262,144.
- **SeedFlood** ([arXiv 2602.18181](https://arxiv.org/abs/2602.18181), Feb 2026): floods (seed, scalar) pairs over a gossip graph. 16×OPT-1.3B used 400 KB of communication *in total*.

**Properties that follow.** Some of this is my own analysis [U]:

1. **The aggregation is exactly checkable.** Given the committed list of (seed_i, f_i), anyone can rebuild Δ = Σ f_i·ε(seed_i) bit-exactly, if the RNG and reduction order are fixed. Integer or fixed-point accumulation, as in EGGROLL's int8 path, removes float ambiguity. Orchestrators need no gradient upload to agree on the update.
2. **Each scalar is checkable with forward passes only.** To check f_i, run the same forward pass on (θ, data batch, seed_i). No backprop and no optimizer state are needed, so the cost is a fraction of training. The verifier's cost per sample equals the prover's, and sampling k of N members costs k/N. Forward-only also makes TOPLOC-style tolerance checks practical.
3. **Attack power is limited by construction.** A malicious worker can only choose scalars, not directions. The update is confined to the span of random directions fixed by the protocol, so steering toward a target direction g only achieves roughly the projection of g onto N random directions (about √(N/d)·‖g‖). Clipping or rank-normalizing scalars (as in OpenAI-ES) bounds this further. This turns "arbitrary vector poisoning" into "bounded scalar noise".
4. **Seeds must be unpredictable before commitment.** Derive them from a post-commit randomness beacon such as `H(round_beacon ‖ worker_id ‖ i)`. Otherwise a worker can grind seeds whose directions happen to align with a backdoor.

**DUST (activation / node perturbation) is different.** Q Labs adds Gaussian noise to each linear layer's output *per token*. A token's noise is rewarded by that token's loss change. The weight-gradient estimate is the outer product of the estimated output error with the layer *input* ([Q Labs](https://qlabs.sh/research/dust)) [P]. Consequences [U]:

- **Not seed+scalar reconstructible.** The estimate depends on the input activations x_s, so rebuilding it needs a forward pass over the worker's data. Communication cannot shrink to seeds and scalars the way EGGROLL's can. This is why v1.5 uploads weight-shaped estimates.
- **Still deterministic given (θ, batch_id, seed, kernel determinism).** A verifier can recompute any *sampled* (layer, token-range) slice with forward passes only. The per-token scalars c_s = (L(+a)_s − L(−a)_s)/2 can be committed as a Merkle tree, so a challenger can bisect to one token or one layer (Verde-style) without replaying everything.
- **Float sensitivity is the open question.** Antithetic differences of nearly equal losses are small numbers. The cross-GPU bf16 noise floor may be close to the signal for small σ. **Measure this first.** If it dominates, either use RepOps-style fixed-order kernels for the perturbed forward (paying 30–60%), or verify the weight-shaped estimate statistically (cosine with recomputation ≥ τ) instead of bit-exactly. The second option reopens the PoL tolerance-band problem.
- **Hybrid design** (recommended in Section 8):
  - Workers upload a compressed weight-shaped estimate plus a Merkle commitment to per-token c_s and to activation fingerprints.
  - Aggregation uses the uploads.
  - Verifiers spot-recompute sampled slices.
  - A weight-space ES lane (EGGROLL-style) can coexist for parameters where seed+scalar exactness is worth the population cost, such as norms, embeddings or LoRA-like adapters.

---

## 3. Consensus and ledger options for orchestrators

| Option | Fault model | Finality | Sybil resistance for validators | Cost / ops | Fit for CrowdGPT |
|---|---|---|---|---|---|
| Nakamoto proof of work | Byzantine, open | Probabilistic, minutes to hours | Wasted energy | Mining is pure waste. "Proof of useful work" (ML) is not cheaply verifiable (Section 2.1) | **No.** Its liveness ideal is right, but no viable useful-work puzzle exists |
| **Tendermint / CometBFT** ([arXiv 1807.04938](https://arxiv.org/abs/1807.04938)) | Byzantine, <1/3 voting power | Instant, ~seconds | Any weighting the app chooses (stake, *reputation*) | Mature Go stack. ABCI app in any language. ~100–200 validators practical [U from Cosmos experience] | **Best fit.** Training rounds last minutes, so latency is irrelevant. **Halts if >1/3 of voting power goes offline** |
| HotStuff / Jolteon | Byzantine | Fast, linear view change | Same | Fewer production-grade off-the-shelf stacks | Viable alternative |
| Narwhal/Bullshark, **Mysticeti** ([arXiv 2310.14821](https://arxiv.org/abs/2310.14821)) | Byzantine, DAG-based | Sub-second (Sui) | Same | High throughput, more complex | Overkill. Throughput is not the bottleneck |
| Raft | **Crash-only** | Fast | None | Simple | Only *inside* one orchestrator operator's cluster |
| Existing L1: **Solana** (Psyche) | Inherited | ~seconds | Inherited (SOL stake) | Needs SOL for fees. Program-upgrade keys are a central point. Psyche is still on devnet [S] | Fast to build. Adds a token dependency and a foreign governance dependency |
| Ethereum rollup (OP Stack: Gensyn, Base Sepolia: Prime) | Inherited plus sequencer | Minutes to fraud-window | Inherited | Gas in ETH. Sequencer is usually centralized | Same concerns, more expensive |
| Celestia DA | Data availability only | n/a | Inherited (TIA) | Pays per blob | Possible later for publishing round data |
| App-specific Cosmos SDK chain | As CometBFT | As CometBFT | App-defined | Most ops work. Full sovereignty | The CometBFT row, packaged |
| **CT-style log:** Trillian → **Tessera** / tlog-tiles, **Rekor v2** (GA 10 Oct 2025, [blog](https://blog.sigstore.dev/rekor-v2-ga)) | One log operator plus witness cosigning | Append-only, publicly checkable | n/a, not consensus | Static tiles behind a CDN, very cheap. Sigstore kept v1 as default for now ([Jun 2026](https://blog.sigstore.dev/rekor-evolution)) | **Use as the public audit mirror**, not as the decision layer |
| CRDTs + signed gossip (libp2p gossipsub v1.1 peer scoring, [arXiv 2007.02754](https://arxiv.org/abs/2007.02754); iroh-gossip) | Eventually consistent | None | Peer scoring only | Cheap | Fine for metrics and telemetry. **Cannot** decide "which update was applied" |

**The central tension: liveness versus finality.**

- Bitcoin survives with any non-zero hash power. BFT stops when more than 1/3 of the weight disappears.
- Prior art for recovery is Ethereum's **inactivity leak** [P, from memory]. Offline validators lose weight until the online ones regain 2/3, which works because Ethereum has a fork-choice chain that stays available.
- CometBFT has no built-in equivalent. A halted chain cannot process validator-set changes. Recovery needs an out-of-band "restart from the last finalized state root with the set that signs a restart message", which amounts to a coordinated restart.
- Design mitigations [U]:
  - Short validator epochs that drop inactive orchestrators *while the chain is still live*.
  - Caps on per-orchestrator voting power.
  - A written, scripted, deterministic re-genesis procedure that any surviving set can run.
  - Anchoring state roots to an external timestamp: OpenTimestamps or Bitcoin (free/cheap), or Rekor.

---

## 4. Storing and distributing checkpoints and deltas

**What others do** [P]:

- **Psyche:** iroh (QUIC, Ed25519 node IDs, ~90% hole-punch success claimed versus ~70% for libp2p), iroh-gossip (HyParView/PlumTree) for metadata, iroh-blobs for gradients. Checkpoints go to the Hugging Face Hub or peers. iroh reached **1.0 in Jul 2026** ([blog](https://www.iroh.computer/blog/the-road-to-iroh-1-0)).
- **Prime SHARDCAST:** sharded checkpoints on an HTTP relay tree, streamed through a pipeline, SHA-256 checked when reassembled. Relays keep the last ~5 versions.
- **templar:** each peer owns a **Cloudflare R2** bucket and posts read credentials on-chain. Uploads are time-stamped and auditable.
- **IOTA:** shared S3 through the orchestrator, which makes the orchestrator central.
- **Pluralis:** hivemind DHT plus direct RPC.

**Options:**

| Option | Strengths | Weaknesses |
|---|---|---|
| **iroh-blobs** (BLAKE3 verified streaming) | Content-addressed, resumable, verified at chunk level. Fits Rust orchestrators | Needs seeders. Relay fallback costs bandwidth |
| BitTorrent v2 (BEP 52, SHA-256 Merkle) | Proven swarm distribution of very large files. Many clients | Tracker/DHT churn. Peers expose IPs |
| IPFS (Kubo) | Content addressing, ecosystem | Content discovery is slow and unreliable for multi-GB files at volunteer scale [U, anecdotal]. Pinning cost |
| Hugging Face Hub (Xet chunk-level dedup) | Free public hosting, dedup of near-identical checkpoints | One company. Rate limits. ToS |
| S3 / R2 | R2 has no egress fees. Simple | Someone pays. Account-level central point |
| Reed–Solomon erasure coding across orchestrators | k-of-n durability without full replicas | More complex. Repair traffic |

**Sizing.**

- A 1B-parameter bf16 delta is ~2 GB. At 20 Mbps up that is ~13 minutes per upload, uncompressed [U arithmetic].
- At 27B MoE the full checkpoint is ~54 GB in bf16.
- So compression is required (SparseLoCo-class >100×, or seeds and scalars), and so is "catch-up by replaying signed small updates". Gauntlet stores sign-aggregated updates so late joiners can replay them instead of downloading checkpoints often.

---

## 5. Incentives and Sybil resistance

**With tokens, the failure modes are on record:**

- **Bittensor:**
  - Emissions attract weight-copying and gaming. SN9's winner-take-all design encouraged model hoarding ([IOTA primer](https://arxiv.org/html/2507.17766v1)) [P].
  - Governance disputes moved price by 15–18% in a day and drove the largest training team out [S].
- **Gensyn:** the $AI token launched Apr 2026 with 54.6% insider allocation. By Oct 2026 it traded near all-time lows, and the training product is paused [S].
- **Nous:** a token "under consideration" behind a $1B-valuation SAFT. The participant upside is undefined [S].
- **Risks of adding a token:**
  - Securities and regulatory exposure.
  - Mercenary participants who leave when the price falls.
  - Incentives to game the metric rather than improve the model.
  - Operator treasury and custody burden.
  - For this repo specifically, AGENTS.md §36.4 forbids any financial commitment without explicit approval.

**Without money, the prior art is:**

- Volunteer computing: BOINC credit, Folding@home points. Both are non-transferable credit plus leaderboards plus redundancy. Folding@home reached exascale in 2020 with no token [P, widely documented].
- Gauntlet's assigned-versus-random proof-of-computation, which needs no money.
- Rank-based reputation (OpenSkill) with decay.
- **Proof of personhood** (World ID, BrightID, Idena, Gitcoin/Human Passport):
  - Raises the cost of each Sybil identity.
  - Adds privacy and exclusion concerns, and dependence on a third party.
  - Best used only as an *optional* booster for validator admission or recognition, never as a requirement to contribute compute [U].
- **Proof of useful work as Sybil cost:** voting power grows only with *verified* contributions, accrues slowly (for example, square root or log of verified FLOPs), decays, and is capped per identity [U].
  - It costs a Sybil attacker real compute.
  - Doing that work also improves the model, which is a desirable property.

---

## 6. Scaling schedules: pivoting model size mid-run

**Growth operators:**

| Method | Type | Evidence |
|---|---|---|
| Net2Net ([arXiv 1511.05641](https://arxiv.org/abs/1511.05641)) | Function-preserving width and depth | Foundational |
| bert2BERT ([arXiv 2110.07143](https://arxiv.org/abs/2110.07143)), LiGO ([arXiv 2303.00980](https://arxiv.org/abs/2303.00980)) | Width/depth mapping, learned (LiGO) | BERT/GPT scale |
| **Masked Structural Growth (MSG)** ([arXiv 2305.02869](https://arxiv.org/abs/2305.02869)) | Strict function preservation via masks, grows width and depth | Up to ~1B |
| **Stacking Your Transformers (G_stack)** (Du et al., NeurIPS 2024, [arXiv 2405.15319](https://arxiv.org/abs/2405.15319)) | Depth stacking | **7B, 750B tokens. Reached the 300B-token scratch loss with 194B tokens (54.6% speedup)**. Gives guidelines for when and how much to grow [P] |
| **HyperCloning** (Apple, [arXiv 2409.12903](https://arxiv.org/abs/2409.12903)) | Function-preserving width expansion | Small to medium Llama-class |
| Zero/one-layer expansion (Bu, Meta, [arXiv 2511.04981](https://arxiv.org/abs/2511.04981)) | Depth, with hyperparameter transfer and LR-schedule guidance | Claims 3–5× compute efficiency on Llama3/DeepSeekV3 scaling fits [P] |
| Model families by progressive expansion ([arXiv 2504.00623](https://arxiv.org/abs/2504.00623)) | Train size i, then expand to i+1 | Directly matches a 1B→4B→9B ladder [P] |
| **Sparse upcycling** (Komatsuzaki et al., [arXiv 2212.05055](https://arxiv.org/abs/2212.05055)) | Dense → MoE: copy the MLP into E experts, add a router | Standard path to MoE |
| MoE orthogonal growth ([arXiv 2510.08008](https://arxiv.org/abs/2510.08008)) | Add layers or experts to a converged MoE. Interpositional copying beats stacking past Chinchilla-optimal tokens. Add noise to copied experts | Says growth at a fixed budget can match or beat training from scratch [P] |

**Caveats:**

- Some studies find growth gains fade when data is plentiful (survey, [NSF PAR](https://par.nsf.gov/servlets/purl/10650843)) [P].
- All of these results use **backprop**. None tests growth under DUST or ES [U]. One open question is whether newly added (zero-initialized) layers get usable DUST credit. In principle they should, because DUST credits each layer by the outer product of its output error and its *input*, and inputs to new layers are non-zero. This needs a test.
- Q Labs reports that larger models are *more* population-efficient. That is measured only up to 243M parameters [P/U].

**Activation mechanics: how chains pivot autonomously** [P, from memory]:

- **Bitcoin:** subsidy halving every 210,000 blocks, difficulty retarget every 2,016 blocks. Pure height rules, no vote.
- **Soft-fork activation:**
  - BIP9 version bits: 95% signaling over a 2016-block window, with a timeout.
  - BIP8: adds a mandatory lock-in at a height (LOT=true).
  - Taproot "Speedy Trial": 90% threshold, activated at block 709,632 in Nov 2021.
- **Ethereum** schedules forks by block height (later by timestamp) built into client releases.

**Applied to CrowdGPT** [U]:

- Put a **milestone table** in genesis. Each phase activates when *either*:
  - (i) a protocol-measured trigger is met (cumulative verified tokens ≥ T_k, or held-out loss slope below ε for W rounds) *and* ≥90% of orchestrator voting power signals that its software supports phase k+1, *or*
  - (ii) a hard height H_k is reached.
- The transition itself must be a **deterministic function** of the last finalized checkpoint and the genesis spec, for example G_stack ×2 depth then HyperCloning width, with a seed fixed by the beacon. Every orchestrator then computes the same grown checkpoint hash and votes on that hash.

---

## 7. Live metrics architectures

- **Psyche:**
  - `psyche.network` run pages are fed by an **indexer** over the on-chain coordinator. CI deploys an indexer and frontend for both devnet and mainnet previews ([CI docs](https://docs.psyche.network/development/ci.html)).
  - Clients have a TUI and JSON log modes.
  - Weakness: pages are JS-rendered and participant counts are not independently auditable. psyche.network was unreachable on 2026-10-07 [S].
- **Prime:** a hosted dashboard with per-run metrics and pinned charts (changelog, 23 Oct 2026). The INTELLECT-1 era had a public map of contributors. W&B exports are private [S].
- **Pluralis:** `dashboard.pluralis.ai` shows a global node map, loss, tokens/s, pipeline stage health, "implied model ownership", and a leaderboard [P].
- **templar:** validators set weights on-chain, and docs describe a metrics-logging pipeline. The page was 404 when checked, so details are unverified [U]. Many Bittensor subnets use W&B or Grafana [U].
- **Gensyn:** the RL Swarm dashboard showed a leaderboard and counts from on-chain identity. It is now read-only, with "historical data preserved on-chain" [P].

**Pattern to copy:**

- **Ledger events → deterministic indexer → read API / static JSON on a CDN → website.**
- Self-reported telemetry (Prometheus, OpenTelemetry) is collected separately and *labelled* "self-reported".
- Headline numbers come only from the ledger: verified tokens, contributor count, eval loss attested by ≥k orchestrators.

---

## 8. Recommendations for CrowdGPT v2

### 8.1 Recommended stack per layer

| Layer | Recommendation | Why (prior art) |
|---|---|---|
| **Training algorithm** | Keep DUST as the research lane. Make the backprop route (DiLoCo/SparseLoCo outer loop with streaming fragments) the **production fallback** until DUST shows ≥1B parameters and ≥10B tokens at matched compute. Hold milestone 1 until that measurement exists | Only backprop+DiLoCo-family has worked at scale (INTELLECT-1, Consilience, Covenant-72B). DUST evidence is ≤243M and extrapolated |
| **Determinism** | Counter-based RNG (Philox/Threefry) keyed by `H(beacon ‖ round ‖ worker ‖ batch)` for all noise and data selection. Pinned kernels and tokenizer. Fixed-order reductions for anything that must match bit-for-bit (aggregation in integer or fixed point). Bench cross-GPU noise floor of DUST's c_s **before** fixing tolerances | Salimans/EGGROLL seed reconstruction. RepOps. Srivastava et al.'s note on `torch.randn` non-determinism |
| **Contribution format** | Compressed weight-shaped estimate (top-k + low-bit + error feedback, SparseLoCo-style) **plus** a Merkle commitment to per-token antithetic scalars and activation fingerprints. Optional EGGROLL lane: (seed, scalar) only | Bandwidth (Section 4). Commitments make bisection possible (Verde). Bounded-influence property of seeded ES (Section 2.6) |
| **Verification** | Layered: (1) cheap checks for shape, norm and liveness. (2) Gauntlet-style held-out loss delta on assigned vs unassigned data. (3) **Optimistic acceptance with sampled forward-only replay** by ≥2 randomly chosen verifiers, picked after commitment. Challenges bisect to a layer or token range. (4) Exact recomputation of the aggregation by every orchestrator. Failed challenges cost reputation, not money | PoL breaks (verifier picks, post-commit randomness, norm bounds). Gauntlet. Verde. Sarmenta spot-checking |
| **Aggregation** | Per-contribution norm clipping, then centered clipping or trimmed mean across verified contributions, weighted by capped reputation. Global skip-step / auto-rollback if held-out loss jumps > kσ (OLMo-style) | Byzantine literature and its limits. Consilience rollback |
| **Orchestrator consensus** | **CometBFT app-chain**, no token. Voting power = capped, decaying function of verified useful work, plus a minimum age. Epoch-wise ejection of inactive validators. Scripted deterministic re-genesis for the >1/3-offline case. Orchestrator logic in Go or Rust, separate from the Python worker | BFT safety and instant finality fit minute-scale rounds. Avoids SOL/ETH dependencies and token risk (Bittensor, Gensyn lessons) |
| **Immutable public record** | Mirror every finalized block and round record into a **tlog-tiles (Tessera) log** served as static files, with C2SP witness cosignatures from independent orchestrators. Periodically anchor state roots with **OpenTimestamps** (Bitcoin) | Rekor v2/Tessera: cheap, CDN-friendly, publicly checkable. External anchor protects against a colluding validator set rewriting history |
| **P2P / data plane** | **iroh 1.0**: QUIC, hole-punching, iroh-gossip for control, iroh-blobs (BLAKE3) for deltas. Checkpoints mirrored to HF Hub plus R2/S3-compatible storage plus BitTorrent v2 magnets. Reed–Solomon across orchestrators for archival checkpoints | Proven in Psyche. Content addressing makes every artifact referable by hash from the ledger |
| **Incentives / Sybil** | No token. Non-transferable contribution credits on the ledger. Reputation-gated orchestrator admission. Optional proof of personhood only for public recognition | Folding@home/BOINC precedent. Token failure modes. §36.4 constraints |
| **Scale schedule** | Genesis milestone table (1B → 4B → 9B → … → 27B MoE). Triggers = verified tokens and loss plateau, plus ≥90% software-readiness signaling, plus a hard height fallback. Growth operator: G_stack depth + HyperCloning width for dense steps, sparse upcycling for MoE. All deterministic and hash-agreed | BIP9/BIP8/Speedy Trial. G_stack 54.6% speedup at 7B. Progressive model families. Sparse upcycling |
| **Live metrics** | Indexer over the ledger → static JSON/tiles on a CDN → website. Separate, labelled self-reported Prometheus/OTel feed. Eval loss computed by ≥3 orchestrators on a fixed held-out set, median published | Psyche indexer pattern. Gensyn read-only on-chain history. Avoids the Psyche "unauditable counts" critique |

### 8.2 Top 10 risks

1. **DUST does not scale to 1B+ parameters at useful compute efficiency.** All billion-scale decentralized pretraining evidence uses backprop. Mitigation: a gated go/no-go measurement before v2 build-out, with the backprop route ready.
2. **Floating-point non-determinism makes DUST verification either expensive (fixed-order kernels, +30–60%) or exploitable (tolerance bands, the PoL attacks).** Measure c_s noise across GPU types first.
3. **The BFT liveness cliff.** Losing more than 1/3 of orchestrator voting power halts the chain, which conflicts with "survives while any orchestrator exists". Re-genesis must be scripted, tested and permission-free.
4. **Sybil capture of the orchestrator set without stake.** Reputation earned by useful work can be farmed slowly by a well-resourced attacker. Needs caps, decay, minimum age and diversity limits.
5. **Undetected poisoning or backdoors.** Loss-delta scoring and robust aggregation do not detect targeted backdoors. No validated defense exists for decentralized LLM training. Needs data-provenance pinning (hash-pinned shards), canary and backdoor evals, and rollback.
6. **Bandwidth and hardware floor.** 1B bf16 deltas are ~2 GB, and 27B MoE checkpoints are ~54 GB. Without >100× compression or seed-scalar lanes, home uplinks exclude most volunteers (Covenant needed 8×B200 per peer).
7. **Free-riding and copying.** Peers can replay others' uploads or grind seeds. Needs commit-reveal, post-commit beacons, and assigned-versus-random data checks (Gauntlet).
8. **Governance and upgrade-key capture.** Psyche's single-key devnet programs and Bittensor's "triumvirate" show the pattern. Protocol upgrades must use the same signaling and activation path as milestones, with no admin key.
9. **Too little compute to be relevant.** Decentralized runs are about 1000× below the frontier. Volunteer churn is high (Gensyn swarms paused, Node0 ran 3 weeks). The schedule must be compute-adaptive (triggers based on verified tokens, not dates).
10. **Legal, data and financial exposure:**
    - Dataset licensing and illegal content.
    - Export-control or classification questions for model weights at larger sizes (see AGENTS.md skill 63).
    - Pressure to add a token, with securities risk and §36.4 prohibitions.
    - Someone must pay for mirrors and storage (R2/HF/S3 costs need explicit approval before provisioning).

### 8.3 Unverified items to confirm before relying on them

- Gensyn mainnet date (22 Apr 2026), $AI details, and cross-device bit-exactness claims (secondary sources only).
- How far Consilience trained (~1.2–1.25T tokens) and its paused status (secondary sources and the Covenant paper).
- templar's metrics pipeline (docs page 404).
- The Pluralis poisoning-defense preprint (seen only through a survey).
- arXiv ids given from memory in Sections 2.1, 2.5, 3 and 6. Re-check before citing externally.
- All [U] analysis in Section 2.6, including the bounded-influence argument and DUST reproducibility. It needs a small experiment.
