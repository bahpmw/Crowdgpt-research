# Open-source ecosystem sweep — Crowd-v1.5 / DUST volunteer training

Survey date: 2026-11-16. Scope: what already exists that we should borrow, fork, or interoperate with instead of rebuilding.
Everything below was confirmed from a page I actually fetched or from a search result that quoted the repo. Anything I could
not confirm is in **Unverified** at the bottom.

---

## 0. Preamble — what we already have, and therefore must NOT rebuild

I read the tree before writing this, so the "don't rebuild" list is grounded, not assumed. What exists today:

| Already built | Where | Consequence |
|---|---|---|
| Plain 1B transformer + DUST activation-perturbation estimator | `CrowdGPT/`, `client.py` | Do not import an ES library. Our perturbation axis is *tokens*, not weights. |
| DiLoCo outer optimizer + sparse payload + bf16 transport | `CrowdGPT/crowdgpt_ng/diloco.py` (`SparsePayload`, `OuterOptimizer`, `compress/average`) | Do not fork OpenDiLoCo/prime. Diff semantics, don't adopt code. |
| EGGROLL low-rank reference + step + benchmark | `CrowdGPT/crowdgpt_ng/eggroll.py` | Already a baseline. No need for `HyperscaleES`. |
| Wire protocol: ed25519 tickets/receipts, merkle roots + proofs, round certificates, checkpoint refs | `CrowdGPT/crowdgpt_ng/protocol.py` | This is *ahead* of every project in this survey. Keep it. |
| Loss-gated eval with shard manifests, bootstrap CI, uniform-logits control | `CrowdGPT/crowdgpt_ng/eval.py`, `loss_gate.py` | Right tool for per-round gating. Do not replace with lm-eval. |
| 1-bit blockwise codec (1024 params → fp16 scale + 1024 sign bits) on deltas, framed | `CrowdGPT-Server/orchestrator/compression.py` | Do not add a quantization dep for this. |
| Content-addressed weight blobs + HTTP `Range` resume + `.part` failover | `client.py` (`hf_weights/blobs/<role>__<sha256>__<size>.safetensors`, `Range: bytes=`) | **Item 4 of the brief is largely already built.** See §4. |
| Weight-signing / fingerprint checks | `orchestrator/fingerprints.py`, `sign.py` | Keep. |
| 119 tests incl. codec, cert, diloco, eggroll, loss-gate, ROCm defaults | `CrowdGPT/tests/test_ng_*.py` | Use as the oracle when borrowing anything below. |

**Licence constraint that governs every recommendation:** our repo is **GPL-3.0** (`CrowdGPT/LICENSE`). MIT, BSD-3 and
Apache-2.0 code can be pulled in and the result stays GPL-3. GPL-3 code can be pulled in. **Code with no licence
(DisTrO) cannot be copied at all**, and GPL-incompatible terms are a hard no. Every "take" below is licence-clean under
that rule unless I say otherwise.

**The two real gaps** (everything else is duplication): (a) published benchmark numbers, (b) a compute-matched
comparison harness.

*Correction (2026-11-16, after this sweep was written):* the third gap named here — error feedback — is **already
implemented**. `crowd_v15/codec.py` is stateless by design (it is a pure encoder/decoder), but the *client* carries the
quantization residual across rounds in `crowd_v15/feedback.py` (commit `2bf5142`, `--error-feedback`, 11 tests). It
recovers ~71 % of the accuracy the 1-bit codec gives up at the same payload: 30 identity rounds, tiny preset, final loss
3.7947 dense / 3.8863 `ngd1+ef` (15.8x smaller) versus 4.1121 for plain `ngd1` (16.4x smaller). So the borrow listed
below is a *cross-check*, not a missing feature.

---

## 1. Table first

| project | what it is | licence | maturity | take in ≤1 day? | why |
|---|---|---|---|---|---|
| [qlabs-eng/dust](https://github.com/qlabs-eng/dust) | official minimal DUST reference (607-line `dust.py`) | MIT | 5 commits, Oct 2026; NVIDIA bf16 only | **yes** | Cross-check our estimator; steal head-slab + credit-matrix tricks |
| [huskydoge/dust](https://github.com/huskydoge/dust) | DUST fork: AdamW baseline, DUST-under-Adam, backprop→DUST continuation | MIT | 12 commits, Oct 2026 | **yes** | Gives us the *missing* compute-matched Adam baseline |
| [qlabs-eng/noise](https://github.com/qlabs-eng/noise) | fixed-model/data/score speedrun benchmark, 3 tracks | (not shown) | active Oct–Nov 2026 | partial | Scoring harness + pinned-data hashing pattern |
| [PrimeIntellect-ai/OpenDiloco](https://github.com/PrimeIntellect-ai/OpenDiloco) | DiLoCo on hivemind + FSDP | Apache-2.0 | 596★, 43 commits, **unmaintained** | diff only | Outer-optimizer semantics vs our `OuterOptimizer` |
| [PrimeIntellect-ai/prime-diloco](https://github.com/PrimeIntellect-ai/prime-diloco) | production `prime`: ElasticDeviceMesh, int8 all-reduce | Apache-2.0 | 312 commits, active | no | Too heavy; borrow the int8-allreduce *idea* only |
| [learning-at-home/hivemind](https://github.com/learning-at-home/hivemind) | DHT, fault-tolerant averaging, `hivemind.Optimizer`, compression | MIT | 1.1.12 Jan 2026, 599 commits | partial | `load_state_from_peers` = origin offload; timeout semantics |
| [bigscience-workshop/petals](https://github.com/bigscience-workshop/petals) | block-sharded serving over a swarm | MIT | 522 commits, v2.2.0 | partial | Per-block checkpoint layout; reroute-on-failure |
| [NousResearch/DisTrO](https://github.com/NousResearch/DisTrO) | paper + README only (no code) | **none** | 15 commits, 1051★ | no | Read the PDF; **do not copy** (no licence) |
| [gensyn-ai/rl-swarm](https://github.com/gensyn-ai/rl-swarm) | RL swarms over the internet | MIT | 1.7k★, 240 commits | no | Different problem (RL, not pretraining) |
| [yandex-research/swarm](https://github.com/yandex-research/swarm) | SWARM pipeline parallelism | (not shown) | 3 commits, stale | no | We run whole model per node; no pipeline |
| [pytorch `powerSGD_hook`](https://docs.pytorch.org/docs/main/ddp_comm_hooks.html) | PowerSGD incl. error feedback, warm start | BSD-3 (torch) | in-repo, stable | **yes** | Cross-check our error feedback (`crowd_v15/feedback.py`); borrow *warm start*, which we still lack |
| [epfml/powersgd](https://github.com/epfml/powersgd) | reference PowerSGD, ~200 lines | MIT | 59 commits | yes | Readable version of the same algorithm |
| [DeepSpeed 1-bit Adam/LAMB](https://www.deepspeed.ai/tutorials/onebit-adam) | 1-bit compressed momentum w/ error compensation | Apache-2.0 | **code removed Sep 2026** | docs only | Momentum-mask gotcha; code is gone from master |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | blockwise k-bit quant, 8-bit optimizers | MIT | 0.50.2, very active | no | 69 MB wheel + CUDA kernels; we have no optimizer state |
| [pytorch/ao (torchao)](https://github.com/pytorch/ao) | int4/int8/float8 weight-only quant | BSD-3 | very active | maybe | Only to *serve* 4-bit init weights to 16 GB nodes |
| [safetensors](https://github.com/huggingface/safetensors) | zero-copy tensor container | Apache-2.0 | 3.8k★, mature | already used | `get_slice` for range-friendly shards; UMA mmap gotcha |
| [hf_xet / Xet](https://huggingface.co/docs/huggingface_hub/main/guides/download) | chunk dedupe transfer (`hf_transfer` retired) | Apache-2.0 | default since hub 0.32 | partial | Chunk-hash dedupe for near-identical round weights |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 60+ benchmarks, pluggable backends | MIT | 14.4k★, v0.4.13 Aug 2026 | **yes** | Citable final numbers without writing benchmarks |
| [CyberAgentAILab/cmaes](https://github.com/CyberAgentAILab/cmaes) | CMA-ES, ask/tell, pure numpy | MIT | v0.13.1 Aug 2026, ~450★ | maybe | Tune our 2L scalar ES / hyperparams |
| [RobertTLange/evosax](https://github.com/RobertTLange/evosax) | 30+ ES in JAX | Apache-2.0 | v0.3.2 Oct 2026 | no | Wrong perturbation space; JAX rewrite |
| [nnaisense/evotorch](https://github.com/nnaisense/evotorch) | EC library on PyTorch | Apache-2.0 | 1.1k★, v0.6.1 May 2025 | no | Weight-space ES primitives we don't use |
| [facebookresearch/nevergrad](https://github.com/facebookresearch/nevergrad) | derivative-free optimisation toolbox | MIT | 4.2k★, 1.0.12 Apr 2025 | no | HPO tool, not a training method |
| [openai/evolution-strategies-starter](https://github.com/openai/evolution-strategies-starter) | distributed ES, master/worker | MIT | **archived Apr 2026**, 1.6k★ | read only | 200 lines of ES math worth reading, nothing to run |
| [ESHyperscale / EGGROLL](https://eshyperscale.github.io/) | low-rank ES at scale | site repo **GPL-3.0** | ICML 2026 | no | We already have `eggroll.py` |
| [pytorch/torchtitan](https://github.com/pytorch/torchtitan) | FSDP2/TP/PP reference training | BSD-3 | very active | no | Backprop-native; our stack has no backward pass |

---

## 2. Zeroth-order / DUST side

### 2.1 `qlabs-eng/dust` — MIT, 607-line reference
The one project that *already does DUST*. Read it before inventing anything else. Four concrete pieces I would lift
into our estimator within a day, all visible in the published source:

1. **Head jitter on cached logits, one 256-column slab at a time.** Their `head_error` path perturbs the LM head
   directly on cached logits in column slabs instead of materialising a `[tokens, vocab]` noise tensor. At our 1B scale
   with a large vocab this is the difference between fitting on a 16 GB card and not. This is the single highest-value
   copy in the whole survey.
2. **The attention credit matrix.** They build `credit[t, s] = ((lag >= 0) & (lag <= window)) * gamma ** lag` — a
   per-layer decay that assigns credit from a perturbation at token `t` to the loss at token `s`, then reuse it across
   all draws for that block. Our version should be compared numerically against theirs on one batch.
3. **Per-layer-type draw allocation (`Recipe` dataclass).** Direct-loss draws, attention-internal draws and head draws
   are separate budgets, with separate tuned defaults per population size. Their population ladder is explicit:
   256/1024 → 1 GPU, 4096 → 4 GPUs, 16384 → 8 GPUs. Useful sanity anchors for our coordinator's expected client count.
4. **Two-sided weight-space ES for the `2L` residual scalars only** (`resid_lambdas`, `x0_lambdas`), everything else
   activation-perturbed. If our 1B has residual-mix scalars, confirm we do the same rather than sampling them in
   activation space.

Cost: none beyond reading — MIT, no install, and the file has no dependency beyond torch. Caveat from their own README:
"preserves the paper's estimator and tuned defaults while omitting execution optimizations" — so it is a *correctness*
reference, not a speed reference. It also assumes NVIDIA bf16 (`--population` defaults to 16384 on 8 GPUs); it will run
on our ROCm APU only by forcing the fp32 path.

Note the model it trains is not a vanilla GPT: value embeddings on alternating blocks, `SSSL` window pattern, squared-ReLU
MLP, untied head, logits soft-capped at 15. If our "plain transformer" diverges from that, our DUST numbers are not
comparable to the paper's, and that matters more than any code we take.

### 2.2 `huskydoge/dust` — MIT fork, adds exactly the baselines the paper omits
Independent reproduction (37.7M params, ≤10M tokens, populations to 16384) that adds `dust_adam.py`,
`dust_cpt.py` (continuation from a backprop checkpoint), `baselines/backprop_adam.py`, and a written report. Two uses:
the **AdamW baseline** (the official release only ships SGD — our comparisons are SGD-only without this), and the
**DUST-on-a-backprop-checkpoint (CPT) protocol**, which is the cheapest way to test "can we keep training an existing
1B checkpoint with DUST" before we commit a volunteer round to it. Their report also documents the failure mode we
should expect: the DUST/backprop gap widens with steps and a pre-trained checkpoint needs a smaller LR.

### 2.3 `qlabs-eng/noise` — the compute-matched comparison harness we were going to write
Fixed model (37.7M), fixed data (30M FineWeb tokens, pinned hashes), fixed score (val cross-entropy), three tracks
(val loss / speedrun to a target loss / open architecture), and a hard rule "no autograd, no hand-written backward".
Each run writes `config.json`, `metrics.jsonl`, `final.pt`, `result.json`; `utils/eval.py` rescores any checkpoint.
That conventions set is worth copying wholesale into `crowd-v15/bench/` — it makes "tokens-to-target-loss" curves
directly comparable to a published leaderboard, which is the thing our internal tests cannot give us.
Licence was not shown on the page I fetched — treat as **read-for-method, do not copy files** until checked.

### 2.4 ES libraries — mostly the wrong axis
`evosax` (Apache-2.0, JAX), `evotorch` (Apache-2.0, PyTorch), `nevergrad` (MIT), `openai/evolution-strategies-starter`
(MIT, archived) all optimise in **weight space**. Our perturbation lives in activation space, where a token is a
population member, so their sampling/update machinery does not transfer. The only weight-space ES we run is the `2L`
residual scalars — a `2L`-dimensional problem where 20 lines of antithetic sampling beats any dependency.
`cmaes` (MIT, pure numpy, no torch) is the one exception worth keeping in the toolbox, for tuning perturbation scale
and per-layer draw allocation rather than for training.
`openai/evolution-strategies-starter` is still worth 15 minutes for its rank-shaping + `max_unroll` guidance, and
`hardmaru/estool/es.py` (525 lines, single file) is the readable transcription of the same math.

### 2.5 EGGROLL — already ours
`ESHyperscale/HyperscaleES` and `nano-egg` are the canonical implementations; the project page repo is GPL-3.0
(compatible with us), but **check each code repo's own licence before copying a file** — the page licence does not
automatically cover them. We already have `crowdgpt_ng/eggroll.py` with `eggroll_step` and a benchmark, so the useful
borrow is their single-file **pure-int8 pretraining** implementation (their release notes describe it explicitly),
which is directly relevant to a 1-bit/int8 wire format.

---

## 3. Compression / codec — the real gap is semantics, not bits

### 3.1 PyTorch `powerSGD_hook` (BSD-3) and `epfml/powersgd` (MIT) — borrow the bookkeeping, not the rank truncation
PowerSGD itself is wrong for us (we send a weight-shaped estimate, not a gradient, and low-rank truncation would fight
the DUST update direction). But two of its mechanisms are exactly what a stateless 1-bit codec is missing:

- **Error feedback:** `state.error_dict[bucket] = input_before - input_after`, added back on the next step. DeepSpeed's
  1-bit Adam paper calls the same thing "error compensation" and proves the accumulated error cancels — *for
  linear-in-gradient optimizers only*. We should measure, not assume, whether it helps our estimator, because the DUST
  update is not a gradient and error feedback can bias a noise-based estimator toward stale directions.
- **Warm start:** reuse the previous step's `p`/`q` memory so the first iteration of the next round is not from scratch.
  Our analogue is reusing the previous round's scale vector per block instead of recomputing fp16 absmax from zero.
- **`min_compression_rate` gate:** PowerSGD skips compression for tensors where it would not pay. We already do a
  version of this in `diloco._budget_per_tensor`; worth aligning the threshold semantics so both paths agree.

Cost: PyTorch built-in — zero install. The reference repo is ~200 lines and MIT if we want a version without DDP hooks.

### 3.2 DeepSpeed 1-bit Adam/LAMB — documentation only, code is gone
The tutorials are still up, but the implementation was **removed from DeepSpeed master** (cleanup PR merged 2026-09-12:
compression library and 1-bit/0-1 Adam/LAMB deleted). So: read the two tutorials, do not add DeepSpeed as a dependency
hoping to import these. The one transferable gotcha is the **momentum mask**: 1-bit compression cannot represent exact
zero, so parameters whose gradients are permanently zero (e.g. embedding rows beyond the trained sequence length)
accumulate phantom error forever. Our 1B almost certainly has such rows in `wte`/value embeddings; a per-block "known
zero" mask is a one-afternoon fix and prevents a silent quality leak.

### 3.3 bitsandbytes (MIT) / torchao (BSD-3) — not worth it for the codec
bitsandbytes' blockwise absmax codec is close to ours, but the wheel went from 122 MB to 69 MB and it still ships CUDA
kernels; ROCm support is partial and version-coupled to torch. We have zero optimizer state, which is bitsandbytes' main
selling point, so the dependency buys nothing.
torchao has exactly one relevant job: serving **int4/int8 weight-only** copies so a 16 GB volunteer can *initialize* a
node fast. That must be dequantized to bf16 before the DUST forward pass, or the perturbation scale is wrong. Treat as
a serving-only optimization, not a training path.

---

## 4. Serving / weight distribution — mostly already solved, two borrows left

Reality check against our own client: we already use **safetensors blobs named by sha256 and size**, HTTP `Range` with
resume, `.part` files, and per-role (model vs engram) grouping. That covers the bulk of "don't hammer the origin" for a
single origin. The remaining three mechanisms are worth a day each:

### 4.1 `hivemind.load_state_from_peers()` (MIT)
The cheapest origin-offload that exists: a joining volunteer fetches state from a *random peer* already in the run,
not from the coordinator. `hivemind.Optimizer` exposes it directly, and its `matchmaking_time` / `averaging_timeout`
pair is a ready-made vocabulary for "wait this long for peers, then proceed" that maps onto our round deadline.
We already have `hivemind_bridge.py`, so this is a code path, not a dependency.
Caveat before adopting: hivemind pulls in a libp2p daemon binary and has historically been version-pinned against torch
(the OpenDiLoCo README pins both torch and hivemind, and the Decoupled DiLoCo fork warns the tree has conflicts). Only
take this if peer-to-peer downloading is a real requirement.

### 4.2 safetensors `get_slice` (Apache-2.0) — shard for range friendliness
`safe_open(...).get_slice("embedding")` lets a client materialise `[i:j]` of a tensor without scanning the file, which
is what makes a *partial* download resumable at tensor granularity rather than byte granularity. Combined with our
existing content addressing, a volunteer that drops at 80% can re-fetch only the missing shards.
**Gotcha for our ROCm APU box:** `safe_open` always mmaps; on unified-memory devices (Apple Silicon, Grace/Blackwell,
some AMD APUs) that doubles peak memory and can OOM a checkpoint that fits (safetensors issue #758, closed 2026-05).
For any UMA node, stream instead of mmapping.

### 4.3 Xet (`hf_xet`) — content-defined chunk dedupe
`hf_transfer` is retired: `huggingface_hub` ≥ 0.32 installs `hf_xet`, and 1.0 removed `hf_transfer` support entirely
(`HF_HUB_ENABLE_HF_TRANSFER` is now ignored). Xet splits files into immutable chunks and serves them by hash, with
`HF_XET_NUM_CONCURRENT_RANGE_GETS` (default 16) controlling parallelism. For us the interesting property is dedupe:
consecutive round checkpoints are near-identical, so hash-addressed chunks would let a volunteer pull only what changed.
We are not going to run a Xet server, but the *pattern* — chunk by content hash, advertise the manifest, fetch missing
chunks — is the right next step past per-file `Range` and can be implemented on top of our blob store.

### 4.4 Petals (MIT) — two patterns, not the codebase
(1) They ship a **converted model repo** (`bigscience/bloom-petals`) with per-block files so a server downloads only the
blocks it hosts. Our equivalent is already per-role safetensors; the improvement is per-*layer* granularity so a
volunteer with 16 GB can host a slice. (2) Their fault tolerance is "client keeps intermediate activations and reroutes
the block to another server hosting the same block". Our analogue is retrying a submission at another coordinator;
the reusable idea is that the *client* holds the retry state, not the server.
Petals does not train from scratch (it explicitly says so and points at hivemind), so there is no training code to take.

### 4.5 torchtitan (BSD-3) — no
Clean FSDP2/TP/PP reference, all of it predicated on backward passes and optimizer state we deliberately do not have.
Nothing here applies.

---

## 5. Evaluation harnesses

### 5.1 `lm-evaluation-harness` (MIT) — adopt for *published* numbers only
We should not replace `eval.py`: our loss gate is per-round, cheap, and tied to signed `EvalResult` records, which is
exactly what lm-eval cannot do. What lm-eval gives us is **citable, reproducible few-shot benchmark numbers** for the
final model. The ≤1-day path is a custom backend: subclass `lm_eval.api.model.LM` and wrap our checkpoint — that is the
documented escape hatch and roughly 50 lines (`loglikelihood`, `loglikelihood_rolling`, `generate_until`).
Two costs to plan for: since v0.4.12/0.4.13 the base install **excludes torch and transformers** (install
`lm_eval[hf]`), and a full task group run takes hours on one GPU — pick two or three tasks (ARC-easy, HellaSwag, LAMBADA)
rather than everything. At 1B, a handful of tasks is enough to say "this is a working LM", which is the claim we need.

### 5.2 Compute-matched comparison — use `noise` conventions (§2.3), not `nanogpt`
The correct comparison is **tokens-to-target-loss at matched compute**, not loss-at-step. The `noise` repo's fixed
data/model/score plus `result.json` + `utils/eval.py` rescorer is the smallest thing that produces a comparable curve.
Our own `eval.evaluate` with bootstrap CI is a good measurement layer *underneath* that; the missing piece is only the
pinned dataset + a documented compute budget per point.

### 5.3 Tokenizer
The DUST release ships a **fixed BPE-4096 tokenizer with a training recipe** (`tokenizer/`). If our vocab differs, every
loss number we report is incomparable to the paper — worth checking before any benchmark work. `tiktoken` is not needed;
we already depend on `transformers`.

---

## 6. Do not bother (and why)

- **Forking `prime` / OpenDiLoCo.** Different topology (peer-to-peer all-reduce, FSDP-over-many-GPUs), unmaintained
  upstream or 300-commit heavyweight, and we already implement the pieces we need in `diloco.py`. Diff the outer
  optimizer math, take nothing else.
- **`DisTrO`.** The repo contains a README and a PDF — no code. It also has **no licence**, so nothing in it is
  copyable. Read the preliminary report for the "stateless client" idea; the production implementation lives in Psyche
  (`shared/modeling/src/distro.rs`, Rust).
- **`SWARM` / `torchtitan` / pipeline parallelism.** All three assume sharding a model across nodes. Our model fits on
  one 16 GB card; pipeline parallelism would add failure modes for zero benefit.
- **`gensyn rl-swarm`.** RL post-training with GRPO, on-chain identity. Not our problem, and it drags in web3 deps.
- **`evosax` / `evotorch` / `nevergrad` as training libraries.** Wrong perturbation space (weights, not activations).
  `evosax` additionally forces a JAX rewrite; `evotorch`'s value is Ray scheduling we don't need.
- **`openai/evolution-strategies-starter`.** Archived April 2026, EC2/Packer/MuJoCo setup, spot-instance orchestration.
  Read `es.py`, don't run it.
- **bitsandbytes.** 69 MB wheel, CUDA kernels, partial ROCm — for an optimizer-state problem we don't have.
- **DeepSpeed.** 1-bit Adam/LAMB were deleted from master in Sep 2026; adopting DeepSpeed now means adopting the whole
  engine to reach code that no longer exists.
- **Building our own benchmark suite.** lm-eval is MIT, mature, and citable. Our time is better spent on the estimator.
- **`torchao` for training.** Only defensible as a "download 4-bit init weights" serving trick.

---

## 7. Unverified

- **`gresyn-ai/noloco` licence and API.** Gensyn's NoLoCo (gossip instead of all-reduce, arXiv 2506.10911, repo
  `github.com/gensyn-ai/noloco`) is interesting if we ever remove the coordinator, but I did not fetch the repo — licence
  and completeness unknown.
- **`yandex-research/swarm` licence.** Repo page I fetched showed no licence line; 3 commits, stale, prototype only.
- **`qlabs-eng/noise` licence.** Page did not show one. Method-level borrowing only until checked.
- **`ESHyperscale/HyperscaleES` and `nano-egg` licences.** The project *page* repo is GPL-3.0; that does not establish
  the code repos' licences. Verify per repo before copying.
- **`hardmaru/estool` licence.** Cited from a search result (`es.py`, 525 lines) without confirming the licence file.
- **`torchao` ROCm/APU behaviour.** Docs emphasise CUDA and XPU paths; int4 tinygemm on our 30 GB ROCm APU is untested
  by me, and the `int4_packing_format` defaults differ per platform.
- **No other DUST reimplementation found.** GitHub searches for "DUST pretraining", "node perturbation transformer",
  "activation perturbation" surfaced only `qlabs-eng/dust`, `qlabs-eng/noise` and `huskydoge/dust`. If a third-party
  DUST fork exists, it is not indexed under those terms.
- **"Together's `open-diloco`"** from the brief does not appear to exist — the `OpenDiloco` repo is
  `PrimeIntellect-ai/OpenDiloco`. Together's public org shows API clients and cookbooks, no Diloco repo.
