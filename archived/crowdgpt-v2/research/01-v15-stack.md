DRAFT — requires human review

# 01 — Crowd-v1.5 stack analysis (input to CrowdGPT v2 scoping)

Scope: working tree of `/Users/patwoodward/Documents/docker/opencode/projects/Crowd-v1.5` on 2026-11-17, including uncommitted edits (`README.md`, `client.py`, `coordinator.py`, `dust.py`, `server/app.py`, `store.py`, `tools/prepare_tokens.py`) and untracked `audit.py`, `tests/test_{audit,shards,time_to_target}.py`, `tools/{bench_alignment,bench_audit,time_to_target}.py`. Test status: `.venv/bin/python -m pytest -q -x` → **173 passed in 21.1 s** (README still says 89/119).

## Key findings (summary)

1. The audit is the only real anti-poisoning check, and it skips (or rejects under `audit_require`) real-token `copy` training, accelerator-generated noise, error feedback, draw averaging / local steps, and cross-device bf16. The fast path and the trusted path are currently mutually exclusive.
2. The cosine ≥ 0.5 threshold permits an attacker component up to ~1.73× the honest estimate's norm; honest estimates align with the true gradient at only cos 0.02–0.13. The audit is scale-invariant, so magnitude inflation is bounded only by `max_abs_grad` (default 100).
3. Client identity is a self-chosen unsigned string; two fake ids meet the default quorum of 2; equal-weight averaging. Clients choose their own noise seed and data cursor (grinding).
4. README/code mismatches: README says token-weighted, `finalize` averages equally; no `200m` preset in code.
5. Updates are not reproducible across replicas (in-memory momentum, arrival-order-dependent sums, digest tolerant to sub-bf16 differences); records thin (unsigned jsonl, client *count* only, last 50 rounds).
6. HF baton is single-writer, check-then-commit race (no `parent_commit`), unverified pushes, throughput ≤ one GPU.
7. Scaling blockers: full fp32 grad buffer per client, full fp32 downloads per round, ~127 MB upload at 1B (ngd1), coordinator reloads weights on every submit and `/status`; `plain.py` bans moe/expert/router names (conflicts with MoE goal); ~7× population needed at 1B to match 10m alignment.
8. Throughput evidence weak: unpublished `runs/ttt` — Adam 10m eval 5.31 on 2M tokens in 22 s vs DUST pop 8 eval 7.21 on 147k tokens in 116 s.
9. Keep: codec + hardened parser, framing, partial-layer re-scoring, sharded downloads, measurement tools. Rebuild: coordinator, store, baton.

## 1. Architecture overview

Components (`crowd_v15/`):

| module | role |
|---|---|
| `model.py` | plain GPT (RMSNorm, fused-QKV attn, SwiGLU, tied embeddings); every hidden linear routed through `_apply_lin` so noise can be injected and inputs recorded (`model.py:172-193`) |
| `dust.py` | DUST estimator `dust_gradients` (`dust.py:162-278`), `DustOptimizer`, `probe_cosine` vs backprop |
| `tasks.py` | identity / induction / random synthetic batches + flat token file `token_batch` |
| `codec.py` / `protocol.py` | NGD1/NGD1R/dense containers; `<u32 len><JSON header><container>` framing |
| `feedback.py` | client-side error feedback for the biased 1-bit codec |
| `coordinator.py` | accept/reject, optional audit, quorum finalize, (outer) momentum |
| `audit.py` | salt-selected partial re-score of a DUST submission |
| `store.py` | atomic safetensors + `status.json` + `submissions.jsonl`, shard plan |
| `server/app.py` | FastAPI surface |
| `client.py` | sync → train → encode → submit loop |
| `accel.py` | CUDA/ROCm/MPS/CPU selection, sync, memory |
| `tools/hf_baton.py` | serverless mode via a Hugging Face repo |

One round end to end:

1. **Fetch.** `DustClient.sync` (`client.py:185-219`) GETs `/model/manifest` then `/status` (round id), downloads weights shard-by-shard with `Range` + per-shard sha256 and 3 retries (`client.py:221-242`), recomputes `weights_digest` and compares to `manifest.baseSha256` (`client.py:204-208`), runs `check_plain_parameter_names` (`client.py:209`), copies into the local model. Weights are served as **fp32** (`store.py:80`) — full re-download every round (`client.py:372`).
2. **Compute.** `client.gradient` (`client.py:276-288`): batch = `client_batch(task, seed, round_id, cursor)` (`client.py:121-143`); `base_seed = seed + round_id*1_000_003` (`client.py:279`) — **seed chosen by the client**. `dust_gradients`: clean pass records layer inputs; per perturbed layer (`mode="layer"`) antithetic ± passes with noise `derive_seed(base_seed, step, draw, layer)` (`dust.py:76-81`); reward = reverse discounted cumsum of per-token loss deltas; `G_W += (noise·reward)ᵀ x / (σ·tokens)` (`dust.py:235-261`); head gradient computed **exactly** from softmax (`dust.py:227-232`). Allocates a full fp32 gradient buffer for every parameter (`dust.py:200-201`). Optional `--draws k` averaging, `--local-steps`, error feedback (`client.py:350-398`).
3. **Submit.** `encode` flattens sorted names (`protocol.py:56-72`), optional EF (`client.py:314-318`), builds header with estimator fields + `replay` block (`client.py:260-274, 319-345`), POST `/submit/dust`.
4. **Validate.** `app.submit` peeks header route and requires URL==header route (`app.py:134-145`); `Coordinator.submit` **reloads the whole weight file from disk** (`coordinator.py:198`), `decode_body` (size/shape/finite guards), `_check_against_model` (names/sizes, `coordinator.py:222-231`), `_accept` (`coordinator.py:250-327`): route open, `roundId == current`, `baseSha256 == current`, no route mixing, one submission per `clientId`, DUST needs `tokens ≥ min_tokens`, `population ≥ 1`, `sigma > 0`; vector normalised to unit direction, norm kept; optional `_audit` (`coordinator.py:233-248`); append to `submissions.jsonl`.
5. **Aggregate.** When `acceptedThisRound ≥ min_submissions` (default 2), `finalize` (`coordinator.py:335-424`) averages unit directions **with equal weight** (`:349-351`), multiplies by `min(mean declared norm, max_abs_grad)` (`:352-354`), optional global clip, then plain SGD/momentum or DiLoCo-style Nesterov outer step (`:366-389`).
6. **Publish.** `store.save_weights` atomic temp+`os.replace` (`store.py:75-91`), new digest, round+1, round record appended to `history` (kept to last 50, `coordinator.py:415`), `status.json` rewritten. Clients see the new round on their next `/status`.

Concurrency: the `async` submit handlers call the synchronous coordinator inline (`app.py:200-206`), so decode + audit + finalize block the event loop — this incidentally serialises state mutation, but one audit stalls every request.

## 2. Protocol & wire formats

- **Framing** (`protocol.py:117-129`): `struct "<I"` header length, compact sorted-key JSON header (≤64 KiB, `:48`), then container. Header fields (`protocol.py:101-113`, `client.py:319-345`): `schema="crowd-v15-submit-1"`, `route ∈ {dust, backprop}`, `clientId`, `roundId`, `baseSha256`, `codec`, `names`, `sizes`, `params`, `clientVersion`, plus `tokens, population, sigma, gamma, draws, baseSeed, seed, step, localSteps, averagedDraws, errorFeedback, residualNorm, replay{preset, task, batch, seqLen, dataCursor, computeDtype, sigma, sigmaMode, sigmaRel, gamma, mode, noiseDevice, population, drawChunk, device}`.
- **Header guards** (`protocol.py:183-210`): schema, route, codec, equal-length names/sizes, unique non-empty names, positive int sizes, `params == sum(sizes) ≤ model params`, 64-char digest, non-negative round, non-empty clientId. Estimator fields read defensively (`:213-227`). Replay fields are **not** validated or bounded anywhere.
- **Containers** (`codec.py:18-25`): `dense` = raw bf16 RNE (`:244-272`); `ngd1` = zlib(`NGD1|u64 n|u32 block|fp16 scale×nb|sign bits`), 1024-value blocks, scale = mean |x|; `ngd1r` adds residual scale + residual sign plane. Frames of ≤4 M values (`protocol.py:49`) wrapped in `NGD2`/`NGD3`. Decoder is hardened: expected raw size precomputed, bounded inflate, rejects trailing/oversized/truncated data, non-finite scales, container/codec mismatch (`codec.py:115-237`), final NaN/Inf check (`protocol.py:178`).
- **Sizes**: ngd1 ≈ 1 bit + 16 bits/1024 per value; measured 16.4× (ngd1) / 8.1× (ngd1r) smaller than dense on the tiny preset (README:193-199); 10m e2e 1.70 MB vs 14.69 MB dense (README:371). At 1B: dense ≈2.03 GB, ngd1 ≈127 MB, ngd1r ≈254 MB, low-rank r=8 ≈10 MB (not on wire) (`research/README.md:61-66`). `max_body_bytes` = 1 GiB (`coordinator.py:65`), so a dense 1B submission is refused by default.
- **Binding**: `baseSha256 = sha256(Σ name‖0x00‖bf16(fp32 values))` sorted by name (`store.py:38-47`). Deliberately invariant to dtype/container — so **two different fp32 weight files that round to the same bf16 share a digest** (tested in `tests/test_baton.py:55`). Binding is to round + digest only; there is **no client signature, no nonce, no binding of the body to the clientId**.
- **Seeds**: noise is a pure function of `(baseSeed, step, draw, layer)` via blake2b (`dust.py:76-81`), CPU generator by default for cross-device bit-equality (`dust.py:84-95`); `noise_device="device"` breaks this (`dust.py:48-55`). Batch seed = blake2b(`seed:round:batch`) (`client.py:129-134`). Both seed and data cursor are client-chosen.

## 3. Trust / verification mechanisms

What is checked today:

| check | where | defeats |
|---|---|---|
| body size, header length, JSON, decode bounds, finite values | `app.py:81-90`, `protocol.py:132-210`, `codec.py:115-237` | zip bombs, malformed/oversized uploads, NaN injection |
| URL route == header route | `app.py:140-145` | relabelling a DUST estimate as exact backprop |
| layout equals served model | `coordinator.py:222-231` | shape confusion |
| round + base digest | `coordinator.py:255-260` | stale / off-point gradients, replay into later rounds |
| one per clientId per round, no route mixing | `coordinator.py:261-272` | duplicate submission under the *same* id |
| estimator sanity (tokens, population, sigma) | `coordinator.py:274-280` | trivially impossible records |
| unit-normalise + cap magnitude | `coordinator.py:286-290, 352-360` | one client dominating by scale *direction-wise* |
| plain-model guard | `plain.py:17-31`, used in `coordinator.py:127`, `client.py:209`, `model.py:84-105` | engram/MoE/LoRA tables sneaking into a checkpoint |
| audit re-score (opt-in, `audit_layers=k`) | `audit.py:81-197` | random noise (cos 0.000), wrong seed (0.006), wrong batch (−0.002), true backprop (0.07), half-faked layers (caught 55/85/100 % at k=1/2/4) (README:414-422) |
| admin secret, fail-closed, constant-time compare | `app.py:34-59` | unauthenticated bootstrap/finalize |

Audit mechanics: layers picked by `blake2b(salt:client:round:layer)` with a per-process random salt (`coordinator.py:79`, `audit.py:63-73`); `dust_gradients(..., layers=chosen)` rows are bit-identical to the full estimate (`dust.py:173-188`, `tests/test_audit.py:39`), so k layers cost ≈k/n_layers of a client step (10m fp32 on M5 Max: full ≈11-14 s, k=4 ≈2 s). Pass requires pooled cosine, **every** checked layer, and (when checked) head cosine ≥ threshold 0.5 (`audit.py:164-184`). Honest NGD1R scores 0.92-0.93 (head 0.80).

What cannot be checked / skipped (`audit.py:41-60, 107-115`): accelerator-generated noise; `localSteps≠1`; `averagedDraws≠1`; error feedback; **`task=copy` (i.e. all real-token pretraining)**; bf16 from a different device class. These are reported `skipped` and **accepted** unless `audit_require`. The backprop route is never audited. Batch fairness is unchecked (README:440-442).

Known attack surfaces (from code reading; none have tests):

1. **Sybil**: `clientId` is a free string, default `client-<pid>-<time>` (`client.py:171`), no auth on `/submit/*`. Quorum is 2; equal-weight averaging (`coordinator.py:349-351`) means k fake ids get k votes.
2. **Additive poisoning under the audit band**: cos ≥ 0.5 permits adding an arbitrary vector with up to √3 ≈ 1.73× the honest estimate's norm per layer; after normalisation up to ~87 % of the submitted direction can be attacker-chosen. Honest estimates align with the true gradient at only cos 0.012-0.128 (README:452-457), so one passing cheater's targeted component can exceed the true-gradient signal of many honest clients. Unaudited layers (with k<n) can be arbitrary.
3. **Magnitude inflation**: the audit is scale-invariant; the declared norm is measured from the client's own vector (`coordinator.py:286`), and the round applies the *mean* norm capped only by `max_abs_grad=100` (`coordinator.py:64, 352-354`). README:244-245 and research item #9 already flag it.
4. **Seed/data grinding**: the client picks `seed`, hence noise and synthetic batch, and the data cursor; it can compute many genuine estimates and submit the most attacker-aligned one — it passes any replay audit.
5. **Audit DoS**: replay `population`, `batch`, `drawChunk` are unbounded (`audit.py:130-140`); `mode="all"` forces a full replay (`audit.py:143`). The audit runs on the event loop.
6. **Evidence inflation**: `tokens` is declared, not verified against `population·batch·seq`; it is recorded and summed into round history (`coordinator.py:408`).
7. **Doc/code mismatch**: README:163-164 says DUST submissions are token-weighted; `weight` is stored (`coordinator.py:295-297`, `tests/test_coordinator.py:243`) but `finalize` never uses it.
8. **Verifiability**: the salt is secret and not persisted, so no third party can reproduce an audit verdict; receipts are unsigned JSON (`coordinator.py:205-220`).
9. **Robust aggregation is not a fallback**: `research/communication-efficiency.md:254-275` argues Krum/median degenerate on near-orthogonal unit vectors. In this design, replay audit is the *only* viable anti-poisoning primitive.

## 4. State / persistence and the HF baton

`ModelStore` (`store.py`): `weights.safetensors` (fp32, atomic), `status.json` (atomic, round, digest, last 50 round records, current `pending` client ids), `submissions.jsonl` (plain append, unsigned, not hash-chained, `store.py:211-213`). Shards: tensor-aligned byte ranges ≤256 MiB with per-shard sha256, cached by mtime/size (`store.py:132-168`). `manifest()` **loads all weights and recomputes the digest on every call** (`store.py:191-198`), and `/status` calls it on every poll (`app.py:164-174`) — O(model) per status request.

Restart semantics: pending submissions are in memory and dropped (`coordinator.py:147-154`); momentum / outer-momentum buffers are in memory only (`coordinator.py:118-119`), so a restart silently changes the optimizer trajectory. The audit salt also regenerates. Round records list only a client *count* (`coordinator.py:398`), not which clients or submission hashes, so "who trained what" exists only in the unsigned jsonl keyed by self-asserted ids.

HF baton (`docs/HF_BATON.md`, `tools/hf_baton.py`): one repo holds `weights.safetensors` (fp32, 1.22 GB at 300M), `status.json`, `manifest.json`, `seed.json` (genesis), `lock.json` (advisory 24 h lease, `hf_baton.py:56, 207-217`). Trainers pull → train locally for N steps with local SGD+momentum → push with round+1. Guards: `push_problem` refuses remote ≥ local round (`:220-232`), but `cmd_push` reads the remote status and *then* `create_commit` without `parent_commit` (`:315-339`, `:98-110`) — a TOCTOU race where two writers can both pass. `verify` (`:270-312`) only checks internal consistency (digest == status == manifest); any write-token holder can push arbitrary weights with consistent documents. It is single-writer by construction: peers take turns, so throughput never exceeds one GPU. The doc's "N peers multiply linearly" (`HF_BATON.md:57`) holds for total contributed time, not for parallel rate.

Single points of failure: the coordinator process + its disk (state, salt, momentum, admin secret); the admin secret (bootstrap can replace the model, `app.py:228-246`; `force` finalize); one HF repo + one write token in baton mode; HF Hub availability.

## 5. Model presets & scaling

Presets (`model.py:52-76`; counts from `param_report`, `model.py:108-118`): tiny 1.08 M (4L/128/ctx64/vocab256); 10m 7.34 M (6L/256/512); 100m 116 M (12L/768/1024); 300m 306.2 M (18L/1024/2048); 1b 1,015,085,056 (15L/2048/2048); 1b-deep 1.35 B (20L); 2b 2.73 B (32L/2304/18 heads). Vocab is fixed at 4096 (DUST BPE-4096, pinned by sha256 in `tools/prepare_tokens.py`). README's `200m` preset (README:8, 36) does not exist in code; the title says "Crowd-v1.1".

Ripple of a size change:

- Client memory ≈ weights (bf16 2 B/param) + **fp32 grad buffer 4 B/param** (`dust.py:200`) + activations × draw_chunk. 1B step peaks at 6.06-6.47 GB (README:304, 323); at 9B the buffer alone is 36 GB and at 27B 108 GB — no layer or pipeline sharding exists, although `layers=` subsets (`dust.py:181-188`) are a natural hook for per-layer assignment.
- Downloads: full fp32 per round (4 GB at 1B, ~108 GB at 27B); no delta or bf16 serving.
- Uploads: ~1/8 B/param with ngd1 → ~1.1 GB at 9B, ~3.4 GB at 27B; exceeds `max_body_bytes` 1 GiB.
- Coordinator RAM: fp32 weights + one fp32 direction per pending client (`coordinator.py:308`) + float64 accumulator (`:350`) + momentum — infeasible past a few B without streaming aggregation.
- Estimator quality: cosine to backprop falls ~2.7× from 10m to 1B; matching 10m at 1B needs ~7× population (README:459-463); one 1B pop-128 step ≈580 s on M5 Max.
- MoE: `plain.py:17` rejects any parameter containing `moe`, `expert` or `router`, and its rationale (sparse rows degrade a ZO estimate, `plain.py:3-12`) applies to experts. **The 27B MoE roadmap is in direct conflict with the current method and guard**; it is an open research question, not a config change.
- Evidence ceiling: no ZO method has trained from scratch beyond 243M params / 20M tokens (`research/README.md:72-78`, `research/zo-es-lineage.md:242-249`).

## 6. Metrics / observability today

- `GET /status` (`coordinator.py:169-192` + manifest fields): round, baseSha256, routes, lr/momentum/outer settings, acceptedThisRound, acceptedByRoute, **pending client ids** (public), lastRound record (`gradNorm`, `declaredNorm`, `clipped`, clients, tokens, `newSha256`). `GET /admin/round` adds `canFinalize`.
- `submissions.jsonl`: per-submission tokens/population/sigma/weight/errorFeedback/audit verdict.
- No logging module, metrics exporter, held-out eval or loss of any kind in `crowd_v15/` — the coordinator cannot tell whether the model is improving.
- Loss and throughput live only in tools: `scale_test.py` (jsonl with trainLoss/evalLoss/tokensPerSecond/uploadBytes/peakVramGb), `time_to_target.py`, `bench_*`. `publish_results.py` pushes per-machine jsonl/csv/hand-rolled SVG/RESULTS.md to the HF repo on a loop; `publish_summary.py` builds a cross-machine table from median step time, with a hardcoded two-machine `HARDWARE` dict (`publish_summary.py:26`). These are periodic snapshots, not live telemetry, and are unsigned.

## 7. Documented performance

- 1B DUST step (seq 32, bs 1, pop 2, 153 forwards; "evidence tokens" = pop × tokens): RTX 4060 Ti 2.08 s / 30.7 tok/s / 6.47 GB; Radeon 890M 8.18 s / 7.8 tok/s / 6.06 GB; M5 Max MPS 1.16-1.31 s / 48.9-55.2 tok/s (README:296-348).
- 300M baton, 4060 Ti, batch 2 × seq 512, pop 32, mode all: 57.5 s/step, 17.9 evidence tok/s, 8.46 GB; 300M × 20 tok/param ≈ 6.1 B tokens ⇒ infeasible on one card (`HF_BATON.md:50-57`).
- 10m e2e over HTTP, 2 clients, pop 32: MPS ~2.1 s/client step, ~4,000 tok/s, round 3.1 s; CPU ~13.1 s, 625 tok/s (README:362-371).
- Learning (tiny, identity — toy): outer lr 0.7 / Nesterov 0.9 takes final loss 4.11 → 0.50 (README:233-239); tiny identity held-out 4.84 → 2.76 in 30 rounds, 6.6 s MPS vs 259 s CPU (README:373-376); error feedback recovers ~71 % of the 1-bit loss gap (README:193-201); draws 1→4 loss 1.45 → 0.45 (README:268-272); draws 1→8 cosine 0.17 → 0.44 at 10m (README:258-263).
- Audit: see §3.
- **Unpublished local artifact** (`runs/ttt`, gitignored, sweep incomplete — only p8 ran): 10m on M5 Max, bf16, seq 128, batch 8. Adam reached eval 5.31 on 2.0 M tokens in 22 s (~91 k tok/s); DUST pop 8 (outer lr 0.7, ngd1r, device noise) reached best eval 7.21 on 147 k tokens in 116 s (~1.27 k tok/s, ~0.95 MB upload/step). Time-to-target vs backprop is the open gate and currently unfavourable.

## 8. Reusability for v2

| module | verdict | reason |
|---|---|---|
| `codec.py` | **keep** | hardened, bounded, v1-compatible; add low-rank/token-row codec later |
| `protocol.py` | **adapt** | keep framing + guards; add signature, pubkey id, nonce, submission hash, bounded/validated `replay`, codec-space commitments |
| `dust.py` | **keep (adapt)** | estimator, deterministic seeds, `layers=` subset are the audit foundation; needs per-layer sharded estimation, bounded config, deterministic fp32 path |
| `audit.py` | **adapt heavily** | keep salt-picked partial replay; make the salt a public post-commit beacon; tighten the comparison (codec-space or elementwise tolerance on fp32 replay, not cos ≥ 0.5); verify `tokens`; bound replay cost; support real-token data via content-addressed shards |
| `feedback.py` | **keep, gate** | useful but makes submissions unauditable; allow only where verifiable or drop |
| `model.py` | **adapt** | fine to ~2-3B; needs larger vocab, context, presets, sharding, and an MoE story |
| `plain.py` | **replace/rethink** | contradicts the MoE roadmap; keep as a per-epoch architecture whitelist |
| `coordinator.py` | **replace** | single authority, in-memory momentum, order-dependent float math, unweighted mean, Sybil-open; salvage acceptance rules + RDA/outer-Nesterov logic into a deterministic state-transition function |
| `store.py` | **adapt** | atomic writes, shard plan and canonical digest are reusable; add exact content hash (fp32 bytes) next to the bf16 digest; content-addressed shards; stop O(model) work per manifest |
| `server/app.py` | **replace** | single-node HTTP; no peer protocol, auth, rate limits, async offload |
| `client.py` | **adapt** | sync/verify/shard-download/encode loop is reusable; seeds and data must come from the network, not the client |
| `accel.py` | **keep** | small, correct |
| `tasks.py` | **adapt** | `token_batch` repeats one window `batch` times (`tasks.py:76-77`); needs deterministic, network-assigned shard sampling |
| `tools/hf_baton.py` | **replace** (keep as bootstrap/mirror) | single-writer, racy, unverified |
| `tools/bench_*`, `scale_test`, `time_to_target`, `publish_*` | **keep** | measurement harness for epoch-pivot gates; turn publish into signed live telemetry |

## 9. Gaps vs v2 goals

- **Decentralization**: one coordinator, one admin secret, one disk; the baton is a serial single-writer. No peer discovery, gossip, multi-orchestrator agreement, or deterministic replicated state transition (momentum is in memory; `np.linalg.norm` and summation order are platform/arrival dependent; the digest tolerates sub-bf16 divergence). Needed: a fully specified, order-canonical, bit-reproducible `apply(prev_state, sorted_accepted_submissions, config) → next_state` with optimizer state hashed into the round.
- **Zero trust**: no identities or signatures; Sybil-open quorum; client-chosen seeds and data; audit optional, skips real-data and fast-path submissions, and tolerates large additive poison and unbounded scale; backprop route unaudited; audit not publicly reproducible. Needed: keypair identities + stake/reputation/PoW-style Sybil cost; network-assigned seeds and data shards from a randomness beacon; commit-then-reveal so the audit salt is fixed after the commit; mandatory, tight, publicly re-checkable audits; redundant auditing across orchestrators; median/bounded norms; verified token counts.
- **Consensus / immutable ledger**: only an unsigned jsonl and a 50-entry history; round records do not name contributors or submission hashes. Needed: hash-chained (Merkle) round blocks containing parent digest, exact weight hash, optimizer-state hash, accepted submission hashes + signer keys, audit verdicts + beacon, config hash, timestamp. The v1 `CrowdGPT-Server/orchestrator/{records,receipts,identity}.py` (Ed25519, Merkle roots, canonical JSON) under `crowdgpt-v2/reference/` is the stated template (README:483-485).
- **Live metrics**: no coordinator-side eval, no metrics endpoint, no logging; `/status` is O(model). Needed: deterministic held-out eval per round as part of the block (verifiable by any peer), per-contributor accounting from the ledger, streaming metrics.
- **Scale roadmap / epoch pivots**: presets stop at 2b, vocab 4096; full-model fp32 buffers and downloads per client; coordinator holds N full vectors; uploads exceed limits beyond ~8B; MoE is banned by design and unproven for DUST; alignment degrades with size and time-to-target vs Adam is unfavourable at 10m. Needed: per-layer/pipeline work assignment (build on `layers=`), streaming aggregation, delta/bf16 weight distribution, explicit go/no-go gates per epoch (cosine-vs-size, time-to-target), and a defined pivot mechanism (architecture whitelist + growth/initialisation rule) recorded on the ledger.
