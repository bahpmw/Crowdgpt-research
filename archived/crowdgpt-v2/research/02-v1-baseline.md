DRAFT — requires human review

# 02 — CrowdGPT v1 Baseline (client + coordinator)

**Sources.** Two public clones, read at HEAD on 2026-11-17:
- `C/` = `/Users/patwoodward/Documents/docker/opencode/projects/Crowd-v1.5/crowdgpt-v2/reference/CrowdGPT-v1` (971 commits, last `00192ee` 2026-11-16)
- `S/` = `/Users/patwoodward/Documents/docker/opencode/projects/Crowd-v1.5/crowdgpt-v2/reference/CrowdGPT-Server` (71 commits, last `04940ef` 2026-11-16)

All `file:line` references point into these trees. "Inference" marks a conclusion that is not stated in the code or docs. Values marked as defaults can be overridden with environment variables. The public server is **not guaranteed to match production**. `S/krum_stream.py:1-8` says "the private deployment's version may use a different streaming strategy".

---

## 1. System architecture

**Components**

| Component | What it is | Evidence |
|---|---|---|
| Client | `C/client.py` (4,914 lines, v3.5) is a PyTorch training wrapper. It also includes a desktop UI (`client.html`), a TUI, an installer and Docker overlays. It runs no code sent by the server. | `C/client.py:197`, `C/docs/ARCHITECTURE.md:17-21` |
| Coordinator | `S/server.py` (5,377-line FastAPI monolith, port 5006) plus an HTTP-free `S/orchestrator/` package backed by SQLite WAL | `S/server.py:136`, `S/docs/ORCHESTRATOR_MODULES.md:1-18` |
| Model store | HF model repo `Vxtzq/Crowd-v1`: `model_bf16.safetensors`, `engram_bf16.safetensors`, optional fp32 copies, `training_state.json`, `rounds.jsonl`, `nutrition.json` | `S/server.py:240,246,2620-2665` |
| Dataset | HF dataset `Vxtzq/CrowdGPT`, pre-tokenized `chunks/chunk_XXXX.bin` (uint32, Qwen3 tokenizer). Clients read 10 MB HTTP range slices. | `S/server.py:241,262-265`; `C/docs/DATA.md:13-16` |
| Data intake | HF dataset `Vxtzq/CrowdGPT-Pending` (text PRs), plus a Cloudflare-Worker form at `upload.crowdgpt.net`. The maintainer reviews and tokenizes manually. | `C/docs/DATA.md:23-27`; `C/docs/index.html:5976` |
| Status artifacts | `status.json`, `leaderboard.json`, `badge_tokens.json`, `public_log.json`, `loss_history.json`, committed to the **model repo** | `S/server.py:2685-2754`; `S/public_log.py:10-14` |
| Website | `crowdgpt.net` = `C/docs/index.html` (GitHub Pages, `C/docs/CNAME`). It reads HF raw files, with `server.crowdgpt.net/api/*` as fallback. | `C/docs/index.html:3761-3763,5216-5217` |
| CrowdEval / loss check | Separate `Vxtzq/CrowdEval` benchmark repo (README link). In-repo: a held-out set `C/eval/ng-eval-v1`, `C/crowdgpt_ng/eval.py`, `loss_gate.py`, `C/tools/hivemind_eval_service.py` (serves `/eval` and `/gate`), and `S/tools/eval_service.py` | `C/README.md:21`; `C/docs/LOSS_CHECK_RUNBOOK.md:11-22` |
| Hivemind bridge | Three modules kept for a planned integration with the external HiveMind swarm and OpenRouter gateway. HiveMind serves Crowd-v1 as `hivemind/crowd-v1` and hot-reloads from `status.json`. | `C/docs/HIVEMIND_INTEGRATION.md:1-24`; `C/docs/HIVEMIND_SYNERGY.md:30`; `C/hivemind_bridge.py:37-38` |

**End-to-end flow of one contribution.** This flow uses the default cross-checked pool path.

1. **Wait and task.** The client polls `GET /fl/round_status`, then `GET /fl/task?skip_weights=true` (`C/client.py:2266-2340`).
   - It sends self-reported `X-Client-*` hardware headers.
   - The server identifies the caller as `user:<uuid>[:dev:<id>]` (from a bearer token) or as `ip:<ip>` (`S/server.py:1306-1316`).
   - The server checks bans and the version gate (HTTP 426), then runs matchmaking into a 3-client pool (`S/server.py:4097-4160`, `_matchmake` at `1913`).
   - Pools are hardware-matched (backend, precision, optimizer, arch, compute capability). The same UUID or same IP cannot share a pool (`S/server.py:452-469`).
2. **Pool activation.** The server assigns the next chunk from an unseeded `random.shuffle` epoch permutation (`S/server.py:1471-1475,1494-1512`), a shared seed, and step, batch and accum targets (`S/server.py:1677-1722`). It issues an Ed25519-signed ticket bound to `baseWeights="step:<N>"` (`S/orchestrator/admission.py:25-76`; `S/orchestrator/domain.py:57-71`).
3. **Weights.** The client downloads the pinned HF revision. It checks the SHA-256 the server publishes from the HF LFS tree (`S/server.py:2607-2618`), using a content-addressed cache and resumable downloads (`C/docs/NG_CLIENT_CACHE.md`). If HF fails, it streams the weights from the coordinator, and that fallback has no hash check (`C/client.py:3404-3424`).
4. **Data.** The client fetches range slices from `.../resolve/main/chunks/...`. This is **not revision-pinned and not hashed** (`C/client.py:2076,2096-2098`). In cohort mode the slot is forced to 0 and the client never advances to a new sub-chunk (`C/client.py:3465`, `2109-2112`). *Inference:* each pool trains only on the first 10 MB slice of its assigned chunk.
5. **Local training.** AdamW at lr 1e-4 with fresh optimizer state per task (`C/client.py:3502`). The run uses a fixed micro-step count (default 500), accumulation from the task payload (`C/client.py:3501`), and a deterministic batch seed `cc_seed*100003 + step*1009` (`C/client.py:3580`).
6. **Upload.** The client sends `delta = final − initial` as bf16, plus a sparse engram-row tail, gzipped, with a JSON header (`C/client.py:3631-3662`). It uses chunked `/fl/submit/init|chunk|finalize` in a background thread, or presigned `PUT /fl/object` (`S/server.py:4177-4206,4241-4514`).
7. **Admission checks.** On submit the server checks:
   - ticket validity (`S/server.py:602-623`)
   - optional receipt (`S/server.py:632-650`)
   - NaN/Inf
   - norm caps, with a subnet ban above the absolute cap (`S/server.py:~3355-3360`)
   - plausible loss between 1 and 20 nats (`S/orchestrator/adversarial.py:8-15`)
   - minimum training time of 900 s
   - token clamps (`S/server.py:3547-3560`)
   - pool membership

   It then records an idempotent submission (`S/server.py:657-678`).
8. **Pool cross-check.** When all deltas are in, or after a 4 h timeout, the server compares the deltas within the pool (§3). It merges survivors into one token-weighted pool delta (`S/server.py:1788-1883,2340-2375`).
9. **Round end.** After a fixed 4 h round (`S/server.py:3072-3183`), the watchdog runs `apply_round_aggregation`, an optional loss ratchet, writes a checkpoint, publishes a signed round record, starts a background HF weights push and status update, and opens the next round.

**Scale of the plumbing.** The production coordinator was **CPU-only: 2 cores and 15 GB RAM** (`C/docs/CROWDGPT_GPU_LOSS_CHECK.md:3`). Memory guards **discard pools** (`S/server.py:2116-2121`) or **skip aggregation entirely** (`S/server.py:3758-3764`) under pressure.

## 2. Training algorithms

**What actually ran: synchronous pooled FedAvg (local SGD) with server LR 1.0.**

- **Within a pool:** a token-weighted average of the cross-check survivors (`S/server.py:3705-3707,1816-1836`).
- **Across pools:** `pool_w = tokens_pool / tokens_round` (A3). The update is `w += lr(step) · pool_w · step_scale · clip · Δ`, where step_scale is `clamp(n_ref/optimizerSteps, 0.5, 2)` (`S/server.py:3731-3906,4914-4923`).
- **Learning rate:** cosine from 1.0 to a 0.05 floor over 50k steps (`S/server.py:2772-2779`), weight decay 0 (`S/server.py:751`).
- **No outer momentum.**
- **Clipping:** a per-delta clip at `MAX_DELTA_APPLY_NORM` 12000 (`S/server.py:755,3829`).
- **Norm guard:** if the post-merge weight-norm ratio falls outside [0.70, 1.50], the server reloads the weights from disk (`S/server.py:3878-3889`).
- **Engram:** sparse row-average with a row-norm cap of 8 and a trust ramp (`S/server.py:3846-3903`).

**Diagnosis recorded by the maintainers.** Commit `e333bdc` (S, 2026-11-16) says: "A round's aggregate was not a descent step: measured cos(grad L_heldout, −delta) sat at the noise floor (~0)". The commit also found that the flat N(0, 0.02) init, which scaled the LayerNorm gains to 0.02, gave "36x less held-out BPB per unit of weight movement" (`S/server.py:284-289`; `S/README.md:16-35`). A1 and A3 are the fix: the server dictates accum and step normalisation, and pools are weighted by tokens. *Inference:* much of the v1 run's compute did not reliably descend held-out loss, and a fresh "v1.5" run was being prepared.

**Prototypes. None of these are wired into production.**

| Prototype | What it does | Status and evidence |
|---|---|---|
| DiLoCo (`C/crowdgpt_ng/diloco.py`) | Top-k magnitude sparsification to a 50 MB budget, bf16 values, and an `OuterOptimizer` | Docstring says "Not wired into client.py" (`diloco.py:14-16`). No error feedback. The "Nesterov" is `g + m(g − g_prev)`, not DiLoCo's momentum-buffer Nesterov (`diloco.py:176-187`). |
| EGGROLL (`C/crowdgpt_ng/eggroll.py`, `tools/bench_eggroll_model.py`) | Low-rank evolution strategies | Toy and 105k-parameter mechanism checks only. Fitness is trusted. Verification is open (`C/docs/NG_EGGROLL.md:29-32,144-146`). |
| Multi-Krum (`C/hivemind_bridge.py:503-564`) and RMS-collapse guard (`:579-610`) | Library code | Never called by the server. `HARDEN["KRUM_F"]` is only logged (`S/server.py:797,3576-3580`). `TOKEN_CAP_FACTOR` and `DUP_REL_DIST` are defined but unused (`S/server.py:799-800`). The technical paper's claims of "Krum-style selection" and "near-duplicate flagging" (`C/docs/technical_paper.md:137`) are not in the public code. |
| `krum_stream.py` | Not Krum. Pairwise L2 plus streaming add. | Despite its docstring, it holds **all** vectors in RAM (`S/krum_stream.py:39-60`). `plan_round` returns None (`:79-80`). |

**Delta codecs.**
- **Default:** dense bf16 plus gzip, with a header of `<u32 len><json>` followed by the base delta, then `<u32 count><u32 idx…><bf16 rows…>` for engram rows (`S/krum_stream.py:15-37,66-77`). That is about 1.3 GB raw per delta.
- **Opt-in NGD1/NGD2:** a 1-bit sign codec with an fp16 mean-abs scale per 1024 values. It is lossy (about 60% relative error on small values) and has no error feedback. It projects to about 83 MB (`C/docs/NG_DELTA_CODEC.md`; `S/orchestrator/compression.py`). The server decoder exists, with bomb-safe bounded inflation (`compression.py:56-96,111-150`). The client never uses NGD1 when engram rows exist (`C/client.py:3646`).

**What got merged on the last day.**
- Server PR #21 (A1–A3) merged.
- Client PR #129 merged at 22:14, then **reverted by hand at 23:01** (`00192ee`, removing `optimizerSteps`/`accum` reporting). The README still documents those fields (`C/README.md:173-185`).
- At HEAD the server's step-normalisation therefore always takes the legacy path (`S/server.py:3836-3838`).
- HF links were switched from `Crowd-v1.5` back to `Crowd-v1` the same evening (`1831bce`).

## 3. Identity and trust: what is verified, and how

| Layer | Mechanism | Strength / gap |
|---|---|---|
| Accounts | PBKDF2-SHA256 at 100k iterations. Bearer tokens are stored **in plaintext** in `users.json` (`S/server.py:939-941,964-972`). Anonymous identity is the IP. | No proof of personhood. Sybils are cheap (one account plus IP per slot). |
| Node identity | Ed25519 key in `orchestrator_node_key.pem`, mode 0600 and unencrypted (`S/orchestrator/identity.py:28-49`). Node ID is `node-<pubkey[:16]>` (`nodes.py:69-70`). | One key, co-located with the server. |
| Tickets | Ed25519-signed, canonical JSON `ng-ticket-1`. Binds client, task, pool, round, budgets, nonce and `baseWeights="step:N"` (`domain.py:57-71`). Checked at submit (`S/server.py:602-623`). | `REQUIRE_TASK_TICKET` defaults to 0 (`S/server.py:794`). `baseWeights` is a step label, not a weights hash. The table is keyed by client, so re-issue overwrites (`admission.py:53-69`). |
| Client receipts | `StepReceipt` is Ed25519-signed by a client key (`receipts.py:27-44`; `S/server.py:632-650`) and includes `fingerprint_root`. | **The shipped client never sends receipts or `X-Client-Pubkey`.** A grep of `C/client.py` finds only `UPDATE_PUBKEYS`. `_check_receipt` returns None when no receipt is present. The claim in `C/docs/NG_AUDIT.md:15-18` is aspirational. |
| Activation fingerprints | 64-bit SimHash per token over a fixed 1536×64 Gaussian projection, in 258-byte blocks per 32 tokens, rooted with SHA-256 (`fingerprints.py`). Sampled replay comparison in `validation.py:11-48`. | Library only. Never invoked server-side. Exact-bit comparison (`compare_blocks`) will break under cross-hardware float nondeterminism. |
| Pool cross-check (the **only** live defence) | 3 hardware-matched clients train the same task from the same seed. The closest pair forms a "core". Another member is kept if its distance to the nearest core member is ≤ `max(5.0×core_dist, 0.02×min_norm)`. Two-member pools are kept if `d ≤ 10%` of norm. Fewer than 2 survivors discards the pool (`S/server.py:2098-2157,3633-3708`). | Threshold defaults to 5.0 in code but README says 2.0 (`S/server.py:832` vs `S/README.md:91`). The paper admits that 3 colluding clients pass, and subtle bias is undetectable (`C/docs/technical_paper.md:156`). Hardware headers are self-reported, so an attacker can steer pool placement. |
| Power users | Admin-designated accounts train solo and skip the cross-check. Only finite and norm checks apply. Weight defaults to **1.0**, although the comment says 0.5 (`S/server.py:1724-1779`). The README advertises them as able to "train by themselves, without any checks" (`C/README.md:257-259`). | Fully trusted bypass. |
| Plausibility | Loss between 1 and 20 nats. Tokens ≤ steps×max-per-step and ≤ age×1M tok/s. Training time ≥ 900 s. Upload, disk and concurrency caps (`S/server.py:785-814,3547-3560`). | All inputs are **client-reported** (loss, tokens). Token claims set pool weight in aggregation (`S/server.py:3744-3750`). |
| Bans | 3 flags trigger a **/24 subnet** ban (`S/server.py:1376-1391`). | Ban and flag dicts live in memory (`S/server.py:476-481`) and are lost on restart. A /24 ban hits innocent neighbours behind CGNAT. |
| Held-out loss ratchet | Stages candidate weights at a bearer-token URL, then POSTs to an evaluator. Rejects when Δloss > tol (default 0) and rolls back in `enforce` mode (`S/server.py:2827-2920`). | **Defaults to `off`** (`S/server.py:807`), fails open (`:811`). Operates per round, not per contribution. The eval set is **public in the repo** (`C/eval/ng-eval-v1`), so poisoning can target it. One evaluator is a single point of failure and capture (decision D11, `C/docs/LOSS_CHECK_RUNBOOK.md:117-125`). |
| HiveMind loss-check receipts | Evaluators sign base and candidate losses on N seeded windows with **per-evaluator HMAC**. The CPU backend recomputes k of 2N values chosen by `HMAC(server_secret, receipt)` (`C/hivemind_loss_check.py:325-470`; `C/docs/CROWDGPT_GPU_LOSS_CHECK.md:10-21`). | A good spot-check design. Because HMAC is symmetric, third parties cannot verify it. Not wired in. |
| Signed client updates | Ed25519-signed manifest (version, SHA-256 of `client.py`/`client.html`, expiry) with two keys (main and recovery) (`C/client.py:197-206,326-372`; `C/SECURITY.md`; `C/sign.py` → `tools/sign_release.py`). HEAD hashes match `C/docs/version.txt`. | Solid. Weaknesses: a **hand-rolled pure-Python Ed25519 verifier** (`C/client.py:208-211`), `install.sh` is unsigned (`C/SECURITY.md` "does not protect"), and there is a single maintainer key. |

**Open attacks at HEAD:**
- (a) Sybil collusion: 3 accounts on 3 IPs fill a pool and submit identical poisoned or backdoored deltas.
- (b) Power-user poisoning.
- (c) Inflated token claims, which raise pool weight.
- (d) Lazy or free-riding deltas. Nothing checks that the work was done; only distance agreement within the pool is checked.
- (e) Data-source tampering: unpinned `resolve/main`, no hashes.
- (f) A malicious coordinator. The paper admits it is "fully trusted" (`C/docs/technical_paper.md:189-191`).
- (g) Backdoors that keep eval loss flat.
- (h) Admin secret in **query strings** (e.g. `S/server.py:4729,4748`), which leaks into proxy logs.
- (i) An unauthenticated `/api/stats` computes the norm of 658M parameters per call (`S/server.py:5104`), which is a DoS lever.
- (j) Past secret exposure ("secrets were exposed… being rotated", `C/docs/technical_paper.md:209`). The client history also shows a `tokens.json` added and edited on 2026-08-21; its contents were not inspected.

## 4. Ledger, hash chain, Merkle, and recovery

**What exists (added 2026-11-09 to 11-14).**
- **Append-only SQLite tables:** `publications.round` is the primary key and the first write wins. Each new row is mirrored to `data/public_rounds.jsonl` with fsync (`S/orchestrator/publication.py:24-51`). An `events` table is append-only by convention (`state.py:168-174`).
- **Signed `ng-1` RoundRecord per round:** open/close times, `base_weights`/`aggregate_weights` digests, leader, `witnesses=()`, Merkle roots over tickets, submissions and "deltas", `prev_record_hash` (a **hash chain**), eval, byte counters, code versions, and a list of certificates (`S/orchestrator/records.py:92-124`; `S/server.py:2288-2326`).
- **Audit API:** `/api/audit/{rounds,checkpoint,proof,chain}` and `/audit` give a Merkle root over record hashes, inclusion proofs, and chain verification (`S/orchestrator/audit.py`; `S/server.py:4576-4631`).
- **Mirrors:** `rounds.jsonl` and `nutrition.json` are copied into each HF weights upload (`S/server.py:2647-2650`). `CROWDGPT_AUDIT_MIRROR` is an optional extra mirror.

**Defects that matter for v2's "immutable record of what was trained":**
1. **`base_weights` and `aggregate_weights` hash the same post-round tensor.** Both use `_model_digest()` with only the step label different (`S/server.py:2300-2306`). The record never commits to the pre-round weights.
2. **Deltas are not committed.** The leaves are `sha256(ticketId:client-supplied checksum)` plus a **local file path** (`S/server.py:661-665,2213-2214`). Delta files are deleted after aggregation (`S/server.py:3867-3869`), so a third party cannot reproduce the aggregate.
3. **Data is not committed.** The chunk index per pool, the dataset revision and the shuffle seed are absent from the record.
4. **Server and client Merkle trees are incompatible.** The server hashes each leaf string first (`S/orchestrator/records.py:27-37`); the client does not (`C/crowdgpt_ng/protocol.py:107-120`). Proof keys differ (`position` vs `side`). Server timestamps use `isoformat()` with microseconds and `+00:00`, which `RoundRecord.validate` rejects (`protocol.py:316-324`). Server `eval` dicts do not match the strict `EvalResult` schema (`protocol.py:556-629`). The documented client-side verification would therefore fail.
5. **No domain separation, and padding duplicates the last node.** Leaves are sorted, so order is lost (`records.py:29`; `protocol.py:112`). This invites CVE-2012-2459-style ambiguity.
6. **One self-held signing key and no witnesses.** The nutrition label admits it: "A leader signature is not a witness quorum until P6.6" (`S/server.py:2277`). Verifications are **overwritten** with `ON CONFLICT DO UPDATE` (`verification.py:24-29`), and so are tickets.
7. **`code_versions` is hard-coded to `"3.2"`** (`S/server.py:210,2312`).

**Restart behaviour.**
- On boot (`S/server.py:5318-5352`) the server loads `users.json`, `global_state.json`, the model and engram, and the dataset cursor (`_crash_recovery` at `1514-1529` skips a consumed position).
- `_restore_orchestrator_state` (`S/server.py:680-745`) rebuilds round, pools, tasks, live tickets and delta paths, but **only if** the persisted round is `open` and `== global_step + 1`.
- `S/tests/test_restart_drill.py` SIGKILLs a child process and asserts three things: committed round, pool and ticket rows survive; a mid-transaction write rolls back; a half upload leaves no committed object.
- **Not covered:**
  - in-memory bans, flags and active-client state
  - `_active_uploads`
  - weights mid-aggregation. `atexit` save does not run on SIGKILL. Default `MAX_CHECKPOINTS` is **1** in code (`S/server.py:749`) but the README says 5.
  - HF divergence: `sync_huggingface_state` re-uploads if HF lags (`S/server.py:2667-2682`).

## 5. Single points of failure and operational pain

- **One coordinator** on a 2-core, 15 GB box (§1). It is the trust root as well as the point of failure (`C/docs/technical_paper.md:208`). Tasks and pools were in memory only until P1.1 (`S/README.md:142`). Client "failover" (`CROWDGPT_ORCHESTRATORS`, `C/client.py:2773-2787`) just rotates URLs. A second orchestrator has no shared state, so a mid-task failover can only fail.
- **Hugging Face carries everything:** weights, dataset, data intake, status, logs and the website feed. A single `HF_TOKEN` grants write access to the model repo. Status is pushed as a **git commit every 300 s** (`S/server.py:2756-2762`), about 288 commits a day to the model repo, and weights after every round.
- **Admin secret.** A single shared secret, passed in query strings, controls bans, resets, forced updates, reloads, power users and stat edits (`S/server.py:4644-5068`). `set_user_stats` (`:4801`) can rewrite leaderboard totals. A per-host object-store HMAC secret and an unencrypted node key sit in the app directory (`S/docs/DEPLOYMENT.md:11-21`).
- **Release key.** A single maintainer controls signing, data review and HF.
- **Memory and disk fragility.** About 1.3 GB per delta and about 4 GB fp32 checkpoints. Guards skip aggregation or discard pools under pressure (`S/server.py:2116-2121,3758-3764,3496-3514`).
- **Matchmaking friction.** "Stuck on Pool matchmaking 1/3" is a documented normal state (`C/README.md:212`). Solo-dissolve timers exist to break pathological loops (`S/server.py:861-879`). The docs record a "0.1 % download-to-contribution funnel" (`C/docs/NG_EFFICIENCY.md:73`).
- **Precision drift.** bf16 cross-hardware divergence flags honest users (`C/docs/technical_paper.md:143-154`). `POOL_BF16 = False` is then coerced to `"arch"` (`S/server.py:845-846`).

**Signs of why v1 stalled.** These are inference from evidence.
1. **Ineffective optimisation.** The aggregate was not a descent step (commit `e333bdc`), and a fresh init plus v1.5 restart was in progress (`C/README.md:52-53`; the README badge points at `Crowd-v2`).
2. **Inconsistent HEAD on 2026-11-16.** The client A1–A3 change was merged and then reverted within 47 minutes. The site links rolled back from v1.5 to v1. The server and client wire contracts diverged.
3. **Maintainability.** The server README says "vibe-coded slop… only way to maintain a constantly changing project" (`S/README.md:148-150`). The paper is labelled "somewhat AI slop" (`C/docs/technical_paper.md:1`). Many commits came from an automated agent (`pi agent`, S log). The public server differs from the private deployment.
4. **Trust and security debt.** Held-out evaluation was never enforced (`C/docs/technical_paper.md:202`). Secrets leaked earlier. One operator owned every role.
5. **Contributor concentration.** See §7.

## 6. Metrics and leaderboard

- **Producer:**
  - `update_and_save_status` builds `status.json` (`S/server.py:2707-2720`): participants, contributions, `activeGpus` (seen within 900 s; the site text says 10 min), total tokens, loss change, step, round, HF revision, cross-check config, matchmaking with `poolBreakdown` by hardware (`S/server.py:1567-1580`), and byzantine flag counts by **subnet**.
  - `leaderboard.json` holds per-user `totalTokens`, `contributions`, `trustScore` (`0.5 + contrib/200 − 0.1×flags`, `S/server.py:974-979`), flags and the power flag.
  - `public_log.py` emits PII-scrubbed events (submission, pool_passed/discarded, flag, aggregation, aggregation_reverted, loss_ratchet_reject, hf_upload). It keeps a JSONL log rotated at 20 MB, the last 1,000 events, and `loss_history.json` with per-pool and per-round token-weighted **training** loss (`S/public_log.py:102-158`).
- **Consumer:** `C/docs/index.html` polls HF `status.json` every 15 s, the leaderboard and loss history every 30 s, logs every 5 s, and server health every 30 s (`index.html:5405,5762,6198-6200`). It falls back to `/api/stats` (`:5529`).
  - The page shows: contributions, tokens, active clients, held-out loss (labelled "Awaiting evaluation" when absent), a token goal of 100e9 (`:3765`), the training-loss chart, pool-by-hardware, a paginated leaderboard and a live log.
- **Freshness:** at most about 5 min (push interval) plus the HF CDN. It is never truly live, and status updates add model-repo commits.
- **Gaps for v2:**
  - There is no live per-user status (current task, pool, progress, upload state, verdicts). `yourContribution` is only a percentage for the authenticated caller (`S/server.py:5088-5092`).
  - The site uses the **self-reported** leaderboard. `/api/leaderboard/verified` exists (`S/server.py:5170-5176`; `S/orchestrator/leaderboard.py`) but is not shown.
  - Held-out loss was effectively never populated.
  - `C/crowdgpt_ng/records.py:1-12` records that status and leaderboard totals disagree with each other.

## 7. Scale achieved

- **Model "Crowd-v1":** a 657,787,392-parameter dense decoder plus a 350,000,640-parameter hashed-bigram engram, about 1.0 B total. It has 24 layers, d=1536, GQA 16/4, SwiGLU 2560, context 2048, Qwen3 vocab 151,669, and tied embeddings (`C/docs/technical_paper.md:38-52`; `S/server.py:249-282`).
- **Corpus:** advertised as about 500 B tokens in 5,659 chunks (`C/docs/technical_paper.md:58`).
- **Snapshot at 2026-11-09 (`C/data/summary.json`):**
  - round 338, step 337
  - 92 accounts and **18 active GPUs**
  - 26.96 B tokens reported (25.55 B attributed)
  - 6,624 contributions
  - reported training loss 3.155
  - **Concentration:** top-1 holds 29.0% of tokens, top-3 70.6%, top-10 88.7% (HHI 0.187). 33 accounts have zero tokens.
- **Per round late in the run** (`C/data/rounds.jsonl`): about 4 h rounds, 94–107 deltas per round, and 0.42–0.48 B tokens per round, all `verification: unverified`.
- **Throughput:** about 450–500 tok/s on a Radeon 890M at the full model size (`C/docs/NG_EFFICIENCY.md:38-50`).
- **Benchmarks:** an image only (`C/docs/benchmarks/benchmarks.png`), with the disclaimer "does not mean Crowd-v1 is an intelligent model" (`C/README.md:50`).
- **No multi-model or MoE work exists.**

## 8. Reusable assets for v2

| Module | Verdict | Why |
|---|---|---|
| `C/crowdgpt_ng/protocol.py` | **Keep, then adapt** | Canonical JSON with floats as decimal strings, strict schemas, Ed25519 envelopes, and Ticket/StepReceipt/EvalResult/CheckpointRef/NodeRecord/RoundRecord with a **multi-certificate** list. It is the best seed for a v2 wire spec. Fix the Merkle tree (domain separation, ordered leaves) and reconcile it with the server. |
| `S/orchestrator/records.py` | Merge into protocol | It duplicates the protocol with an incompatible Merkle tree. |
| `identity.py`, `nodes.py` | **Keep, then adapt** | Node keys and a registry with **key-rotation overlap windows**. Add encrypted or hardware-backed keys and gossip. |
| `admission.py` (signed tickets) | Adapt | The binding concept is good. Bind content-hash base weights, the data shard hash and the issuer. Keep history rather than upserting. |
| `submission.py` | Keep | Content-hash idempotency and finalize-replay are directly reusable. |
| `verification.py`, `audit.py`, `publication.py` | Adapt | Verdicts with evidence, an append-only publication with a public mirror, and chain and inclusion verification. Replace with a replicated, hash-linked log. Make verdicts immutable. |
| `objects.py` | **Keep** | Presigned, SHA-256-bound, content-addressed PUT grants, designed for S3/R2. This is the correct control-plane/data-plane split. |
| `compression.py` | Adapt | A safe bounded decoder. Add error feedback before relying on 1-bit compression. |
| `fingerprints.py`, `validation.py`, `receipts.py` | Adapt (research) | Seeds for a verification layer: SimHash activation fingerprints, a seeded replay sample, and signed client receipts. They need tolerance-based comparison and a client implementation. |
| `C/hivemind_loss_check*.py`, `tools/hivemind_eval_service.py` | **Keep the concept** | Evaluator receipts with **unpredictable k-of-2N spot recompute** fit a zero-trust held-out gate well. Switch HMAC to signatures and add multiple independent evaluators. |
| `C/crowdgpt_ng/eval.py`, `loss_gate.py`, `eval/ng-eval-v1`, `tools/reproduce_eval.py` | Keep | Hash-pinned manifest, bootstrap CI, a `ln(vocab)` uniform sanity check, and a refusal to invent numbers. Add a secret or rotating held-out split (commit-reveal) so the gate cannot be gamed. |
| `leaderboard.py`, `proposals.py` | Adapt | The split between verified and reported figures is right. Token-weighted voting is a seed for milestone and pivot governance, but it is plutocratic and Sybil-exposed. |
| `public_log.py` | Adapt | A PII-scrubbing event stream is a good base for live public metrics. Move it off HF commits to a streaming or CDN feed. |
| `state.py` (SQLite WAL) | Keep only as a local node cache | A single writer cannot serve as the consensus layer ("P3 replaces it with one writer per node", `S/docs/ORCHESTRATOR_MODULES.md:24-26`). |
| `C/hivemind_bridge.py` (Multi-Krum, RMS guard), `diloco.py` | Adapt | Reference robust-aggregation code and an outer-loop sketch. Fix Nesterov and add error feedback. |
| `C/client.py` pieces | Mine selectively | Hardware detection, the VRAM degrade ladder, the content-addressed cache, resumable downloads, the signed updater (replace the hand-rolled Ed25519), the activity monitor, and the TUI. Discard the monolith. |
| `S/server.py` | Discard as architecture | Keep the ideas: hardware-cohort matchmaking, plausibility clamps, norm-ratio guard, loss ratchet, same-actor guard. |
| `krum_stream.py`, `adversarial.py`, `eggroll.py`, `crowd_jepa.py` | Discard or reference | Stubs, trivial code or research toys. |
| `C/crowdgpt_ng/records.py`, `data/*.jsonl` | Keep | Legacy exporter plus the v1 history as a baseline. |
| `sign.py`, `tools/sign_release.py` | Keep the pattern | Signed manifest with expiry and main plus recovery keys. Move to threshold or multi-maintainer signing. |

**Concepts that already anticipate multiple orchestrators.**
- `NodeRecord` roles `{leader, witness, validator, scheduler, storage}`, `bond_rounds` (a stake-like bond), `status` draining/offline, and endpoints (`C/crowdgpt_ng/protocol.py:44-45,692-760`).
- `RoundRecord.witnesses` and a certificate list designed for a "caller decides how many distinct certificates" quorum (`protocol.py:925-931`).
- The node registry and rotation (`S/orchestrator/nodes.py`).
- The ticket `issuer` and `attempt` fields (`protocol.py:364-378`).
- Publication JSONL "that… another orchestrator can consume" (`S/orchestrator/publication.py:1-5`).
- Planned "one writer per node" (P3) and "witness quorum" (P6.6).
- Client orchestrator failover lists.
- The requirement for a second independent evaluator before going fail-closed (`C/docs/LOSS_CHECK_RUNBOOK.md:117-125`).
- The presigned object-store abstraction.

All of these are schema-level only. No gossip, replication, leader election or quorum logic was implemented.
