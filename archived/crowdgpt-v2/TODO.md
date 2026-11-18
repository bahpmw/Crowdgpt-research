DRAFT — requires human review

# CrowdGPT v2 — TODO

Status: ✅ done · 🔄 in progress · ⬜ not started. Phases are described in
`00-GAMEPLAN.md` §6. Estimates are in agent-hours, are guesses unless a
basis is named, and exclude human wait time (listed per phase).

Basis for guesses: the v1.5 stack took 23 commits over roughly 5–6 active
hours (git log, 2026-11-16 → 2026-11-17).

## Phase 0 — Scoping and measurement

| # | Status | Description |
|---|---|---|
| 0.1 | ✅ | Clone v1 client and server references into `reference/` |
| 0.2 | ✅ | Research: v1.5 stack, v1 baseline, prior art, governance constraints (`research/`) |
| 0.3 | ✅ | Draft gameplan docs `00`–`05`, `DECISIONS.md`, `adr/` |
| 0.4 | ⬜ | Owner review of this gameplan and answers to `DECISIONS.md` |
| 0.5 | ⬜ | Measure cross-GPU replay noise for DUST at 10M, 100M, 300M, 1B (CUDA, ROCm, MPS; bf16 and fp32) against the size of the antithetic loss difference |
| 0.6 | ⬜ | Time-to-target at 100M and 300M: DUST vs DiLoCo-style backprop vs SOLO-style local modules vs seeded weight-space ES, same tokens, same hardware, published with run files |
| 0.7 | ⬜ | DUST at 1B on real tokens for long enough to see whether held-out loss keeps falling |
| 0.8 | ⬜ | Red-team the tightened audit: best steering attack that passes replay at the measured tolerance |
| 0.9 | ⬜ | Measure update sizes after compression at 300M and 1B; check against median home upload bandwidth |
| 0.10 | ⬜ | Write the lane decision for E1 with numbers (ADR) |
| 0.11 | ⬜ | DUST split across 2–4 machines as a pipeline crew on real home connections: throughput, stalls, and whether loss matches the single-machine run |
| 0.12 | ⬜ | Activation compression between stages: how far bf16 activations can be shrunk (low-rank, 8-bit) before DUST's loss changes are swamped |
| 0.13 | ⬜ | DUST signal vs number of stages: perturb all layers, one stage per pass, and one layer per pass at 4, 8, 16 stages; cosine to the true gradient and loss after a fixed token budget |
| 0.14 | ⬜ | Small independent-paths test: train 4–8 paths of a tiny shared-plus-experts model separately, combine, and compare with one model trained on the same tokens |
| 0.15 | ⬜ | Write `adr/0005` (model splitting) up from Proposed with the 0.11–0.14 and 0.16–0.17 numbers |
| 0.16 | ⬜ | Replicate SOLO (arXiv 2609.35440) with K=2 and K=4 modules at 100M and 300M on the v1.5 model and tokens; add it as a fourth arm to 0.6. Read the owner's MOSAIC lab results (`../research/lab/`) first and reuse its harness |
| 0.17 | ⬜ | SOLO-style crew across 2–4 machines on home links: one module per machine, stale output-layer copy, one slow and one dropped module; throughput and loss vs one machine |
| 0.18 | ⬜ | Verify one SOLO module by replay from signed input activations; measure cross-GPU replay tolerance (same method as 0.5) |

Estimate (a guess): 0.5–0.9 are 10–20 agent-hours of setup and analysis;
0.11–0.15 add a further 10–20 agent-hours (pipeline plumbing is new code
with no comparable work in this repo); 0.16–0.18 add 8–15 agent-hours,
less if the lab harness already covers SOLO. The runs themselves need GPU
wall-clock time that depends on the hardware the owner provides. Human
wait: owner review, GPU access, a second home connection for 0.11 and 0.17.

## Phase 1 — One orchestrator on the v2 protocol

| # | Status | Description |
|---|---|---|
| 1.1 | ⬜ | Record format: canonical JSON, entry types, hash chain, Merkle roots, Ed25519 (`cryptography` package); round-trip and tamper tests |
| 1.2 | ⬜ | Identities: register, rotate, revoke; probation and caps as pure functions of the record |
| 1.3 | ⬜ | Beacon v1: record-derived plus single-orchestrator commit-reveal |
| 1.4 | ⬜ | Deterministic assignment function (shards, seed, slice) and tests that any node computes the same assignment |
| 1.5 | ⬜ | Commit-then-reveal submissions; idempotency keys; presigned or content-addressed upload |
| 1.6 | ⬜ | Replay verification v2 |
| 1.6a | ⬜ | Public audit beacon replaces secret salt |
| 1.6b | ⬜ | Cover every shipped training mode or remove the mode |
| 1.6c | ⬜ | Tolerance from Phase 0, token counts recomputed, replay sizes bounded and run off the event loop |
| 1.7 | ⬜ | Deterministic aggregation: per-update clipping, fixed-point sum in fixed order, outer optimizer state hashed |
| 1.8 | ⬜ | Held-out gate and canary probes; rollback entry; checkpoint states |
| 1.9 | ⬜ | Contributor client v2: key generation, assignment fetch, commit, upload, local queue, failover list |
| 1.10 | ⬜ | Independent replay tool: rebuild weights from genesis plus record plus blobs and match every hash |
| 1.11 | ⬜ | `/health`, `/metrics`, JSON logs, correlation IDs |

Exit gate: a second machine replays the record and reproduces every weight
hash. Estimate (a guess): 25–45 agent-hours. Human wait: review of the
record format before it is frozen.

## Phase 2 — Federation of known orchestrators

| # | Status | Description |
|---|---|---|
| 2.1 | ⬜ | Signed HTTPS gossip among a static peer list |
| 2.2 | ⬜ | Partial sums, cross-orchestrator robust combination, recompute and co-sign |
| 2.3 | ⬜ | Proposer rotation; committee checkpoints; finality |
| 2.4 | ⬜ | Committee commit-reveal beacon |
| 2.5 | ⬜ | Provisional branches, fork choice, merge entries |
| 2.6 | ⬜ | Recovery committee with time lock |
| 2.7 | ⬜ | Equivocation proofs and automatic ejection |
| 2.8 | ⬜ | Dispute flow: panels, bisection, committee replay |
| 2.9 | ⬜ | Chaos tests: kill a minority, a majority, partition 50/50, slow links, clock skew; measure safe branch length before merge |

Exit gate: training continues through each chaos scenario and reconciles
without a person. Estimate (a guess): 30–60 agent-hours. Human wait:
finding 3–7 orchestrator operators.

## Phase 3 — Live metrics (can start after 1.1 is frozen)

| # | Status | Description |
|---|---|---|
| 3.1 | ⬜ | Indexer: record and event ingestion, signature checks, Postgres |
| 3.2 | ⬜ | Public read API (REST plus Server-Sent Events) and JSON snapshots |
| 3.3 | ⬜ | Static site: overview, contributor, orchestrator, round explorer, alerts |
| 3.4 | ⬜ | Freshness states on every value; stale banner |
| 3.5 | ⬜ | Alert rules |
| 3.6 | ⬜ | Accessibility review |

Estimate (a guess): 15–30 agent-hours. Human wait: hosting and alert
channel decisions (`DECISIONS.md` D2).

## Phase 4 — Open admission and hardening

| # | Status | Description |
|---|---|---|
| 4.1 | ⬜ | Observer → candidate → active orchestrator track |
| 4.2 | ⬜ | Peer-to-peer gossip and blob layer (dependency ADR first) |
| 4.3 | ⬜ | Known-answer jobs for verifiers |
| 4.4 | ⬜ | Rate limits and abuse controls per key and per address |
| 4.5 | ⬜ | Signed releases with threshold maintainer keys |
| 4.6 | ⬜ | Public testnet red-team round |
| 4.7 | ⬜ | External security review |

Estimate (a guess): 20–40 agent-hours. Human wait: external review,
red-team volunteers.

## Phase 5 — Epoch machinery

| # | Status | Description |
|---|---|---|
| 5.1 | ⬜ | Genesis roadmap table and epoch config schema |
| 5.2 | ⬜ | Per-epoch allowed-architecture list (replaces `plain.py` global ban) |
| 5.3 | ⬜ | Trigger evaluation: tokens, plateau, rehearsal jobs, signalling, deadline |
| 5.4 | ⬜ | Deterministic growth recipes |
| 5.4a | ⬜ | Depth copy and function-preserving widening, with a test that the grown model matches the old model's outputs |
| 5.4b | ⬜ | Sparse upcycling to mixture-of-experts (research) |
| 5.5 | ⬜ | Work-role re-tiering against the genesis hardware floor |
| 5.5a | ⬜ | Pipeline crews: stage assignment, redundant holders per stage, routing around dead stages, per-hop signed input/output hashes |
| 5.5b | ⬜ | Per-stage replay verification and per-stage credit |
| 5.5c | ⬜ | Independent paths: document routing, path assignment, combining paths at round boundaries (research) |
| 5.6 | ⬜ | bf16 weights, delta downloads, slice-only buffers, compressed uploads |
| 5.7 | ⬜ | Rehearsal: 100M → 300M pivot end-to-end on the testnet without a person |

Estimate (a guess): 20–40 agent-hours. Human wait: roadmap table approval.

## Phase 6 — Epoch 1 (1B)

| # | Status | Description |
|---|---|---|
| 6.1 | ⬜ | Tokenizer and E1 dataset fixed, licences recorded |
| 6.2 | ⬜ | Go / no-go gate (`05` §7) recorded |
| 6.3 | ⬜ | Genesis entry signed |
| 6.4 | ⬜ | Launch |

Human wait: owner go decision; export review if weights will be published.
