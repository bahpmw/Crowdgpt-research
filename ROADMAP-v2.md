DRAFT — requires human review

# CrowdGPT v2 roadmap (short version)

Date: 2026-11-18. This is the cut-down plan. The full first draft is in `archived/crowdgpt-v2/` (start at
`00-GAMEPLAN.md`). The evidence behind the changes is in `research/papers/SYNTHESIS-round2.md` and
`research/papers/SYNTHESIS-round2-addendum.md`.

## Goal

Train one open model family on volunteer GPUs with no single point of failure. Check every contribution and keep a
permanent public record of what was trained, when, and by whom. Grow the model automatically at planned milestones.
The design has to keep working from tens of GPUs up to, in principle, millions.

## The five big changes from the first v2 draft

| # | Change | Was | Now | What we gain | Why (evidence) |
|---|---|---|---|---|---|
| 1 | **Training engine** | DUST (forward-only perturbation) as the main lane | Local modules: the model is cut into 2–4 slices, each trained with its own small loss head. Volunteers recompute earlier slices themselves (MOSAIC) instead of waiting for them | Model quality close to normal training; slices fit small cards; each slice is a self-contained, checkable job | Our lab: SOLO matched normal training from 10M to 120M parameters with up to 6 slices (5.12–5.14 vs 5.17). Every forward-only method, DUST included, stalled at 6.6–8.3. The SOLO paper (arXiv 2609.35440) shows the same up to 2B |
| 2 | **Unit of work** | Whole-model rounds | Small **cells**: one slice, one data block, one round, with inputs and outputs hashed into the record | Volunteers can join or leave at any time; any cell can be re-run and checked alone; it works like a "share" in a mining pool | Agora (8.6B, 330 machines) lost throughput to delays and coordination, not bandwidth; failures at large scale are constant (SPARe: one every few minutes at 600k GPUs) |
| 3 | **Checking** | Redundant audit of every update | Network assigns the data, volunteers commit first, then the beacon picks a sample of cells to re-run on the same class of hardware. Pairs are rotated so colluders can't keep vouching for each other. Merges use exact integer maths | Checking costs a small fraction of compute instead of 2× or more; colluders and faulty cards get caught | Poisoning 0.1% of training data can plant a backdoor that works about 92% of the time at 4B (2610.37741). Small-scale step audits cut attacks from 99.5% to 7.2% at 2.8% overhead (2607.06643). Cross-vendor exact replay costs about 12× (OPEN-1B) |
| 4 | **Syncing** | Every update goes to the orchestrators | Three levels: groups of ~16 average through a relay; a committee per slice merges groups; the root record holds only signed checkpoints. Rounds close when the work is covered, not when everyone reports | No machine has to take in traffic from everyone; slow or vanished volunteers never stall a round | One receiver would need about 2.76 Tbit/s at 1M volunteers (lab model); 1–2 peer gossip fell behind full averaging as groups grew (GASLoC) |
| 5 | **Growth plan** | Dense 1B → 4B → 9B → 27B, then mixture of experts | Become a mixture of experts (MoE) early, at about 1B, then grow in ×2 steps. A step is triggered when verified participation doubles, with a minimum spacing and a total token horizon fixed at launch | Growth events become predictable "halving-style" moments the whole network can prepare for. Volunteers hold only their experts | Converting dense to MoE later did worse than a half-size MoE trained from scratch (Expert Upcycling); doubling is the most reliable growth step (Warmstarting); the best split point depends on the planned horizon (Optimal Splitting) |

Smaller changes that ride along: one fixed, published delay for every hand-off, so cells stay replayable; equal-token
work units, so fast cards take more units and merging stays fair; FP8 weight storage, 8-bit optimizer state and system
RAM, so 16 GB cards can hold 3–5B parameters; AdamW on volunteers with the heavier optimizer maths done at merge time;
a Gated DeltaNet hybrid architecture, roughly 1.3× fewer tokens to reach the same loss at 1B by the paper's own
projection; and real jobs for phones and small cards (verifying, scoring, data filtering) instead of DUST.

## What stays the same

- No cryptocurrency or token. The permanent record is a signed, append-only log published with witness signatures.
- Training continues on separate branches during an outage, and the record reconciles them afterwards.
- Every number on the live website comes from the record and is labelled verified, provisional or self-reported.
- Nothing is claimed faster or better until we measure it.

## Phases

Each phase ends with a gate. If the gate fails, we change the plan before building more.

| Phase | What we build | Gate to move on | Agent-hours (guess) |
|---|---|---|---|
| **0. Prove the engine** | Lab experiments X1–X9 from the addendum: MOSAIC vs passing activations between machines vs normal training at 100M/300M; same-class replay tolerance; groups of 16 with relay averaging; expert doubling; largest slice on a 16 GB card; churn simulation; small poisoning red team | MOSAIC within ~3% of normal training at 300M; honest replays pass and planted faults fail; poisoning drill caught | 26–52 |
| **1. One orchestrator, new protocol** | Signed identities, cells, leases, beacon-assigned data, commit-then-check, integer merges, signed record. The MOSAIC lane on a small MoE | A second machine rebuilds the same weights hash from the record alone | 25–45 |
| **2. Many orchestrators** | 3–7 orchestrators, slice committees, three-level sync, coverage-based round close, outage and rejoin drills | Kill any orchestrator mid-round; training continues and the record reconciles | 30–60 |
| **3. Live metrics** | Indexer that reads the record, live feed, static website first | Every number traces to a record entry; stale data is shown as stale | 15–30 |
| **4. Open doors** | Public admission with probation, device reputation and quarantine, per-identity caps, external red team | Red team cannot get poisoned data accepted at the target rate | 20–40 |
| **5. Growth machinery** | Epoch table with token horizon, ×2 expert doubling, rehearsal growth from 100M to 300M on testnet | Growth is bit-for-bit reproducible from the record and passes the post-growth quality check | 20–40 |
| **6. Launch** | First public epoch | Owner sign-off, plus legal and export review | — |

Total: about 136–267 agent-hours, plus human waits for hardware, review and legal. This is a guess. The phase 1–5
figures come from the first v2 draft; Phase 0 replaces its old experiment list. The only build in this repository to
compare with is v1.5: 23 commits over about 5–6 active hours. v2 is far bigger because it adds a distributed record,
checking, and growth machinery, and Phase 0 is research with repeat runs.

## Model milestones (to be fixed in the genesis epoch table)

| Epoch | Model | Trigger |
|---|---|---|
| E0 testnet | Small MoE, ~100M → 300M by one ×2 step | Rehearsal only |
| E1 | ~1B active parameters, sparse from the start | Launch |
| E2 onward | ×2 expert doubling | Verified participation doubles **and** at least as many tokens trained since the last doubling **and** 90% of orchestrators running the new software |
| Later | Split into independent specialist paths | At the point the splitting law predicts for the published horizon |

At very large scale, data runs out before compute does. Spare GPUs go to data filtering, evaluation and checking
rather than ever-larger single models.

## Decisions still needed from the owner

- D2: which outside services we may use (Hugging Face, storage, a domain) and any spending.
- D4: legal and export review before publishing large weights and cryptographic code.
- D8: confirm the switch from a dense first epoch to sparse-early growth.
- D9: access to a 16 GB CUDA card for Phase 0 memory tests.
- Talk to the upstream author (Vxtzq) about where v2 lives.
