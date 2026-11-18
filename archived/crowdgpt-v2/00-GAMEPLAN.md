DRAFT — requires human review

# CrowdGPT v2 — Gameplan

Local-only scoping document. Not committed. Every claim about other projects
comes from `research/03-prior-art.md`, which tags each one as primary,
secondary, or unverified; check the tag there before repeating it.

## 1. What v2 has to do

The owner set five requirements (2026-11-17). Each one maps to a design doc.

| # | Requirement (owner's words, shortened) | Where it is designed |
|---|---|---|
| R1 | Distributed by design: anyone can run an "orchestrator"; training continues while any orchestrators survive; scale by adding orchestrators | `01-architecture.md` |
| R2 | Zero trust: verify everything, rule out poisoning by bad actors | `02-trust-and-verification.md` |
| R3 | Self-governing mesh: orchestrators reach conclusions on their own and keep an immutable record of what was trained, when, and by whom | `03-consensus-and-record.md` |
| R4 | Live metrics: the website shows the real status of every contributor, their work, and the whole system | `04-live-metrics.md` |
| R5 | Long-term roadmap built in: 1B → 4B → 9B → 27B mixture-of-experts, with automatic pivots at milestones (like Bitcoin halvings) | `05-scale-roadmap.md` |

Supporting files: `TODO.md` (phased task list), `DECISIONS.md` (questions only
the owner can answer), `adr/` (proposed architecture decisions),
`research/` (evidence).

## 2. Where we start from

- **v1 (CrowdGPT, defunct 2026-11-16)** proved people will show up: 92
  accounts, 18 active GPUs, ~27B tokens by round 338. It also showed what
  breaks: one server on a 2-core box, Hugging Face as the database, trust
  based on three clients agreeing (three fake accounts pass), self-reported
  token counts setting merge weight, a signed record that committed to the
  wrong weight hash, and merged updates that did not reliably lower held-out
  loss. Several useful pieces exist in the v1 server (Ed25519 node identity
  with key rotation, canonical JSON, Merkle roots, presigned uploads, node
  roles for leader/witness/validator) but none of the multi-orchestrator
  logic was ever built. Detail: `research/02-v1-baseline.md`.
- **v1.5 (this repo)** is a cleaner, smaller stack: a plain transformer,
  DUST (activation-perturbation) training on the client, a coordinator, a
  replay audit, 173 passing tests. Its weak points are the same class of
  problem: unsigned free-text client IDs, clients choose their own seed and
  data, the audit skips several training modes, a cosine threshold of 0.5
  lets an attacker steer most of an update, and nothing is replayable
  because state lives in memory. Detail: `research/01-v15-stack.md`.

**Carry forward:** v1.5 codec, device layer, DUST kernel, measurement tools;
v1 wire schemas, node identity registry, content-hashed idempotent uploads,
presigned upload grants, hash-pinned evaluation, signed-release pattern.
**Rebuild:** the coordinator (as a deterministic state machine), the server,
the record, the client's work-assignment logic.

## 3. The one risk that sits above all the others

Nobody has yet pretrained a billion-parameter model from scratch with a
zero-order / evolution-strategy method like DUST. The largest published DUST
result is 243M parameters on 20M tokens. Our own unpublished 10M-parameter
time-to-target run was far behind ordinary backprop (Adam reached eval loss
5.31 in 22 s; DUST reached 7.21 in 116 s). Meanwhile, permissionless
backprop-based runs have reached 72B parameters (Covenant-72B).

So v2 is designed with **several training lanes behind one protocol**
(`adr/0004`):

1. **DUST lane** (research): cheap for contributors, easy to replay and
   check, but unproven at scale.
2. **Backprop lane** (fallback): DiLoCo-style local steps with compressed
   updates, proven at scale, harder to verify.
3. **Local-module lane** (strong candidate): SOLO-style training (arXiv
   2609.35440), where the model is cut into a few modules that each learn
   locally. It was measured within about 3.5% of ordinary training at 2B
   parameters with 2 modules, and it splits the model across cards. The
   owner's lab design MOSAIC (`../research/lab/REPORT.md`) builds on it;
   results pending.
4. **Seeded evolution-strategies lane** (optional): strongest verification,
   weakest signal.

Everything in the trust, record, metrics, and roadmap design is shared. The
lane is a per-epoch setting in the roadmap. Milestone 1 (1B) starts only
after a measured go/no-go gate picks the lane (`TODO.md` Phase 0).

**Small GPUs stay useful at every size.** Instead of every contributor
holding the whole model, larger epochs split it: dense models into
pipeline stages run by "crews" that act as one contributor (like a mining
pool), sparse models into independent paths. The genesis roadmap fixes a
hardware floor (proposed 16 GB) that no epoch may exceed. Splitting solves
memory, not total compute: a 1T dense model would take tens of years even
on 10,000 cards, so very large epochs grow through sparse experts instead
(`05` §5, `adr/0005`).

## 4. Design principles

1. **The record is the truth.** Weights, leaderboards, credit, and the
   website are all derived from a signed, hash-chained public record. If it
   is not in the record, it did not happen.
2. **Nothing is trusted because of who sent it.** Identity comes from
   signatures. Every contribution is checked by parties chosen after the
   contributor committed to it.
3. **Contributors do not choose what they are checked on.** The network
   picks seeds, data shards, and audit samples from a public randomness
   beacon revealed after commitment.
4. **Determinism where it matters.** Aggregation, model growth, and
   pivots are pure functions of the record, so any orchestrator can
   recompute the next model and get the same hash.
5. **Training keeps going in a partition; the record catches up.** Model
   updates can be merged after the fact in a way money cannot, so v2 favours
   staying alive over stopping for agreement (see §5).
6. **No token, no money.** Credit is non-transferable and earned from
   verified work. This keeps v2 out of securities, tax, and speculation
   problems (owner decision pending, `DECISIONS.md` D3).
7. **Measure before claiming.** Every performance or safety claim in these
   docs that lacks a measurement is labelled as a target or a guess.

## 5. How v2 answers "keeps going as long as any orchestrator survives"

Bitcoin stays alive under outages because it accepts temporary forks and
picks the heaviest chain later. Classic fault-tolerant agreement (the kind
used by CometBFT and similar) stops making progress if more than one third
of voting power goes offline. These pull in opposite directions; the
research flagged this as risk #3.

The v2 plan splits the problem in two:

- **Training liveness:** any orchestrator that can reach contributors keeps
  assigning work, checking it, and producing *provisional* rounds on its
  branch of the record.
- **Record finality:** a rotating committee of orchestrators signs
  *checkpoints*. When the committee is healthy, provisional rounds become
  final within minutes. When it is not, branches keep training.
- **Reconciliation:** when branches reconnect, the record picks a canonical
  branch by a fixed rule (most verified work), and the model updates from
  the other branches are merged as one more outer-optimizer step, subject to
  the same held-out-loss check as any update. Work is not thrown away;
  contributors on the losing branch still get credit if their work verifies.
- **Committee recovery:** if no committee checkpoint appears for a set time,
  a recovery rule re-forms the committee from orchestrators that are
  provably still active. This is the riskiest piece and gets its own ADR
  and chaos tests.

Detail and alternatives: `03-consensus-and-record.md`, `adr/0002`.

## 6. Delivery phases (summary)

Full task tables are in `TODO.md`. Phases overlap where noted.

| Phase | Goal | Exit gate |
|---|---|---|
| 0 | Measure the unknowns: DUST at 300M→1B, cross-GPU rounding noise, time-to-target vs backprop, update sizes, DUST split across machines | Lane decision for epoch 1, written up with numbers |
| 1 | One orchestrator speaking the v2 protocol: signed identities, beacon-assigned work, commit-then-reveal, replay verification, hash-chained record, deterministic aggregation | Any second machine can replay the record and reproduce the weight hash |
| 2 | Federation: 3–7 known orchestrators, committee checkpoints, gossip, branch-and-merge, chaos tests | Kill any minority, then a majority; training continues and reconciles |
| 3 | Live metrics: indexer, public read API, live site, operator dashboards | Site shows verified per-contributor state within one round of the record |
| 4 | Open orchestrator admission, abuse hardening, external security review | Public testnet survives a red-team round |
| 5 | Epoch machinery: roadmap in genesis, readiness signalling, deterministic growth, first pivot rehearsal on small models | 100M → 300M pivot runs end-to-end without a human |
| 6 | Mainnet epoch 1 (1B) | Owner go decision |

Phases 1 and 3 can run in parallel once the record format (Phase 1) is
fixed. Phase 5 can be prototyped on tiny models during Phase 2.

**Size estimate (a guess):** the whole v1.5 stack (client, coordinator,
codec, protocol, DUST, 89 tests at the time) took 23 commits over roughly
5–6 active hours (git log, 2026-11-16 20:45 → 2026-11-17 10:38). v2 Phases
1–5 are a distributed system with consensus, adversarial testing, and a
website; the phase guesses in `TODO.md` add up to roughly 110–215
agent-hours of build time (about 20–40x the v1.5 build), before any
measurement runs in Phase 0. What makes it bigger: multi-node agreement and
recovery, chaos and red-team testing, a public site, and model-growth code,
none of which v1.5 has. Sums of per-part guesses tend to run high. This is a
guess, not a commitment; the dominant uncertainty is Phase 0's outcome.
Human wait time (owner decisions, legal export review, external security
review, recruiting independent orchestrator operators) is separate and
likely longer than build time.

## 7. Top risks (owned in the docs listed)

| # | Risk | Where handled |
|---|---|---|
| 1 | DUST does not scale to 1B+ | Four lanes incl. SOLO-style local modules; Phase 0 gate (`05`, `adr/0004`) |
| 2 | Cross-GPU rounding noise swamps DUST's signal, making replay checks loose or expensive | Phase 0 measurement; tolerance design (`02`) |
| 3 | Consensus stalls when many orchestrators drop | Branch-and-merge (`03`) |
| 4 | Fake identities capture the orchestrator set | Admission by verified work, caps, decay (`03`) |
| 5 | Subtle poisoning or backdoors slip through | Layered checks, held-out canaries, rollback (`02`) |
| 6 | Bandwidth and VRAM floors rise with model size and shut out small GPUs | Fixed hardware floor; pipeline crews and independent paths (`05` §5, `adr/0005`) |
| 7 | Free-riding: copying others' work, grinding seeds | Beacon assignment, commit-then-reveal (`02`) |
| 8 | Whoever controls upgrades controls the network | Roadmap fixed at genesis, signalled activation (`05`, `03`) |
| 9 | Too little total compute | Honest roadmap triggers; no fixed dates (`05`) |
| 10 | Legal, data, export, and financial exposure | `DECISIONS.md` |
