# CrowdGPT v2 — Scoping Workspace

Local-only (excluded via `.git/info/exclude`). Everything here is
`DRAFT — requires human review` and is not committed.

| File | What it covers |
|---|---|
| `00-GAMEPLAN.md` | Start here: requirements, starting point, principles, phases, risks |
| `01-architecture.md` | Roles, round lifecycle, horizontal scaling, storage, per-component declarations |
| `02-trust-and-verification.md` | Threat model, identity, layered checks, rollback |
| `03-consensus-and-record.md` | The permanent record, how orchestrators agree, self-healing, upgrades |
| `04-live-metrics.md` | Indexer, live site, alerts, operator observability |
| `05-scale-roadmap.md` | Epoch table, triggers, automatic pivots, hardware floor, splitting the model across contributors |
| `TODO.md` | Phased task list |
| `DECISIONS.md` | Questions only the owner can answer |
| `adr/` | Proposed architecture decisions |
| `research/` | Evidence behind the plan |
| `../research/papers/`, `../research/lab/` | Owner's paper survey (SOLO, SOMA, and 10 others) and MOSAIC experiments, at the repo root, git-ignored |
| `reference/` | Clones of the original CrowdGPT client and server |
