DRAFT — requires human review

# 04 — Governance Constraints for CrowdGPT v2 Design

Source: Anchor governance files in this repository (FASTPATH.md, AGENTS.md, BAH.md, GOVERNMENT.md, PROJECT.md, skills/06, 13, 16, 20, 22, 28, 30, 31, 41, 50, 63, 68). Precedence when rules conflict: law/contract > GOVERNMENT.md > BAH.md > AGENTS.md > prompts (GOVERNMENT.md §2). Items marked **[ASK]** need an explicit user decision before work proceeds.

## 1. Repository and document handling

- **Local-only workspace.** PROJECT.md: "Anchor files: local… do not stage or commit them" and "`crowdgpt-v2/` is a local-only, git-excluded workspace… Nothing in it is committed or pushed." Keep all v2 docs here.
- **DRAFT marking.** AGENTS §36.7: LLM-generated docs must "use synthetic/redacted data only," be marked "`DRAFT — requires human review`," and "not [be committed] without human approval." Do not put generated docs in `docs/`.
- **Check before you print.** GOVERNMENT §9.3: check name, path and first line before reading or quoting a file. This applies to `crowdgpt-v2/reference/` (it contains `CROWDGPT_COLLABORATION_AGREEMENT.md`, a legal-style document).
- **No GitHub activity** without explicit approval (AGENTS §2, §36.3). The PROJECT.md remotes (`origin`, `fork`) are not authorization to push.

## 2. Required document formats

- **TODO.md** (skill 06): Legend ✅/🔄/⬜, "Current Sprint / Phase" table `| # | Status | Description |`, "Completed Sprints" table, Notes for decisions and deviations; sub-tasks as `5a`, `5b`. The v2 roadmap (1B → 27B MoE) should use this shape.
- **Estimates** (AGENTS §50.6, skill 06 §6.2): "Never give a time estimate without its basis"; size in agent-hours; list human wait time separately; keep an Estimate Log table in PROJECT.md. Without comparable history, label the number "a guess."
- **Architecture docs** (skill 16): architecture doc, ADRs in an `adr/` folder, OpenAPI at `/docs`. Use the same pattern inside `crowdgpt-v2/`.
- **DEEPDIVE.md trigger** (skill 16): update whenever a change affects "system boundaries, … data flow, dependency assumptions, or operational failure modes."
- **Avoid drift** (skill 16.3): "One home per topic"; "Delete before rewriting"; rank drift by harm. A wrong security claim blocks release.
- **Per-component declarations** (GOVERNMENT §16.1, §17.1, §18.1, §19.1, §20.1, §21.1; FASTPATH Edit Gates): for every significant service (orchestrator, client, ledger, metrics site), declare **degraded mode, rollback constraint, identity dependency, observability mode, trusted artifact import path, sustainment assumption**, or write `N/A` with a reason.
- **Writing style** (BAH §3): plain English, short sentences, active voice; define acronyms on first use; no idioms. BAH §3.7: do not use "BAH" for Booz Allen Hamilton in deliverables.
- **Honest claims** (AGENTS §50.6; GOVERNMENT §11.1): no performance claims without measurements; never call the system "compliant," "authorized," or "zero trust" by branding alone (GOVERNMENT §13.1.3).
- **Review tier** (BAH §5.1): verification, consensus and ledger design are security-sensitive and architecture-heavy, so **Tier 3**: plan first, explicit human review, explicit validation, no acceptance based on AI output alone. Record AI use (BAH §4.5).

## 3. Default tech stack (skill 28), filtered by minimalism (skill 50)

- API/orchestrator: FastAPI + Pydantic + Uvicorn + SQLAlchemy + PostgreSQL; ML: FastAPI + PyTorch; high performance: Go; CLI: Typer or Cobra; real-time: WebSocket for two-way, SSE for server→client (fits live metrics).
- Public web app: Next.js + Tailwind + shadcn/ui; cache: Redis. Avoid MongoDB, Django, plain Flask/Express.
- **Minimalism wins conflicts** (§50.1): v1.5 already uses FastAPI, so reuse it. A static metrics page fed by SSE or JSON polling may meet Rung 3 before Next.js does. Record shortcuts as tradeoff comments with an upgrade trigger (§50.2).
- **Carve-outs that need full rigor** (§50.3): input validation at trust boundaries, data-loss prevention, security, accessibility.
- **New dependencies** (GOVERNMENT §24): answer all five questions (why needed, existing alternative, data touched, external-service/telemetry/approval risk, license) or do not add it. Pin versions.

## 4. Zero-trust verification, identity and consensus

- GOVERNMENT §13.1.4: every significant design must answer: what is the policy enforcement point; what identities are involved; what attributes drive the access decision; what happens when signals are stale or unavailable; how is lateral movement constrained if one identity or workload is compromised. Apply to user-run orchestrators and contributor nodes.
- §13.1.2: prefer strong service-to-service identity over shared static secrets; design logs so reviewers can answer why access was granted, denied or degraded.
- §18.1: every signing key, token issuer or automation credential needs an owner, purpose, expiry expectation and rotation path; test expired certificates, revoked credentials, clock skew. §18.3: tag non-human identities in the audit trail.
- skill 31.6: identity comes from the authenticated session/signature, never from a client-controlled field. Node ID and orchestrator ID come from signatures, not payload fields.
- skill 13: Pydantic validation on all inputs, rate limiting on public endpoints, no secrets in code.
- skill 31.2: audit log fields (actor, action, target, changes, IP, user-agent, timestamp, success/error).

## 5. Ledger, provenance and evidence-first patterns

- skill 68.1: application-owned, verified evidence controls what the system calculates, claims and displays. For v2, the ledger is the source of truth for "who trained what."
- skill 68.3: render tables, charts and downloads from one structured result object. Leaderboards and metrics come from ledger records, never rebuilt separately.
- skill 68.5/68.7: every displayed number exists in verified records; do not present stale data as live; use explicit states (`validating`, `partial`, `unavailable`, `failed`).
- skill 20: for each external integration document rate limits, auth, error codes/retry, health check, timeouts, idempotent retried writes, fallback states, provenance, and validate responses against a closed schema.
- GOVERNMENT §15.2 and §20.1: dependency inventory plus vulnerability, secret, IaC and container scanning; SBOM, provenance attestations, signed releases. Require signed artifacts and hash verification; never use mutable `latest` tags. Checkpoints and shards carry signed manifests (v1.5 already lists per-shard sha256).
- GOVERNMENT §21.1: classify restore points as trusted, suspect or contaminated; after compromise default to rebuild-from-known-good. Maps to rolling back poisoned checkpoints.
- GOVERNMENT §14.2: traceability between requirements, architecture, design, code and test evidence; plan data, model and artifact rights early.
- GOVERNMENT §17.1–17.2: volunteers upgrade at different times — additive interfaces, tolerant readers, strict writers, explicit schema/payload versions; do not couple a release to immediate fleet-wide convergence.
- GOVERNMENT §16.1: define behavior when the orchestrator, ledger, identity or telemetry backend is down: local queueing, resumable sync, idempotent commands, bounded retries, explicit freshness indicators.

## 6. Observability shapes

- skill 41.1: structured JSON logs (`timestamp`, `severity`, `message`, `logger`, `trace_id`, `correlation_id`).
- skill 41.2: OpenTelemetry trace propagation. skill 41.4: `X-Correlation-ID` on every request.
- skill 41.3: Prometheus `/metrics` with `http_requests_total`, `http_request_duration_seconds`, `http_requests_in_progress`, `errors_total`, plus application metrics (rounds, accepted/rejected submissions, audit catch rate, tokens trained).
- skill 41.5: redact PII before logging, including emails, tokens and **IP addresses**. skill 41.6: SLO table (availability, p50/p95/p99, error rate, error budget).
- skill 30: `/health` returns `{status, version, uptime_seconds, timestamp, subsystems:{name:{status, message, latency_ms}}}`; §30.5: report dependency status and last-success separately from process liveness.
- GOVERNMENT §19: separate audit-grade events from debug telemetry; answer who changed what, when, from where, under which identity; never log tokens, keys or raw payloads.

## 7. Data handling for an open public project

- GOVERNMENT §5.1: Public means intentionally approved for public release; unclassified is not automatically public. **[ASK]** Confirm all v2 material (docs, code, data, weights) is meant to be public, and who approves release.
- GOVERNMENT §6: no real sensitive data in examples; use clearly fake values (`TESTUSER_001`, `example.test`); sanitize before logging request bodies.
- GOVERNMENT §7 (least dissemination): the public ledger publishes the minimum identity needed (pseudonymous key IDs, not emails or IPs). Wallet addresses and hardware fingerprints count as identifiers.
- Training corpus licensing and PII are data-owner decisions. **[ASK]**
- No government, customer or program data in this public project.

## 8. Export control (skill 63; not legal sign-off)

- §63.0: classification is a Legal/Export-Compliance determination, not an engineering one. §63.3: "already open source" does not automatically resolve export-control status.
- **[ASK — route to Legal]** (background context, not from the governance files): (a) whether open publication of trained weights (up to 27B MoE) and training code is EAR-exempt as "published"; (b) whether ledger/signing cryptography needs open-source encryption notification; (c) sanctions and geography rules for a global volunteer pool and any token payments; (d) hosting regions.

## 9. External tools, SaaS, crypto and spending

- AGENTS §36.4: no paid services, domains, certificates or anything that could cost money without asking. Covers cloud GPUs, the metrics-site domain, a CDN, and blockchain RPC or fees.
- AGENTS §36.4a: no `.github/workflows/`, Dependabot, Renovate or GitHub Apps without explicit per-workflow approval.
- BAH §4.7: check each exact product is allowed on company devices (AI tools, web storage/sync — relevant to Hugging Face Hub and IPFS-style storage). If status cannot be verified, treat as unavailable.
- **Crypto tokens / chains [ASK]**: financial commitment, new external dependency and data flow, legal/financial/export questions. Out of scope until explicitly approved. Prefer a ledger that needs no token (signed append-only log, Merkle-root publication) per §50.1 Rung 1.
- BAH §4.2: **[ANSWERED 2026-11-17]** v2 is a community open-source project, not Booz Allen work product. The owner contributes as an individual employee (like company engineers on the Linux kernel). A future internal Booz Allen deployment on company hardware is possible but not certain, so the design should be able to run as a private permissioned network. Owner to confirm any company outside-activity or open-source contribution approval through normal channels.
- skill 70 / AGENTS §36.8: never change a live environment without showing the exact command, target and identity.

## 10. Process for writing the doc set (skill 22)

- Default to sequential handoff: research → architecture → component specs → roadmap, with a check at each handoff.
- Define termination criteria: max iterations, consensus threshold, escalation path, context snapshot.

## 11. Consolidated "ask the user first" list

1. Release approval and data class for every v2 artifact; company relationship and OSS approval.
2. Any external service: Hugging Face, blockchains, cloud GPUs, CDN, domain/DNS, hosted AI tools.
3. Any spending, including transaction fees, domains, certificates and paid runners.
4. Any GitHub-side automation, PRs, issues or pushes.
5. Export-control review of weights, crypto code, volunteer geography and tokens (Legal).
6. Licenses for code, weights and data; training-corpus PII and licensing owner.
7. Any new dependency that fails the five questions in GOVERNMENT §24.
8. Any live-environment change.
9. Escalate anything on the GOVERNMENT §28 trigger list (unclear approval boundary, new external data flow, undefined degraded or rollback behavior).
