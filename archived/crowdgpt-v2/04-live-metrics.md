DRAFT — requires human review

# 04 — Live Metrics and the Website

Answers requirement R4: the website shows the real status of each
contributor, their work, and the whole system, fast enough to react.

## 1. Two kinds of numbers, never mixed

| Kind | Source | Shown as |
|---|---|---|
| **Verified** | The signed record and signed orchestrator events | Default everywhere; drives leaderboards and credit |
| **Reported** | Contributors' own optional telemetry (GPU model, tokens per second, temperature, client version) | Labelled "self-reported", shown separately, never ranked |

v1's leaderboard ranked self-reported token counts. v2 ranks only verified
work.

## 2. Every number has a state

Each value on the site carries one of: `live`, `provisional` (on the record
but not yet final), `final`, `validating`, `stale` (older than its freshness
limit), `unavailable`, `failed`. A stale value is never shown as live; the
page shows its age and a banner instead.

## 3. Data path

```
orchestrators ──signed events──┐
                               ├─► indexer ──► public read API ──► website
record (gossip, mirrors) ──────┘     │          (REST + Server-Sent Events)
                                     └─► JSON snapshots on mirrors/CDN
```

- **Signed events** are short messages orchestrators already produce:
  assignment issued, commitment received, verdict signed, round proposed,
  round co-signed, checkpoint signed, automatic action taken. The indexer
  subscribes to several orchestrators, checks signatures, and drops
  duplicates. One orchestrator going quiet does not blank the site.
- **The record** is the authority. Event-derived numbers are `live` until
  the matching record entry arrives, then `provisional` or `final`.
- **Read API:** REST for pages and history; Server-Sent Events (SSE) for
  live updates. SSE is plain HTTP and needs no extra dependency.
- **Snapshots:** every round the indexer writes static JSON files that any
  mirror or CDN can serve, so the site survives the indexer being down (it
  shows the snapshot's age).
- Anyone can run an indexer. The project's site is one instance.

## 4. Views

**Network overview**
- Current epoch, model size, training lane, round number, model hash.
- Held-out loss curve (verified), tokens verified this epoch, progress
  toward the next milestone trigger (`05`).
- Contributors active now (training, verifying, idle), orchestrators active,
  committee health, finality lag (rounds since last checkpoint), beacon
  source in use.
- Acceptance and rejection rates by reason, verifier disagreement rate.
- Share of accepted weight held by the top 10 keys (v1 reached 88.7%).

**Contributor page** (by key ID; optional display name signed by the key)
- Current status: `idle`, `assigned`, `training`, `committed`, `verifying`,
  `accepted`, `rejected (reason)`, `disputed`.
- Verified tokens and rounds, verification jobs done, reputation,
  probation state, cap usage, history by round.
- Self-reported telemetry in its own labelled panel.

**Orchestrator page**
- Uptime, contributors served, acceptance rate, recompute agreement with
  peers, partials submitted, proposals made and missed, committee terms,
  any suspensions with evidence links.

**Round explorer**
- Every round entry, its contributions, verdicts, eval result, and
  co-signatures, with links to the raw signed entries so anyone can check
  them.

**Alerts feed**
- Every automatic action from the self-healing playbook (`03` §4), as it
  happens.

## 5. Reacting faster: alerts

Alert rules evaluated by the indexer and by each orchestrator (thresholds
are starting guesses, tuned in Phase 3):

| Alert | Trigger |
|---|---|
| Loss spike | Round rejected by the held-out gate, or loss rises N rounds in a row |
| Finality stall | No checkpoint for longer than the recovery period's first half |
| Orchestrator drop | Active orchestrators fall by more than a third within an hour |
| Rejection surge | Rejection rate doubles against the trailing day |
| Concentration | Any key or group near its cap for several rounds |
| Verifier disagreement | Disagreement rate above its trailing baseline |
| Branching | More than one branch active |
| Indexer lag | Record-to-site lag above one round |

Where alerts go (email, chat webhook, pager) is an owner decision because
each is an outside service (`DECISIONS.md` D2).

## 6. Operator observability

Per the governance overlay's observability standard:

- JSON logs with timestamp, level, round, correlation ID
  (`X-Correlation-ID` carried across orchestrators), no IP addresses.
- Prometheus `/metrics` on every orchestrator and indexer.
- `/health` returning per-subsystem states (record sync, peer gossip,
  beacon, storage, verification queue, aggregation).
- OpenTelemetry traces for the round lifecycle as a later addition.

Service targets (guesses, to be measured): record-to-site lag under one
round at the 95th percentile; signed event to site under 30 seconds.

## 7. Privacy

Public views use key IDs and optional signed display names. No IP
addresses, locations, wallet addresses, or hardware serials. Self-reported
telemetry is opt-in. The record is public and permanent, so contributors
are told before registering that their key's work history will be public.

## 8. Build order

1. Read API and JSON snapshots from the Phase 1 record (one orchestrator).
2. Static site (plain HTML, CSS, and JavaScript) reading the API and SSE.
   No framework until a measured need appears.
3. Alerts.
4. Multi-orchestrator event merging (Phase 2).
5. Accessibility review against the UI guardrails (semantic HTML, keyboard
   access, contrast, screen-reader labels on charts).
