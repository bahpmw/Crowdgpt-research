DRAFT — requires human review

# ADR 0003 — Network-assigned work from a post-commit randomness beacon

Status: Proposed

## Context

In v1.5 contributors choose their own seed and data cursor, and the replay
audit uses a secret per-process salt. That allows grinding (trying inputs
until a bad update looks good) and makes audits impossible for anyone else
to repeat. Proof-of-Learning was broken by attacks that exploit knowing what
will be checked.

## Decision

- Assignments (data shards, noise seed, layer slice) are a pure function of
  `(assignment beacon, contributor key, slot)`.
- Contributors commit to their update and per-sample scalars before the
  audit beacon is revealed; the audit beacon then picks what is replayed
  and by whom.
- Beacon source, in order of preference once available: committee
  commit-reveal mixed with drand (if approved, D2); committee commit-reveal
  alone; record-derived hash as the fallback. Each round entry records which
  source it used.

## Consequences

- Any party can recompute assignments and audits.
- Contributors cannot pick favourable data or seeds.
- The last revealer in commit-reveal can still choose to abort; penalised by
  reputation, removed if drand or a threshold scheme is adopted later.
