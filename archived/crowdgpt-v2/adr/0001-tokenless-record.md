DRAFT — requires human review

# ADR 0001 — Tokenless signed record instead of a cryptocurrency

Status: Proposed (owner decision D3)

## Context

R3 asks for an immutable record of what was trained, when, and by whom,
maintained by orchestrators without a central server. Bitcoin and most
decentralised-training projects (Bittensor, Gensyn, Psyche on Solana) pair
their record with a token.

## Decision

Use an append-only, hash-chained, signed log (canonical JSON, Ed25519,
Merkle roots), replicated on every orchestrator, with committee checkpoints
and optional witness co-signatures and OpenTimestamps anchoring. No token,
no wallets. Contributor credit is non-transferable and derived from verified
work on the record.

## Consequences

- Removes securities, tax, money-transmission, and speculation exposure,
  and the class of attackers who want the token rather than the model.
- Loses money as a Sybil cost and as a reward. Sybil cost comes from
  verified work, probation, and caps instead (`02` §2, `03` §5); the reward
  is credit, reputation, and access to the model.
- Can still anchor to Bitcoin for timestamps without holding coins.
- Revisit if volunteer numbers stall and the owner decides paid incentives
  are wanted; that would need its own legal review.
