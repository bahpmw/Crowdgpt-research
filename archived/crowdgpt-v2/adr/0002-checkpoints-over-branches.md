DRAFT — requires human review

# ADR 0002 — Committee checkpoints over provisional branches

Status: Proposed

## Context

R1 wants training to continue while any orchestrators survive. Classic
fault-tolerant agreement (for example CometBFT) stops when more than one
third of voting power is offline. Bitcoin-style heaviest-chain agreement
keeps going but needs a hard-to-fake weight measure and only gives
probabilistic finality.

Model training differs from payments in one useful way: two copies of a
model trained separately from the same starting point can be merged
afterwards (DiLoCo outer steps, model averaging), with the merge checked by
held-out loss. A fork loses no progress if it can be merged.

## Decision

- Orchestrators keep producing provisional rounds on their branch whenever
  the committee is unreachable.
- A committee chosen from the record signs checkpoints; two thirds of its
  signatures make history final.
- If no checkpoint appears for a recovery period, orchestrators with recent
  heartbeats form a recovery committee after a time lock.
- On reconnection, fork choice picks the branch with the most final
  checkpoints, then the most verified work; the other branch's net change is
  merged as a gated outer step.

## Consequences

- Training liveness survives large outages; finality is strong only while
  the committee holds.
- Two partitions recovering independently can both finalise; credit must be
  reconciled by the merge entry. Acceptable because no money is at stake
  (`adr/0001`).
- Merging fails when branches drift too far; branch length needs a limit
  measured in Phase 2 chaos tests.
- More code than a single-leader design; the recovery path is the riskiest
  part and gets dedicated tests.

## Alternatives

Single leader plus witnesses (single point of failure); pure BFT chain
(halts); pure heaviest chain (weak finality, unclear weight measure).
