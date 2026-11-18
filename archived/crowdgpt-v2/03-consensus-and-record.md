DRAFT — requires human review

# 03 — Consensus and the Permanent Record

Answers requirement R3: orchestrators reach decisions on their own, fix
problems without people, and keep an immutable record of what was trained,
when, and by whom.

## 1. What the network decides without people

| Decision | Made by | Basis |
|---|---|---|
| Which updates count this round | Verifiers and orchestrators | Checks in `02` |
| The next weights | Every orchestrator, independently | Deterministic aggregation; mismatching hashes are rejected |
| Whether a round is rejected and rolled back | Every orchestrator | Held-out and canary gate |
| Dispute outcomes | Verifier panels, then committee | Bisection replay |
| Reputation, probation, caps | Every orchestrator | Pure function of the record |
| Committee membership | Pure function of the record and beacon | Verified work, uptime, caps |
| Orchestrator admission and ejection | Pure function of the record | Observer track record; equivocation proofs; inactivity |
| Epoch pivots | Pure function of the record | Triggers in the genesis roadmap (`05`) |
| Software version activation | Orchestrator signalling | Threshold or deadline (§6) |

**Still needs people:** changing the roadmap itself, approving a new dataset,
signing releases, and legal or policy matters. All of those go through
proposals recorded on the record (§6) so the network can enforce the result.

## 2. The record

An append-only log of signed entries in canonical JSON, each naming its
parent's hash. Merkle roots summarise long lists. Every orchestrator keeps a
full copy; anyone can rebuild it from the genesis hash and mirrors.

### Entry types

- `genesis` — first entry: roadmap, epoch 0 config, first committee and
  release-signing keys, held-out commitments.
- `register`, `rotate_key`, `revoke_key` — identities.
- `round` — one model update (below).
- `checkpoint` — committee signatures that make rounds final.
- `merge` — reconciles a branch (§3.4).
- `verdict`, `dispute`, `dispute_result` — verification outcomes.
- `rollback` — marks checkpoints suspect or contaminated and names the
  rebuild point.
- `signal`, `proposal`, `activation` — upgrades and roadmap changes.
- `epoch_start` — a pivot (`05`).
- `heartbeat` — orchestrator liveness, used by committee recovery.

### A `round` entry answers "what, when, and by whom"

```
round:
  parent_hash, round_number, epoch_id, branch_id
  proposer_key, timestamp_utc
  beacon: {assignment_value, audit_value, source}
  model_before_hash, model_after_hash          # exact fp32 hashes
  optimizer_state_before_hash, optimizer_state_after_hash
  epoch_config_hash, software_version
  accepted: Merkle root over
    {contributor_key, assignment_id, shard_ids, update_hash,
     clipped_norm, weight, verifier_keys, verdict_hashes}
  rejected: Merkle root over {contributor_key, update_hash, reason_code}
  partials: [{orchestrator_key, partial_hash, inputs_root}]
  eval: {held_out_loss, canary_results_hash, gate: pass|fail}
  co_signatures: [{orchestrator_key, signature}]
```

v1's signed round record put post-round weights in both the "before" and
"after" fields and committed to client-supplied checksums and local file
paths. v2 commits to hashes the orchestrators computed themselves.

### Training data provenance

`shard_ids` point to shards pinned in the epoch config, which records each
shard's source, licence, and filter version. Together with the per-update
entries, the record states which data shaped which round. v1 did not record
training data.

### Public anchoring

- **Witness co-signatures:** independent parties (other orchestrators, or
  outside witnesses) sign checkpoint hashes, so a node showing different
  histories to different people is caught. The research points at
  Tessera-style transparency logs for this.
- **Timestamps:** checkpoint hashes can be anchored in Bitcoin through
  OpenTimestamps, which is free and needs no token or wallet. It is still an
  outside service (`DECISIONS.md` D2).

## 3. How orchestrators agree

### 3.1 Options considered

| Option | Keeps running when many nodes drop? | Final answer? | Notes |
|---|---|---|---|
| Single leader plus witnesses (what v1's schemas anticipated) | No | Yes | One point of failure; rejected |
| Classic fault-tolerant agreement (for example CometBFT) | Stops if more than one third of voting power is offline | Yes, fast | Proven; conflicts with "keep going while any orchestrator survives" |
| Bitcoin-style heaviest chain | Yes | Only probabilistic | Needs a costly, hard-to-fake measure of weight; verified training work is the candidate but is harder to check than proof-of-work |
| **Hybrid: committee checkpoints over provisional branches** | Yes | Yes when the committee is healthy | Proposed (`adr/0002`) |

### 3.2 Normal operation

- A committee (size a guess: 7–21) is drawn each term from eligible
  orchestrators, weighted by verified work and uptime, capped per operator,
  using the beacon.
- Each round has a proposer chosen by the beacon from all active
  orchestrators. It assembles partials and proposes the round entry; others
  recompute and co-sign.
- Every N rounds the committee signs a checkpoint. Two thirds of committee
  signatures make everything up to it final.

### 3.3 Outages

- If the committee cannot reach two thirds, orchestrators keep producing
  **provisional** rounds. Contributors keep training. The site shows
  "provisional, not final".
- If no checkpoint appears for a recovery period, the orchestrators with
  recent heartbeats form a **recovery committee** by a fixed rule and
  continue. The entry is flagged, and the switch happens after a time lock
  so a brief outage does not trigger it.
- Two partitions could both recover and both finalise. That is a weaker
  guarantee than a payments system needs, but it does not lose model
  progress, because branches merge (§3.4). It does mean credit totals on the
  two sides must be reconciled, which the merge entry does.

### 3.4 Branch reconciliation

When branches reconnect:

1. **Fork choice:** the canonical branch is the one with the most
   committee-final checkpoints, then the most verified work.
2. **Merge:** the other branch's net change (its tip weights minus the
   fork-point weights) is applied to the canonical tip as one extra outer
   step, clipped and passed through the held-out gate like any other update.
   A `merge` entry lists the branch's round hashes.
3. **Credit:** verified work on the merged branch keeps its credit. Work on
   a branch whose merge fails the gate is credited as verified work but
   marked as not used.

Merging two separately trained copies of a model is well understood for
small divergence (DiLoCo outer steps, model averaging) and becomes risky
when branches drift far apart. Branch length before a forced merge attempt
is a tunable limit; Phase 2 chaos tests measure the safe range.

## 4. Self-healing playbook

| Incident | Automatic response |
|---|---|
| Orchestrator goes offline | Contributors fail over to the next orchestrator in their list; work in flight is re-sent with the same idempotency key |
| Proposer misses its slot | Next proposer by beacon order takes over after a timeout |
| A block's weight hash does not match | Peers refuse to co-sign; proposer's reputation drops; repeated faults lead to suspension |
| An orchestrator signs two conflicting blocks | Anyone can post both signatures as an equivocation proof; ejection is automatic |
| Held-out loss spike | Round rejected; next round starts from previous weights |
| Late-found bad update | Dispute, then rebuild without it (`02` §6) |
| Verifiers disagree | Larger panel, then bisection, then committee |
| Committee loses quorum | Provisional mode, then recovery committee (§3.3) |
| Mirror loses a blob | Clients try other mirrors; orchestrators re-seed from their own copies |
| Old client version | Accepted until the epoch's minimum version; rejected after with a clear reason code |

People are alerted on every automatic action (`04`), but none of these
actions wait for them.

## 5. Joining and leaving as an orchestrator

Like Bitcoin, anyone can run the software. Unlike Bitcoin, there is no
mining cost, so admission needs a different cost that fake identities
cannot cheaply fake:

1. **Observer:** run the orchestrator read-only, recompute every round, and
   publish signed recomputation hashes. Wrong hashes mean the node is
   broken.
2. **Candidate:** after a set number of correct rounds and minimum uptime,
   register as a candidate. The operator's contributor key must have a
   record of verified work (the stake is work, not money; v1's schemas
   called this `bond_rounds`).
3. **Active:** can accept contributors, propose rounds, and count toward
   committee selection. Voting weight grows with verified work and uptime,
   is capped per operator, and decays.
4. **Ejection:** automatic for equivocation, for repeated wrong hashes, or
   for inactivity across an epoch. Ejection is a record entry with the
   evidence attached.

During Phases 2–3 admission is by invitation (a known federation), with
the same rules running so they can be tested before they are relied on.

## 6. Upgrades and changes to the plan

- **Software releases** are signed by a threshold of maintainer keys named
  in genesis. Orchestrators choose to run them.
- **Activation:** a release that changes rules carries a version bit.
  Orchestrators signal readiness in round entries. The change activates when
  a threshold (for example 90% of a signalling window) is reached, or at a
  hard deadline. This is the Bitcoin BIP9 and BIP8 pattern.
- **Roadmap changes** (`05`) need a recorded proposal, a supermajority of
  active orchestrators signalling, and a delay before taking effect, so a
  small group cannot rush a change.
- **No single stop switch.** Maintainers can publish advisories and decline
  to sign releases. Halting the network takes a committee supermajority
  vote recorded on the record. A single emergency key is a capture risk
  (research risk #8).

## 7. Why not a blockchain with a token

A tokenless signed log does everything R3 asks for: an immutable,
independently checkable history of what was trained, when, and by whom. A
token adds securities, tax, money-transmission, and speculation problems and
attracts attackers whose goal is the token rather than the model. Covenant
left Bittensor in April 2026 over governance. Proposed: no token, no
wallets, non-transferable credit (`adr/0001`, owner decision `DECISIONS.md`
D3).
