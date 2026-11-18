DRAFT — requires human review

# 01 — Architecture

Answers requirement R1 (distributed by design, scale by adding
orchestrators). Trust rules are in `02`, the record and decision-making in
`03`.

## 1. Roles

| Role | Who runs it | Hardware | What it does |
|---|---|---|---|
| **Contributor** | Volunteers | Any supported GPU (CUDA, ROCm, Apple MPS); CPU for verification-only work at small sizes | Fetches an assignment, trains, uploads an update, runs verification jobs when assigned |
| **Orchestrator** | Volunteers who pass admission (`03` §5) | CPU box with good bandwidth and disk; a GPU is optional and lets it do its own spot checks | Hands out assignments, accepts commitments and uploads, schedules verification, computes its share of the aggregate, proposes and co-signs rounds, gossips with peers, stores the record |
| **Mirror** | Anyone | Disk and bandwidth | Serves content-addressed blobs (weights, updates, data shards). Untrusted: every blob is checked against its hash in the record |
| **Indexer** | Anyone; the project runs one for the site | CPU, database | Reads the record and live events, serves the public read API and the website (`04`) |
| **Beacon** | Derived, not a server | — | Public randomness nobody can predict or bias alone (§4) |

One machine may hold several roles. Roles never imply trust: an orchestrator
is checked by other orchestrators exactly as a contributor is checked by
verifiers.

## 2. Round lifecycle

A **round** is one model update. An **epoch** is a long run of rounds with a
fixed model shape and training lane (`05`).

1. **Open.** The record states the current model hash, optimizer-state hash,
   epoch config hash, and the round's assignment beacon value.
2. **Assign.** A contributor asks any orchestrator for work. The assignment
   (data shard IDs, noise seed, layer slice, batch size) is a pure function
   of `(beacon, contributor public key, slot number)`. Every orchestrator
   computes the same assignment, so contributors cannot shop for an easy or
   favourable one, and an orchestrator cannot quietly favour a friend.
3. **Train.** The contributor downloads the model (bf16 delta from the last
   checkpoint it holds, verified by hash) and the assigned shards, trains,
   and produces an update.
4. **Commit.** Before the commit deadline the contributor signs and sends a
   commitment: update hash, Merkle root of its per-sample training scalars
   (DUST lane) or per-step loss trace (backprop lane), assignment ID, client
   version. The blob goes to a mirror or a presigned upload slot; only the
   hash travels through orchestrators.
5. **Reveal randomness.** After the deadline, the audit beacon value is
   revealed. It picks, for each accepted commitment, which samples, layers,
   or steps get replayed, and which verifiers replay them (at least two,
   different keys, not on the contributor's orchestrator).
6. **Verify.** Verifiers replay the selected pieces and sign a verdict.
   Disagreements go to dispute resolution (`02` §5).
7. **Aggregate.** Each orchestrator clips and sums the verified updates it
   took in, in fixed-point arithmetic in a fixed order, and publishes the
   partial sum with a Merkle list of its inputs. The round's update combines
   the partials with a robust rule across orchestrators (§3). The
   outer optimizer step is deterministic, so every orchestrator computes the
   same next weights and weight hash.
8. **Gate.** The new weights are scored on the epoch's held-out evaluation
   set and on hidden canary probes (`02` §6). A loss spike rejects the round
   and the record moves forward from the previous weights.
9. **Record.** The proposer for the round writes a round block (`03` §2)
   and gossips it. Others recompute and co-sign. Committee checkpoints make
   rounds final.

In epochs that split the model (`05` §5, `adr/0005`), step 2 assigns a
whole **crew**: one or more contributors per pipeline stage, chosen by the
beacon. Steps 3–6 happen per stage. Activations travel directly between
crew members, and each stage commits to its own update and hop hashes. The
orchestrator treats the crew's combined update as one contribution but
records and credits each stage separately.

## 3. How adding orchestrators adds capacity

v1 and v1.5 both funnel every upload through one server, and the v1.5
coordinator holds a full fp32 vector per pending client and reloads the
weights on every submit. That does not scale past a handful of contributors
at 1B.

v2 splits the work so each orchestrator handles a slice:

- **Intake is partitioned.** Contributors pick an orchestrator (nearest,
  least loaded). Each orchestrator only receives commitments, not blobs.
- **Verification is spread across contributors.** Replay jobs are assigned
  by the beacon to contributor GPUs, so verification capacity grows with the
  crowd, not with the orchestrator count.
- **Aggregation is two-level.** Per-update clipping and a fixed-point sum
  split cleanly across orchestrators, so each orchestrator only downloads
  the updates it accepted. The cross-orchestrator step works on one partial
  per orchestrator, which is small. Treating orchestrators as the units of a
  robust rule (trimmed mean or median over partials) also limits damage
  from a dishonest orchestrator.
- **Blobs never touch orchestrators.** Mirrors and peer-to-peer transfer
  carry weights, updates, and data, as v1's presigned uploads did.

Tradeoff: robust rules that look at individual updates (Krum, coordinate
median across all contributors) do not split this way. The plan accepts
that and relies on clipping plus verification at the update level and
robust combination at the orchestrator level. Revisit if measured poisoning
tests show the gap matters.

## 4. Randomness beacon

Assignment and audit choices must be unpredictable before commitment and
checkable after. Options, simplest first:

1. **Record-derived:** hash of the last final checkpoint plus the round
   number. Free and simple, but the checkpoint's proposer can try several
   blocks and pick a favourable hash ("grinding").
2. **Orchestrator commit-reveal:** each committee member commits a secret,
   then reveals; the beacon is the hash of all reveals. A member who refuses
   to reveal is penalised; the last revealer can still choose to abort.
3. **Threshold signature beacon** among the committee (unbiasable while
   most members are honest), more code.
4. **External public beacon** (drand / League of Entropy): unbiasable and
   free, but an outside dependency (`DECISIONS.md` D2).

Proposal: start with (2) plus (1) as a fallback, mix in (4) if approved.
See `adr/0003`.

## 5. Networking

Decision ladder applied:

- **Phases 1–2 (known peers):** HTTPS between orchestrators with signed
  messages and a static peer list. No new networking dependency.
- **Phase 4 (open admission):** a peer-to-peer gossip and blob layer. The
  research names iroh (gossip plus content-addressed blobs) and libp2p as
  candidates; picking one needs the five dependency questions from
  GOVERNMENT §24 and an ADR.

## 6. Storage

| Data | Source of truth | Format |
|---|---|---|
| Record (rounds, checkpoints, registrations, verdicts) | Append-only, content-addressed files replicated on every orchestrator | Canonical JSON, Ed25519 signatures, hash chain, Merkle roots |
| Weights and optimizer state | Content-addressed blobs on mirrors, hash in the record | Exact fp32 hash for checkpoints; bf16 deltas for transfer |
| Updates | Content-addressed blobs, kept for the dispute window and then archived or dropped by policy | v1.5 NGD1 codec (adapted) |
| Training data | Content-addressed shards pinned in the epoch config | v1.5 tensor-aligned shards |
| Local query index | Rebuildable from the record | SQLite for orchestrators (easy home deployment); Postgres for the public indexer |

Hugging Face becomes one mirror among several rather than the database.

## 7. Tech stack (reference implementation)

- Python 3.10+, FastAPI and Pydantic (already used in v1.5), PyTorch for
  training and replay.
- Ed25519 via the `cryptography` package (v1's hand-written verifier is not
  carried forward).
- New dependencies only through the GOVERNMENT §24 five questions, pinned.
- A faster orchestrator in Go or Rust is a later option if profiling shows
  Python is the bottleneck. Not before.

## 8. Per-component declarations

Required by the governance overlay for every new component.

| Component | Degraded mode | Rollback constraint | Identity dependency | Observability | Trusted artifact import | Sustainment assumption |
|---|---|---|---|---|---|---|
| Contributor client | Queues finished work locally if no orchestrator answers; retries with the same idempotency key; abandons work older than the round's commit deadline | Old clients must keep working within an epoch; payload versions are explicit | Contributor's own Ed25519 key, generated locally | JSON logs locally; optional signed self-telemetry, labelled as self-reported | Weights and shards checked against record hashes; client releases signed | Runs unattended for days; auto-update is opt-in |
| Orchestrator | Keeps assigning, verifying, and producing provisional rounds on its own branch when peers or the committee are unreachable | Record is append-only; a bad round is reverted by a later round, never by deleting history | Operator's Ed25519 key, registered in the record; rotation through the record | JSON logs, Prometheus `/metrics`, `/health` with subsystem states, correlation IDs; no IPs in public outputs | Releases signed; config pinned by hash in the record | Volunteer-run; must survive restarts by replaying its local record |
| Record | Branches allowed; reconciled by fixed rule (`03`) | Never rewritten; rollbacks are new entries | Signatures only; no account system | Every entry is public and checkable | Genesis hash pinned in client and orchestrator releases | Must be readable by a fresh node with only the genesis hash and mirrors |
| Mirror | Any mirror can vanish; clients try others | None (content-addressed) | None (hash-checked) | Optional | Hash in record | Best effort |
| Indexer and site | Shows "stale" or "unavailable", never old data as live | Rebuildable from the record | None for reads | Freshness on every view (`04`) | Reads only signed record data plus labelled telemetry | One project instance; anyone can run another |
| Beacon | Falls back to record-derived value, flagged in the round block | n/a | Committee keys | Each round block names the beacon source used | n/a | n/a |
