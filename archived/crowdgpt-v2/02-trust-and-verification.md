DRAFT — requires human review

# 02 — Trust and Verification

Answers requirement R2 ("zero trust": verify everything, keep bad actors
from poisoning the model). "Zero trust" here means a concrete property, not
a label: **no message, update, verdict, or block is accepted because of who
sent it; each is accepted only after a check that someone other than the
sender chose after the sender committed.**

## 1. Who we defend against

| Attacker | Goal | Main defences |
|---|---|---|
| Fake-identity farm (Sybil) | Outvote checks, capture weight or orchestrator seats | Reputation from verified work only, per-identity caps, probation (§2) |
| Poisoner | Steer the model or plant a backdoor | Network-assigned data and seeds, replay checks, clipping, held-out and canary gates, rollback (§3–§6) |
| Free-rider | Get credit without doing the work (copy, random noise, partial work) | Per-key unique assignments, replay of randomly chosen parts (§3) |
| Seed or data grinder | Pick inputs that make a bad update look good | Assignments fixed by a beacon the contributor cannot influence |
| Dishonest orchestrator | Censor contributors, favour friends, fake verdicts, publish a wrong aggregate | Other orchestrators recompute everything; contributors can go elsewhere (§7) |
| Lazy or lying verifier | Rubber-stamp or block work | Two or more independent verifiers, hidden known-answer jobs (§5) |
| Mirror tampering | Serve bad weights or data | Everything content-addressed and hash-checked against the record |
| Supply chain | Ship a malicious client or orchestrator build | Signed releases, reproducible builds as a later goal, version recorded per submission |
| Flooding (denial of service) | Exhaust orchestrators or verifiers | Signed requests, per-key and per-address rate limits, bounded replay sizes (v1.5's audit had unbounded replay parameters on the event loop) |

Out of scope for now: nation-state network attacks, hardware-level
attacks on contributors' machines, attacks on the people running
orchestrators.

## 2. Identity and Sybil cost

- Every contributor, verifier, and orchestrator is an **Ed25519 key pair**
  generated locally. Identity comes only from signatures; payload fields
  such as `clientId` are never trusted (v1.5 used a free unsigned string).
- Registration is a signed record entry. Key rotation and revocation are
  also record entries, signed by the old key, so the history of every key
  is public.
- **New keys start on probation:** low weight cap, verification jobs and
  small training assignments first. Weight and eligibility grow only with
  verified work, and decay when idle. Creating a thousand keys gets a
  thousand probationers, not a thousand votes.
- **Caps:** no single key, and no group of keys that share an orchestrator
  and registered in the same window, can supply more than a set share of a
  round's accepted weight. v1 had 10 accounts supplying 88.7% of tokens;
  caps make that concentration visible and bounded.
- Optional stronger identity (for example linking a GitHub account) is an
  owner decision because it trades privacy for Sybil resistance
  (`DECISIONS.md` D6).

Public outputs show key IDs only. IP addresses, wallet addresses, and
hardware fingerprints are identifiers and never appear in public outputs.

## 3. Checking an update

Checks are layered; each one is cheap enough to run on every update or on
a random sample, and each one closes a gap the one before leaves.

| Layer | Check | Catches | Cost |
|---|---|---|---|
| L0 Shape | Signature, nonce, version, assignment ID matches the beacon-derived assignment for this key, size and norm within epoch bounds | Garbage, replays, wrong assignment, giant updates | Trivial |
| L1 Commit first | Update hash and Merkle root of per-sample scalars (DUST) or per-step losses (backprop) signed before the audit beacon is revealed | Choosing what to fake after seeing what is checked (the attack that broke Proof-of-Learning) | Trivial |
| L2 Replay | Verifiers chosen by the audit beacon replay randomly chosen samples and layers and compare to the commitment | Copying, noise, partial work, steering | Tunable fraction of the contributor's work (Phase 0 measures) |
| L3 Usefulness score | Does the update lower loss on its assigned data more than on unrelated data, and not raise held-out loss | Honest-looking but useless or harmful updates | One forward pass on small batches |
| L4 Aggregation limits | Clip each update relative to the round's median norm; fixed-point sum; robust combination across orchestrators; per-identity share cap | Whatever slips through L0–L3, limited to bounded damage per round | Trivial |
| L5 Model gate | Hidden held-out set and canary probes on the new weights; reject the round on a spike | Round-level damage | One eval per round |
| L6 Dispute and rebuild | Anyone can challenge an accepted update during the dispute window; if upheld, rebuild later checkpoints without it | Anything found late | Rare, aggregation-only recompute |

### L2 details by lane

**DUST lane.** v1.5's audit already replays a few salt-chosen layers and
compares by cosine similarity. v2 changes four things:

1. The salt becomes the public **audit beacon** revealed after commitment,
   so anyone can re-run the audit (v1.5's salt was a secret per process and
   not reproducible).
2. The audit covers every training mode that ships (v1.5 skipped
   real-token data, device-generated noise, error feedback, draw averaging,
   local steps, and cross-device bf16). Anything that cannot be audited is
   not allowed on the network.
3. The comparison tightens from cosine ≥ 0.5 (which lets an attacker
   control most of the update's direction) to a tolerance measured in
   Phase 0 from honest cross-GPU replays.
4. Token counts are recomputed from the assignment, never self-reported.

**Backprop lane.** Exact replay of backprop across different GPUs is not
bit-for-bit. The plan uses replay of randomly chosen steps with a loss
tolerance, plus the L3 usefulness score. This is scoring, not proof; the
research found no live system that proves pretraining gradients. Weight
caps and L4 matter more in this lane.

**Local-module lane (SOLO-style, `adr/0004`).** Each module has its own
loss, so a verifier can replay a sampled step of one module from its signed
input activations, its weights, its assigned data, and the frozen
output-layer copy, without the rest of the model. Like the backprop lane it
uses local backprop, so replay matches only within a measured tolerance
(`TODO.md` 0.18), and caps plus L4 still matter.

**Weight-space evolution strategies (optional lane).** If an update is
fully determined by public seeds plus per-sample scalars (EGGROLL style),
any orchestrator can rebuild the whole update from the scalars without
the data, and contributors only choose scalars. This is the strongest
verification story available and is worth a Phase 0 experiment.

**Split models (pipeline crews and paths, `adr/0005`).** Each stage signs
the hash of the activations it received and the activations it sent, plus
its update. Because a stage's output depends only on its input, its weights,
and its seeded noise, a verifier holding only that stage can replay one hop,
which is far cheaper than replaying a whole model. Two extra risks:

- A dishonest stage corrupts every stage after it in that batch. Mismatched
  hashes between neighbours point to the culprit: if stage k's signed
  output does not match stage k+1's signed input, one of them lied, and a
  replay of stage k settles it. The crew's batch is dropped and only the
  guilty key is penalised.
- Stages can collude within a crew. Crew membership comes from the
  assignment beacon, not from contributors' choice, and caps limit how many
  stages of one crew a single identity or group can hold.

Independent paths are checked like full-model updates of a smaller model.

### The rounding-noise question

DUST's signal is a small difference between two antithetic losses.
Different GPUs round differently. If honest rounding differences are close
in size to the signal, replay checks must either be loose (which reopens
cheating) or use deterministic kernels (the research cites a 30–60%
slowdown, unverified). **Phase 0 measures this before any trust design is
locked.**

## 4. Data poisoning

Contributors never choose data. Shards are pinned by hash in the epoch
config and assigned by the beacon. This moves the poisoning risk to dataset
curation, which is handled once per epoch, in the open, with:

- source and licence recorded for every shard;
- a published filtering pipeline that anyone can re-run;
- shard selection approved through the roadmap process (`05`).

## 5. Verifiers

- At least two verifiers per sampled check, with different keys, not on the
  contributor's orchestrator, chosen by the audit beacon.
- **Known-answer jobs:** orchestrators slip in replay jobs whose correct
  answer they already know (including deliberately bad updates). A verifier
  that approves a known-bad job loses reputation. This catches verifiers
  that approve everything without doing the work.
- **Disputes:** if verifiers disagree, a larger panel replays the disputed
  piece; if still split, the replay is narrowed step by step to a single
  layer and batch (bisection), which a committee replays in full. The loser
  loses reputation. No money is at stake.

## 6. Model-level protection

- **Held-out set:** a fresh slice per epoch, committed by hash at epoch
  start and revealed at epoch end, so nobody can train on it. v1's held-out
  gate was off by default, failed open, and used a set published in the
  repo.
- **Canary probes:** fixed prompts checked every round for sudden behaviour
  changes. Backdoors are not reliably detectable by any known method; the
  research found no validated defence for decentralised LLM training as of
  Nov 2026. Canaries reduce the risk; they do not remove it.
- **Checkpoint states** (from the governance overlay): `trusted`,
  `suspect`, `contaminated`. A spike or an upheld dispute marks checkpoints
  suspect; confirmed contamination triggers a rebuild from the last trusted
  one.
- **Rebuild is cheap because aggregation is deterministic.** Updates are kept
  for the dispute window. To remove a bad update, re-run aggregation from the
  last trusted checkpoint without it. Later updates were computed against
  slightly different weights; DiLoCo-style training already tolerates that
  kind of staleness, and the L5 gate re-checks the result.

## 7. Checking orchestrators

- Assignments are deterministic, so any party can check an orchestrator
  gave the right assignment.
- Partial sums list their inputs by Merkle root; peers re-download a random
  sample of the inputs and confirm each one passed verification and is
  included correctly.
- The final combination and outer optimizer step are recomputed by every
  orchestrator; a round block with a wrong weight hash is rejected.
- Contributors may send the same commitment to several orchestrators. If
  one ignores valid commitments, it shows up in its public acceptance rate
  and the contributor's work still lands through another.

## 8. Zero-trust checklist (governance §13.1.4)

| Question | Answer |
|---|---|
| Where is each decision enforced? | At every receiver: contributors check weights and assignments, orchestrators check commitments and blocks, verifiers check updates, the indexer checks the record |
| Which identities exist? | Ed25519 keys for contributors, verifiers, orchestrators, release signers; nothing else |
| Which attributes drive decisions? | Reputation, probation state, caps, and role eligibility, all derived from the record |
| How are stale signals handled? | Every assignment, commitment, and verdict names its round and expires; old reputation decays |
| How is lateral movement limited? | Roles carry no implicit trust; a compromised orchestrator key can sign wrong blocks, which peers reject, but cannot change the record others hold |

## 9. Key management

| Key | Owner | Expiry | Rotation and revocation |
|---|---|---|---|
| Contributor | Volunteer | None by default; reputation decays when idle | Signed rotation entry; revocation by old key or by committee after proven compromise |
| Orchestrator | Operator | Re-registration each epoch | Rotation entry; committee can suspend on proven misbehaviour |
| Committee checkpoint | Each member's orchestrator key | Committee term | Term-based |
| Release signing | Maintainers (2-of-N threshold proposed) | Yearly | Published in the record; genesis pins the first set |

Audit log fields for every orchestrator decision: timestamp (UTC), round,
actor key ID, action, target ID, outcome, reason code, correlation ID.
