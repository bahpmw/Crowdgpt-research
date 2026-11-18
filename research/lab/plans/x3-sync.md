DRAFT — requires human review

# X3: sync shape, merge rule, work units and inner beta2 for CrowdGPT v2

Experiment X3 from `research/papers/SYNTHESIS-round2-addendum.md` (ideas 9, 10, 3, 4 there; G2 mining top-5 #1-#4).
Code: `research/lab/methods/hsync.py` (new; no existing method file or the harness was changed).
Arms: `research/lab/arms/x3-sync.txt`, run with `sh round1.sh arms/x3-sync.txt`.
Results: `research/lab/results/x3-*.json`, logs `research/lab/logs/x3-*.log`.

## Design

Common to every arm:
- 10m preset (8.39M params incl. tied embedding), real FineWeb-Edu tokens, seq 128, IID data.
- Every replica runs H=30 local AdamW steps (lr 3e-3, betas 0.9/0.95, clip 1, wd 0, AdamW state kept per replica).
- Replica -> node messages are SparseLoCo: Delta = theta_node - theta_local, error feedback (beta 0.95),
  chunk Top-64 of every 64x64 block, 2-bit values (methods/sparseloco.py `compress`), outer SGD alpha=1.
  Message = 214.5 KB per replica per round (0.20 bits/param).
- Fixed budget: 9,830,400 training tokens per arm (all replicas together), so arms compare at equal compute.
- Byte accounting per replica per round, object-store model: upload your message once, download every other
  message of your node.

Sync shapes (Q1, seq 128 x batch 4 per replica step, so 32 replicas = 20 rounds and 64 replicas = 10 rounds):
- **flat**: one global model; all W replicas merge every round (plain SparseLoCo at W replicas).
- **hier**: groups of 16 replicas. Inside a group: exactly flat (full mean of the 16 compressed messages).
  Between groups: sparse value gossip with 1 (or 2) partner groups per round chosen by a seeded "beacon" random
  matching that changes every round.
- **gossip** (GASLoC-like): every replica is its own node, keeps its full local update and mixes only through
  sparse value gossip with 2 beacon-chosen peers per round.
- Sparse value gossip (SPICE-style, exact form assumed): each node keeps an accumulator a = 0.95 a + (own change
  this round), sends its current parameter values on the chunk Top-`gk` coordinates of |a| (bf16 value + 12-bit
  index; gk=64 -> 408 KB, gk=512 -> 3.3 MB per message), and zeroes a there. The receiver averages those
  coordinates 50/50 with its own. 1-D tensors are averaged dense. In hier, a group's gossip message is uploaded
  once (relay) and downloaded by every member of the partner group.
- Reported loss = eval of the average of all node models (needs one final all-reduce); `node0Loss` = a single
  node's model with no final sync; `spread` = mean ||x_node - x_mean|| / ||x_mean||.

Merge rules (Q2, flat 16 replicas, batch 8, 20 rounds):
- **mean**: equal-weight mean.
- **mdir**: clip every message to the lower-median message norm, take the mean, rescale the non-embedding part
  of the mean to that median norm (tied embedding keeps the clipped mean, per Decoupled DiLoCo App. D.2).
- Attacks: byz = 2 of 16 (1-of-8) send -5 x their message (sign flip, c=5 as in earlier lab runs);
  lazy = 4 of 16 (25%) send 0.01 x their message (near-zero).
- mdir with alpha=0.5 as a step-size control (mdir makes the step ~sqrt(M) larger than mean).

Work units (Q3, flat 16, batch 8): 4 of 16 workers run at 1/4 speed.
- sync: everyone does 30 steps; the round waits for the slowest (sim time 120 per round).
- budget, equal weight: each worker does round(30 x speed) steps (slow = 8), equal-weight merge.
- budget, token weight: same work, merge weighted by tokens (Decoupled DiLoCo).
- units: work is cut into equal-token 8-step units, each started from the round's weights; fast workers do
  4 units, slow ones 1; equal weights. Sim time 30 per round for budget and units.

Inner optimizer (Q4, flat 16): AdamW beta2 0.95 vs the replica rule beta2 = 0.95^(1/16) = 0.9968.

RESULTS_PENDING
