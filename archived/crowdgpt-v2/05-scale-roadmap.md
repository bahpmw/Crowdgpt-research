DRAFT — requires human review

# 05 — Scale Roadmap and Automatic Pivots

Answers requirement R5: a long-term plan built into the system, so that at
each milestone the network pivots everyone's work on its own, the way a
Bitcoin halving changes the rules at a known point without anyone deciding
it on the day.

## 1. The idea in one paragraph

The full milestone table is written into the **genesis entry** of the record
before training starts. Each milestone ("epoch") has public **triggers**.
Progress toward each trigger is visible on the site at all times. When the
triggers are met, every orchestrator independently computes the same next
step: freeze the current model, grow it into the next shape with a fixed
recipe, publish the new starting weights' hash, and switch every
contributor's work to the new epoch's tasks. No meeting, no flag day set by
a person. Changing the table itself is possible but slow and public
(`03` §6).

Unlike a halving, triggers are based on progress and capacity, not on a
clock or block count, because the network cannot know in advance how fast
volunteers will arrive.

## 2. Proposed epoch table

Sizes follow the owner's sketch. Token targets use the common rule of thumb
of about 20 training tokens per parameter for a compute-efficient run
(Chinchilla); treat them as placeholders for the owner and Phase 0 to set.

| Epoch | Model | Shape (to be fixed) | Lane | Token target (placeholder) | Starts from |
|---|---|---|---|---|---|
| E0 Testnet | 10M → 100M → 300M | v1.5 presets | Both, for comparison | Small, enough to rehearse pivots | Random init |
| E1 | 1B dense | v1.5 `1b` preset (15 layers, width 2048) or a re-tuned shape | Chosen by Phase 0 gate | ~20B | Random init, or grown from E0's 300M |
| E2 | 4B dense | Deeper and wider | Gate at E1 end | ~80B | Grown from E1 |
| E3 | 9B dense | Deeper and wider | Gate at E2 end | ~180B | Grown from E2 |
| E4 | ~27B total mixture-of-experts | E3 attention and shared layers plus experts copied from E3's feed-forward layers | Gate at E3 end | Set at E3 end | Upcycled from E3 |

Notes:

- **Tokenizer:** v1.5 uses a 4,096-entry vocabulary, which is small for
  1B and larger. Changing the tokenizer mid-run throws away much of the
  learned embedding, so the tokenizer for E1–E4 should be chosen once,
  before E1 (`DECISIONS.md` D7).
- **Mixture-of-experts is blocked in today's code.** `plain.py` refuses
  module names containing `moe`, `expert`, or `router`. In v2 the allowed
  architecture becomes a per-epoch whitelist in the epoch config instead of
  a global ban.
- **No training method has been shown to pretrain a mixture-of-experts
  model with zero-order / evolution-strategy updates.** E4 is a research
  milestone, not a plan of record.
- **Splitting the model (§5.1–5.2, `adr/0005`)** keeps a 16 GB card useful
  at every epoch. Epochs past E4 (for example a "1T-class" sparse model)
  would grow total parameters through paths and experts, not dense width;
  §5.3 explains why.

## 3. Triggers

An epoch ends when **all** of these hold:

| Trigger | Measured how | Why |
|---|---|---|
| **Training done** | Verified tokens reach the target, or held-out loss improves less than a set percentage over a set window (plateau) | Do not grow a model that is still learning cheaply, or keep training one that has stopped |
| **Capacity ready** | During a readiness window, the network sends **rehearsal jobs** sized for the next epoch. Contributors who complete them correctly are counted. The trigger needs a minimum count in each work role | VRAM and bandwidth cannot be self-reported honestly; doing a real next-size job proves it |
| **Software ready** | At least 90% of active orchestrators signal they run a release that supports the next epoch | Avoids a split where half the network cannot follow |
| **Quality checks recorded** | The end-of-epoch evaluation suite has run and its results are on the record | Every published model has a known scorecard |

**Deadline rule:** if the training and capacity triggers hold but software
signalling lags, the pivot activates at a hard deadline (the Bitcoin BIP8
pattern), so a few holdouts cannot block it. If capacity is missing, the
network stays on the current epoch and keeps improving it; it does not
force a pivot nobody can run.

## 4. What happens at a pivot

All of the following are pure functions of the record, so every
orchestrator gets the same hashes:

1. **Freeze** the final checkpoint of the ending epoch and mark it as a
   release candidate. Publishing weights publicly may need the export
   review in `DECISIONS.md` D4 first; the network can keep training either
   way.
2. **Grow** the model with the recipe fixed in the genesis table:
   - dense to bigger dense: copy layers to add depth, and widen with
     function-preserving expansion, so the new model starts with the old
     model's behaviour;
   - dense to mixture-of-experts: copy each feed-forward layer into several
     experts and add a router (sparse upcycling).
   The result's exact hash is written in the `epoch_start` entry.
3. **Switch configs:** model shape, lane, round length, batch sizes, data
   shards, held-out commitment, verification fractions, caps, minimum client
   version, allowed architecture list.
4. **Re-tier work.** Contributors move into roles based on the rehearsal
   jobs they passed (§5).
5. **Clients switch at the activation round.** The code for the next epoch
   ships during the readiness window, so the switch is a config change, not
   a download.

## 5. Work roles change as the model grows

This is where the "halving" feel comes from: each milestone changes what a
given GPU does for the network. The rule that keeps small GPUs in the
network is a **fixed hardware floor** (proposed: 16 GB of GPU memory plus a
stated minimum upload speed) written into the genesis roadmap. No epoch may
require more. When the model outgrows one card, it is split into more
pieces; the pieces never get bigger (`adr/0005`).

| Role | Needs | Typical from |
|---|---|---|
| Full-model trainer | Holds the whole model in memory | E0–E1 on most GPUs; E2 on 24 GB+ |
| Stage trainer | Holds one pipeline stage (~2.5–3B parameters at 16 GB) as part of a crew (§5.1) | E2 onward for cards below the full-model size |
| Path trainer | Holds the shared modules plus one assigned path or set of experts (§5.2) | Sparse epochs (E4 and beyond, research) |
| Verifier | Replays sampled pieces of others' work; with splitting, one stage or path at a time | All epochs |
| Evaluator | Runs held-out and canary checks | All epochs |
| Data worker | Tokenises and checks new shards | All epochs, CPU-friendly |

### 5.1 Pipeline crews (dense epochs)

The model's layers are split into stages. A **crew** has one contributor
per stage, ideally two or three per stage so one dropout does not stall it
(SWARM-style routing to any live holder of the next stage). A batch flows
through the crew; the last stage computes the loss and sends the per-token
loss changes back to every stage; each stage updates its own layers locally.

```
data → [stage 1] → [stage 2] → … → [stage N] → loss
         ▲ activations over the internet ▲          │
         └──── per-token loss changes (KB) ─────────┘
```

To the rest of the network a crew looks like one full-model contributor,
the way a mining pool looks like one large miner. Assignment (`01` §2)
hands out work per crew; orchestrators check and credit each stage
separately.

Where the mining-pool comparison breaks:

- **Mining shares are independent; stages are chained.** A crew runs at the
  speed of its slowest stage, and one bad stage corrupts everything after
  it.
- **Traffic between stages is large.** Rough arithmetic, not a measurement:
  one 2,048-token sequence at width 25,600 (a plausible 1T dense shape) is
  about 105 MB in bf16 per hop, about 21 s at 40 Mbit/s upload. About 100×
  compression (as Pluralis reports) brings that near 0.2 s. Without
  compression, crews are not practical on home connections.
- **DUST signal dilution.** Perturbing every layer at once dilutes each
  layer's signal by about √(number of layers); perturbing one layer per pass
  costs a pass per layer. The likely compromise is one perturbed stage per
  pass, with different crews covering different stages. Unmeasured
  (`TODO.md` 0.13).
- **SOLO-style modules avoid the dilution problem.** Each module learns
  from its own local loss, with only forward activations crossing between
  machines (arXiv 2609.35440, `adr/0005`). It is proven only for 2–4
  modules, so a large crew would be a few modules, each split across 2–4
  cards. Unmeasured over home links (`TODO.md` 0.16–0.17).

### 5.2 Independent paths (sparse epochs)

The closest match to a mining pool is to make the pieces nearly
independent. The model is a set of shared modules plus many experts. Whole
documents are routed to one **path** (shared modules plus a few experts),
so a contributor trains a self-contained 1–3B path with almost no traffic to
anyone else. Paths are combined at round boundaries. This is the route to a
"1T-class" sparse model that a 16 GB card can help train. SOMA (arXiv
2609.37899, `../research/papers/notes/2609.37899.md`) shows the idea for
zero-order training, but only with ~8M-parameter LSTMs. Other prior work
(DiPaCo, Branch-Train-MiX, both 2024) is recalled, not yet verified in
`research/`, and was done with backprop, not DUST.

Per-token expert routing across the internet is rejected: every token would
visit other machines at every layer (`adr/0005`).

### 5.3 Splitting fixes memory, not compute

A mining pool lets small miners take part but adds no hash rate. Splitting
the model is the same: it lets a 16 GB card help, but the total work to
train the model does not shrink. Rough bound using the standard estimate of
6 × parameters × tokens operations for backprop training, ~20 tokens per
parameter, and an assumed 6 TFLOPS of useful work per consumer card (an
assumption, not a measurement; v1.5's measured DUST speed is well below
this):

| Target | Operations | Card-years | With 10,000 cards |
|---|---|---|---|
| 1T dense, ~20T tokens | ~1.2 × 10²⁶ | ~630,000 | ~63 years |
| 1T sparse, ~32B active, ~15T tokens | ~2.9 × 10²⁴ | ~15,000 | ~1.5 years |

So past E3, the roadmap should grow **total** parameters through sparse
paths and experts while keeping **active** parameters per token modest. A
1T dense epoch is not on the table.

### Memory and bandwidth floors (weights alone)

| Model | fp32 weights | bf16 weights | One bf16 update before compression |
|---|---|---|---|
| 1B | ~4 GB | ~2 GB | ~2 GB |
| 4B | ~16 GB | ~8 GB | ~8 GB |
| 9B | ~36 GB | ~18 GB | ~18 GB |
| 27B | ~108 GB | ~54 GB | ~54 GB |

These are arithmetic (parameters × bytes), not measurements. Today's v1.5
client also keeps a full fp32 gradient buffer (4 bytes per parameter) and
downloads full fp32 weights each round, and the 9B and 27B uploads would
exceed the coordinator's 1 GiB request limit. v2 needs bf16 weights,
delta downloads, slice-only buffers, and compressed uploads before E2.
With splitting (§5.1–5.2), these floors apply to the whole model; a
contributor holds only its stage or path.

## 6. How long might this take?

Measured basis: v1.5 ran a 1B DUST step at 30.7 tokens per second on one
RTX 4060 Ti. At that rate, E1's ~20B-token target would take about:

- 18 such GPUs (v1's active count): ~419 days;
- 180 such GPUs: ~42 days.

That is the 1B epoch alone, before verification overhead, and before
accounting for DUST possibly needing many more tokens than backprop to
reach the same loss (our 10M test needed far more time to reach a worse
loss). Each later epoch costs roughly (size ratio)² more: E2 about 16 times
E1, E3 about 5 times E2. This is why triggers are not dates, and why the
Phase 0 lane decision matters more than any other choice in this plan.

## 7. Go / no-go gate before each epoch

Recorded with numbers on the record before the epoch's genesis or
`epoch_start` entry:

1. The chosen lane reached a target held-out loss at the previous size
   within a measured time.
2. Honest replays across the GPU types in use agree within a tolerance
   that still rejects the red-team's best attack.
3. Verification overhead is within budget (proposed: no more than a quarter
   of total network compute; a guess to revisit).
4. Upload and download sizes fit the bandwidth of the median contributor,
   and no role needs more than the genesis hardware floor.
5. Enough rehearsal-proven contributors in each role; for crew epochs,
   enough to fill every stage with redundancy.
6. Legal and export review of the next model size is done if weights will
   be published.

## 8. The genesis roadmap entry

Sketch of the fields every epoch row carries:

```
epoch:
  id, name
  model: {family, layers, width, heads, vocab, experts?, active_experts?}
  hardware_floor: {gpu_memory_gb, min_upload_mbit}   # same for every epoch
  split: {mode: none | crew | paths, stages?, per_stage_redundancy?,
          activation_compression?, paths?, routing?}
  allowed_modules: [...]
  lane: dust | backprop | local_modules | es_seeded
  growth_from_previous: {recipe, recipe_version, params}
  data: {shard_set_hash, filter_version}
  held_out_commitment
  token_target, plateau_rule
  readiness: {rehearsal_job_spec, min_per_role, signal_threshold, deadline_rounds}
  rounds: {length_seconds, commit_window, dispute_window}
  verification: {replay_fraction, verifiers_per_check, tolerance}
  caps: {per_key_share, per_group_share}
  min_client_version
```
