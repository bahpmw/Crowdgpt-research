DRAFT — requires human review

# ADR 0005 — Split the model across contributors instead of raising the hardware floor

Status: Proposed

## Context

The first roadmap draft assumed every contributor holds the whole model.
That puts a hard ceiling on who can help: bf16 weights alone are ~18 GB at
9B and ~54 GB at 27B (`05` §5), so a 16 GB card drops out around E2–E3,
and a 1T model is out of reach for every consumer card.

The owner asked (2026-11-17) whether contributors could each work on a
small chunk, the way a Bitcoin mining pool bundles many small miners into
what looks like one large miner.

Evidence from `research/03-prior-art.md`:

- Pluralis Node0 trained a 7.5B model with pipeline parallelism on consumer
  GPUs from 16 GB, with >300 participants, compressing the activations
  passed between stages about 100× (subspace networks, arXiv 2506.01260).
  Pluralis has since reported an 8B run and a 13B systems test.
- SWARM parallelism (arXiv 2301.11913) keeps pipelines running on
  unreliable volunteer nodes by holding several nodes per stage and routing
  each batch to any live one.
- Macrocosmos IOTA ran a 100B model in 16 stages, but on data-centre GPUs
  with a central orchestrator, and only 1.1B tokens.

Evidence from the owner's paper survey (`../research/papers/`, notes in
`notes/<id>.md`):

- **SOLO** (arXiv 2609.35440, Yin et al., 28 Sep 2026) cuts the model into K
  modules with no gradient flowing between them. Each module learns from its
  own next-token loss through a small head that reads out through a frozen,
  periodically refreshed copy of the final output layer. Measured at
  340M–2B parameters on 15B tokens: K=2 lands within 0.5 points on average
  zero-shot score and 3.1–3.6% on WikiText perplexity of ordinary training.
  At K=4 the gap shrinks from 11.2% to 6.1% between 340M and 2B. Only
  forward activations cross between modules. At 1 Gb/s it loses 1.2% of its
  throughput, against 51% for a standard pipeline. Extra compute is 11–13%
  at K=2. Limits: only K=2 and K=4 tested (K=8 cost +89% compute), on one
  machine, with no slow or hostile modules.
- **SOMA** (arXiv 2609.37899, Chaubard et al., 29 Sep 2026) trains several
  small zero-order "experts", each on its own data cluster with no
  communication, and averages them at inference. Separate losses cut
  zero-order noise about 3–4.6× at 4 experts. Tested only on ~8M-parameter
  LSTMs, so it supports the independent-paths idea in principle, not at our
  scale.

Not yet in our research files (recalled, not verified): DiPaCo (Google
DeepMind, 2024) and Branch-Train-MiX (Meta, 2024) train separate "paths"
or expert models with little communication and combine them afterwards.

## Decision

1. **Fix a contributor hardware floor in the genesis roadmap** (proposed:
   16 GB of GPU memory and a stated minimum upload speed). No epoch may
   require more. Growing the model means more pieces, not bigger downloads.
2. **Two ways to split the model, chosen per epoch:**
   - **Pipeline crews** for dense epochs: the layers are split into
     stages; a crew is one contributor (or a small redundant group) per
     stage; a batch flows through the crew. The crew acts as one
     full-model contributor to the rest of the network.
   - **Independent paths** for sparse "very large" epochs: the model is a
     set of shared modules plus many experts; whole documents are routed to
     one path, so a contributor trains a self-contained 1–3B path with
     almost no traffic to others. Paths are combined at round boundaries.
3. **Every stage and path is checked on its own.** A hop's output is a
   pure function of its input activations, its weights, and its seeded
   noise, so a verifier holding only that piece can replay it (`02` §3).
4. **The 1T-class goal stays a research milestone** until Phase 0 shows the
   chosen lane learns when split, and the compute table in `05` §5 says the
   network can finish it in a sensible time.

## Why DUST fits pipelines better than backprop

Backprop sends a correction signal back through every stage that is as large
as the forward activations, and each stage must hold its activations until
that signal arrives. DUST only runs forward. The last stage sends back the
per-token loss changes (kilobytes per sequence), and each stage computes its
own layer updates locally from its stored inputs and its own noise. That
roughly halves traffic between stages and removes the wait.

The open problem: DUST's signal for one layer is diluted by about
√(number of perturbed layers) when every layer is perturbed at once
(`crowd_v15/dust.py`, `mode="all"`). The clean alternative (`mode="layer"`)
costs one extra pass per perturbed layer, which across hundreds of stages is
too slow. The likely middle ground is to perturb one stage per pass, with
different crews perturbing different stages. SOMA's result (separate losses
cut zero-order noise roughly by the number of groups) points the same way.
This has not been measured on our model.

## SOLO-style crews: the measured alternative

SOLO gives crews the same traffic pattern as DUST (forward activations
only, no waiting for a return signal) but with local backprop inside each
module, which is why it learns far better per token. Each module also has
its own loss, so a verifier can replay one module from its signed input
activations, its weights, its data, and the frozen output-layer copy,
without the rest of the crew. That makes per-stage checking simpler than for
DUST, where every stage depends on the last stage's loss changes.

The catch is the module count. SOLO has only been shown with 2–4 modules of
at least 6 layers each. A 16 GB floor at large sizes needs many more
stages, so a crew would likely use a few SOLO modules, each run as a short
ordinary pipeline across 2–4 cards. Whether that works over home links is
the point of `TODO.md` 0.16–0.17.

The owner's lab design MOSAIC (`../research/lab/REPORT.md`) builds on this:
SOLO modules, SparseLoCo-style compressed uploads, capped aggregation per
module group, and spot-check replay. Its results are not yet written up.

## Consequences

- The "layer-slice trainer" role (which still ran the full model forward)
  is replaced by stage and path roles (`05` §5).
- New Phase 0 measurements gate this design (`TODO.md` 0.11–0.14 and
  0.16–0.17).
- If SOLO-style modules win Phase 0, crews use the `local_modules` lane
  (`adr/0004`) and DUST's dilution problem stops mattering for splitting.
- A crew is only as fast as its slowest stage and only as honest as its
  least honest stage; redundancy per stage and per-hop checks are required,
  not optional.
- Crews need direct connections between contributors, which moves part of
  the networking work (Phase 4 peer-to-peer layer) earlier for crew epochs.
- Memory stops being the main limit. Total compute and bandwidth become the
  limits, and the roadmap has to say so honestly.

## Alternatives considered

- **Keep full-model contributors and raise the floor each epoch.** Simple,
  but shuts out most volunteers by E3, which is what the crowd exists for.
- **Per-token mixture-of-experts routing across the internet.** Every token
  visits experts on other machines at every layer; too much traffic and
  latency for home connections.
- **Fully sharded data parallelism (FSDP-style weight streaming).** Each
  card streams all weights through itself every step; traffic per step is
  the whole model.
