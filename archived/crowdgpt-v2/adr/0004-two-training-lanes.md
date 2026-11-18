DRAFT — requires human review

# ADR 0004 — Several training lanes behind one protocol

Status: Proposed

## Context

DUST is cheap for contributors and replayable, but its largest published
result is 243M parameters on 20M tokens, and our own 10M time-to-target run
trailed backprop badly. Backprop-based decentralised training has reached
72B parameters (Covenant-72B) but is harder to verify.

A third option appeared in the owner's paper survey
(`../research/papers/INDEX.md`): SOLO (arXiv 2609.35440) cuts the model into
2–4 modules that each learn locally with a small prediction head, so only
forward activations cross between modules. It was measured from 340M to 2B
parameters on 15B tokens: with 2 modules it lands within about 3.5% of
ordinary training on WikiText perplexity, and the gap shrinks as the model
grows. The owner's lab (`../research/lab/REPORT.md`) proposes MOSAIC, which
pairs SOLO-style modules with SparseLoCo-style compressed uploads, capped
aggregation, and spot-check replay. MOSAIC's results sections are still
empty, so it is a proposed design, not a measured one.

## Decision

The protocol, record, trust layers, metrics, and roadmap are lane-neutral.
Each epoch config names its lane:

| Lane | What a contributor does | Strength | Weakness |
|---|---|---|---|
| `dust` | Forward-only activation perturbation | Cheapest, easiest to replay | Unproven above 243M |
| `backprop` | DiLoCo-style local steps, compressed updates | Proven to 72B | Hardest to verify; whole model per card |
| `local_modules` | SOLO-style: one module plus a local head, local backprop, compressed updates (MOSAIC) | Measured to 2B; splits the model; little traffic between modules | Tested only with 2–4 modules, on one machine, with no hostile nodes |
| `es_seeded` | Weight-space evolution strategies rebuilt from seeds and scalars | Strongest verification | Weakest learning signal |

Phase 0 measures all four at the same token budget, and an ADR records the
E1 choice.

## Consequences

- The project does not bet the whole system on an unproven method.
- Verification code has lane-specific replay paths (`02` §3).
- A lane can change at an epoch boundary without rebuilding the network.
- `local_modules` is also a way to split the model across cards
  (`adr/0005`), so the lane choice and the splitting choice are linked.
