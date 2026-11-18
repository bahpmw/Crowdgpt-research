DRAFT — requires human review.

# B3 runners-up and takeaways: modular, sparse and growth-based training

Bucket B3 covers independent or decentralized experts, mixture-of-experts (MoE) training with low communication, merging of separately trained
models, and model growth or upcycling. Round 2, written 2026-11-18.

**Downloaded and read in full** (notes in `research/papers/notes/`):

- 2602.11543 SPES: sparse expert synchronization over slow links.
- 2603.19149 Optimal Splitting: scaling law for when to fork a model into specialists.
- 2604.19835 Expert Upcycling: growing the number of experts mid-training.

**Read level for the runners-up below:** A = abstract or search snippet only. S = abstract plus a partial skim of the HTML
version. None was read in full, so treat every number as unverified until someone reads the paper. Dates are arXiv v1 dates
checked on the abs page unless marked "(listing)", which means the date came from a search result or listing only.

## Runners-up (found, not downloaded), most relevant first

1. **2606.19025** (2026-06-17), *FoMoE: Breaking the Full-Replica Barrier with a Federation of MoEs*. [A]
   Each worker holds and syncs only its assigned experts. An overlap factor sets how many replicas each expert has.
   Tokens routed to experts the worker does not hold are skipped, so no activations cross the WAN (wide-area network).
   Reported: 1.42× less traffic than full-replica DiLoCo-style training, 45.44× less than plain data parallel, and 1.4×
   throughput. This is the missing piece for SPES, where every node holds every expert. Quality appears to be shown only
   at about 54M parameters.

2. **2605.06663** (2026-05-07), *EMO: Pretraining Mixture of Experts for Emergent Modularity*. [A]
   All tokens in a document must use one router-chosen pool of experts. A 1B-active / 14B-total model trained on 1T
   tokens matches a standard MoE. Keeping 25% or 12.5% of experts costs only 1 or 3 points; models and code are
   released. This is a building block for sending whole documents to the volunteers that hold the matching pool. It
   bounds per-node memory and avoids per-token traffic.

3. **2605.13405** (2026-05-13), *When is Warmstarting Effective for Scaling Language Models?* [S]
   Dense growth tested from 14M to 1.2B parameters. 2× growth is the most reliable. Above some growth factor, training
   from scratch is just as good. Gains fade beyond about 20 tokens per parameter, and exact function preservation is not
   needed. This questions the roadmap's 4× pivot from 1B to 4B and argues for smaller, more frequent growth steps.

4. **2605.11689** (2026-05-12, listing), *Slicing and Dicing: Configuring Optimal Mixtures of Experts*. [A]
   Over 2,000 pretraining runs up to 6.6B total parameters. Quality keeps rising with total expert parameters at fixed
   active parameters, even at very high sparsity, and the best expert size depends only on active parameters. This
   supports fixing expert size once, from the per-volunteer compute budget, and adding experts as volunteers join.

5. **2605.13247** (2026-05, listing; v2 dated 2026-05-14), *EMO: Frustratingly Easy Progressive Training of Extendable
   MoE*. Not the AI2 EMO paper. [A]
   Grows the expert pool in stages, using a sparsity-aware scaling law to set each stage's token budget. Final loss
   1.017 vs 0.994 for a fixed 128-expert run, with 10% fewer GPU-hours. A second recipe for "grow experts as the network
   grows"; it saves less than Expert Upcycling.

6. **2601.06857** (2026-01-11), *MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models*. [A]
   Each "shared backbone + one expert" submodel trains on its own data cluster with zero communication. The experts
   are then concatenated, the backbones averaged, and the merged model fine-tuned briefly on strong GPUs. Claims parity
   with full training at 47.6–69.5% lower cost on Qwen1.5-MoE-2.7B and Llama-MoE-3.5B. A cheap "train apart, then merge
   into one MoE" path. It is unclear whether training starts from scratch, and the final joint step needs datacenter
   GPUs.

7. **2605.25846** (2026-05, listing), *On the Limits of Model Merging for Multilinguality in Pre-Training*. [A]
   Linearly merging separately pretrained ~2B monolingual models collapses to near-random. Their representations
   diverge unless they share pretraining first. This is a warning against "everyone pretrains their own model and we
   merge later", and it supports a shared trunk before any fork.

8. **2606.07404** (2026-06-05, listing), *Reversible Foundations: Training a 120B Sparse MoE through State-Preserving
   Scaling*. [A]
   Experience report: one model grown in four stages, from a dense seed through 5B and 9B MoEs to 120B (460 experts,
   top-12) on single 8-GPU nodes. It lists silent failure modes for each growth operator. Useful as a checklist for
   epoch-pivot operators; it is not a controlled benchmark.

9. **2602.08818** (2026-02-09), *FlexMoRE: A Flexible Mixture of Rank-heterogeneous Experts for Efficient
   Federatedly-trained LLMs*. [A]
   Replacing FlexOlmo's full-size experts with lower-rank experts of varying rank scored 47.18 vs 45.46, using 10.75B
   vs 33.27B parameters. Differently sized experts could match 8 GB vs 24 GB volunteers. The experts were extracted after
   training, not pretrained this way.

10. **2603.18297** (2026-03, listing), *Path-Constrained Mixture-of-Experts (PathMoE)*. [A]
    Sharing router weights across blocks of layers concentrates tokens onto fewer expert "paths", with better perplexity
    at 0.9B and 16B and no auxiliary loss. Fewer, steadier paths are easier to place on one volunteer.

11. **2604.00801** (2026-04-01, listing), *Routing-Free Mixture-of-Experts* (EMNLP 2026). [A]
    Each expert decides on its own whether to activate, with no central router or top-K selection. Tested only up to
    0.8B. A model without a central router might let volunteers add or drop experts without retraining one.

**Considered, lower relevance:**

- 2604.18473 BAR: modular post-training, not pretraining.
- Merging of fine-tuned checkpoints: 2511.21437 and 2607.11997.
- Not pretraining: 2610.07809 MASKerade (vision-language tuning), 2606.16456 SPRI (speech-translation upcycling), 2605.14289
  MetaMoE (federated, privacy).
- Datacenter expert-parallel systems: 2610.14798 PlaceMoE, 2610.07333 RelayMoE, 2608.28511.
- Upcycling papers whose abstracts were not read: 2610.12599 STAPLE, 2610.11317 DivMoE, 2604.13508, 2603.29765.

**Non-arXiv leads** (not eligible for the index, flagged for a human):

- Parallax tech report (Chutes, version of 2026-10-10, parallax.chutes.ai). A 7.77B-total / 1.20B-active ternary MoE trained on up to
  208 RTX 5090s, in 8-GPU hosts, over the public internet. Trunk and experts sync separately: sparse 4-bit trunk packets go
  through CPU relays, and expert owners publish their experts. Median per-host send rate was about 10–12 Mbit/s. This
  is the closest real-world match to a volunteer MoE. The report calls itself an "AI-generated living document, not
  fully reviewed by people", so treat it as unverified.
- Core Automation blog, "The Hitchhiker's Guide to Training on Compute Islands" (2026-11-02). A ~32B MoE trained
  across two 160-GPU clusters about 9,000 miles apart, linked at 25 Gbps. Pairwise "butterfly" mixing matched standard
  training. Closer to bucket B7.

## Ranked takeaways for training across ~1M consumer GPUs

1. **Scale by experts, not dense width: go sparse early and grow by duplicating experts.** Doubling experts at fixed
   top-K matched a 13B MoE trained from scratch: validation loss 1.263 vs 1.267, 11-task average 56.4 vs 56.7, with
   24–32% fewer GPU-hours. Dense→MoE upcycling at matched tokens ended worse than even a half-size MoE trained from
   scratch at every sparsity tested (2604.19835 Tables 1 and 4; Table 4 uses small models only). Total parameters can then
   track the volunteer count while per-token work stays flat. Before committing to the roadmap's "dense to 9B, upcycle
   once" plan, A/B test "sparse at E1/E2, then ×2 experts per pivot" on small models. Paper rules of thumb: grow at
   13–38% of a stage, train at least 50% as many tokens again after growing, and copy the experts with the largest
   gradients.

2. **Make "own a few experts" the unit of volunteer work, but close three gaps first.** SPES (2602.11543) owners
   train the shared trunk plus their own experts and upload only those. At 2.1B that is 2.8 vs 8.3 GB per round and
   34.1 vs 54.5 GB per GPU, at centralized quality. The gaps:
   - Every node still holds every expert, and a ~27B MoE needs about 54 GB in bf16. Fix with FoMoE-style skipping or
     by routing documents to expert pools.
   - Each expert has one owner, a single point of both failure and trust. Use 2–3 replicas per expert, robust averaging,
     and per-expert loss scoring.
   - The trunk is about 87% of each upload (our arithmetic) and was sent uncompressed over 13–17 Gbps links. It needs
     SparseLoCo-class compression (Covenant-72B) to fit 20–100 Mbit/s.

3. **Route whole documents across the internet, never single tokens.** Document-level constraints look cheap. EMO's
   per-document expert pools match a standard MoE at 14B / 1T tokens (abstract). A 32-token prefix picks the right
   cluster 88% of the time (2603.19149 Table 5). A coordinator or small router can assign each document to a pool and
   send it to volunteers holding that pool. That means no activation traffic and memory bounded by pool size, consistent
   with `adr/0005` rejecting per-token WAN routing.

4. **Cap global sync with "trunk, then fork".** The whole network trains one shared trunk, then forks into K
   sub-networks that never talk. The fork time comes from a scaling law fit on cheap small runs; fit at ≤1.3B, it
   predicted 2.7B with mean absolute error 0.012. At the optimum, 47% (1.3B) to about 63% (2.7B) of tokens are shared.
   The 16 specialists then beat a same-size single model by 1.5–2 points, and the 2.7B specialists beat a 6.7B dense
   model (2603.19149). Merging without a shared phase collapses (2605.25846, abstract). This bounds sync cost at
   million-GPU scale and confines poisoning to one cluster. The costs are K× total parameters and per-document routing.

5. **Make pivots deterministic and checkable, and remember the evidence gap.** Expert copying with seeded noise, fork
   times computed from logged token counts, and recorded utility scores can all be "pure functions of the record". A
   post-growth gate can reject malformed pivots: loss should land near the source (1.38 vs 1.32), not near random
   (about 10.5). However, every B3 result uses backprop with AdamW on datacenter links, 13–17 Gbps at best. None tests home
   uplinks, churn, mixed 8–24 GB cards, Byzantine peers, or zero-order (DUST) updates. First experiments:
   - SPES-style expert ownership with compressed trunk sync at a simulated 40 Mbit/s.
   - Expert duplication under DiLoCo-style local steps.
