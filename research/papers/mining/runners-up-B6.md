DRAFT — requires human review

# B6 (trust for decentralized training): runners-up and takeaways

Round-2 survey, bucket B6. Written 2026-11-18.

**Chosen and downloaded** (full notes in `research/papers/notes/`):
- 2609.17380, *OPEN-1B: A Fully Auditable Training Run*: bit-exact replay of any training step on other
  hardware, at a measured 5–12× speed cost.
- 2609.37367, *Backdoor Mitigation in Decentralized LLM Fine-Tuning (Chorus)*: receivers test each update's
  behavior and vote. It catches every backdoored update in its tests.
- 2606.19129, *Giskard*: Byzantine-robust aggregation (aggregation that tolerates malicious senders),
  costed at 10⁶ parties using committee trees.

**How to read the list below.** None of these runners-up was read in full. Numbers come from abstracts,
arXiv HTML pages or author summaries, as the last column says. Dates are arXiv v1 submission dates,
checked on the abstract page unless marked. The arXiv search API started refusing requests (HTTP 429)
partway through the search, so coverage may have gaps.

## Runners-up (found, not downloaded), ranked by relevance

| # | arXiv | v1 date | Title | Why it could matter for training across ~1M consumer GPUs | Source read |
|---|---|---|---|---|---|
| 1 | [2610.37741](https://arxiv.org/abs/2610.37741) | 2026-10-27 | Agentic Backdoors via Pretraining Poisoning (Hu, Hu, Loughridge, He, Carlini, Jagielski et al.) | Poisoning 0.1% of pretraining data gave about 92% attack success (Pass@32) at 4B. Success rises with model size, stays above zero at 0.0005%, and survives later fine-tuning and preference training. At a million equal-share peers that is about 1,000 peers, or about 5 (our inference). Outside the strict bucket (centralized poisoning). | abstract, author summary |
| 2 | [2606.04819](https://arxiv.org/abs/2606.04819) | 2026-06-03 | The Usefulness Gap in Proof-of-Useful-Work: An Empirical Study of Pearl's cuPOW Protocol (Basu) | A live "useful work" network of about 320,000 GPU-equivalents (8,012 workers measured) did zero useful AI computation. Its verifier accepted random matrices. This is real evidence at our target scale that unverified work gets gamed. | abstract, HTML skim |
| 3 | [2607.06643](https://arxiv.org/abs/2607.06643) | 2026-07-07 | The Power of Backdoor Absorption in Community Training (Seddik, Souihi, Tamaazousti, Tucci Piergiovanni) | Models how spot-check rate, detection rate and reputation penalties drive backdoor success. Auditing 10% of steps cut attack success from 99.52% to 7.20% with half the trainers malicious, at 2.76% overhead (Table 2). A sizing method for our sampled replay, but tested on ResNet-18 only. | HTML, §1–5 |
| 4 | [2606.05433](https://arxiv.org/abs/2606.05433) | 2026-06-03 | Zero knowledge verification for frontier AI training is possible (Peigné-Lefebvre, Nguyen, Wang) | Reports same-hardware bit-exact determinism at 1.6–8.2% end-to-end overhead (Llama 7B, 8×H100). That is far below OPEN-1B's 5–12× for matching across vendors, so auditing within one GPU class could be much cheaper. This is a policy feasibility proposal. | HTML snippets |
| 5 | [2604.02372](https://arxiv.org/abs/2604.02372) | 2026-03-31 | Backdoor Attacks on Decentralised Post-Training (Ersoy, Blagoev, te Lintelo, Koffas et al.) | An attacker holding one middle pipeline stage implants a misalignment backdoor: alignment falls from 80% to 6% with the trigger, and it survives later safety training in 60% of cases. Training 27B on 8–24 GB cards forces pipeline stages held by strangers, so this applies directly. ICLR 2026 workshop paper. | abstract |
| 6 | [2609.37011](https://arxiv.org/abs/2609.37011) | 2026-09-29 | OPFL: Optimistic Verification of Federated Learning via Empirical Boundary (Su, Yao, Wang, Viswanath) | Replays training and accepts it within a gradient-difference boundary calibrated on honest replays across mixed hardware. Reports 0% attack success and 0% false rejection, including on Qwen3-0.6B and an adaptive attack. This is the tolerance-band alternative to bit-exact replay. It is a sibling of round-1 OVIG (2606.21045), so it may be redundant. | HTML snippets |
| 7 | [2610.08258](https://arxiv.org/abs/2610.08258) | 2026-10-06 | zkLLMPoT: Efficient Zero Knowledge Proof of Training for Large Language Models (Liang et al.) | A zero-knowledge proof of a committed checkpoint's loss on auditor-chosen text. Proving takes 41–59 s at 1.1–1.5B (covered operators) and verification under 0.5 s. It proves evaluation, not training. It could let orchestrators publish a verifiable held-out loss without revealing the held-out set. | abstract |
| 8 | [2602.15671](https://arxiv.org/abs/2602.15671) | 2026-02-17 | Revisiting Backdoor Threat in Federated Instruction Tuning from a Signal Aggregation Perspective (Zhao, Hu, Liu) | Poison spread thinly across many clients (under 10% of data) reaches over 85% attack success, and per-client defenses fail. This is the Sybil pattern: many identities, each carrying a little poison, each looking honest. | abstract |
| 9 | [2610.07510](https://arxiv.org/abs/2610.07510) | 2026-10-05 | Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training (Zhan et al.) | Attackers can make backdoors survive later honest training: 20% → 74% after supervised fine-tuning, 76% after reinforcement learning, on Qwen2.5-Coder-7B. This undercuts defenses that count on honest updates washing a backdoor out. | abstract |
| 10 | [2608.18736](https://arxiv.org/abs/2608.18736) | 2026-08-19 | FedLNS: Leverage LayerNorm Signature Modeling to Mitigate Adversarial Manipulation in Federated LLMs (Li, Park, Joe-Wong, Ni, Dressler) | A cheap server-side screen that looks only at normalization-layer changes and needs no extra upload. With 200 clients training small LMs from scratch, it beats six baselines when 40% of clients are malicious. An adaptive attacker could likely fake the signature (untested). | abstract, HTML skim |
| 11 | [2605.15520](https://arxiv.org/abs/2605.15520) | 2026-05-15 | On the Fragility of Data Attribution When Learning Is Distributed (Gao, Hui, Sun, Ku) | One participant inflated its contribution score without hurting accuracy: normalized Shapley value 0.0547 → 0.1682, rank 10 → 2 (CIFAR-10, ResNet-18, Table 1). Scores based on marginal benefit, the family LossScore belongs to, are themselves an attack surface. Image-only. | abstract, HTML skim |
| 12 | [2604.04611](https://arxiv.org/abs/2604.04611) | 2026-04-06 | Dynamic Free-Rider Detection in Cross-Silo Federated Learning via Simulated Attack Patterns (Nakamura) | Free-riders who start honest and later just resubmit the global model's last change (the "delta weight attack") evade existing detectors. This cheap copy attack is the one a permissionless network will see first. Image-only. | abstract, HTML skim |
| 13 | [2601.00566](https://arxiv.org/abs/2601.00566) | 2026-01-02 | Low Rank Comes with Low Security: Gradient Assembly Poisoning Attacks against Distributed LoRA-based LLM Systems (Dong et al.) | The two low-rank adapter factors each look harmless while their product is malicious. Any factored or low-rank compression of updates must be checked as the update actually applied, not factor by factor. | abstract |
| 14 | [2607.08651](https://arxiv.org/abs/2607.08651) | 2026-07-09 | Secure Decentralized Federated Learning via Gossip and Virtual Voting (Taherpour, Wang) | Derives agreement on which updates were accepted from the gossip history itself (Hashgraph-style voting), with no separate chain, and flags lazy and invalid senders. A possible orchestrator record, but tested only up to 100 nodes. | abstract |
| 15 | [2611.11477](https://arxiv.org/abs/2611.11477) | ~2026-11-08 (search metadata, unverified) | Robust and Private Swarm Learning via Zero-Knowledge Validator Admission and Consensus Aggregation | Admits validators only after a zero-knowledge proof that their validation data meets a quality bar, addressing "who judges the judges". | abstract snippet only |

**Lower relevance, listed so they are not re-found:**
- 2610.12711 (2026-10-08), HMP-GAE: Byzantine defense for federated LoRA text classification; small scale.
- 2603.12512 (2026-03-12), Byz-NSGDM: robust optimizer, tested on a character-level GPT only.
- 2603.15144 (2026-03-16), Byz-DM21: compressed plus robust aggregation; numerical experiments only.
- 2512.13666 (2025-12-15), SEDULity: proof-of-learning blockchain; simulation only.
- 2610.17546 (2026-10-13), FLAI: requires trusted hardware enclaves.
- 2605.19969 (2026-05-19), image-only precursor of Chorus.

## Takeaways (ranked)

1. **Verify which tokens each peer actually trained on. Everything else is secondary.** The only check found
   that gives proof rather than a score is exact replay of a step sampled after the peer commits to its
   result (OPEN-1B). Its cost grows with the sampling rate, not the number of peers, so it is the one trust
   mechanism that scales to a million GPUs. Matching bits across vendors costs 5–12× (OPEN-1B, Table 10).
   Same-hardware determinism is reported at 1.6–8.2% (2606.05433, unverified). So audit within GPU classes,
   use cross-vendor replay only for disputes, and measure the cost on consumer cards first.
2. **Assume a handful of fake identities can plant a backdoor, and that scoring will not see it.** Poisoning
   0.1% of pretraining data gave about 92% attack success at 4B, the effect grows with scale, and it survives
   later training (2610.37741). In a million-peer run that is about 1,000 equal-share peers, or about 5 for
   the still-effective 0.0005% (our inference). Loss-based scores, norm clipping and benchmark gates let such
   updates through, and thinly spread poison defeats per-peer detectors (2602.15671). The defense is
   beacon-assigned data, sampled exact replay, and caps on any identity's token share.
3. **Scale checking with committees drawn from a public random beacon, which decide by honest majority.
   Drop privacy machinery.** Committee trees give low per-party cost up to 10⁶ parties (Giskard). But its
   secret-shared median costs at least 146 GB per party for a single number, which rules it out for LLM
   updates, and Crowd wants updates public anyway. Our arithmetic: at 10⁶ parties with 10% fake identities,
   an honest-majority committee needs about 67 members, while one that must stay below 1/4 corrupted needs
   about 353 (thousands at 20%). The scaling lever is how much an identity costs to create.
4. **Screen behavior before aggregation on a sample, and do not count on backdoors fading.** Chorus caught
   100% of backdoored updates, rejecting at most 4.7% of honest ones. Several receivers probe each update
   against their own model and vote, at 3–4× compute per screened update. Defenses that rely on backdoors
   fading under honest training (2607.06643) are contradicted by pretraining backdoors (2610.37741) and by
   deliberately persistent ones (2610.07510). Use behavioral screening for post-training now, and test its
   memorization test on pretraining.
5. **Reward only verified, assigned work.** A deployed network of about 320,000 GPU-equivalents did no
   useful work because nothing enforced usefulness (2606.04819). Contribution scores can be inflated without
   hurting accuracy (2605.15520), and free-riders resubmit the global model's last change (2604.04611).
   Reputation, voting weight and recognition should come only from replay-verified work on beacon-assigned
   data, with commit-before-reveal and checks for copies and replayed changes.
