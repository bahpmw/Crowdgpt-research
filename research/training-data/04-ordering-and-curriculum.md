DRAFT — requires human review

# 04 — Data order, staging, and curriculum for compute-limited pretraining

Research note for CrowdGPT v2 (1B → 4B → 9B → 27B MoE). Compiled 2026-11-18 from primary sources (arXiv papers and official technical reports and blogs). Every number below comes from the cited source. Where I draw a conclusion the source does not state directly, I mark it **[inference]**. Where I could not confirm something against a primary source, I mark it **[unverified]**.

Bottom line: **order matters, but mostly in one way.** The strongest, most-repeated finding is "train on broad, good data at a high learning rate, then put your best and most specialised data into the final learning-rate decay." Fine-grained "easy-to-hard" curricula give smaller gains that are easy to wipe out. Most of those gains show up only with a constant learning rate.

---

## 1. Key findings

### 1.1 Warmup-Stable-Decay (WSD) and annealing on high-quality data

| Finding | Numbers | Source |
|---|---|---|
| WSD matches cosine. Most of the loss drop happens in the decay. | A decay over 10% of tokens was enough for the best result. A 2.5% decay fell short (0.036B model). | MiniCPM, arXiv 2404.06395, §4.3 |
| Putting SFT-style and high-quality data **in the decay** beats adding it only at SFT. | 2.4B model, same SFT afterwards: MMLU 44.6 → 50.9, GSM8K 27.7 → 42.3, HumanEval 27.7 → 30.4 (Table 1, A-1 vs A-2). 1.2B model: doubling SFT tokens from 6B to 12B did nothing (MMLU 47.9 vs 47.9). The decay-data version reached MMLU 49.6 and MATH 10.5 vs 7.9, but GSM8K went **down** (34.2 → 31.8). | MiniCPM, Table 1 |
| Cooldown length: gains level off around 20% of steps. A "1-sqrt" decay shape beats linear decay. | With 1-sqrt, a 5% cooldown nearly matched cosine on a 20B-token run (124M–210M models). | Hägele et al., arXiv 2405.18392, §3 |
| Decaying linearly to zero (D2Z) beats the usual decay to 10% of peak. The benefit grows with tokens per parameter. | A 610M model at 80 tokens/param with D2Z had lower loss than at 200 tokens/param with 10x decay. | Bergsma et al., arXiv 2502.15938 |
| Upsampling domain data at the end ("domain upsampling") gives large benchmark gains. The best share of training is **10–20%**. | 7B model, 1T tokens: up to +6.90 MMLU, +8.26 GSM8K, +6.17 HumanEval. Tested durations ran from 5% to 30%. | Blakeney et al. "Does your data spark joy?", arXiv 2406.03476 (abstract) |
| Llama 3: annealing on small amounts of high-quality code and math helps small models, not the largest. | 8B: GSM8K +24.0%, MATH +6.4% (validation). 405B: "negligible". The final 40M tokens were annealed to LR 0 with upsampled high-quality sources, then checkpoints were averaged. | Llama 3, arXiv 2407.21783, §3.1.3, §3.4.3 |
| Annealing is a cheap way to **evaluate** a dataset. | Llama 3: anneal a 50%-trained 8B over 40B tokens, with 30% new dataset and 70% default mix. Olmo 3 "microanneal": 5B target + 5B web, compared against a 10B web-only anneal. | Llama 3 §3.1.3; Olmo 3, arXiv 2512.13961, §3.5.1 |
| OLMo 2 mid-training (5–10% of FLOPs, LR decayed linearly to 0) gives **the biggest relative gain to the smallest model**. | 1B: avg 31.9 → 43.7, MMLU 26.9 → 44.3, GSM8K 3.3 → 43.8 (Table 9). Relative gain: +37.0% at 1B, +18.7% at 7B, +15.9% at 13B, +12.3% at 32B. Without mid-training, the 1B "struggles to break out of random performance" on multiple-choice tasks. | OLMo 2, arXiv 2501.00656, Table 9, App. B.1 |
| A higher peak LR in the stable phase is cancelled out by the anneal. | Final OLMES scores differ by <2 points across 3e-4 to 12e-4 (Table 8). | OLMo 2 §4.1 |
| Averaging several anneals run with **different data orders** ("souping") helps. | Mix B: best single 75.3 vs 3x soup 77.3 OLMES-MCF; GSM* 73 → 77. 7B used 3 x 50B anneals. | OLMo 2 §4.5, Table 14 |
| DeepSeek LLM uses a stepped LR schedule as an alternative to WSD. | Drops to 31.6% of peak at 80% of tokens and to 10% at 90%. Final quality "essentially consistent" with cosine, and the stable checkpoint can be reused. | DeepSeek LLM, arXiv 2401.02954, §2.3 |
| WSD lets you **branch** a decay off the main run and keep training the main run. WSD-S resumes from a decayed checkpoint. | Tested from 0.1B to 1.2B. | Wen et al., arXiv 2410.05192 |
| Checkpoint merging can stand in for a decay phase. | WSM vs WSD: +3.5% MATH, +2.9% HumanEval, +5.5% MMLU-Pro. | arXiv 2507.17634 |

### 1.2 Multi-stage mixes in released small models

- **SmolLM2 1.7B (11T tokens, 4 stages, WSD: 2,000 warmup steps, peak 5e-4, linear decay to 0 over the last 10%)**, arXiv 2502.02737, §4:
  - Stage 1 (0–6T): web as 60% FineWeb-Edu / 40% DCLM, code 10% (StarCoderData), **no math**.
  - Stage 2 (6–8T): web 75%, code 20%, math 5% (OpenWebMath).
  - Stage 3 (8–10T): math about 10%, web ratio flipped to 40/60 FineWeb-Edu/DCLM, Stack-Edu replaces StarCoderData.
  - Stage 4 (decay, 10–11T): web 58%, Stack-Edu 24%, math 14% (FineMath4+, InfiWebMath3+), Cosmopedia v2 4%.
  - Table 3 (stage-end averages): Math 3.21 → 3.70 → 7.27 → **22.07**; Code 8.87 → 10.56 → 16.75 → **23.21**; Knowledge/Reasoning 55.50 → 56.76 → 57.47 → 60.24. Most math and code gains came in the 10% decay.
- **SmolLM2 135M and 360M (2T and 4T tokens)** used a **single stage** of "consistently high-quality data", with Stack-Edu, FineMath, InfiMM-WebMath and Cosmopedia from the start, and WSD with **20% decay** (SmolLM2 §6). This is the closest public precedent for a small, tightly budgeted run.
- **SmolLM3 3B (11.2T tokens, 3 stages, decay over the last 10%)**, HF blog 2025-07-08:
  - Stage 1 (0–8T): web 85% / code 12% / math 3%.
  - Stage 2 (8–10T): 75 / 15 / 10, adding Stack-Edu, FineMath4+ and MegaMath.
  - Stage 3 (decay, 10–11.1T): 63 / 24 / 13, adding instruction and reasoning sets such as OpenMathReasoning.
  - Mixes were chosen with 3B ablations on 50–100B tokens.
- **Nemotron Nano 2 (12B, 20T tokens)**: 3 phases. Phase 1 is a diversity mix. Phases 2 and 3 are mainly high-quality data. Switches happen at **60% and 90%** of training (arXiv 2508.14444, §2.3).
- **NVIDIA two-phase study** (8B, 1T tokens; also 15T and 25B), arXiv 2412.15285: two phases beat random ordering by **3.4%** and the natural distribution by **17%** average accuracy. Best phase-2 share was **40%** (Table 14 average: 0% → 37.68, 10% → 50.48, 20% → 51.58, 30% → 52.28, **40% → 52.86**, 50% → 51.96). Blends designed on downsampled data at 1T still worked at 15T "if the quality and epochs of the datasets are maintained." *Caveat:* this study's "phase 2" is longer than a WSD decay. The 40% result does not contradict the 10–20% decay finding. It measures how long to run on the high-quality blend, not how long to decay.
- **Phi-3**: two sequential phases. Phase 1 is mostly web. Phase 2 is a more heavily filtered subset of that web plus synthetic data (arXiv 2404.14219, §2).
- **Llama 3** changed the mix during training: more non-English data, more math, more recent web, and less of the subsets later found to be low quality (§3.4.1).

### 1.3 Curriculum learning (difficulty or quality order): evidence and negative results

- **Positive, but tied to the LR schedule.** Zhang et al. 2025 (arXiv 2506.11300) trained 200+ models (0.5B and 1B, up to 100B tokens). Curricula sped up early and mid training. Used as a *warmup*, they gave lasting gains of up to 3.5%. Compression ratio, lexical diversity (MTLD) and readability were the best difficulty signals. **All runs used a constant LR of 2e-4** (§3.4).
- **Key negative/qualifying result.** Luo et al., "How Learning Rate Decay Wastes Your Best Data…" (arXiv 2511.18903, ICLR 2026), used 1.5B models on 30B tokens. An ascending-quality curriculum clearly beats random shuffling under a constant LR. That advantage **mostly disappears under WSD or cosine decay**, because the best data arrives just as the LR gets small. Two fixes: (a) a moderate decay, where the final LR stays only moderately below peak, and (b) replacing the decay with weighted checkpoint averaging. Combined, these gave +1.64% average over random order plus standard decay.
- **[inference]** Together these two papers explain why "curriculum" mostly means *a few large mix stages plus annealing* in practice, and not a finely sorted document order. Annealing works by putting the best data where the low LR is. A sorted curriculum fights the LR schedule unless you redesign the schedule.
- **Order noise is real.** OLMo 2 gets a measurable gain just from averaging anneals that differ only in data order (§1.1). So a single run's result partly depends on its random order.

### 1.4 When to add code and math, and how big the anneal should be

- Code from the start at about 10–12%: SmolLM2 stage 1 (10%), SmolLM3 stage 1 (12%). Muennighoff et al. found **up to 50% code caused no drop on natural-language tasks** in their data-limited setting (arXiv 2305.16264, §7).
- Math: small amounts early (0–3%), raised mid-way (5–10%), and highest in the decay (13–21%). Figures: SmolLM2 0 → 5 → ~10 → 14%; SmolLM3 3 → 10 → 13%; OLMo 2 Dolmino 50B mix 20.8% math.
- Small math sets work even at low shares in the anneal. In OLMo 2 microanneals, a 10/90 math/web mix reached GSM* 61 vs 63.5 for 35/65, from a 28.5 baseline (§4.4.2).
- Anneal share: **10–20% of tokens** (Blakeney; Hägele plateau at ~20%; MiniCPM 10% enough; SmolLM2-1.7B 10%; SmolLM2-small 20%). Large-scale runs use less: OLMo 2 5–10% of FLOPs; Llama 3 only the last 40M tokens.
- Benchmark-format data in the anneal: SmolLM2 added AugGSM8K at 0.02% and OLMo 2 added the GSM8K train split. Llama 3 **left benchmark train sets out on purpose** so it could measure true few-shot ability. **[inference]** If Crowd results will be compared publicly, follow Llama 3, or clearly report that benchmark train splits were included.

### 1.5 Mixing-ratio methods and transfer from small proxy models

| Method | Result | Source |
|---|---|---|
| DoReMi | 280M proxy sets weights for 8B. +6.5 points few-shot, 2.6x fewer steps on The Pile. | arXiv 2305.10429 |
| RegMix | 512 models of 1M params x 1B tokens predict the best mix for a 1B / 25B-token model (best of 64 candidates). Web corpora correlate most positively with downstream scores, more than "high-quality" Wikipedia. | arXiv 2407.01492 |
| Data Mixing Laws | 1B / 100B tokens on RedPajama: tuned mix ≈ default mix trained 48% longer. | arXiv 2403.16952 |
| AutoScale | **Best mixes change with scale.** Wikipedia and papers help most at small scale and then flatten. CommonCrawl keeps helping. | arXiv 2407.20177 |
| Aioli (negative) | **No existing method consistently beat stratified (equal) sampling** on average test perplexity. Some were up to 6.9 points worse. | arXiv 2411.05735 |
| Olmo 3 | Swarm of 30M-param proxies trained on 3B tokens. Repetition of any domain capped at **~4–7x**. "Quality-aware upsampling" beats flat filtering. | arXiv 2512.13961, §3.4.4, App. A.2 |
| Nemotron-CLIMB | Cluster-and-search mix: 1B on 400B tokens beat Llama-3.2-1B by 2.0%. | arXiv 2504.13161 |

**[inference]** For Crowd: a hand-set, stratified-style mix based on the released recipes is a strong baseline. Treat proxy-model searches as optional refinements. Re-check them at each model size (AutoScale).

### 1.6 Repetition limits and upsampling small high-quality sets

- Muennighoff et al. (arXiv 2305.16264): up to **4 epochs** of repeated data is about as good as unique data. Returns fall off steeply past about **16 epochs** (fitted R_D* ≈ 15). Downstream scores were flat up to about 4 epochs, then dropped. *Caveat:* they repeated the whole dataset. They did not upsample one source inside a mix.
- SmolLM2 kept most sources near **4–5 epochs**. Infi-WebMath4+ plateaued after about 10 epochs (80B tokens), while FineMath4+ did not (§3.3.2, §4).
- OLMo 2 microanneal: math data used once scored GSM* 61, two copies 66, four copies 65 (§4.4.2). Its 300B mix repeated math, Stack Exchange and Wikipedia 4x (Table 13).
- NVIDIA two-phase: math and task data gave diminishing returns past about 8 epochs **[unverified exact threshold; snippet-level reading of §5.2]**.
- **[inference]** For Crowd: keep large web sources at ≤1–2 epochs and specialist sets at ≤4 epochs overall. Allow anneal-only sets about 2–4 passes. **Count repetitions across model sizes if a grown model inherits weights that already saw the data** (see §3).

### 1.7 Instruction and reasoning data in pretraining

- Helps base-model scores when it goes **in the anneal or mid-training**:
  - Olmo 3 tested instruction plus thinking-trace data inside a 100B mid-training mix. It improved **every** base metric (avg 48.8 → 50.7; math 43.1 → 48.7; Table 10).
  - OLMo 2's 50B Dolmino mix had FLAN at 16.6% and StackExchange Q&A at 2.45%.
  - MiniCPM's decay results are in §1.1. SmolLM3's decay includes OpenMathReasoning.
  - NVIDIA "Reuse, Don't Retrain" (15B) found that adding QA/alignment-style data in a second continued-pretraining distribution gave **+9%** average accuracy (arXiv 2407.07263).
- **Formatting trap.** Olmo 3 found that chat **special tokens** not seen in pretraining wrecked evaluations: GSM8K 49.43 → 0. A chat template written with ordinary text was fine (46.02) (§3.5.4). Keep special tokens for SFT.
- From scratch, Instruction Pre-Training (arXiv 2406.14491) synthesised instruction-response pairs from raw text. A 500M model on 100B tokens matched a 1B model on 300B (Pythia). Simply *mixing raw SFT data* into pretraining ("Mix PT") was mixed: for 500M, BoolQ went 57.5 → 46.8 and ARC-e 50.3 → 52.8 (Table 1).
- OLMo 2 suggested that below some size, models "may require the inclusion of task-specific data, such as that seen in SFT" to beat random on hard multiple-choice tasks (App. B.1). The authors state this as a hypothesis.
- **[inference]** Early in training, raw SFT data has weak and mixed evidence. In the decay, the evidence for it is consistent.

### 1.8 Context length schedule

- Llama 3 trained long sequences only at the end, because attention cost grows with the square of length. It extended in 6 steps over about 800B tokens (§3.4.2).
- SmolLM2 extended 2k → 8k from a checkpoint taken 75B tokens before the end of the decay: 40% long documents plus 60% decay mix, RoPE base 130k (§4.6).
- SmolLM3 used +100B tokens (4k → 32k → 64k, 50B each). Upsampling special long-document sources did not help beyond what was naturally long in the mix.
- SkyLadder (arXiv 2503.15450): under a fixed token budget, **shorter context windows trained better**. A short-to-long ramp (from 8 tokens, linear, over about 60% of tokens for a 32K target) gave up to +3.7% on standard benchmarks and up to 22% faster training (1B and 3B, 100B tokens).
- Dataset Decomposition (arXiv 2405.13226): a variable-length curriculum trained an 8k 1B model at the cost of a 2k baseline and reached target accuracy up to 6x faster.
- ProLong (arXiv 2410.02660): long data must be **mixed with high-quality short data**.
- Olmo 3 lists extension budgets from <20B tokens (Nemotron Nano 2, AFM) up to 800B (Llama 3.1) (§3.6).

### 1.9 Model growth and warm-starting

- G_stack depth stacking (arXiv 2405.15319): small models trained only **10B tokens** were stacked 4x (g = 4). The 7B stacked model reached the 300B-token scratch loss in 194B tokens (54.6% speed-up).
- "When is Warmstarting Effective…" (arXiv 2605.13405, May 2026): **2x growth is the most reliable**. There is an **upper growth factor** beyond which training from scratch wins. Gains are largest under **≤20 tokens/param** and shrink as the budget grows. Experiments used width growth on GPT-2-style models from 14M to 1.2B.
- Function-preserving width growth (HyperCloning, arXiv 2409.12903) saves GPU hours. A 2026 small-scale study found that preserving the small model's function is not a reliable way to choose a warm start (arXiv 2604.04281).
- Continuing training on new data:
  - Re-warming the LR, re-decaying it, and replaying old data (5% for a weak shift, 25% for a strong one) matched full retraining at 405M and 10B (arXiv 2403.08763; see also 2308.04014).
  - The same paper reports that DeepSeek-V2 continued training from a **non-decayed** checkpoint with 30% replay.
  - NVIDIA's continued-pretraining recipe starts from the old minimum LR, switches distributions at η_max/5, and decays to η_min/100 (arXiv 2407.07263).
- No source I found tests **whether to reuse the anneal mix after growth**. The closest evidence is that blends and anneal recipes carried across sizes when re-sized:
  - OLMo 2 used the same Dolmino recipe at 7B, 13B and 32B. It scaled the token count (50B → 100B → 300B) to keep **the same number of anneal steps** (§4.5). Its recipe was developed on a 1B model and "generalized well" (App. B.1).
  - NVIDIA's blends transferred to 15T and 25B (arXiv 2412.15285, Table 5).
- Mixture of experts (MoE): upcycling Nemotron-4 15B on 1T tokens reached MMLU 67.6 vs 65.3 for continued dense training (arXiv 2410.07524). Sparse upcycling beat dense models using about 50% of the original sunk compute (arXiv 2212.05055).

### 1.10 Order and mix with local-SGD, federated or noisy optimizers

- DiLoCo (arXiv 2311.08105): **non-IID shards** were built by k-means clustering on a pretrained model's features (8 workers).
  - Non-IID converges more slowly but ends at similar perplexity (§3.1, Fig. 5).
  - Table 3 (IID / non-IID): 8 replicas 15.08 / 15.02; 16 replicas 15.02 / 14.91; 64 replicas 14.95 / 14.96.
  - It still worked when started from random weights.
  - Limits: 60M–400M models, C4 only. The authors say the distributed setup "reduces the FLOP and data efficiency."
- Photon / federated pretraining (arXiv 2405.10853, 2411.02908): reports that statistical heterogeneity is "much less impactful for overparameterized LLMs" than for small FL models. Heterogeneity was tested with Pile sources split per client. **[unverified: I did not extract per-condition numbers.]**
- DiLoCo quality drops relative to data-parallel training as the number of workers and inner steps grows (stated in IsoLoCo, arXiv 2607.03011). DiLoCo scaling laws: when well tuned, it scales predictably with model size (arXiv 2503.09799).
- Deliberate domain specialisation is a design choice, not something to drift into:
  - BTX trains domain experts separately and then merges them into an MoE (arXiv 2403.07816).
  - DiPaCo routes data along paths (arXiv 2403.10616).
  - For **zeroth-order** training, SOMA (arXiv 2609.37899) trained separate experts per data cluster. This cut ZO gradient variance, at the cost of shared representations across domains.
  - ZO-Curv (arXiv 2611.02688) states that ZO gradient estimates "break down in pretraining". It uses them only for curvature.

---

## 2. Recommended staged regime for a compute-limited 1B run

Illustrative budget: **B = 100B tokens**. Scale the percentages for 50–300B. Starting mixes are **copied or averaged from the cited recipes and not tuned for Crowd**. Confirm them with microanneals (§2.2).

| Phase | Share of B | LR | Data (starting point) | Context | Why (source) |
|---|---|---|---|---|---|
| 0. Warmup | ~1–2% (or ~2,000 steps) | Linear 0 → peak | Phase-1 mix | 2048 | SmolLM2, SmolLM3 and OLMo 2 all use 2,000 warmup steps |
| 1. Stable, broad | 0 → ~70–75% | Constant peak | Web ~75–80% (FineWeb-Edu : DCLM ≈ 60 : 40; Nemotron-CC / Ultra-FineWeb high tiers can stand in); code ~10–12% (Stack-Edu); math ~3–5% (FineMath3+, MegaMath web); reference ~3–5% (Wikipedia, peS2o, arXiv); Cosmopedia ≤ ~3%. No chat or SFT data. | 2048 | SmolLM2 Table 1 and stage 1; SmolLM3 stage 1; SmolLM2-small used high-quality sources from the start |
| 2. Stable, higher quality (optional; merge into phase 1 if B < ~100B) | ~70–75% → ~85% | Constant peak | Web ~70% (higher quality threshold); code ~15–18%; math ~8–10% (FineMath4+, MegaMath); reference ~3–5% | 2048 | SmolLM3 stage 2; SmolLM2 stages 2–3; Nemotron Nano 2 switch at 60% |
| 3. Decay / anneal | final **~15%** (allowed range 10–20%) | Peak → **0**, linear or 1-sqrt | Top-quality web ~50–60%; code ~15–24% (Stack-Edu, upsampled); math ~12–20% (FineMath4+, MegaMath, Nemotron-CC-Math, rewritten Q&A); reference ~8–12% (Wikipedia, peS2o); Cosmopedia ~4%; instruction / QA / reasoning ~5–15% (SmolTalk, Tulu 3, OpenMathReasoning) written in **plain text, without new special tokens**, decontaminated | 2048 | Blakeney 10–20%; Hägele ~20% plateau and 1-sqrt; D2Z; OLMo 2 Dolmino 50B (web 47.2, FLAN 16.6, math 20.8, wiki 7.1, peS2o 5.85, SE 2.45); SmolLM2 stage 4; SmolLM3 stage 3; Olmo 3 Table 10 and special-token finding |
| 4. Context extension (optional) | last ~5% of B, overlapping the decay; or defer to a later size | Continue the decay | ~40% long documents + ~60% decay mix | 2048 → 4096 (8192 only if memory allows) | SmolLM2 §4.6; ProLong (keep short high-quality data) |

Operational rules attached to the template:

1. **Keep the pre-decay checkpoint** at the end of phase 2. Run the decay as a **branch** (WSD). That checkpoint is the starting point for growth to 4B (§3) (MiniCPM §4.3; Wen et al. 2410.05192; DeepSeek-V2 practice reported in 2403.08763).
2. **Run 2–3 anneals with different data-order seeds** on separate volunteer groups and average the weights. This is cheap parallel work that matches how volunteers join and leave (OLMo 2 §4.5; Llama 3 checkpoint averaging).
3. Repetition caps: web ≤ ~1–2 epochs; specialist sets ≤ ~4 epochs over the whole run; anneal-only sets ~2–4 passes (Muennighoff; SmolLM2; OLMo 2 microanneal; Olmo 3 4–7x cap).
4. Do not sort documents easy-to-hard inside a phase under a decaying LR. The evidence says the decay cancels most of the benefit (Luo et al. 2511.18903). If someone wants to try it, pair it with a moderate decay plus checkpoint averaging and test against a random-order control.
5. **[unverified for our stack]** A short-to-long context ramp in phase 0–1 (SkyLadder) might be a cheap gain. It was tested for 8K–32K targets, not 2048. Low priority.

### 2.2 How to tune the mixes cheaply

- **Phase 1 mix:** start from the template, which is close to stratified sampling across good sources. Aioli shows such baselines are hard to beat. If you search, use a small proxy swarm (Olmo 3: 30M params x 3B tokens; RegMix: 1M x 1B). Expect the optimum to move with scale (AutoScale).
- **Anneal mix:** microanneal from the stable checkpoint. Olmo 3 used 5B target + 5B web vs 10B web-only. Llama 3 used 30% new + 70% default over 40B. **[inference]** At 1B with a 100B budget, a microanneal of about 1–2B + 1–2B is probably enough to rank sources. Validate this scaled-down setting once.
- Hold out evaluation tasks that are never used for mix decisions (OLMo 2 §2.5).

---

## 3. What changes for 4B, 9B, 27B MoE and growth pivots

**Bigger models get less from annealing, but still need it.** OLMo 2's relative mid-training gain fell from +37% at 1B to +12.3% at 32B. Llama 3 405B gained "negligible" amounts from annealing. **[inference]** Keep the decay at about 10–15% for 4B and 9B. A 5–10% decay is defensible for the MoE.

**Tokens per parameter collapse.** With tens to a few hundred billion tokens per size, 9B and 27B will be far below 20 tokens/param.
- Repetition is not the main risk at those sizes; Muennighoff's limits will rarely bind.
- Data *diversity* becomes more valuable than narrow "high-quality" sources (AutoScale; RegMix: web correlates best).
- **[inference]** Lower the reference/Wikipedia weight and keep broad web. Keep specialist data concentrated in the decay.

**Model growth looks most useful exactly in this low-budget regime.** Warm-start gains are largest at ≤20 tokens/param (arXiv 2605.13405). But the factor matters:

| Pivot | Factor | Evidence fit |
|---|---|---|
| 1B → 4B | ~4x | G_stack's depth g = 4 worked. The 2026 width study found 2x most reliable, with an upper bound. Unclear. |
| 4B → 9B | ~2.25x | Closest to the "reliable" range |
| 9B → 27B MoE | Upcycling, not plain growth | Upcycled MoE beat continued dense training (NVIDIA 15B) |

**[unverified for our architectures]** Plan a from-scratch control at small scale before committing to a 4x pivot.

**Pivot recipe** (assembled from continual-pretraining evidence; no growth paper tests it directly):
1. Grow from the **stable, pre-decay** checkpoint of the smaller model, not the annealed release (DeepSeek-V2 practice; WSD branching). WSD-S shows resuming from a decayed checkpoint can work at fixed size. **[unverified for growth]**
2. **Re-warm** the LR after growth (Gupta et al. 2308.04014; Ibrahim et al. 2403.08763).
3. Start the larger model on the **phase-1/2 mix**, not the anneal mix. Add a little "replay" of whatever the small model last saw if the mix changes (5–25% per 2403.08763).
4. **Reuse the anneal recipe, re-sized**: keep similar source proportions, scale anneal tokens so the anneal has a similar number of optimizer steps (OLMo 2 50B → 100B → 300B), and re-run microanneals at the new size (AutoScale caveat).
5. **Count repetitions across sizes. [inference]** If the 1B's anneal branch is *not* the parent of the 4B, the anneal data has not been "spent". If you grow from an annealed checkpoint, the 4B inherits weights that already saw that data, so its repetitions add up.

**MoE specifics.** I found no primary source testing data order specifically for MoE pretraining. **[unverified]** Large mix switches could change router load or expert specialisation. Watch load-balance metrics at each phase switch. BTX-style deliberate domain experts are a separate, tested design.

---

## 4. Risks for distributed and noisy optimizers, and what to do

| Risk | Evidence | Mitigation |
|---|---|---|
| **Accidental non-IID shards.** Volunteers hold whatever shards they downloaded, so each worker sees a different domain mix. | DiLoCo copes with non-IID k-means shards but converges more slowly (§1.10). This was only tested at ≤400M on C4. DiLoCo quality falls as workers and inner steps grow (2607.03011). | Default: **every worker samples IID from the same current-phase mix** using a deterministic global sampler. Domain-specialised shards only as a deliberate experiment (BTX/DiPaCo-style). |
| **Mix drift from volunteer churn.** If code-heavy workers drop out, the realised mix changes. | Not directly studied **[inference]**. | Log tokens actually trained per source per outer round. Re-weight the sampler to hit the target mix. Report the realised mix, not the planned one. |
| **Phase switch in the middle of an inner loop.** Workers' pseudo-gradients mix two distributions. | Not directly studied **[inference]**. | Switch mixes and LR phases only at outer-sync boundaries, on all workers at once. |
| **Curriculum vs LR schedule, now with two LRs** (inner AdamW and outer Nesterov). | Luo et al. shows quality ordering interacts strongly with LR decay. No study covers DiLoCo inner/outer schedules **[unverified]**. | Use stage mixes plus anneal only. Decay the inner LR in the anneal. Run a small test of whether the outer LR or momentum should also change in the decay. |
| **High-variance updates in the decay** (zeroth-order "DUST" lane, and DiLoCo at many workers). | D2Z explains late decay as averaging updates to fight gradient noise. ZO gradients "break down in pretraining" (2611.02688). Cluster-sharded ZO reduces variance but loses shared representations (2609.37899). | **[inference]** Do not spend the anneal data (the scarcest, most valuable tokens) on the ZO lane. If ZO runs at all, use it in the stable phase on broad web. Prefer weight averaging and souping at the end for noisy lanes (OLMo 2; WSM; Luo et al.). |
| **SOLO-style local-module training and data order.** | No primary source found **[unverified]**. | Treat it like the ZO lane until tested: stable-phase broad data only. |
| **Contamination and special tokens** in volunteer-supplied instruction data. | Olmo 3: special tokens made GSM8K drop 49.43 → 0. | Normalise all instruction data to plain text centrally. Decontaminate against eval sets before distribution. |

---

## 5. Unverified items and open questions

- Whether the 10–20% anneal share holds at 50–300B tokens for a 1B model. Most evidence comes from runs of 1T tokens or more, except MiniCPM and Hägele, which are small. **[unverified]**
- The best microanneal size for a 1B / 100B run. **[unverified]**
- Whether a short-to-long context ramp helps at a 2048 target. **[unverified]**
- NVIDIA two-phase "~8 epochs" threshold for math and task data. **[unverified exact value]**
- Photon heterogeneity effect sizes. **[unverified numbers]**
- Growth factor limits for depth growth (1B → 4B) on our architecture. Sources disagree (G_stack g = 4 vs 2x most reliable). **[unverified]**
- Any interaction between data order and DiLoCo outer momentum, ZO estimators, or SOLO local modules. **No primary evidence found.**
- Whether to reuse the exact anneal mix after growth. Only indirect evidence (OLMo 2 and NVIDIA blend transfer). **[inference]**

## Sources (primary)

SmolLM2 arXiv 2502.02737 · SmolLM3 blog huggingface.co/blog/smollm3 (2025-07-08) · OLMo 2 arXiv 2501.00656 (v3) · Olmo 3 arXiv 2512.13961 · Llama 3 arXiv 2407.21783 · MiniCPM arXiv 2404.06395 · Blakeney et al. arXiv 2406.03476 · Hägele et al. arXiv 2405.18392 · Straight to Zero arXiv 2502.15938 · WSD river valley arXiv 2410.05192 · WSM arXiv 2507.17634 · DeepSeek LLM arXiv 2401.02954 · NVIDIA two-phase arXiv 2412.15285 · Nemotron Nano 2 arXiv 2508.14444 · Phi-3 arXiv 2404.14219 · Luo et al. arXiv 2511.18903 · Zhang et al. arXiv 2506.11300 · Muennighoff et al. arXiv 2305.16264 · DoReMi 2305.10429 · RegMix 2407.01492 · Data Mixing Laws 2403.16952 · AutoScale 2407.20177 · Aioli 2411.05735 · Nemotron-CLIMB 2504.13161 · Instruction Pre-Training 2406.14491 · Reuse Don't Retrain 2407.07263 · Ibrahim et al. 2403.08763 · Gupta et al. 2308.04014 · G_stack 2405.15319 · Warmstarting 2605.13405 · HyperCloning 2409.12903 · Width-growth selection 2604.04281 · NVIDIA upcycling 2410.07524 · Sparse upcycling 2212.05055 · SkyLadder 2503.15450 · Dataset Decomposition 2405.13226 · ProLong 2410.02660 · DiLoCo 2311.08105 · DiLoCo scaling laws 2503.09799 · IsoLoCo 2607.03011 · Photon 2411.02908 and 2405.10853 · BTX 2403.07816 · DiPaCo 2403.10616 · SOMA 2609.37899 · ZO-Curv 2611.02688 · Mid-training survey 2510.06826.
