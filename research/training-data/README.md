# Training data for CrowdGPT: a dozen datasets and a training order

DRAFT — requires human review. Compiled 2026-11-18 from public papers, dataset cards and the Hugging Face API.
Numbers are the authors' own unless the detail files say otherwise. `[unverified]` and `[inference]` mark claims
we could not check or that we derived ourselves. Nothing here has been run on Crowd yet.

## Files

| file | scope |
|---|---|
| [`01-web-corpora.md`](01-web-corpora.md) | 22 web corpora: DCLM, FineWeb-Edu, FinePhrase, FinePDFs, Essential-Web, Nemotron-CC, Ultra-FineWeb, licences, overlap |
| [`02-code-math.md`](02-code-math.md) | ~30 code and math sets: Stack-Edu, SwallowCode/Math, Crane*, MegaMath, FineMath, UltraData, rejects |
| [`03-reference-synthetic-instruct.md`](03-reference-synthetic-instruct.md) | Wikipedia, papers, PDFs, synthetic rewrites, Nemotron specialised sets, Dolmino anneal, instruction data in pretraining |
| [`04-ordering-and-curriculum.md`](04-ordering-and-curriculum.md) | does order matter, WSD decay, curriculum, mixing methods, repetition, growth 1B→27B, distributed caveats |

## Why these and not others

Crowd-v1's measured loss gaps against Qwen3-0.6B (`CrowdGPT/docs/DATA_MIX_PROPOSAL.md`) set the priorities:

| domain | gap (nats) | datasets below that target it |
|---|---|---|
| Python code | +3.84 | 8, 9, 12 |
| math | +2.44 | 10, 11 |
| wikitext | +1.80 | 6 (note: inflated by eval overlap, see "Fix first") |
| SciQ | +1.50 | 4, 5, 7 |
| fineweb-edu | +1.04 | 1, 2, 3 |
| own chunks | +0.36 | — |

A dataset made the list if it (a) beat a strong baseline in a controlled ablation at 0.5B–8B scale, (b) is
licensed for open redistribution or training without a signed agreement, and (c) fits a 2048-token context.
"Punches above its weight" here means a large gain per token, not a large dataset.

## The dozen

Licences were read from the Hugging Face API on 2026-11-18. None of the twelve is gated.

| # | dataset | HF id | size | licence | headline evidence | where it goes |
|---|---|---|---|---|---|---|
| 1 | **DCLM-Baseline 1.0** | `mlfoundations/dclm-baseline-1.0` | 3.8T | CC-BY-4.0 | 7B/280B: MMLU 50.8 vs FineWeb-Edu 37.3; best real-text partner for synthetic data (14.32 vs 13.77 alone, FinePhrase study) | main backbone; replaces raw FineWeb |
| 2 | **FineWeb-Edu** (score ≥3; ≥4 for decay) | `HuggingFaceFW/fineweb-edu` | 1.3T | ODC-By | MMLU 33.6% after 38B tokens; other corpora needed ~300B to match. Weak commonsense, so pair with DCLM | main (≥3), decay (≥4) |
| 3 | **FinePhrase** | `HuggingFaceFW/finephrase` | 486B | ODC-By (SmolLM2-1.7B generator, Apache-2.0) | 1.2B/21B: table variant 17.18 vs DCLM 13.77 vs Nemotron-HQ-synth 13.54; holds 0.5B–6.2B | main + decay; cap synthetic share, always mix with real text |
| 4 | **FinePDFs-Edu** | `HuggingFaceFW/finepdfs-edu` | 350B+ | ODC-By | 1.67B/36B beats Nemotron-CC v2 (figures only); HF best mix is web + ≤25% PDFs | main + decay; needs a PII pass (only emails and IPs removed) |
| 5 | **Essential-Web STEM-w-DCLM** | `EssentialAI/eai-taxonomy-stem-w-dclm` | ~0.9T | ODC-By | 2.3B/80B: MMLU-STEM 34.5 vs DCLM 27.7 vs FineWeb-Edu 26.7 | main + decay; targets the SciQ gap |
| 6 | **FineWiki** | `HuggingFaceFW/finewiki` | ~9B (en) | CC-BY-SA-4.0 + GFDL | Wikipedia with math and tables kept; Wikipedia is in every strong anneal mix (OLMo 2 Dolmino 7.1%) | small share in main, upsampled in decay |
| 7 | **peS2o (openly licensed)** | `common-pile/peS2o_filtered` | ~43B | per-paper open licences (no HF tag) | part of the Dolmino anneal that took OLMo 2 1B MMLU 26.9→44.3 | small share main + decay |
| 8 | **Stack-Edu** (permissive files only) | `HuggingFaceTB/stack-edu` | ~125B, 15 langs | per-file `license_type`; contents mirrored in `allenai/dolma3_mix-6T` (ODC-By) | SmolLM2 1.7B: Python HumanEval 20.7→25.6, C++ 16.7→24.8 vs StarCoder2Data | main code; replaces `github-code-clean` |
| 9 | **SwallowCode-v2** (+ CraneCode) | `tokyotech-llm/swallow-code-v2`; CraneCode in `allenai/dolma3_dolmino_pool` | 49.8B (+18.8B) | Apache-2.0, also "subject to Stack v2 terms" (CraneCode ODC-By) | 8B/50B: +20.7 HumanEval vs Stack-Edu; CraneCode +13.5 HumanEval at 12.5B microanneal | decay only |
| 10 | **MegaMath** | `LLM360/MegaMath` | 371.6B | ODC-By (Web-Pro rewritten by Llama-3.3-70B → Llama terms) | Llama-3.2-1B, 100B tokens: GSM8K 8.5→25.7, MATH 4.6→9.5 | Web in main; Web-Pro/code/synth in decay |
| 11 | **SwallowMath-v2** (+ CraneMath) | `tokyotech-llm/swallow-math-v2`; CraneMath in `allenai/dolma3_dolmino_pool` | 32B (+5.6B) | Apache-2.0 (CraneMath ODC-By) | CraneMath microanneal +18.5 MATH / +27.4 GSM8K; SwallowMath v1 +12.4 GSM8K vs FineMath-4+ at 8B | late main + decay |
| 12 | **Nemotron-Pretraining-Specialized**, selected subsets | `nvidia/Nemotron-Pretraining-Specialized-v1`, `-v1.1`, `-v1.2` | 7.3B Code-Concepts, 35B Fact-Seeking, 82.5B STEM-SFT | v1.1/v1.2 CC-BY-4.0; v1 mixed (CC-BY-SA, GFDL, "other") | Code-Concepts HumanEval 73→79 (30B-A3B; built from HumanEval concepts, so check other code evals); Fact-Seeking SimpleQA-proxy 40.2→50.2 | decay only; STEM-SFT filtered to ≤2048 tokens and plain text |

Short notes:

- **Biggest likely wins per token:** SwallowCode-v2, CraneMath/SwallowMath-v2 and FinePhrase. All three are rewrites of
  already-good data by a small open model. Rewrites gain most at small scale, which suits a 1B start.
- **Most important swap:** DCLM + FineWeb-Edu instead of raw FineWeb. This is the best-evidenced change on the list.
- **Overlap:** FineWeb, FineWeb-Edu, Ultra-FineWeb, Ultra-FineWeb-L3 and FinePhrase all come from the same Common
  Crawl document pool. Budget epochs per *pool*, not per dataset, or deduplicate by URL.
- **Token counts** are in each source's tokenizer. Re-measure in Qwen3 (or the v1.5 4096-vocab tokenizer) before
  setting epochs.

### Honorable mentions and needs-a-decision

| dataset | why not in the dozen |
|---|---|
| Nemotron-CC v1 high-quality | strongest web result (8B/1T MMLU 59.0 vs DCLM 53.4), but Common Crawl terms and not on HF |
| Ultra-FineWeb-L3 (already used) | Apache-2.0 tag, but the card forbids unchanged redistribution or re-hosting without written permission. `Vxtzq/CrowdGPT` re-hosts it tokenized |
| UltraData-Stack-v3-L2 | 245B, ODC-By, EvalPlus 30.0 vs 25.4 at 1B, but self-reported only |
| UltraData-Math | +3.6 MATH500 over Nemotron-CC-Math at 1.2B; check the seed is not Nemotron-derived |
| OpenMathReasoning, Nemotron-Math-v2 (already used) | CC-BY-4.0, not gated; keep, but only short samples (≤2048) and only in the decay |
| Cosmopedia v2 | weaker than DCLM alone (10.33 vs 13.77); ≤3% at most |
| FLAN (as in Dolmino) | helps the anneal but is the largest contamination source; decontaminate first |

Rejected for licence or access: Nemotron-CC v2/v2.1 and Nemotron-CC-Math (NVIDIA agreement), ClimbMix, REWIRE,
MegaMath-Web-Pro-Max, MathPile, Institutional Books (non-commercial), SwallowCode/Math v1 (Llama licence),
Stack v2 bulk (signed agreement). Deferred for length: OpenThoughts3 (~14.5k tokens avg) and SmolTalk2 (~7.4k).

## Fix first: problems in the current mix

1. **Raw FineWeb** → replace with DCLM (1) and FineWeb-Edu (2).
2. **`codeparrot/github-code-clean`** carries an Apache-2.0 label but contains GPL, AGPL, LGPL, EPL and MPL files,
   and is lower quality than Stack-Edu. Replace with Stack-Edu filtered to permissive licences.
3. **`allenai/tulu-3-sft-mixture`** includes No Robots (CC-BY-NC) and GPT-4 WildChat responses. **`HuggingFaceTB/smoltalk`**
   includes Llama-3.1-405B outputs (Llama naming clause). Drop those subsets or the whole sets.
4. **The `Vxtzq/CrowdGPT` card says MIT.** The contents are ODC-By, CC-BY, Apache and others, plus the Ultra-FineWeb-L3
   clause. The card should list each source and its licence. Ask openbmb for permission or stop re-hosting L3.
5. **SFT and chat data should not be in the main phase.** Put it in the decay only, as plain text, and do not add chat
   special tokens the model has not seen. Olmo 3 saw GSM8K drop from 49.4 to 0 from unseen special tokens.
6. **Eval overlap.** WikiText comes from Wikipedia, so adding FineWiki will make the wikitext gap shrink for the wrong
   reason. Build `ng-eval-v2` with reserved chunk indices (as `docs/NG_EVAL_SET.md` recommends), drop eval documents by
   ID from every new source, and decontaminate FLAN-style and Q&A sets against GSM8K, MATH, HumanEval, SciQ and MMLU.

## Does order matter? Yes, at the phase level

Detail and sources: [`04-ordering-and-curriculum.md`](04-ordering-and-curriculum.md).

- **What works:** a warmup-stable-decay (WSD) schedule. Train on broad data at a constant learning rate, then put the
  best and most specialised data into the final 10–20% while the learning rate decays to zero. Evidence: SmolLM2's
  10% decay took math 7.3→22.1 and code 16.8→23.2; OLMo 2 1B's 50B-token anneal took MMLU 26.9→44.3 and GSM8K 3.3→43.8
  (part of that is the decay itself); MiniCPM 2.4B with instruction data in the decay went GSM8K 27.7→42.3.
- **What does not:** sorting documents from easy to hard inside a phase. The decay erases most of the effect
  (Luo et al., ICLR 2026, arXiv 2511.18903).
- **Mixing algorithms** rarely beat a sensible stratified mix (Aioli). Use the template below and tune it with
  microanneals.

### Proposed regime for the 1B run

Illustrative budget B = 100B tokens. Shares are percent of each phase's tokens. **[inference]** They are starting points
built from the SmolLM2, SmolLM3, OLMo 2 and Olmo 3 recipes and mapped onto the dozen. They have not been tuned for Crowd.

| phase | span of B | LR | mix |
|---|---|---|---|
| 0. warmup | ~1–2% | 0 → peak | phase-1 mix |
| 1. stable, broad | to ~70% | constant | DCLM 28 · FineWeb-Edu≥3 25 · FinePhrase 10 · FinePDFs-Edu 7 · Essential-STEM 5 · Stack-Edu 12 · MegaMath-Web + FineMath-3+ 5 · FineWiki 3 · peS2o 3 · Nemotron Fact-Seeking 2 |
| 2. stable, higher quality | ~70–85% | constant | DCLM 20 · FineWeb-Edu≥4 20 · FinePhrase 10 · FinePDFs-Edu 7 · Essential-STEM 6 · Stack-Edu 15 · SwallowCode-v2 3 · math 10 (MegaMath-Web-Pro, FineMath-4+, SwallowMath-v2 textbook) · FineWiki 4 · peS2o 3 · Fact-Seeking 2 |
| 3. decay | final ~15% (10–20%) | peak → 0 (linear or 1-sqrt) | web 45 (FineWeb-Edu≥4 18, DCLM 10, FinePhrase 8, FinePDFs-Edu 5, Essential-STEM 4) · code 18 (SwallowCode-v2 8, CraneCode 4, Stack-Edu Python 4, Code-Concepts 2) · math 18 (SwallowMath-v2 8, CraneMath 4, MegaMath Pro/synth 3, short OpenMathReasoning/Nemotron-Math-v2 3) · reference 9 (FineWiki 5, peS2o 4) · instruction/QA 10 (STEM-SFT ≤2048 5, Fact-Seeking 2, decontaminated FLAN-style 3), all plain text |

If B is below ~100B, merge phases 1 and 2. At 100B the decay uses about 1.2B SwallowCode-v2 tokens, 0.6B CraneMath and
under 4B FineWiki in total, so no set comes near its repetition cap (web ≤1–2 epochs, specialist ≤4, anneal-only 2–4).

Rules that go with it:

1. **Keep the checkpoint from the end of phase 2.** Run the decay as a branch. Grow to 4B from the pre-decay checkpoint,
   not the decayed one.
2. **Run 2–3 decays** with different data-order seeds on separate volunteer groups and average the weights. This fits
   volunteer churn and is a known cheap gain (OLMo 2 souping).
3. **Same mix on every worker.** DiLoCo tolerated different domains per worker only at ≤400M on one dataset. Log tokens
   per source and reweight to correct for volunteers dropping out. Change phase or mix only at outer sync points.
4. **Keep decay data off the zeroth-order and SOLO lanes** until tested; give them broad web only. **[unverified]** Nobody
   has measured how these lanes interact with data order.

### Growing to 4B, 9B and the 27B MoE

- Bigger models gain less from the decay (OLMo 2: +37% at 1B, +12.3% at 32B) but still gain. Use a 10–15% decay at 4B
  and 9B, 5–10% for the MoE.
- At 9B and 27B, tokens per parameter will be far below 20. Favor broad web diversity in the main phase and keep the
  specialised sets in the decay.
- After each growth step, re-warm the learning rate and replay some earlier-phase data. Reuse the decay mix, scaled to a
  similar number of optimizer steps. **[unverified]** whether a 1B decay mix transfers unchanged.
- Growth works best when the smaller model has seen ≤20 tokens per parameter, and doubling is the most reliable step.
  4B→9B (2.25×) is closer to that than 1B→4B (4×).
- Build the MoE by upcycling the 9B dense model (NVIDIA: MMLU 67.6 vs 65.3 from scratch).

## How to validate before committing volunteer compute

1. **Eval first.** Build `ng-eval-v2`, decontaminate, and fix the held-out tasks that are never used for mix decisions.
2. **Proxy ranking.** Rank web sources with a small model (Olmo 3 used 30M params × 3B tokens). Rankings at 150M predict
   1B about 80% of the time, so treat them as a filter.
3. **Microanneals from a stable checkpoint.** For each decay candidate, anneal ~1–2B target + 1–2B web and compare with
   ~2–4B web-only. **[inference]** This is scaled down from Olmo 3's 5B + 5B; validate once.
4. **Measure with `tools/domain_eval.py`.** Track the six gaps above. Each dataset's row says which gap it should close.
   Success looks like the code and math gaps shrinking first, as they did in SmolLM2's decay.

## Open questions

- Will openbmb grant permission to re-host Ultra-FineWeb-L3, or should it be dropped?
- Is Stack-Edu content fetched from the allenai mirrors enough, or is Software Heritage S3 needed? Ask before using S3:
  it may cost money.
- Which FineMath subset does `Vxtzq/CrowdGPT` already contain?
- How do data phases interact with the DiLoCo outer optimizer, DUST and SOLO? No published evidence.
- What decay length is best at 1B under volunteer churn, where decay steps may arrive unevenly?
- Does SwallowCode-v2's "subject to Stack v2 terms" note need legal review before re-hosting?
