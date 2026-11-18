DRAFT — requires human review

# Code and math pretraining / annealing data for CrowdGPT v2

Research date: 2026-11-18. Sources are dataset cards on Hugging Face (HF) and papers on arXiv. I read each one directly for this note. Every number below comes from one of those sources. Anything I could not confirm is tagged **[unverified]**.

## How to read this

- **Token counts use each source's own tokenizer** (Llama-2, Llama-3, SmolLM2, GPT-2 and others). The same text will give a different count with the Qwen3 tokenizer CrowdGPT uses. Treat sizes as rough.
- **"Licence"** is the licence on the dataset card. For web-derived sets (ODC-By), the licence covers the compilation, not the copyright in each page. CrowdGPT already accepts that for FineWeb, so I treat ODC-By web compilations the same way. For code, what matters is the licence of each file, so I flag it wherever files with no licence or a copyleft licence can get in.
- **"Model-licence carry-over"** means the data was written by a model whose licence adds terms to anything trained on its outputs. Examples: Llama 3.x ("Built with Llama", name prefix), Qwen2/2.5-72B (Qwen licence), DeepSeek, Phi-4. These do not block use, but each one must be recorded per shard.
- **Phase**: "main" means the stable learning-rate phase. "anneal" means the late learning-rate decay or mid-training phase. Most of the evidence comes from annealing or continual-pretraining runs on models that were already trained, not from training from scratch. That matches CrowdGPT's per-token goal, but it means the gains may not carry over in full to a 1B model trained from scratch.

## 1. Summary table

| Dataset | HF id | Domain | Tokens (source tokenizer) | Licence (card) | Evidence headline | Best phase | Verdict |
|---|---|---|---|---|---|---|---|
| SwallowCode-v2 | `tokyotech-llm/swallow-code-v2` | Python (LLM-rewritten) | 49.8B | Apache-2.0 (card also says "subject to The-Stack-v2 licensing terms") | +20.7 HumanEval / +21.9 HumanEval+ vs Stack-Edu, Llama-3.1-8B, 50B-token continual pretraining (CPT) (card) | anneal (+ some main) | **Top code pick #1** |
| SwallowCode (v1) | `tokyotech-llm/swallow-code` | Python (rewritten) | 16.1B | Llama 3.3 Community | +17.0 HumanEval vs Stack-Edu, 8B, 50B CPT (arXiv 2505.02881) | anneal | Superseded by v2; Llama licence |
| CraneCode (OLMo 3) | inside `allenai/dolma3_dolmino_pool` (`data/cranecode`) | Python (Qwen3 rewrite of SwallowCode recipe) | 18.8B | ODC-By (Dolmino card) | 5B-token microanneal +5.0 HumanEval (SwallowCode: +10.3); 12.5B microanneal +13.5 (arXiv 2512.13961 §A.3.2) | anneal | Good alternative or complement to #1 |
| Stack-Edu | `HuggingFaceTB/stack-edu` (IDs only); contents in `allenai/dolma3_mix-6T` / `dolma3_dolmino_pool` | 15 languages, edu-filtered | ~125B | No licence on card; per-file `license_type` field; refers to Stack v2 terms | Python HumanEval 20.7→25.6 vs StarCoder2Data; C++ 16.7→24.8 (SmolLM2 Table 2, 1.7B anneal) | main + anneal | **Top code pick #2** |
| UltraData-Stack-v3-L2 | `openbmb/UltraData-Stack-v3-L2` | 11 languages, algorithm-relevant selection from Stack v3 | 245.5B | ODC-By | 1B model, 10B CPT: EvalPlus 30.00 vs 25.37, MultiPL-E 22.75 vs 17.30 vs raw Stack-v3-train (card) | main | **Top code pick #3** |
| UltraData-Code L2 / L3 | `openbmb/UltraData-Code` | 11 languages; L3 = synthetic exercises | L2 ~400B, L3 ~150B | Apache-2.0 (but "users must comply with each source repo licence") | L2 vs Stack-Edu +4.37 EvalPlus / +3.05 MultiPL-E; L2+L3 +8.42 / +8.07 over L2 alone (1B, 10B CPT; arXiv 2610.23960) | L2 main, L3 anneal | Strong numbers; **licence blocker**: no per-file licence field, source not licence-filtered |
| The Stack v3 (train) | `HuggingFaceCode/stack-v3-train` | 713 languages, repo-grouped, contents inline | ~3.6T | ODC-By; files `permissive` or `no_license` only | No quality ablation found **[unverified]** | main (raw pool) | Good licence-filtered pool; too raw to use directly |
| The Stack v2 / StarCoder2Data | `bigcode/the-stack-v2*` | 600+ languages | ~900B (StarCoder2Data) | "other": bulk download needs a Software Heritage/INRIA agreement; original licences apply | Baseline others improve on | main | Use the filtered versions above instead |
| Common Pile stackv2_edu | `common-pile/stackv2_edu_filtered` | Code, openly licensed only | 67.8B (Comma tokenizer) | Per-file open licences (Blue Oak list) | Used to train Comma v0.1 7B; no head-to-head code ablation found **[unverified]** | main | Safest licence fallback |
| OpenCoder annealing corpus | `OpenCoder-LLM/opc-annealing-corpus` | algorithmic code + synthetic QA/snippets | 12.44B algorithmic + 2.71B snippets + 0.91B textbooks (paper Table 3) | ODC-By (source code from Stack v2; synthetic from Qwen2-72B-Instruct) | Ablation is shown only as a figure (1.5B); no numbers in text | anneal | Optional; mixed licence sources |
| OpenCoder FineWeb code/math recall | `OpenCoder-LLM/opc-fineweb-code-corpus`, `opc-fineweb-math-corpus` | code/math web pages | 55B combined (card) | MIT (code) / ODC-By (math) | EAI table: HumanEval+ 26.2 vs DCLM-baseline 28.0 | main | Skip: no gain shown |
| Nemotron-Pretraining-Code v1/v2/v3 | `nvidia/Nemotron-Pretraining-Code-v*` | GitHub metadata + synthetic | v2 adds ~340B; v3 adds 173B | v1/v2: NVIDIA Data Agreement (gated); v3: CC-BY-4.0 | — | — | Reject: metadata only (needs a GitHub re-crawl); v1/v2 terms forbid redistribution |
| Nemotron-CC-Code-v1 | `nvidia/Nemotron-CC-Code-v1` | code web pages | 427.9B | NVIDIA Data Agreement (gated, manual) | — | — | Reject (redistribution terms) |
| EAI-Taxonomy Code w/ DCLM | `EssentialAI/eai-taxonomy-code-w-dclm` | code web pages | 564B | ODC-By | HumanEval+ 28.7 vs DCLM 28.0 (within error); MMLU-CS 47.0 vs 32.0 (card) | main | Optional, for CS knowledge, not code generation |
| codeparrot/github-code-clean (current) | `codeparrot/github-code-clean` | 30 languages | 873 GB raw source | Card says Apache-2.0, but files include GPL-2/3, AGPL-3, LGPL, EPL, MPL | Python subset ≈ The-Stack-v2-smol in Swallow ablation | — | **Replace, or filter to permissive licences** |
| SwallowMath-v2 | `tokyotech-llm/swallow-math-v2` | math (FineMath-3+ rewritten, Q&A + textbook) | 32.0B (13.6B Q&A + 18.3B textbook) | Apache-2.0 | Card: beats other open math sets on GSM-Plus and BBH; slightly below Nemotron-CC-Math-4+ on MATH (figure only; numbers **[unverified]**) | anneal | **Top math pick #1** |
| CraneMath (OLMo 3) | `allenai/dolma3_dolmino_pool` (`data/cranemath`) | FineMath-4+ rewritten with Qwen3 | 5.62B | ODC-By | Microanneal +18.5 MATH / +27.4 GSM8K (SwallowMath v1: +16.0 / +24.5 using 3.6B tokens) (arXiv 2512.13961) | anneal | Use with #1 (same idea, small, clean licence) |
| SwallowMath (v1) | `tokyotech-llm/swallow-math` | FineMath-4+ rewritten | 2.3B | Llama 3.3 Community | +12.4 GSM8K / +7.6 MATH vs FineMath-4+, 8B, 50B CPT | anneal | Superseded; Llama licence |
| MegaMath | `LLM360/MegaMath` | math web + math code + synthetic | 371.6B total (Web 263.9B, Web-Pro 15.1B, Code 28.1B, Synth 64.5B) | ODC-By (Web-Pro and Q&A refined with Llama-3.3-70B / Qwen2.5-72B) | Llama-3.2-1B, 100B tokens: GSM8K 8.5→25.7, MATH 4.6→9.5 (CoT, Table 22) | Web main; Pro + Synth anneal | **Top math pick #2** |
| UltraData-Math | `openbmb/UltraData-Math` | math web L1/L2 + synthetic L3 | L1 170.5B, L2-preview 33.7B, L3 88B | Apache-2.0 (seed data includes gated Nemotron-CC-Math) | MiniCPM-1.2B, ~100B decay: MATH500 37.02 (+3.62 vs Nemotron-CC-Math 4+), GSM8K 61.79 (+3.34) (card) | L2 main, L3 anneal | **Top math pick #3** (check provenance first) |
| Nemotron-CC-Math-v1 | `nvidia/Nemotron-CC-Math-v1` | math web | 3+: 133.26B; 4+: 52.32B | NVIDIA Data Agreement (gated) | 8B, 100B tokens: MATH 40.6 vs FineMath-4+ 35.8; HumanEval+ 34.82 vs 32.16 (arXiv 2508.15096 Table 2) | main/anneal | Best evidence, but **reject on licence** (no redistribution) |
| Nemotron-Pretraining-Specialized-v1 | `nvidia/Nemotron-Pretraining-Specialized-v1` | synthetic STEM: Math-Textbooks 25.1B, STEM-SFT 82.5B, Sci-Coding 1.2B, RQA 134.6B | see left | CC-BY-4.0 (Sci-Coding and Wiki-Rewrite: CC-BY-SA-4.0 + GFDL) | Evidence is in the Nemotron 3 Nano report **[unverified]** | anneal | Candidate; Qwen/DeepSeek/Phi-4 carry-over; share-alike parts |
| FineMath 3+/4+, InfiWebMath 3+/4+ | `HuggingFaceTB/finemath` | math web | FM3+ 34B, FM4+ 10B (paper; card table says 9.6B), IWM3+ 20.5B, IWM4+ 8.5B | ODC-By | FineMath4+: 2x GSM8K, 6x MATH vs InfiMM-WebMath (SmolLM2 §3.3.2, 1.7B anneal) | 3+ main, 4+ anneal | Already in the mix: make sure 4+ is upsampled in the decay phase |
| MegaMath-Web-Pro-Max | `OctoThinker/MegaMath-Web-Pro-Max` | math web, Llama-refined | ~70B (paper) | ODC-By, but the gate requires agreeing to **non-commercial use only** | OLMo 3: +7.0 MATH / +13.3 GSM8K microanneal | anneal | Reject (non-commercial gate) |
| OpenWebMath | `open-web-math/open-web-math` | math web | 14.7B | ODC-By | Beaten by FineMath and Nemotron-CC-Math (tables above) | — | Superseded |
| Proof-Pile-2 / AlgebraicStack | `EleutherAI/proof-pile-2` | arXiv + OWM + math code | 55B (29B arXiv, 15B OWM, 11B AlgebraicStack) | No licence on card; arXiv and code have mixed licences | Llemma (7B/34B) CPT | anneal | Only with per-source licence filtering |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | GSM8K/MATH-style problem and solution pairs | 14M pairs | CC-BY-4.0 (written by Llama-3.1-405B) | SFT of Llama-3.1-8B-Base: MATH 67.8 vs 51.9 for Llama3.1-8B-Instruct (abstract) | anneal / SFT | Low priority: overlaps with existing math SFT sets |
| MathPile | `GAIR/MathPile` | arXiv/textbooks/web math | 9.5B | CC-BY-NC-SA-4.0 | — | — | **Reject** (non-commercial, share-alike) |
| EAI-Taxonomy Math w/ FM | `EssentialAI/eai-taxonomy-math-w-fm` | math web | 34B | ODC-By | GSM8K 22.4 vs FineMath-3+ 26.4 (card) | — | Skip: below FineMath, which CrowdGPT already has |

## 2. Per-dataset notes

### Code

**SwallowCode-v2** (`tokyotech-llm/swallow-code-v2`; card: https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2; method paper: https://arxiv.org/abs/2505.02881)
- Built from The Stack v2 Python files. Pipeline: ruff auto-format, length filter, LLM quality score (SeedCoder prompt), a one-stage rewrite with Qwen3-235B-A22B-Instruct, then a final ruff pass.
- The card's own ablations, all on Llama-3.1-8B with 50B-token CPT, found:
  - one-stage and two-stage rewriting score about the same, but two-stage costs about twice the GPU hours;
  - Instruct and Thinking rewriters score about the same;
  - rewriting "medium-quality" seed code slightly beat rewriting "high-quality" seed code.
- Headline result: +20.7 HumanEval and +21.9 HumanEval+ vs Stack-Edu (card, Llama-3.1-8B).
- v1 paper detail (Appendix tables, 50B tokens, code was 16% of the mix):
  - final SwallowCode: HumanEval 0.5396 / HumanEval+ 0.5445;
  - the-stack-v2-smol Python: 0.3689 / 0.3720;
  - "codeparrot-clean Python subset": 0.3616 / 0.3634. Which codeparrot dataset this is **[unverified]**; it may be `codeparrot/codeparrot-clean`, not `github-code-clean`.
- Independent check: OLMo 3 found SwallowCode v1 gave +10.3 HumanEval in a 5B-token microanneal (arXiv 2512.13961 §A.3.2).
- Licence: Apache-2.0 (Qwen3 output is Apache-2.0). The card also says use is "subject to The-Stack-v2's licensing terms". Whether files with no licence in the source were excluded is **[unverified]**. Python only.
- Overlap with CrowdGPT: none at the document level. It targets the biggest gap (Python, +3.84 nats).
- Caveat: the gains are measured on an 8B model that already knew code. On a 1B model trained from scratch the gain is probably smaller **[unverified]**.

**CraneCode** (OLMo 3; `allenai/dolma3_dolmino_pool`, path `data/cranecode`)
- A permissively licensed re-creation of SwallowCode by AllenAI. Pipeline: Python from the-stack-v2-smol, then syntax and linter filters, then the two-stage SwallowCode rewrite with Qwen3. 18.8B tokens; 10B tokens of it were used in the Olmo 3 midtraining mix (Table 5).
- Microanneal results: +5.0 HumanEval with 5B tokens, against +10.3 for SwallowCode; +13.5 HumanEval with 12.5B tokens. The paper does not name the model in the quoted passage; it is likely Olmo 3 7B **[unverified]**.
- Licence: the Dolmino card says ODC-By.
- Useful as a second rewritten-Python source when CrowdGPT runs out of unique SwallowCode-v2 tokens.

**Stack-Edu** (`HuggingFaceTB/stack-edu`; SmolLM2 paper https://arxiv.org/abs/2502.02737 §3.4)
- Construction: StarCoder2Data, the 15 largest languages (~450B tokens), scored 0–5 by per-language StarEncoder classifiers trained on Llama3-70B labels. Threshold 3 (2 for Java). Result: ~125B tokens; Python is 21.8B.
- Evidence (SmolLM2 Table 2, annealing a 1.7B SmolLM2 checkpoint taken at 3T tokens), MultiPL-E, StarCoder2Data → Stack-Edu:

  | Language | Before | After |
  |---|---|---|
  | Python | 20.7 | 25.6 |
  | C++ | 16.7 | 24.8 |
  | JavaScript | 18.2 | 22.4 |
  | Java | 17.6 | 22.7 |

- Labs that use it: SmolLM2's decay (24% of the mix), SmolLM3's stable-2 and decay phases (HF blog), and Olmo 3's pretraining ("Stack-Edu (Rebalanced)") and midtraining (FIM version).
- Access: the HF repo holds Software Heritage IDs only. Contents must be fetched from the `softwareheritage` S3 bucket (needs AWS credentials; may cost money for transfer **[unverified]**). Copies with contents included appear in `allenai/dolma3_mix-6T` (`data/stack_edu-*`) and `allenai/dolma3_dolmino_pool` (`stack_edu_fim-*`, FIM-formatted). Whether those copies keep the per-file `license_type` field is **[unverified]**.
- Licence: the card has no licence field and refers to The Stack v2 terms. Each file carries `license_type` (permissive / no_license). **Keep only `permissive` files** to meet CrowdGPT's open-licence rule.
- Overlap: StarCoder2Data overlaps heavily with codeparrot/github-code-clean, since both come from GitHub. Stack-Edu should replace it, not sit alongside it.

**UltraData-Stack-v3-L2** (`openbmb/UltraData-Stack-v3-L2`; tech report arXiv 2610.23960, dated 2026-10-09)
- Built from The Stack v3 train split. Steps: cleaning, then a per-language selector (file-role prediction + algorithmic-relevance score + quality constraint) using Qwen3-Embedding-0.6B features. Keeps 20.10% of the pool: 245.5B tokens across 11 languages, Python 38.94B.
- Evidence (card; 1B model, 10B-token CPT, same settings throughout): EvalPlus 30.00 vs 25.37 and MultiPL-E 22.75 vs 17.30, against unfiltered Stack-v3-train.
  - Whether it beats Stack-Edu directly is shown only in a figure **[unverified]**.
  - The sister set UltraData-Code-L2 beat Stack-Edu by +4.37 EvalPlus / +3.05 MultiPL-E (multilingual) and by +8.46 EvalPlus on Python.
- Licence: ODC-By, inherited from Stack v3. Stack v3 already excludes `non_permissive` files, but `no_license` files remain. Whether the per-file licence field is kept here is **[unverified]**.
- Caveat: very new, and all evidence comes from the authors' own lab.

**UltraData-Code L2/L3** (`openbmb/UltraData-Code`)
- Source: an independent crawl of ~192M GitHub repositories. L2 is ~400B tokens of algorithm-relevant code; L3 is ~150B tokens of synthetic "task, analysis, solution, test" exercises grounded in L2 files.
- Evidence (1B model, 10B CPT): L2 beats L1 by +7.80 EvalPlus / +5.13 MultiPL-E, and beats Stack-Edu by +4.37 / +3.05. An L2+L3 1:1 mix adds +8.42 / +8.07 over L2 alone. At 100B tokens, L2+L3 reaches EvalPlus 57.06 / MultiPL-E 39.54, which is +10.11 / +12.39 over L2 alone.
- **Licence blocker:** the card says Apache-2.0 "does not override" each source repository's licence. "Public availability of a repository is not a grant of… training rights." The L2 row schema has no licence field. CrowdGPT cannot record a per-file licence, so I do not recommend it unless openbmb publishes licence metadata.
- Which model generated L3 is **[unverified]**.

**The Stack v3** (`HuggingFaceCode/stack-v3-train`, created 2026-07-23)
- A direct GitHub crawl (cutoff 2025-08-07). Train split: 15.9 TB, ~3.6T tokens (v3.1, after fixing a duplication bug), 713 languages, grouped by repository, contents inline. ODC-By, not gated.
- Licence handling: ScanCode per-file detection. `non_permissive` files are excluded; files are labelled `permissive` or `no_license`. Opt-out and StarPII redaction are applied.
- There are no quality-filter ablations on the card. Use it as a licence-clean raw pool (keep `permissive` files) or through UltraData-Stack-v3-L2. It is far too large and raw to use as-is with CrowdGPT's token budget.

**The Stack v2 / StarCoder2Data** (`bigcode/the-stack-v2`; https://arxiv.org/abs/2402.19173)
- Terms of use:
  1. bulk download needs an agreement with Software Heritage and INRIA;
  2. training must follow the Software Heritage principles for LLM training;
  3. the original file licences apply.
- In practice CrowdGPT should consume it only through Stack-Edu, SwallowCode-v2 or CraneCode.

**Common Pile stackv2_edu** (`common-pile/stackv2_edu_filtered`; https://arxiv.org/abs/2506.05209)
- The Stack v2 filtered to repositories whose detected licences are all on the Blue Oak Council list. Per-document licence is in metadata.
- 67.8B tokens (Comma tokenizer). In Comma v0.1-1T it was repeated 2× in the main stage and 0.1× in the cooldown (comma_v0.1_training_dataset card).
- I found no head-to-head code ablation **[unverified]**. It is the licence-safest code source, and the right fallback if the project decides `no_license` files are not acceptable.

**OpenCoder annealing and RefineCode** (https://arxiv.org/abs/2411.04905)
- The annealing mix (Table 3): RefineCode 83.94B, Algorithmic Corpus 12.44B, High-Quality Code Snippets 2.71B, Code Textbooks 0.91B.
- §6.2 says removing the algorithmic and synthetic data "drops performance a lot" for a 1.5B model. The numbers are only in Figure 9 **[unverified]**.
- RefineCode itself is released as metadata only.
- Code textbooks were written with Qwen2-72B-Instruct, so Qwen licence carry-over applies. The algorithmic corpus is sampled from The Stack v2 (mixed licences).
- Overall: superseded by SwallowCode-v2 and CraneCode.

**Nemotron code sets**
- Pretraining-Code-v1/v2 are gated under the **NVIDIA Data Agreement for Model Training**. §2.2.2 forbids redistributing or making the data available to others, and §2.2.3 forbids use "in any manner that would cause them to become subject to an open-source license". That conflicts with publishing shards in an open CrowdGPT dataset.
- Pretraining-Code-v3 is CC-BY-4.0, but it is only commit IDs and paths (8.2 GB of metadata, 173B tokens if re-crawled).
- Nemotron-CC-Code-v1 (427.9B tokens) is under the same agreement as v1/v2.
- All rejected.

**codeparrot/github-code-clean (currently in the mix)**
- The dataset card says Apache-2.0. The source dataset's own card lists 15 per-file licences, including gpl-2.0, gpl-3.0, agpl-3.0, lgpl-2.1, lgpl-3.0, epl-1.0 and mpl-2.0. https://huggingface.co/datasets/codeparrot/github-code
- It is a BigQuery snapshot from March 2022, with exact deduplication and simple filters only.
- Action: drop it, or at minimum keep only files where `license` is mit/apache-2.0/bsd-2/bsd-3/isc/cc0/unlicense, and record the licence per shard. Stack-Edu (permissive files only) is the better-evidenced replacement.

### Math

**SwallowMath-v2** (`tokyotech-llm/swallow-math-v2`)
- Built from FineMath-3+: length filter, then math-text extraction with Qwen3-32B, then rewriting with Qwen3-235B-A22B-Instruct into two styles:
  - Q&A: 13.6B tokens, 12.6M samples;
  - Textbook: 18.3B tokens, 13.3M samples.
- Style ablation (card): Q&A was best on GSM8K/GSM-Plus, Textbook was best on BBH, and both helped MATH. The card says it beats other open math datasets on GSM-Plus and BBH and is slightly below Nemotron-CC-Math-v1-4+ on MATH. Exact numbers are only in a figure **[unverified]**.
- Apache-2.0.
- Related evidence: the v1 version (2.3B tokens, Llama-3.3 rewrite of FineMath-4+) gave +12.4 GSM8K / +7.6 MATH vs FineMath-4+ at 8B with 50B CPT (arXiv 2505.02881). In that run, FineMath-4+ alone reached GSM8K 0.5292 / MATH 0.2400 at 50B (Appendix Table 19). OLMo 3 independently measured +16.0 MATH / +24.5 GSM8K for SwallowMath v1 with 3.6B tokens.
- Overlap: it comes from the same source documents as the FineMath CrowdGPT already has. It is a rewrite, not a duplicate, but avoid upsampling both heavily in the same phase.

**CraneMath** (OLMo 3, `allenai/dolma3_dolmino_pool` → `data/cranemath`)
- AllenAI rewrote FineMath-4+ using the SwallowMath prompt with Qwen3, specifically to avoid the Llama licence. 5.62B tokens.
- Microanneal: +18.5 MATH / +27.4 GSM8K. This is the highest per-token math gain I found in a primary source.
- The same pool also holds **MegaMatt** (3.88B tokens), a Qwen3 re-generation of MegaMath-Web-Pro-Max-style data. Its own microanneal number is **[unverified]**.
- Licence: ODC-By.

**MegaMath** (`LLM360/MegaMath`; https://arxiv.org/abs/2504.02807)
- 371.6B tokens (Llama-2 tokenizer):
  - Web: 263.9B;
  - Web-Pro: 15.1B (FineMath classifier + Llama-3.3-70B refinement);
  - Code: 28.1B (math-related Stack v2 code);
  - Synthetic: 64.5B (Q&A 7.0B from Qwen2.5-72B + Llama-3.3-70B; translated code 7.2B; text-and-code blocks 50.3B).
- Evidence:
  - Llama-3.2-1B continual-trained for 100B tokens: GSM8K 8.5→25.7, MATH 4.6→9.5, CoT average 22.3→42.2 (Table 22).
  - TinyLlama-1B ablations: text-and-code blocks raised the PAL average from 19.5 (raw code) to 28.1 when mixed, and to 46.5 when used alone (Table 7). This suggests it helps both math and Python.
  - Nemotron-CC-Math Table 2 found MegaMath-Pro below FineMath-4+ on MATH (34.0 vs 35.8) at 8B / 100B tokens.
- Licence: ODC-By. Web-Pro and Q&A carry Llama-3.3 (and Qwen) carry-over. MegaMath-Code carries Stack v2 per-file licences.
- Best use: Web in main pretraining (optional; overlaps FineMath's Common Crawl sources); text-and-code blocks and Web-Pro in the anneal phase.

**UltraData-Math** (`openbmb/UltraData-Math`; framework paper arXiv 2602.09003)
- L0 is a custom math HTML parser. L1 is 170.5B tokens of heuristically cleaned web math. L2-preview is 33.7B tokens selected by a classifier. L3 is 88B tokens of synthetic Q&A, dialogue, multi-style and textbook data, generated with Qwen2.5-72B, Qwen3-32B and GLM-4.5.
- Evidence (card): MiniCPM-1.2B decay with ~100B tokens (30% target data / 70% general). MATH500 37.02 (+3.62 over Nemotron-CC-Math 4+), GSM8K 61.79 (+3.34). The tier-by-tier and per-baseline tables are figures only **[unverified]**.
- Licence: Apache-2.0 on the card. Two concerns:
  1. The card lists Nemotron-CC-Math, MegaMath and FineMath as "Seed Data". If L3 rewrites Nemotron-CC-Math documents, NVIDIA's agreement may still apply **[unverified — needs a human licence check]**.
  2. Qwen2.5-72B carry-over applies to the L3 parts it generated.

**Nemotron-CC-Math-v1** (https://arxiv.org/abs/2508.15096; ICLR 2026)
- Pipeline: Lynx rendering + Phi-4 cleanup of Common Crawl. Sizes: 3+ is 133.26B tokens, 4+ is 52.32B.
- Nemotron-T 8B, 100B tokens (Table 2), Nemotron-CC-Math-4+ vs FineMath-4+:

  | Benchmark | Nemotron-CC-Math-4+ | FineMath-4+ |
  |---|---|---|
  | MATH | 40.60 | 35.80 |
  | GSM8K | 76.27 | 75.97 |
  | HumanEval+ | 34.82 | 32.16 |
  | MBPP+ | 45.11 | 28.88 |

- At 300B tokens, 3+ vs FineMath-3+: MATH 44.2 vs 34.6.
- This is the strongest math evidence I found, and it also helps code. **But** the card gates it under the NVIDIA Data Agreement, which forbids redistribution and open-source re-licensing (see the Nemotron code entry above). Do not put it in a published CrowdGPT dataset without legal review.

**Nemotron-Pretraining-Specialized-v1** (`nvidia/Nemotron-Pretraining-Specialized-v1`, created 2025-12-15)
- Synthetic subsets: Math-Textbooks 25.1B, STEM-SFT 82.5B, Scientific-Coding 1.2B, RQA 134.6B, InfiniByte-Reasoning 19.4B, Wiki-Rewrite 7.9B.
- Licences: CC-BY-4.0, except Scientific-Coding and Wiki-Rewrite (CC-BY-SA-4.0 + GFDL, which are share-alike). The card warns that models trained on it may carry Qwen, DeepSeek and Phi-4 licence terms.
- Math-Textbooks is generated from Nemotron-CC-Math documents, but is itself released as CC-BY-4.0.
- No ablation numbers on the card; it cites the Nemotron 3 Nano report **[unverified]**.
- Worth a human licence review as a decay-phase STEM source.

**FineMath / InfiWebMath** (`HuggingFaceTB/finemath`; SmolLM2 §3.3.2)
- Sizes: FineMath4+ 10B tokens (6.7M docs), FineMath3+ 34B, InfiWebMath4+ 8.5B, InfiWebMath3+ 20.5B. All decontaminated against GSM8K, MATH and MMLU.
- Evidence (annealing a 1.7B SmolLM2 checkpoint): FineMath4+ gives 2x GSM8K and 6x MATH vs InfiMM-WebMath. InfiWebMath4+ plateaus after about 80B tokens (~10 epochs); FineMath4+ does not.
- Who uses it: SmolLM2's decay used FineMath4+ + InfiWebMath-3+. SmolLM3 added FineMath4+, InfiWebMath4+ and MegaMath in stable-2. Olmo 3 pretrains on FineMath-3+.
- CrowdGPT already includes `HuggingFaceTB/finemath`. Which subset is used should be confirmed **[unverified]**. Upsample 4+ in the decay phase; this costs nothing.

**Others, briefly**
- **OpenWebMath** (14.7B, ODC-By): superseded by FineMath and Nemotron-CC-Math in head-to-head tables.
- **Proof-Pile-2** (55B): no licence on the card. The arXiv subset comes from RedPajama (arXiv licences vary). AlgebraicStack is code with mixed licences. Use only after licence filtering per source.
- **OpenMathInstruct-2** (CC-BY-4.0; Llama-3.1-405B outputs): SFT-style; overlaps with the existing OpenMathReasoning, Nemotron-Math-v2, NuminaMath and OpenR1 sets.
- **MathPile**: CC-BY-NC-SA-4.0. Reject.
- **MegaMath-Web-Pro-Max**: the gate asks users to agree to "non-commercial use ONLY". Reject; the ODC-By CraneMath and MegaMatt re-creations are the clean alternative.

## 3. Ranked picks for CrowdGPT

What I optimised for: (a) the most capability per token for 1B–9B models on a budget of 10–100s of billions of tokens; (b) open licences that can be redistributed and recorded per shard; (c) the measured gaps, Python first (+3.84 nats), then math (+2.44).

### Code

1. **SwallowCode-v2** (49.8B Python tokens, Apache-2.0).
   - Why: the largest measured per-token Python gains of any open set, targeting CrowdGPT's biggest gap: +20.7 HumanEval vs Stack-Edu at 8B / 50B CPT. The rewrite recipe is confirmed independently by OLMo 3 (SwallowCode microanneal +10.3 HumanEval with 5B tokens).
   - How to use: mostly in the decay/anneal phase. Some in the main phase for the 1B model, where the whole run is short.
   - CraneCode (18.8B, ODC-By) is a second, independent rewrite to mix in.
   - Check before use: whether source files with no licence were excluded.
2. **Stack-Edu, permissive files only** (~125B tokens before filtering, 15 languages).
   - Why: the best-replicated multilingual code filter (SmolLM2, SmolLM3, Olmo 3). +4.9 HumanEval Python vs StarCoder2Data at 1.7B.
   - Licence: the per-file `license_type` field lets CrowdGPT drop `no_license` files and record a licence per file.
   - How to use: the code backbone in the main phase, replacing codeparrot/github-code-clean. Get contents from the AllenAI Dolma 3 mirrors or the Software Heritage S3 bucket.
3. **UltraData-Stack-v3-L2** (245.5B tokens, ODC-By, 2026).
   - Why: newer source (GitHub as of August 2025) and an algorithm-focused selection. +4.6 EvalPlus over raw Stack v3 at 1B / 10B CPT. The sister L2 set beat Stack-Edu by +4.37 EvalPlus.
   - Why only #3: single-lab evidence, and the licence field is **[unverified]**.
   - How to use: main phase, if CrowdGPT needs more unique multilingual code than Stack-Edu provides.
   - Licence-strict fallback: Common Pile `stackv2_edu` (67.8B tokens, only Blue Oak-listed licences).

Remove or licence-filter `codeparrot/github-code-clean`: it holds GPL, AGPL and LGPL files under an Apache-2.0 dataset label.

### Math

1. **SwallowMath-v2** (32B tokens, Apache-2.0) **plus CraneMath** (5.6B tokens, ODC-By).
   - Why: rewritten FineMath has the highest measured per-token math gains in primary sources. CraneMath microanneal: +18.5 MATH / +27.4 GSM8K. SwallowMath v1: +12.4 GSM8K / +7.6 MATH over FineMath-4+ at 8B. Both rewrites were done with Apache-licensed Qwen3, so there is no Llama carry-over.
   - How to use: decay/anneal phase.
   - Overlap: same source documents as CrowdGPT's FineMath. Treat them as an upgrade of those documents, not extra unique content.
2. **MegaMath** (ODC-By).
   - Why: the text-and-code blocks (50.3B tokens) teach math through Python. That should help both the math and the Python gaps; PAL average 28.1 vs 19.5 with raw code on TinyLlama-1B. On Llama-3.2-1B, 100B tokens took GSM8K from 8.5 to 25.7.
   - How to use: text-and-code blocks and Web-Pro in the anneal phase. Web is optional in the main phase.
   - Licence: record Llama-3.3 carry-over for Web-Pro and Q&A, and Qwen2.5 carry-over for Q&A.
3. **UltraData-Math L2 (+ L3)** (Apache-2.0, 2026).
   - Why: on a 1.2B model, a 100B-token decay reported MATH500 +3.62 and GSM8K +3.34 over Nemotron-CC-Math-4+.
   - Condition: a human must confirm that the L3 synthetic data is not derived from Nemotron-CC-Math documents in a way NVIDIA's agreement covers.
   - Fallback if that fails: simply upsample **FineMath-4+ / InfiWebMath-4+**. They are already in the pipeline, add no new licence risk, and SmolLM2 showed FineMath4+ gives 2x GSM8K and 6x MATH vs InfiMM-WebMath at 1.7B.

Not recommended despite strong numbers: Nemotron-CC-Math-v1 and all NVIDIA-agreement sets (no redistribution), MegaMath-Web-Pro-Max (non-commercial gate), MathPile (NC-SA).

## 4. Open questions and unverified items

- Whether SwallowCode-v2, CraneCode and UltraData-Stack-v3-L2 kept files whose source licence was `no_license` **[unverified]**.
- Whether the Stack-Edu copies inside `allenai/dolma3_*` keep `license_type` metadata **[unverified]**.
- Exact numbers in figures only: the SwallowMath-v2 comparison, the UltraData-Math per-baseline table, the UltraData-Stack-v3-L2 Python table, and OpenCoder Figure 9 **[unverified]**.
- The model used in the OLMo 3 microanneals (likely Olmo 3 7B) **[unverified]**.
- The Nemotron 3 Nano report's evidence for Nemotron-Pretraining-Specialized-v1 **[unverified]**.
- Whether downloading from the Software Heritage S3 bucket incurs transfer charges **[unverified]**. Ask before running it (no-spend rule).
- Which FineMath subset CrowdGPT currently uses **[unverified]**.
- How well gains measured by continual pretraining of 1B–8B models carry over to CrowdGPT's from-scratch, possibly zeroth-order or DiLoCo training. I found no primary source that tests this **[unverified]**.
