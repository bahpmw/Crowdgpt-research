DRAFT — requires human review

# Training data research 03: reference, synthetic, and instruction/reasoning data

Research date: 2026-11-18. Scope: (A) reference / academic / books text, (B) synthetic pretraining data, (C) instruction and reasoning data used in mid-training or late annealing of base models.

## How to read this

- Every number below comes from a primary source I opened: the arXiv paper, the Hugging Face (HF) dataset card or HF API, or the publisher's own blog. The source is linked next to the number.
- **[unverified]** means I could not confirm the claim from a primary source, or I am inferring something. Check these before acting on them.
- Token counts use the **source's tokenizer**, not Qwen3's. Expect different counts after re-tokenization.
- "Generator terms" means licence conditions attached to the model that wrote synthetic text. These can apply even when the dataset itself has a permissive licence.
- **Redistribution matters more for CrowdGPT than for most projects.** CrowdGPT ships shards to volunteers through `Vxtzq/CrowdGPT`. Any dataset whose terms ban redistribution cannot go into that mix, even if it allows training. This rules out the NVIDIA Data Agreement datasets and Institutional Books.
- CrowdGPT's measured loss gaps against Qwen3-0.6B, in nats: code +3.84, math +2.44, encyclopedic (wikitext) +1.80, science (SciQ) +1.50. I mark which gap each dataset targets.

---

## 1. Summary table

Verdicts: **ADOPT** = recommended. **CONSIDER** = useful but has a caveat. **DEFER** = wrong fit now; revisit for later model sizes. **AUDIT** = already in CrowdGPT and its licence needs a check. **REJECT** = licence or fit blocks it.

| Dataset | HF id | Cat. | Tokens (source tokenizer) | Licence (+ generator terms) | Evidence headline | Best phase | Verdict |
|---|---|---|---|---|---|---|---|
| FineWiki | `HuggingFaceFW/finewiki` | A | en: 6.61M pages, 35.1 GB parquet; ~9B tokens per SYNTH paper (Pleias tokenizer) | CC-BY-SA-4.0 + GFDL | No isolated small-model ablation found. Keeps tables and math, which `wikimedia/wikipedia` strips | Main, upweighted in anneal | **ADOPT** (instead of wikimedia/wikipedia) |
| Wikipedia (2023 dump) | `wikimedia/wikipedia` | A | — | CC-BY-SA-3.0 + GFDL | Superseded by FineWiki | — | REJECT (superseded) |
| FinePDFs | `HuggingFaceFW/finepdfs` | A | ~3T total; eng 1,190.65B | ODC-By 1.0 + Common Crawl ToU | 1.67B model, 36B tokens: close to the SmolLM3 web mix; mixing helps. HF recommends keeping PDFs under 25% | Main | CONSIDER (prefer the Edu subset) |
| FinePDFs-Edu | `HuggingFaceFW/finepdfs-edu` | A | 350B+ across 69 languages (English share [unverified]) | ODC-By (education scores labelled by Qwen3-235B; filter only) | Blog: beats Nemotron-CC v2 on its own; best mix is FW-Edu + DCLM + 25% FinePDFs-Edu (charts; exact deltas [unverified]) | Main + anneal | **ADOPT** |
| peS2o v2 | `allenai/peS2o`; open subset `common-pile/peS2o_filtered` | A | 38.97M docs, 42.01B whitespace tokens; Common Pile filtered 43.3B (Comma tokenizer) | ODC-By on the collection; papers carry mixed open-access licences. Common Pile keeps only openly licensed papers | Part of OLMo 2 Dolmino anneal; no isolated delta | Main (small) + anneal | CONSIDER (use the Common Pile version) |
| arXiv (open licences) | `common-pile/arxiv_papers(_filtered)` | A | 321,336 papers, 21 GB; 6.0B tokens (Comma) | CC BY / BY-SA / CC0, per paper | No isolated delta | Main (small) | CONSIDER |
| StackExchange | `common-pile/stackexchange(_filtered)` | A | 33.4M docs, 103.7 GB raw; 23.9B tokens filtered (Comma) | CC-BY-SA, per document | In Dolmino (1.26B) and the Comma mix (6× repeats); no isolated delta | Main + anneal | CONSIDER (helps code and science Q&A) |
| Project Gutenberg / pre-1929 books | `common-pile/project_gutenberg` | A | 71,810 books, 26.2 GB; 5.7B tokens (Comma) | Public domain per metadata | No per-token evidence on target benchmarks | Main (low weight) | Low priority |
| Common Pile v0.1 (Comma mix) | `common-pile/comma_v0.1_training_dataset` | A | 463.6B tokens, main stage | Openly licensed, per source | Comma 7B (1T tokens) is comparable to Llama-1 7B. At 1.7B params / 28B tokens it beats other open-licence corpora | Source for A shards | ADOPT as a **source** for A shards |
| Institutional Books 1.0 | `institutional/institutional-books-1.0` | A | 983K vols, 242B o200k tokens | IDI Terms of Use: gated, non-commercial, no redistribution | None | — | **REJECT** (terms) |
| FinePhrase | `HuggingFaceFW/finephrase` | B | 486.4B completion tokens (faq 148.1B, math 98.4B, table 92.4B, tutorial 147.4B) | ODC-By; generator SmolLM2-1.7B-Instruct (Apache-2.0); source FineWeb-Edu | 1.2B model, 21B tokens: FinePhrase-Table 17.18 vs DCLM 13.77 (+3.41) and vs Nemotron-HQ-Synth (+3.63) | Main (mixed with real web) | **ADOPT** |
| Nemotron-Pretraining-Specialized v1 / v1.1 / v1.2 | `nvidia/Nemotron-Pretraining-Specialized-v1`, `-v1.1`, `-v1.2` | B/C | v1: RQA 134.6B, STEM-SFT 82.5B, Math-Textbooks 25.1B, InfiniByte 19.4B, Wiki-Rewrite 7.9B, Sci-Coding 1.2B. v1.1: Code-Concepts 7.29B. v1.2: Fact-Seeking 35.0B, MCQ 6.1B | CC-BY-4.0 (Wiki-Rewrite and Sci-Coding: CC-BY-SA + GFDL). Generator terms vary by subset (see §2) | Code-Concepts (10B tokens in last 100B of Nemotron Nano v3): HumanEval 73→79. Fact-Seeking: proxy SimpleQA 40.2→50.2 | Anneal / late main | **ADOPT selected subsets** |
| Nemotron-CC (v1) synthetic | Hosted on Common Crawl (not HF) | B | 1.9T synthetic; Diverse QA 499.5B | Common Crawl ToU [redistribution terms unverified]. Generator: Mistral NeMo 12B Instruct | 8B model, 1T tokens: HQ subset +5.6 MMLU over DCLM. Diverse QA: +1.5 GenQA in OLMo 3 microanneal; 14.58 vs DCLM 13.77 in FinePhrase study | Main | CONSIDER (Diverse QA only; check terms) |
| Nemotron-CC-v2 / v2.1 | `nvidia/Nemotron-CC-v2`, `-v2.1` | B | 6,585.8B (v2) | NVIDIA Data Agreement: internal training only, **no distribution**; gated | — | — | **REJECT** (no redistribution) |
| PleIAs SYNTH | `PleIAs/SYNTH` | B | 79.6M samples, ~75B tokens (Pleias tokenizer) | Card: CC-BY-4.0. **But** generators are LoRAs on Gemma-3-12B-base, trained on Gemini 2.5 Pro outputs | Own (600M, Table 8): multiple-choice avg 42.2 vs FineWiki 26.1 / FinePDFs-Edu 25.1. Independent (1.2B/21B, FinePhrase study): 10.03 vs DCLM 13.77 | Anneal (if cleared) | CAUTION (licence chain) |
| Cosmopedia v1 | `HuggingFaceTB/cosmopedia` | B | 25B tokens, 30M files | Apache-2.0; generator Mixtral-8x7B-Instruct | FinePhrase study (alone): 10.33 vs DCLM 13.77. BeyondWeb: about 5.1pp behind it at 8B | Main (low weight) | Low priority |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus` (cosmopedia-v2) | B | 28B tokens (SmolLM blog) | ODC-By; generator Mixtral-8x7B-Instruct | Used at 4% in SmolLM2 stage 4; no isolated delta | Main (low weight) | Low priority |
| BeyondWeb | not released | B | — | Proprietary (DatologyAI) | 3B model on 180B tokens beats an 8B Cosmopedia model; +2.6pp over Nemotron-Synth at 8B | — | Method only |
| WRAP | method; no dataset | B | — | — | C4 rephrasing: ~3× faster pretraining; >2% zero-shot on 13 tasks (1.3B model) | — | Method only |
| TinyStories | `roneneldan/TinyStories` | B | small | CDLA-Sharing-1.0; GPT-3.5 / GPT-4 outputs (OpenAI terms) | Models under 10M params write fluent stories; little transfer to target benchmarks | — | REJECT |
| phi-1 "textbooks" | not released | B | 1B synthetic + 6B filtered web | — (GPT-3.5 outputs) | phi-1 (1.3B): HumanEval 50.6%, MBPP 55.5% | — | Evidence only |
| Dolmino Mix 1124 | `allenai/dolmino-mix-1124` | C | 843B pool; 50 / 100 / 300B mixes | ODC-By collection + per-source licences + Common Crawl ToU | OLMo 2 1B after 50B Dolmino tokens: avg 31.9→43.7, MMLU 26.9→44.3, GSM8K 3.3→43.8 (part of this is the learning-rate anneal alone) | Anneal | **ADOPT recipe** (non-web parts) |
| Dolma 3 Dolmino Mix | `allenai/dolma3_dolmino_mix-100B-1125` | C | 100B | ODC-By, **but** includes GPT-4o-mini (Reddit-to-Flashcards) and Gemini reasoning traces | OLMo 3: adding instruction + thinking data lifts base avg 48.8→50.7 and GenQA 43.1→48.7 (Table 10) | Anneal | CONSIDER (drop proprietary-generated parts) |
| FLAN (Dolmino version) | in `allenai/dolmino-mix-1124`; `Muennighoff/flan` | C | 17.0B (Dolmino) | ODC-By (Dolmino); per-task source licences | Helps QA in OLMo 3 microanneals. **Also the largest contamination source found** | Anneal, decontaminated only | CAUTION |
| Nemotron-Post-Training-Dataset-v1 | `nvidia/Nemotron-Post-Training-Dataset-v1` | C | ~25.7M samples | CC-BY-4.0, not gated. Generators: DeepSeek-R1-0528, Qwen3-235B-A22B | No small-model pretraining ablation found | Anneal / SFT | CONSIDER |
| Nemotron-Pretraining-SFT-v1 | `nvidia/Nemotron-Pretraining-SFT-v1` | C | 336B (per Front-Loading paper) | Gated (manual), licence "other" [likely NVIDIA Data Agreement — unverified] | 8B model, 1T tokens, 20% of mix: base avg 52.70→64.09 | — | REJECT (licence). Use Specialized-v1 STEM-SFT instead |
| OpenThoughts3-1.2M | `open-thoughts/OpenThoughts3-1.2M` | C | 1.2M rows; avg ~14.5k tokens per example (~16.5B tokens) | Apache-2.0; generator QwQ-32B | 8B / 1T / 20% mix: avg 52.70→54.98. Strong SFT data (OpenThinker3-7B) | SFT / long-context mid-training | **DEFER** (too long for 2048 context) |
| SmolTalk2 | `HuggingFaceTB/smoltalk2` | C | Mid: 4.78M rows, 35.2B tokens. SFT: ~3.4M rows | Mixed: new subsets Apache-2.0, others per source | SmolLM3 ran 140B tokens (4 epochs) of reasoning mid-training. No isolated delta; it hurt long-context RULER | Mid / SFT at ≥8k context | DEFER |
| Tulu 3 SFT mixture (already in) | `allenai/tulu-3-sft-mixture` | C | 939,344 samples | ODC-By, **but** includes CC-BY-NC No Robots and GPT-4 outputs (WildChat) | — | SFT | **AUDIT** |
| SmolTalk v1 (already in) | `HuggingFaceTB/smoltalk` | C | ~1M samples | Apache-2.0 for new parts; Smol-Magpie-Ultra comes from Llama-3.1-405B | — | SFT | **AUDIT** (Llama naming clause) |

---

## 2. Dataset notes

### A. Reference, academic, and books

#### FineWiki — `HuggingFaceFW/finewiki`
- **What:** A re-extraction of Wikipedia from HTML dumps (August 2025), 325 languages. Unlike `wikimedia/wikipedia`, it renders templates and keeps headings, lists, code, tables and **math**. It drops redirects, disambiguation pages, and most reference and "See also" sections, and flags pages that contain math (`has_math`). English: 6,614,655 pages, 35.1 GB. Total: 61.55M pages. Source: [dataset card](https://huggingface.co/datasets/HuggingFaceFW/finewiki).
- **Tokens:** The SYNTH paper describes English FineWiki as "~9B tokens" with its 65k-vocabulary tokenizer ([arXiv 2609.37891](https://arxiv.org/html/2609.37891v1), §4.2). Qwen3 count: [unverified].
- **Licence:** CC-BY-SA-4.0 + GFDL. ShareAlike applies to redistributed shards: record it per shard.
- **Evidence:** I found no isolated small-model ablation of "add Wikipedia, measure gain". Indirect evidence:
  - Wikipedia is upweighted in every OLMo 2 Dolmino anneal mix (100% of source in 50B and 100B mixes, 400% in the 300B mix) ([dolmino card](https://huggingface.co/datasets/allenai/dolmino-mix-1124)).
  - Synthetic data derived from FineWiki helps factual recall: NVIDIA Fact-Seeking raised a proxy SimpleQA from 40.2 to 50.2 ([card](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.2)).
  - Negative data point: a 600M model trained only on FineWiki (~17 epochs) stayed near chance on multiple choice (26.1) ([SYNTH paper Table 8](https://arxiv.org/html/2609.37891v1)). So Wikipedia is a supplement, not a whole diet.
- **Gap targeted:** encyclopedic (+1.80). Wikipedia text is the domain of the wikitext evaluation.
- **Contamination:** **High for CrowdGPT's own metric.** WikiText-103 is built from Wikipedia articles, so training on FineWiki will lower wikitext loss partly through overlap rather than skill. Use a held-out, deduplicated Wikipedia eval slice, or remove wikitext articles from the training shards. SQuAD and TriviaQA also draw on Wikipedia.
- **Overlap:** FineWeb and Ultra-FineWeb may already contain some Wikipedia pages [unverified]. CrowdGPT has no dedicated Wikipedia source.
- **Phase:** Main pretraining at a few percent, upweighted (2–4 epochs) in the anneal, following Dolmino practice.

#### FinePDFs / FinePDFs-Edu — `HuggingFaceFW/finepdfs`, `HuggingFaceFW/finepdfs-edu`
- **What:** Text extracted from Common Crawl PDFs (2013 to Feb 2025): about 3T tokens, 475M documents, 1,733 languages. English: 206.9M docs, 1,190.65B tokens ([card](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)). FinePDFs-Edu keeps the top 10% by an education classifier: 350B+ tokens, 69 languages ([card](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu)). The Qwen3-235B-A22B-Instruct-2507 labels only trained the filter, so they add no generator terms to the text.
- **Licence:** ODC-By 1.0, plus Common Crawl Terms of Use.
- **Evidence:** HF trained 1.67B models on 36B tokens at 4096 context ([blog appendix](https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog)).
  - FinePDFs alone is "nearly on par" with the SmolLM3 web mix.
  - FinePDFs-Edu alone "already surpasses Nemotron-CC v2".
  - FW-Edu + DCLM + 25% FinePDFs-Edu was "the best performing dataset across all our ablations".
  - The card recommends keeping PDFs **under 25%** of the mix.
  - Exact benchmark deltas are only shown in interactive charts: [unverified numbers].
- **Gaps targeted:** science (+1.50) and encyclopedic (+1.80). PDFs skew towards education, legal and technical documents.
- **Contamination:** Exam papers and textbooks in PDFs could contain benchmark items. HF ships a small test split per language. A benchmark decontamination pass is not documented on the card [unverified].
- **Overlap:** None. CrowdGPT has no PDF-derived source.
- **Phase:** Main pretraining (10–25%) and anneal. Long documents are fine at 2048 context after chunking.

#### peS2o — `allenai/peS2o` (open subset: `common-pile/peS2o_filtered`)
- **What:** About 40M open-access papers and abstracts from S2ORC. v2: 38.97M docs, 42.01B whitespace tokens, cutoff 2023-01-03 ([card](https://huggingface.co/datasets/allenai/peS2o)).
- **Licence:** The collection is ODC-By, but individual papers carry many different open-access licences. Common Pile kept only openly licensed papers: 43.3B Comma tokens in its main stage ([Comma mix card](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset)). Prefer the Common Pile version for per-shard licence records.
- **Evidence:** Used in OLMo 2's anneal: 58.6B tokens in the Dolmino pool, 5.85–19.4% of mixes. It is part of the reference bundle in the 1B result in §2C. No isolated delta found.
- **Gap targeted:** science (+1.50).
- **Phase:** Small share of main pretraining, more in the anneal.

#### arXiv (openly licensed) — `common-pile/arxiv_papers(_filtered)`
- **What:** 321,336 papers under CC BY, CC BY-SA or CC0, converted from LaTeX through LaTeXML to text. 21 GB; 6.0B Comma tokens ([card](https://huggingface.co/datasets/common-pile/arxiv_papers)). RedPajama arXiv is larger but mixes licences: [unverified per-paper licensing].
- **Evidence:** None isolated. It is a modest source of dense math and science text.
- **Gaps targeted:** math (+2.44) and science (+1.50).

#### StackExchange — `common-pile/stackexchange(_filtered)`
- **What:** All questions, answers and comments, ordered by votes with the accepted answer first. 33.4M docs, 103.7 GB raw; 23.9B Comma tokens filtered ([card](https://huggingface.co/datasets/common-pile/stackexchange)).
- **Licence:** CC-BY-SA, per document. Stack Exchange stopped publishing public dumps in July 2024; Common Pile used community-uploaded Internet Archive dumps from December 2024. Whether newer Stack Exchange terms restrict LLM training on those dumps: [unverified — check before use].
- **Evidence:** In Dolmino (1.26B tokens, upsampled 2–4×) and the Comma mix (repeated 6×). No isolated delta found.
- **Gaps targeted:** code (+3.84; Stack Overflow Q&A with code) and science.
- **Contamination:** Math and science sites may contain benchmark problems [unverified].
- **Overlap:** Partly overlaps `codeparrot/github-code-clean` only in topic, not in source.

#### Project Gutenberg / public-domain books — `common-pile/project_gutenberg`
- **What:** 71,810 English public-domain books (plus PG-19), 26.2 GB; 5.7B Comma tokens. Common Pile also has pre-1929 books (12.4B) and Library of Congress (9.5B).
- **Evidence:** No per-token gains on CrowdGPT's weak areas. Old prose has limited value for code, math or science.
- **Verdict:** Low priority. Small weight only, for long-range prose.

#### Common Pile v0.1 — `common-pile/*`
- **What:** 8 TB of openly licensed text from 30 sources ([arXiv 2506.05209](https://arxiv.org/abs/2506.05209)).
- **Evidence:**
  - Comma v0.1-1T (7B) is competitive with Llama-1 7B and MPT-7B. Comma v0.1-2T is competitive with OLMo-7B-Twin-2T and Llama-2 7B.
  - With identical 1.7B models trained on 28B tokens, the Comma mix beats other openly licensed corpora and beats the Pile on all but two tasks (paper Fig. 7).
- **Use for CrowdGPT:** The best place to get A-category shards (peS2o, arXiv, StackExchange, Gutenberg) **with per-document licence metadata**, which CrowdGPT's per-shard licence record needs.

#### Institutional Books 1.0 — `institutional/institutional-books-1.0`
- **What:** 983,004 public-domain volumes (242B o200k tokens) from Harvard Library's Google Books scans ([arXiv 2506.08300](https://arxiv.org/abs/2506.08300), [card](https://huggingface.co/datasets/institutional/institutional-books-1.0)).
- **Licence:** IDI Terms of Use for Early-Access: gated, **non-commercial, no sharing or redistribution** ([Open Future summary](https://openfuture.eu/blog/common-pile-and-institutional-books-datasets-chart-two-pathways-for-ai-commons/)).
- **Verdict:** **REJECT.** The terms conflict with CrowdGPT redistributing shards to volunteers.

### B. Synthetic pretraining data

#### FinePhrase — `HuggingFaceFW/finephrase` (headline pick)
- **What:** FineWeb-Edu (sample-350BT) rephrased by **SmolLM2-1.7B-Instruct** into four formats: FAQ, math word problem, table plus Q&A, and tutorial. Result: 1.354B samples and 486.4B completion tokens ([card](https://huggingface.co/datasets/HuggingFaceFW/finephrase); paper [arXiv 2604.13977](https://arxiv.org/html/2604.13977), COLM 2026). Note: the `text` column holds the **source** document; the rephrase is in `rollout_results[0].text`.
- **Licence:** ODC-By. The generator, SmolLM2, is Apache-2.0 and FineWeb-Edu is ODC-By, so this is the cleanest licence chain of any large synthetic set I found.
- **Evidence:** 1.2B Qwen2-architecture model, 21B tokens, 12 benchmarks, macro average ×100. Tables 2–7 of the paper:

  | Dataset or prompt | Macro average |
  |---|---|
  | DCLM | 13.77 |
  | Ultra-FineWeb | 13.00 |
  | FineWeb-Edu-HQ | 11.82 |
  | Nemotron-HQ-Synth | 13.54 |
  | REWIRE | 13.49 |
  | Cosmopedia | 10.33 |
  | SYNTH | 10.03 |
  | FinePhrase prompt: math | 15.31 |
  | FinePhrase prompt: table | 14.83 |
  | FinePhrase prompt: faq | 14.45 |
  | FinePhrase prompt: tutorial | 14.30 |
  | **FinePhrase-Table (final, SmolLM2 generator)** | **17.18** (+3.41 over DCLM, +3.63 over Nemotron-HQ-Synth) |

  - The gain holds for 500M, 1.2B, 2.9B and 6.2B students (Table 5).
  - Seed variance is ±0.16–0.49, against a +4.71–5.23 gain.
  - Sweeping the synthetic share from 10% to 90% beat the baseline at every point; the peak was 60–80% synthetic.
- **Directly relevant to CrowdGPT:**
  - Ultra-FineWeb, which CrowdGPT uses, scored 13.00 in the same harness. FinePhrase is measured against it.
  - Synthetic data **hurts HellaSwag and PIQA** (e.g. table format: HS 7.66 vs DCLM 9.25) and must be mixed with real web text. It wins on ARC, SQuAD and DROP.
  - A ~1B generator is enough; larger generators gave no extra gain (§4.2). **CrowdGPT's volunteers could therefore generate their own rephrasings on 16 GB GPUs with an Apache-2.0 1–2B model.** That would be a new kind of volunteer work [design idea, not evidence].
- **Gaps targeted:** encyclopedic, science, and math (the math prompt scored best).
- **Contamination:** 10-gram audit (Appendix E.4): 735 flagged docs per billion tokens, versus 841 for the FineWeb-Edu-HQ source. Rephrasing slightly reduced contamination.
- **Overlap:** The source is FineWeb-Edu, and CrowdGPT already has FineWeb and Ultra-FineWeb. FinePhrase restates similar knowledge in new formats; it adds little new knowledge. The paper's main finding is exactly that the format helps.
- **Phase:** Main pretraining. Note the paper's ratios (50–80% synthetic) are measured at 21B-token horizons only. Long-horizon behaviour: [unverified].

#### Nemotron-Pretraining-Specialized v1 / v1.1 / v1.2 — `nvidia/Nemotron-Pretraining-Specialized-*`
These are CC-BY datasets and are not gated. They are **not** covered by the NVIDIA Data Agreement, unlike Nemotron-CC-v2. Each card lists the generator for each subset. Picking subsets by generator avoids most generator-licence issues:

| Subset | Tokens | Generator(s) | Licence | Generator terms note |
|---|---|---|---|---|
| Code-Concepts (v1.1) | 7.29B | gpt-oss-20b / 120b | CC-BY-4.0 | Apache-2.0 generators [gpt-oss licence: unverified this session]. Clean |
| Fact-Seeking (v1.2) | 35.0B | Qwen3-30B-A3B-Instruct-2507 | CC-BY-4.0 | Apache-2.0 generator [unverified this session]. Clean |
| Wiki-Rewrite (v1) | 7.9B | Qwen3-30B-A3B | CC-BY-SA + GFDL | Clean; ShareAlike |
| Math-Textbooks (v1) | 25.1B | Qwen3-30B-A3B, Qwen3-235B-A22B | CC-BY-4.0 | Clean |
| RQA, STEM reasoning Q&A (v1) | 134.6B | Qwen3-235B-A22B-Thinking-2507, gpt-oss-120b | CC-BY-4.0 | Clean |
| STEM-SFT (v1) | 82.5B | DeepSeek-R1-0528, Qwen2.5-32B | CC-BY-4.0 | R1-0528 is MIT and Qwen2.5-32B is Apache-2.0 [both unverified this session]. Card warns generically about Qwen, DeepSeek and Phi-4 terms |
| Multiple-Choice (v1.1, v1.2), Generative (v1.2) | 1.6B / 6.1B / 0.7B | **DeepSeek-v3**, Qwen3-235B | CC-BY-4.0 | Card: models trained on it "may be subject to … DeepSeek License Agreement". **Exclude** |

- **Evidence:**
  - Code-Concepts: adding 10B tokens into the **final 100B** of Nemotron Nano v3 pretraining raised HumanEval from 73 to 79; other benchmarks were roughly unchanged ([NVIDIA blog](https://huggingface.co/blog/nvidia/synthetic-code-concepts)).
  - Fact-Seeking: on an intermediate Nemotron 3 Nano base checkpoint, proxy SimpleQA rose from 40.2 to 50.2 ([card](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.2)).
  - Both are measured on a 30B-total / 3B-active model, **not** a small dense model. Transfer to 1B: [unverified].
- **Gaps targeted:** Code-Concepts → code (+3.84). Fact-Seeking and Wiki-Rewrite → encyclopedic. RQA, STEM-SFT and Math-Textbooks → science and math.
- **Contamination:** **Code-Concepts was built from the 91 concepts found in HumanEval prompts**, so it is benchmark-targeted. Judge it on MBPP, LiveCodeBench, or a private code eval, not HumanEval alone. The Multiple-Choice sets are built around MMLU auxiliary-train topics.
- **Phase:** Anneal / last 10–20% of pretraining (that is how NVIDIA validated them).

#### Nemotron-CC (v1) and Nemotron-CC-v2 synthetic rephrases
- **v1:** 6.3T tokens, of which 1.9T are synthetic. Diverse QA alone is 499.5B. Rephrased by Mistral NeMo 12B Instruct. Hosted at [data.commoncrawl.org](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html) under Common Crawl terms; redistribution terms: [unverified].
- **Evidence:**
  - With 8B models trained on 1T tokens, the HQ subset scores MMLU 59.0 vs DCLM 53.4 (+5.6) ([arXiv 2412.02595](https://arxiv.org/abs/2412.02595)).
  - In OLMo 3 microanneals, only **Diverse QA** beat top-quality web (GenQA +1.5, MC non-STEM +1.9). Distill, Extract Knowledge, Knowledge List and WRAP-medium did worse than natural data ([OLMo 3, arXiv 2512.13961](https://arxiv.org/html/2512.13961) §3.5.2).
  - The FinePhrase study agrees: Diverse QA 14.58 vs DCLM 13.77; Extract Knowledge 11.81.
- **v2 / v2.1 (HF):** The [NVIDIA Data Agreement](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/raw/main/LICENSE.md) §2.1–2.2 allows only "internal training" and forbids "distribute … or otherwise make available to others". It also forbids making the data subject to an open-source licence. **This is incompatible with CrowdGPT's shard distribution.** Access is gated with manual approval.
- **Verdict:** v1 Diverse QA is CONSIDER, pending a Common Crawl terms review. v2 and v2.1 are REJECT.

#### PleIAs SYNTH — `PleIAs/SYNTH`
- **What:** 79,648,272 samples, ~75B tokens (Pleias tokenizer), built from 58,698 Wikipedia "vital" and specialist articles. Covers memorisation Q&A, RAG, arithmetic, editing, creative writing and multiple choice, with reasoning traces. About 20% is non-English ([card](https://huggingface.co/datasets/PleIAs/SYNTH); paper [arXiv 2609.37891](https://arxiv.org/html/2609.37891v1)).
- **Licence chain (flag):** The card says CC-BY-4.0. The paper says:
  - The query generator and other auxiliary generators are **LoRAs fine-tuned on Gemma-3-12B-base**. Arithmetic uses Qwen-3-8B.
  - Their Stage-1 training data came from **Gemini 2.5 Pro**: 81,264 calls, 133.2M output tokens (Appendix C, Table 3).
  - The [Gemma Terms of Use](https://ai.google.dev/gemma/terms) §1.1(e) define "Model Derivatives" to include models trained on "synthetic data Outputs by Gemma … in order to cause that model to perform similarly to Gemma". Whether that reaches a model trained on SYNTH needs a legal reading.
  - Gemini API terms on competitive use: [unverified].
  - **Human review required before use.**
- **Evidence:**
  - Own results: Baguettotron (321M, 80 layers) on ~200B SYNTH tokens. In a matched 600M ablation (Table 8), SYNTH scored 42.2 multiple-choice average vs 26.1 (FineWiki) and 25.1 (FinePDFs-Edu). The web models got only SmolTalk + MMLU-aux post-training, so the comparison favours SYNTH's built-in instruction format.
  - Independent cloze-format evaluation (FinePhrase study): SYNTH 10.03 vs DCLM 13.77, **below** curated web.
  - So SYNTH's advantage shows up in instruction-format / generative evals, not in cloze pretraining evals.
- **Gap targeted:** encyclopedic. **Phase:** possibly a small anneal share, only if the licence is cleared.

#### Cosmopedia v1 / v2
- **v1:** 25B tokens, 30M files; Mixtral-8x7B-Instruct; Apache-2.0; the card describes decontamination ([card](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)).
- **v2:** 28B tokens, in `smollm-corpus` (ODC-By). HF found no gain from swapping in Llama-3-70B, Mixtral-8x22B or Qwen1.5-72B as generators ([SmolLM blog](https://huggingface.co/blog/smollm)).
- **Evidence:**
  - On its own, Cosmopedia trails DCLM (10.33 vs 13.77, FinePhrase study). Used as the mix-in partner it is also weak (Table 8: 12.99 vs DCLM 14.32).
  - BeyondWeb reports beating Cosmopedia by up to 5.1pp at 8B ([arXiv 2508.10975](https://arxiv.org/abs/2508.10975)).
  - SmolLM2 used it at only 4% in its final stage ([arXiv 2502.02737](https://arxiv.org/abs/2502.02737)).
- **Verdict:** Low priority. Newer rephrase-based sets beat it per token.

#### BeyondWeb, WRAP, TinyStories, phi-1 (evidence and methods only)
- **BeyondWeb** ([arXiv 2508.10975](https://arxiv.org/abs/2508.10975)): not released.
  - At 8B / 180B tokens: 63.7% average on 14 tasks vs Nemotron-Synth (+2.6pp) and RedPajama (+7.1pp). Reaches the RedPajama target after 23.2B tokens and the Nemotron-Synth target after 66.2B.
  - Its 3B model beats the 8B Cosmopedia model.
  - Lessons: diversity matters, and gains from the rephraser flatten at 3–8B.
- **WRAP** ([arXiv 2401.16380](https://arxiv.org/abs/2401.16380), ACL 2024): rephrasing C4 into Wikipedia-style and Q&A style. ~3× faster pretraining, >10% perplexity improvement on Pile subsets (abstract), >2% zero-shot over 13 tasks (1.3B model). This is the origin of the rephrasing method.
- **TinyStories** ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759)): GPT-3.5 / GPT-4 stories for models under 10M params. Not useful for CrowdGPT's gaps; OpenAI outputs; CDLA-Sharing. **REJECT.**
- **phi-1** ([arXiv 2306.11644](https://arxiv.org/abs/2306.11644)): 1.3B model, 6B filtered web + 1B GPT-3.5 textbooks: HumanEval 50.6%, MBPP 55.5%. Strong evidence that "textbook-quality" code data punches above its weight, but the data is unreleased and proprietary-generated.

### C. Instruction and reasoning data for mid-training / annealing

#### OLMo 2 Dolmino Mix 1124 — `allenai/dolmino-mix-1124` (headline small-model evidence)
- **What:** An anneal pool of 843B tokens. In practice OLMo 2 annealed on 50B, 100B or 300B mixes: about 50% DCLM, plus FLAN (17.0B), peS2o (58.6B), Wikipedia and Wikibooks (3.7B), StackExchange (1.26B) and math (TuluMath, TinyGSM-MIND, MathCoder2 and others) ([card](https://huggingface.co/datasets/allenai/dolmino-mix-1124)).
- **Evidence, OLMo 2** ([arXiv 2501.00656](https://arxiv.org/abs/2501.00656), Table 9):
  - **OLMo 2 1B**, after 4T pretraining tokens, then 50B Dolmino tokens: average 31.9→43.7, MMLU 26.9→44.3, ARC-C 26.1→51.3, GSM8K 3.3→43.8, DROP 25.1→34.0.
  - 7B: average 53.0→62.9.
  - **Caveat:** Part of this is the learning-rate anneal. On a 7B model, annealing on the pretraining mix alone gave OLMES +4.4 and MMLU +2.0, with GSM* −1.5 (Table 11; MMLU 59.8→61.8).
  - Adding instruction data ("+Ins") lifted GSM* from 28.5 to 46.0. Adding math lifted it to 52.0 (Table 11).
- **Licence:** ODC-By for the collection, plus Common Crawl terms. Per-source: StackExchange and Metamath are CC-BY-SA; MathCoder2 is Apache-2.0; GSM8K is MIT. Generator terms for TinyGSM-MIND and TuluMath: [unverified]. TuluMath was persona-generated; its generator model is not stated on the card.
- **Use for CrowdGPT:** Copy the **recipe**: ~50% best web, plus reference sources (FineWiki, peS2o, StackExchange), plus decontaminated FLAN, plus math, for the last ~5–10% of tokens. CrowdGPT already has better math sources. Use OLMo 2's **microanneal** method (anneal on a 50/50 mix of candidate source and web; OLMo 3 used 5B + 5B) to test each candidate cheaply on volunteer compute.

#### OLMo 3 Dolma 3 Dolmino Mix — `allenai/dolma3_dolmino_mix-100B-1125`
- **Evidence** ([arXiv 2512.13961](https://arxiv.org/html/2512.13961) §3.5):
  - With total tokens held fixed in a 100B integration test, including instruction and thinking data improved every base metric (Table 10): average 48.8→50.7, MC STEM 63.6→64.9, GenQA 43.1→48.7, math 66.7→68.1, code 23.3→24.4. Model size for this test: [unverified, likely 7B].
  - Microanneals: Wiki-to-RCQA (Wikipedia passages → reading-comprehension Q&A by Qwen2.5-32B-Instruct) gave about +2 GenQA. Reddit-to-Flashcards gave +2 MC non-STEM and +3 MMLU.
- **Licence flags:**
  - **Reddit-to-Flashcards was written by GPT-4o-mini** (OpenAI terms).
  - The mix also includes "Gemini Reasoning Traces" (246M tokens) and OpenThoughts2 traces.
  - Exclude the GPT and Gemini parts. Wiki-to-RCQA does not appear to be released on its own [unverified; searched HF].
- **Contamination:** OLMo 3 found **FLAN and Nemotron were the main contamination sources**, including complete validation and test splits inserted through templates. DROP, Minerva and SQuAD scores fell noticeably after decontamination.

#### FLAN
- The Dolmino version holds 17.0B tokens. The `Muennighoff/flan` task list includes `arc_challenge`, `arc_easy`, `hellaswag`, `piqa`, `openbookqa`, `bool_q`, `squad_v2`, `drop` and `trivia_qa`. **These are the standard base-model evals.**
- Already partly in CrowdGPT: Tulu 3 contains 89,982 FLAN v2 prompts.
- **Verdict:** Use only after decontaminating against every eval CrowdGPT reports.

#### Front-Loading Reasoning (evidence for putting SFT-style data into pretraining)
- [arXiv 2510.03264](https://arxiv.org/abs/2510.03264), ICLR 2026. Setup: 8B model, 1T tokens, 80% general data and 20% reasoning data, 8k context.
- Base-model averages (Table 1):
  - No reasoning data: 52.70
  - OpenThoughts3 (small, high quality): 54.98
  - Nemotron-Pretraining-SFT-v1 (336B, diverse): 64.09
  - Both combined: 64.07
- After SFT the gap widens: 26.62 vs 35.92 average (Table 2).
- **Lessons:** Diversity and scale matter most in pretraining; quality matters most in SFT.
- **Caveat:** the winning dataset (Nemotron-Pretraining-SFT-v1) is gated with an "other" licence. Its card names Nemotron-Pretraining-Specialized-v1 as its successor, "with updated naming". Whether STEM-SFT in Specialized-v1 has the same content: [unverified].

#### Nemotron-Post-Training-Dataset-v1 — `nvidia/Nemotron-Post-Training-Dataset-v1`
- CC-BY-4.0, not gated. Responses from DeepSeek-R1-0528 (24.6M samples) and Qwen3-235B-A22B (1.06M) ([card](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1)). Both generator licences are permissive [MIT and Apache-2.0: unverified this session].
- It has reasoning-on and reasoning-off samples, and splits for math, code, STEM, chat and tool use. Filter to samples of 2048 tokens or fewer for CrowdGPT.
- I found no small-model pretraining ablation. Treat it as a CONSIDER anneal ingredient.

#### OpenThoughts3-1.2M and SmolTalk2 (defer)
- **OpenThoughts3:** Apache-2.0. QwQ-32B wrote 16 answers per question for 850k math, 250k code and 100k science questions ([card](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M), [arXiv 2506.04178](https://arxiv.org/abs/2506.04178)).
  - The **average example is ~14.5k tokens** (SmolTalk2 statistics). At CrowdGPT's 2048 context, nearly every example would be cut off before its answer.
  - It is also mostly math, which overlaps OpenMathReasoning, Nemotron-Math-v2 and OpenR1-Math.
- **SmolTalk2:** the Mid subset is 35.2B tokens of R1 and QwQ traces (average 7.4k tokens). SmolLM3 trained 4 epochs (~140B tokens) on it and later had to repair long-context loss by model merging ([SmolLM3 blog](https://huggingface.co/blog/smollm3)).
- **Verdict:** Revisit both when CrowdGPT trains ≥4B models at ≥16k context.

#### Already in CrowdGPT: licence audit needed
- **`allenai/tulu-3-sft-mixture`:** The card says "Some portions of the dataset are non-commercial". It includes **No Robots (CC-BY-NC-4.0, 9,500 prompts)** and **WildChat GPT-4 responses (100,000)**. Evol-CodeAlpaca is GPT-generated [unverified generator]. Persona subsets: generator not stated on card [unverified].
- **`HuggingFaceTB/smoltalk`:** Smol-Magpie-Ultra (400K) was generated with **Llama-3.1-405B-Instruct**. The [Llama 3.1 licence](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE) requires that a distributed model trained on Llama outputs include **"Llama" at the start of its name**, and display "Built with Llama".
- **Recommendation:** Drop or replace these subsets, or record the obligations per shard.

---

## 3. Ranked top picks for CrowdGPT (2 per category)

### A. Reference / academic / books
1. **FinePDFs-Edu (English subset)** — `HuggingFaceFW/finepdfs-edu`
   - Why: The best-evidenced new reference source. HF found it beats Nemotron-CC v2 on its own and gives the best mix at 25% alongside web data (1.67B / 36B-token ablations).
   - Targets the science and encyclopedic gaps. ODC-By, 350B+ tokens, so it never runs out.
   - How: 10–20% of main pretraining, more in the anneal. Start below HF's 25% ceiling.
2. **FineWiki (English)** — `HuggingFaceFW/finewiki`
   - Why: The direct fix for the encyclopedic gap. Better extraction than `wikimedia/wikipedia`: it keeps math and tables, which also helps the math gap.
   - Every strong open mix (Dolmino, Comma) upweights Wikipedia.
   - How: a few percent in main pretraining, 2–4 epochs in the anneal.
   - **Fix the wikitext eval first**, or the measured gain will be inflated. Record CC-BY-SA on its shards.
   - Runners-up: **Common Pile StackExchange (filtered)** for the code gap (CC-BY-SA; check Stack Exchange terms) and **Common Pile peS2o (filtered)** for science.

### B. Synthetic
1. **FinePhrase** — `HuggingFaceFW/finephrase`
   - Why: The largest measured per-token gain in a controlled, independent-baseline study. +3.4 macro points over DCLM at 1.2B / 21B, holding from 0.5B to 6.2B students.
   - Cleanest licence chain (Apache-2.0 generator, ODC-By source and output). 486B tokens.
   - Its paper also shows a 1B-class generator is enough, which suits a future volunteer-generated synthetic data stream.
   - How: start around 20–40% of main-pretraining tokens, always mixed with Ultra-FineWeb [suggestion; the paper's 60–80% optimum is only shown at 21B tokens]. Prioritise the **math** and **table** configs.
2. **Nemotron-Pretraining-Specialized, selected subsets** — Code-Concepts (v1.1), Fact-Seeking (v1.2), Wiki-Rewrite, Math-Textbooks, RQA (v1)
   - Why: Each subset targets one CrowdGPT gap directly: code (HumanEval +6 when added late), encyclopedic (proxy SimpleQA +10), science and math.
   - CC-BY licensed, not gated. The chosen subsets use Apache-licensed Qwen3 / gpt-oss generators.
   - How: late-pretraining / anneal ingredients (5–15% of the final 10–20% of tokens).
   - Skip the DeepSeek-v3 Multiple-Choice and Generative subsets.
   - Check Code-Concepts on non-HumanEval code evals.

### C. Instruction / reasoning for mid-training / annealing
1. **OLMo 2 Dolmino-style anneal** (non-web parts of `allenai/dolmino-mix-1124`, combined with A and B picks)
   - Why: The strongest published small-model evidence for mid-training. OLMo 2 **1B** after 50B Dolmino tokens: MMLU 26.9→44.3, GSM8K 3.3→43.8, average 31.9→43.7. Some of this is the learning-rate anneal itself.
   - OLMo 3 adds that instruction and thinking data improve base metrics even before SFT (average 48.8→50.7).
   - How: last 5–10% of each model's token budget. ~50% Ultra-FineWeb top slice; the rest FineWiki, peS2o, StackExchange, **decontaminated** FLAN, and the existing math sets.
   - Use microanneals (5B + 5B tokens, as in OLMo 3) to accept or reject each source.
2. **Nemotron-Pretraining-Specialized-v1 STEM-SFT** (82.5B; DeepSeek-R1-0528 + Qwen2.5-32B), backed by **Nemotron-Post-Training-Dataset-v1** short samples
   - Why: Front-Loading Reasoning shows large, diverse SFT-style Q&A in pretraining beats small, curated long-chain-of-thought data. At 8B / 1T: average 52.70→64.09, versus 54.98 for OpenThoughts3.
   - These are the openly licensed, redistributable relatives of the gated dataset that won that study.
   - How: filter to ≤2048 tokens, then use as ~5–20% of the late-pretraining / anneal mix.
   - Science and code subsets first, since math is already covered by OpenMathReasoning and Nemotron-Math-v2.
   - Confirm STEM-SFT ≈ Nemotron-Pretraining-SFT-v1 content [unverified].

### Explicit "do not use" for this project
- Institutional Books (non-commercial, no redistribution).
- Nemotron-CC-v2 / v2.1 and Nemotron-Pretraining-SFT-v1 (NVIDIA Data Agreement or gated "other" licence: no redistribution).
- TinyStories (OpenAI outputs).
- Proprietary-generated parts of Dolma 3 Dolmino (GPT-4o-mini, Gemini).
- SYNTH, until a legal review of the Gemma and Gemini chain.

---

## 4. Unverified items (check before acting)
1. Qwen3-tokenizer token counts for every dataset. All counts above use source tokenizers.
2. English token share of FinePDFs-Edu, and exact FinePDFs / FinePDFs-Edu benchmark deltas (interactive charts only).
3. Generator licences restated from general knowledge, not re-fetched this session: DeepSeek-R1-0528 (MIT), QwQ-32B / Qwen3 / Qwen2.5-32B (Apache-2.0), gpt-oss (Apache-2.0), Mistral NeMo / Mixtral (Apache-2.0), SmolLM2 (Apache-2.0).
4. Gemini API and OpenAI terms on using outputs to build competing models (relevant to SYNTH, Dolma 3 Reddit-to-Flashcards, Tulu 3 WildChat, TinyStories).
5. Whether the Gemma "Model Derivatives" clause reaches models trained on SYNTH.
6. Common Crawl terms on redistributing Nemotron-CC v1 synthetic text.
7. Stack Exchange's post-2024 dump terms for LLM training.
8. Whether FineWeb / Ultra-FineWeb already contain Wikipedia pages (overlap with FineWiki).
9. Model size used in OLMo 3 Table 10 integration tests.
10. Whether Nemotron-Pretraining-Specialized-v1 STEM-SFT equals the Nemotron-Pretraining-SFT-v1 data used in Front-Loading Reasoning.
11. Generator models for Tulu 3 persona subsets, Evol-CodeAlpaca, TuluMath and TinyGSM-MIND.
12. All transfer of results from 1.2B–8B study models to CrowdGPT's 1B dense / 27B MoE path at 2048 context. No study used exactly CrowdGPT's setup.
13. The mixing percentages in §3 are suggestions derived from the cited studies, not measured results for CrowdGPT.
