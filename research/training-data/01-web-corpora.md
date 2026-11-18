DRAFT — requires human review

# 01 — General-purpose web pretraining corpora for CrowdGPT

Research date: 2026-11-18. Scope: English (plus optional multilingual) web corpora that give the most benchmark
capability per training token at small scale (100M–7B parameters, ≤1T tokens). Every number below was read from
the primary source linked next to it (arXiv paper or Hugging Face dataset card) unless it is tagged **[unverified]**.

Ground rules used when reading the evidence:

- Token counts are in each source's own tokenizer (GPT-2, gpt-neox, Llama-3, MiniCPM5, ...). Counts under the
  Qwen3 tokenizer CrowdGPT uses will differ. **[unverified for Qwen3]** everywhere.
- "Licence" is the dataset-level licence on the card. ODC-By (and similar collection licences) covers the
  compilation only; it does not make the underlying web documents openly licensed (Common Pile paper,
  arXiv:2506.05209, §2 "Use of collection licenses"). Per-shard licence recording should note this.
- Most ablations are run by the dataset's own authors on their own evaluation suite. Cross-paper comparisons are
  only made where one paper trained on both datasets under one setup.

CrowdGPT's budget (tens to a few hundred billion tokens per model size) is far below the unique-token size of
every candidate here, so forced repetition is not the issue. The issue is choosing the highest-yield slice and not
paying twice for the same FineWeb documents.

---

## 1. Summary table

| Dataset | HF id | Tokens (source tokenizer) | Licence | Evidence headline | Verdict for CrowdGPT |
|---|---|---|---|---|---|
| DCLM-Baseline 1.0 | `mlfoundations/dclm-baseline-1.0` | 3.8T (paper) | CC-BY-4.0 | 1B/29B tok: CORE 30.2 vs FineWeb-Edu 26.6; 7B/0.28T: MMLU 50.8 vs FineWeb-Edu 37.3 (arXiv:2406.11794 Tables 33 and main table). Best non-synthetic corpus and best "mix-in" for synthetic data in FinePhrase study (1.2B/21B). | **Adopt** (rank 1). Replace raw FineWeb as the real-text backbone. |
| FineWeb-Edu (int_score ≥3) | `HuggingFaceFW/fineweb-edu` | 1.3T (≥2 variant: 5.4T, `fineweb-edu-score-2`) | ODC-By | 1.71B/350B: MMLU 33.6% at 38B tokens; Matrix needs 300B for the same (arXiv:2406.17557 Fig. 11). Weak on commonsense (HellaSwag/PIQA/CSQA). | **Adopt** (rank 2), paired with DCLM. Dedup against CrowdGPT's FineWeb. |
| FinePhrase (synthetic rephrase of FineWeb-Edu) | `HuggingFaceFW/finephrase` | 486.4B completion tokens | ODC-By | 1.2B/21B, 50/50 with real web: table-format variant macro 17.18 vs DCLM 13.77 (+3.41) and Nemotron-HQ-Synth 13.54 (arXiv:2604.13977 §6, Fig. 5). | **Adopt as complement** (rank 3) to Ultra-FineWeb-L3; cap synthetic share. |
| FinePDFs-Edu | `HuggingFaceFW/finepdfs-edu` | 350B+ across 69 languages (English share **[unverified]**) | ODC-By | 1.67B/36B: card/blog say it beats Nemotron-CC v2 with no mixing; numbers shown only in figures **[unverified numerically]**. | **Trial** (rank 4). New, non-overlapping source; check PII. |
| Ultra-FineWeb (L2, English) | `openbmb/Ultra-FineWeb` | ~1T en, 120B zh | Apache-2.0 card; card tells users to check upstream licences (FineWeb = ODC-By) | 1.2B/100B: avg-English 45.89 vs FineWeb-Edu 44.56 vs FineWeb 42.28 (arXiv:2505.05427 Table 4). In FinePhrase's independent run: 13.00 vs DCLM 13.77. | Optional real-text partner for L3; lower priority than DCLM. |
| Ultra-FineWeb-L3 (already used) | `openbmb/Ultra-FineWeb-L3` | 245B en QA + 164B en multi-style (MiniCPM5 tok.) | Apache-2.0 **plus** "no unauthorized unchanged redistribution" clause | Card: 100B-token from-scratch runs, L3 "highest average late in training" vs FinePhrase, DCLM, FineWeb-Edu; figure only **[unverified numerically]**. | Keep, but **check the redistribution clause** against Vxtzq/CrowdGPT. |
| Nemotron-CC v1 (incl. HQ subset) | not on HF; `data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/` | 6.3T (4.4T real + 1.9T synthetic); HQ subset 1.1T | Common Crawl Terms of Use | 8B/1T (73% CC): HQ avg 60.1 / MMLU 59.0 vs DCLM 57.0 / 53.4 vs FineWeb-Edu 53.2 / 42.9 (arXiv:2412.02595 Table 5). | Strong evidence; **conditional** on human review of CC ToU for re-hosting. |
| Nemotron-CC v2 / v2.1 | `nvidia/Nemotron-CC-v2`, `nvidia/Nemotron-CC-v2.1` | v2: 6.6T total (3,359.8B English CC); v2.1: +2.5T | NVIDIA Data Agreement for Model Training; gated; Qwen/DeepSeek licence pass-through | No standalone ablation numbers found; FinePDFs blog uses it as the SoTA reference. | **Exclude** unless legal review clears it. Terms restrict use to model training. |
| Essential-Web v1.0 | `EssentialAI/essential-web-v1.0` | 24T, 23.6B docs | ODC-By | Taxonomy-labelled pool. STEM-w-DCLM subset: MMLU-STEM 34.5% vs DCLM 27.7% vs FineWeb-Edu 26.7% (2.3B, 80B-token anneal; arXiv:2506.14111 Table 9). | Use as a **selection tool**; STEM-w-DCLM subset is the SciQ-targeted alternative to rank 4. |
| FineWeb-HQ | `epfml/FineWeb-HQ` | top 10% of FineWeb; 2.45B docs; tokens **[unverified]** | ODC-By | 1B models: avg rank 1.83 vs DCLM 2.39 vs FineWeb-Edu 2.44 (card table). | Skip: full overlap with FineWeb, gains small vs DCLM/FW-Edu. |
| DCLM-Edu | `HuggingFaceTB/dclm-edu` | **[unverified]** | CC-BY-4.0 | 360M/200B beats DCLM and FineWeb-Edu on MMLU/ARC (figure only); gains not consistent at 1.7B (card). | Candidate for the 1B stage only. |
| Zyda-2 | `Zyphra/Zyda-2` | 5.07T (gpt-neox) | ODC-By | Card claims it beats each constituent; figure only. | Useful as a **pre-cross-deduplicated** DCLM + FineWeb-Edu pair. |
| TxT360 | `LLM360/TxT360` | ~5T after global dedup | ODC-By | 8x8B MoE, 1.5T tokens vs FineWeb: loss/eval curves only. | Low priority. |
| GneissWeb | `ibm-granite/GneissWeb` | ~10T (recipe + bloom filter over FineWeb v1.1.0) | Apache-2.0 | 1.4B/350B: +2.73 pp avg over FineWeb on 11 tasks (arXiv:2502.14907). | Skip: subset of FineWeb, compared mainly to FineWeb. |
| Dolma 3 pool | `allenai/dolma3_pool` | 9.31T (Common Crawl 8.14T) | ODC-By | Not a standalone ablation; supplies 24 topics × 20 quality buckets. Quality-aware upsampling beats flat filtering (arXiv:2512.13961 Table 39). | Use for topic/quality-targeted top-ups, not as a whole. |
| FineWeb2 / FineWeb2-HQ | `HuggingFaceFW/fineweb-2`, `epfml/FineWeb2-HQ` | FineWeb2: 20 TB, 5B docs, 1000+ languages (no English) | ODC-By | FineWeb2-HQ: 1B/70B–119B matches baseline MMLU with 15% of the tokens (arXiv:2502.10361). | Optional, small share only if multilingual matters. |
| Common Pile v0.1 (Comma) | `common-pile/comma_v0.1_training_dataset` | 8 TB raw (openly licensed) | per-source open licences | 1.7B/28B: beats other open-licence corpora; FineWeb best on most tasks; Comma best on MMLU/ARC, worse on HellaSwag/PIQA/CSQA. | Licence-safest tier; not a per-token winner. |
| MixtureVitae | `ontocord/MixtureVitae-v1` | 422B (only ~34B web) | card front-matter says ODC-By; HF tag empty **[unverified]** | 1.7B/300B: surpasses FineWeb-Edu, approaches DCLM (arXiv:2509.25531). | Mostly synthetic instruction data, not a web corpus; overlaps CrowdGPT's SFT-style sources. |
| Nemotron-ClimbMix | `nvidia/Nemotron-ClimbMix` | 400B | **CC-BY-NC-4.0** | 1B/400B beats Llama-3.2-1B by 2.0% (arXiv:2504.13161). | **Exclude** (non-commercial). |
| REWIRE | `facebook/recycling_the_web` | 44.4B | **CC-BY-NC-4.0** + Llama 3.3 licence terms | +1.0 / 1.3 / 2.5 pp at 1B / 3B / 7B vs filtered web only (arXiv:2506.04689). | **Exclude** (non-commercial, Llama naming rule). |
| BeyondWeb | none found | — | not released | 3B/180B beats an 8B on Cosmopedia; +2.6 pp over Nemotron-Synth (arXiv:2508.10975). | Not available. |
| CuraWeb | none found on HF | 2T (paper) | **[unverified]** | 3B/1T: +1.8% avg over FineWeb, Dolma3, DCLM (arXiv:2607.22662). | Watch; no public dataset found. |

---

## 2. Per-dataset notes

### 2.1 DCLM-Baseline 1.0

- **Source**: DataComp-LM, arXiv:2406.11794. Card: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- **Size / licence**: 3.8T tokens (paper §1), CC-BY-4.0 (HF tag), ungated. Essential-Web paper quotes 3.6–3.65T.
- **Filtering**: Resiliparse extraction from Common Crawl, RefinedWeb-style heuristics, sharded dedup, then a
  fastText classifier with positives from OpenHermes-2.5 and r/ExplainLikeImFive; keeps top ~10%.
- **Evidence** (all from arXiv:2406.11794):
  - 1B params / 29B tokens: CORE 30.2 vs FineWeb-Edu 26.6, RefinedWeb 25.1, C4 23.7 (Table 33). MMLU at that
    scale is near chance for all (DCLM 23.8, FineWeb-Edu 26.3).
  - 412M / 8B tokens: CORE 18.0 vs FineWeb-Edu 18.3 (Table 33) — no DCLM advantage at the smallest scale.
  - 7B / 0.14T: CORE 44.1 / MMLU 38.3 vs FineWeb-Edu 38.7 / 26.3. 7B / 0.28T: 48.9 / 50.8 vs 41.9 / 37.3.
  - 7B / 2.6T (with StarCoder + ProofPile2): MMLU 63.7.
  - Independent replication: FinePhrase study (arXiv:2604.13977 Table 2, 1.2B, 21B tokens): DCLM 13.77 macro,
    best of the non-synthetic sets (Ultra-FineWeb 13.00, FineWeb-Edu score 4–5 "FWHQ" 11.82).
  - Best "mix-in" partner for synthetic data: with FWHQ-sourced rephrases, mixing in DCLM scored 14.32 vs
    FWHQ 14.30 vs FWLQ 13.61 (arXiv:2604.13977 Table 8).
- **Weaknesses**: card says not intended for code or math. Roughly 80% fuzzy duplicates because dedup was sharded
  (Nemotron-CC paper §2.4). Classifier rewards instruction/Q&A-like style. Contamination: MMLU-overlap flag rate
  0.007% (arXiv:2406.11794 Appendix O); 365 docs per billion tokens with a 10-gram benchmark overlap, the lowest of
  the real-web sets audited in arXiv:2604.13977 Table 25.
- **Overlap with CrowdGPT**: different extraction (Resiliparse) and selection from FineWeb, but many of the same
  URLs. Zyda-2's `dclm_crossdeduped` config is DCLM with FineWeb-Edu-2 duplicates removed.

### 2.2 FineWeb-Edu (and the score-threshold question)

- **Source**: arXiv:2406.17557. Card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **Size / licence**: 1.3T (int_score ≥3); 5.4T for `fineweb-edu-score-2` (≥2). ODC-By. Card changelog v1.4.0
  (2025-07-11) added six 2025 snapshots up to CC-MAIN-2025-26; the post-update total is not stated **[unverified]**.
- **Filtering**: Llama-3-70B-Instruct scored 460k pages 0–5; a regression classifier was trained on those labels;
  threshold 3 chosen (F1 82%). Paper: threshold 3 gave "the best trade-off between performance on knowledge and
  reasoning intensive benchmarks and the performance on other benchmarks like HellaSwag".
- **Evidence**: 1.71B model / 350B tokens; MMLU 33.6% at 38B tokens, Matrix reaches similar accuracy at 300B
  (Fig. 11). Paloma perplexity: FineWeb-Edu fits Wikipedia, S2ORC and arXiv better; FineWeb fits general web,
  forums and social media better (§4.2, Fig. 12). That matches CrowdGPT's wikitext and fineweb-edu loss gaps.
- **Weaknesses**:
  - Commonsense regression vs raw FineWeb at 1.2B/100B: CSQA 31.53 vs 34.32, PIQA 72.14 vs 73.29, HellaSwag
    42.17 vs 42.91 (arXiv:2505.05427 Table 4). The Ultra-FineWeb authors also note FineWeb beats FineWeb-Edu on
    Winogrande, HellaSwag and PIQA at 380B tokens.
  - Contamination: 841 docs/B tokens with a 10-gram overlap for FWHQ (arXiv:2604.13977 Table 25), mostly
    MMLU-Redux and HellaSwag; MMLU-overlap flag rate 0.009% (DCLM paper Appendix O).
  - At 7B it falls well behind DCLM on MMLU (see 2.1).
- **Thresholds**: ≥3 is the authors' choice. FinePhrase used ≥4 ("FWHQ") as its high-quality source and mix-in.
  No primary source compares ≥3 vs ≥4 at CrowdGPT's budget **[unverified]**.
- **Overlap**: FineWeb-Edu documents are a subset of FineWeb and carry FineWeb `id`s, so they can be deduplicated
  against CrowdGPT's FineWeb shards by `id`.

### 2.3 FinePhrase (2026)

- **Source**: arXiv:2604.13977 (COLM 2026). Card: https://huggingface.co/datasets/HuggingFaceFW/finephrase.
  Code: https://github.com/huggingface/finephrase
- **Size / licence**: 1.35B samples, 486,367,076,933 completion tokens across `faq` (148.1B), `math` (98.4B),
  `table` (92.4B), `tutorial` (147.4B). ODC-By. Generator: SmolLM2-1.7B-Instruct (Apache-2.0). Source:
  FineWeb-Edu `sample-350BT`.
- **Evidence** (1.2B Qwen2-architecture student, ~21B tokens, 50/50 synthetic/real):
  - FinePhrase-table macro 17.18 vs DCLM 13.77 (+3.41) and Nemotron-HQ-Synth 13.54 (§6, Fig. 5).
  - Over the DCLM baseline: math +1.54, table +1.06, faq +0.68, tutorial +0.53 (§4.1, Table 4; Gemma-3-1B
    generator in that ablation).
  - Generator size: gains stop at about 1B parameters; same pattern for 500M–6.2B students (Table 5).
  - Pure synthetic underperforms; mixing with real text restores HellaSwag/PIQA/WinoGrande (Table 7). The
    mix-ratio sweep peaks at 60–80% synthetic (16.8) at this short horizon (Appendix E.2).
  - Contamination 735 docs/B tokens, lower than its FineWeb-Edu source at 841 (Appendix E.4).
- **Weaknesses**: only a 21B-token horizon was tested; the generator can hallucinate (card); DCLM still wins on NLU
  (PIQA, HellaSwag). No head-to-head with Ultra-FineWeb-L3 in the paper. The L3 card claims L3 is ahead of
  FinePhrase late in training, but only as a figure **[unverified numerically]**.
- **Overlap**: content derives from FineWeb-Edu, so it partly overlaps CrowdGPT's FineWeb/L3 *source* documents,
  but the outputs are new text in different formats.

### 2.4 FinePDFs / FinePDFs-Edu

- **Source**: card https://huggingface.co/datasets/HuggingFaceFW/finepdfs, blog
  https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog. No arXiv paper found. The card's arXiv tag points to an
  unrelated paper (2506.18421, a table-reasoning benchmark).
- **Size / licence**: FinePDFs about 3T tokens, 475M docs, 1,733 languages, English 1,190.65B. FinePDFs-Edu 350B+
  tokens across 69 languages; English share **[unverified]**. Both ODC-By. Edu classifier labels came from
  Qwen3-235B-A22B-Instruct-2507. Globally deduplicated.
- **Evidence**: 1.67B models, 36B tokens, 4096 context (blog appendix). Blog: FinePDFs is "on par" with the SmolLM3
  web mix (FineWeb-Edu + DCLM). FW-Edu + DCLM + 0.25 FinePDFs "outperforms even the current SoTA (Nemotron-CC v2)".
  FinePDFs-Edu "without any mixing ... surpasses Nemotron-CC v2". All shown only in figures
  **[unverified numerically]**.
- **Weaknesses**: PII removal only covers emails and IP addresses (blog), and top domains include government and
  legal hosts. A PII pass is advisable given CrowdGPT's "no PII-heavy sources" rule. Odd long-tail content (for
  example horse-racing records from equibase.com survived dedup).
- **Overlap**: none with CrowdGPT's HTML-derived sources.

### 2.5 Ultra-FineWeb (L2) and Ultra-FineWeb-L3 (already in CrowdGPT)

- **Source**: arXiv:2505.05427; L3 card https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3; tiered framework
  arXiv:2602.09003.
- **Size**: Ultra-FineWeb about 1T English + 120B Chinese. A 2026-08-20 update added an L1/L2 set up to
  CC-MAIN-2025-51; size of the new L2 slice **[unverified]**. L3 English: QA 245B + multi-style 164B (MiniCPM5
  tokenizer). Generators: MiniCPM4 and Qwen3-30B-A3B.
- **Licence**: Apache-2.0 on both cards. The Ultra-FineWeb card says to check upstream licences. The **L3 card adds**:
  "Without prior written permission ... strictly prohibited from directly reposting, mirroring, re-hosting ... any
  artifacts of this project". If Vxtzq/CrowdGPT re-hosts L3 text or tokens, a human should check this.
- **Filtering**: fastText classifier (threshold 0.5) trained on verified seeds (FineWeb-Edu-annotated ≥4,
  OH-2.5, ELI5, textbooks, synthetic educational text); seeds chosen with a cheap annealing test on a 1B model.
- **Evidence** (MiniCPM-1.2B, 100B tokens): English avg 45.89 vs FineWeb-Edu 44.56 vs FineWeb 42.28 (Table 4);
  on the DCLM pool, Ultra-DCLM 47.25 vs DCLM-edu 46.59 vs DCLM-pool 45.58 (Table 8). Independent: 13.00 vs DCLM
  13.77 (arXiv:2604.13977 Table 2). Contamination 785 docs/B (ibid. Table 25).
- **Implication**: the FinePhrase result that rephrased data needs a high-quality real mix-in applies directly. OpenBMB
  uses L3 for the decay phase on top of Ultra-FineWeb. CrowdGPT currently pairs L3 with raw FineWeb. Raw FineWeb
  was not tested as a mix-in in arXiv:2604.13977, but low-quality FineWeb as a mix-in was clearly worse (with an
  FWHQ source: FWLQ mix-in 13.61 vs DCLM 14.32; with an FWLQ source: FWLQ mix-in 9.63 vs FWHQ 12.99, Tables 8/9).

### 2.6 Nemotron-CC v1, v2, v2.1

- **v1**: arXiv:2412.02595 (ACL 2025). Justext extraction, ensemble of 3 classifiers into 20 buckets, fewer
  heuristic filters, global fuzzy and exact-substring dedup; synthetic rephrasing with Mistral-NeMo-12B (Wikipedia
  style for low quality; Diverse QA, Distill, Extract Knowledge and Knowledge List for high quality).
  6.3T = 4.4T unique real + 1.9T synthetic; HQ subset 1.1T (0.6T real + 0.5T synthetic) (Table 4).
  Licence: Common Crawl Terms of Use; hosted on Common Crawl's bucket, not on HF.
  - 8B / 1T tokens, 73% from the tested CC set: Nemotron-CC-HQ avg 60.1, MMLU 59.0; DCLM 57.0 / 53.4; full
    Nemotron-CC 57.8 / 53.0; FineWeb-Edu 53.2 / 42.9; FineWeb-Edu-2 53.2 / 42.4 (Table 5).
  - 8B / 15T: +5.0 MMLU over Llama 3.1 8B (70.3 vs 65.3; Table 6).
  - Caveats: evidence is only at 8B. The HQ subset is part Diverse-QA synthetic text whose format resembles
    multiple-choice benchmarks (the paper does not quantify this). Nemotron-HQ-Synth was only on par with DCLM in
    the independent 1.2B/21B test (13.54 vs 13.77, arXiv:2604.13977 Table 2).
- **v2** (card https://huggingface.co/datasets/nvidia/Nemotron-CC-v2): v1 plus eight 2024–2025 snapshots, Qwen3-30B-A3B
  rephrasing. English CC 3,359.8B; English synthetic CC 1,257.3B; Diverse QA 692.9B; translated Diverse QA
  558.2B; total 6,585.8B including math/code/SFT. **v2.1** adds 2.5T, of which 2,122.8B is Medium-High-Quality-Synthetic.
- **Licence (v2/v2.1)**: NVIDIA Data Agreement for Model Training, manual gating; card says use is limited to model
  training, and models trained on it "may be subject to" the Qwen and DeepSeek licence terms. Not compatible with
  re-hosting in an open dataset repo as far as the card text indicates. The agreement text itself was not read
  **[unverified]**.
- **Overlap**: CrowdGPT already uses `nvidia/OpenMathReasoning` and `nvidia/Nemotron-Math-v2`, which have their own
  licences. Those are separate datasets.

### 2.7 Essential-Web v1.0

- **Source**: arXiv:2506.14111; card https://huggingface.co/datasets/EssentialAI/essential-web-v1.0. ODC-By, 24T
  tokens, 23.6B documents, from DCLM-pool plus 12 newer snapshots (through CC-MAIN-2024-38), with snapshot-level
  MinHash.
- **What it is**: every document carries a 12-field taxonomy (FDC subject, Bloom level, document type, reasoning
  depth, technical correctness, education level) labelled by EAI-Distill-0.5b. It is meant to be filtered with SQL.
- **Evidence**: two 2.3B models pretrained for 320B tokens on DCLM-baseline, then annealed for 80B tokens on each
  candidate. MMLU-STEM: STEM-w-DCLM 34.5% ±0.8, STEM 29.1%, DCLM 27.7%, FineWeb-Edu 26.7% (Table 9).
- **Released subsets**: `EssentialAI/eai-taxonomy-stem-w-dclm` (ODC-By), plus math, code and medical variants.
  Size discrepancy: the card says 1,742B tokens; paper Table 8 lists STEM = 1.74T and STEM-w-DCLM = 0.91T
  **[unverified which is correct]**.
- **Weaknesses**: only measured by annealing and only on MMLU-STEM for the STEM subset. Heavy overlap with DCLM
  documents.

### 2.8 FineWeb-HQ / FineWeb2 / FineWeb2-HQ

- arXiv:2502.10361 (NeurIPS 2025 D&B); arXiv:2506.20920 (FineWeb2).
- **FineWeb-HQ** (`epfml/FineWeb-HQ`, ODC-By): top 10% of FineWeb by an XLM-RoBERTa-embedding classifier;
  2,445,786,576 rows (datasets-server). The parquet is 15.8 TB because it ships embeddings. Token count
  **[unverified]**. Card table (1B models): average rank 1.83 vs DCLM 2.39 vs FineWeb-Edu 2.44 vs FineWeb 3.33;
  MMLU 34.0 vs FineWeb-Edu 34.7 vs DCLM 31.6. The token budget for that table was not confirmed **[unverified]**.
- **FineWeb2** (`HuggingFaceFW/fineweb-2`, ODC-By): 20 TB, about 5B documents, 1,000+ languages, no English.
- **FineWeb2-HQ** (`epfml/FineWeb2-HQ`, ODC-By): top 10% for 20 languages; 1B / 70B–119B tokens matched baseline
  MMLU with as little as 15% of the tokens (abstract).
- **Verdict**: FineWeb-HQ is fully contained in FineWeb and adds little over DCLM + FineWeb-Edu. FineWeb2-HQ is the
  right choice if CrowdGPT wants a small multilingual slice; SmolLM3 used FineWeb2 and FineWeb2-HQ for its 12%
  multilingual web share (HF blog https://huggingface.co/blog/smollm3).

### 2.9 DCLM-Edu

- Card https://huggingface.co/datasets/HuggingFaceTB/dclm-edu, CC-BY-4.0. DCLM re-scored with the FineWeb-Edu
  classifier, keeping score ≥2; the card recommends ≥3 for small models. Used for SmolLM2-135M/360M. 360M/200B
  ablation favours DCLM-Edu on MMLU/ARC (figure only). At 1.7B, mid-training gains "weren't consistent". Token
  count **[unverified]**. A reasonable choice for the 1B stage; evidence above 1B is weak.

### 2.10 Zyda-2

- arXiv:2411.06068; card https://huggingface.co/datasets/Zyphra/Zyda-2; ODC-By; 5,070.2B gpt-neox tokens:
  `dclm-crossdeduped` 3,348.9B, `fwe3` (FineWeb-Edu ≥3) 1,319.2B, `zyda-crossdeduped-filtered` 163.6B,
  `dolma_cc-crossdeduped-filtered` 238.4B. Recommended token weights: DCLM 4.0, FWE3 4.0, Zyda 0.16, Dolma-CC 0.24.
- Evidence: the card claims it beats each constituent (figure only). The paper's 1.4B/50B ablation shows that
  keeping the top 10–20% by the NeMo quality classifier helps Zyda-1 and Dolma-CC, while re-filtering DCLM gave
  "negligible gain".
- **Practical value**: a ready-made DCLM + FineWeb-Edu pair with cross-duplicates removed.

### 2.11 TxT360

- Card https://huggingface.co/datasets/LLM360/TxT360, ODC-By. Global dedup across 99 CC snapshots and 14 curated
  sources, about 5T tokens; v1.1 adds `TxT360_BestOfWeb` (ProX-filtered; size **[unverified]**) and `TxT360_QA`
  (Mistral-7B QA appended). Evidence: one 8x8B MoE, 1.5T-token comparison against FineWeb, loss and eval curves
  only. No small-model, per-token evidence vs DCLM/FineWeb-Edu was found.

### 2.12 GneissWeb

- arXiv:2502.14907; card https://huggingface.co/datasets/ibm-granite/GneissWeb (Apache-2.0). Released as a recipe
  plus a bloom filter over FineWeb v1.1.0 (about 10T), not as new text. 1.4B / 350B: +2.73 pp average over FineWeb
  on 11 tasks (+1.75 pp on 20). The card also lists 7B / 350B ablation models for FineWeb-Edu, but no headline
  delta vs FineWeb-Edu was found **[unverified]**.

### 2.13 Dolma 3 pool (OLMo 3)

- arXiv:2512.13961; card https://huggingface.co/datasets/allenai/dolma3_pool; ODC-By. 9.31T tokens: Common Crawl
  8.14T, olmOCR science PDFs 972B, plus code, math and Wikipedia pointers.
- Web data is split into 24 WebOrganizer topics × 20 quality vigintiles. The quality classifier is DCLM-style
  fastText with OH-2.5, ELI5, UltraChat and WildChat positives.
- Finding relevant to CrowdGPT: in data-constrained settings, quality-aware upsampling (several copies of the top
  ~5%, single copies of the rest) beat flat top-quartile filtering at every repetition factor tested (Table 39;
  numbers not extracted **[unverified]**).
- Use: pull top-vigintile documents for under-represented topics (science, technology). Shares the DCLM-pool
  extraction, so it overlaps DCLM.

### 2.14 Licence-first corpora: Common Pile v0.1, MixtureVitae

- **Common Pile** (arXiv:2506.05209): 8 TB of public-domain and openly licensed text from 30 sources. Its web part,
  CCCC, keeps only domains with all content under CC licences. 1.7B / 28B tokens: beats OLC, Common Corpus and
  KL3M; FineWeb is best on most benchmarks; Comma is best on MMLU/ARC but "significantly worse" on HellaSwag, PIQA
  and CSQA. Comma v0.1 7B at 1T/2T tokens is comparable to Llama 1/2 7B.
- **MixtureVitae** (arXiv:2509.25531): 422B tokens, but only about 34B is web; about 178B is synthetic
  instruction/reasoning data. 1.7B / 300B: surpasses FineWeb-Edu and approaches DCLM late in training.
- Verdict: the right sources if the project wants a "clean licence" tier. Neither is the per-token leader for
  general web text.

### 2.15 Excluded or unavailable

- **Nemotron-ClimbMix** (`nvidia/Nemotron-ClimbMix`): CC-BY-NC-4.0. Strong 1B result (beats Llama-3.2-1B by 2.0%
  after 400B tokens, arXiv:2504.13161), but non-commercial.
- **REWIRE** (`facebook/recycling_the_web`): CC-BY-NC, and the text is Llama 3.3 output; the card requires "Llama"
  at the start of derived model names.
- **BeyondWeb** (arXiv:2508.10975): DatologyAI did not release the data; no HF dataset found.
- **CuraWeb** (arXiv:2607.22662, 2T tokens, 3B / 1T, +1.8% avg over FineWeb, Dolma3, DCLM): no public dataset found
  on HF **[unverified]**.

---

## 3. Ranked shortlist for CrowdGPT

CrowdGPT's measured gaps that web data can address are wikitext (+1.80 nats), SciQ (+1.50) and fineweb-edu
(+1.04). Code and math gaps belong to other reports. The current web side is Ultra-FineWeb-L3 (synthetic) plus raw
FineWeb.

1. **DCLM-Baseline 1.0** (CC-BY-4.0). Use it to replace most of the raw FineWeb share.
   - Best real-web corpus in the only independent same-setup comparison found (1.2B/21B: 13.77 vs Ultra-FineWeb
     13.00 vs FineWeb-Edu-HQ 11.82).
   - Best mix-in for rephrased data (Table 8).
   - Strong on the commonsense tasks where FineWeb-Edu is weak.
   - Lowest measured n-gram contamination of the real-web sets.
   - To avoid paying twice for FineWeb-Edu duplicates, take it from Zyda-2's `dclm_crossdeduped` split.
2. **FineWeb-Edu, int_score ≥3** (ODC-By), paired roughly evenly with DCLM.
   - Strongest evidence for knowledge and science benchmarks per token at short horizons (MMLU 33.6% at 38B
     tokens).
   - Better fit for Wikipedia and academic text (Paloma).
   - SmolLM2/3 and Zyda-2 (weights 4.0 : 4.0) both pair it with DCLM.
   - Dedup against CrowdGPT's FineWeb by `id`.
   - Ablating ≥4 for the 1B stage is worthwhile; no primary source settles ≥3 vs ≥4 at this budget.
3. **FinePhrase** (ODC-By), mainly the `math` and `table` configs, as a second synthetic source next to
   Ultra-FineWeb-L3.
   - Best measured per-token gains of any open synthetic web set (+3.41 macro over DCLM at 1.2B/21B).
   - Generator is Apache-2.0, so it is licence-cleaner than Qwen-, DeepSeek- or Llama-generated sets.
   - The paper's mixing results argue for keeping real text at ≥20–50% of the web stream.
   - Overlaps L3 in role, so it adds format diversity rather than new facts.
4. **FinePDFs-Edu, English config** (ODC-By), as a new non-HTML source of long science, technical and legal
   documents.
   - No overlap with anything CrowdGPT has.
   - Reported to beat Nemotron-CC v2 at 1.67B/36B, figure only.
   - Requires a PII pass.
   - **Alternative**, if SciQ is the priority: `EssentialAI/eai-taxonomy-stem-w-dclm` (ODC-By; MMLU-STEM 34.5 vs
     DCLM 27.7 in a 2.3B anneal).

**Conditional (needs human licence review)**: Nemotron-CC v1 HQ subset has the strongest controlled small-horizon
result (8B/1T: MMLU 59.0 vs DCLM 53.4). It is under Common Crawl ToU and is not on HF. If re-hosting under those terms
is acceptable, it is a direct competitor for slot 1 or 2. Nemotron-CC v2/v2.1 should stay out unless the NVIDIA
agreement is cleared.

**Practical cautions**

- **Eval leakage in CrowdGPT's own metric.** Training on FineWeb-Edu will lower the "fineweb-edu" validation loss
  partly because it is in-distribution. Make sure the eval documents are excluded by `id` from every
  FineWeb-derived shard (FineWeb, FineWeb-Edu, FinePhrase, Ultra-FineWeb/L3). Treat a lower loss there as a
  distribution signal, not proof of capability.
- **FineWeb family overlap.** FineWeb, FineWeb-Edu, FineWeb-HQ, GneissWeb, Ultra-FineWeb, Ultra-FineWeb-L3 and
  FinePhrase all draw from the FineWeb document pool. DCLM, Essential-Web and Dolma 3 all draw from DCLM-pool
  extraction. Budget tokens per *pool*, not per dataset name.
- **Pick the mix with small proxies, not intuition.** DataDecide (arXiv:2504.11393) found that rankings at 150M
  parameters predict 1B rankings in about 80% of pairwise comparisons. Continuous-likelihood metrics on MMLU, ARC
  and HellaSwag were more than 80% predictable at 0.01% of the compute. This suits CrowdGPT's scarce compute.
- **Optimizer interaction.** No primary source was found on how data choice interacts with zeroth-order or
  DiLoCo-style training. All evidence above uses AdamW-class optimizers **[unverified for CrowdGPT's optimizer]**.
- **Licence recording.** Record per shard: dataset licence (CC-BY-4.0 / ODC-By / Apache-2.0), Common Crawl ToU where
  applicable, generator-model terms for synthetic text (SmolLM2: Apache-2.0; L3: MiniCPM4 + Qwen3), and the L3
  redistribution clause.
