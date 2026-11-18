# Paper index (round 1 survey)

Local only (git-ignored). Each paper: `<id>.pdf` (full paper from arxiv.org/pdf/<id>),
`<id>.txt` (full text extracted with pypdf), `notes/<id>.md` (method + MVP recipe).

Selection: 2026 papers only, biased to the newest (6 of 12 from Sep-Nov 2026), split
between ZO-estimator complements that attack DUST's variance/cost, forward-only or
local-learning replacements, and backprop-based low-communication / trustless
replacements that set the bar DUST has to beat for a volunteer crowd.

| # | arXiv | Date | Title | Authors | Role | Why it matters for Crowd |
|---|-------|------|-------|---------|------|--------------------------|
| 1 | [2611.08712](https://arxiv.org/abs/2611.08712) | 2026-11-05 | An Educated Guess: Deriving Statistically Aligned Gradient Estimates for Zeroth-Order Optimisation | Huebotter, Thill, van Gerven, Ahmad | Complement | Better-aligned ZO estimator family (SZO-κ): same forwards, higher cosine - attacks DUST's core weakness (variance) |
| 2 | [2611.16317](https://arxiv.org/abs/2611.16317) | 2026-11-12 | Wanda-Guided Sparse Perturbations for Efficient Zeroth-Order Optimization (WIS-MeZO) | Wen, Song, Wang, Luo, Zeng, Li et al. | Complement | Perturb only salient weights: fewer effective dimensions -> lower variance per draw |
| 3 | [2611.04797](https://arxiv.org/abs/2611.04797) | 2026-11-03 | Training Specialist LLMs on Your Phone: Forward-Only Optimization via Block-Spectral Coordinates (MobileSpec) | Tan, Zhao, Liu, Su, Zhang, Zhang | Complement / replacement | Forward-only training on weak devices; perturb in spectral coordinates (tiny search space) |
| 4 | [2609.10980](https://arxiv.org/abs/2609.10980) | 2026-09-10 | EGGROLL, Unrolled: Understanding and Improving Low-Rank Evolution Strategies at Scale | Kaya, Hashemi | Complement | Low-rank ES at scale: cheap batched perturbations, seed-only communication |
| 5 | [2609.37899](https://arxiv.org/abs/2609.37899) | 2026-09-29 | Scaling Zero-Order Pretraining through Model Sharding (SOMA) | Chaubard, Kochenderfer, Ré | Replacement | ZO pretraining that shards the model into independently trained experts - no gradient exchange |
| 6 | [2602.17155](https://arxiv.org/abs/2602.17155) | 2026-02-19 | Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization | Lang, Wang, Zhang, Hong, Zhang, Yin et al. | Complement | Subspace ZO + Muon-style orthogonalised updates: conditioning fix for noisy ZO directions |
| 7 | [2604.14669](https://arxiv.org/abs/2604.14669) | 2026-04-16 | Zeroth-Order Optimization at the Edge of Stability | Song, Zhang, Li, He, Muehlebach, Oh | Complement | Stability limit for ZO step sizes - explains and prevents the divergence seen in our DUST runs |
| 8 | [2608.03020](https://arxiv.org/abs/2608.03020) | 2026-08-04 | LoCA: Forward-Only LLM Tuning after One-Shot Calibration with Local Credit Assignment | Xia, Liu, Zhang, Wang, Shen, Zhu | Complement | Forward-only updates with block-local credit from a one-shot calibration: removes global-loss variance |
| 9 | [2609.35440](https://arxiv.org/abs/2609.35440) | 2026-09-28 | SOLO: Pretraining Billion-Parameter Language Models with Shared-Output Local Learning | Yin, Wang, Pan, Li | Replacement | Local learning with a shared output head: block-parallel pretraining, no end-to-end backprop |
| 10 | [2603.08163](https://arxiv.org/abs/2603.08163) | 2026-03-09 | Covenant-72B: Pre-Training a 72B LLM with Trustless Peers Over-the-Internet | Lidin, Sarfi, Miahi, Anthony, Chauhan, Pappas et al. | Replacement | Permissionless internet pretraining at 72B: SparseLoCo compression + Gauntlet peer scoring |
| 11 | [2604.21428](https://arxiv.org/abs/2604.21428) | 2026-04-23 | Decoupled DiLoCo for Resilient Distributed Pre-training | Douillard, Rush, Donchev, Charles, Fallen, Dubey et al. | Replacement | Async learners + quorum aggregation: stragglers and dropouts, the normal state of a volunteer fleet |
| 12 | [2609.39357](https://arxiv.org/abs/2609.39357) | 2026-09-30 | Robustifying Asynchronous SGD via Soft Throttling | Otsuka, Meyer, Takezawa, Yamada, Koloskova | Complement | Byzantine-robust async aggregation: untrusted peers without synchrony |

Considered and not selected (overlap or not MVP-able on one Mac): HeLoCo 2606.00271
(overlaps #11), SparseLoCo 2508.15706 (inside #10), Greedy Local Learning 2610.04867
(overlaps #9), Infer-ZO 2605.28760 (needs an inference engine), SENTINEL 2603.03592,
verification papers 2609.33915 / 2611.09889 / 2606.21045, Factored Gossip DiLoCo
2606.22768, model-merging DiLoCo 2607.03011, Subspace Networks 2506.01260, MAPL 2606.05484.
