# Zeroth-order / evolutionary / perturbation-training lineage — Crowd-v1.5

Survey date: 2026-11-16. Scope: what descends from, competes with, or could improve DUST, with emphasis on
variance reduction and on evidence at 1B scale. Every entry below was read from an abstract page, PDF, or
publisher page I actually fetched. Anything I could not open is in **§11 Unverified**.

Sibling doc: `open-source-ecosystem.md` (same directory) covers repos/licences. This doc covers papers only.

---

## 0. Why this matters for v1.5

v1.5 is a bet that a token-axis virtual population (DUST) is a cheaper variance-reduction device than a
weight-axis population (ES). That bet is really a claim about **estimator variance per forward pass**, and the
variance-reduction literature is the only place where that claim can be checked before we spend GPU-months.
Three things follow. (a) DUST's own numbers stop at 243M params / 20M tokens, so the 1B step is extrapolation
and we should buy every cheap variance win available. (b) Almost all ZO-at-scale evidence is *fine-tuning*
evidence, where the model is already at a good point; pretraining-from-scratch is a different regime and only
DUST, DeepZero, and the ZO-pretraining line address it. (c) Several recent results are directly negative or
directly free — ZO-Adam is worthless, per-layer noise scales matter, control variates work — and those are
cheaper to exploit than any architectural change.

---

## 1. Foundations: where the activation-vs-weight argument actually comes from

### 1.1 Werfel, Xie & Seung — node vs weight perturbation, quantified (2003/2005)

- **Authors/venue:** Justin Werfel, Xiaohui Xie, H. Sebastian Seung. NIPS 2003; journal version *Neural Computation* 17(12):2699–2718, 2005.
- **Link:** doi:10.1162/089976605774320539 · PMID 16212768.
- **Mechanism:** analyses three online learners on a linear perceptron — direct gradient descent, node perturbation (noise on activations), weight perturbation (noise on weights) — and derives learning speed as the exponential decay rate of the learning curve, with the update-size scalar tuned to maximise it.
- **Measured claim (analytic, no experiments to quote):** node perturbation is slower than direct gradient descent by a factor equal to **the number of output units**; weight perturbation is slower by **an additional factor equal to the number of input units**. So for a 512→512 layer the sample-cost ratio NP : WP is 1 : 512. Parallel (batched) perturbation beats sequential by a factor independent of network size.
- **Why it matters here:** this is the cleanest theoretical statement of why DUST perturbs activations. It also says something uncomfortable: the advantage is *only* one factor of `d_in`, i.e. it is a constant, not an asymptotic win. DUST's extra claim (tokens as population) is a second, separate multiplier that Werfel does not model.
- **Implementable in a day:** nothing to build. Use it as the sanity model when choosing which layers get activation perturbation vs (for the 2L residual scalars) weight-space ES — the ratio predicts where weight-space is affordable.

### 1.2 Hiratani, Mehta, Lillicrap & Latham — node perturbation scales in width but is unstable (NeurIPS 2022)

- **Link:** NeurIPS 2022 proceedings, `.../2022/hash/cf38eb1549024cce4b3d2c1bb87a6c27-Abstract-Conference.html`.
- **Mechanism:** analytic study in deep linear networks plus simulations in deep nonlinear ones, asking how node perturbation's stability and convergence depend on network size.
- **Measured claim:** in deep linear nets with one hidden layer, **both learning time and performance depend very weakly on hidden layer size** — the width-scaling fear is largely absent. But: "unlike stochastic gradient descent, when there is model mismatch between the student and teacher networks, node perturbation is **always unstable**", triggered by weight diffusion to very large weights.
- **Why it matters here:** the good half supports DUST's empirical "bigger models are more population-efficient". The bad half is a live risk for a 1B volunteer run: pretraining is the maximal model-mismatch regime, and instability manifests as weight-norm growth, which a coordinator averaging heterogeneous volunteer updates will see late and confusingly.
- **Implementable in a day:** log per-layer weight L2 norm and update-norm ratio every round; alarm on monotone growth over a window. Costs one pass over the state dict; catches the failure mode Hiratani predicts.

### 1.3 Züge et al. — weight perturbation often matches or beats node perturbation (PRX 2023)

- **Link:** *Phys. Rev. X* 13, 021006 (2023), doi:10.1103/PhysRevX.13.021006. Title (from ADS): "Weight Perturbation Often Performs Similarly or Better than Node Perturbation…". Full author list not verified — see §11.
- **Mechanism:** compares WP and NP learning rules on temporally extended / online tasks where NP has been the standard biological-plausibility benchmark.
- **Claim:** NP's assumed superiority over WP does not hold in general; WP is often similar or better. (Direction verified from the ADS abstract snippet; magnitudes not.)
- **Why it matters here:** a direct counterweight to §1.1. It does not refute the dimensional argument, but it says the dimensional argument is not the whole story in temporal credit assignment — which is exactly the regime (`reverse_discounted_cumsum`, γ≈0.98) where v1.5 lives.
- **Implementable in a day:** an A/B you can run on one 4060 Ti: same token budget, same population, DUST-style activation noise vs weight-space ES on the MLP blocks only. If WP is within noise at small scale, the whole "activations are the only option" framing deserves a footnote.

### 1.4 Nesterov & Spokoiny — the dimension penalty, and why it is the wrong penalty (2017)

- **Link:** *Foundations of Computational Mathematics* 17(2):527–566, doi:10.1007/s10208-015-9296-2.
- **Mechanism:** Gaussian smoothing of a convex objective; each iteration draws a random direction, evaluates two function values at ±σ, and forms the two-point estimator.
- **Claim:** complexity bounds carry a factor linear in dimension `d`, which is why ZO is "known not to scale" — and is the conventional wisdom DUST argues against empirically.
- **Implementable in a day:** none; it is the null hypothesis to state in the writeup.

### 1.5 Yue et al. — replacing `d` with an effective-dimension factor (2023)

- **Link:** arXiv:2307.05753, "Zeroth-order Optimization with Weak Dimension Dependency".
- **Mechanism:** introduces `ED_α = sup_x Σ_i σ_i^α(∇²f(x))` (sum of α-powers of Hessian singular values) as the real measure of dimensionality, then re-derives ZO complexities in terms of it rather than `d`.
- **Claim:** complexities "significantly reduced" when measured in `ED_α`; the Nesterov–Spokoiny algorithm on quadratics gets a rate in `ED_α`, not `d`.
- **Why it matters here:** this is the theory that rescues ZO-at-scale *if and only if* the loss Hessian is effectively low-rank near the current point. It predicts that DUST's population requirement should track the **effective rank of the gradient/Hessian**, not parameter count — testable, and a much better prior than "1B params needs 1B samples".
- **Implementable in a day:** estimate effective rank cheaply from the per-layer update history: take the last ~32 rounds of a layer's estimated gradient, stack them, and report the scree/participation ratio. If effective rank ≪ d, populations can stay small and §1.4 is not our problem.

---

## 2. Forward-only training: what is actually usable for a sequence model

### 2.1 Baydin et al. — forward gradient (2022)

- **Link:** arXiv:2202.08587, "Gradients without Backpropagation".
- **Mechanism:** pick a random tangent `v`, compute the directional derivative `∇f·v` exactly in one forward-mode pass, return `(∇f·v)v` as an unbiased gradient estimate. One forward run, no reverse pass.
- **Claim:** unbiased, single forward run; "training up to twice as fast in some cases" on the problems tested.
- **Why it matters here:** this is the ancestor of DUST's estimator with `v` drawn per *model* rather than per *token*. The variance is `O(d)` — which is precisely the problem DUST solves by making each token its own tangent.
- **Implementable in a day:** nothing; we already have the better version.

### 2.2 Ren, Kornblith, Liao & Hinton — local losses + activation perturbation (ICLR 2023)

- **Link:** arXiv:2210.03310.
- **Mechanism:** two changes to forward gradient. (i) Perturb **activations rather than weights**, which "substantially reduce[s] the variance" because the perturbation lives in `d_out` not `d_out × d_in`. (ii) Many **local greedy losses**, each with few parameters, plus a `LocalMixer` architecture suited to local learning.
- **Claim:** matches backprop on MNIST and CIFAR-10; "significantly outperforms previously proposed backprop-free algorithms on ImageNet". No exact ImageNet gap given in the abstract.
- **Why it matters here:** this is DUST's direct methodological parent, and it is the strongest published evidence that activation-space perturbation is the right axis. Note what it needed to get there: **local losses and a bespoke architecture**. DUST's claim to novelty is getting the same benefit on a stock transformer by using the token axis instead of local losses.
- **Implementable in a day:** the cheap half — a per-block auxiliary loss (predict the block's own input from its output, or a per-block LM head on a small vocab slab) gives each block a dense local signal instead of relying entirely on the reverse-discounted token reward. This is a real architectural change, so realistically 2–3 days, and it invalidates nothing in `dust.py`.

### 2.3 Fournier et al. — can forward gradient match backprop? (ICML 2023)

- **Link:** PMLR 202:10249–10264.
- **Mechanism:** systematic study of (gradient target × gradient guess) combinations; instead of isotropic random guess vectors, **bias the guess toward directions from small local auxiliary networks** (local-loss gradients, feedback alignment, etc.).
- **Claim:** "using gradients obtained from a local loss as a candidate direction **drastically improves** on random noise in Forward Gradient methods."
- **Why it matters here:** a variance-reduction recipe that does not need more forwards — it needs a *better guess distribution*. In DUST terms: the noise `a_t` need not be isotropic Gaussian; it could be shaped toward a cheap surrogate direction (e.g. the direction implied by the previous round's estimate, or by a local linear model of the block).
- **Implementable in a day:** none cleanly — this is the "shaped noise" idea, and shaping noise in a shared-seed protocol means every volunteer must derive the same shaping, which is fine if it comes from the previous round's published update. Call it a 2-day experiment.

### 2.4 Flügel et al. — multi-tangent forward gradients with orthogonal projection (IJCNN 2025)

- **Link:** arXiv:2410.17764.
- **Mechanism:** instead of one tangent per step, compute `k` directional derivatives and combine them via **orthogonal projection**; the Gram matrix of tangents is inverted so each contributes only its new information.
- **Claim:** "increasing the number of tangents improves both approximation quality and optimization performance across various tasks."
- **Why it matters here:** the multi-tangent combination rule is exactly what a population estimator needs, and orthogonal projection across tangents is a variance-reduction device we are not using (our `K` draws are iid and antithetic, not orthogonalised).
- **Implementable in a day:** yes — orthogonalise the `K` noise draws per layer (QR or modified Gram–Schmidt on the `K × d_out` block, per token chunk), keeping the same seeds. Cost is `O(K² d)` per layer, negligible for K≤64. See §7.5 for the caveat that this is a variance win only in specific senses.

### 2.5 Dalm et al. — node perturbation as directional derivative + input decorrelation (2023)

- **Link:** arXiv:2310.00965, "Node Perturbation Can Effectively Train Multi-Layer Neural Networks".
- **Mechanism:** re-derives NP as a directional-derivative estimator and adds **input decorrelation at every layer**; also handles the case where the noise process itself is inaccessible.
- **Claim:** "large improvements in parameter convergence and much higher performance on the test data, **approaching that of BP**" on multi-layer nets; enables learning in noisy systems.
- **Why it matters here:** input decorrelation is a *free* variance reduction in the outer-product step `Σ_t ĝ_t x_tᵀ`: if the layer inputs `x_t` are highly correlated across tokens (they are, in natural text), the outer product is ill-conditioned. Whitening or even just mean-centering `x_t` before the outer product changes the estimator's conditioning without changing its expectation much.
- **Implementable in a day:** yes, in `dust.py`'s outer-product accumulation — subtract the per-batch mean of `x_t` (per feature) before the outer product, and optionally scale by the inverse per-feature std. One-line change, one A/B on a 10M-token run.

### 2.6 Hinton — Forward-Forward (2022), and where it stands in 2025

- **Link:** arXiv:2212.13345. Follow-up with numbers: Self-Contrastive Forward-Forward, *Nature Communications* 2025, doi:10.1038/s41467-025-61037-0.
- **Mechanism:** two forward passes, one on real data and one on corrupted data; each layer locally maximises a "goodness" contrast, so no backward pass and no stored activations across layers.
- **Claim:** FF supervised CIFAR-10 rose "from below 60% to **84.7%**" once random direct feedback connections were added (SCFF intro, citing prior work); SCFF extends FF to unsupervised tasks on MNIST/CIFAR-10/STL-10/Tiny ImageNet, and to a spoken-digit sequential task.
- **Why it matters here:** FF remains a **vision-benchmark method**. I found no FF result at LM scale, let alone 1B. It also needs positive/negative pairs, which for next-token prediction means inventing a corruption process — a research project, not a config flag.
- **Implementable in a day:** no. Explicitly out of scope for v1.5; note it as the main "other" backprop-free family so we are not accused of ignoring it.

### 2.7 Predictive coding (PC-ALM, 2026) — the local-dynamics alternative

- **Link:** arXiv:2605.31022, Seely & Gould, "Augmented Lagrangian predictive coding" (cited by DUST as the local-learning alternative).
- **Mechanism:** credit propagates through local predictive-coding dynamics with Lagrange multipliers; uses local derivatives only.
- **Claim:** evaluated on image classification (per DUST's related-work description); no LM-scale claim.
- **Implementable in a day:** no.

---

## 3. ZO at LLM scale: the memory argument, and the evidence about parameter count

### 3.1 MeZO (NeurIPS 2023) — the reference point

- **Authors/link:** Malladi, Gao, Nichani, Damian, Lee, Chen, Arora. arXiv:2305.17333.
- **Mechanism:** classical ZO-SGD made memory-free: perturb parameters **in place** with a seeded `z`, run two forward passes (or one with a stored seed), form the scalar projection, restore. No gradient buffer, no optimizer state (momentum is reconstructed from the seed history rather than stored).
- **Claim:** 30B model fine-tuned on **one A100-80GB** where backprop fits only 2.7B; up to **66B**; "up to 12× memory reduction and up to 2× GPU-hour reduction"; beats in-context learning and linear probing; compatible with LoRA/prefix.
- **Why it matters here:** this is the strongest existing evidence that forward-only optimisation is *usable* at ≥1B. It is also fine-tuning-only evidence, and the 2× GPU-hour win is against a memory-bound baseline, not against a well-tuned SGD.
- **Implementable in a day:** the seed-reconstruction trick for momentum is already conceptually in `DustOptimizer`. Nothing new.

### 3.2 Zhang et al. — the ZO benchmark that says which knobs matter (ICML 2024)

- **Link:** arXiv:2402.11592.
- **Mechanism:** first broad benchmark of ZO fine-tuning: 5 model families (RoBERTa, OPT, LLaMA, Vicuna, Mistral), 3 task complexities, 5 tuning schemes; adds block-wise descent, hybrid ZO/FO training, and gradient sparsity.
- **Claim:** "task alignment" dominates; the forward-gradient method matters; there is a real complexity-vs-performance tradeoff; block-wise and sparse variants help. No single headline number.
- **Why it matters here:** it is the paper to cite when someone asks whether a fancier ZO estimator is worth it. Its answer: estimator sophistication buys less than prompt/task alignment and than *which parameters you perturb*.
- **Implementable in a day:** block-wise descent maps directly onto "perturb a subset of blocks per round" — a scheduling change in `coordinator.py`, one day.

### 3.3 MeZO-SVRG (ICML 2024) — the control-variate result, at scale

- **Authors/link:** Gautam, Park, Zhou, Raman, Ha. arXiv:2404.08080, "Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models".
- **Mechanism:** SVRG-style variance reduction for ZO: keep a **snapshot** point and a snapshot gradient estimate; each step's estimator is `(current probe estimate) − (snapshot probe estimate at the same noise) + (snapshot gradient)`. The shared noise makes the two probe terms correlated, so their difference has small variance. Implemented with in-place ops to keep MeZO's memory profile.
- **Claim:** up to **+20% test accuracy** over MeZO on GLUE, full- and partial-parameter; often reaches MeZO's peak accuracy with **2× fewer GPU-hours**; **2× less memory than first-order SGD** for autoregressive models; removes the need for task-specific prompts.
- **Why it matters here:** this is the single most transferable variance-reduction idea for v1.5, and the one with scale evidence. The catch for us is the snapshot: it is weight-shaped. At 1B that is 2 GB in bf16 — affordable on the 30 GB APU, not on an 8 GB card, and it dents the "no extra state" selling point. A compressed snapshot (the same 1-bit codec we already ship, or a rank-`r` sketch) would keep the promise.
- **Implementable in a day:** the estimator change is ~30 lines in `dust.py` *if* we accept a full-precision snapshot; with a compressed snapshot it is 2 days plus a codec round-trip test. Highest expected payoff of anything in this document — see §8.

### 3.4 Sparse MeZO (NeurIPS 2025) — perturb fewer, better-chosen parameters

- **Link:** arXiv:2402.15751. Liu, Zhu, Gong, Cheng, Hsieh, You.
- **Mechanism:** observation — ZO estimation error hurts more when added to **large weights** than small ones; therefore select a subset of parameters (by magnitude/structure) and apply ZO only there, with a memory-optimised sparse mask.
- **Claim:** **+9% absolute accuracy and 3.5× speedup over MeZO on RTE**; fine-tunes LLaMA-30B on a single A100 at inference-level memory.
- **Why it matters here:** the "error hurts more on large weights" observation is directly relevant to a pretraining run where weight norms grow — it predicts the estimator degrades as training proceeds, which is a *different* failure mode from interference and would look like a plateau.
- **Implementable in a day:** yes — restrict each round's perturbation mask to the top-`p` fraction of blocks by ||W|| (per layer), computed on the coordinator and broadcast. One day, and it composes with everything else.

### 3.5 HiZOO (ICLR 2025) — diagonal Hessian preconditioning

- **Link:** arXiv:2402.15173. Zhao et al.
- **Mechanism:** estimate a diagonal Hessian from the same finite differences and use it to precondition the ZO step (Hessian-informed ZO), with a low-memory variant (HiZOO-L) at ~10% of the Hessian memory.
- **Claim:** **8× speedup and +1.55% absolute accuracy** vs MeZO.
- **Why it matters here:** preconditioning is an alternative to variance reduction — it does not make the estimate less noisy, it makes the step better conditioned. But storing a diagonal costs `d` floats (4 GB at 1B fp32), which breaks the memory story; the low-memory variant is the only version worth considering, and even then it is a second buffer.
- **Implementable in a day:** no (violates the no-state constraint); listed so we can say we considered it.

### 3.6 LOZO (ICLR 2025) — low-rank ZO estimator

- **Link:** arXiv:2410.07698. Chen, Zhang, Cao, Yuan, Wen.
- **Mechanism:** constrain the ZO probe to a **low-rank** factorisation (perturb `A Bᵀ` with rank `r`), estimate the gradient in that subspace, and prove convergence by framing it as subspace optimisation; low rank makes momentum affordable.
- **Claim:** outperforms existing ZO methods and "closely approach[es] the performance of FO algorithms" across model sizes; negligible extra memory for momentum.
- **Why it matters here:** if the gradient is effectively low-rank (§1.5), then a rank-`r` probe has variance `∝ r` instead of `∝ d`. **This is the biggest theoretical lever in the document** — and it is the weight-space analogue of what ZO-Act (§3.8) does with activations.
- **Implementable in a day:** in weight space, a rank-`r` DUST means perturbing each layer's output along `r` random directions shared across tokens rather than `d_out` independent ones. That collapses the token-axis population advantage, so it is *not* obviously a win here. Read it as theory support, not a patch.

### 3.7 On Adaptivity in ZO (2026) — ZO-Adam does not pay

- **Link:** arXiv:2605.03869. Dbouk, Gazagnadou, Reisser, Louizos.
- **Mechanism:** tests adaptive ZO (ZO-Adam and relatives) against well-tuned ZO-SGD; argues that in high dimensions ZO gradients "lack coordinate-wise heterogeneity", so per-coordinate second moments carry no information; proposes **MEAZO**, a single scalar global step-size adaptation.
- **Claim:** "adaptive ZO methods such as ZO-Adam offer **no convergence advantage** over well-tuned ZO-SGD, while incurring significant memory overhead"; MEAZO matches ZO-Adam at ZO-SGD's memory.
- **Why it matters here:** this is a negative result that protects our design. It also means the DUST-under-Adam result in the paper (Adam helps DUST and backprop, not EGGROLL) should be read cautiously — Adam's win there may be a step-size win, which MEAZO gets for one scalar.
- **Implementable in a day:** yes — one scalar `s_t` adapted from the ratio of consecutive estimate norms, applied as a global lr multiplier. Half a day, removes the temptation to add per-parameter state.

### 3.8 ZO-Act (2026) — activation-derived low-rank subspace, one-shot

- **Link:** arXiv:2607.01125. Dong, Xu, Wang, Li, Yin, Yang.
- **Mechanism:** for each linear layer, compute a small **activation basis once at initialisation**; freeze it; optimise only lightweight coefficient matrices using forward-only loss evaluations. Formally ZO over a restricted coefficient space, with an explicit bias/variance split: perturbing low-dimensional coefficients reduces both the variance-dependent convergence term and the finite-difference error, at the cost of controlled subspace bias.
- **Claim:** consistent gains over strong ZO baselines on Llama-3-8B, OPT-13B and INT4 Llama-3-8B across understanding, QA, and commonsense tasks. (Headline percentages not in the abstract.)
- **Why it matters here:** we are the one setting where the activation basis is *free* — the volunteer already has activations, and DUST's whole trick is that they are available. Perturbing in a rank-`r` activation subspace per layer reduces the effective perturbation dimension from `d_out` to `r`, which is exactly the quantity that sets variance.
- **Implementable in a day:** no, but a 3-day spike is realistic: compute `U ∈ R^{d_out × r}` from a batch of activations at round 0 (randomised SVD), publish `U` once per layer as part of the round manifest, and have clients perturb `U c` with `c ∈ R^r`. The protocol change is real (extra per-layer tensors), but they are tiny (`d_out × r` ≈ 2048×16).

### 3.9 P-GAP (2025) — projected gradient-aligned perturbations, plus a variance-vs-model-size theory

- **Link:** arXiv:2510.18228. Mi, Tan, Zhang, Xu, Yuan, Huang.
- **Mechanism:** estimate a low-dimensional gradient space, then align perturbations with the **projected gradient direction** inside that space; fewer perturbed parameters and lower variance.
- **Claim:** up to **+6%** accuracy on classification and **+12%** on generation tasks, with **~81% fewer training iterations** and **~70% fewer GPU-hours**. Notably, the paper states a theoretical result on "how the variance of standard ZO estimators **scales with the model size**" and how subspace restriction mitigates it.
- **Why it matters here:** that variance-vs-model-size statement is the theoretical question v1.5 most needs answered, and this is the paper that claims to answer it. Worth reading in full before the 1B run; I only verified the abstract.
- **Implementable in a day:** no; but its "align the probe with a projected gradient" is the same family as §2.3 and could reuse the previous round's published update as the surrogate direction.

---

## 4. Weight-space ES: the competitor we are defined against

### 4.1 Salimans et al. (2017) — the template

- **Link:** arXiv:1703.03864.
- **Mechanism:** ES as black-box RL: sample population of weight perturbations, evaluate fitness, update `θ ← θ + α/(Nσ) Σ F_i ε_i`. Introduces **antithetic sampling** (evaluate `+ε` and `−ε` together) and **rank-based fitness shaping**; common random numbers let workers share seeds so only scalars cross the wire.
- **Claim:** solves 3D humanoid walking in 10 minutes; competitive on most Atari games after one hour; scales to >1000 parallel workers.
- **Why it matters here:** every one of these tricks is available to us and two are already in `dust.py` (antithetic). **Rank-based fitness shaping is not**, and it is free.
- **Implementable in a day:** yes — replace raw `c_s` with its within-draw rank or z-score before forming the reward. One function.

### 4.2 EGGROLL (2025) — low-rank perturbations for GPU efficiency

- **Link:** arXiv:2511.16652. Sarkar et al. (20 authors).
- **Mechanism:** structure each population member's perturbation as a **rank-`r`** matrix `A Bᵀ/√r` instead of a dense Gaussian, so the batched matmul becomes one dense matmul plus cheap rank-`r` corrections; prove consistency with full ES as dimension grows; overall update is full-rank because it sums `N` rank-`r` terms.
- **Claim:** "hundredfold increase in training speed for billion-parameter models at large population sizes", **up to 91% of pure batch-inference throughput**; enables integer-only recurrent LM pretraining; competitive with GRPO for post-training.
- **Why it matters here:** EGGROLL is the baseline DUST beats by 10³–10⁴× in the DUST paper. Its real contribution is an *implementation* insight (low-rank = high arithmetic intensity), which is orthogonal to our token-axis trick and could be combined.
- **Implementable in a day:** no.

### 4.3 EGGROLL, Unrolled (2026) — rank-one variance is small, and LOO halves MSE

- **Link:** arXiv:2609.10980. Kaya & Hashemi.
- **Mechanism:** analyses the EGGROLL mean field at finite rank and radius (it is a resolvent applied to the smoothed-objective gradient, which can be non-conservative and can reverse local stability); quantifies the extra sampling variance of rank-one perturbations vs dense Gaussian; introduces **LOO-ROLL**, a leave-one-out estimator replacing EGGROLL's two antithetic evaluations per direction with one.
- **Claim:** rank-one extra variance "decays inversely with matrix width" and is only **0.098% at width 4096**; LOO-ROLL **halves estimator MSE at equal evaluation cost** in transformer blocks; across 14 post-training settings up to 14B parameters, matched-time comparisons give 11 improvements and no significant loss.
- **Why it matters here:** two transferable facts. (i) Rank-one/low-rank structure is nearly free in variance at realistic widths — so weight-space ES is not variance-crippled, it is *throughput*-crippled; that sharpens what DUST's advantage actually is. (ii) Leave-one-out baselines can halve MSE at no extra cost — and our per-token centred loss is already a (token-axis) leave-one-out-style baseline, which is a nice retroactive justification to document.
- **Implementable in a day:** the LOO version of our per-layer baseline — use the other draws' mean rather than all-draws mean — is a small change and may reduce the small bias the current centering introduces. Half a day.

### 4.4 Evolution Strategies at Scale (ICML 2026) — ES fine-tunes billions

- **Link:** arXiv:2509.24372. Qiu, Gan, Hayes, Liang, Xu, Dailey, Meyerson, Hodjat, Miikkulainen.
- **Mechanism:** full-parameter ES (no dimensionality reduction) applied to LLM post-training, with the usual population/shaping machinery.
- **Claim:** "first successful application of ES to **full-parameter fine-tuning of LLMs at the billion-parameter scale**"; better tolerance to long-horizon/delayed rewards, less reward hacking, more stable than RL baselines.
- **Why it matters here:** the strongest counter-evidence to "ZO cannot touch 1B". Note it is *fine-tuning*, i.e. the search starts inside a good basin — the same caveat as MeZO.
- **Implementable in a day:** no.

---

## 5. DUST itself: what it claims, what it does not

### 5.1 DUST (Q Labs, Oct 2026)

- **Authors/link:** Samip Dahal, Bishwas Mandal, Serdar Gülbahar, Akshay Vegesna. https://qlabs.sh/research/dust · code: github.com/qlabs-eng/dust.
- **Mechanism (one paragraph):** add Gaussian noise to each linear layer's output **independently per token**; reward each token's noise by the centred loss reduction at that token plus, with decay γ, at later tokens it reaches through attention (`r_t = Σ_{s≥t} γ^{s−t} c_s`, `c_s = ℓ̃_s − ℓ_s`); the reward-weighted noise `ĝ_t = −(1/Kσ) Σ_i r_t^(i) a_t^(i)` is the estimated output error, and its outer product with the already-computed layer input, summed over tokens, is the weight gradient. No backward pass anywhere; embeddings become scatter-adds. Interference is controlled by perturbing different layer *types* in separate passes with per-type noise scales, and by running the LM head on cached logits over a vocabulary slab.
- **Measured claims:** at 100k–1M tokens DUST beats backprop with a few hundred to ~1k draws; at 10M/20M it trails by **0.06 / 0.17** at 16k draws; the 20M power-law fit's limit is **4.431 (95% CI 3.89–4.58)** vs backprop **4.633**, with the ladder still falling. EGGROLL at 16k (256× the population) still does not reach DUST at 64 draws and stays **0.4–0.6** above at 1M–20M; extrapolated EGGROLL need is "several thousand to ~10⁴×" DUST's population. Test loss at 10M tokens, 16k draws: 2M **5.171**, 38M **5.036**, 243M **5.086**; backprop 5.180/5.015/5.048. Cosine law `cos(K)=c_max/√(1+c/K)` with **RMSE < 0.06** per layer type. Base config 8L/512w/seq 2048/batch 8 (16k tokens/step).
- **What it does not show (read this before quoting the 1B number):** the largest **trained** model is **243M parameters at 10M tokens**; the largest training run is **20M tokens on ~38M parameters**. The "1B tokens" figure in the abstract and Figure 5 is a **cosine measurement of DUST's estimator evaluated on backprop-trained checkpoints**, not a DUST training run. The 243M model was also *worse* than 38M at every population — the "bigger is more population-efficient" claim holds up to 38M and then weakens.
- **Why it matters here:** v1.5 at 1B is roughly **4× beyond the largest model DUST trained and 50× beyond its token budget**, on a model shape (1B, ~24L, ~2048w) where every per-token perturbation dimension is 4× larger than the 512-wide test bed. That is the actual size of the extrapolation, and it should be stated in one sentence rather than buried.
- **Implementable in a day:** nothing — this is already `crowd_v15/dust.py`.

### 5.2 Precursor: decoupling search from learning (2025)

- **Link:** arXiv:2509.10973, Vegesna & Dahal.
- **Mechanism:** two-phase training — evolutionary **search in activation space** for diverse representations, then gradient regression from parameters to those representations.
- **Claim:** searched representations are learnable; regression approaches SGD on MNIST/CIFAR-10/CIFAR-100; performance improves with search compute up to saturation; resulting models follow different representational trajectories than SGD models.
- **Why it matters here:** this is the intellectual ancestor and it contains the honest caveat DUST inherits: search quality saturates with compute, and the learner still needs gradients in phase two. DUST is "phase one only, with the regression replaced by an outer product".

### 5.3 Commentary and critique: the HN thread is the real review

- **Link:** news.ycombinator.com/item?id=49970871 (280 points, 82 comments, ~41 days old at time of writing).
- **Substance:** the top critique is not empirical but complexity-theoretic: derivative-free optimisation on smooth/Lipschitz objectives has a provable complexity gap versus first-order methods, and "the more parameters you have, the more useful [the gradient] becomes"; the commenter links a 2026 proof (arXiv:2607.13335 — **I did not open it**). Secondary critique: every few years a derivative-free method gets this treatment and none displaces SGD. Defenders argue energy/biology plausibility.
- **Coverage-level critique (AIGuys/Medium, Vishal Rajput, 2026-11):** states the largest run is "20M tokens on a ~38M-parameter model", that compute per update is "thousands of times backprop's", and that the authors themselves say orders of magnitude more efficiency are needed. The article also teases a community "speedrun" that cut time-to-loss-4.2 "from …" — **the number is behind the member paywall and I could not read it**.
- **Why it matters here:** these are the objections our v1.5 writeup must pre-empt, because they are correct as stated. The defensible position is not "ZO scales" but "the token axis buys a specific constant factor, and we are measuring whether it is enough at 1B on volunteer hardware" — plus the fact that volunteer hardware makes memory, not FLOPs, the binding constraint.

---

## 6. Federated ZO: the closest architectural competitor

### 6.1 DeComFL (ICLR 2025) — O(1) communication per round

- **Link:** arXiv:2405.15861.
- **Mechanism:** federated ZO where each client evaluates a **scalar** directional derivative along a shared random direction `z` (common seed) and uploads only that scalar; the server aggregates scalars and updates with `ĝ = ⟨z, ∇f⟩ z`. Communication drops from `O(d)` to `O(1)` in both directions.
- **Claim:** dimension-free communication per round; convergence guarantees for the ZO-SGD-style scheme.
- **Why it matters here:** this is a fundamental challenge to our upload design. We currently upload a **weight-shaped** (1-bit compressed) update per volunteer. DeComFL uploads **one float**. The tradeoff is exactly the one DUST resolved in the other direction: a weight-shaped estimate amortises one volunteer's forward passes over all parameters, whereas a scalar estimate throws away everything but one projection. Our design is the variance-efficient choice; DeComFL is the bandwidth-efficient choice. Worth stating explicitly, and worth an ablation mode.
- **Implementable in a day:** yes as an **optional client mode**, not a default: client computes `⟨z, ĝ_local⟩` (dot product of its DUST estimate with the round's shared direction), uploads 4 bytes. Everything needed already exists (shared seeds via `derive_seed`). Half a day for the plumbing, one run to measure the variance cost. This is the single most informative *communication* experiment available to us.

### 6.2 Hessian-informed federated ZO (2026)

- **Link:** arXiv:2506.02370 ("HiSo"), reported as up to **5× faster convergence than DeComFL** with higher accuracy than all ZO baselines.
- **Mechanism:** federated ZO plus Hessian-informed preconditioning (successor to HiZOO in the FL setting).
- **Why it matters here:** shows the federated-ZO line is active and that preconditioning is where its gains are coming from. Same memory objection as HiZOO.

---

## 7. Compression foundations (already implemented — included for completeness)

### 7.1 Seide et al. (Interspeech 2014) — 1-bit SGD with error feedback

- **Link:** doi:10.21437/Interspeech.2014-274.
- **Claim:** gradients can be quantised "to but one bit per value" "at no or nearly no loss of accuracy" **if the quantisation error is carried forward across minibatches**. Without error feedback "training quickly" degrades.
- **Status here:** implemented in `feedback.py`; the docstring's framing (direction improves, magnitude does not) matches the paper's scope.

### 7.2 Karimireddy et al. (ICML 2019) — why error feedback is safe

- **Link:** arXiv:1901.09847.
- **Claim:** **EF-SGD with an arbitrary compression operator achieves the same convergence rate as SGD**, with no additional assumptions; sign-based compressors can diverge without it; error feedback also improves generalisation.
- **Status here:** implemented; this is the citation that justifies calling our 1-bit codec "free" rather than "a tradeoff".

**Gap worth noting:** both papers assume the compressor is applied to a *gradient*. We apply it to a *ZO estimate*, whose variance is already large relative to its mean. Error feedback reduces compression error but cannot reduce estimator variance, so the two error terms add. The 71% recovery we measured is consistent with that: feedback removes one of two error sources, and the residual is the estimator's own noise. That is a testable prediction — recovery should improve as population `K` grows. Nobody has published this combination (ZO estimate + 1-bit EF + Federated averaging); we may be first, which is a claim worth checking before we make it.

---

## 8. Ranked: what to actually do

Ranked by (expected payoff at 1B) ÷ (cost), with evidence quality stated honestly. "Evidence" = published, at what scale, for what task (FT = fine-tuning; PT = pretraining from scratch).

| idea | expected payoff at 1B | implementation cost | evidence quality |
|---|---|---|---|
| **Per-layer σ and population allocation tuned by split-half cosine SNR** (own extension of DUST §2.3's cosine-tuning, without needing backprop) | **High** — wrong σ per layer is the dominant silent failure; DUST shows per-type σ and allocation change results materially | **1 day** in `dust.py` + a coordinator knob | **Derived, not published.** DUST tunes by cosine-to-backprop; we have no backprop. Split-half cosine is my proposal. Validate on a 100k-token run first. |
| **MeZO-SVRG control variate across rounds** (Gautam et al., ICML 2024) with a **compressed** snapshot | **High** — up to +20% accuracy / 2× GPU-hours at FT scale; attacks the dominant error term directly | **2 days** (1 day if snapshot is full-precision bf16 and we accept 2 GB) | **Strong** at ≤66B fine-tuning; **none** at pretraining from scratch. The snapshot is the memory problem. |
| **Rank/standardised reward shaping** (Salimans 2017; standard in EGGROLL) | **Medium-high** — heavy-tailed per-token loss deltas are exactly what rank shaping fixes; free variance reduction | **0.5 day** — one function replacing raw `c_s` | **Strong** in ES literature; not tested in a DUST-style per-token reward. Low risk. |
| **Block-coordinate perturbation scheduling** (Zhang et al. 2024 block-wise; DUST's own layer-type separation) | **Medium-high** — interference is DUST's stated main cost at affordable populations; fewer co-perturbed layers per pass = less interference per draw | **1 day** — schedule in `coordinator.py`, no estimator change | **Strong** conceptually (DUST §2.3 does it by layer type); block-wise at LM scale benchmarked in Zhang 2024. |
| **Low-rank activation-basis perturbation** (ZO-Act 2026; theory from LOZO 2025 + Yue 2023) | **High but uncertain** — variance `∝ r` instead of `∝ d_out`; but no evidence at pretraining, and subspace bias may hurt early training | **3 days** — randomised SVD at round 0, publish `U` per layer, perturb `Uc` | **Medium**: consistent gains at 8B/13B fine-tuning; **zero** PT evidence. |
| **Input-decorrelated outer product** (Dalm et al. 2023) | **Medium** — better-conditioned `Σ ĝ_t x_tᵀ`; cheap | **0.5 day** — centre `x_t` before the outer product | **Medium**: published gains on multi-layer nets, small scale, not transformers. |
| **Orthogonalised draws across the K axis** (Flügel 2025 multi-tangent; orthogonal MC) | **Low-medium** — removes shared component across draws; helps most at small K where we actually operate | **1 day** — QR on `K × d_out` per layer, same seeds | **Medium**: multi-tangent paper shows gains; orthogonal-MC variance claims are for linear functionals — our estimator is quadratic in the noise, so the guarantee may not carry. Cheap to test, do not assume. |
| **MEAZO single-scalar step-size adaptation; explicitly NOT ZO-Adam** (Dbouk et al. 2026) | **Low-medium** — robustness, not speed; prevents a tempting mistake | **0.5 day** | **Strong** (2026, multi-family LLMs): ZO-Adam has no convergence advantage and costs memory. |

Two things deliberately **not** recommended: HiZOO-style diagonal preconditioning (violates the no-state constraint at 1B: a diagonal is 4 GB fp32) and forward-forward / predictive coding (no LM-scale evidence; would be a research programme, not a v1.5 feature).

---

## 9. The synthesis in five sentences

1. The activation-vs-weight argument is a **constant factor of `d_in`** (Werfel), not an asymptotic win, and it is contested in temporal settings (Züge).
2. DUST's genuinely new multiplier is the **token axis**, and it is bounded by tokens-per-step (2048 × batch), not by parameter count — so its population cannot grow the way an ES population can.
3. The field's best variance reductions at scale are **control variates (MeZO-SVRG)** and **subspace restriction (LOZO/ZO-Act/P-GAP)**, both of which trade a little bias or a little state for a large variance drop.
4. ZO-at-billion evidence is overwhelmingly **fine-tuning** evidence; pretraining-from-scratch evidence stops at 243M params / 20M tokens (DUST) and at CIFAR-scale (DeepZero).
5. Therefore v1.5's defensible claim is narrow and testable: *the token-axis virtual population plus error-feedback compression is enough to make 1B volunteer pretraining cheaper in memory than any backprop alternative* — not that ZO matches backprop at 1B.

---

## 10. One-paragraph reading order, if someone has two hours

Werfel 2005 (§1.1) for the dimensional argument → Ren 2023 (§2.2) for why activations → MeZO (§3.1) for the memory claim → MeZO-SVRG (§3.3) for the variance claim → DUST §4.1 and §4.2 (in the paper) for the only pretraining-scale evidence → EGGROLL Unrolled (§4.3) for the honest read on low-rank variance → DeComFL (§6.1) to see how our upload design differs from the federated-ZO mainstream.

---

## 11. Citations I could not verify

Listed so nobody quotes them as if I had read them.

1. **Züge et al., PRX 13, 021006 (2023)** — I verified the title fragment ("Weight Perturbation Often Performs Similarly or Better than Node Perturbation…") and first author initial from an ADS snippet, and the journal/volume/article number from the DOI link. I did **not** open the paper (APS blocked the fetch behind a JS challenge), so I cannot state the full author list, the task set, or any number. Treat §1.3's claim as direction-only.
2. **arXiv:2607.13335** — cited by an HN commenter as a proof of a complexity gap between derivative-free and first-order methods on Lipschitz convex objectives. I did not open it and cannot confirm title, authors, or that the proof says what the commenter says.
3. **Spall (1992), SPSA** — the canonical simultaneous-perturbation reference; I did not open it, so I omitted it from the sections rather than cite it from memory. If a foundations citation is needed, verify it first.
4. **Allaire et al., ICPRAM 2025 (doi:10.5220/0013261100003905) and KronZO, SN Computer Science 7 (2026), art. 162** — the ZO-pretraining line DUST cites as prior art. Taken from DUST's reference list; I did not open either. KronZO (low-rank Kronecker perturbations, selective directional updates) may be more relevant to us than its obscurity suggests, and should be read before the 1B run.
5. **"Zeroth-Order Fine-Tuning of LLMs in Random Subspaces", ICCV 2025 (Yu et al.)** — I saw a CVF open-access PDF URL and a one-line snippet about variance; I did not open it, and I could not confirm the arXiv ID. Do not cite the author list from me.
6. **HiSo / Hessian-informed federated ZO, arXiv:2506.02370** — the "5× faster than DeComFL" number comes from a PMC snippet quoting the paper, not from the paper itself.
7. **The community "speedrun" of DUST** (referenced in the AIGuys/Medium piece as cutting time-to-loss-4.2) — the number is paywalled. I have no figure and no primary source. If this matters, ask the Q Labs repo's issues/discussions rather than trusting secondary coverage.
8. **DUST appendix numbers** — I read the main paper page in full but not `qlabs.sh/research/dust/appendix.html`. The per-cell test losses above come from Figure 2/4 and the surrounding text; the exact Table 1 grid and the tuning grid are in the appendix I did not fetch.
9. **"MeZO-SVRG" memory claim exactness** — the abstract says "2× [reduction] for autoregressive models" relative to first-order SGD; I did not verify the measurement conditions (batch size, sequence length).
10. **Whether anyone has published ZO-estimate + 1-bit error feedback + federated averaging** — I found no such paper, but absence of evidence from a two-hour sweep is weak. Check before claiming novelty in §7's gap note.

---

*Method note: abstracts were fetched from arXiv/publisher pages on 2026-11-16 and quoted from those pages; numbers in backticks are as printed there. Where I quote a number that came from a secondary summary rather than the paper, the sentence says so.*
