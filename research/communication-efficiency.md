# Communication-Efficient & Asynchronous Distributed Training — Literature Sweep for Crowd-v1.5

Compiled 2026-11-16. Scope: what a small volunteer network training a 1B dense transformer with
**DUST** (zeroth-order, activation-space, no backward pass) could realistically adopt in 2026.
DUST: <https://qlabs.sh/research/dust> · <https://github.com/qlabs-eng/dust>.

## Why this matters for v1.5 — the assumptions that differ

Almost every paper below assumes the worker submits an **unbiased stochastic gradient** of a shared
objective and that its *magnitude* carries information. Crowd-v1.5 submits neither: the upload is a
**biased, very high-variance Monte-Carlo estimate** whose norm (~1.5e5 on 1B for a 64-token draw) is an
artefact of noise scale σ and token count N, not of distance from an optimum — hence we **normalise to
a unit direction** and re-apply a capped magnitude. Three consequences recur: (i) cross-round error
feedback is meaningless on a unit-normalised direction change; (ii) every method needing `σ ≪ ‖g‖` is
violated by construction; (iii) per-coordinate magnitude of a ZO estimate is nearly uninformative, so a
1-bit codec loses far less than first-order theory predicts — consistent with our 71% EF recovery, and
a reason to expect sign-family methods to over-perform here relative to their published results.

## 1. Compression

**1.1 EF21 / EF21+ — error feedback done properly.** Richtárik, Sokolov, Fatkhullin, *EF21*, NeurIPS
2021, <https://arxiv.org/abs/2106.05203>. Classic EF (Seide 2014 / Stich 2018) carries
`e_t = e_{t−1} + g_t − C(g_t)` on the raw gradient. EF21 instead carries a *state vector* `g_t`
approximating the gradient, communicates `C(∇f(x_t) − g_t)` (a **difference**), and sets
`g_{t+1} = g_t + C(∇f(x_t) − g_t)`; because the compressed quantity is a small residual, the
contraction factor θ enters the rate linearly instead of through a 1/θ blow-up. EF21+ overwrites
`g_{t+1}` with `C(∇f(x_t))` when that is better and tolerates much larger stepsizes.
*Measured:* first **O(1/T)** rate for EF (previous best O(1/T^{2/3}), and only under bounded
gradients) and first **linear rate under Polyak–Łojasiewicz** without unbiased compressors; stepsize
sweeps 512×–4096× show EF oscillating where EF21/EF21+ converge. Nuance: *"Tight analyses of
first-order methods with error feedback"* (OpenReview `hlPk6Hi43e`) computes numerically that EF and
EF21 have **identical** worst-case optimal contraction (difference 1e-5…1e-7) — so EF21's practical
win is **stepsize tolerance**, not the rate.
*One day:* nothing free. Our block codec is already EF21-style residual feedback *within* a submission.
Cross-round EF is the part that does not transfer.

**1.2 The uncomfortable result: EF alone does not converge at small batch.** Fatkhullin, Tyurin,
Richtárik, *Momentum Provably Improves Error Feedback!*, ICML 2024, <https://arxiv.org/abs/2305.15155>.
A convex 2-D L-smooth counterexample shows **EF21-SGD-ideal does not converge for any fixed stepsize
at batch size 1**; a mini-batch of Ω(σ²/ε²) is *necessary*, and larger batches do not rescue practical
EF21 (their Figs. 1a/4a). The fix is Polyak momentum on the EF state (EF21-SGDM), giving near-optimal
sample complexity, plus a double-momentum variant. *Why it is load-bearing for us:* our round is **one
draw** — the smallest possible batch — at a σ/‖g‖ ratio far beyond their counterexample, and their
prescription (momentum on the **server**, not the compressor) is exactly the DiLoCo outer-Nesterov
design (§2.1). *One day:* server-side Nesterov/EMA over aggregated unit directions, momentum 0.9.

**1.3 signSGD, EF-SignSGD, 1-bit Adam.** Bernstein, Wang, Azizzadenesheli, Anandkumar, *signSGD:
Compressed Optimisation for Non-Convex Problems*, ICML 2018 <https://arxiv.org/abs/1802.04434> (the
majority-vote variant is the ICLR 2019 follow-up, <https://arxiv.org/abs/1810.05291>) ·
Karimireddy, Rebjock, Stich, Jaggi, *Error Feedback Fixes SignSGD*, ICML 2019
<https://arxiv.org/abs/1901.09847> · Tang et al., *1-bit Adam*, ICML 2021
<https://arxiv.org/abs/2102.02888>. Signs must be **rescaled by ‖g‖** (else magnitude is lost forever)
and the sign error **accumulated locally** (else direction is lost); 1-bit Adam adds that Adam's second
moment stabilises after warmup (BERT-Large: flat from ≈23K steps), so `v` can be **frozen** and the
momentum compressed to 1 bit. *Measured:* up to **5× less end-to-end communication**
(`1/(warmup + (1−warmup)/16)`, warmup 6–15%), **3.3× throughput** on BERT-Large seq-128, 2.9× on
SQuAD, 5.48×/6.17× faster *in the compressed stage*, 93.32 vs 93.33 F1. Same O(1/√(nT)) rate.
Honest ceiling: max speedup is inversely linear in the warmup fraction, and the scheme rests on `v`
stabilising — validated for BERT-class workloads only.
*Relevance:* "per-1024-block mean-abs scale + sign bits" **is** sign-with-rescaling at block
granularity, i.e. the EF-SignSGD prescription. What we have not applied is the **warmup**.
*One day:* run the first ~5% of rounds at fp16 (or 8× residual planes), then drop to 1-bit+EF.

**1.4 QSGD and bucketing — the theory for why 1024 blocks is right.** Alistarh, Grubic, Li, Tomioka,
Vojnovic, NeurIPS 2017, <https://arxiv.org/abs/1610.02132>. Randomised rounding to `s` levels per
bucket of `d` consecutive coordinates, plus Elias coding. Variance blow-up ≤
`min(n/s², √n/s)·B`, expressed in the **bucket size d, not the model dimension n**.
*Measured:* with `d=512, s=16` the variance increase is bounded by `√512/16 ≈ 1.41×` — that one number
justifies block-wise scales. Bandwidth floor ≈`2.8n + 32` bits for ≤2× variance (≈5.7× saving).
8-bit/512-bucket recovers full accuracy everywhere tested; 4-bit/8192 loses 0.68% top-1 on AlexNet;
2-bit/64 loses 1.73%. *One day:* sweep block size 256/1024/4096 on the tiny model — the `√d/s` form
predicts 4096 blocks are nearly free and cut fp16 overhead 4×.

**1.5 PowerSGD / low-rank — good numbers, wrong tool here.** Vogels, Karimireddy, Jaggi, NeurIPS 2019,
<https://arxiv.org/abs/1905.13727>. Reshape each ≥2-D parameter to a matrix M; one warm-started step of
subspace iteration: `P = M·Q`, `Q = Mᵀ·orth(P)`, send `(P,Q)`. Linear in the update, so it composes with
all-reduce and EF. Biases/1-D tensors excluded. *Measured:* **>120×** compression; comms time −54%
(ResNet18/CIFAR-10) and −90% (LSTM/WikiText-2); **end-to-end −24% and −55%**; rank-2 matches SGD at
94.4% vs 94.3% test accuracy, 239 vs 312 ms/batch. At 128× it is the only scheme in their sweep to hit
target accuracy; at ~32× (sign-level) everything is competitive, so low-rank wins specifically in the
*aggressive* regime. *Why not for us:* cost is `O(n·m·r)` matmuls per layer, while our whole client
round is ~64 forward passes over 64 tokens — PowerSGD would cost as much client time as the training,
and requires shipping a rank-r basis back down. Worse, rank truncation keeps the **largest** directions,
which for a ZO estimate are the **highest-variance** ones: it would preferentially transmit noise.
*One day:* measure the **stable rank** of one 1B DUST update (SVDs on sampled layers). Build PowerSGD
only if stable rank ≲ 4.

**1.6 DeMo — momentum buffer *is* the error accumulator.** *DeMo: Decoupled Momentum Optimization*,
<https://arxiv.org/abs/2411.19870> (authorship unverified, §8). Exchange a **compressed momentum
buffer**; each worker subtracts the decoded update from its own buffer, so the buffer *is* the implicit
EF accumulator and explicit EF memory disappears (blockwise transform + top-k; SGD-m / Lion / Muon).
Useful framing: **EF and momentum are one object seen twice.** Needs a dense momentum buffer per
participant — 4 GB at 1B, which a 16 GB volunteer cannot spare.
*One day:* no. Supports §2.2's "the coordinator has the memory, volunteers do not".

**1.7 Zeroth-order-specific compression — the under-served intersection.** These are the only papers
that treat a ZO estimate's magnitude as uninformative *by construction*.
**FeedSign** (IEEE 2026, doc 11455968) — upload *and* download of **exactly 1 bit per aggregation
step** as a (seed, sign) pair from shared PRNGs; memory drops to inference-only; **exponential
convergence under Polyak–Łojasiewicz**. **ZO-SignSGD for FL** (IEEE 2025,
doi:10.1109/ieeeconf67917.2025.11443933) — transmits only signs of ZO *scalars* + seeds; claims
LLM-finetuning wins in communication and Byzantine robustness. **z-SignFedAvg**
(<https://arxiv.org/abs/2302.02589>) — symmetric-noise perturbation that tunes the sign-bias/
convergence tradeoff; first sign method permitting multiple local steps, matches uncompressed rates
under uniform noise. **DeComFL** (<https://arxiv.org/abs/2405.15861>) — ZO gradients decompose into a
*scalar* × a *seed-reproducible direction*; ≈1 MB **total** per client to fine-tune a multi-billion
parameter model.
*Why it matters most:* if our activation-space direction is at all seed-reproducible, "one bit per
parameter" is not the floor — **one scalar per draw** is. Highest-ceiling idea in this document.
*One day:* nothing. But measure the **cosine similarity between two DUST draws at the same θ**: if
≈0, magnitude is noise and §1.3/§1.7 is the right family; if high, there is a shared systematic
direction we should be extracting instead of averaging away.

## 2. Local-SGD / DiLoCo family

**2.1 DiLoCo — the reference design.** Douillard, Feng, Rusu, Chhaparia, Donchev, Kuncoro, Ranzato,
Szlam, Shen, arXiv:2311.08105 (Nov 2023), <https://arxiv.org/abs/2311.08105>. `H` inner steps with an
inner optimizer, then average the **parameter-space delta** `Δ_m = θ^{t−H} − θ_m^t` and apply one outer
step to `θ^{t−H}`. The paper is explicit that Δ is **not the gradient of any function** — a
*pseudo-gradient*, which is exactly the epistemic status of our upload. Config: inner AdamW lr 4e-4,
H=500, **outer Nesterov lr 0.7 momentum 0.9**, batch 512×seq 1024, 8 workers.
*Measured:* 8 workers match fully-synchronous training with **500× less communication**; 150M model at
8× effective batch — DiLoCo **15.02** perplexity vs 15.30 (8× data-parallel) vs 16.23 (no replicas) vs
14.72 (8× updates oracle), at the same wall-clock as the DP run. Outer-optimizer ablation: **SGD
(=FedAvg) and Adam both performed poorly; Adam was "particularly unstable with a high second-order
momentum norm"** and needed ε=0.1. Nesterov won.
*One day:* outer Nesterov on the coordinator's aggregated unit direction, lr 0.7, momentum 0.9. ~20
lines, and the best-evidenced single change available.

**2.2 OpenDiLoCo — where large H breaks at 1B.** Jaghouar, Hagemann et al. (Prime Intellect),
arXiv:2407.07852, <https://arxiv.org/abs/2407.07852>. Hivemind implementation; FP16 all-reduce of the
outer gradient with no degradation; ran across 3 countries at 127–935 Mbit/s.
*Measured — the key negative:* 1.1B, 4 workers, batch 2048. H=125 → **10.76** vs 10.52 for
equal-compute 4×-batch DP (gap 0.24, ~125× less communication). **H=500 → 11.14** — visibly
under-converged, and a 0.38 gap from its own H=125 run. DiLoCoX (§2.5) blames exactly this on
"excessively large H causing gradient staleness" and uses H=125 at 1.3B. Utilisation 90–95%; comms
only **6.9%** of training time at 1B over the open internet.
*Interpretation:* **local-step count scales inversely with model size**, not with it. At 150M H=500 is
fine; at 1B H=125 is fine and H=500 is not. Anything built around "many local draws" must target the
1B regime, not our tiny-model v1 numbers.

**2.3 Scaling laws for DiLoCo — the tuning map.** Douillard et al., NeurIPS 2025,
<https://arxiv.org/abs/2503.09799>. Sweeps 35M–2.4B, M∈{1,2,4,8}, H∈{1,5,10,30,100,300},
η∈{0.05…1.0}. (1) **Optimal outer lr is independent of N** for N ≥ 335M — only M and H matter
(≈0.4 at M=1, ≈0.6–0.8 at M=8, H=30). (2) Optimal outer η **increases with H** (counter-intuitive but
measured). (3) `γ ≈ 22256·N^−0.883·M^0.293`; `B ≈ 0.00709·N^0.470·M^0.340`;
`L ≈ 19.226·N^−0.0985·M^0.0116`. (4) Validated at 4B/10B: DiLoCo M=1/M=2 **beats** DP (2.219/2.220 vs
2.224 at 4B; 2.086 vs 2.090 at 10B); M=4 is 0.18–0.29% worse.
*One day:* adopt the rule instead of sweeping — set outer η from (M,H) only, and note **more clients
want a larger outer lr**, the opposite of "average more, trust more".

**2.4 Streaming DiLoCo — fragments + overlap + FP4.** Douillard et al., arXiv:2501.18512,
<https://arxiv.org/abs/2501.18512>. Three orthogonal changes: synchronise **one parameter fragment at a
time** (24 layers / fragment 3 → 8× peak-bandwidth cut), **overlap** a fragment's communication with
the next τ inner steps (mixing factor α; α=0 discards interim local work, α=1 is no communication), and
**FP4-quantise** the outer gradient. *Measured,* 1B on Dolma at 1 Gbit/s: 25B tokens → **0.88 h vs
109 h** for data-parallel; 250B tokens → **8.75 h vs 1097 h**; eval loss 2.45 vs 2.45; **400× fewer
bits total**, **8× lower peak bandwidth**; communication may take as long as a full compute step.
Utilisation 95% at 1–5 Gbit/s. Their advice: **cap overlap at ~5 inner steps**. *One day:* the
fragment-by-fragment upload is implementable (stream 1/24 of the tensor) and would cut coordinator peak
memory and any single-request timeout. FP4 is a second-day item; 1-bit already beats it.

**2.5 DiLoCoX — 107B on 1 Gbps, with an honest ablation of the codec's cost.** Qi, Zhu, Li, Wu, Wu, He,
Gao et al., arXiv:2506.21263, <https://arxiv.org/abs/2506.21263>. Pipeline parallelism + "dual optimizer
policy" + one-step-delay overlap + adaptive compression. Bandwidth limited to 1 Gbps with Linux `tc`
(**simulated**, not real internet).
*Measured:* 107B (modified Qwen1.5, 78 layers) on 160×A800 — **3,728 vs 10.4 tok/s** for AllReduce
(**357×**), 1.35× over CocktailSGD; loss 4.20 vs 3.90. The ablations are the useful part: **without
adaptive compression, loss improves 4.20 → 4.02 but throughput falls 3,728 → 1,168** (the codec costs
~0.18 nats and buys 3.2× speed); **without one-step-delay overlap, loss 4.15 but throughput halves to
2,197**. At 1.3B: AllReduce 4.06, DiLoCoX 4.27, OpenDiLoCo 5.37, CocktailSGD 5.79. Caveat (0G, Mar
2026): 4,000 steps on WikiText-103 is proof-of-concept and bandwidth was emulated.

**2.6 Decoupled DiLoCo — async learners, quorum, and *our own aggregation rule*.** Google DeepMind,
arXiv:2604.21428 (Apr 2026), <https://arxiv.org/abs/2604.21428>. Independent "learners" run inner steps
and asynchronously ship **parameter fragments** to a synchroniser, which merges arrivals subject to a
**minimum quorum K**, an **adaptive grace window** ξ_grace, and **dynamic token weighting**. Two pieces
are directly applicable:

- **The weight function.** `w = c_tokens × (c_tokens / c_steps)` — *quantity × quality*: contribute
  more if you processed more tokens, trust more if those tokens were amortised over **fewer** steps.
  The only published, empirically validated answer we found to §4.3's question.
- **Radial-Directional Averaging.** `RDA(v₁…v_M) = (mean‖vᵢ‖)·φ(mean φ(vᵢ))`, φ(x)=x/‖x‖ — average
  magnitudes and directions separately, renormalise the mean direction, multiply. **Exactly what
  Crowd-v1.5 already does.** Motivation: per-learner outer gradients are **near-pairwise-orthogonal**,
  so plain averaging gives norm ≈ R/√M and forces outer-lr re-tuning whenever M changes; RDA makes the
  norm invariant to M.

*Measured:* 2B Gemma-2 / 26B tokens — at **M=8** no consistent winner, RDA-on-model + average-on-embedding
comparable or better; at **M=16 RDA on the non-embedding part "drastically improves performance"**,
while RDA on the *embedding* is often worse than plain averaging. Async/quorum: M=8 at 1.2M simulated
chips, 1-year per-chip MTBI → **88% goodput vs 58%** for elastic data-parallel, **zero global
downtime**, text-avg 69.8 vs 70.1 (no measurable ML cost). Heterogeneous TPU v5e/v5p with injected 10%
speed skew: K=1 + grace window matches the blocking K=M=8 baseline on text (55.3 vs 54.4) and vision
(37.0 vs 37.6) at 100% vs 84% goodput.
*Critical caveat for us:* RDA's justification is that honest submissions are near-orthogonal. Our bias
is **shared across clients** (same smoothing, same σ, same activation-space projection), so submissions
contain a **coherent component**; mean-of-norms then overstates aggregate magnitude in proportion to
how much of each submission is shared bias rather than independent noise. Our cap is load-bearing — and
should arguably come from the **median** norm, not the mean. *One day:* nothing to implement. Harvest
two things: `w = N_tokens × (N_tokens/steps)` if draws ever vary, and the measured **"RDA everywhere
except embeddings"** finding.

## 3. Asynchronous / staleness-tolerant methods

**3.1 FedBuff — buffered async aggregation.** Nguyen, Malik, Zhan, Yousefpour, Rabbat, Malek, Huba
(Meta), AISTATS 2022, <https://arxiv.org/abs/2106.06639>. Clients train asynchronously and deposit
updates in a server buffer; the moment **K** updates are present the server applies one optimizer step
immediately. Decouples server-step rate from the slowest client, stays compatible with secure
aggregation. *Measured:* **3.3–3.8× more efficient than FedAvgM**, 2.5× than FedAsync (CelebA to 90%:
27.1K vs 104K vs 28.7K client trips), 1.7× on Sent140; gains **consistent across half-normal, uniform
and exponential staleness** (4.65×/3.83×/4.34× over FedAvgM); wall-clock 3.6× faster at K=1000, and
still 25–41% faster than FedAvgM with 30% over-selection. **K=10 "a good setting across benchmarks and
does not require tuning."**
*One day:* highest-value systems change available. A buffer of K=8–16 aggregated on arrival removes the
per-round latency floor; buffer + outer Nesterov = async DiLoCo in two features. Over-issuing 2K and
applying the first K bounds τ by construction — the only staleness bound we can actually enforce
(FedBuff's own ablation: 30% over-selection narrows, but does not close, the 25–41% straggler gap).

**3.2 Staleness theory — and why it is the least of our problems.** Arjevani, Shamir, Srebro, ALT 2020
<https://proceedings.mlr.press/v117/arjevani20a.html>: matching bounds on quadratics — **without noise,
delay is catastrophic** (error as bad as GD on 1/τ of the gradients); **with stochastic noise the effect
of delays becomes negligible**. Koloskova, Stich, Jaggi, *Sharper Convergence Guarantees for
Asynchronous SGD*, NeurIPS 2022 <https://arxiv.org/abs/2206.08307>: rates
`O(σ²/ε² + √(τ_avg·τ_max)/ε)` improve to `O(σ²/ε² + τ_avg/ε)` with a **delay-adaptive stepsize**
(`η_t = η` for `τ_t ≤ 2τ_C`, `min{η, 1/(4Lτ_t)}` otherwise) — equivalently, **dropping the top half of
delayed updates costs at most 2×**. Mishchenko, Jaggi, Koloskova, Stich, NeurIPS 2021: `O(σ²/ε⁴ +
τ/ε²)` with τ the *average* delay. Mishchenko et al., *Async SGD Beats Minibatch SGD Under Arbitrary
Delays*, NeurIPS 2022 <https://arxiv.org/abs/2206.07638>: bounds depend on the **number of devices, not
the delay**.
*Composite message:* our σ is enormous, so by Arjevani et al. staleness should be nearly free, and by
Koloskova et al. we can tolerate *unbounded* delays with an adaptive stepsize at no theoretical cost.
**But all of it assumes unbiasedness**, and a bias evaluated at a stale point does not cancel: delay
turns a biased estimator into a *consistently outdated* biased one, and the floor becomes
`‖bias(θ^{t−τ}) − bias(θ^t)‖`. Practical rule: bound τ mechanically (FedBuff's buffer) rather than
reason about it analytically. *One day:* `lr_scale = 1/max(1, τ/2)` on top of whatever outer optimizer
we adopt — three lines, no tuning, theoretically free.

**3.3 Gossip / consensus SGD.** Assran, Loizou, Ballas, Rabbat, *Stochastic Gradient Push*, ICML 2019
<https://arxiv.org/abs/1811.10792> — PushSum + SGD over directed, sparse, time-varying topologies;
O(1/√(nK)) rate. Measured at 256 GPUs / 32 nodes over 10 Gbps Ethernet: AllReduce SGD reaches 76.2%
top-1 on ResNet-50/ImageNet; Overlap-SGP reaches **76.2% in 1/3 the time** and **77.1% in 1/2 the
time**; on Transformer WMT'16 En-De, ~1.5× faster and +0.6 BLEU. Koloskova, Loizou, Boreiri, Jaggi,
Stich, ICML 2020 <https://arxiv.org/abs/2003.10422> gives the unified theory, with a `1/(1−β)`
spectral-gap penalty plus a heterogeneity term. For a coordinator-served-weights architecture this is a
rewrite, and the penalty is worst for exactly the sparse, churn-prone topologies volunteers have. **Not
recommended for v1.5.** The transferable idea is elasticity: **Bamboo** (Thorpe et al., *NSDI 2023*,
<https://arxiv.org/abs/2204.12013>) fills pipeline bubbles with redundant computation so a preempted
node's work is already redone — **3.7× throughput over checkpointing, 3.6× lower cost** than
on-demand — our analogue being "issue 2K, accept the first K", i.e. FedBuff with over-issue.

## 4. Aggregation defences and weighting

**4.1 Krum / trimmed mean / median — already in v1, and violated by our setting.** Blanchard, El
Mhamdi, Guerraoui, Stainer, NeurIPS 2017 <https://arxiv.org/abs/1703.02757>; Yin, Chen, Kannan,
Bartlett, ICML 2018 <https://arxiv.org/abs/1803.01498>. Krum picks the update minimising squared
distances to its `n−f−2` nearest neighbours (O(n²d); Multi-Krum averages the m best). Coordinate-wise
median has breakdown 1/2 but an extra `1/n` rate term; trimmed mean has the better rate
`Õ(α/√n + 1/√(nm))` but needs β ≥ f, and both decide each coordinate in isolation "so a clever
adversary can stay just inside the kept band on every coordinate while still pushing a coherent
direction".
*The condition we violate:* Krum's guarantee needs `η(n,f)·√d·σ < ‖g‖`, η(n,f)=O(√n) for f=O(1). As
*Fall of Empires* (AISTATS 2020, <https://proceedings.mlr.press/v115/xie20a.html>) puts it, this "is
never guaranteed when variance is non-zero, and the better SGD converges the less likely it is to
hold"; that paper also proves Krum breaks under inner-product manipulation, and Krum's breakdown point
deteriorates to 1/4 (via the ACM *Byzantine ML Primer*, 2024).
*But note what unit-normalisation does:* every submission has norm 1, so `d(i,j)² = 2 − 2cos`, and if
submissions are near-orthogonal in 1B dimensions — the *same* fact Decoupled DiLoCo relies on for RDA
(§2.6) — then `cos ≈ 0` for essentially every pair and **all Krum scores concentrate at
≈2·(n−f−2)**. Honest-vs-adversarial differences shrink below the spread and selection degenerates
toward a coin flip. **Distance-based filters are weakest exactly where RDA is strongest; it is one
observation.** Coordinate-wise median on a near-orthogonal unit vector is likewise close to a random
half of the coordinates — no better than the mean, and it destroys the sign structure our codec needs.
*One day:* add no machinery. **Instrument**: log the pairwise-cosine distribution per round. That one
histogram tells us whether Krum/median do anything at all, and whether RDA's premise holds for us.

**4.2 FedNova — and why Crowd-v1.5 does not need it.** Wang, Liu, Wang, Song, Liu, Zhang, Sahu, Cadena,
NeurIPS 2020, <https://arxiv.org/abs/2007.07481>. When clients run different local step counts τᵢ, naive
weighted averaging converges to a stationary point of a **mismatched objective** `F̃ = Σ wᵢFᵢ` whose
weights are set by the τᵢ rather than by data sizes pᵢ. FedNova aggregates the *normalised* update
`dᵢ = Δᵢ/τᵢ` weighted by pᵢ, with `τ_eff = Σ pᵢτᵢ`; works with any local solver and any server
optimizer. *Measured:* non-IID CIFAR-10, 16 clients, Dir(0.1), VGG-11 — FedAvg 60.68% vs **FedNova
66.31%**; with local momentum 65.26 vs **73.32**; random τᵢ∈U(2,5): 70.44 vs **77.07**. Local+server+VR
momentum reaches 79.19%, a combination FedAvg cannot even define.
*We are exempt, and this is a real architectural advantage.* FedNova corrects **client drift** — local
optimiser progress on non-IID shards pulling θᵢ toward a client-specific optimum. DUST has **no local
optimizer state and no local progress**: k draws at the same θ are just a bigger Monte-Carlo batch of
the *same* estimator at the *same* point, so `Δᵢ` from a client doing 64 draws and one doing 640 differ
only in **variance**, not in **bias toward a different objective** — the `χ²κ²` floor is structurally
absent. Do not import its remedies.

**4.3 Weighting submissions with different evidence.** Three distinct answers, not interchangeable:

| basis | rule | source | assumption |
|---|---|---|---|
| sample count | `w ∝ nᵢ` | FedAvg; Mime | unbiased, i.i.d.-ish |
| local progress | `w ∝ pᵢ`, `dᵢ = Δᵢ/τᵢ` | FedNova | drift is the pathology |
| quantity × quality | `w = c_tokens·(c_tokens/c_steps)` | Decoupled DiLoCo | async, per-fragment |
| inverse variance | `w ∝ 1/σᵢ²` | estimator theory | independent, unbiased |

For a Monte-Carlo estimator the correct rule is inverse-variance, and for a mean of N independent draws
`Var ∝ 1/N`, so **`w ∝ N_tokens` *is* inverse-variance weighting** — same answer as FedAvg's data-size
weighting, reached for a different reason. The Decoupled-DiLoCo "quality" factor is the genuinely new
term: `tokens/steps` proxies per-draw SNR, and `tokens/draw` is a *direct* measurement of the same
quantity (constant at 64 for us; a real lever once clients choose their draw size).
*Honest limit:* inverse-variance weighting removes **variance** differences, not **shared bias**. If
every client carries the same b(θ), then `Σ wᵢ ĝᵢ = g(θ) + b(θ)` for any wᵢ. Weighting is a
second-order fix; the bias is first-order and untouched.

## 5. Server-side amortisation

**5.1 FedOpt / FedAdam / FedYogi.** Reddi, Charles, Zaheer, Garrett, Rush, Konečný, Kumar, McMahan,
ICLR 2021, <https://arxiv.org/abs/2003.00295>. Treat the aggregated client delta as a **pseudo-gradient**
and run a server-side adaptive optimizer on it; server momentum alone (FedAvgM) is the m=1 case. Gains
over FedAvg are largest exactly in our regime (high heterogeneity, many local steps).
*Counter-evidence:* DiLoCo found outer Adam "particularly unstable with a high second-order momentum
norm" (needed ε=0.1) and outer SGD (=FedAvg) also poor, with Nesterov winning (§2.1); Mime (§5.2) shows
FedAvgAdam collapsing on Shakespeare (**18.5% vs MimeAdam 54.1%**). Consensus: **outer momentum yes,
outer adaptivity not without careful ε** — and a second moment over *unit-norm* pseudo-gradients is
especially suspect, since `v_t` degenerates into a running mean of per-coordinate squared signs, nearly
constant, which is exactly the degeneracy 1-bit Adam exploits deliberately by freezing it.

**5.2 Mime / MimeLite.** Karimireddy, Jaggi, Kale, Mohri, Reddi, Stich, Suresh, ICML 2021,
<https://arxiv.org/abs/2008.03606>. Compute momentum **at the server** (so it stays unbiased w.r.t. the
global objective) and apply it *inside* each client's local update, with an SVRG-style control variate
for drift; MimeLite keeps only the server-statistic trick. *Measured:* about **7× faster than FedAvg**
on natively federated benchmarks, Mime and MimeLite curves essentially identical; proves for the first
time that local steps can beat the server-only lower bound (MimeMVR, when Hessian dissimilarity δ ≪ L).
Known failure: **sparse gradients** (StackOverflow 12.5 vs FedAvgSGD 23.8).
*One day:* no — needs server state pushed *down* and (for Mime proper) doubles upload volume. The
transferable lesson is negative and useful: **do not put adaptivity inside the client's estimator.**
DUST has no client optimizer by design; Mime's sparse-gradient failure is the warning about adding
state to a deliberately stateless path.

## 6. Where the assumptions break, ranked by damage

1. **`σ ≪ ‖g‖`** (Krum/trimmed-mean/median, half the compression theory). Violated by orders of
   magnitude: distance-based filters lose discriminative power once submissions are unit-normalised
   and near-orthogonal (§4.1), and "robust aggregation improves the rate" does not transfer.
   *Mitigation:* none from this literature — measure the cosine spectrum first.
2. **Unbiasedness** (all of §3, §4.3, every EF convergence theorem). EF, FedNova, inverse-variance
   weighting and the delay analyses all bound `E[ĝ] = g`. A shared bias survives every averaging rule,
   weighting scheme and staleness bound. EF recovers 71% of our *codec* error because codec error is
   zero-mean; nothing analogous exists for smoothing bias, and any EF variant accumulating residuals
   **across rounds** accumulates bias rather than cancelling it.
3. **"Magnitude is meaningful"** (PowerSGD rank truncation, top-k, tiered scales). Our magnitude is an
   artefact of σ and N. Rank truncation keeps the *largest* directions — for a ZO estimate those are
   the highest-variance ones, so **PowerSGD would preferentially transmit noise.** Stronger reason to
   skip it than its compute cost.
4. **"Local steps create drift"** (FedNova, SCAFFOLD, FedProx, the H sweeps). Does not apply: no local
   optimizer state ⇒ k draws at fixed θ is a bigger batch, not a divergent trajectory. **But the
   empirical H limits still bind** (OpenDiLoCo: H=125 ok, H=500 not, at 1B) — for us a *client
   wall-clock* budget, not a correctness budget. That asymmetry is the most exploitable fact here.
5. **"One sample per round suffices because gradients are unbiased."** Contradicted by EF21-SGDM's
   necessity result (§1.2): at batch 1 the compressor alone is insufficient; momentum must move to
   the server.

## 7. Ranked shortlist

Payoff is my judgement for *our* numbers (1-bit codec + EF shipped, ~1.5e5 raw norm, 71% recovery).

| idea | payoff at 1B | cost | assumes unbiased? | evidence |
|---|---|---|---|---|
| **1. Server-side outer Nesterov (lr 0.7, mom 0.9) on aggregated unit directions** | High — best-evidenced change; also exactly the fix EF21-SGDM's theory demands at batch 1 | ~20 lines, coordinator only | **No** — DiLoCo is a *pseudo*-gradient method and explicitly says Δ is no function's gradient | **Strong**: DiLoCo + DiLoCoX + Scaling Laws agree on mom 0.9; 4B/10B validated |
| **2. FedBuff buffer (K≈8–16) + over-issue 2K, apply on arrival** | High — removes the per-round latency floor; 3.3–3.8× fewer client-trips measured | small: queue + apply-on-arrival | Yes in theory, but the *mechanism* (bound τ structurally) is assumption-free | **Strong** on efficiency (3.8×/2.5×, three staleness distributions); ML-quality evidence small-scale |
| **3. More draws per client before upload — our "local steps"** (k independent draws at fixed θ, averaged) | High — variance ∝ 1/k, zero extra bytes, and **no drift** because there is no local optimizer | small: loop the existing draw k× client-side; coordinator unchanged | **No** — pure Monte-Carlo variance reduction on a biased estimator: shrinks variance, leaves bias | **Strong indirectly**: EF21-SGDM proves batch-1 EF cannot converge; DiLoCo/OpenDiLoCo fix the H budget. Our own tiny-model data says variance dominates |
| **4. Token-count (inverse-variance) weighting, `w ∝ N_tokens`** | Medium — correct rule, nearly free, but does nothing about shared bias | trivial (one weight field) | **Essentially yes** — optimal only for independent unbiased estimates | **Strong** as statistics, **weak** as a measured 1B win (never isolated from data-size weighting) |
| **5. Decoupled-DiLoCo weight `w = tokens × (tokens/steps)`** | Medium — only valuable once clients choose their own draw counts | trivial | Yes | **Weak–moderate**: one paper, one scale (2B/26B tokens), no ablation isolating the weight from quorum/grace-window effects |
| **6. Uncompressed (or 8-bit) warmup for the first 5–15% of rounds** | Medium — 1-bit Adam's own precondition; our variance is highest early too | one coordinator flag | No — the warmup argument is about variance, of which we have more | **Moderate**: strong for BERT (6–15% warmup, 5× volume, 3.3× throughput); extrapolated by analogy to ZO |
| **7. Refine the shipped RDA cap: median norm, and plain averaging on embeddings** | Medium — median is more outlier-robust than mean; the embedding exception is a measured finding | trivial to try | **No — and that is the caveat**: RDA's norm-preservation argument assumes near-orthogonal submissions; our shared bias makes it overstate magnitude | **Moderate**: M=16 improvement real and large; M=8 a wash; embeddings explicitly excluded |
| **8. PowerSGD rank-r low-rank compression** | Low-to-negative at 1B — 120× is real but we are already at 1 bit/param; client matmul ≈ the cost of the whole round; rank truncation keeps our noisiest coordinates | 2–3 days (client power iteration + basis distribution) | **No — actively wrong here**: it assumes large-magnitude directions are signal | **Strong** for first-order 16-GPU wall-clock (−24%/−55%); **not applicable** to a ZO estimate |
| **9. QSGD block-size sweep (1024 → 4096) + scale format** | Low–medium — `<2×` variance at the `√d/s` bound; win is halving fp16-scale overhead | half a day; harness exists | Partly — the variance bound is machinery-independent | **Strong** bound and accuracy sweeps; marginal win at our bit rate |
| **10. Krum / trimmed-mean / median upgrades** | Low-to-negative — precondition `σ ≪ ‖g‖` violated; near-orthogonal unit vectors make distances concentrate | already shipped; *adding* more is the risk | **Yes, strongly** | **Strong that the precondition fails** (*Fall of Empires*); the failure mode is well documented |
| **11. Gossip / consensus SGD (decentralised)** | Low for v1.5 — a rewrite with a `1/(1−β)` penalty worst for churn-prone topologies | weeks | Yes | **Weak here** — theory only; no 1B-scale numbers verified |

## 8. Citations I could not verify

Listed separately so they are not mistaken for the verified set above.

- **DeMo** (arXiv:2411.19870). Read abstract + momentum-subtraction mechanism; **did not verify the
  author list or venue**, and found no independently numbers-verified summary. Mechanism good, claims
  unverified. §1.6
- **"Tight analyses of first-order methods with error feedback"** — only an OpenReview PDF
  (`hlPk6Hi43e`) without authors or venue I could confirm; the EF ≡ EF21 worst-case equivalence (max
  contraction difference 1e-5…1e-7) is read from their text, unattributed. §1.1
- **DiLoCoX authorship.** arXiv shows "Ji Qi, WenPeng Zhu, Li Li, Ming Wu, …" (truncated); HuggingFace
  credits "submitted by Arthur Douillard". Full author list unresolved. §2.5
- **"Achieving Extremely Low Communication Overhead in FL via ZO SignSGD"** (IEEE 2025,
  doi:10.1109/ieeeconf67917.2025.11443933) — abstract only; **no numbers verified**, authors not
  retrieved. §1.7
- **FeedSign** (IEEE 2026, doc 11455968) — abstract only; the PL exponential-rate claim is from the
  abstract, **no experiment numbers verified**, authors not retrieved. §1.7
- **NF4 / Int4 gradient quantization.** **No** credible source found applying NF4 (a *weight* datatype
  from QLoRA) to *gradients*, and no Int4-with-EF result beyond the QSGD family. The Int4 claims
  actually evidenced here are QSGD 4-bit/512-bucket (variance ≤1.41×) and Streaming DiLoCo's FP4 outer
  gradient. **Do not treat "NF4 for gradients" as established.**
- **Bamboo** — cited as NSDI 2023 from the USENIX proceedings PDF; I read the 3.7×/3.6× figures from that
  PDF but did **not** verify that arXiv:2204.12013 is the same version. §3.3
- **Mime's "7× faster"** — read in the body and a figure caption, but the dataset is unresolved in the
  retrieved text; §5.2 numbers come from verified Table 2 accuracies. **Krum breakdown point ¼** and the
  *Fall of Empires* inner-product attack — read second-hand via the ACM *Byzantine ML Primer* (2024),
  not from Farhadkhani et al. directly. §4.1
- **Our own DUST numbers** (1.5e5 raw norm; 3.7947 / 4.1121 / 3.8863 at 30 rounds; 71% recovery) are
  taken from the v1 handover as given, **not** independently verified here.
