# eos: ZO at the Edge of Stability (2604.14669), trace-set MeZO lr

## Faithful
- Gaussian two-point MeZO with n directions. Polyak momentum as in the paper: m = beta*m + g,
  w -= lr*m (no dampening).
- Stability rule (Thm 2): the band is [2(1-beta)/(Tr + 2 lambda_max/(1+beta)), 2(1-beta)/Tr],
  with lr*lambda_max < 1 - beta^2. I checked the PDF: beta squared (the recursion's beta2 terms are
  beta^2), not (1-beta)^2.

## Our extrapolation (not in the paper)
- One extra clean forward per step gives the forward-only trace sample (f+ + f- - 2 f0)/eps^2,
  which is unbiased for Tr H at Gaussian z up to O(eps^2).
- EMA (beta_tr=0.99, bias-corrected, clamped > 0) sets lr = min(lr_max, c*2(1-beta)*n/Tr).
  lambda_max is ignored. The factor n comes from the paper's App. B n-query operator.

## Diagnostic: `eos_check.py` (backprop HVPs, labelled diagnostic)
| point | Hutchinson (20) | forward-only (64) | lambda_max | predicted n=1 lr, beta=0 / 0.9 |
|---|---|---|---|---|
| init | 827 +- 138 | 873 +- 96 | about -2.5 | 2.3e-3 / 2.3e-4 |
| adam30 | -3215 +- 1758 | 630-720 +- 60-90 | -3358 | fwd: 2.8e-3 / 2.8e-4 |

At adam30 the -3358 eigenvector (92% tok_emb) is a micro-scale feature. I confirmed it in float64:
the second difference along it is -3377 at h=1e-5, -2620 at 1e-4, and about 0 at h>=1e-3. It
dominates Hutchinson but is invisible at the ZO radius, so the forward estimator is the relevant
number.

## Measured in smoke (12-15 s, n=4, eps=1e-3)
| arm | result |
|---|---|
| c=0.5 (spec default) | diverges by step 49 (lr about 7e-3) |
| c=0.2 | diverges |
| c=0.1 | stable but loss rises to 8.57: Tr falls, lr rises (EoS feedback loop) |
| c=0.05, beta=0 | 8.366 -> 8.328 (lr about 4e-4) |
| c=0.05, beta=0.9 | 8.366 -> 8.324 (lr about 4e-5) |

The full-batch bound is about 10-20x too loose for batch 8x128. Likely cause: data noise. The
usable lr shrinks by roughly ||E g||^2 / E||g_b||^2, which the theory leaves out.

probe on ckpt: n=4 gives 0.0000, n=16 gives 4e-4. This is the noise floor, the same as mezo.

## Round-1 arms (120 s each)
```
P=../../.venv/bin/python
$P harness.py eos n=4 eps=0.001 beta=0   c=0.05              --label r1-eos-c05-b0
$P harness.py eos n=4 eps=0.001 beta=0.9 c=0.05              --label r1-eos-c05-b9
$P harness.py eos n=4 eps=0.001 beta=0   c=0.02              --label r1-eos-c02-b0
$P harness.py eos n=16 eps=0.001 beta=0  c=0.05              --label r1-eos-n16-c05
$P harness.py mezo k=4 eps=0.001 lr=0.0001                    --label r1-mezo-k4
$P harness.py mezo k=4 eps=0.001 lr=0.0004                    --label r1-mezo-k4-lr4e4
# diagnostic only (not a training arm), about 15 s each:
$P eos_check.py ; $P eos_check.py --ckpt ckpt/adam30.pt
```
The eos arms cost 9 forwards per step against 8 for mezo k=4.

## Proves out if
- `r1-eos-c05-b0` matches or beats the better fixed-lr MeZO arm without a sweep, and no eos arm
  diverges.
- The beta=0.9 arm lands near the beta=0 arm's loss. In effective-lr terms the paper predicts no
  benefit from ZO momentum.
- `r1-eos-n16-c05` runs at about 4x the lr of n=4 and stays stable. That tests the linear-in-n
  law.
