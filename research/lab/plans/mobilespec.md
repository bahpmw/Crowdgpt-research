# mobilespec: MobileSpec / FuRA-S block-spectral ES (2611.04797)

## Faithful
- Column-block SVD of every 2-D block linear (W_k = U_k diag(s_k) V_k^T, frozen U, V).
- Only block singular values train, relatively: s' = s(1+delta).
- Exact rewrite W' = W0 + sum_k U_k diag(s_k delta_k) V_k^T (paper Eq. 4-5), so delta=0 is
  bit-exact.
- D = mu_rel |theta0| becomes a plain mu_rel in delta coordinates.
- Antithetic ES: `pairs`=15 (N=30), Gaussian z, population z-scored fitness (f = -loss), one
  shared batch, alpha=1, mu_rel=3e-2, no momentum.
- Logs the "beats center" fraction (paper Fig. 5).

## Simplified / chosen
- Block width b = 64 (the paper's b is about sqrt(n), i.e. 16/32 here). There are 12,288
  trainable values.
- tok_emb/head and RMSNorm gains are frozen. There is no forward-only exact path for them, and
  the paper adapts linears only.
- No LM-head offsets, no ZO-Adam variant (Eq. 16).
- `estimate()` maps the delta-space ES step to weight space for probe.py.
- Upload = seed + (f+, f-) per pair = 128 B/step.

## Measured in smoke
| run | result |
|---|---|
| scratch, 15 s | 8.366 -> 8.359 (89 steps): stalls, as expected |
| from adam30, 15 s | 5.5977 -> 5.5982 (best 5.5971), flat; beatCenter about 0.4-0.47 |
| ft alpha=4 | worse (5.607) |
| ft mu_rel=0.1 | 5.5959 (slightly better) |
| ft alpha=4 block=16 | 5.610 |

probe on ckpt (CPU, 3 repeats, 30 forwards): mobilespec 0.0062, mu_rel=0.1 0.0037,
mezo k=15 0.0002. That is about 30x better block alignment at equal forwards, but it is confined
to a 12k-dim space.

## Round-1 arms (120 s each)
```
P=../../.venv/bin/python
$P harness.py mobilespec pairs=15 mu_rel=0.03 alpha=1                         --label r1-ms-scratch
$P harness.py mobilespec pairs=15 mu_rel=0.03 alpha=1 --init ckpt/adam30.pt   --label r1-ms-ft
$P harness.py mobilespec pairs=15 mu_rel=0.1  alpha=1 --init ckpt/adam30.pt   --label r1-ms-ft-mu10
$P harness.py mobilespec pairs=15 mu_rel=0.03 alpha=1 --init ckpt/adam30.pt --batch 32 --label r1-ms-ft-b32
$P harness.py mezo k=15 eps=0.001 lr=0.0001 --init ckpt/adam30.pt             --label r1-ft-mezo-k15
$P harness.py mezo k=15 eps=0.001 lr=0.0001 --init ckpt/adam30.pt --batch 32  --label r1-ft-mezo-k15-b32
$P harness.py adam lr=0.003 --init ckpt/adam30.pt                             --label r1-ft-adam
```
- `r1-ft-mezo-k15` is the control: full-parameter ZO at the same 30 forwards per step.
- `r1-ft-mezo-k15-b32` is the batch-32 control.
- `r1-ft-adam` is the backprop ceiling.
- The paper scores every member on 128-1024 examples, so batch 32 tests whether batch noise
  causes the stall.

## Proves out if
- From the checkpoint, the best mobilespec arm beats `r1-ft-mezo-k15` at the matching batch on
  final eval loss at equal wall time, with upload of 128 B/step.
- beatCenter stays above 0 throughout.
- From scratch it is expected to stall near the start loss. Any real descent there would be a
  surprise.
