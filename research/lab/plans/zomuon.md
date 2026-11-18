# zomuon: ZO-Muon, subspace RGE + msign (2602.17155)

## Faithful
- Per 2-D matrix (m x n, stored [out, in]): orthonormal P (m x r) from QR of a seeded Gaussian,
  redrawn every `refresh` = 100 steps.
- nq probes perturb all matrices at once by mu P Psi_i. These are one-sided forward differences
  against one shared clean loss: nq+1 forwards per step.
- C = (1/nq) sum_i s_i Psi_i, then W -= eta * P msign(C). msign uses the quintic Muon
  Newton-Schulz (5 steps, (3.4445, -4.7750, 2.0315)). No momentum.
- `ortho=false`: W -= eta * P C, the Subspace-MeZO control.

## Simplified / chosen
- tok_emb is treated as a [vocab x d] matrix: the head's [out, in], with P on the vocab side
  (4096 x 64).
- RMSNorm gains are frozen. The paper gives them a MeZO update with no stated lr.
- `scale=muon` (default) multiplies by sqrt(max(1, m/n)): qkv x1.73, up/gate x2, tok_emb x4.
  `scale=none` gives x1.
- Upload is C per matrix in bf16: 1.61 MB/step at r=64. This is the spec's accounting; a
  seed-replay worker could send only nq scalars.
- `estimate()` returns P C (not orthogonalised).

## Measured in smoke (10 s unless noted; shared GPU)
| arm | result |
|---|---|
| eta=0.01, 15 s | 8.366 -> 8.338 (270 steps) |
| eta=0.03 | -> 8.325 |
| eta=0.05 | -> 8.311 |
| eta=0.1 | 8.367 (too large) |
| eta=0.03 scale=none | -> 8.331 |
| ortho=false eta=1e-4 / 3e-4 / 1e-3 / 3e-3 | 8.366 / 8.363 / 8.356 / 8.335 |
| ortho=false eta=0.01 | -> 8.282 |
| ortho=false eta=0.03 | -> 8.300 |

In smoke the plain subspace control beats msign.

probe on ckpt (CPU, 3 repeats): nq=8 gives 6e-4 (9 fwd), nq=16 gives 8e-4 (17 fwd), mezo k=4
gives -1e-4 (8 fwd). The cosine is capped near sqrt(r/m) per matrix by design.

## Round-1 arms (120 s each)
```
P=../../.venv/bin/python
$P harness.py zomuon r=64 nq=8 mu=0.001 eta=0.03                --label r1-zomuon-e03
$P harness.py zomuon r=64 nq=8 mu=0.001 eta=0.05                --label r1-zomuon-e05
$P harness.py zomuon r=64 nq=8 mu=0.001 eta=0.01 ortho=false    --label r1-subzo-e01
$P harness.py zomuon r=64 nq=8 mu=0.001 eta=0.03 ortho=false    --label r1-subzo-e03
$P harness.py mezo k=4 eps=0.001 lr=0.0001                      --label r1-mezo-k4
```
`r1-mezo-k4` runs at 8 forwards per step vs 9 for the subspace arms.

Optional fine-tune pair: add `--init ckpt/adam30.pt` to the best ortho and best plain arm.

## Proves out if
- The best `zomuon` (msign) arm beats the best `ortho=false` arm on final loss. That is the
  paper's central claim (msign helps once the estimate is low-dimensional and nq > 1). Both
  should beat MeZO at equal forwards.
- msign arms never diverge across eta in {0.03, 0.05}, given the bounded step norm.
- If plain beats msign again at 120 s, as in smoke, that counts as a negative result for
  pretraining from scratch.
