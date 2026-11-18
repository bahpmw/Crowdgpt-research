# wismezo: WIS-MeZO / WISD-MeZO (2611.16317)

## Faithful
- Wanda score S_ij = |W_ij| * ||X_j||_2 from calibration activations of each linear's input.
  The inputs are recorded via `Perturb(record={})`.
- Row-wise top-floor(r*n) binary mask M. Direction U = Z*M/sqrt(r). Two-point SPSA on one batch.
  Update W -= lr * (L+ - L-)/(2 eps) * U. Constant lr, no momentum, no weight decay.
- WISD refresh every `tau` steps. `tau=0` gives a one-time mask, i.e. plain WIS.
- Seed replay: upload is (seed, scalar) = 12 B per direction. The mask is recomputable from (W, public batch).
- `mask=random` gives the paper's Random-MeZO control: same r, random seeded support.

## Simplified / chosen
- k directions per step averaged (default 4; the paper uses 1). Compare to `mezo` at the same k.
- Tied tok_emb/head is masked in its head view: rows = vocab, inputs = final hidden state.
  `emb=dense` leaves it dense. RMSNorm gains get dense noise.
- Calibration uses `cal=32` fresh training sequences (paper: 128 task examples).
  Its tokens and forward are counted.

## Measured in smoke (shared GPU, so step counts are noisy)
- Scratch, 15 s: `wismezo` reached 8.366 -> 8.328 in 491 steps. `mezo k=4 lr=1e-4` reached
  8.366 -> 8.350 in 306 steps (8.341 at step 332 for WIS). Any edge is within noise.
- Energy capture rho = ||M*G||^2/||G||^2 on blocks:
  - At random init, rho is about r (0.105 at r=0.1, 0.052 at r=0.05). Wanda equals random at init, as the notes predicted.
  - On `ckpt/adam30.pt`, rho is 0.145 / 0.200 / 0.342 at r = 0.05 / 0.1 / 0.2, i.e. 2.9x / 2.0x / 1.7x r.
    The predicted alignment gain sqrt(rho/r) is about 1.7x at r=0.05.
- probe (CPU; probe.py cannot run on MPS, see report), ckpt, 3 repeats:
  - k=4: all arms are at the noise floor (+-4e-4).
  - k=64, 2 repeats: mezo 0.0008, wis r=0.1 0.0013, wis r=0.05 0.0009. Not resolvable without about 20 repeats.

## Round-1 arms (120 s each)
```
P=../../.venv/bin/python
# from scratch
$P harness.py mezo    k=4 eps=0.001 lr=0.0001                     --label r1-mezo-k4
$P harness.py wismezo k=4 eps=0.001 lr=0.0001 r=0.1 tau=100        --label r1-wis-r10
$P harness.py wismezo k=4 eps=0.001 lr=0.0001 r=0.05 tau=100       --label r1-wis-r05
$P harness.py wismezo k=4 eps=0.001 lr=0.0001 r=0.1 tau=100 mask=random --label r1-wisrand-r10
# paper regime: fine-tune from the Adam checkpoint
$P harness.py mezo    k=4 eps=0.001 lr=0.0001 --init ckpt/adam30.pt                --label r1-ft-mezo-k4
$P harness.py wismezo k=4 eps=0.001 lr=0.0001 r=0.05 tau=100 --init ckpt/adam30.pt --label r1-ft-wis-r05
$P harness.py wismezo k=4 eps=0.001 lr=0.0001 r=0.05 tau=100 mask=random --init ckpt/adam30.pt --label r1-ft-wisrand-r05
```
Optional lr probe: repeat the `r1-ft-*` mezo and wis arms at `lr=0.0003`.

## Proves out if
- Fine-tune: `r1-ft-wis-r05` beats both `r1-ft-mezo-k4` and `r1-ft-wisrand-r05` on final eval loss
  by more than the run-to-run spread (about 0.005 at this scale; re-seed once if close).
- From scratch, we expect no gain (rho is about r at init). A gain there would be a surprise worth re-checking.
- For the crowd, the bar is a loss win at an identical 48 B/step upload. A tie is a null result.
