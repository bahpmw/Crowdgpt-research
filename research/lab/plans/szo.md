# szo — SZO-κ activation-space ZO (2611.08712)

Files: `methods/szo.py` (method), `actzo.py` (estimator engine, fixed),
`szo_probe.py` (probe.py with an MPS-safe cosine).

## Faithful vs simplified
- Faithful: per-sample (per-sequence) credit; one-sided probes against one clean
  pass; iterative probing (one layer per forward); NP control, per-sample NTWP
  (Alg. 2 normaliser k/‖ΔH‖²), SZO-1 via App. I (`Ψ_b = G_b X_b/‖X_b‖`,
  realised budget `c = ‖Ψ‖²`, Ψ never materialised); App. G activation
  awareness (node credit for np/ntwp, per-row probe-energy-weighted slope² for
  szo1); optional adaptive per-tensor clip (5× lower median of last 100, after
  50 warm-up steps).
- Simplified: GPT instead of MLP/CNN (tokens play the role of conv patches, so
  ntwp is not unbiased here: it carries the per-sample C_b bias); relative
  eps 1e-2 instead of absolute 1e-4 (MPS fp32); SGD momentum 0.9 + global clip
  1.0 (DUST's outer optimizer) instead of plain SGD; RMSNorm weights frozen;
  exact head gradient + NP embedding term; slope² normalised to mean 1 (SiLU).
- actzo fixes: `aware` was silently ignored for szo1; szo1 readout lacked the
  `1/‖Ψ‖²` normaliser and `k` factor (now on the same LR scale as np/ntwp);
  probes are processed in chunks (memory bound at large n); docstring no longer
  claims token-level ntwp is unbiased.

## Probe (block-weight cosine to backprop, 1–2 repeats, batch 8×128)
Forwards are full-model passes over the batch. Real DUST forwards are
`reported × chunk`: pop=8 chunk=8 = 504, pop=32 chunk=16 = 2016 (see note).

| arm | fwd | init | ckpt adam30 |
|---|---|---|---|
| szo np n=16 | 497 | 0.028 | 0.020 |
| szo ntwp n=16 | 497 | 0.139 | 0.014 |
| szo szo1 n=16 | 497 | 0.131 | 0.029 |
| dust pop=8 chunk=8 | 504 | 0.034 | 0.005 |
| szo np n=64 | 1985 | 0.058 | 0.022 |
| szo ntwp n=64 | 1985 | 0.268 | 0.041 |
| szo szo1 n=64 | 1985 | 0.254 | 0.052 |
| szo szo1 n=64 aware | 1985 | 0.249 | 0.072 |
| szo ntwp n=64 aware | 1985 | — | 0.071 |
| dust pop=32 chunk=16 | 2016 | 0.067 | 0.035 |

At init SZO-1/NTWP give ~4–5× NP/DUST cosine at equal forwards (the paper's
claim holds strongly). From the Adam checkpoint the gap shrinks to ~1.5–2.5×
(szo1) and ~2× with awareness, consistent with the paper's caveat that the
isotropic-V prior is an init argument. All absolute cosines are small.

## Round-1 arms (120 s each)
```
../../.venv/bin/python harness.py dust pop=8 chunk=8 lr=0.05 --label r1-szo-dust8   # = base-dust8
../../.venv/bin/python harness.py szo kind=np   n=16 --label r1-szo-np16
../../.venv/bin/python harness.py szo kind=ntwp n=16 --label r1-szo-ntwp16
../../.venv/bin/python harness.py szo kind=szo1 n=16 --label r1-szo-szo1-16
../../.venv/bin/python harness.py szo kind=szo1 n=16 aware=true --label r1-szo-szo1-16a
../../.venv/bin/python harness.py szo kind=szo1 n=4 aware=true --label r1-szo-szo1-4a
../../.venv/bin/python harness.py szo kind=szo1 n=16 aware=true adaclip=5 clip=0 --label r1-szo-szo1-16a-ada
```
Defaults: eps=0.01 relative, lr=0.05, momentum 0.9, clip 1.0. n=16 matches
DUST pop=8 forwards (497 vs 504). n=4 tests the paper's "few probes, more
steps" rule. Optional cosine table:
`../../.venv/bin/python szo_probe.py "szo kind=np n=16" "szo kind=szo1 n=16" "szo kind=szo1 n=16 aware=true" "dust pop=8 chunk=8" --repeats 5 [--ckpt ckpt/adam30.pt]`.

## Proves out if
- Probe: szo1/ntwp cosine ≥ 1.5–2× np and DUST at matched forwards, at init and
  from the checkpoint (met at init, borderline at the checkpoint without awareness).
- Training: szo1 (or szo1 aware) eval loss at 120 s is below both np n=16 and
  DUST pop=8 by more than seed noise (rerun the best two with `--seed 1`).

## Notes for the coordinator
- `probe.py` crashes on MPS (`.double()` before `.cpu()` in `block_cosine`);
  `szo_probe.py` is the same CLI with the order swapped.
- `methods/dust.py` reports `fwd = passes × tokens`, but each pass runs
  `chunk` replicas, so DUST compute is under-counted by `chunk`× (8× for
  base-dust8, 16× for base-dust32). Compare arms on wall time, or rescale.
