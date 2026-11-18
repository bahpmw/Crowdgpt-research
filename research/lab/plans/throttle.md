# throttle — Robust async SGD via soft throttling (2609.39357)

File: `methods/throttle.py` (shared helper `crowd.corrupt`).

## Faithful vs simplified
- Faithful:
  - Event-driven asynchronous SGD. Each worker computes one backprop
    minibatch gradient (batch 8×128) at the snapshot it was handed.
  - Worker duration = (1 + 0.2·U)/speed. Default speeds are
    `[1,1,1,1,0.5,0.5,0.25,0.25]` (n=8).
  - THROTTLE rule:
    - A worker's first arrival in a round gets weight 1/n and is center-clipped:
      g = a + clip_τ(v − a). The anchor a is the mean of the previous round's
      clipped first arrivals.
    - Its c-th repeat gets weight q^−c and is clipped to λ.
    - A round ends when all n workers have delivered. The server then
      hard-restarts: every worker gets the current model, and in-flight work
      is discarded.
  - The default q is 1.3, between the paper's 1.1 and 1.5.
- Control: `rule=mean` is plain async SGD. Every arrival gets weight 1, with a
  global-norm clip `clip=1` (same as the adam baseline; 0 = none).
- Simplified:
  - Server optimizer: Adam (lr 1e-3) by default, fed weight·g as its gradient.
    `opt=sgd` (mom 0.9) is closer to the paper's plain SGD, but SGD at lr 0.3
    blew up even with no attack.
  - Clipping radii are fixed: τ=2, λ=1. Radii adapted to the median norm fed
    back on themselves and exploded.
  - No G-ball projection of the anchor.
- Attacks: `byz=f` makes workers 0..f−1 Byzantine. Those are the speed-1
  workers in the default list.
  - `flip`: −c·g, c=10 (the paper's NG attack).
  - `noise`: Gaussian noise with norm c·‖g‖.
  - `fast`: flip, plus the attackers run `fastx`=4× faster than the fastest
    honest worker (flooding).
- Logged: `simTime`, `rounds`, `w`, `gradNorm` (raw), `byzShare` (the
  attackers' share of Σ weight·‖g‖ applied), and `wasteFrac` (discarded
  in-flight compute).
- Accounting:
  - One harness step = one arrival.
  - up = 4 B/param per arrival (`bytes=2` rounds to bf16).
  - down = 4 B/param per hand-out, and n× at a restart.
  - Discarded work counts in fwd/bwd by the fraction already done.

## Smoke results (GPU shared with other runs)
- Clean, 12 s: throttle 8.366 → 6.79 (696 arrivals, 34 rounds, about 11% of
  compute wasted at restarts). mean 8.366 → 6.76 (549 arrivals).
- 200 arrivals, `byz=2`:

| attack | rule | eval after 200 arrivals | byzShare | notes |
|---|---|---|---|---|
| flip | mean | 7.39 | 0.38 | |
| flip | throttle q=1.3 | 7.76 | 0.41 | |
| flip | throttle q=3 | 7.43 | 0.39 | |
| fast | mean | 54.9 | 0.71 | diverges |
| fast | throttle q=1.3 | 17.7 | 0.58 | |
| fast | throttle q=3 | 10.8 | 0.44 | |

**Surprise:** THROTTLE reduces flooding damage a lot compared with mean, but
it does not neutralize it in our setting. There are two reasons:
1. Small-batch honest gradients have norm about 0.5–0.7, below λ=1, while
   attackers always sit at the clip radius. Attackers therefore get more
   influence per message than honest workers.
2. Repeat weights q^−c are much larger than 1/n. Whoever is fast holds about
   1/(q−1) of the round's weight.

Expect `lam≈0.5`, `tau≈1` (about the honest norm) and larger q to help.

## Round-1 arms (120 s each)
```
P=../../.venv/bin/python
$P harness.py throttle rule=mean --label r1-th-mean                        # async SGD control, clean
$P harness.py throttle --label r1-th-q13                                   # throttle clean
$P harness.py throttle q=3 lam=0.5 tau=1 --label r1-th-q3-tight            # throttle clean, tight clip
$P harness.py throttle rule=mean byz=2 attack=fast --label r1-th-mean-fast
$P harness.py throttle byz=2 attack=fast --label r1-th-q13-fast
$P harness.py throttle q=3 lam=0.5 tau=1 byz=2 attack=fast --label r1-th-q3-tight-fast
$P harness.py throttle rule=mean byz=2 attack=flip --label r1-th-mean-flip
$P harness.py throttle q=3 lam=0.5 tau=1 byz=2 attack=flip --label r1-th-q3-tight-flip
```
Optional purity check: `opt=sgd lr=0.05` versions of `r1-th-mean` and
`r1-th-q13` (the learning rate is untuned; 0.3 diverged).

## Proves out if
1. Clean: throttle matches or beats `mean` at equal computed gradients (tokens,
   including the waste) and does not lose much in simTime.
2. Under `fast` flooding, throttle stays within a few percent of its clean loss
   while `mean` diverges. Today it limits the damage but does not stay close,
   so a tight-clip arm that does would count.
3. byzShare under throttle stays near the attackers' head-count share (0.25)
   rather than their arrival share (about 0.7).
