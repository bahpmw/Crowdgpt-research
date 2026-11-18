# Lab spec for MVP methods (read fully before writing code)

Goal: one quick, faithful MVP per paper so we can see what proves out for a
crowd (tens of thousands of consumer GPUs, slow uplinks, untrusted peers)
training a GPT from scratch. MVP = the paper's core mechanism on our model, not
a full reproduction. Name simplifications in a short module docstring.

## Where things are
- Repo root: `/Users/patwoodward/Documents/docker/opencode/projects/Crowd-v1.5`
- Lab (gitignored): `research/lab/`. Run everything from there with
  `../../.venv/bin/python`.
- Papers: `research/papers/<id>.pdf` and `.txt` (txt has control bytes; strip with
  `tr -d '\000-\010\013\014\016-\037'` into the temp dir before reading).
  Notes with MVP recipes: `research/papers/notes/<id>.md` (read these first).
- Model: `crowd_v15/model.py` (`GPT`, `ModelConfig.preset("10m")`: vocab 4096,
  6 layers, d 256, 8 heads, ctx 512, tied tok_emb/head, RMSNorm, SwiGLU MLP,
  `Perturb` hook object that can record layer inputs and add output noise).
- Repo DUST: `crowd_v15/dust.py` (`dust_gradients`, `DustConfig`, `_param_map`).
- Harness: `research/lab/harness.py` (read it). Shared helpers: `zo.py`
  (seeded weight-space perturbation, `spsa_pair`, `trainable`), `actzo.py`
  (activation-space ZO engine with np/ntwp/szo1 kinds, `backprop`, `cosine`).
- Existing methods to copy style from: `methods/adam.py`, `methods/dust.py`,
  `methods/mezo.py`.

## Method interface
`research/lab/methods/<name>.py` defines `class Method` with
`__init__(self, model, data, cfg: dict, device)` and `step(self, it) -> Cost`.
- `data.next(batch=None, worker=0, workers=1, noniid=False)` returns
  `(idx, tgt)` int64 on device, shape (batch, seq). Default batch 8, seq 128.
- `Cost(tokens, fwd, bwd, up, down, extra)`: tokens = training tokens consumed
  this step (all simulated workers); fwd = forward token-passes; bwd = backward
  token-passes; up/down = bytes per step a crowd worker set would send/receive
  over the internet (sum over simulated workers); extra = small dict of scalars
  to log (e.g. gradNorm, lr, cosine).
- Read config with `cfg.get("key", default)`; values arrive JSON-parsed.
- Estimator-style methods (anything that forms a gradient estimate per step)
  must also expose `estimate(self, idx, tgt, it) -> (grads: dict, forwards: int)`
  with grads keyed like `zo.trainable(model)` (named_parameters, dedup; the
  tied weight is `tok_emb.weight`) so `probe.py` can measure cosine to
  backprop on the 2-D block weights at matched forwards.
- Methods may keep their own extra models/state (e.g. local heads, worker
  replicas). Evaluation always uses `harness.eval_loss(model, ...)` on the
  `model` passed in, so whatever you train must end up in that model (for
  distributed methods: the global/consensus model; for SOLO: the full stack).
- No backprop in ZO arms (wrap in `torch.no_grad()`), except where the paper
  itself uses it (LoCA calibration probe, SOLO/DiLoCo local backprop, EoS
  diagnostics that are clearly labelled as diagnostics).
- Use `torch.Generator(device=...)` seeded per (step, worker, probe) so a
  coordinator could replay noise from a seed.

## Running
- Smoke test only: `../../.venv/bin/python harness.py <name> k=v ... --seconds 15 --label smoke-<name> --eval-every 5`
  The GPU is shared with other runs, so keep smoke tests <= 20 s and do not
  launch long runs; the coordinator will run the real 120 s benchmarks
  sequentially later. Also smoke `probe.py "<name> k=v" --repeats 1` for
  estimator methods.
- Make sure loss goes down (or explain why not) and nothing errors.
- Do not edit `harness.py`, `zo.py`, `actzo.py`, `probe.py` or other agents'
  methods. If you need a shared helper, put it in your own module under
  `research/lab/` with a distinct name.
- Do not touch anything outside `research/lab/` and do not git add/commit.

## Deliverable per method
1. `methods/<name>.py` working.
2. Write `research/lab/plans/<name>.md`: method name, paper id, what is faithful vs simplified, the exact
   arm commands (harness configs, 120 s each) the coordinator should run for
   round 1 including the baseline/control arm(s) for this paper, and what
   result would count as "proves out".
3. Report back briefly: files, smoke results, recommended arms.
