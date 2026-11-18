DRAFT — requires human review

# CrowdGPT v2 MVP — build spec (test day 2026-11-18)

Goal: a small, working, end-to-end v2 network on one Mac that exercises the
roadmap (`ROADMAP-v2.md`) for real: several orchestrators with no single point of
failure, signed contributors, beacon-assigned work, local-module (MOSAIC) training
cells, compressed uploads, commit-then-sampled-replay checking, robust
deterministic merging, a hash-chained signed record with co-signatures, and live
metrics. It is an MVP: correct and tested, not fast or hardened. Name every
shortcut in a module docstring or a `# mvp:` comment.

## Where things are
- Repo root: `/Users/patwoodward/Documents/docker/opencode/projects/Crowd-v1.5`
  (`R` below). Python: `R/.venv/bin/python` (3.12, torch 2.15 with MPS,
  fastapi, uvicorn, httpx, pydantic 2, numpy, pytest, safetensors,
  `cryptography` 46.0.3 for Ed25519). Do not add other dependencies.
- MVP code: `R/v2-mvp/` — package `R/v2-mvp/crowd_v2/`, tests `R/v2-mvp/tests/`,
  runnable scripts under `R/v2-mvp/` (`sim.py`, dashboard under
  `R/v2-mvp/dashboard/`), outputs under `R/v2-mvp/runs/` (create it).
- Model: `R/archived/crowd-v1.5/crowd_v15/model.py` (`GPT`, `ModelConfig.preset`,
  presets `tiny` 1.08M, `10m` 7.34M: vocab 4096, 6 layers, d 256, ctx 512, tied
  tok_emb/head, RMSNorm, SwiGLU, blocks called as `block(h, mask, None, name)`).
  Import it by putting `R/archived/crowd-v1.5` on `sys.path` in one place
  (`crowd_v2/__init__.py` or a `paths.py`).
- Token data: `R/archived/crowd-v1.5/runs/data/tok/{train,val}.bin` (uint16,
  40M / 2M tokens, vocab 4096). MVP shortcut: every node reads the same local
  file; in v2 these become content-addressed shards.
- Reference training code to adapt (copy what you need, do not import from the
  lab): `R/research/lab/methods/mosaic.py` (module split, `_local_loss`, MOSAIC
  inner loop, median clipping, outer step), `methods/solo.py` (`LocalHead`),
  `methods/sparseloco.py` (`compress`, CHUNK=64, top-k per 64x64 chunk + 2-bit),
  `crowd.py` (`corrupt`). Lab defaults that worked: K=2, H=30 inner AdamW steps
  lr 3e-3 betas (0.9, 0.95), clip 1.0, topk=64, error feedback beta 0.95, outer
  SGD alpha 1.0 (or Nesterov olr 0.7 mom 0.9), median-norm clipping.
- Prior v1 schemas worth borrowing ideas from (read only):
  `R/CrowdGPT-Server/orchestrator/{records,receipts,identity}.py`.

## Design (keep it this simple)

### Identities and signing (`crowd_v2/crypto.py`)
- Ed25519 keys (`cryptography`). Key id = first 16 hex of sha256(public key).
- Canonical JSON: `json.dumps(obj, sort_keys=True, separators=(",", ":"))`, UTF-8.
- Every client and orchestrator message that changes state is signed; identity
  comes from the verified signature, never from a payload field.
- Tensor hash: sha256 over the exact fp32 little-endian bytes of tensors in
  sorted-name order (helper `hash_tensors(dict) -> hex`).

### Record (`crowd_v2/record.py`)
- Append-only JSONL, one entry per line: `{"n", "type", "round", "prev", "body",
  "hash", "sigs": {key_id: sig_hex}}`; `hash` = sha256(canonical json of
  `{n,type,round,prev,body}`); `prev` = previous entry hash (genesis: 64 zeros).
- Entry types: `genesis` (config hash, orchestrator public keys, quorum, model
  preset, K, initial weight hash per module), `register` (client public key),
  `round` (round, beacon, leader, accepted cells with signer key ids + payload
  hashes, audit verdicts, per-module new weight hashes, eval loss, tokens),
  `verdict` (rejections/strikes/bans), `rollback`, `orchestrator_down/up`.
- An entry is **final** when it carries signatures from >= quorum orchestrators
  (quorum = 2 of 3 by default) and **provisional** with fewer.
- `verify_chain(path, genesis_keys)` checks hashes, links, signatures, and
  reports final/provisional counts. A test must show tampering is detected.

### Beacon and assignment (`crowd_v2/beacon.py`)
- `beacon_r = sha256("beacon" || hash of the last final entry before round r)`.
  mvp: record-derived beacon; note the leader-grinding risk in a docstring.
- Cells: a round has `K` modules; registered, non-banned clients that asked for
  work by the open deadline are sorted by sha256(beacon || key id) and dealt
  round-robin onto modules so every module gets coverage.
- Data: each cell's batches come from a numpy Generator seeded with
  int(sha256(beacon || key id || "data")) — so neither the client nor the
  orchestrator chooses the data, and anyone can recompute it.
- Audit sample: after commits close, `audit_seed = sha256(beacon || merkle root of
  the round's commits)`; sample fraction `p_audit` (default 0.5 for the MVP,
  1.0 for clients still on probation, i.e. their first 3 rounds) of cells for
  replay. Contributors cannot know which cells are checked before committing.

### Training cell (`crowd_v2/cell.py`) — shared by client and auditor
- Input: round weights (all modules; untied head), K, module k, data seed, H,
  inner hyper-parameters, device. Fresh AdamW every round (so a cell is
  replayable from public inputs). MOSAIC: run earlier modules forward at round
  weights (no activations sent between peers), train module k + its local head
  (k < K-1) or norm_f + head (last module) with W_bar = round head (frozen).
- Output: dense delta for module k (round weights minus trained weights, like
  mosaic.py), then compressed message with error feedback (client keeps the EF
  buffer across rounds for that module; if the client's module changes, its EF
  for the old module is dropped — note this in the code).
- Wire payload: safetensors bytes of the compressed message as sparse
  (indices + 2-bit codes + per-chunk scale) or, mvp-acceptable, the
  decompressed dense tensors in bf16 plus an honest `wire_bytes` count computed
  with the SparseLoCo formula. Pick one and say which.
- Must run on cpu (default) and mps.

### Checking (`crowd_v2/audit.py`)
- Layer 0: signature, round, assignment (module, data seed), shapes, finite
  values, size limits.
- Layer 1: commit first. Client sends a signed commit `{round, cell, payload_hash,
  ef_hash, tokens}` before the commit deadline, then uploads the payload whose
  hash must match.
- Layer 2: sampled replay. For sampled cells the client must reveal its EF
  buffer (hash must match `ef_hash`); the auditor reruns the cell on its own
  device from public inputs and compares the replayed compressed message with
  the submitted one (cosine of decompressed vectors and norm ratio). Thresholds
  come from a calibration function that measures honest replay error (same
  device and cpu-vs-mps) — record the measured numbers in the run report.
- Each sampled cell is replayed by an orchestrator other than the one that
  received it (chosen by the audit seed). Verdicts are signed.
- Failed check → submission rejected, `verdict` entry, one strike; 3 strikes →
  ban (key cannot be assigned work).
- Layer 3 (cheap, unsampled): reject messages whose norm exceeds 4x the module
  median, or whose cosine to the module's mean direction is strongly negative.

### Merge (`crowd_v2/merge.py`)
- Per module: decompress accepted messages, clip each to the median norm, mean
  with equal weights (cells are equal-token work units), outer step
  (SGD alpha 1.0 default; Nesterov optional — outer momentum state is part of
  the state hash).
- Deterministic across machines: convert each clipped message to int64 fixed
  point (`round(x * 2**24)`), sum integers, divide once, all on cpu, in
  sorted payload-hash order. New weights hash goes into the round entry.
- Model gate: eval loss on a fixed held-out batch (val.bin, seed 12345 windows,
  like the lab harness); if loss rises by more than `gate_tol` (default 0.25)
  over the previous round, write a `rollback` entry and keep old weights.

### Orchestrator (`crowd_v2/orchestrator.py`, FastAPI, `python -m crowd_v2.orchestrator`)
- Config: port, own key file, peers (URLs + public keys), genesis file, data
  path, device, round timing (open, commit and upload deadlines in seconds;
  defaults suitable for a laptop sim), `p_audit`.
- Endpoints (JSON, pydantic models, explicit `v` schema version):
  `POST /v2/register`, `GET /v2/round` (round, beacon, leader, deadlines,
  module weight hashes), `POST /v2/work` (signed; returns assignment),
  `GET /v2/blob/{sha256}` (content-addressed weights), `POST /v2/commit`,
  `POST /v2/upload`, `GET /v2/audit/{key_id}` (pending reveal requests),
  `POST /v2/reveal`, `GET /v2/record?since=n`, `GET /v2/status` (per contributor
  and per orchestrator, value states live/provisional/final/stale),
  `GET /v2/events` (Server-Sent Events: round_open, commit, upload, audit,
  verdict, round_final, rollback, orchestrator_down/up), `GET /health`
  (subsystem status), `GET /metrics` (Prometheus text). Peer endpoints:
  `POST /v2/peer/submission` (forwarded signed commit+payload),
  `POST /v2/peer/propose`, `POST /v2/peer/cosign`, `POST /v2/peer/verdict`.
- Federation (3 orchestrators by default): a client may talk to any orchestrator;
  the receiver verifies and forwards commits/payloads to peers. Leader for round
  r = live orchestrators sorted by sha256(beacon || orchestrator key id), first.
  At the upload deadline the leader proposes `{round, accepted payload hashes,
  verdicts}`; each peer fetches anything missing, recomputes the merge itself
  (deterministic), checks the weight hash, and co-signs. Entry final at quorum.
  If the leader is silent past a timeout, the next orchestrator in order
  proposes. A lone orchestrator keeps training with provisional entries; a
  restarted orchestrator catches up by pulling the record and blobs from a peer,
  verifying every entry and re-running merges before co-signing.
- Structured JSON logs with round and correlation ids; no IPs or secrets logged.

### Client (`crowd_v2/client.py`, `python -m crowd_v2.client`)
- Args: orchestrator URLs (failover list), key file (created if missing), device,
  behaviour (`honest` default; attack modes for the sim: `flip`, `scale`,
  `noise`, `zero` (lazy/near-zero update), `copy` (re-sends another client's
  last payload, fetched from the record/blob store), `wrongdata` (trains on data
  it chose), `stale` (trains from last round's weights)), `--rounds`, churn
  options (leave after n rounds / rejoin).
- Loop: register → get round → request work → download needed module blobs
  (verify hashes) → train cell → commit → upload → answer reveal requests → next.
  Failover to the next orchestrator on connection errors.

### Simulation (`v2-mvp/sim.py`)
- Starts N orchestrators and M clients as subprocesses on localhost
  (ports 8701+), with a scenario file (JSON) for behaviours, churn and chaos
  (kill orchestrator i at round a, restart at round b). Torch threads per
  process limited (e.g. 2) so 3 orchestrators + 8 clients fit on 18 cores.
- Writes `runs/<scenario>/report.json`: per round eval loss, accepted/rejected by
  behaviour (true/false positives), audit replay error stats, bytes up/down per
  client per round, round wall time, final vs provisional entries, chain verify
  result.

### Dashboard (`v2-mvp/dashboard/index.html`)
- One static HTML file, no build step, plain JS: polls `/v2/status` and listens to
  `/v2/events`; shows network overview (round, finality, loss curve as inline
  SVG, orchestrators up/down), contributor table (key id, module, accepted,
  rejected, strikes, last seen, state), live event feed. Label verified vs
  self-reported values. Semantic HTML, keyboard accessible.

## Tests (pytest, `R/.venv/bin/python -m pytest v2-mvp/tests -q`)
Required: crypto sign/verify/tamper; record chain verify + tamper detection +
finality counting; beacon assignment determinism and coverage; data seed
reproducibility; cell replay equals original on same device (exact or within
tolerance) and differs under each attack; merge determinism (two runs, shuffled
input order → identical weight hash) and median clipping bounds a huge update;
model gate rollback; orchestrator HTTP contract tests with FastAPI TestClient;
one multi-process smoke (3 orchestrators, 3 clients, tiny preset, 2 rounds) that
reaches final entries. Tests must not depend on order or use sleep for sync
(poll with a deadline instead).

## Rules
- Do not touch anything outside `R/v2-mvp/` except reading. No git commands.
- No dead code, no stubs, no bare except, no silent failures; validate all
  network input with pydantic and size limits.
- Keep files short and readable; prefer one obvious way.
