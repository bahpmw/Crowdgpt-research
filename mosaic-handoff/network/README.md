DRAFT — requires human review

# CrowdGPT v2 MVP

A small, end-to-end v2 training network that runs on one Mac. Several orchestrators share the
work, so there is no single point of failure. Contributors sign everything they send. A public
beacon assigns the work. Each contributor trains one MOSAIC module (a "cell") locally and uploads
a compressed update. Orchestrators check the uploads by commit-then-sampled-replay, merge them
robustly and deterministically, and write the result to a hash-chained record that they co-sign.
A dashboard shows live metrics.

It is an MVP: correct and tested, not fast or hardened. The build spec is `SPEC.md`. Measured
results are in `runs/BUILD-SUMMARY.md` and `runs/<scenario>/report.{json,md}`.

All commands below run from this directory (`network/`) with a Python that has `../requirements.txt` installed:

```bash
PY=python   # or the path to your virtual environment's python
```

## Tests

```bash
$PY -m pytest -q
```

The suite covers: crypto (sign, verify, tampering); the record (chain checks, tamper detection,
finality counting); beacon assignment and data seeds; cell replay (equal on the same device,
different under each attack); merge determinism and median clipping; the model-gate rollback;
round close, strikes and bans; auditor choice when a peer restarted mid-round; HTTP contract
tests (FastAPI TestClient); and one multi-process
smoke test (3 orchestrators, 3 clients, tiny preset, 2 rounds, final entries). The smoke test
uses ports 8761-8763.

## Simulations

```bash
$PY sim.py scenarios/baseline.json   # 3 orchestrators, 8 honest clients (one on mps), 10 rounds
$PY sim.py scenarios/attacks.json    # 2 honest + 7 attackers (flip, scale, noise, zero, copy, wrongdata, stale)
$PY sim.py scenarios/chaos.json      # 6 honest; leader killed twice, both restarted; 2 clients leave and rejoin
$PY sim.py scenarios/chaos.json --report-only   # rebuild the report from an existing run
```

Each run writes `runs/<name>/`: orchestrator data (`orchN/record.jsonl`, `orchN/blobs/`), logs
(`orchN.log`, `clientN.log`), per-round client metrics (`clientN.jsonl`), and `report.json` /
`report.md`. The report includes per-round held-out loss, final vs provisional entries,
accept/reject counts by behaviour (true and false positives), replay error (genesis calibration
pairs, including cpu vs mps when `calib_devices` is set, plus live honest audits by client
device), bytes up and down per client per round, round wall time, and the `verify_chain` result
for each orchestrator.

Scenario keys: `orchestrators`, `hyper` (see `crowd_v2.nets.Hyper`), `p_audit`,
`calib_devices`, `timing` (seconds), `rounds`, `clients` (`behaviour`, `count`, `device`),
`chaos` (`at_round`, `action`: `kill_leader` | `kill` + `orch` | `restart`), `churn`
(`client`, `leave_after`, `rejoin_at_round`), `base_port`, `max_minutes`.

Each sim uses about 12 processes with 2 torch threads each. With the `10m` preset the runs took 5
to 9 minutes on 2026-11-18 (baseline 497 s, attacks 305 s, chaos 310 s). `runs/chaos-before-fix/`
and `runs/chaos-run2/` keep the reports from the chaos runs made before the auditor fix (see
`runs/BUILD-SUMMARY.md`).

## Dashboard

While a sim runs, open `http://127.0.0.1:8701/dashboard` (any orchestrator port works). The page
is one static HTML file (`dashboard/index.html`) that polls `/v2/status` and listens to
`/v2/events`. It labels which values come from the signed record (verified) and which the
clients report themselves.

## Running nodes by hand

```bash
$PY -m crowd_v2.genesis --out /tmp/net/genesis --orchestrators 3 --hyper '{"preset":"tiny"}'
$PY -m crowd_v2.orchestrator --config node.json      # NodeConfig JSON, see crowd_v2/node.py
$PY -m crowd_v2.client --urls http://127.0.0.1:8701,http://127.0.0.1:8702 --key c0.key
```

`crowd_v2.launch.make_network` writes the genesis and one node config per orchestrator. It is
the easiest way to get a working `node.json`.

## Module map (`crowd_v2/`)

| Module | Role |
|---|---|
| `paths.py` | The only place that knows where the v1.5 model code and token files are |
| `crypto.py` | Ed25519 identities, canonical JSON, signed envelopes, `hash_tensors`, merkle root |
| `wire.py` | safetensors bytes, base64, content ids |
| `blobs.py` | Content-addressed blob store (module weights, payloads) |
| `record.py` | Append-only hash-chained JSONL record, co-signatures, `verify_chain` |
| `beacon.py` | Beacon, cell assignment, data seeds, audit sampling, leader and auditor order |
| `nets.py` | GPT with untied head, MOSAIC module split, local heads, token windows, held-out loss |
| `cell.py` | One training cell (client trains, auditor replays), chunk top-k + 2-bit compression, EF |
| `audit.py` | Layer 0 checks, replay comparison, calibration, layer 3 checks |
| `merge.py` | Median-clipped fixed-point merge, outer SGD step, model gate |
| `rounds.py` | Chain-derived state and `close_round`, the one deterministic path from round inputs to entries |
| `chain.py` | Propose, verify, co-sign, commit, catch up from peers |
| `peers.py` | Signed orchestrator-to-orchestrator HTTP and liveness |
| `node.py` | Orchestrator state and the round loop (open, commit, upload and audit, propose) |
| `schemas.py` | Pydantic models for every network message |
| `api.py` | FastAPI endpoints (`/v2/*`, `/health`, `/metrics`, `/dashboard`) |
| `status.py` | Read-only views for status, metrics and the dashboard |
| `logs.py` | Structured JSON logs |
| `genesis.py` | Keys, calibration, initial weights, signed genesis entry |
| `orchestrator.py` / `client.py` | Command-line entry points (`client.py` also has the attack behaviours) |
| `launch.py` | Start local networks as subprocesses and poll them (used by `sim.py` and the smoke test) |

## MVP-only shortcuts

The code marks each of these with an `mvp:` note.

- One tool creates every orchestrator key and co-signs genesis (`genesis.py`).
- Every node reads the same local token files. In v2 these become content-addressed shards (`paths.py`).
- The beacon is derived from the record, so the last finalizer could grind it. v2 needs a VRF or an external beacon (`beacon.py`).
- Round deadlines assume all nodes share one clock (`node.py`). Reveals go to every orchestrator, not only the assigned auditor.
- Uploads are dense bf16 safetensors in base64 JSON. `wire_bytes` reports the honest SparseLoCo size, which is about 90x smaller for this model (`cell.py`).
- Weights download as full fp32 module blobs each round. There are no deltas.
- Late co-signatures rewrite the record file atomically. Entries themselves never change (`record.py`).
- Token ids are taken modulo the preset vocabulary, so `tiny` can use the 4096-token files (`nets.py`).
- Auditors replay on cpu. cpu-vs-mps error is measured at genesis and in live audits of mps clients.
- There is no TLS, no rate limiting beyond request size limits, and no persistent client reputation outside the record.
