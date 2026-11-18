DRAFT — requires human review

# Scenario `baseline`

3 orchestrators, 8 clients, hyper `{"preset": "10m", "seq": 128, "H": 30, "batch": 8, "K": 2}`, p_audit 0.5.

Held-out loss: 8.366322 at genesis -> 7.008598 after 10 rounds. Finality: 10 final, 0 provisional round entries (as held by the most complete orchestrator at stop time).

## Rounds

| round | eval loss | rolled back | leader | sigs | final | cells | sampled | statuses | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.930899 | False | orch0 | 3 | True | 8 | 8 | accepted:8 | 65.83 |
| 2 | 7.666419 | False | orch0 | 3 | True | 8 | 8 | accepted:8 | 53.72 |
| 3 | 7.530239 | False | orch2 | 3 | True | 8 | 8 | accepted:8 | 50.69 |
| 4 | 7.389488 | False | orch2 | 3 | True | 8 | 6 | accepted:8 | 42.91 |
| 5 | 7.355344 | False | orch2 | 3 | True | 8 | 4 | accepted:8 | 36.41 |
| 6 | 7.309073 | False | orch1 | 3 | True | 8 | 4 | accepted:8 | 31.68 |
| 7 | 7.232473 | False | orch1 | 3 | True | 8 | 5 | accepted:8 | 40.15 |
| 8 | 7.197921 | False | orch0 | 3 | True | 8 | 6 | accepted:8 | 39.84 |
| 9 | 7.061529 | False | orch0 | 3 | True | 8 | 6 | accepted:8 | 39.95 |
| 10 | 7.008598 | False | orch0 | 3 | True | 8 | 3 | accepted:8 | 30.61 |

## Detection by behaviour

| behaviour | cells | accepted | rejected (strike) | excluded (no strike) | reasons |
|---|---|---|---|---|---|
| honest | 80 | 80 | 0 | 0 | {} |

Bans: {}

## Replay error of honest cells (live audits)

Thresholds {"cos_min": 0.9799, "ratio_hi": 1.0501, "ratio_lo": 0.9499}.

- client cpu vs auditor cpu: {"cos": {"n": 52, "min": 1.0, "mean": 1.0, "max": 1.0}, "ratio": {"n": 52, "min": 1.0, "mean": 1.0, "max": 1.0}, "failed": 0}
- client mps vs auditor cpu: {"cos": {"n": 6, "min": 0.985484, "mean": 0.996472, "max": 1.0}, "ratio": {"n": 6, "min": 0.999868, "mean": 1.000039, "max": 1.000419}, "failed": 0}

Genesis calibration pairs: [{"cos": 1.0, "module": 0, "pair": "cpu-same", "ratio": 1.0}, {"cos": 1.0, "module": 0, "pair": "cpu-threads", "ratio": 1.0}, {"cos": 0.999873, "module": 0, "pair": "cpu-mps", "ratio": 0.999933}, {"cos": 1.0, "module": 1, "pair": "cpu-same", "ratio": 1.0}, {"cos": 1.0, "module": 1, "pair": "cpu-threads", "ratio": 1.0}, {"cos": 0.999957, "module": 1, "pair": "cpu-mps", "ratio": 1.000002}]

## Traffic per client per round (bytes)

- up (actual HTTP, dense bf16 + base64 JSON): {"n": 80, "min": 11196168, "mean": 20428159.4375, "max": 41962617}
- down (actual HTTP, fp32 weights): {"n": 80, "min": 37772849, "mean": 37838518.675, "max": 37889142}
- SparseLoCo wire formula for the same message: {"n": 80, "min": 122368, "mean": 137729.0, "max": 153090}

Round wall time (s): {"n": 10, "min": 30.61, "mean": 43.179, "max": 65.83}

## Chain verification

- orch0: {"ok": true, "entries": 19, "final": 19, "provisional": 0, "errors": []}
- orch1: {"ok": true, "entries": 19, "final": 19, "provisional": 0, "errors": []}
- orch2: {"ok": true, "entries": 19, "final": 19, "provisional": 0, "errors": []}

## Timeline

- {"t": 42.5, "round": 0, "event": "orchestrators_up"}
- {"t": 42.8, "round": 0, "event": "clients_started", "n": 8}
- {"t": 110.8, "round": 1, "event": "round_recorded"}
- {"t": 167.3, "round": 2, "event": "round_recorded"}
- {"t": 220.2, "round": 3, "event": "round_recorded"}
- {"t": 265.6, "round": 4, "event": "round_recorded"}
- {"t": 304.0, "round": 5, "event": "round_recorded"}
- {"t": 338.1, "round": 6, "event": "round_recorded"}
- {"t": 380.2, "round": 7, "event": "round_recorded"}
- {"t": 422.5, "round": 8, "event": "round_recorded"}
- {"t": 464.3, "round": 9, "event": "round_recorded"}
- {"t": 497.2, "round": 10, "event": "stopping", "reached": 10, "timed_out": false, "converged": true}
