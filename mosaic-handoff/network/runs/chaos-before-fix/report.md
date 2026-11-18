DRAFT — requires human review

# Scenario `chaos`

3 orchestrators, 6 clients, hyper `{"preset": "10m", "seq": 128, "H": 30, "batch": 8, "K": 2}`, p_audit 0.5.

Held-out loss: 8.366322 at genesis -> 7.143219 after 10 rounds. Finality: 10 final, 0 provisional round entries (as held by the most complete orchestrator at stop time).

## Rounds

| round | eval loss | rolled back | leader | sigs | final | cells | sampled | statuses | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.935205 | False | orch1 | 3 | True | 6 | 6 | accepted:6 | 39.17 |
| 2 | 7.7074 | False | orch0 | 3 | True | 6 | 6 | accepted:6 | 28.11 |
| 3 | 7.632754 | False | orch0 | 3 | True | 5 | 5 | accepted:5 | 29.05 |
| 4 | 7.531731 | False | orch2 | 3 | True | 5 | 2 | accepted:5 | 26.85 |
| 5 | 7.403761 | False | orch2 | 3 | True | 4 | 2 | accepted:4 | 23.21 |
| 6 | 7.381507 | False | orch0 | 3 | True | 5 | 3 | accepted:5 | 35.92 |
| 7 | 7.328093 | False | orch0 | 3 | True | 5 | 2 | accepted:5 | 34.59 |
| 8 | 7.257438 | False | orch0 | 3 | True | 5 | 1 | accepted:4, missing_audit:1 | 320.06 |
| 9 | 7.198651 | False | orch1 | 3 | True | 6 | 3 | accepted:5, missing_audit:1 | 312.11 |
| 10 | 7.143219 | False | orch0 | 3 | True | 6 | 2 | accepted:6 | 28.63 |

## Detection by behaviour

| behaviour | cells | accepted | rejected (strike) | excluded (no strike) | reasons |
|---|---|---|---|---|---|
| honest | 53 | 51 | 0 | 2 | {"missing_audit": 2} |

Bans: {}

## Replay error of honest cells (live audits)

Thresholds {"cos_min": 0.9799, "ratio_hi": 1.0501, "ratio_lo": 0.9499}.

- client cpu vs auditor cpu: {"cos": {"n": 30, "min": 1.0, "mean": 1.0, "max": 1.0}, "ratio": {"n": 30, "min": 1.0, "mean": 1.0, "max": 1.0}, "failed": 0}

Genesis calibration pairs: [{"cos": 1.0, "module": 0, "pair": "cpu-same", "ratio": 1.0}, {"cos": 1.0, "module": 0, "pair": "cpu-threads", "ratio": 1.0}, {"cos": 0.999881, "module": 0, "pair": "cpu-mps", "ratio": 0.999916}, {"cos": 1.0, "module": 1, "pair": "cpu-same", "ratio": 1.0}, {"cos": 1.0, "module": 1, "pair": "cpu-threads", "ratio": 1.0}, {"cos": 0.999957, "module": 1, "pair": "cpu-mps", "ratio": 1.000002}]

## Traffic per client per round (bytes)

- up (actual HTTP, dense bf16 + base64 JSON): {"n": 53, "min": 11194555, "mean": 17689594.622642, "max": 41962640}
- down (actual HTTP, fp32 weights): {"n": 53, "min": 37772012, "mean": 37987037.811321, "max": 38665034}
- SparseLoCo wire formula for the same message: {"n": 53, "min": 122368, "mean": 139178.150943, "max": 153090}

Round wall time (s): {"n": 10, "min": 23.21, "mean": 87.77, "max": 320.06}

## Chain verification

- orch0: {"ok": true, "entries": 21, "final": 21, "provisional": 0, "errors": []}
- orch1: {"ok": true, "entries": 21, "final": 21, "provisional": 0, "errors": []}
- orch2: {"ok": true, "entries": 21, "final": 21, "provisional": 0, "errors": []}

## Timeline

- {"t": 30.6, "round": 0, "event": "orchestrators_up"}
- {"t": 30.8, "round": 0, "event": "clients_started", "n": 6}
- {"t": 71.5, "round": 1, "event": "round_recorded"}
- {"t": 101.1, "round": 2, "event": "round_recorded"}
- {"t": 131.4, "round": 3, "event": "kill_orchestrator", "orch": 1}
- {"t": 131.4, "round": 3, "event": "round_recorded"}
- {"t": 158.9, "round": 4, "event": "round_recorded"}
- {"t": 182.7, "round": 5, "event": "kill_orchestrator", "orch": 2}
- {"t": 182.7, "round": 5, "event": "client_rejoin", "client": 4}
- {"t": 182.7, "round": 5, "event": "round_recorded"}
- {"t": 219.4, "round": 6, "event": "round_recorded"}
- {"t": 254.4, "round": 7, "event": "restart_orchestrator", "orch": 1}
- {"t": 254.5, "round": 7, "event": "round_recorded"}
- {"t": 576.3, "round": 8, "event": "restart_orchestrator", "orch": 2}
- {"t": 576.4, "round": 8, "event": "client_rejoin", "client": 5}
- {"t": 576.4, "round": 8, "event": "round_recorded"}
- {"t": 890.9, "round": 9, "event": "round_recorded"}
- {"t": 921.5, "round": 10, "event": "stopping", "reached": 10, "timed_out": false, "converged": true}
