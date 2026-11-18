DRAFT — requires human review

# Scenario `chaos`

3 orchestrators, 6 clients, hyper `{"preset": "10m", "seq": 128, "H": 30, "batch": 8, "K": 2}`, p_audit 0.5.

Held-out loss: 8.366322 at genesis -> 7.202351 after 10 rounds. Finality: 10 final, 0 provisional round entries (as held by the most complete orchestrator at stop time).

## Rounds

| round | eval loss | rolled back | leader | sigs | final | cells | sampled | statuses | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.92778 | False | orch1 | 3 | True | 6 | 6 | accepted:6 | 42.27 |
| 2 | 7.686115 | False | orch1 | 3 | True | 6 | 6 | accepted:6 | 39.07 |
| 3 | 7.533398 | False | orch1 | 3 | True | 5 | 5 | accepted:5 | 32.65 |
| 4 | 7.450009 | False | orch0 | 3 | True | 5 | 2 | accepted:5 | 34.56 |
| 5 | 7.408943 | False | orch0 | 3 | True | 4 | 3 | accepted:4 | 26.27 |
| 6 | 7.399363 | False | orch2 | 3 | True | 5 | 4 | accepted:5 | 34.25 |
| 7 | 7.388942 | False | orch2 | 3 | True | 5 | 2 | accepted:5 | 18.67 |
| 8 | 7.372063 | False | orch2 | 3 | True | 5 | 4 | accepted:1, missing_audit:4 | 34.35 |
| 9 | 7.350762 | False | orch1 | 3 | True | 6 | 2 | accepted:5, missing_audit:1 | 304.08 |
| 10 | 7.202351 | False | orch2 | 3 | True | 6 | 4 | accepted:6 | 28.6 |

## Detection by behaviour

| behaviour | cells | accepted | rejected (strike) | excluded (no strike) | reasons |
|---|---|---|---|---|---|
| honest | 53 | 48 | 0 | 5 | {"missing_audit": 5} |

Bans: {}

## Replay error of honest cells (live audits)

Thresholds {"cos_min": 0.9799, "ratio_hi": 1.0501, "ratio_lo": 0.9499}.

- client cpu vs auditor cpu: {"cos": {"n": 33, "min": 1.0, "mean": 1.0, "max": 1.0}, "ratio": {"n": 33, "min": 1.0, "mean": 1.0, "max": 1.0}, "failed": 0}

Genesis calibration pairs: [{"cos": 1.0, "module": 0, "pair": "cpu-same", "ratio": 1.0}, {"cos": 1.0, "module": 0, "pair": "cpu-threads", "ratio": 1.0}, {"cos": 0.999873, "module": 0, "pair": "cpu-mps", "ratio": 0.999933}, {"cos": 1.0, "module": 1, "pair": "cpu-same", "ratio": 1.0}, {"cos": 1.0, "module": 1, "pair": "cpu-threads", "ratio": 1.0}, {"cos": 0.999957, "module": 1, "pair": "cpu-mps", "ratio": 1.000002}]

## Traffic per client per round (bytes)

- up (actual HTTP, dense bf16 + base64 JSON): {"n": 53, "min": 11195554, "mean": 18639311.698113, "max": 41962639}
- down (actual HTTP, fp32 weights): {"n": 53, "min": 21048307, "mean": 36323346.566038, "max": 38612474}
- SparseLoCo wire formula for the same message: {"n": 53, "min": 122368, "mean": 139178.150943, "max": 153090}

Round wall time (s): {"n": 10, "min": 18.67, "mean": 59.477, "max": 304.08}

## Chain verification

- orch0: {"ok": true, "entries": 21, "final": 21, "provisional": 0, "errors": []}
- orch1: {"ok": true, "entries": 21, "final": 21, "provisional": 0, "errors": []}
- orch2: {"ok": true, "entries": 21, "final": 21, "provisional": 0, "errors": []}

## Timeline

- {"t": 37.3, "round": 0, "event": "orchestrators_up"}
- {"t": 37.4, "round": 0, "event": "clients_started", "n": 6}
- {"t": 81.4, "round": 1, "event": "round_recorded"}
- {"t": 122.8, "round": 2, "event": "round_recorded"}
- {"t": 157.2, "round": 3, "event": "kill_orchestrator", "orch": 1}
- {"t": 157.3, "round": 3, "event": "round_recorded"}
- {"t": 192.9, "round": 4, "event": "round_recorded"}
- {"t": 219.9, "round": 5, "event": "kill_orchestrator", "orch": 0}
- {"t": 219.9, "round": 5, "event": "client_rejoin", "client": 4}
- {"t": 219.9, "round": 5, "event": "round_recorded"}
- {"t": 254.4, "round": 6, "event": "round_recorded"}
- {"t": 273.7, "round": 7, "event": "restart_orchestrator", "orch": 1}
- {"t": 273.7, "round": 7, "event": "round_recorded"}
- {"t": 309.5, "round": 8, "event": "restart_orchestrator", "orch": 0}
- {"t": 309.5, "round": 8, "event": "client_rejoin", "client": 5}
- {"t": 309.6, "round": 8, "event": "round_recorded"}
- {"t": 614.7, "round": 9, "event": "round_recorded"}
- {"t": 644.5, "round": 10, "event": "stopping", "reached": 10, "timed_out": false, "converged": true}
