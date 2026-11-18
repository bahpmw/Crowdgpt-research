DRAFT — requires human review

# Scenario `attacks`

3 orchestrators, 9 clients, hyper `{"preset": "10m", "seq": 128, "H": 30, "batch": 8, "K": 2}`, p_audit 0.5.

Held-out loss: 8.366322 at genesis -> 7.282719 after 8 rounds. Finality: 8 final, 0 provisional round entries (as held by the most complete orchestrator at stop time).

## Rounds

| round | eval loss | rolled back | leader | sigs | final | cells | sampled | statuses | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.869157 | False | orch1 | 3 | True | 9 | 9 | accepted:4, audit:replay:5 | 68.67 |
| 2 | 7.838231 | False | orch0 | 3 | True | 9 | 8 | accepted:2, audit:replay:6, layer0:replayed_payload:1 | 51.39 |
| 3 | 7.755637 | False | orch1 | 3 | True | 9 | 8 | accepted:2, audit:replay:6, layer0:replayed_payload:1 | 61.11 |
| 4 | 7.623402 | False | orch1 | 3 | True | 4 | 1 | accepted:2, audit:replay:1, layer0:replayed_payload:1 | 19.82 |
| 5 | 7.623402 | True | orch0 | 3 | True | 2 | 0 | accepted:2 | 10.65 |
| 6 | 7.464667 | False | orch2 | 3 | True | 2 | 1 | accepted:2 | 17.38 |
| 7 | 7.403587 | False | orch0 | 3 | True | 2 | 0 | accepted:2 | 11.44 |
| 8 | 7.282719 | False | orch0 | 3 | True | 2 | 1 | accepted:2 | 18.52 |

## Detection by behaviour

| behaviour | cells | accepted | rejected (strike) | excluded (no strike) | reasons |
|---|---|---|---|---|---|
| copy | 4 | 1 | 3 | 0 | {"layer0:replayed_payload": 3} |
| flip | 3 | 0 | 3 | 0 | {"audit:replay": 3} |
| honest | 16 | 16 | 0 | 0 | {} |
| noise | 3 | 0 | 3 | 0 | {"audit:replay": 3} |
| scale | 3 | 0 | 3 | 0 | {"audit:replay": 3} |
| stale | 4 | 1 | 3 | 0 | {"audit:replay": 3} |
| wrongdata | 3 | 0 | 3 | 0 | {"audit:replay": 3} |
| zero | 3 | 0 | 3 | 0 | {"audit:replay": 3} |

Bans: {"00ae7e197c146399": {"round": 3, "behaviour": "scale"}, "5d5532f53a220261": {"round": 3, "behaviour": "flip"}, "75e36976d77d8735": {"round": 3, "behaviour": "zero"}, "b5f291d571f2de78": {"round": 3, "behaviour": "wrongdata"}, "daf55ed4ad04c691": {"round": 3, "behaviour": "noise"}, "e68f97f2e90c29b1": {"round": 4, "behaviour": "stale"}, "f268e6b6cee04f4c": {"round": 4, "behaviour": "copy"}}

## Replay error of honest cells (live audits)

Thresholds {"cos_min": 0.9799, "ratio_hi": 1.0501, "ratio_lo": 0.9499}.

- client cpu vs auditor cpu: {"cos": {"n": 8, "min": 1.0, "mean": 1.0, "max": 1.0}, "ratio": {"n": 8, "min": 1.0, "mean": 1.0, "max": 1.0}, "failed": 0}

Genesis calibration pairs: [{"cos": 1.0, "module": 0, "pair": "cpu-same", "ratio": 1.0}, {"cos": 1.0, "module": 0, "pair": "cpu-threads", "ratio": 1.0}, {"cos": 0.999873, "module": 0, "pair": "cpu-mps", "ratio": 0.999933}, {"cos": 1.0, "module": 1, "pair": "cpu-same", "ratio": 1.0}, {"cos": 1.0, "module": 1, "pair": "cpu-threads", "ratio": 1.0}, {"cos": 0.999957, "module": 1, "pair": "cpu-mps", "ratio": 1.000002}]

## Traffic per client per round (bytes)

- up (actual HTTP, dense bf16 + base64 JSON): {"n": 39, "min": 11195415, "mean": 19158463.846154, "max": 41962614}
- down (actual HTTP, fp32 weights): {"n": 39, "min": 8902, "mean": 32779930.461538, "max": 48449928}
- SparseLoCo wire formula for the same message: {"n": 39, "min": 122368, "mean": 138910.615385, "max": 153090}

Round wall time (s): {"n": 8, "min": 10.65, "mean": 32.3725, "max": 68.67}

## Chain verification

- orch0: {"ok": true, "entries": 40, "final": 40, "provisional": 0, "errors": []}
- orch1: {"ok": true, "entries": 40, "final": 40, "provisional": 0, "errors": []}
- orch2: {"ok": true, "entries": 40, "final": 40, "provisional": 0, "errors": []}

## Timeline

- {"t": 33.1, "round": 0, "event": "orchestrators_up"}
- {"t": 33.3, "round": 0, "event": "clients_started", "n": 9}
- {"t": 103.6, "round": 1, "event": "round_recorded"}
- {"t": 156.6, "round": 2, "event": "round_recorded"}
- {"t": 219.8, "round": 3, "event": "round_recorded"}
- {"t": 241.1, "round": 4, "event": "round_recorded"}
- {"t": 253.5, "round": 5, "event": "round_recorded"}
- {"t": 272.4, "round": 6, "event": "round_recorded"}
- {"t": 284.9, "round": 7, "event": "round_recorded"}
- {"t": 305.2, "round": 8, "event": "stopping", "reached": 8, "timed_out": false, "converged": true}
