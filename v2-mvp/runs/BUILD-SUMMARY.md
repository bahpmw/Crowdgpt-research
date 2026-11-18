DRAFT — requires human review

# CrowdGPT v2 MVP: build summary (2026-11-18)

All numbers below are copied from `runs/<scenario>/report.json`. Setup for every scenario:
3 orchestrators, 10m preset (seq 128, H 30, batch 8, K 2), p_audit 0.5 (1.0 for a client's first
3 cells), genesis calibration on cpu and mps, everything on one Mac with other GPU jobs running.

## What works

- The full loop runs end to end with real processes over HTTP: register, beacon-assigned MOSAIC
  cells, signed commit, then upload, then sampled replay by a non-receiving orchestrator, then a
  median-clipped fixed-point merge, then the model gate, then a co-signed hash-chained record.
  Held-out loss fell in all three scenarios.
- No single point of failure. Leaders rotate by beacon. When a leader is killed the next
  orchestrator proposes. A lone orchestrator keeps training with provisional entries. Restarted
  orchestrators catch up by re-running every merge, then co-sign, which made the lone-period
  entries final.
- Checks catch every attack in the scenario. With probation sampling at 1.0, each attacker was
  banned after 3 strikes, by round 3 or 4. There were 0 false positives in 149 honest cells
  across the three scenarios.
- `verify_chain` was ok on every orchestrator in every run, and all round entries were final at
  stop.
- Tests: 39 passed (`../.venv/bin/python -m pytest -q`, 36.1 s).
- The dashboard is served at `/dashboard` (HTTP 200, 9.7 KB). `/v2/status` and `/metrics` answered
  during the chaos run. I did not check the page visually in a browser in this session.

## Results per scenario

### Baseline (runs/baseline/report.json)

3 orchestrators, 8 honest clients (7 cpu, 1 mps), 10m preset, seq 128, H 30, batch 8, K 2, p_audit 0.5.
- Held-out loss 8.366322 (genesis) -> 7.930899 (r1) -> 7.530239 (r3) -> 7.309073 (r6) -> 7.008598 (r10); falls every round, no rollbacks.
- Finality: 10/10 round entries final with 3/3 signatures; verify_chain ok on all 3 orchestrators (19 entries, 19 final, 0 errors each).
- Detection: 80 honest cells, 80 accepted, 0 false positives.
- Replay error, live audits: client cpu vs auditor cpu 52 audits, cos 1.0 and ratio 1.0 (all exact); client mps vs auditor cpu 6 audits, cos min 0.985484 / mean 0.996472, ratio 0.999868-1.000419, 0 failed. Thresholds cos >= 0.9799, ratio 0.9499-1.0501.
- Genesis calibration: cpu-same / cpu-threads cos 1.0; cpu-mps cos 0.999873 (module 0), 0.999957 (module 1).
- Bytes per client per round: up mean 20.4 MB (min 11.2, max 42.0 MB, includes the EF reveal when sampled); down mean 37.8 MB; SparseLoCo formula 122-153 KB.
- Round wall time: mean 43.2 s (min 30.6, max 65.8), 497 s for the whole run.
- Leaders rotated: orch0, orch2, orch1 all led rounds.

### Attacks (runs/attacks/report.json)

3 orchestrators, 2 honest + 7 attackers (flip, scale, noise, zero, copy, wrongdata, stale), 8 rounds, same hyper.
- Detection: every attacker was rejected with a strike in each round its attack was active. 3 strikes led to a ban: flip, scale, noise, zero and wrongdata banned at round 3; stale and copy banned at round 4. Honest: 16 cells, 16 accepted, 0 false positives.
  - flip/scale/noise/zero/wrongdata: 3 cells each, 3 true positives (`audit:replay`), 0 missed.
  - stale: 4 cells, 3 true positives (`audit:replay`), 1 "missed" = round 1, where it has no previous weights and so trains honestly (replay cos 1.0).
  - copy: 4 cells, 3 true positives (`layer0:replayed_payload`), 1 "missed" = round 1, where nothing has been accepted yet to copy, so it trains honestly.
- Attacker replay numbers: scale cos 1.0 / ratio 8.0; flip cos -1.0; zero ratio 0.001; noise cos -0.0024 to 0.005; wrongdata cos 0.036-0.146; stale (rounds 2-4) cos 0.038-0.123. Honest live audits: 8, all cos 1.0 / ratio 1.0.
- Held-out loss 8.366322 -> 7.869157 (r1) -> 7.623402 (r4) -> 7.282719 (r8). Round 5 was rolled back by the model gate (candidate 8.207703 vs kept 7.623402, gate_tol 0.25). Both round-5 cells were honest and accepted, one per module, so there was no averaging.
- Finality: 8/8 round entries final (3 sigs); verify_chain ok on all 3 orchestrators (40 entries, all final, 0 errors).
- Bytes per client per round: up mean 19.2 MB (11.2-42.0 MB); down mean 32.8 MB (max 48.4 MB for the copy client, which also fetches the record and a payload); formula 122-153 KB.
- Round wall time mean 32.4 s (10.7-68.7 s); rounds get shorter once 7 of 9 clients are banned. Whole run took 305 s.

### Chaos, run 3, final code (runs/chaos/report.json)

6 honest clients. The round-3 leader (orch1) was killed after round 3 and the round-5 leader (orch2)
after round 5. orch1 restarted after round 7 and orch2 after round 8. Clients 4 and 5 left after
2 and 4 rounds and rejoined at rounds 5 and 8.
- Held-out loss 8.366322 -> 7.936575 (r1) -> 7.743634 (r3) -> 7.485785 (r5) -> 7.216949 (r7) ->
  6.954011 (r10). Round 3 went up slightly (7.708974 -> 7.743634, within gate_tol), with no
  rollbacks.
- Finality: 10/10 round entries final with 3/3 signatures at stop. verify_chain ok on all 3
  orchestrators (21 entries, 21 final, 0 errors). `orchestrator_down` was recorded in rounds 4
  and 6, and `orchestrator_up` in rounds 8 and 9. Rounds 6-7 ran while orch1 and orch2 were both
  dead, so they were written by orch0 alone.
- Catch-up: restarted orch1 adopted 6 entries (re-running each merge) and restarted orch2 adopted
  5. Round wall time just after the restarts was 26.2 s (round 8) and 24.0 s (round 9). Run 1
  took 320.1 s and 312.1 s for those rounds.
- Round wall time mean 27.2 s (16.1-40.4 s). Whole run 310.4 s (run 1: 921.5 s).
- Detection: 53 honest cells, 52 accepted, 0 strikes, 1 excluded as `missing_audit` (round 8).
  orch1 had just restarted with a different beacon and orch2 was down, so no peer could replay
  that cell.
- Replay: 32 honest audits, cos 1.0 / ratio 1.0. Bytes per client per round: up mean 19.9 MB,
  down mean 37.8 MB, formula 122-153 KB.

## What did not work or was cut

- Run 1 of chaos exposed a stall: rounds just after a restart took about 5 minutes. It is fixed
  (see run notes below). The underlying split remains. After a restart, two orchestrators can
  run the same round with different beacons, because "the last final entry" changes when a
  restarted node co-signs. The fix only stops the stall. The affected cells are still excluded
  (no strike).
- Nesterov outer momentum (optional in the spec) is not implemented. Outer SGD alpha 1.0 only.
- The sparse wire format was not built. Uploads are dense bf16 (11-14 MB per cell over HTTP),
  and the SparseLoCo size is reported as a formula (122-153 KB).
- The dashboard was not checked visually in this session.

## Deviations from SPEC.md

- The beacon mixes in the round number (`beacon.py`). Without it, a lone orchestrator would
  repeat the same data every round.
- The leader proposes once every payload and sampled audit is in, or at the audit deadline. The
  spec says "at the upload deadline". The commit phase also closes early once every assigned cell
  has committed.
- The probation rule (p_audit 1.0) counts a client's first 3 recorded cells, not its first 3
  rounds.
- Layer 3 direction uses the mean of the replay-verified messages, not a leave-one-out mean of
  all messages (reason in `audit.py`).
- A receiver replays its own cell only as a no-stall fallback. Such an audit never accepts a cell
  unless the receiver is the lone live orchestrator.
- `/health` also reports the open round and the commits held, so peers can choose auditors.

## Known gaps

- The cpu-vs-mps replay margin is thin. Genesis calibration (initial weights, empty EF) measured
  cos 0.99987. Live audits of the mps client went down to cos 0.985484 against a threshold of
  0.9799. Calibration should be repeated on trained weights with a non-empty EF.
- After a rollback, clients keep the error-feedback buffer from the rejected round. In the
  attacks run, the round-5 rollback happened with only one honest cell per module.
- The sim reports finality as held at stop time, not when each entry was written.
- Bytes: download is a full fp32 copy of both modules every round (about 37.8 MB). Upload
  includes a dense fp32 EF reveal when sampled (up to 42 MB).
- Everything shares one machine clock, and nothing uses TLS. See `README.md`, "MVP-only
  shortcuts".

## Next steps

1. Make a restarted orchestrator join the round already open at its peers (same beacon and
   deadlines, with commits pulled from a peer) instead of opening its own.
2. Re-run calibration on trained weights and on mps clients, and set thresholds from the live
   distribution.
3. Add the sparse wire format and weight deltas to cut about 20 MB up and 38 MB down per cell.
4. Reset or roll back client EF when a round is rolled back.
5. Use an external beacon (VRF or drand) to remove leader grinding.

## Run notes (chronological)

- Code read end to end. The suite started at 35 passing tests.
- `sim.py` now writes the report even if orchestrators do not converge within 60 s. Scenarios can
  set `calib_devices` and a per-client `device`. The report splits honest live-audit replay error
  by client device. `kill_leader` now kills the first *live* orchestrator in leader order (the
  beacon's leader may already be dead).
- Standalone calibration check: cpu-same / cpu-threads cos 1.0; cpu-vs-mps cos 0.999873 /
  0.999957; 36 s.

### Chaos, run 1, before the auditor fix (copy in runs/chaos-before-fix/report.json)

3 orchestrators, 6 honest clients, 10 rounds. The round-3 leader (orch1) was killed after round 3 and the round-5 leader (orch2) after round 5. orch1 restarted after round 7 and orch2 after round 8. Client 4 left after 2 rounds and rejoined at round 5; client 5 left after 4 and rejoined at round 8.
- Held-out loss 8.366322 -> 7.935205 (r1) -> 7.632754 (r3) -> 7.403761 (r5) -> 7.328093 (r7) -> 7.143219 (r10). It fell every round with no rollbacks.
- Failover: rounds 4-5 ran with 2 orchestrators (quorum 2, final at once). orch1 was recorded `orchestrator_down` in round 4 and orch2 in round 6. Rounds 6-7 were written by orch0 alone (provisional, one signature). After the restarts, catch-up re-ran those merges and co-signed them. At stop, all 10 round entries were final with 3/3 signatures, and verify_chain was ok on all 3 orchestrators (21 entries, 21 final, 0 errors). `orchestrator_up` was recorded in rounds 8 and 9.
- Detection: 53 honest cells, 51 accepted, 0 strikes (0 false positives), 2 excluded as `missing_audit` (no strike), in rounds 8 and 9.
- Round wall time: 23-39 s normally. Rounds 8 and 9 (just after each restart) took 320.1 s and 312.1 s; mean 87.8 s. Whole run 921.5 s.
- Live replay: 30 honest audits, cos 1.0 / ratio 1.0. Bytes per client per round: up mean 17.7 MB, down mean 38.0 MB.
- Bug found: the 5-minute rounds. A restarted orchestrator catches up and co-signs the provisional entries, which makes a newer entry final. It then opens the same round number from that newer final entry, so it gets a different beacon from the orchestrator already running the round. It refuses that orchestrator's forwarded commits ("no assignment"). But the receiver still picked it as auditor for sampled cells, because it was "up". The receiver then waited until the audit deadline (open 4 + commit 120 + upload 30 + audit 150 = 304 s after open), and the cell was excluded as `missing_audit`.
- Fix (`node.py` `_auditor`, `/health` now reports the open round, and `peers.py` stores it): a peer can be the auditor only if it is up and had the same round open, with the same beacon, before the commit arrived. Otherwise the receiver replays the cell itself, so the round does not stall. That cell is still excluded as `missing_audit` with no strike, because a receiver may not audit its own cells unless it is alone. Test: `tests/test_node.py` (3 tests).
- Test suite after the fix: 38 passed in 34.82 s (log: runs/pytest.log). Chaos re-run started.

### Chaos, run 2, with the first fix (copy in runs/chaos-run2/report.json)

The first fix let a peer audit if it had the round open, with the same beacon, before the commit arrived. Leaders killed: orch1 after round 3 and orch0 after round 5. orch1 restarted after round 7 and orch0 after round 8.
- Loss 8.366322 -> 7.92778 (r1) -> 7.408943 (r5) -> 7.388942 (r7) -> 7.202351 (r10). All 10 rounds final with 3 sigs at stop; verify_chain ok on all 3 (21 entries).
- Round 8 now took 34.4 s (before the fix: 320 s), but 4 of 5 cells were excluded as `missing_audit`. The receiver audited them itself because no peer qualified.
- Round 9 still took 304.1 s. The restarted orch0 opened round 9 with the same beacon, before the commits arrived, but after the work requests were dealt. So it had no assignment and refused the commits. The rule "opened before the commit" was not enough.
- Honest: 53 cells, 48 accepted, 0 strikes, 5 `missing_audit`.
- Second fix (current code): `/health` now reports the commits each orchestrator holds for its open round. A peer can audit a cell only if it holds that commit. The receiver falls back to auditing the cell itself only after `AUDIT_GRACE` (3 s) of heartbeats, so a stale heartbeat cannot trigger it. A peer's audit replaces a receiver's self-audit (`on_verdict`), and a late self-audit no longer overwrites a peer audit (`_audit` uses setdefault). `tests/test_node.py` now has 4 tests.
- Test suite after the second fix: 39 passed in 36.10 s. Chaos run 3 started (results above).
