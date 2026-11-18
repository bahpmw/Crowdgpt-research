"""Run a scenario: ``python sim.py scenarios/baseline.json`` -> runs/<name>/report.{json,md}.

Starts N orchestrators and M clients as local subprocesses, applies chaos
(kill/restart orchestrators at given rounds) and churn (clients leave and
rejoin), then reads the records, client metrics and genesis calibration to
write the report. All waiting is polling with deadlines.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import httpx
import torch

from crowd_v2 import beacon, launch
from crowd_v2.crypto import Identity
from crowd_v2.nets import Hyper
from crowd_v2.paths import MVP
from crowd_v2.record import verify_chain
from crowd_v2.rounds import NO_STRIKE

log = logging.getLogger("sim")


class Sim:
    def __init__(self, sc: dict, out: Path):
        self.sc, self.out = sc, out
        self.base = int(sc.get("base_port", 8701))
        self.n = int(sc["orchestrators"])
        self.urls = [f"http://127.0.0.1:{self.base + i}" for i in range(self.n)]
        self.orchs: dict[int, subprocess.Popen] = {}
        self.clients: dict[int, subprocess.Popen] = {}
        self.timeline: list[dict] = []
        self.killed: list[int] = []
        self.specs = [b for c in sc["clients"] for b in [c["behaviour"]] * int(c.get("count", 1))]
        self.devices = [d for c in sc["clients"] for d in [c.get("device", "cpu")] * int(c.get("count", 1))]
        self.kids: list[str] = []
        self.client_kids: dict[str, tuple[int, str]] = {}
        self.t0 = time.time()
        self.chaos_done: set[int] = set()
        self.rejoined: set[int] = set()

    def note(self, what: str, **kw: object) -> None:
        row = {"t": round(time.time() - self.t0, 1), "round": self.round(), "event": what, **kw}
        self.timeline.append(row)
        print(json.dumps(row), flush=True)

    def round(self) -> int:
        return launch.max_round([self.urls[i] for i in self.orchs])

    def start_orch(self, i: int) -> None:
        self.orchs[i] = launch.start_orchestrator(self.cfgs[i], self.out / f"orch{i}.log")

    def kill_orch(self, i: int) -> None:
        p = self.orchs.pop(i)
        p.kill()
        p.wait()
        self.killed.append(i)
        self.note("kill_orchestrator", orch=i)

    def start_client(self, j: int, leave_after: int | None) -> None:
        urls = self.urls[j % self.n:] + self.urls[:j % self.n]
        self.clients[j] = launch.start_client(
            urls, self.out / "keys" / f"c{j}.key", self.out / f"client{j}.log",
            self.out / f"client{j}.jsonl", self.specs[j], int(self.sc["rounds"]), leave_after,
            self.devices[j])

    def leader_index(self) -> int:
        """The first live orchestrator in this round's leader order (the beacon's leader may be dead)."""
        for i in self.orchs:
            try:
                info = httpx.get(self.urls[i] + "/v2/round", timeout=3).json()
            except httpx.HTTPError as exc:
                log.warning("round info failed: %s", exc)
                continue
            order = beacon.leader_order(info["beacon"], sorted(self.kids)) if info["beacon"] else []
            return next((self.kids.index(k) for k in order if self.kids.index(k) in self.orchs), i)
        return next(iter(self.orchs))

    def apply_chaos(self) -> None:
        r = self.round()
        for idx, ev in enumerate(self.sc.get("chaos", [])):
            if idx in self.chaos_done or r < ev["at_round"]:
                continue
            self.chaos_done.add(idx)
            if ev["action"] == "kill_leader":
                self.kill_orch(self.leader_index())
            elif ev["action"] == "kill":
                self.kill_orch(int(ev["orch"]))
            elif ev["action"] == "restart":
                i = self.killed.pop(0)
                self.start_orch(i)
                self.note("restart_orchestrator", orch=i)
        for j, ch in enumerate(self.sc.get("churn", [])):
            c = int(ch["client"])
            p = self.clients.get(c)
            if p is not None and p.poll() is not None and r >= ch["rejoin_at_round"] and j not in self.rejoined:
                self.rejoined.add(j)
                self.start_client(c, None)
                self.note("client_rejoin", client=c)

    def run(self) -> dict:
        self.t0 = time.time()
        hp = Hyper(**self.sc["hyper"])
        self.cfgs = launch.make_network(self.out, self.n, hp, float(self.sc.get("p_audit", 0.5)),
                                        self.sc["timing"], self.base,
                                        tuple(self.sc.get("calib_devices", ())))
        self.kids = [Identity.load_or_create(self.out / "genesis" / f"orch{i}.key").kid for i in range(self.n)]
        (self.out / "keys").mkdir(exist_ok=True)
        self.client_kids = {Identity.load_or_create(self.out / "keys" / f"c{j}.key").kid: (j, b)
                            for j, b in enumerate(self.specs)}
        for i in range(self.n):
            self.start_orch(i)
        launch.wait_for(lambda: all(launch.health(u) for u in self.urls), 120, "orchestrators")
        self.note("orchestrators_up")
        churn = {int(c["client"]): int(c["leave_after"]) for c in self.sc.get("churn", [])}
        for j in range(len(self.specs)):
            self.start_client(j, churn.get(j))
        self.note("clients_started", n=len(self.specs))
        deadline = self.t0 + 60 * float(self.sc.get("max_minutes", 20))
        target, last = int(self.sc["rounds"]), 0
        while time.time() < deadline and self.round() < target:
            self.apply_chaos()
            r = self.round()
            if r != last:
                last = r
                self.note("round_recorded")
            time.sleep(0.5)
        try:
            launch.wait_for(lambda: len({h["round"] for h in map(launch.health, [self.urls[i] for i in self.orchs])
                                         if h}) <= 1, 60, "orchestrators to converge")
            converged = True
        except TimeoutError as exc:
            # Still write the report: per-orchestrator chain results show who lagged.
            log.warning("%s", exc)
            converged = False
        self.note("stopping", reached=self.round(), timed_out=self.round() < target, converged=converged)
        launch.stop(list(self.clients.values()) + list(self.orchs.values()))
        return report(self.out, self.sc, self.kids, self.client_kids, self.timeline)


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": round(min(xs), 6), "mean": round(statistics.fmean(xs), 6),
            "max": round(max(xs), 6)}


def report(out: Path, sc: dict, kids: list[str], clients: dict[str, tuple[int, str]],
           timeline: list[dict]) -> dict:
    genesis = json.loads((out / "genesis" / "genesis.json").read_text())
    keys = genesis["body"]["orchestrators"]
    chains = {}
    for i in range(len(kids)):
        path = out / f"orch{i}" / "record.jsonl"
        chains[i] = verify_chain(path, keys) if path.exists() else {"ok": False, "errors": ["missing"]}
    best = max(chains, key=lambda i: (chains[i].get("ok", False), chains[i].get("entries", 0)))
    entries = [json.loads(x) for x in (out / f"orch{best}" / "record.jsonl").read_text().splitlines()]
    quorum = genesis["body"]["quorum"]
    behaviour = {k: b for k, (_, b) in clients.items()}
    per_client = []
    for j in range(len(clients)):
        path = out / f"client{j}.jsonl"
        if path.exists():
            per_client += [r for r in map(json.loads, path.read_text().splitlines()) if r["status"] == "submitted"]
    device = {r["key"]: r.get("device", "cpu") for r in per_client}
    rounds, det, honest_audit, attack_audit, bans = [], {}, {}, [], {}
    for e in entries:
        if e["type"] == "verdict" and e["body"]["banned"]:
            bans[e["body"]["key"]] = e["round"]
        if e["type"] != "round":
            continue
        b = e["body"]
        statuses: dict[str, int] = {}
        for c in b["cells"]:
            beh = behaviour.get(c["key"], "unknown")
            statuses[c["status"]] = statuses.get(c["status"], 0) + 1
            d = det.setdefault(beh, {"cells": 0, "accepted": 0, "rejected_strike": 0, "excluded_no_strike": 0,
                                     "by_reason": {}})
            d["cells"] += 1
            if c["status"] == "accepted":
                d["accepted"] += 1
            elif c["status"] in NO_STRIKE:
                d["excluded_no_strike"] += 1
            else:
                d["rejected_strike"] += 1
            if c["status"] != "accepted":
                d["by_reason"][c["status"]] = d["by_reason"].get(c["status"], 0) + 1
        for a in b["audits"]:
            if behaviour.get(a["key"]) == "honest":
                pair = honest_audit.setdefault(f"client {device.get(a['key'], 'cpu')} vs auditor cpu",
                                               {"cos": [], "ratio": [], "failed": 0})
                pair["cos"].append(a["cos"])
                pair["ratio"].append(a["ratio"])
                pair["failed"] += not a["ok"]
            else:
                attack_audit.append({"round": e["round"], "behaviour": behaviour.get(a["key"]),
                                     "cos": a["cos"], "ratio": a["ratio"], "ok": a["ok"]})
        inp = b["inputs"]
        rounds.append({"round": e["round"], "eval_loss": b["eval_loss"], "rolled_back": b["rolled_back"],
                       "leader": b["leader"], "leader_index": kids.index(b["leader"]),
                       "sigs": len(e["sigs"]), "final": len(e["sigs"]) >= quorum,
                       "cells": len(b["cells"]), "statuses": statuses, "sampled": len(b["sampled"]),
                       "round_seconds": round(inp["t_close"] - inp["t_open"], 2), "tokens": b["tokens"],
                       "down_recorded": inp["down"], "up_recorded": inp["up"]})
    for beh, d in det.items():
        attacker = beh != "honest"
        d["true_positive" if attacker else "false_positive"] = d["rejected_strike"]
        d["missed" if attacker else "true_negative"] = d["accepted"]
    rep = {
        "scenario": sc, "genesis_eval_loss": genesis["body"]["eval_loss"],
        "final_eval_loss": rounds[-1]["eval_loss"] if rounds else None,
        "rounds": rounds, "detection": det,
        "bans": {k: {"round": r, "behaviour": behaviour.get(k)} for k, r in bans.items()},
        "replay": {"calibration": genesis["body"]["config"]["calibration"],
                   "thresholds": genesis["body"]["config"]["thresholds"],
                   "honest_live": {k: {"cos": _stats(v["cos"]), "ratio": _stats(v["ratio"]), "failed": v["failed"]}
                                   for k, v in sorted(honest_audit.items())},
                   "attacker_audits": attack_audit},
        "bytes": {"up_per_client_round": _stats([r["bytes_up"] for r in per_client]),
                  "down_per_client_round": _stats([r["bytes_down"] for r in per_client]),
                  "wire_formula_per_client_round": _stats([r["wire_bytes"] for r in per_client]),
                  "per_client_round": [{k: r.get(k) for k in ("round", "key", "behaviour", "module", "bytes_up",
                                                              "bytes_down", "wire_bytes", "train_s")}
                                       for r in per_client]},
        "round_seconds": _stats([r["round_seconds"] for r in rounds]),
        "finality": {"final_rounds": sum(r["final"] for r in rounds),
                     "provisional_rounds": sum(not r["final"] for r in rounds)},
        "chain_verify": {f"orch{i}": c for i, c in chains.items()},
        "timeline": timeline,
    }
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    (out / "report.md").write_text(markdown(rep))
    return rep


def markdown(rep: dict) -> str:
    sc = rep["scenario"]
    lines = ["DRAFT — requires human review", "", f"# Scenario `{sc['name']}`", "",
             f"{sc['orchestrators']} orchestrators, {sum(int(c.get('count', 1)) for c in sc['clients'])} clients, "
             f"hyper `{json.dumps(sc['hyper'])}`, p_audit {sc.get('p_audit', 0.5)}.", "",
             f"Held-out loss: {rep['genesis_eval_loss']} at genesis -> {rep['final_eval_loss']} "
             f"after {len(rep['rounds'])} rounds. Finality: {rep['finality']['final_rounds']} final, "
             f"{rep['finality']['provisional_rounds']} provisional round entries "
             "(as held by the most complete orchestrator at stop time).", "",
             "## Rounds", "",
             "| round | eval loss | rolled back | leader | sigs | final | cells | sampled | statuses | seconds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["rounds"]:
        lines.append(f"| {r['round']} | {r['eval_loss']} | {r['rolled_back']} | orch{r['leader_index']} | "
                     f"{r['sigs']} | {r['final']} | {r['cells']} | {r['sampled']} | "
                     f"{', '.join(f'{k}:{v}' for k, v in sorted(r['statuses'].items()))} | {r['round_seconds']} |")
    lines += ["", "## Detection by behaviour", "",
              "| behaviour | cells | accepted | rejected (strike) | excluded (no strike) | reasons |",
              "|---|---|---|---|---|---|"]
    for beh, d in sorted(rep["detection"].items()):
        lines.append(f"| {beh} | {d['cells']} | {d['accepted']} | {d['rejected_strike']} | "
                     f"{d['excluded_no_strike']} | {json.dumps(d['by_reason'])} |")
    lines += ["", f"Bans: {json.dumps(rep['bans'])}", "", "## Replay error of honest cells (live audits)", "",
              f"Thresholds {json.dumps(rep['replay']['thresholds'])}.", ""]
    lines += [f"- {k}: {json.dumps(v)}" for k, v in rep["replay"]["honest_live"].items()]
    lines += ["",
              "Genesis calibration pairs: " + json.dumps(rep["replay"]["calibration"]), "",
              "## Traffic per client per round (bytes)", "",
              f"- up (actual HTTP, dense bf16 + base64 JSON): {json.dumps(rep['bytes']['up_per_client_round'])}",
              f"- down (actual HTTP, fp32 weights): {json.dumps(rep['bytes']['down_per_client_round'])}",
              f"- SparseLoCo wire formula for the same message: {json.dumps(rep['bytes']['wire_formula_per_client_round'])}",
              "", f"Round wall time (s): {json.dumps(rep['round_seconds'])}", "", "## Chain verification", ""]
    lines += [f"- {k}: {json.dumps(v)}" for k, v in rep["chain_verify"].items()]
    lines += ["", "## Timeline", ""] + [f"- {json.dumps(t)}" for t in rep["timeline"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from an existing run")
    a = ap.parse_args()
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sc = json.loads(a.scenario.read_text())
    out = MVP / "runs" / sc["name"]
    if a.report_only:
        meta = json.loads((out / "sim-meta.json").read_text())
        rep = report(out, sc, meta["kids"], {k: tuple(v) for k, v in meta["clients"].items()}, meta["timeline"])
    else:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        sim = Sim(sc, out)
        try:
            rep = sim.run()
        finally:
            launch.stop(list(sim.clients.values()) + list(sim.orchs.values()))
            (out / "sim-meta.json").write_text(json.dumps(
                {"kids": sim.kids, "clients": sim.client_kids,
                 "timeline": sim.timeline}))
    print(json.dumps({"final_eval_loss": rep["final_eval_loss"], "start": rep["genesis_eval_loss"],
                      "finality": rep["finality"], "detection": rep["detection"]}, indent=1))


if __name__ == "__main__":
    main()
