"""Multi-process smoke: 3 orchestrators + 3 clients, tiny preset, 2 rounds, final entries."""

import json

import httpx

from crowd_v2 import launch
from crowd_v2.nets import Hyper
from crowd_v2.record import verify_chain

BASE = 8761
TIMING = {"open": 2.0, "commit": 60.0, "upload": 20.0, "audit": 60.0,
          "propose_timeout": 4.0, "final_timeout": 8.0}


def _first_event(url: str) -> dict:
    with httpx.stream("GET", url + "/v2/events", timeout=10.0) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                return json.loads(line[6:])
    return {}


def test_three_orchestrators_three_clients_two_rounds(tmp_path):
    cfgs = launch.make_network(tmp_path, 3, Hyper(preset="tiny", H=6, batch=4), 0.5, TIMING, BASE)
    urls = [f"http://127.0.0.1:{BASE + i}" for i in range(3)]
    procs = [launch.start_orchestrator(c, tmp_path / f"orch{i}.log") for i, c in enumerate(cfgs)]
    try:
        launch.wait_for(lambda: all(launch.health(u) for u in urls), 90, "orchestrators healthy")
        for i in range(3):
            procs.append(launch.start_client(urls[i:] + urls[:i], tmp_path / f"c{i}.key",
                                             tmp_path / f"c{i}.log", tmp_path / f"c{i}.jsonl", rounds=2))
        launch.wait_for(lambda: min(h["round"] for h in map(launch.health, urls) if h) >= 2,
                        240, "all orchestrators at round 2")
        assert _first_event(urls[0])["type"]
        keys = json.loads((tmp_path / "genesis" / "genesis.json").read_text())["body"]["orchestrators"]
        records = []
        for i in range(3):
            rep = verify_chain(tmp_path / f"orch{i}" / "record.jsonl", keys)
            assert rep["ok"], rep
            records.append([json.loads(x)["hash"] for x in
                            (tmp_path / f"orch{i}" / "record.jsonl").read_text().splitlines()])
        common = min(len(r) for r in records)
        assert all(r[:common] == records[0][:common] for r in records)
        entries = [json.loads(x) for x in (tmp_path / "orch0" / "record.jsonl").read_text().splitlines()]
        rounds = [e for e in entries if e["type"] == "round"]
        assert len(rounds) >= 2
        assert all(len(e["sigs"]) >= 2 for e in rounds[:2])
        accepted = sum(c["status"] == "accepted" for e in rounds[:2] for c in e["body"]["cells"])
        assert accepted >= 4
        assert rounds[1]["body"]["eval_loss"] < entries[0]["body"]["eval_loss"]
        status = httpx.get(urls[1] + "/v2/status", timeout=10).json()
        assert status["chain"]["final"] >= 3 and len(status["contributors"]) == 3
    finally:
        launch.stop(procs)
