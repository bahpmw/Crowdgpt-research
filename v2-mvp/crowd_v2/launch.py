"""Start a local network as subprocesses and wait on it by polling (no fixed sleeps)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .crypto import Identity
from .genesis import make_genesis
from .nets import Hyper
from .paths import MVP


def make_network(root: Path, n_orch: int, hp: Hyper, p_audit: float, timing: dict,
                 base_port: int = 8701, calib_devices: tuple[str, ...] = ()) -> list[Path]:
    """Genesis plus one NodeConfig JSON per orchestrator; returns the config paths.

    ``calib_devices`` (e.g. ``("mps",)``) adds cpu-vs-device pairs to the replay calibration.
    """
    gdir = root / "genesis"
    make_genesis(gdir, n_orch, hp, p_audit, timing, list(calib_devices))
    pubs = [Identity.load_or_create(gdir / f"orch{i}.key").pub for i in range(n_orch)]
    paths = []
    for i in range(n_orch):
        cfg = {"port": base_port + i, "key": str(gdir / f"orch{i}.key"),
               "genesis": str(gdir / "genesis.json"), "init": str(gdir / "init.safetensors"),
               "data": str(root / f"orch{i}"),
               "peers": [{"url": f"http://127.0.0.1:{base_port + j}", "pub": pubs[j]}
                         for j in range(n_orch) if j != i]}
        path = root / f"orch{i}.json"
        path.write_text(json.dumps(cfg, indent=1))
        paths.append(path)
    return paths


def _spawn(args: list[str], log_path: Path) -> subprocess.Popen:
    env = os.environ | {"PYTHONPATH": str(MVP), "OMP_NUM_THREADS": "2"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = log_path.open("a")
    return subprocess.Popen([sys.executable, "-m", *args], cwd=MVP, env=env, stdout=out,
                            stderr=subprocess.STDOUT)


def start_orchestrator(cfg: Path, log_path: Path) -> subprocess.Popen:
    return _spawn(["crowd_v2.orchestrator", "--config", str(cfg)], log_path)


def start_client(urls: list[str], key: Path, log_path: Path, metrics: Path, behaviour: str = "honest",
                 rounds: int = 10, leave_after: int | None = None, device: str = "cpu") -> subprocess.Popen:
    args = ["crowd_v2.client", "--urls", ",".join(urls), "--key", str(key), "--behaviour", behaviour,
            "--rounds", str(rounds), "--log", str(metrics), "--device", device]
    if leave_after is not None:
        args += ["--leave-after", str(leave_after)]
    return _spawn(args, log_path)


def health(url: str) -> dict | None:
    try:
        resp = httpx.get(url + "/health", timeout=2.0)
    except httpx.HTTPError:
        return None
    return resp.json() if resp.status_code == 200 else None


def wait_for(check, timeout: float, what: str, interval: float = 0.25):
    """Poll ``check()`` until it returns a truthy value or the deadline passes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = check()
        if got:
            return got
        time.sleep(interval)
    raise TimeoutError(f"timed out waiting for {what}")


def max_round(urls: list[str]) -> int:
    rounds = [h["round"] for h in (health(u) for u in urls) if h]
    return max(rounds, default=0)


def stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
