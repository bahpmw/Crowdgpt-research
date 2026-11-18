"""Project round results onto a consumer-GPU crowd.

The Mac measures loss per token and compute per token; a crowd adds the
network. For each run this reports, per worker step on a consumer box:
compute seconds (Mac step time scaled by --gpu-factor), upload seconds at
--uplink-mbps, download seconds at --downlink-mbps (when the method downloads
per step), the share of the step spent on the network, and coordinator
ingress for --crowd workers. It does not model large-batch efficiency: loss
per token comes from the single-machine run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def workers(cfg: dict) -> int:
    for key in ("workers", "w", "n"):
        if key in cfg:
            return max(int(cfg[key]), 1)
    return 1


def project(r: dict, *, gpu_factor: float, up_mbps: float, down_mbps: float, crowd: int) -> dict:
    t, steps = r["totals"], max(r["steps"], 1)
    secs = r["curve"][-1]["seconds"] or 1.0
    w = workers(r.get("cfg", {}))
    step_s = secs / steps * gpu_factor / w
    up = t["up"] / steps / w
    down = t.get("down", 0) / steps / w
    up_s = up * 8 / (up_mbps * 1e6)
    down_s = down * 8 / (down_mbps * 1e6)
    total = step_s + up_s + down_s
    return {
        "label": r["label"],
        "workerComputeS": step_s,
        "upMB": up / 1e6,
        "upS": up_s,
        "downS": down_s,
        "netShare": (up_s + down_s) / total if total else 0.0,
        "tokensPerWorkerStep": t["tokens"] / steps / w,
        "workerTokPerS": t["tokens"] / steps / w / total if total else 0.0,
        "ingressGbps": crowd * up * 8 / total / 1e9 if total else 0.0,
        "bestLoss": r["bestLoss"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="*", default=[""])
    ap.add_argument("--gpu-factor", type=float, default=1.6,
                    help="consumer step time / M5 Max step time (RTX 4060 Ti DUST 1b: 2.08/1.31)")
    ap.add_argument("--uplink-mbps", type=float, default=20.0)
    ap.add_argument("--downlink-mbps", type=float, default=200.0)
    ap.add_argument("--crowd", type=int, default=10_000)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.stem.startswith("smoke") or not any(p.stem.startswith(x) for x in a.prefix):
            continue
        r = json.loads(p.read_text())
        if "curve" not in r or not r["curve"]:
            continue
        rows.append(project(r, gpu_factor=a.gpu_factor, up_mbps=a.uplink_mbps,
                            down_mbps=a.downlink_mbps, crowd=a.crowd))
    head = ["run", "best", "worker step s", "up MB", "up s", "net share",
            "tok/worker-step", "worker tok/s", f"ingress Gb/s @{a.crowd:,}"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for x in sorted(rows, key=lambda x: x["bestLoss"]):
        lines.append("| " + " | ".join([
            x["label"], f"{x['bestLoss']:.3f}", f"{x['workerComputeS']:.3f}", f"{x['upMB']:.3f}",
            f"{x['upS']:.3f}", f"{x['netShare']:.0%}", f"{x['tokensPerWorkerStep']:,.0f}",
            f"{x['workerTokPerS']:,.0f}", f"{x['ingressGbps']:.2f}",
        ]) + " |")
    text = "\n".join(lines)
    print(text)
    if a.md:
        a.md.write_text(text + "\n")


if __name__ == "__main__":
    main()
