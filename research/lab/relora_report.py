"""Print the ReLoRA study table (Markdown) from results/*.json. Used for plans/relora.md."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "archived" / "crowd-v1.5"))
from crowd_v15.model import ModelConfig  # noqa: E402

LABELS = ["s-bp-128x8", "rl-full", "rl-r32", "rl-r128", "s-mo-k2w8", "rl-mo-k2w8-base", "rl-mo-k2w8-dense", "rl-mo-k2w8-r32-prod",
          "rl-mo-k2w8-r32-fac", "rl-mo-k2w8-r128-prod", "rl-mo-k2w8-r128-fac",
          "rl-mo-k2w8-r32-favg", "rl-100m-full", "rl-100m-r128"]

print("| arm | preset | method / settings | tokens | final eval loss | best | "
      "upload MB per worker per 30 steps | download MB per worker per 30 steps | wall s |")
print("|---|---|---|---|---|---|---|---|---|")
for lab in LABELS:
    f = Path(__file__).parent / "results" / f"{lab}.json"
    if not f.exists():
        print(f"| {lab} | | not finished | | | | | | |")
        continue
    r = json.loads(f.read_text())
    cfg = " ".join(f"{k}={v}" for k, v in r["cfg"].items() if k not in ("total", "tokens"))
    t = r["totals"]
    if r["method"].startswith("mosaic"):
        w = r["cfg"].get("workers", 4)
        up = r["curve"][-1].get("upPerWorkerMB", t["up"] / r["steps"] / w / 1e6)
        down = t["down"] / r["steps"] / w / 1e6
        up_s = f"{up:.3f} (last round)"
    else:
        n = ModelConfig.preset(r["preset"]).n_params * r["curve"][-1]["trainFrac"]
        down = 0.0
        up_s = f"{4 * n * 30 / 1e6:.1f} (dense fp32 grads of trained params, last phase)"
    print(f"| {lab} | {r['preset']} | {r['method']} {cfg} | {t['tokens'] / 1e6:.1f}M "
          f"| {r['finalLoss']:.4f} | {r['bestLoss']:.4f} | {up_s} | {down:.3f} | {r['trainSeconds']:.0f} |")
