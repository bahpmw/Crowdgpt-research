"""Shared MVP harness: one small GPT, real FineWeb-Edu tokens, one budget, one log.

Every method is a class with ``step(it) -> Cost``. The harness owns data, the
model, evaluation and accounting so arms differ only in how they turn tokens
into a weight update.

Accounting (per optimizer step, summed over simulated workers):
  tokens   training tokens consumed (the data a crowd would have to ship)
  fwd      forward token-passes (1 per token per forward)
  bwd      backward token-passes counted as 2 forwards each
  up/down  bytes a worker would upload/download over the internet
Wall time is measured on the Mac; "compute" = fwd + 2*bwd forward-equivalents.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crowd_v15.model import GPT, ModelConfig  # noqa: E402

LAB = Path(__file__).resolve().parent
# Extra evals when cumulative tokens cross these, so fast and slow arms share
# loss-at-tokens points (eval time is not counted as training time).
TOKEN_MARKS = [x * 10 ** e for e in range(3, 9) for x in (1, 2, 5)]
RESULTS = LAB / "results"


@dataclass
class Cost:
    tokens: int = 0
    fwd: float = 0.0
    bwd: float = 0.0
    up: float = 0.0
    down: float = 0.0
    extra: dict = field(default_factory=dict)


class Data:
    """Random windows from uint16 token files; worker shards for non-IID tests."""

    def __init__(self, root: Path, seq: int, batch: int, seed: int, device: torch.device):
        self.train = np.memmap(root / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(root / "val.bin", dtype=np.uint16, mode="r")
        self.seq, self.batch, self.device = seq, batch, device
        self.rng = np.random.default_rng(seed)
        vrng = np.random.default_rng(12345)
        starts = vrng.integers(0, len(self.val) - seq - 1, size=64)
        self.val_batch = self._windows(self.val, starts)

    def _windows(self, arr, starts):
        x = np.stack([arr[s:s + self.seq + 1] for s in starts]).astype(np.int64)
        t = torch.from_numpy(x).to(self.device)
        return t[:, :-1], t[:, 1:]

    def next(self, batch: int | None = None, worker: int = 0, workers: int = 1,
             noniid: bool = False):
        """A training batch. With ``noniid`` each worker draws only from its own
        contiguous slice of the stream (documents differ by slice)."""
        n = batch or self.batch
        lo, hi = 0, len(self.train) - self.seq - 1
        if noniid and workers > 1:
            span = hi // workers
            lo, hi = worker * span, (worker + 1) * span
        return self._windows(self.train, self.rng.integers(lo, hi, size=n))


def build_model(preset: str, seed: int, device: torch.device) -> GPT:
    torch.manual_seed(seed)
    return GPT(ModelConfig.preset(preset)).to(device)


def loss_fn(model, idx, tgt, st=None):
    logits = model(idx, st) if st is not None else model(idx)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))


@torch.no_grad()
def eval_loss(model, data: Data, chunks: int = 4) -> float:
    model.eval()
    idx, tgt = data.val_batch
    total = 0.0
    for i, j in zip(idx.chunk(chunks), tgt.chunk(chunks)):
        total += float(loss_fn(model, i, j))
    model.train()
    return total / chunks


def nbytes(*tensors) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors))


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run(method_name: str, cfg: dict, *, preset: str, seconds: float, seq: int, batch: int,
        seed: int, device: str, eval_every: float, out: Path | None, label: str | None,
        save: Path | None = None, init: Path | None = None, tokens: int = 0) -> dict:
    dev = torch.device(device)
    data = Data(ROOT / "runs/data/tok", seq, batch, seed, dev)
    model = build_model(preset, seed, dev)
    if init:
        model.load_state_dict(torch.load(init, map_location=dev))
    mod = importlib.import_module(f"methods.{method_name}")
    method = mod.Method(model, data, cfg, dev)
    label = label or method_name
    curve, tot = [], Cost()
    t_train, it, next_eval, mark = 0.0, 0, 0.0, 0
    loss0 = eval_loss(model, data)
    curve.append({"it": 0, "seconds": 0.0, "tokens": 0, "compute": 0.0, "up": 0.0,
                  "down": 0.0, "evalLoss": loss0})
    print(f"[{label}] start eval {loss0:.4f}", flush=True)
    best = loss0
    diverged = False
    while t_train < seconds and not (tokens and tot.tokens >= tokens):
        sync(dev)
        t0 = time.perf_counter()
        c = method.step(it)
        sync(dev)
        t_train += time.perf_counter() - t0
        it += 1
        tot.tokens += c.tokens
        tot.fwd += c.fwd
        tot.bwd += c.bwd
        tot.up += c.up
        tot.down += c.down
        hit_tok = tot.tokens >= TOKEN_MARKS[mark] if mark < len(TOKEN_MARKS) else False
        while mark < len(TOKEN_MARKS) and tot.tokens >= TOKEN_MARKS[mark]:
            mark += 1
        done = t_train >= seconds or bool(tokens and tot.tokens >= tokens)
        if t_train >= next_eval or done or hit_tok:
            if t_train >= next_eval:
                next_eval = t_train + eval_every
            ev = eval_loss(model, data)
            best = min(best, ev)
            row = {"it": it, "seconds": round(t_train, 3), "tokens": tot.tokens,
                   "compute": tot.fwd + 2 * tot.bwd, "up": tot.up, "down": tot.down,
                   "evalLoss": ev, **{k: v for k, v in c.extra.items()
                                      if isinstance(v, (int, float, str))}}
            curve.append(row)
            print(f"[{label}] it {it} t {t_train:6.1f}s tok {tot.tokens/1e6:7.3f}M "
                  f"eval {ev:.4f} up/step {c.up/1e6:.3f}MB {c.extra}", flush=True)
            if not math.isfinite(ev) or ev > loss0 + 2.0:
                diverged = True
                print(f"[{label}] diverged", flush=True)
                break
    result = {
        "schema": "crowd-lab-run-1", "label": label, "method": method_name, "cfg": cfg,
        "preset": preset, "params": model.n_params(), "seq": seq, "batch": batch,
        "seed": seed, "device": device, "budgetSeconds": seconds, "budgetTokens": tokens,
        "trainSeconds": round(t_train, 3), "steps": it,
        "totals": asdict(tot) | {"compute": tot.fwd + 2 * tot.bwd}, "startLoss": loss0,
        "finalLoss": curve[-1]["evalLoss"], "bestLoss": best, "diverged": diverged,
        "upBytesPerStep": tot.up / max(it, 1), "curve": curve,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save)
    return result


def parse_cfg(pairs: list[str]) -> dict:
    cfg = {}
    for p in pairs:
        k, v = p.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("cfg", nargs="*", help="key=value method settings (JSON values)")
    ap.add_argument("--label")
    ap.add_argument("--preset", default="10m")
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--eval-every", type=float, default=10.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tokens", type=int, default=0, help="stop after this many tokens (0 = time only)")
    ap.add_argument("--save", type=Path, help="write final state_dict here")
    ap.add_argument("--init", type=Path, help="start from this state_dict")
    a = ap.parse_args()
    out = a.out or RESULTS / f"{a.label or a.method}.json"
    r = run(a.method, parse_cfg(a.cfg), preset=a.preset, seconds=a.seconds, seq=a.seq,
            batch=a.batch, seed=a.seed, device=a.device, eval_every=a.eval_every,
            out=out, label=a.label, save=a.save, init=a.init, tokens=a.tokens)
    print(json.dumps({k: r[k] for k in ("label", "steps", "startLoss", "finalLoss",
                                        "bestLoss", "diverged", "upBytesPerStep")}
                     | {"tokens": r["totals"]["tokens"]}))


if __name__ == "__main__":
    sys.path.insert(0, str(LAB))
    main()
