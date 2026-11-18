#!/usr/bin/env python3
"""Does a DiLoCo-style outer optimizer beat a plain step on DUST rounds?

The aggregated estimate is not a minibatch gradient. ``finalize`` averages unit
directions and rescales them to the mean declared norm, so the result has a
*declared* magnitude that does not shrink as the model approaches an optimum.
DiLoCo's scaling work therefore uses a separate pseudo-gradient learning rate
(0.7) with Nesterov momentum (0.9) rather than a small SGD learning rate.

This sweeps that outer learning rate against the shipped plain path on the same
batch, the same starting weights and the same coordinator, so the only variable
is how the aggregated direction is turned into a weight move.

    python tools/bench_outer.py --rounds 30
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crowd_v15 import protocol  # noqa: E402
from crowd_v15.coordinator import CoordConfig, Coordinator  # noqa: E402
from crowd_v15.dust import DustConfig, dust_gradients  # noqa: E402
from crowd_v15.model import GPT, ModelConfig  # noqa: E402
from crowd_v15.store import ModelStore  # noqa: E402
from crowd_v15.tasks import identity_batch  # noqa: E402
from tools.bench_feedback import clean_loss, load_into, tensors_of  # noqa: E402


def run_arm(name: str, *, model_cfg: ModelConfig, rounds: int, cfg: DustConfig, batch: int,
            seq_len: int, lr: float, outer_lr: float, outer_momentum: float,
            max_abs_grad: float, seed: int) -> dict:
    torch.manual_seed(seed)
    model = GPT(model_cfg)
    coord = Coordinator(
        store=ModelStore(Path(tempfile.mkdtemp(prefix="v15-outer-"))),
        config=CoordConfig(lr=lr, outer_lr=outer_lr, outer_momentum=outer_momentum,
                           min_submissions=1, max_abs_grad=max_abs_grad),
        model_meta={"name": name},
    )
    coord.bootstrap(tensors_of(model))

    generator = torch.Generator().manual_seed(seed + 7)
    idx, tgt = identity_batch(batch, seq_len, model_cfg.vocab_size, torch.device("cpu"), generator)
    losses = [clean_loss(model, idx, tgt)]
    started = time.time()

    for rnd in range(rounds):
        base_seed = seed + rnd * 1_000_003
        grads, stats = dust_gradients(model, cfg, idx, tgt, base_seed=base_seed, step=0)
        names, sizes, vector = protocol.flatten(grads)
        header = protocol.build_header(
            route="dust", client_id="bench", round_id=coord.round_id,
            base_sha256=coord.base_sha256, names=names, sizes=sizes,
            codec_name="ngd1",
            tokens=int(stats["tokens"]), population=int(stats["population"]),
            sigma=float(stats["sigma"]), gamma=float(stats["gamma"]),
            draws=int(stats["draws"]), baseSeed=int(base_seed), seed=seed,
            step=0, localSteps=1, params=int(sum(sizes)),
        )
        reply = coord.submit(protocol.encode_body(header, vector))
        if not reply.get("accepted"):
            raise SystemExit(f"{name}: rejected {reply}")
        load_into(model, coord.store.load_weights())
        losses.append(clean_loss(model, idx, tgt))

    return {
        "arm": name,
        "lr": lr,
        "outerLr": outer_lr,
        "outerMomentum": outer_momentum,
        "firstLoss": losses[1],
        "finalLoss": losses[-1],
        "bestLoss": min(losses[1:]),
        "delta": losses[-1] - losses[1],
        "seconds": round(time.time() - started, 2),
        "lossCurve": [round(x, 6) for x in losses],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--draw-chunk", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-abs-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-lrs", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.7])
    parser.add_argument("--outer-momentum", type=float, default=0.9)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    model_cfg = ModelConfig.preset(args.preset)
    cfg = DustConfig(population=args.population, draw_chunk=args.draw_chunk,
                     sigma=args.sigma, gamma=args.gamma)

    arms = [run_arm("plain", model_cfg=model_cfg, rounds=args.rounds, cfg=cfg,
                    batch=args.batch, seq_len=args.seq_len, lr=args.lr, outer_lr=0.0,
                    outer_momentum=0.0, max_abs_grad=args.max_abs_grad, seed=args.seed)]
    for outer_lr in args.outer_lrs:
        arms.append(run_arm(f"outer-{outer_lr:g}", model_cfg=model_cfg, rounds=args.rounds,
                            cfg=cfg, batch=args.batch, seq_len=args.seq_len, lr=args.lr,
                            outer_lr=outer_lr, outer_momentum=args.outer_momentum,
                            max_abs_grad=args.max_abs_grad, seed=args.seed))

    baseline = arms[0]["finalLoss"]
    for arm in arms:
        arm["vsPlain"] = round(arm["finalLoss"] - baseline, 6)
        print(f"{arm['arm']:>12}  final {arm['finalLoss']:.4f}  best {arm['bestLoss']:.4f}  "
              f"vs plain {arm['vsPlain']:+.4f}  ({arm['seconds']:.1f}s)")

    payload = {"schema": "crowd-v15-outer-bench-1", "rounds": args.rounds,
               "preset": args.preset, "population": args.population, "arms": arms}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
