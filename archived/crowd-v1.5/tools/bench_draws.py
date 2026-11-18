#!/usr/bin/env python3
"""Does averaging k independent DUST draws buy accuracy per uploaded byte?

DUST keeps no optimizer state, so k draws at the *same* weights are just a bigger
Monte-Carlo batch: the variance of the averaged estimate falls as 1/k while the
payload is unchanged (the coordinator still receives one estimate). That is a
different trade from ``local_steps``, which moves the model between draws and so
trades staleness for progress.

Two modes:

    python tools/bench_draws.py --mode cosine --preset 10m --draws 1 2 4 8
    python tools/bench_draws.py --mode loss --draws 1 2 4 8 --rounds 20
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
from crowd_v15.dust import DustConfig, backprop_gradients, dust_gradients, probe_cosine  # noqa: E402
from crowd_v15.model import GPT, ModelConfig, canonical_params  # noqa: E402
from crowd_v15.store import ModelStore  # noqa: E402
from crowd_v15.tasks import identity_batch, induction_batch  # noqa: E402
from tools.bench_feedback import clean_loss, load_into, tensors_of  # noqa: E402


def averaged_draws(model, cfg, idx, tgt, *, base_seed, draws):
    """Mean of ``draws`` independent estimates at the current weights."""
    total = None
    stats = None
    for draw in range(draws):
        grads, stats = dust_gradients(model, cfg, idx, tgt,
                                      base_seed=base_seed + draw * 7919, step=0)
        total = grads if total is None else {n: total[n] + grads[n] for n in total}
    return {n: v / float(draws) for n, v in total.items()}, stats


def cosine_arm(model_cfg, cfg, draws, *, seed, batch, seq_len, task):
    torch.manual_seed(seed)
    model = GPT(model_cfg)
    gen = torch.Generator().manual_seed(seed + 5)
    maker = induction_batch if task == "induction" else identity_batch
    idx, tgt = maker(batch, seq_len, model_cfg.vocab_size, torch.device("cpu"), gen)
    started = time.perf_counter()
    grads, _stats = averaged_draws(model, cfg, idx, tgt, base_seed=seed, draws=draws)
    seconds = time.perf_counter() - started
    reference = backprop_gradients(model, idx, tgt)
    head_key = None
    for name in sorted(grads):
        if name.endswith("emb.weight"):
            head_key = name
            break

    def _cat(names_):
        left = np.concatenate([np.asarray(grads[n], dtype=np.float64).reshape(-1) for n in names_])
        right = np.concatenate([np.asarray(reference[n], dtype=np.float64).reshape(-1) for n in names_])
        norm = np.linalg.norm(left) * np.linalg.norm(right)
        return float(left @ right / (norm + 1e-30))

    names = sorted(grads)
    body = [n for n in names if n != head_key]
    by_type = {}
    for kind, pred in (("attn", lambda n: ".attn." in n),
                       ("mlp", lambda n: ".mlp." in n),
                       ("embedding", lambda n: n == head_key)):
        subset = [n for n in names if pred(n)]
        if subset:
            by_type[kind] = round(_cat(subset), 6)
    return {"draws": draws, "cosine": round(_cat(names), 6),
            "cosineNoHead": round(_cat(body), 6) if body else None,
            "byType": by_type, "seconds": round(seconds, 2),
            "gradNorm": round(float(np.linalg.norm(np.concatenate(
                [np.asarray(grads[n], dtype=np.float64).reshape(-1) for n in names]))), 3)}


def loss_arm(model_cfg, cfg, draws, *, rounds, lr, outer_lr, outer_momentum,
             max_abs_grad, seed, batch, seq_len):
    torch.manual_seed(seed)
    model = GPT(model_cfg)
    coord = Coordinator(
        store=ModelStore(Path(tempfile.mkdtemp(prefix="v15-draws-"))),
        config=CoordConfig(lr=lr, outer_lr=outer_lr, outer_momentum=outer_momentum,
                           min_submissions=1, max_abs_grad=max_abs_grad),
        model_meta={"name": f"draws-{draws}"},
    )
    coord.bootstrap(tensors_of(model))
    gen = torch.Generator().manual_seed(seed + 7)
    idx, tgt = identity_batch(batch, seq_len, model_cfg.vocab_size, torch.device("cpu"), gen)
    losses = [clean_loss(model, idx, tgt)]
    uploaded = 0
    started = time.perf_counter()
    for rnd in range(rounds):
        grads, stats = averaged_draws(model, cfg, idx, tgt,
                                      base_seed=seed + rnd * 1_000_003, draws=draws)
        names, sizes, vector = protocol.flatten(grads)
        header = protocol.build_header(
            route="dust", client_id="bench", round_id=coord.round_id,
            base_sha256=coord.base_sha256, names=names, sizes=sizes, codec_name="ngd1",
            tokens=int(stats["tokens"]), population=int(stats["population"]),
            sigma=float(stats["sigma"]), gamma=float(stats["gamma"]),
            draws=int(stats["draws"]), baseSeed=int(seed + rnd * 1_000_003),
            seed=seed, step=0, localSteps=1, averagedDraws=int(draws),
            params=int(sum(sizes)),
        )
        body = protocol.encode_body(header, vector)
        uploaded += len(body)
        if not coord.submit(body).get("accepted"):
            raise SystemExit(f"draws={draws}: rejected")
        load_into(model, coord.store.load_weights())
        losses.append(clean_loss(model, idx, tgt))
    return {"draws": draws, "rounds": rounds, "firstLoss": round(losses[1], 6),
            "finalLoss": round(losses[-1], 6), "bytesPerRound": uploaded // rounds,
            "seconds": round(time.perf_counter() - started, 2),
            "lossCurve": [round(x, 6) for x in losses]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("cosine", "loss"), default="cosine")
    parser.add_argument("--preset", default="10m")
    parser.add_argument("--draws", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--draw-chunk", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1e-2)
    parser.add_argument("--sigma-mode", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--sigma-rel", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--task", choices=("identity", "induction"), default="induction")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--outer-momentum", type=float, default=0.9)
    parser.add_argument("--max-abs-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--compute-dtype", default="float32")
    args = parser.parse_args(argv)

    model_cfg = ModelConfig.preset(args.preset)
    cfg = DustConfig(population=args.population, draw_chunk=args.draw_chunk,
                     sigma=args.sigma, sigma_mode=args.sigma_mode,
                     sigma_rel=args.sigma_rel, gamma=args.gamma)
    arms = []
    for draws in args.draws:
        if args.mode == "cosine":
            arm = cosine_arm(model_cfg, cfg, draws, seed=args.seed, batch=args.batch,
                             seq_len=args.seq_len, task=args.task)
            print(f"draws={draws:<3} cosine {arm['cosine']:.4f}  |g| {arm['gradNorm']:.3g}  "
                  f"{arm['seconds']:.1f}s")
        else:
            arm = loss_arm(model_cfg, cfg, draws, rounds=args.rounds, lr=args.lr,
                           outer_lr=args.outer_lr, outer_momentum=args.outer_momentum,
                           max_abs_grad=args.max_abs_grad, seed=args.seed,
                           batch=args.batch, seq_len=args.seq_len)
            print(f"draws={draws:<3} final {arm['finalLoss']:.4f}  "
                  f"{arm['bytesPerRound']} B/round  {arm['seconds']:.1f}s")
        arms.append(arm)

    payload = {"schema": "crowd-v15-draws-bench-1", "mode": args.mode, "preset": args.preset,
               "population": args.population, "task": args.task, "arms": arms}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
