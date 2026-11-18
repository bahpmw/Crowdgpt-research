#!/usr/bin/env python3
"""Does carrying the quantization error improve a real round loop?

Four arms see the same batch plan, the same starting weights and the same
coordinator settings. The only difference is what is put on the wire:

    dense   -> bf16    (unbiased, big)
    ngd1    -> 1 bit   (biased, small)
    ngd1r   -> 2 bits  (biased, small, less error)
    ngd1+ef -> ngd1 with the previous round's error added back before encoding

The coordinator is the real one, reached through the real encode/decode path, so
the numbers include the unit-direction averaging that ``finalize`` performs -- the
part that decides whether error feedback can help at all.

    python tools/bench_feedback.py --rounds 30
"""

from __future__ import annotations

import argparse
import json
import sys
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
from crowd_v15.feedback import ErrorFeedback  # noqa: E402
from crowd_v15.model import GPT, ModelConfig, canonical_params  # noqa: E402
from crowd_v15.tasks import identity_batch  # noqa: E402

ARMS = (
    {"name": "dense", "codec": "ngd1", "feedback": False, "codecName": "dense"},
    {"name": "ngd1", "codec": "ngd1", "feedback": False},
    {"name": "ngd1r", "codec": "ngd1r", "feedback": False},
    {"name": "ngd1+ef", "codec": "ngd1", "feedback": True},
    {"name": "ngd1r+ef", "codec": "ngd1r", "feedback": True},
)


def tensors_of(model: GPT) -> dict[str, np.ndarray]:
    return {
        name: param.detach().cpu().float().numpy().copy()
        for name, param in canonical_params(model).items()
    }


def load_into(model: GPT, tensors: dict[str, np.ndarray]) -> None:
    with torch.no_grad():
        for name, param in canonical_params(model).items():
            param.copy_(torch.from_numpy(np.asarray(tensors[name], dtype=np.float32)))


def clean_loss(model: GPT, idx: torch.Tensor, tgt: torch.Tensor) -> float:
    with torch.no_grad():
        st = None
        logits = model(idx, st)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
    return float(loss)


def run_arm(arm: dict, *, model_cfg: ModelConfig, rounds: int, cfg: DustConfig,
            batch: int, seq_len: int, lr: float, max_abs_grad: float, seed: int) -> dict:
    torch.manual_seed(seed)
    model = GPT(model_cfg)
    coord = _coordinator(lr, max_abs_grad, arm["name"])
    coord.bootstrap(tensors_of(model))
    codec_name = arm.get("codecName", arm["codec"])
    engine = ErrorFeedback(codec_name) if arm["feedback"] else None

    generator = torch.Generator().manual_seed(seed + 7)
    idx, tgt = identity_batch(batch, seq_len, model_cfg.vocab_size, torch.device("cpu"), generator)
    losses = [clean_loss(model, idx, tgt)]
    uploaded = 0
    dense_bytes = 0
    residuals = []
    started = time.time()

    for rnd in range(rounds):
        base_seed = seed + rnd * 1_000_003
        grads, stats = dust_gradients(model, cfg, idx, tgt, base_seed=base_seed, step=0)
        names, sizes, vector = protocol.flatten(grads)
        residual_norm = None
        if engine is not None:
            vector, fb_stats = engine.encode(vector)
            residual_norm = fb_stats["residualNorm"]
            residuals.append(residual_norm)
        header = protocol.build_header(
            route="dust",
            client_id="bench",
            round_id=coord.round_id,
            base_sha256=coord.base_sha256,
            names=names,
            sizes=sizes,
            codec_name=codec_name,
            tokens=int(stats["tokens"]),
            population=int(stats["population"]),
            sigma=float(stats["sigma"]),
            gamma=float(stats["gamma"]),
            draws=int(stats["draws"]),
            baseSeed=int(base_seed),
            seed=seed,
            step=0,
            localSteps=1,
            params=int(sum(sizes)),
        )
        body = protocol.encode_body(header, vector)
        uploaded += len(body)
        dense_bytes += int(vector.size) * 2
        reply = coord.submit(body)
        if not reply.get("accepted"):
            raise SystemExit(f"{arm['name']}: rejected {reply}")
        load_into(model, coord.store.load_weights())
        losses.append(clean_loss(model, idx, tgt))

    return {
        "arm": arm["name"],
        "codec": codec_name,
        "feedback": bool(arm["feedback"]),
        "rounds": rounds,
        "firstLoss": losses[1],
        "finalLoss": losses[-1],
        "bestLoss": min(losses[1:]),
        "delta": losses[-1] - losses[1],
        "uploadedBytes": uploaded,
        "denseBytes": dense_bytes,
        "compression": dense_bytes / max(1, uploaded),
        "meanResidual": float(np.mean(residuals)) if residuals else None,
        "seconds": round(time.time() - started, 2),
        "lossCurve": [round(x, 6) for x in losses],
    }


def _coordinator(lr: float, max_abs_grad: float, name: str):
    """A coordinator with an in-memory store, so no files are written."""

    from crowd_v15.store import ModelStore
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="v15-bench-"))
    return Coordinator(
        store=ModelStore(root),
        config=CoordConfig(lr=lr, min_submissions=1, max_abs_grad=max_abs_grad),
        model_meta={"name": name},
    )


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
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    model_cfg = ModelConfig.preset(args.preset)
    cfg = DustConfig(
        population=args.population,
        draw_chunk=args.draw_chunk,
        sigma=args.sigma,
        gamma=args.gamma,
        lr=args.lr,
    )
    results = []
    for arm in ARMS:
        record = run_arm(
            arm,
            model_cfg=model_cfg,
            rounds=args.rounds,
            cfg=cfg,
            batch=args.batch,
            seq_len=args.seq_len,
            lr=args.lr,
            max_abs_grad=args.max_abs_grad,
            seed=args.seed,
        )
        print(json.dumps({k: v for k, v in record.items() if k != "lossCurve"}), flush=True)
        results.append(record)

    payload = {
        "schema": "crowd-v15-feedback-bench-1",
        "config": {
            "rounds": args.rounds,
            "preset": args.preset,
            "params": model_cfg.n_params,
            "batch": args.batch,
            "seqLength": args.seq_len,
            "population": args.population,
            "sigma": args.sigma,
            "gamma": args.gamma,
            "lr": args.lr,
            "maxAbsGrad": args.max_abs_grad,
            "seed": args.seed,
            "task": "identity",
        },
        "arms": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
