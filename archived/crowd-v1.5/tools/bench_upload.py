#!/usr/bin/env python3
"""Is the DUST estimate compressible by *structure* rather than by quantizing it?

A DUST weight update is an outer-product sum over tokens:

    G_W = sum_t (r_t * a_t) x_t^T          ->  G_W = A X^T,  A in R^{out x T}

so the estimate is **exactly rank <= T**, where T is the number of tokens the
client scored. A client that has the activations can therefore upload two thin
factors instead of the dense matrix: ``r * (out + in) * 2`` bytes for rank r,
fp16. That is a different lever from the 1-bit codec (which quantizes every
entry), and the two compose.

This tool measures the loss cost of that truncation against the 1-bit codec, with
byte accounting taken from the real encoders. It re-implements
``Coordinator.finalize`` (unit direction, capped magnitude, one step) rather than
going through HTTP, because the wire format does not carry factors yet; the
end-to-end HTTP path is what ``bench_feedback.py`` covers.

    python tools/bench_upload.py --rounds 30 --ranks 2 4 8 16
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

from crowd_v15 import codec, protocol  # noqa: E402
from crowd_v15.dust import DustConfig, dust_gradients  # noqa: E402
from crowd_v15.feedback import ErrorFeedback, quantize  # noqa: E402
from crowd_v15.model import GPT, ModelConfig, canonical_params  # noqa: E402
from crowd_v15.tasks import identity_batch  # noqa: E402


def _low_rank(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Truncated SVD; returns fp16 factors and the rank actually used."""

    u, s, vh = np.linalg.svd(matrix.astype(np.float32), full_matrices=False)
    r = int(min(rank, s.size))
    left = (u[:, :r] * s[:r]).astype(np.float16)
    right = vh[:r, :].astype(np.float16)
    return left, right, r


def encode_lowrank(grads: dict[str, torch.Tensor], rank: int) -> tuple[dict[str, np.ndarray], int]:
    """Reconstruct the estimate from fp16 factors, and count the bytes it costs."""

    rebuilt: dict[str, np.ndarray] = {}
    total = 0
    for name, tensor in grads.items():
        dense = tensor.detach().cpu().float().numpy()
        if dense.ndim == 2:
            left, right, r = _low_rank(dense, rank)
            rebuilt[name] = (left.astype(np.float32) @ right.astype(np.float32))
            total += int(left.size + right.size) * 2
            del left, right
        else:
            flat = dense.astype(np.float16)
            rebuilt[name] = flat.astype(np.float32)
            total += int(flat.size) * 2
    return rebuilt, total


def flatten_like(rebuilt: dict[str, np.ndarray], reference: dict[str, torch.Tensor]):
    names = sorted(reference)
    sizes = [int(reference[n].numel()) for n in names]
    vector = np.concatenate([np.asarray(rebuilt[n], dtype=np.float32).reshape(-1) for n in names])
    return names, sizes, vector


def arm_vector(grads, names, sizes, flat, arm: dict, engine, feedback_engine):
    """Return (vector to apply, bytes uploaded)."""

    if arm["kind"] == "lowrank":
        rebuilt, nbytes = encode_lowrank(grads, arm["rank"])
        _, _, vector = flatten_like(rebuilt, grads)
        return vector, nbytes

    if arm["kind"] == "dense":
        blob = codec.encode_dense(flat)
        decoded = codec.decode_dense(blob, flat.size)
        return decoded, len(blob)

    if arm["kind"] == "quantized":
        if feedback_engine is not None:
            sent, _ = feedback_engine.encode(flat)
        else:
            sent = quantize(flat, arm["codec"])
        header = protocol.build_header(
            route="dust",
            client_id="bench",
            round_id=0,
            base_sha256="0" * 64,
            names=names,
            sizes=sizes,
            codec_name=arm["codec"],
            tokens=1,
            population=1,
            sigma=1e-3,
            gamma=0.9,
        )
        body = protocol.encode_body(header, sent)
        return sent, len(body)

    raise ValueError(arm["kind"])


def run_arm(arm: dict, *, model_cfg: ModelConfig, rounds: int, cfg: DustConfig,
            batch: int, seq_len: int, lr: float, max_abs_grad: float, seed: int) -> dict:
    torch.manual_seed(seed)
    model = GPT(model_cfg)
    engine = ErrorFeedback(arm.get("codec", "ngd1")) if arm.get("feedback") else None
    generator = torch.Generator().manual_seed(seed + 7)
    idx, tgt = identity_batch(batch, seq_len, model_cfg.vocab_size, torch.device("cpu"), generator)

    def loss() -> float:
        with torch.no_grad():
            logits = model(idx, None)
            return float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)))

    losses = [loss()]
    uploaded = 0
    dense_bytes = 0
    started = time.time()
    params = canonical_params(model)

    for rnd in range(rounds):
        grads, _stats = dust_gradients(
            model, cfg, idx, tgt, base_seed=seed + rnd * 1_000_003, step=0
        )
        names, sizes, flat = protocol.flatten(grads)
        vector, nbytes = arm_vector(grads, names, sizes, flat, arm, None, engine)
        uploaded += nbytes
        dense_bytes += int(flat.size) * 2

        norm = float(np.linalg.norm(vector))
        magnitude = min(norm, max_abs_grad) if norm > 0 else 0.0
        direction = vector if norm == 0 else (vector / norm).astype(np.float32)
        update = direction * magnitude

        offset = 0
        with torch.no_grad():
            for name in names:
                size = sizes[names.index(name)]
                chunk = torch.from_numpy(update[offset: offset + size]).reshape(params[name].shape)
                params[name].sub_(lr * chunk)
                offset += size

        losses.append(loss())

    return {
        "arm": arm["name"],
        "rounds": rounds,
        "firstLoss": losses[1],
        "finalLoss": losses[-1],
        "bestLoss": min(losses[1:]),
        "uploadedBytes": uploaded,
        "denseBytes": dense_bytes,
        "compression": dense_bytes / max(1, uploaded),
        "bytesPerRound": uploaded // max(1, rounds),
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
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    model_cfg = ModelConfig.preset(args.preset)
    cfg = DustConfig(population=args.population, draw_chunk=args.draw_chunk,
                     sigma=args.sigma, gamma=args.gamma, lr=args.lr)
    tokens = args.batch * args.seq_len

    arms = [
        {"name": "dense", "kind": "dense"},
        {"name": "ngd1", "kind": "quantized", "codec": "ngd1"},
        {"name": "ngd1r", "kind": "quantized", "codec": "ngd1r"},
        {"name": "ngd1+ef", "kind": "quantized", "codec": "ngd1", "feedback": True},
    ]
    for rank in args.ranks:
        arms.append({"name": f"lowrank-r{rank}", "kind": "lowrank", "rank": rank})

    results = []
    for arm in arms:
        record = run_arm(arm, model_cfg=model_cfg, rounds=args.rounds, cfg=cfg,
                         batch=args.batch, seq_len=args.seq_len, lr=args.lr,
                         max_abs_grad=args.max_abs_grad, seed=args.seed)
        print(json.dumps({k: v for k, v in record.items() if k != "lossCurve"}), flush=True)
        results.append(record)

    payload = {
        "schema": "crowd-v15-upload-bench-1",
        "config": {
            "rounds": args.rounds, "preset": args.preset, "params": model_cfg.n_params,
            "batch": args.batch, "seqLength": args.seq_len, "tokensPerEstimate": tokens,
            "population": args.population, "sigma": args.sigma, "gamma": args.gamma,
            "lr": args.lr, "maxAbsGrad": args.max_abs_grad, "seed": args.seed,
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
