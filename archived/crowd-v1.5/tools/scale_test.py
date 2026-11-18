#!/usr/bin/env python3
"""Train one arm (DUST or backprop Adam) on real tokens and log a loss curve.

The point is a comparison at equal tokens: a DUST client and a backprop client
consume the same FineWeb windows at the same step index, so the two curves are
paired rather than merely similar. Every token the DUST arm counts is a token
it actually scored (``batch * seq``), not the population it evaluated.

    python tools/scale_test.py --preset 10m --arm dust --minutes 60 \
        --data /tmp/v15data/tok --out runs/scale --device cuda
    python tools/scale_test.py --preset 10m --arm adam --minutes 60 \
        --data /tmp/v15data/tok --out runs/scale --device cuda

Writes ``<out>/<arm>.jsonl`` (one record per log interval), ``<out>/<arm>.ckpt.pt``
every ``--ckpt-seconds`` and on exit, and ``<out>/<arm>.summary.json``.

The DUST arm goes through the real ``Coordinator`` (store, header, codec, unit
direction, cap, outer optimizer); the Adam arm is plain PyTorch so it is the
strong baseline rather than a strawman.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_v15 import accel, protocol  # noqa: E402
from crowd_v15.coordinator import CoordConfig, Coordinator, Rejected  # noqa: E402
from crowd_v15.dust import DustConfig, dust_gradients  # noqa: E402
from crowd_v15.model import GPT, ModelConfig, canonical_params  # noqa: E402
from crowd_v15.store import ModelStore  # noqa: E402

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def window_batch(path: str, step: int, batch: int, seq_len: int, seed: int):
    """Deterministic windows, so both arms see the same tokens at a step."""
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    span = int(tokens.size) - (seq_len + 1)
    if span <= 0:
        raise SystemExit(f"{path} is too short for seq {seq_len}")
    generator = torch.Generator().manual_seed((int(seed) * 1_000_003 + int(step)) & 0x7FFFFFFF)
    offsets = torch.randint(0, span, (batch,), generator=generator).tolist()
    rows = [np.asarray(tokens[o:o + seq_len + 1]) for o in offsets]
    block = np.stack(rows).astype(np.int64)
    return torch.from_numpy(block[:, :-1]), torch.from_numpy(block[:, 1:])


def eval_loss(model, path: str, *, seq_len: int, windows: int, batch: int = 8) -> float:
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    available = int(tokens.size) // (seq_len + 1)
    windows = min(windows, available)
    rows = [np.asarray(tokens[i * (seq_len + 1):(i + 1) * (seq_len + 1)]) for i in range(windows)]
    block = np.stack(rows).astype(np.int64)
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, windows, batch):
            piece = block[start:start + batch]
            idx = torch.from_numpy(piece[:, :-1]).to(next(model.parameters()).device)
            tgt = torch.from_numpy(piece[:, 1:]).to(idx.device)
            logits = model(idx)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1), reduction="sum")
            total += float(loss)
            count += int(tgt.numel())
    model.train()
    return total / max(1, count)


def peak_memory_gb(device) -> float | None:
    return accel.peak_memory_gb(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("dust", "adam"), required=True)
    parser.add_argument("--preset", default="10m")
    parser.add_argument("--data", default="/tmp/v15data/tok")
    parser.add_argument("--out", default="runs/scale")
    parser.add_argument("--device", default=accel.default_device(),
                        help="cuda (incl. ROCm), mps, cpu or auto")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--tokens", type=int, default=0, help="stop after this many tokens (0 = no cap)")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--adam-lr", type=float, default=3e-4)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--draw-chunk", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=1e-2)
    parser.add_argument("--sigma-mode", choices=("absolute", "relative"), default="relative")
    parser.add_argument("--sigma-rel", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--mode", choices=("layer", "all"), default="layer")
    parser.add_argument("--noise-device", choices=("cpu", "device"), default="cpu",
                        help="generate the perturbation noise on the accelerator (fast, device-specific)"
                             " or on the CPU (slow, bit-identical on every device)")
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--outer-momentum", type=float, default=0.9)
    parser.add_argument("--max-abs-grad", type=float, default=1.0)
    parser.add_argument("--codec", default="ngd1r", choices=protocol.codec.CODECS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=2_000_000)
    parser.add_argument("--eval-windows", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-seconds", type=float, default=1800.0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    args = parser.parse_args(argv)

    device = accel.resolve_device(args.device)
    dtype = _DTYPES[args.compute_dtype]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = (out_dir / f"{args.arm}.jsonl").open("a", encoding="utf-8")
    meta_path = out_dir / f"{args.arm}.meta.json"

    model_cfg = ModelConfig.preset(args.preset)
    torch.manual_seed(args.seed)
    model = GPT(model_cfg)
    # DUST runs the served weights in bf16; Adam keeps fp32 weights with bf16
    # autocast, which is what its checkpoint would be served from.
    if args.arm == "dust":
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device, dtype=torch.float32)

    train_bin = str(Path(args.data) / "train.bin")
    val_bin = str(Path(args.data) / "val.bin")
    params = canonical_params(model)
    n_params = int(sum(p.numel() for p in params.values()))

    coord = None
    optimizer = None
    store_dir = None
    start_round = 0
    step = 0
    tokens_seen = 0
    resumed = False

    ckpt_path = out_dir / f"{args.arm}.ckpt.pt"
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        step = int(state.get("step", 0))
        tokens_seen = int(state.get("tokens", 0))
        start_round = int(state.get("round", 0))
        resumed = True
        print(f"resumed at step {step}, {tokens_seen} tokens", flush=True)

    if args.arm == "dust":
        store_dir = Path(tempfile.mkdtemp(prefix="v15-scale-store-"))
        coord = Coordinator(
            store=ModelStore(store_dir),
            config=CoordConfig(lr=args.lr, outer_lr=args.outer_lr,
                               outer_momentum=args.outer_momentum,
                               min_submissions=1, max_abs_grad=args.max_abs_grad),
        )
        flat = {name: p.detach().float().cpu().numpy() for name, p in canonical_params(model).items()}
        coord.bootstrap(flat)
        while coord.round_id < start_round:
            # Rewinding a resumed run: the store only holds the current weights,
            # so replay empty rounds until the counter matches the checkpoint.
            coord._state["round"] = int(coord._state.get("round", 0)) + 1
        if start_round:
            coord._state["round"] = start_round
            coord._persist()
        dust_cfg = DustConfig(population=args.population, draw_chunk=args.draw_chunk,
                              sigma=args.sigma, sigma_mode=args.sigma_mode,
                              sigma_rel=args.sigma_rel, gamma=args.gamma, mode=args.mode,
                              noise_device=args.noise_device)
    else:
        optimizer = torch.optim.AdamW(list(params.values()), lr=args.adam_lr,
                                      betas=(0.9, 0.95), weight_decay=0.01)
        if ckpt_path.is_file():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if state.get("optim"):
                optimizer.load_state_dict(state["optim"])
                for group in optimizer.param_groups:
                    group["lr"] = args.adam_lr

    meta = {
        "arm": args.arm, "preset": args.preset, "params": n_params,
        "device": str(device), "computeDtype": args.compute_dtype,
        "batch": args.batch, "seqLen": args.seq_len,
        "noiseDevice": args.noise_device,
        "population": args.population if args.arm == "dust" else None,
        "drawChunk": args.draw_chunk if args.arm == "dust" else None,
        "sigma": args.sigma if args.arm == "dust" else None,
        "sigmaMode": args.sigma_mode if args.arm == "dust" else None,
        "sigmaRel": args.sigma_rel if args.arm == "dust" else None,
        "gamma": args.gamma if args.arm == "dust" else None,
        "mode": args.mode if args.arm == "dust" else None,
        "lr": args.lr if args.arm == "dust" else args.adam_lr,
        "outerLr": args.outer_lr if args.arm == "dust" else None,
        "codec": args.codec if args.arm == "dust" else None,
        "seed": args.seed, "startedAt": time.time(), "resumed": resumed,
        "torch": torch.__version__,
        "gpu": accel.device_name(device),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, sort_keys=True), flush=True)

    tokens_per_step = args.batch * args.seq_len
    started = time.perf_counter()
    # A resumed run must report this session's throughput, not every token the
    # checkpoint has ever seen (that divided 65,536 inherited tokens by 179 s and
    # published 388 tok/s for a 23 tok/s machine).
    session_start_tokens = tokens_seen
    next_eval = max(args.eval_every, tokens_seen + args.eval_every)
    next_ckpt = time.perf_counter() + args.ckpt_seconds
    recent: list[float] = []
    best: float | None = None

    def checkpoint() -> None:
        payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "step": step, "tokens": tokens_seen,
                   "round": int(coord.round_id) if coord is not None else 0}
        if optimizer is not None:
            payload["optim"] = optimizer.state_dict()
        torch.save(payload, ckpt_path)

    while True:
        elapsed = time.perf_counter() - started
        if elapsed / 60.0 >= args.minutes:
            print("stopping: time budget reached", flush=True)
            break
        if args.tokens and tokens_seen >= args.tokens:
            print("stopping: token budget reached", flush=True)
            break

        idx, tgt = window_batch(train_bin, step, args.batch, args.seq_len, args.seed)
        idx = idx.to(device)
        tgt = tgt.to(device)
        step_started = time.perf_counter()

        if args.arm == "dust":
            grads, stats = dust_gradients(model, dust_cfg, idx, tgt,
                                          base_seed=args.seed + step * 1_000_003, step=step)
            names, sizes, vector = protocol.flatten(grads)
            header = protocol.build_header(
                route="dust", client_id="scale-test", round_id=coord.round_id,
                base_sha256=coord.base_sha256, names=names, sizes=sizes,
                codec_name=args.codec, tokens=int(stats["tokens"]),
                population=int(stats["population"]), sigma=float(stats["sigma"]),
                gamma=float(stats["gamma"]), step=step,
            )
            body = protocol.encode_body(header, vector)
            try:
                receipt = coord.submit(body)
            except Rejected as exc:
                print(f"rejected at step {step}: {exc.reason} {exc.detail}", flush=True)
                break
            weights = coord.store.load_weights()
            with torch.no_grad():
                for name, param in canonical_params(model).items():
                    param.copy_(torch.from_numpy(weights[name]).to(param.device, param.dtype))
            # The coordinator applies the step, so the arm has no local loss unless
            # it scores the batch it just used. Without this the DUST arm logs no
            # train loss at all (`recent` stays empty and the per-step record is
            # suppressed), which silently produced an empty jsonl for a whole run.
            with torch.no_grad():
                logits = model(idx)
                loss = float(torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)))
            upload_bytes = len(body)
            extra = {"gradNorm": receipt.get("gradNorm"), "uploadBytes": upload_bytes,
                     "declaredNorm": receipt.get("declaredNorm"),
                     "population": stats["population"], "forwards": stats["forwards"],
                     "sigma": stats["sigma"], "sigmaMax": stats["sigmaMax"]}
        else:
            warm = min(1.0, (step + 1) / max(1, args.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = args.adam_lr * warm
            with accel.autocast(device):
                logits = model(idx)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
            optimizer.step()
            loss = float(loss.detach())
            upload_bytes = 0
            extra = {}

        seconds = time.perf_counter() - step_started
        step += 1
        tokens_seen += tokens_per_step
        if loss is not None:
            recent.append(float(loss))
            recent = recent[-50:]

        logged = (step % args.log_every == 0)
        if tokens_seen >= next_eval:
            eval_value = eval_loss(model, val_bin, seq_len=args.seq_len, windows=args.eval_windows)
            best = eval_value if best is None else min(best, eval_value)
            record = {
                "arm": args.arm, "step": step, "tokens": tokens_seen,
                "evalLoss": round(eval_value, 6),
                "trainLoss": round(sum(recent) / len(recent), 6) if recent else None,
                "trainLossInstant": round(float(loss), 6) if loss is not None else None,
                "bestEvalLoss": round(best, 6),
                "seconds": round(time.perf_counter() - started, 1),
                "tokensPerSecond": round((tokens_seen - session_start_tokens) / max(1e-9, time.perf_counter() - started), 1),
                "sessionTokens": tokens_seen - session_start_tokens,
                "peakVramGb": peak_memory_gb(device),
                "uploadBytes": upload_bytes or None,
                **extra,
            }
            jsonl.write(json.dumps(record, sort_keys=True) + "\n")
            jsonl.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            next_eval += args.eval_every
        elif logged and recent:
            record = {
                "arm": args.arm, "step": step, "tokens": tokens_seen,
                "trainLoss": round(sum(recent) / len(recent), 6),
                "trainLossInstant": round(float(loss), 6) if loss is not None else None,
                "seconds": round(time.perf_counter() - started, 1),
                "tokensPerSecond": round((tokens_seen - session_start_tokens) / max(1e-9, time.perf_counter() - started), 1),
                "sessionTokens": tokens_seen - session_start_tokens,
                "peakVramGb": peak_memory_gb(device),
                "stepSeconds": round(seconds, 3),
                "uploadBytes": upload_bytes or None,
                **extra,
            }
            jsonl.write(json.dumps(record, sort_keys=True) + "\n")
            jsonl.flush()
            print(json.dumps(record, sort_keys=True), flush=True)

        if time.perf_counter() >= next_ckpt:
            checkpoint()
            next_ckpt = time.perf_counter() + args.ckpt_seconds

        # A DUST step is far more expensive than an Adam step; keep the loop
        # responsive to the time budget without a sleep.
        if step > 0 and step % 500 == 0:
            print(f"... step {step}, {tokens_seen/1e6:.1f}M tokens", flush=True)

    checkpoint()
    summary = {"arm": args.arm, "step": step, "tokens": tokens_seen,
               "bestEvalLoss": best, "seconds": round(time.perf_counter() - started, 1),
               "peakVramGb": peak_memory_gb(device), "meta": meta}
    (out_dir / f"{args.arm}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    jsonl.close()
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
