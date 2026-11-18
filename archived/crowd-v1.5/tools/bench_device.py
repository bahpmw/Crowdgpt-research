#!/usr/bin/env python3
"""Hardware probe: the README's per-device table, reproducible on any backend.

Measures, for one preset at one compute dtype:

- the weights' size on the device;
- one clean forward (default seq 8, batch 1);
- one DUST step (default seq 32, batch 1, population 2, per-layer passes);
- evidence throughput of that step (``population * batch * seq`` tokens / s).

The defaults match the shapes behind the gfx1150 and RTX 4060 Ti rows, so a new
machine's numbers drop straight into that comparison:

    python tools/bench_device.py --preset 1b --device mps
    python tools/bench_device.py --preset 1b --device cuda --noise-device device

Timings are wall clock after a device synchronize; ``--repeat`` takes the median.
Memory is ``max_memory_reserved`` on CUDA/ROCm and ``driver_allocated_memory`` on
MPS (live tensors plus the allocator cache, i.e. the high-water mark).
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_v15 import accel  # noqa: E402
from crowd_v15.dust import DustConfig, dust_gradients  # noqa: E402
from crowd_v15.model import GPT, ModelConfig  # noqa: E402
from crowd_v15.tasks import random_batch  # noqa: E402

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def timed(device: torch.device, fn, repeat: int) -> tuple[float, object]:
    times, result = [], None
    for _ in range(repeat):
        accel.synchronize(device)
        start = time.perf_counter()
        result = fn()
        accel.synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="1b")
    parser.add_argument("--device", default="auto", help="cuda (incl. ROCm), mps, cpu or auto")
    parser.add_argument("--compute-dtype", default="bfloat16", choices=tuple(_DTYPES))
    parser.add_argument("--forward-seq", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--population", type=int, default=2)
    parser.add_argument("--draw-chunk", type=int, default=2)
    parser.add_argument("--mode", choices=("layer", "all"), default="layer")
    parser.add_argument("--noise-device", choices=("cpu", "device"), default="cpu")
    parser.add_argument("--sigma-mode", choices=("absolute", "relative"), default="relative")
    parser.add_argument("--repeat", type=int, default=1, help="median over this many DUST steps")
    parser.add_argument("--warmup", type=int, default=1, help="untimed DUST steps first (kernel compile)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    device = accel.resolve_device(args.device)
    dtype = _DTYPES[args.compute_dtype]
    cfg = ModelConfig.preset(args.preset)
    torch.manual_seed(args.seed)

    build_start = time.perf_counter()
    model = GPT(cfg).to(device=device, dtype=dtype)
    model.eval()
    accel.synchronize(device)
    build_s = time.perf_counter() - build_start
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 3
    after_weights_gb = accel.peak_memory_gb(device)

    gen = torch.Generator().manual_seed(args.seed)
    fwd_idx, _ = random_batch(1, args.forward_seq, cfg.vocab_size, device, gen)
    with torch.no_grad():
        model(fwd_idx)  # warm the kernels
        forward_s, _ = timed(device, lambda: model(fwd_idx), 3)
    forward_gb = accel.peak_memory_gb(device)

    idx, tgt = random_batch(args.batch, args.seq_len, cfg.vocab_size, device, gen)
    dust_cfg = DustConfig(population=args.population, draw_chunk=args.draw_chunk,
                          mode=args.mode, noise_device=args.noise_device,
                          sigma_mode=args.sigma_mode)
    for step in range(args.warmup):
        dust_gradients(model, dust_cfg, idx, tgt, base_seed=args.seed, step=step)
    step_s, (grads, stats) = timed(
        device, lambda: dust_gradients(model, dust_cfg, idx, tgt,
                                       base_seed=args.seed, step=args.warmup),
        args.repeat)
    step_gb = accel.peak_memory_gb(device)
    grad_gb = sum(g.numel() * g.element_size() for g in grads.values()) / 1024 ** 3
    finite = all(bool(torch.isfinite(g).all()) for g in grads.values())

    record = {
        "preset": args.preset, "params": cfg.n_params, "device": str(device),
        "deviceName": accel.device_name(device) or platform.processor(),
        "torch": torch.__version__, "computeDtype": args.compute_dtype,
        "weightsGb": round(weights_gb, 3), "buildSeconds": round(build_s, 2),
        "memAfterWeightsGb": after_weights_gb,
        "forwardSeq": args.forward_seq, "forwardSeconds": round(forward_s, 4),
        "memAfterForwardGb": forward_gb,
        "dust": {"seqLen": args.seq_len, "batch": args.batch, "population": args.population,
                 "drawChunk": args.draw_chunk, "mode": args.mode, "noiseDevice": args.noise_device,
                 "sigmaMode": args.sigma_mode, "repeat": args.repeat},
        "stepSeconds": round(step_s, 3), "forwards": stats["forwards"],
        "forwardsPerSecond": round(stats["forwards"] / step_s, 1),
        "tokens": stats["tokens"], "tokensPerSecond": round(stats["tokens"] / step_s, 1),
        "memAfterStepGb": step_gb, "gradBufferGb": round(grad_gb, 3),
        "gradNorm": stats["gradNorm"], "gradFinite": finite,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
