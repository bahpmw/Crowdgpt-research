#!/usr/bin/env python3
"""Does DUST's gradient estimate keep pointing downhill as the model grows?

The README's first open gate: the estimator was only validated on tiny CPU
models. This measures the cosine between a DUST estimate and the exact
backprop gradient on *real tokens*, at the same weights and the same batch,
across presets and population sizes.

The head's gradient is exact by construction (softmax), so ``cosine`` with the
head included is flattered by it; ``cosineNoHead`` is the number that answers
whether the zeroth-order part scales. Each population is a prefix of the same
draws (noise is a pure function of the draw index), so a larger population is
strictly more evidence on the same batch.

    python tools/bench_alignment.py --presets tiny 10m 100m 300m 1b \
        --populations 8 32 128 --data runs/data/tok --out runs/align/size.json
    # alignment as training proceeds: probe a scale_test checkpoint
    python tools/bench_alignment.py --presets 10m --checkpoint runs/scale/dust.ckpt.pt
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

from crowd_v15 import accel  # noqa: E402
from crowd_v15.dust import (DustConfig, _head_name, backprop_gradients,  # noqa: E402
                            dust_gradients, per_token_ce)
from crowd_v15.model import GPT, ModelConfig, canonical_params  # noqa: E402
from tools.scale_test import window_batch  # noqa: E402

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def _cos(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom == 0.0 else float(a @ b / denom)


def compare(estimate: dict, reference: dict, head_key: str) -> dict:
    """Cosines between two gradient dicts, overall and by parameter group."""
    names = sorted(n for n in estimate if n in reference)

    def flat(src, subset):
        return np.concatenate([src[n].detach().float().cpu().numpy().astype(np.float64).reshape(-1)
                               for n in subset])

    groups = {
        "all": names,
        "noHead": [n for n in names if n != head_key],
        "attn": [n for n in names if ".attn." in n],
        "mlp": [n for n in names if ".mlp." in n],
    }
    out = {}
    for label, subset in groups.items():
        if subset:
            out[label] = _cos(flat(estimate, subset), flat(reference, subset))
    per_layer = [_cos(flat(estimate, [n]), flat(reference, [n]))
                 for n in names if n != head_key and estimate[n].dim() == 2]
    per_layer = [c for c in per_layer if c is not None]
    out["perLayerMedian"] = float(np.median(per_layer)) if per_layer else None
    return out


def probe(model: GPT, cfg: DustConfig, idx, tgt, *, seed: int, reference: dict,
          device) -> dict:
    accel.synchronize(device)
    started = time.perf_counter()
    grads, stats = dust_gradients(model, cfg, idx, tgt, base_seed=seed, step=0)
    accel.synchronize(device)
    seconds = time.perf_counter() - started
    cos = compare(grads, reference, _head_name(model))
    return {"population": cfg.population, "seconds": round(seconds, 3),
            "forwards": stats["forwards"], "sigmaMin": stats["sigma"],
            "sigmaMax": stats["sigmaMax"],
            **{f"cosine{k[0].upper()}{k[1:]}" if k != "all" else "cosine":
               (None if v is None else round(v, 5)) for k, v in cos.items()}}


def run_preset(preset: str, args, device) -> dict:
    model_cfg = ModelConfig.preset(preset)
    if model_cfg.vocab_size < 4096:
        raise SystemExit(f"{preset} has vocab {model_cfg.vocab_size}; real tokens need 4096")
    dtype = _DTYPES[args.compute_dtype]
    torch.manual_seed(args.seed)
    model = GPT(model_cfg)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    model = model.to(device=device, dtype=dtype).eval()
    params = canonical_params(model)
    n_params = int(sum(p.numel() for p in params.values()))
    val_bin = str(Path(args.data) / "val.bin")

    batches = []
    for rep in range(args.repeats):
        idx, tgt = window_batch(val_bin, rep, args.batch, args.seq_len, args.seed + 17)
        batches.append((idx.to(device), tgt.to(device)))

    rows = []
    for rep, (idx, tgt) in enumerate(batches):
        with torch.no_grad():
            loss = float(per_token_ce(model(idx, None).float(), tgt).mean())
        # The reference is computed at the dtype the estimate is measured at, so
        # the cosine measures the estimator rather than bf16-vs-fp32 rounding.
        reference = backprop_gradients(model, idx, tgt)
        for pop in args.populations:
            cfg = DustConfig(population=pop, draw_chunk=min(args.draw_chunk, pop),
                             sigma=args.sigma, sigma_mode=args.sigma_mode,
                             sigma_rel=args.sigma_rel, gamma=args.gamma, mode=args.mode,
                             noise_device=args.noise_device)
            row = probe(model, cfg, idx, tgt, seed=args.seed + 1000 * rep,
                        reference=reference, device=device)
            row.update({"repeat": rep, "loss": round(loss, 4)})
            rows.append(row)
            print(f"{preset:>5} rep {rep} pop {pop:>4}  noHead {row['cosineNoHead']:+.4f}  "
                  f"attn {row['cosineAttn']:+.4f}  mlp {row['cosineMlp']:+.4f}  "
                  f"all {row['cosine']:+.4f}  {row['seconds']:.1f}s", flush=True)
        del reference
        if device.type == "mps":
            torch.mps.empty_cache()

    summary = []
    for pop in args.populations:
        mine = [r for r in rows if r["population"] == pop]
        vals = [r["cosineNoHead"] for r in mine]
        summary.append({
            "population": pop,
            "cosineNoHead": round(float(np.mean(vals)), 5),
            "cosineNoHeadStd": round(float(np.std(vals)), 5),
            "cosineAttn": round(float(np.mean([r["cosineAttn"] for r in mine])), 5),
            "cosineMlp": round(float(np.mean([r["cosineMlp"] for r in mine])), 5),
            "perLayerMedian": round(float(np.mean([r["cosinePerLayerMedian"] for r in mine])), 5),
            "cosine": round(float(np.mean([r["cosine"] for r in mine])), 5),
            "seconds": round(float(np.mean([r["seconds"] for r in mine])), 3),
        })
    out = {"preset": preset, "params": n_params, "layers": model_cfg.n_layers,
           "dModel": model_cfg.d_model, "rows": rows, "summary": summary}
    del model, params, batches
    if device.type == "mps":
        torch.mps.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--presets", nargs="+", default=["10m", "100m", "300m"])
    parser.add_argument("--populations", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--data", default="runs/data/tok")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default=accel.default_device())
    parser.add_argument("--compute-dtype", choices=sorted(_DTYPES), default="float32")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--draw-chunk", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1e-2)
    parser.add_argument("--sigma-mode", choices=("absolute", "relative"), default="relative")
    parser.add_argument("--sigma-rel", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--mode", choices=("layer", "all"), default="layer")
    parser.add_argument("--noise-device", choices=("cpu", "device"), default="device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    device = accel.resolve_device(args.device)
    results = []
    started = time.perf_counter()
    for preset in args.presets:
        results.append(run_preset(preset, args, device))
        if args.out:
            _write(args, device, results, started)
    print("\npreset    params  " + "  ".join(f"pop{p:<5}" for p in args.populations)
          + "  (cosine, no head)")
    for res in results:
        cells = "  ".join(f"{s['cosineNoHead']:+.4f} " for s in res["summary"])
        print(f"{res['preset']:>6} {res['params'] / 1e6:8.1f}M  {cells}")
    if args.out:
        _write(args, device, results, started)
        print("wrote", args.out)
    return 0


def _write(args, device, results, started) -> None:
    payload = {"schema": "crowd-v15-alignment-1", "device": accel.device_name(device),
               "computeDtype": args.compute_dtype, "batch": args.batch,
               "seqLen": args.seq_len, "repeats": args.repeats,
               "sigmaMode": args.sigma_mode, "sigmaRel": args.sigma_rel,
               "sigma": args.sigma, "gamma": args.gamma, "mode": args.mode,
               "noiseDevice": args.noise_device, "checkpoint": args.checkpoint or None,
               "data": args.data, "seconds": round(time.perf_counter() - started, 1),
               "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
