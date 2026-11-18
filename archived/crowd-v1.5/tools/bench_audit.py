#!/usr/bin/env python3
"""Measure the audit primitive: honest vs fabricated re-score cosines, and its cost.

An honest client and several cheats each produce one DUST submission from the
same served weights; the coordinator-side ``audit.rescore`` re-computes it on
the CPU and reports the cosine. The honest row should sit near 1 and every
cheat near 0, across codecs and across a bf16 client on another accelerator.

    python tools/bench_audit.py --presets tiny 10m --devices cpu mps \
        --codecs dense ngd1r --out runs/audit/audit.json

Cheats:
    random      Gaussian noise with the honest estimate's per-tensor norms
    other_seed  a real DUST estimate under a different noise seed
    other_batch a real DUST estimate on a different batch
    backprop    the exact gradient (the head matches; the DUST layers do not)
    half        honest on a random half of the layers, noise on the rest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_v15 import accel, audit, protocol  # noqa: E402
from crowd_v15.client import ClientConfig, DustClient  # noqa: E402
from crowd_v15.dust import backprop_gradients, dust_gradients  # noqa: E402
from crowd_v15.model import canonical_params  # noqa: E402

SCHEMA = "crowd-v15-audit-bench-1"


def _weights(client: DustClient) -> dict[str, np.ndarray]:
    return {n: p.detach().float().cpu().numpy().copy()
            for n, p in canonical_params(client.model).items()}


def _decode(client: DustClient, grads, stats):
    body, _ = client.encode(grads, stats)
    return protocol.decode_body(body, max_params=client.model.n_params() + 1)


def _like(grads, gen):
    return {k: torch.randn(v.shape, generator=gen).to(v) * (v.float().norm() / max(1.0, v.numel()) ** 0.5)
            for k, v in grads.items()}


def _cheats(client: DustClient, grads, stats, rng_seed: int):
    gen = torch.Generator().manual_seed(rng_seed)
    idx, tgt = client.batch()
    other_seed, _ = dust_gradients(client.model, client.dust_cfg, idx, tgt,
                                   base_seed=stats["baseSeed"] + 1, step=0)
    saved = client.round_id
    client.round_id = saved + 1
    idx2, tgt2 = client.batch()
    client.round_id = saved
    other_batch, _ = dust_gradients(client.model, client.dust_cfg, idx2, tgt2,
                                    base_seed=stats["baseSeed"], step=0)
    layers = list(client.model.noise_shapes())
    order = torch.randperm(len(layers), generator=gen).tolist()
    faked = {f"{layers[i]}.weight" for i in order[: len(layers) // 2]}
    noise = _like(grads, gen)
    half = {k: (noise[k] if k in faked else v) for k, v in grads.items()}
    return {
        "random": _like(grads, gen),
        "other_seed": other_seed,
        "other_batch": other_batch,
        "backprop": backprop_gradients(client.model, idx, tgt),
        "half": half,
    }


def run_case(args, preset: str, device: str, codec_name: str) -> list[dict]:
    dtype = args.accel_dtype if device != "cpu" else args.cpu_dtype
    torch.manual_seed(args.seed)
    client = DustClient(ClientConfig(
        client_id=f"bench-{device}", offline=True, preset=preset, task=args.task,
        batch=args.batch, seq_len=args.seq_len, population=args.population,
        draw_chunk=args.draw_chunk, sigma=args.sigma, sigma_mode=args.sigma_mode,
        sigma_rel=args.sigma_rel, gamma=args.gamma, codec=codec_name,
        device=device, compute_dtype=dtype, seed=args.seed))
    client.sync()
    weights = _weights(client)
    accel.synchronize(client.device)
    started = time.perf_counter()
    grads, stats = client.gradient(step=0)
    accel.synchronize(client.device)
    client_seconds = time.perf_counter() - started
    n_layers = len(client.model.noise_shapes())
    base = {"sigmaMode": args.sigma_mode, "auditDtype": args.audit_dtype or dtype,
            "preset": preset, "clientDevice": device, "computeDtype": dtype, "codec": codec_name,
            "layersTotal": n_layers, "clientSeconds": round(client_seconds, 3)}
    rows = []

    header, vector = _decode(client, grads, stats)
    for k in [0, *args.layers]:
        res = audit.rescore(weights, header, vector, layers=k, salt="bench",
                            threshold=args.threshold, device=args.audit_device,
                            dtype=args.audit_dtype or None)
        rows.append({**base, "submission": "honest", "layers": res["layersChecked"],
                     "verdict": res["verdict"], "cosine": res.get("cosine"),
                     "headCosine": res.get("headCosine"), "worstLayer": res.get("worstLayer"),
                     "auditSeconds": res.get("seconds")})

    cheats = _cheats(client, grads, stats, args.seed + 7)
    for name, fake in cheats.items():
        header, vector = _decode(client, fake, stats)
        res = audit.rescore(weights, header, vector, layers=0, threshold=args.threshold,
                            device=args.audit_device, dtype=args.audit_dtype or None)
        row = {**base, "submission": name, "layers": res["layersChecked"],
               "verdict": res["verdict"], "cosine": res.get("cosine"),
               "headCosine": res.get("headCosine"), "worstLayer": res.get("worstLayer"),
               "auditSeconds": res.get("seconds")}
        if name == "half":
            # How often a k-layer spot check catches a client that faked half
            # its layers, over independent salts.
            catch = {}
            for k in args.layers:
                caught = 0
                for trial in range(args.trials):
                    r = audit.rescore(weights, header, vector, layers=k, salt=f"t{trial}",
                                      threshold=args.threshold, device=args.audit_device,
                            dtype=args.audit_dtype or None)
                    caught += r["verdict"] == "fail"
                catch[str(k)] = round(caught / args.trials, 3)
            row["catchRate"] = catch
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--presets", nargs="+", default=["tiny", "10m"])
    parser.add_argument("--devices", nargs="+", default=["cpu", accel.default_device()])
    parser.add_argument("--codecs", nargs="+", default=["dense", "ngd1r"])
    parser.add_argument("--cpu-dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--accel-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--audit-device", default="cpu")
    parser.add_argument("--task", default="induction")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--draw-chunk", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--sigma-mode", default="absolute", choices=["absolute", "relative"])
    parser.add_argument("--sigma-rel", type=float, default=0.05)
    parser.add_argument("--audit-dtype", default="", choices=["", "float32", "bfloat16"],
                        help="re-score dtype; empty uses the client's declared dtype")
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    devices = list(dict.fromkeys(args.devices))
    rows: list[dict] = []
    print(f"{'preset':>6} {'device':>5} {'dtype':>8} {'codec':>6} {'submission':>11} "
          f"{'k':>3} {'cosine':>8} {'head':>8} {'verdict':>7} {'audit s':>8} {'client s':>8}")
    for preset in args.presets:
        for device in devices:
            for codec_name in args.codecs:
                for row in run_case(args, preset, device, codec_name):
                    rows.append(row)
                    cos = "-" if row["cosine"] is None else f"{row['cosine']:+.4f}"
                    head = "-" if row["headCosine"] is None else f"{row['headCosine']:+.4f}"
                    extra = f"  catch {row['catchRate']}" if "catchRate" in row else ""
                    print(f"{preset:>6} {device:>5} {row['computeDtype']:>8} {codec_name:>6} "
                          f"{row['submission']:>11} {row['layers']:>3} {cos:>8} {head:>8} "
                          f"{row['verdict']:>7} {row['auditSeconds']:>8} {row['clientSeconds']:>8}{extra}",
                          flush=True)
                    if args.out:
                        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                        Path(args.out).write_text(json.dumps(
                            {"schema": SCHEMA, "config": vars(args), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
