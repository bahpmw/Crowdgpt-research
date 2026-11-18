"""Re-score a DUST submission: the coordinator's audit primitive.

A DUST estimate is a deterministic function of the served weights, the batch,
the seeds and the estimator settings, all of which a client declares in its
header. With CPU-generated noise (``noiseDevice="cpu"``) every draw is
bit-identical on every machine, so a coordinator can recompute what an honest
client *must* have sent and compare.

Recomputing a whole submission costs as much as producing it. In
``mode="layer"`` it does not have to: each perturbed layer has its own passes,
so the coordinator can re-score ``k`` layers it picks *after* the upload (from
a salt the client never sees) for roughly ``k / n_layers`` of the client's
compute. A client that fakes any sizeable fraction of its layers is caught with
probability that grows with ``k``; one that fakes all of them is caught on the
first check.

The comparison is a cosine rather than equality. The wire codecs are lossy
(sign planes plus block scales) and a bf16 client on another accelerator rounds
differently, so an honest re-score lands near, not at, 1. A fabricated or
replayed estimate is a different random projection and lands near 0; the
default threshold sits between the two and ``tools/bench_audit.py`` measures
both sides.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import torch

from .dust import DustConfig, _head_name, dust_gradients
from .model import GPT, ModelConfig, canonical_params

SCHEMA = "crowd-v15-audit-1"
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def unauditable_reason(header: dict[str, Any]) -> str | None:
    """Why a submission cannot be re-scored, or ``None`` if it can."""
    replay = header.get("replay")
    if header.get("route") != "dust":
        return "route is not dust"
    if not isinstance(replay, dict):
        return "no replay record"
    if replay.get("noiseDevice", "cpu") != "cpu":
        return "noise generated on the accelerator is not reproducible"
    if int(header.get("localSteps", 1)) != 1:
        return "local steps move the weights between estimates"
    if int(header.get("averagedDraws", 1)) != 1:
        return "averaged draws are not replayed"
    if header.get("errorFeedback"):
        return "error feedback adds a carried residual"
    if replay.get("task") == "copy":
        return "copy batches depend on a token file the coordinator may not hold"
    if replay.get("computeDtype", "float32") not in _DTYPES:
        return "unknown compute dtype"
    return None


def pick_layers(shapes: list[str], k: int, *, salt: str, client_id: str,
                round_id: int) -> list[str]:
    """``k`` perturbed layers chosen from a secret salt, so a client cannot predict them."""
    if k <= 0 or k >= len(shapes):
        return list(shapes)
    keyed = sorted(
        shapes,
        key=lambda name: hashlib.blake2b(
            f"{salt}:{client_id}:{round_id}:{name}".encode(), digest_size=8).digest(),
    )
    return sorted(keyed[:k], key=shapes.index)


def _cos(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom == 0.0 else float(np.dot(a, b) / denom)


def rescore(weights: dict[str, np.ndarray], header: dict[str, Any], vector: np.ndarray, *,
            layers: int = 0, salt: str = "", threshold: float = 0.5,
            device: str | torch.device = "cpu", dtype: str | None = None,
            batch_fn=None) -> dict[str, Any]:
    """Recompute (part of) a submission and compare it with what was uploaded.

    ``layers=0`` re-scores every perturbed layer; ``layers=k`` re-scores ``k``
    picked from ``salt``. ``batch_fn`` defaults to the client's own batch
    function, so the coordinator rebuilds exactly the batch the header names.
    ``dtype`` overrides the client's declared compute dtype (a CPU auditor may
    prefer float32 over a slow CPU bfloat16). Returns a record with ``verdict`` ``"pass"``, ``"fail"`` or ``"skipped"``.
    """
    started = time.perf_counter()
    record: dict[str, Any] = {"schema": SCHEMA, "clientId": header.get("clientId"),
                              "roundId": header.get("roundId"), "threshold": threshold}
    reason = unauditable_reason(header)
    if reason:
        record.update(verdict="skipped", reason=reason)
        return record
    replay = header["replay"]
    try:
        model_cfg = ModelConfig.preset(str(replay["preset"]))
    except (KeyError, ValueError) as exc:
        record.update(verdict="fail", reason=f"unknown preset: {exc}")
        return record

    device = torch.device(device)
    declared = replay.get("computeDtype", "float32")
    if declared == "bfloat16" and replay.get("device", device.type) != device.type:
        # bf16 rounds differently on each kind of device and the estimate is a
        # difference of nearby losses, so only a same-device replay matches it.
        record.update(verdict="skipped",
                      reason=f"bf16 estimate from {replay.get('device')} needs a "
                             f"{replay.get('device')} auditor")
        return record
    dtype = _DTYPES[dtype or declared]
    model = GPT(model_cfg)
    params = canonical_params(model)
    if sorted(params) != sorted(weights) or any(
            tuple(params[n].shape) != tuple(np.asarray(weights[n]).shape) for n in params):
        record.update(verdict="fail", reason="declared preset does not match the served model")
        return record
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(torch.from_numpy(np.asarray(weights[name], dtype=np.float32)))
    model = model.to(device=device, dtype=dtype).eval()

    if batch_fn is None:
        from .client import client_batch as batch_fn
    idx, tgt, _cursor = batch_fn(
        replay["task"], seed=int(header.get("seed", 0)), round_id=int(header["roundId"]),
        batch=int(replay["batch"]), seq_len=int(replay["seqLen"]),
        vocab=model_cfg.vocab_size, device=device, cursor=int(replay.get("dataCursor", 0)))
    cfg = DustConfig(population=int(replay.get("population", header.get("population", 1))),
                     draw_chunk=int(replay.get("drawChunk", 8)),
                     sigma=float(replay.get("sigma", header.get("sigma", 1e-3))),
                     sigma_mode=replay.get("sigmaMode", "absolute"),
                     sigma_rel=float(replay.get("sigmaRel", 0.01)),
                     gamma=float(replay.get("gamma", header.get("gamma", 0.98))),
                     mode=replay.get("mode", "layer"), noise_device="cpu")

    shapes = list(model.noise_shapes())
    full = layers <= 0 or layers >= len(shapes) or cfg.mode != "layer"
    chosen = shapes if full else pick_layers(shapes, layers, salt=salt,
                                             client_id=str(header.get("clientId")),
                                             round_id=int(header["roundId"]))
    grads, stats = dust_gradients(model, cfg, idx, tgt, base_seed=int(header["baseSeed"]),
                                  step=int(header.get("step", 0)),
                                  layers=None if full else chosen)

    names = list(header["names"])
    offsets = np.concatenate([[0], np.cumsum([int(s) for s in header["sizes"]])])
    head = _head_name(model)

    def pair(name: str) -> tuple[np.ndarray, np.ndarray]:
        i = names.index(name)
        sent = np.asarray(vector[offsets[i]:offsets[i + 1]], dtype=np.float64)
        mine = grads[name].detach().float().cpu().numpy().reshape(-1).astype(np.float64)
        return sent, mine

    # The head gradient is exact, so a client that sends a true backprop
    # gradient would match it. Score the perturbed layers on their own and the
    # head separately; both have to clear the threshold.
    compared = [f"{n}.weight" for n in chosen if f"{n}.weight" != head]
    per_layer: dict[str, float | None] = {}
    left, right = [], []
    for name in compared:
        sent, mine = pair(name)
        value = _cos(sent, mine)
        per_layer[name] = None if value is None else round(value, 6)
        left.append(sent)
        right.append(mine)
    cosine = _cos(np.concatenate(left), np.concatenate(right)) if left else None
    head_cosine = None
    if full or "tok_emb" in chosen or head != "tok_emb.weight":
        head_cosine = _cos(*pair(head))
    if cosine is None:
        cosine = head_cosine
    # Every checked layer has to clear the threshold on its own: a pooled
    # cosine lets a client that faked its small layers hide behind its large ones.
    worst = min((v for v in per_layer.values() if v is not None), default=None)
    passed = (cosine is not None and cosine >= threshold
              and (worst is None or worst >= threshold)
              and (head_cosine is None or head_cosine >= threshold))
    record.update(
        verdict="pass" if passed else "fail",
        cosine=None if cosine is None else round(cosine, 6),
        headCosine=None if head_cosine is None else round(head_cosine, 6),
        worstLayer=worst, perLayer=per_layer,
        layersChecked=len(chosen), layersTotal=len(shapes), full=full,
        forwards=int(stats["forwards"]),
        seconds=round(time.perf_counter() - started, 3),
    )
    if not passed:
        record["reason"] = (f"cosine {record['cosine']}, worst layer {worst}, "
                            f"head {record['headCosine']}: below {threshold}")
    return record


__all__ = ["SCHEMA", "pick_layers", "rescore", "unauditable_reason"]
