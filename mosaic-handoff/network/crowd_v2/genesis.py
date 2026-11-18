"""Create a network: orchestrator keys, calibrated thresholds, initial weights, genesis entry.

``python -m crowd_v2.genesis --out DIR --orchestrators 3 --preset tiny``

mvp: one tool generates every orchestrator key and co-signs the genesis entry;
in v2 each operator keeps its own key and signs the genesis it reviewed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import audit
from .crypto import Identity, canonical, hash_tensors, sha256_hex
from .nets import Hyper, Tokens, eval_loss, init_weights, module_names
from .record import chain_entries
from .wire import to_bytes

TIMING = {"open": 3.0, "commit": 120.0, "upload": 30.0, "audit": 120.0,
          "propose_timeout": 6.0, "final_timeout": 10.0}


def make_genesis(out: Path, n_orch: int, hp: Hyper, p_audit: float, timing: dict,
                 calib_devices: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    idents = [Identity.load_or_create(out / f"orch{i}.key") for i in range(n_orch)]
    tokens = Tokens(vocab=hp.vocab, seq=hp.seq)
    weights = init_weights(hp)
    names = module_names(hp)
    calib = audit.calibrate(hp, tokens, calib_devices)
    config = {"hyper": hp.to_dict(), "p_audit": p_audit, "timing": timing,
              "thresholds": calib["thresholds"], "calibration": calib["pairs"],
              "shapes": [{n: list(weights[n].shape) for n in m} for m in names]}
    body = {"config": config, "config_hash": sha256_hex(canonical(config)),
            "orchestrators": {i.kid: i.pub for i in idents}, "quorum": n_orch // 2 + 1,
            "preset": hp.preset, "K": hp.K,
            "weights": {str(k): hash_tensors({n: weights[n] for n in m}) for k, m in enumerate(names)},
            "eval_loss": round(eval_loss(hp, weights, tokens), 6)}
    entry = chain_entries("0" * 64, 0, [("genesis", 0, body)])[0]
    entry["sigs"] = {i.kid: i.sign(entry["hash"].encode()) for i in idents}
    (out / "genesis.json").write_text(json.dumps(entry, indent=1, sort_keys=True))
    (out / "init.safetensors").write_bytes(to_bytes(weights))
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--orchestrators", type=int, default=3)
    ap.add_argument("--hyper", default="{}", help='JSON overrides for Hyper, e.g. {"preset":"10m"}')
    ap.add_argument("--timing", default="{}", help="JSON overrides for round timing (seconds)")
    ap.add_argument("--p-audit", type=float, default=0.5)
    ap.add_argument("--calib-devices", default="", help="comma list, e.g. mps")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    entry = make_genesis(a.out, a.orchestrators, Hyper(**json.loads(a.hyper)), a.p_audit,
                         TIMING | json.loads(a.timing), [d for d in a.calib_devices.split(",") if d])
    print(json.dumps({"genesis": entry["hash"], "thresholds": entry["body"]["config"]["thresholds"]}))


if __name__ == "__main__":
    main()
