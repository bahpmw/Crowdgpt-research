#!/usr/bin/env python3
"""Pass a Crowd v1.5 DUST checkpoint around through a Hugging Face repo.

There is no server to run. A repo holds one weight file and one round counter;
a trainer *pulls* the baton, trains locally, and *pushes* it back with the round
incremented. The round guard is what stops a slow trainer from overwriting a
newer round: ``push`` refuses when the remote round is ahead of the local one.

Files in a repo (all under the root by default):

    weights.safetensors   float32 tensors under canonical parameter names
    status.json           crowd-v15-coordinator-1: round, baseSha256, history
    manifest.json         crowd-v15-model-1: names, sizes, shapes, baseSha256
    seed.json             crowd-v15-seed-1: preset, params, seed, digest
    lock.json             crowd-v15-baton-1: advisory lease (holder, expiry)

``baseSha256`` is sha256 over sorted names plus the *float32-rounded-to-bfloat16*
bytes, so a dtype change or a re-save does not move it. ``verify`` recomputes it.

    python tools/hf_baton.py pull  --repo peva3/Crowd-v1.5-dust --dir state
    python tools/hf_baton.py train --repo peva3/Crowd-v1.5-dust --dir state \
        --steps 20 --data /tmp/v15data/tok --device cuda --push
    python tools/hf_baton.py verify --dir state

Authentication: pass a fine-grained write token through the environment, never
on the command line and never in a file that gets committed:

    export HF_TOKEN=hf_...   # write access to this one repo only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_v15 import accel  # noqa: E402
from crowd_v15.dust import DustConfig, dust_gradients, DustOptimizer  # noqa: E402
from crowd_v15.model import GPT, ModelConfig, canonical_params, param_report  # noqa: E402
from crowd_v15.store import weights_digest  # noqa: E402

FILES = ("weights.safetensors", "status.json", "manifest.json", "seed.json", "lock.json")
WEIGHTS = "weights.safetensors"
STATUS = "status.json"
MANIFEST = "manifest.json"
SEED = "seed.json"
LOCK = "lock.json"
LOCK_TTL_S = 24 * 3600
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


class BatonError(RuntimeError):
    """The checkpoint or the hub state is inconsistent."""


# ---------------------------------------------------------------- hub access

def _token(token_env: str) -> str | None:
    value = os.environ.get(token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not value:
        raise BatonError(
            f"{token_env} is not set. Create a fine-grained token with write access "
            "to the checkpoint repo (Settings -> Access Tokens -> Fine-grained -> "
            "Repositories: write) and export it, e.g. "
            f"export {token_env}=hf_...  Never commit the token."
        )
    return value


def _optional_token(token_env: str) -> str | None:
    """Reads are allowed anonymously; a token only raises the rate limit."""
    return os.environ.get(token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def hub_download(repo: str, name: str, dest: Path, *, revision: str | None,
                 token_env: str = "HF_TOKEN") -> Path | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(repo, name, revision=revision,
                               token=_optional_token(token_env),
                               local_dir=str(dest.parent))
    except EntryNotFoundError:
        return None
    return Path(path)


def hub_upload_files(repo: str, directory: Path, names: list[str], message: str,
                     *, token_env: str = "HF_TOKEN") -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    operations = [CommitOperationAdd(path_in_repo=name,
                                     path_or_fileobj=str(directory / name))
                  for name in names]
    commit = api.create_commit(repo_id=repo, operations=operations, commit_message=message,
                               token=_token(token_env))
    return getattr(commit, "commit_id", "") or ""


# ---------------------------------------------------------------- local state

def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BatonError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_weights(directory: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    path = directory / WEIGHTS
    if not path.is_file():
        raise BatonError(f"missing {path}")
    return {k: np.asarray(v, dtype=np.float32) for k, v in load_file(str(path)).items()}


def save_weights(directory: Path, tensors: dict[str, Any]) -> None:
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    payload = {k: np.ascontiguousarray(np.asarray(v, dtype=np.float32))
               for k, v in tensors.items()}
    tmp = directory / (WEIGHTS + ".tmp")
    save_file(payload, str(tmp), metadata={"format": "pt", "dtype": "float32"})
    os.replace(tmp, directory / WEIGHTS)


def model_tensors(model: GPT) -> dict[str, np.ndarray]:
    return {name: p.detach().float().cpu().numpy()
            for name, p in canonical_params(model).items()}


def load_into(model: GPT, tensors: dict[str, np.ndarray]) -> None:
    with torch.no_grad():
        for name, param in canonical_params(model).items():
            if name not in tensors:
                raise BatonError(f"checkpoint has no tensor named {name}")
            value = torch.from_numpy(np.asarray(tensors[name]))
            if value.shape != param.shape:
                raise BatonError(f"{name}: checkpoint shape {tuple(value.shape)} "
                                 f"!= model shape {tuple(param.shape)}")
            param.copy_(value.to(param.device, param.dtype))


# ---------------------------------------------------------------- documents

def seed_document(repo: str, preset: str, params: int, digest: str, *, seed: int,
                  names: list[str]) -> dict[str, Any]:
    return {
        "schema": "crowd-v15-seed-1",
        "name": repo.split("/")[-1],
        "preset": preset,
        "params": int(params),
        "seed": int(seed),
        "tiedEmbeddings": True,
        "canonicalNames": names,
        "weightsDigest": digest,
        "note": "canonical_params lists tied weights once, under tok_emb.weight",
        "createdAt": time.time(),
    }


def manifest_document(name: str, tensors: dict[str, Any], digest: str) -> dict[str, Any]:
    names = sorted(tensors)
    return {
        "schema": "crowd-v15-model-1",
        "name": name,
        "route": "dust",
        "dtype": "float32",
        "params": int(sum(np.asarray(tensors[n]).size for n in names)),
        "bytes": int(sum(np.asarray(tensors[n], dtype=np.float32).nbytes for n in names)),
        "baseSha256": digest,
        "names": names,
        "sizes": [int(np.asarray(tensors[n]).size) for n in names],
        "shapes": [list(np.asarray(tensors[n]).shape) for n in names],
    }


def status_document(round_id: int, digest: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": "crowd-v15-coordinator-1",
        "round": int(round_id),
        "baseSha256": digest,
        "history": list(history or []),
        "routes": ["dust", "backprop"],
        "startedAt": time.time(),
    }


def lock_document(holder: str, round_id: int, digest: str, *, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    return {
        "schema": "crowd-v15-baton-1",
        "holder": holder,
        "round": int(round_id),
        "digest": digest,
        "acquiredAt": now,
        "expiresAt": now + LOCK_TTL_S,
        "note": "advisory lease; the round guard in hf_baton push is the enforcement",
    }


def push_problem(remote_round: int, local_round: int, *, force: bool) -> str | None:
    """Reason a push must be refused, or None when it may proceed."""
    if force:
        return None
    if remote_round > local_round:
        return (f"remote round {remote_round} is ahead of local round {local_round}; "
                "pull again instead of overwriting it")
    if remote_round == local_round:
        return (f"remote round {remote_round} equals the local round; a push would "
                "claim a round that is already published (use --force to rewrite it)")
    return None


# ---------------------------------------------------------------- commands

def cmd_init(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    cfg = ModelConfig.preset(args.preset)
    torch.manual_seed(args.seed)
    model = GPT(cfg)
    tensors = model_tensors(model)
    save_weights(directory, tensors)
    digest = weights_digest(tensors)
    names = sorted(tensors)
    _write_json(directory / SEED, seed_document(args.repo, args.preset, cfg.n_params, digest,
                                                seed=args.seed, names=names))
    _write_json(directory / MANIFEST, manifest_document(args.repo.split("/")[-1], tensors, digest))
    _write_json(directory / STATUS, status_document(0, digest))
    _write_json(directory / LOCK, lock_document(args.holder, 0, digest))
    report = param_report(cfg)
    print(json.dumps({"dir": str(directory), "preset": args.preset, "params": report["total"],
                      "digest": digest, "bytes": (directory / WEIGHTS).stat().st_size},
                     indent=1, sort_keys=True))
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in FILES:
        got = hub_download(args.repo, name, directory / name, revision=args.revision,
                           token_env=args.token_env)
        if got is None:
            missing.append(name)
    if WEIGHTS in missing or STATUS in missing:
        raise BatonError(f"repo {args.repo} is missing {missing}")
    return cmd_verify(argparse.Namespace(dir=str(directory), quiet=args.quiet))


def cmd_verify(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    tensors = load_weights(directory)
    digest = weights_digest(tensors)
    status = _read_json(directory / STATUS)
    manifest = _read_json(directory / MANIFEST)
    seed = _read_json(directory / SEED) if (directory / SEED).is_file() else {}
    problems = []
    if status.get("baseSha256") != digest:
        problems.append(f"status.baseSha256 {status.get('baseSha256')} != {digest}")
    if manifest.get("baseSha256") != digest:
        problems.append(f"manifest.baseSha256 {manifest.get('baseSha256')} != {digest}")
    if seed and int(status.get("round", 0)) == 0 and seed.get("weightsDigest") != digest:
        # seed.json records the *genesis* checkpoint, so it only has to agree
        # with the weights before the first round. After that the baton has
        # moved on and status.json is the authority.
        problems.append(f"seed.weightsDigest {seed.get('weightsDigest')} != {digest}")
    for name in sorted(tensors):
        if name not in set(manifest.get("names") or []):
            problems.append(f"manifest omits {name}")
            break
    if manifest.get("sizes") and manifest.get("names"):
        expected = dict(zip(manifest["names"], manifest["sizes"]))
        for name, expected_size in expected.items():
            actual = int(np.asarray(tensors.get(name, [])).size)
            if actual != expected_size:
                problems.append(f"{name}: {actual} values, manifest says {expected_size}")
                break
    preset = seed.get("preset") if seed else None
    if preset:
        want = ModelConfig.preset(preset).n_params
        if int(seed.get("params", -1)) != want:
            problems.append(f"seed.params {seed.get('params')} != preset {preset} ({want})")
        if int(manifest.get("params", -1)) != want:
            problems.append(f"manifest.params {manifest.get('params')} != {want}")
    payload = {"dir": str(directory), "round": status.get("round"), "digest": digest,
               "params": int(sum(int(np.asarray(v).size) for v in tensors.values())),
               "preset": preset, "problems": problems, "ok": not problems}
    if not args.quiet:
        print(json.dumps(payload, indent=1, sort_keys=True))
    if problems:
        raise BatonError("; ".join(problems))
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    local_status = _read_json(directory / STATUS)
    digest = weights_digest(load_weights(directory))
    if local_status.get("baseSha256") != digest:
        raise BatonError("local status does not describe the local weights; run verify")
    remote_dir = Path(args.dir) / ".remote"
    remote_dir.mkdir(parents=True, exist_ok=True)
    remote = hub_download(args.repo, STATUS, remote_dir / STATUS,
                          revision=args.revision, token_env=args.token_env)
    remote_round = int(_read_json(remote).get("round", -1)) if remote else -1
    problem = push_problem(remote_round, int(local_status.get("round", 0)), force=args.force)
    if problem:
        raise BatonError(problem)
    _write_json(directory / LOCK, lock_document(args.holder, int(local_status["round"]), digest))
    upload = [WEIGHTS, STATUS, MANIFEST, LOCK]
    if (directory / SEED).is_file():
        upload.append(SEED)
    commit = hub_upload_files(args.repo, directory, upload,
                              message=args.message or f"round {local_status['round']} ({digest[:12]})",
                              token_env=args.token_env)
    print(json.dumps({"repo": args.repo, "round": local_status["round"], "digest": digest,
                      "commit": commit, "files": upload}, indent=1, sort_keys=True))
    return 0


def _window_batch(path: str, step: int, batch: int, seq_len: int, seed: int):
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    span = int(tokens.size) - (seq_len + 1)
    if span <= 0:
        raise BatonError(f"{path} is too short for seq {seq_len}")
    generator = torch.Generator().manual_seed((int(seed) * 1_000_003 + int(step)) & 0x7FFFFFFF)
    offsets = torch.randint(0, span, (batch,), generator=generator).tolist()
    rows = [np.asarray(tokens[o:o + seq_len + 1]) for o in offsets]
    block = np.stack(rows).astype(np.int64)
    return torch.from_numpy(block[:, :-1]), torch.from_numpy(block[:, 1:])


def cmd_train(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    seed_doc = _read_json(directory / SEED)
    preset = args.preset or seed_doc.get("preset")
    if not preset:
        raise BatonError("no preset in seed.json; pass --preset")
    cfg = ModelConfig.preset(preset)
    device = accel.resolve_device(args.device)
    dtype = DTYPES[args.compute_dtype]
    tensors = load_weights(directory)
    if int(sum(int(np.asarray(v).size) for v in tensors.values())) != cfg.n_params:
        raise BatonError(f"checkpoint has {sum(int(np.asarray(v).size) for v in tensors.values())} "
                         f"values, preset {preset} expects {cfg.n_params}")
    model = GPT(cfg).to(device=device, dtype=dtype)
    load_into(model, tensors)
    del tensors

    dust_cfg = DustConfig(population=args.population, draw_chunk=args.draw_chunk,
                          sigma=args.sigma, sigma_mode=args.sigma_mode,
                          sigma_rel=args.sigma_rel, gamma=args.gamma, mode=args.mode)
    optimizer = DustOptimizer(model, lr=args.lr, momentum=args.momentum, grad_clip=args.grad_clip)
    train_bin = str(Path(args.data) / "train.bin")
    status = _read_json(directory / STATUS)
    started = time.perf_counter()
    tokens_seen = 0
    step = 0
    losses: list[float] = []
    while step < args.steps:
        if args.minutes and (time.perf_counter() - started) / 60.0 >= args.minutes:
            break
        idx, tgt = _window_batch(train_bin, step, args.batch, args.seq_len, args.seed)
        idx, tgt = idx.to(device), tgt.to(device)
        grads, stats = dust_gradients(model, dust_cfg, idx, tgt,
                                      base_seed=args.seed + step * 1_000_003, step=step)
        optimizer.lr = optimizer.lr_at(step, args.warmup_steps, args.lr)
        optimizer.step(grads)
        with torch.no_grad():
            logits = model(idx)
            loss = float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)))
        losses.append(loss)
        tokens_seen += args.batch * args.seq_len
        step += 1
        if step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"step": step, "tokens": tokens_seen,
                              "loss": round(float(np.mean(losses[-20:])), 6),
                              "sigma": round(float(stats["sigma"]), 6),
                              "seconds": round(time.perf_counter() - started, 1),
                              "tokensPerSecond": round(tokens_seen / max(1e-9, time.perf_counter() - started), 1),
                              "vramGb": accel.peak_memory_gb(device)}), flush=True)

    save_weights(directory, model_tensors(model))
    digest = weights_digest(load_weights(directory))
    history = list(status.get("history") or [])
    history.append({"round": int(status.get("round", 0)) + 1, "clients": 1, "route": "dust",
                    "steps": step, "tokens": tokens_seen, "loss": round(float(np.mean(losses)), 6),
                    "population": args.population, "sigma": args.sigma,
                    "lr": args.lr, "finishedAt": time.time(), "baseSha256": digest})
    _write_json(directory / STATUS, status_document(int(status.get("round", 0)) + 1, digest, history))
    _write_json(directory / MANIFEST, manifest_document(directory.name, load_weights(directory), digest))
    _write_json(directory / LOCK, lock_document(args.holder, int(status.get("round", 0)) + 1, digest))
    print(json.dumps({"trained": step, "tokens": tokens_seen, "round": int(status.get("round", 0)) + 1,
                      "digest": digest}, indent=1, sort_keys=True))
    if args.push:
        return cmd_push(argparse.Namespace(dir=str(directory), repo=args.repo, revision=None,
                                           token_env=args.token_env, force=False, holder=args.holder,
                                           message=None))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    # Both orders work: `hf_baton.py --repo X push` and `hf_baton.py push --repo X`.
    common.add_argument("--repo", default="peva3/Crowd-v1.5-dust")
    common.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--repo", default=argparse.SUPPRESS)
    parser.add_argument("--token-env", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", parents=[common], help="build a fresh seed checkpoint in a local dir")
    init.add_argument("--preset", default="300m")
    init.add_argument("--dir", required=True)
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--holder", default="peva3")

    pull = sub.add_parser("pull", parents=[common], help="download the baton")
    pull.add_argument("--dir", required=True)
    pull.add_argument("--revision", default=None)
    pull.add_argument("--quiet", action="store_true")

    verify = sub.add_parser("verify", parents=[common], help="recompute the digest and check the documents")
    verify.add_argument("--dir", required=True)
    verify.add_argument("--quiet", action="store_true")

    push = sub.add_parser("push", parents=[common], help="upload the baton after training")
    push.add_argument("--dir", required=True)
    push.add_argument("--revision", default=None)
    push.add_argument("--force", action="store_true")
    push.add_argument("--holder", default="peva3")
    push.add_argument("--message", default="")

    train = sub.add_parser("train", parents=[common], help="pull-free local DUST training on the local baton dir")
    train.add_argument("--dir", required=True)
    train.add_argument("--preset", default="")
    train.add_argument("--data", default="/tmp/v15data/tok")
    train.add_argument("--device", default=accel.default_device(),
                       help="cuda (incl. ROCm), mps, cpu or auto")
    train.add_argument("--compute-dtype", default="bfloat16", choices=tuple(DTYPES))
    train.add_argument("--steps", type=int, default=20)
    train.add_argument("--minutes", type=float, default=0.0)
    train.add_argument("--batch", type=int, default=2)
    train.add_argument("--seq-len", type=int, default=512)
    train.add_argument("--population", type=int, default=64)
    train.add_argument("--draw-chunk", type=int, default=16)
    train.add_argument("--sigma", type=float, default=1e-2)
    train.add_argument("--sigma-mode", choices=("absolute", "relative"), default="relative")
    train.add_argument("--sigma-rel", type=float, default=0.05)
    train.add_argument("--gamma", type=float, default=0.9)
    train.add_argument("--mode", choices=("layer", "all"), default="all")
    train.add_argument("--lr", type=float, default=0.05)
    train.add_argument("--momentum", type=float, default=0.9)
    train.add_argument("--grad-clip", type=float, default=0.0)
    train.add_argument("--warmup-steps", type=int, default=10)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log-every", type=int, default=5)
    train.add_argument("--holder", default="peva3")
    train.add_argument("--push", action="store_true")
    return parser.parse_args(argv)


COMMANDS = {"init": cmd_init, "pull": cmd_pull, "verify": cmd_verify,
            "push": cmd_push, "train": cmd_train}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
