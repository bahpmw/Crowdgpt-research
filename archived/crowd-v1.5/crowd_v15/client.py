"""A Crowd-v1.5 DUST client.

What a round looks like
-----------------------
1. ``GET /model/manifest`` -> shapes and the digest of the weights on the server.
2. ``GET /model/weights``  -> the same weights, verified against that digest.
3. Run DUST locally: no backward pass, no optimizer state, no gradient buffer.
   The estimator needs the activations, so it produces a weight-shaped
   *estimate* of the descent direction.
4. ``POST /submit/dust`` with that estimate, tagged with the digest it was
   measured from and the evidence behind it (tokens, population, sigma, gamma).
5. When the coordinator finalizes, the round number changes and the next sync
   pulls the new weights.

Why one gradient per round
--------------------------
A gradient is only valid at the point it was measured.  Taking ten DUST steps
locally and then uploading the tenth step's gradient would hand the coordinator
a direction measured at weights it never saw.  ``--local-steps`` exists so the
staleness is explicit (it is reported in the header), not so it can be hidden.

The client is deliberately runnable without a server (``--dry-run``) and without
a GPU: the same code path trains on CPU, which is the point of the paradigm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import accel, codec, protocol
from .dust import DustConfig, DustOptimizer, dust_gradients, probe_cosine
from .feedback import ErrorFeedback
from .model import GPT, ModelConfig, canonical_params
from .plain import check_plain_parameter_names
from .store import weights_digest
from .tasks import identity_batch, induction_batch, load_tokens, random_batch, token_batch

DEFAULT_TIMEOUT = 120.0


class ClientError(RuntimeError):
    pass


@dataclass
class ClientConfig:
    server: str = "http://127.0.0.1:8000"
    client_id: str = ""
    preset: str = "tiny"
    task: str = "identity"
    data: str = ""
    data_dtype: str = "uint32"
    batch: int = 4
    seq_len: int = 64
    population: int = 32
    draw_chunk: int = 8
    sigma: float = 1e-3
    # "relative" scales each layer's noise to its activation RMS; in bf16 an
    # absolute sigma of 1e-3 sits below the rounding step of a unit activation.
    sigma_mode: str = "absolute"
    sigma_rel: float = 0.01
    # "cpu" noise is reproducible on any machine, so a coordinator can audit it;
    # "device" is faster on an accelerator but cannot be re-scored elsewhere.
    noise_device: str = "cpu"
    gamma: float = 0.98
    lr: float = 1e-3
    momentum: float = 0.0
    grad_clip: float = 0.0
    codec: str = "ngd1r"
    device: str = "cpu"
    compute_dtype: str = "float32"
    local_steps: int = 1
    # Independent DUST draws at the *same* weights, averaged before upload.
    # Variance falls as 1/k, the payload does not change, and (unlike
    # local_steps) the estimate is not measured at a moved point.
    draws: int = 1
    seed: int = 0
    timeout: float = DEFAULT_TIMEOUT
    dry_run: bool = False
    offline: bool = False
    probe_cosine: bool = False
    error_feedback: bool = False

    def check(self) -> None:
        if self.codec not in codec.CODECS:
            raise ValueError(f"codec must be one of {codec.CODECS}")
        if self.task not in ("identity", "induction", "copy", "random"):
            raise ValueError("task must be identity, induction, copy or random")
        if self.task == "copy" and not self.data:
            raise ValueError("--data is required for task=copy")
        if self.local_steps < 1:
            raise ValueError("local_steps must be at least 1")
        if self.draws < 1:
            raise ValueError("draws must be at least 1")


def _http(url: str, *, method: str = "GET", body: bytes | None = None,
          headers: dict[str, str] | None = None, timeout: float = DEFAULT_TIMEOUT) -> bytes:
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:800].decode("utf-8", "replace")
        raise ClientError(f"{method} {url} -> {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ClientError(f"{method} {url} failed: {exc.reason}") from exc


def client_batch(task: str, *, seed: int, round_id: int, batch: int, seq_len: int,
                 vocab: int, device: torch.device, tokens: torch.Tensor | None = None,
                 cursor: int = 0) -> tuple[torch.Tensor, torch.Tensor, int]:
    """A client's batch for one round, as a pure function of what it declares.

    A coordinator re-scoring a submission calls this with the header's values,
    so it must not depend on any client state beyond ``cursor``.
    """
    gen = torch.Generator().manual_seed(
        int.from_bytes(
            hashlib.blake2b(f"{seed}:{round_id}:batch".encode(), digest_size=8).digest(),
            "little",
        ) % (2**31)
    )
    if task == "identity":
        return (*identity_batch(batch, seq_len, vocab, device, gen), cursor)
    if task == "induction":
        return (*induction_batch(batch, seq_len, vocab, device, gen), cursor)
    if task == "random":
        return (*random_batch(batch, seq_len, vocab, device, gen), cursor)
    if tokens is None:
        raise ClientError("task=copy needs --data")
    return token_batch(tokens, batch, seq_len, cursor, device)


class DustClient:
    def __init__(self, config: ClientConfig):
        config.check()
        self.config = config
        self.device = accel.resolve_device(config.device)
        self.model_cfg = ModelConfig.preset(config.preset)
        if config.seq_len > self.model_cfg.context_length:
            raise ValueError(
                f"seq_len {config.seq_len} exceeds the preset's context {self.model_cfg.context_length}"
            )
        self.model = GPT(self.model_cfg).to(
            self.device, torch.float32 if config.compute_dtype == "float32" else torch.bfloat16
        )
        self.dust_cfg = DustConfig(
            population=config.population,
            draw_chunk=config.draw_chunk,
            sigma=config.sigma,
            sigma_mode=config.sigma_mode,
            sigma_rel=config.sigma_rel,
            noise_device=config.noise_device,
            gamma=config.gamma,
            lr=config.lr,
            momentum=config.momentum,
            grad_clip=config.grad_clip,
        )
        self.client_id = config.client_id or f"client-{os.getpid()}-{int(time.time()) % 100000}"
        self.cursor = 0
        self.tokens = None
        if config.data:
            self.tokens = load_tokens(config.data, config.data_dtype)
        self.round_id = 0
        self.base_sha256 = ""
        self.history: list[dict[str, Any]] = []
        # Error feedback lives across rounds, so it is client state, not a pure
        # function of the current gradient.
        self._feedback = ErrorFeedback(config.codec) if config.error_feedback else None

    # ---- server sync -----------------------------------------------------

    def sync(self) -> dict[str, Any]:
        """Fetch the manifest and the weights, and check they agree."""
        if self.config.offline:
            self.base_sha256 = weights_digest(
                {n: p.detach().cpu().float().numpy() for n, p in canonical_params(self.model).items()}
            )
            return {"offline": True, "baseSha256": self.base_sha256, "params": self.model.n_params}
        manifest = json.loads(_http(f"{self.config.server}/model/manifest", timeout=self.config.timeout))
        try:
            status = json.loads(_http(f"{self.config.server}/status", timeout=self.config.timeout))
            manifest["round"] = int(status.get("round", 0))
        except ClientError:
            manifest["round"] = self.round_id
        blob = self._download_weights(manifest)
        import io

        from safetensors.numpy import load as load_buffer

        tensors = {k: np.asarray(v, dtype=np.float32) for k, v in load_buffer(blob).items()}
        digest = weights_digest(tensors)
        if digest != manifest["baseSha256"]:
            raise ClientError(
                f"weights digest {digest[:12]} does not match the manifest {manifest['baseSha256'][:12]}"
            )
        check_plain_parameter_names(tensors)
        if sorted(tensors) != sorted(self._param_names()):
            raise ClientError("served parameter names do not match the local model")
        with torch.no_grad():
            for name, param in self._named_params().items():
                source = tensors[name]
                if tuple(source.shape) != tuple(param.shape):
                    raise ClientError(f"{name}: served shape {source.shape} != local {tuple(param.shape)}")
                param.copy_(torch.from_numpy(source).to(param.device, param.dtype))
        self.base_sha256 = digest
        return manifest

    def _download_weights(self, manifest: dict[str, Any], attempts: int = 3) -> bytes:
        """The weight file, shard by shard when the manifest offers a plan."""
        url = f"{self.config.server}/model/weights"
        shards = manifest.get("shards") or []
        if len(shards) <= 1:
            return _http(url, timeout=self.config.timeout)
        parts = []
        for shard in shards:
            for attempt in range(attempts):
                try:
                    part = _http(url, headers={"Range": f"bytes={shard['start']}-{shard['end'] - 1}"},
                                 timeout=self.config.timeout)
                except ClientError:
                    if attempt + 1 == attempts:
                        raise
                    continue
                if hashlib.sha256(part).hexdigest() == shard["sha256"]:
                    parts.append(part)
                    break
                if attempt + 1 == attempts:
                    raise ClientError(f"shard {shard['index']} failed its sha256 check")
        return b"".join(parts)

    def _named_params(self) -> dict[str, torch.Tensor]:
        return canonical_params(self.model)

    def _param_names(self) -> list[str]:
        return sorted(self._named_params())

    # ---- training --------------------------------------------------------

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx, tgt, self.cursor = client_batch(
            self.config.task, seed=self.config.seed, round_id=self.round_id,
            batch=self.config.batch, seq_len=self.config.seq_len,
            vocab=self.model_cfg.vocab_size, device=self.device,
            tokens=self.tokens, cursor=self.cursor)
        return idx, tgt

    def replay(self, data_cursor: int) -> dict[str, Any]:
        """Everything besides the header's seeds that an auditor needs to re-score."""
        cfg = self.dust_cfg
        return {
            "preset": self.config.preset, "task": self.config.task,
            "batch": int(self.config.batch), "seqLen": int(self.config.seq_len),
            "dataCursor": int(data_cursor), "computeDtype": self.config.compute_dtype,
            "sigma": float(cfg.sigma), "sigmaMode": cfg.sigma_mode,
            "sigmaRel": float(cfg.sigma_rel), "gamma": float(cfg.gamma),
            "mode": cfg.mode, "noiseDevice": cfg.noise_device,
            "population": int(cfg.population), "drawChunk": int(cfg.draw_chunk),
            # bf16 rounding differs between accelerators, so an auditor matching
            # a bf16 estimate exactly needs the same kind of device.
            "device": self.device.type,
        }

    def gradient(self, step: int = 0) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        data_cursor = self.cursor
        idx, tgt = self.batch()
        base_seed = int(self.config.seed) + int(self.round_id) * 1_000_003
        grads, stats = dust_gradients(self.model, self.dust_cfg, idx, tgt, base_seed=base_seed, step=step)
        stats["route"] = "dust"
        stats["baseSeed"] = base_seed
        stats["step"] = int(step)
        stats["dataCursor"] = int(data_cursor)
        stats["roundId"] = self.round_id
        stats["window"] = [int(idx.min()), int(idx.max())]
        stats["localSteps"] = self.config.local_steps
        return grads, stats

    def apply_locally(self, grads: dict[str, torch.Tensor]) -> None:
        DustOptimizer(self.model, self.config.lr, self.config.momentum, self.config.grad_clip).step(grads)

    # ---- upload ----------------------------------------------------------

    def submit(self, grads: dict[str, torch.Tensor], stats: dict[str, Any]) -> dict[str, Any]:
        body, dense_bytes = self.encode(grads, stats)
        reply = _http(
            f"{self.config.server}/submit/dust",
            method="POST",
            body=body,
            headers={"content-type": "application/octet-stream"},
            timeout=self.config.timeout,
        )
        receipt = json.loads(reply)
        receipt["uploadBytes"] = len(body)
        receipt["denseBytes"] = dense_bytes
        self.history.append({"roundId": self.round_id, **stats, "receipt": receipt})
        return receipt

    def encode(self, grads: dict[str, torch.Tensor], stats: dict[str, Any]) -> tuple[bytes, int]:
        """The submission body for one estimate, and its dense bf16 size in bytes."""
        names, sizes, vector = protocol.flatten(grads)
        feedback_stats: dict[str, Any] | None = None
        if self._feedback is not None:
            # Send g + carried error, and carry the new error forward. The
            # coordinator averages unit directions, so this sharpens the
            # direction rather than inflating the step.
            vector, feedback_stats = self._feedback.encode(vector)
        extra = {
            "tokens": int(stats.get("tokens", 0)),
            "population": int(stats.get("population", self.config.population)),
            "sigma": float(stats.get("sigma", self.config.sigma)),
            "gamma": float(stats.get("gamma", self.config.gamma)),
            "draws": int(stats.get("draws", self.config.population)),
            "baseSeed": int(stats.get("baseSeed", 0)),
            "seed": int(self.config.seed),
            "step": int(stats.get("step", 0)),
            "localSteps": int(self.config.local_steps),
            "averagedDraws": int(self.config.draws),
            "params": int(sum(sizes)),
            "replay": self.replay(int(stats.get("dataCursor", 0))),
        }
        if feedback_stats is not None:
            extra["errorFeedback"] = True
            extra["residualNorm"] = float(feedback_stats["residualNorm"])
        header = protocol.build_header(
            route="dust",
            client_id=self.client_id,
            round_id=self.round_id,
            base_sha256=self.base_sha256,
            names=names,
            sizes=sizes,
            codec_name=self.config.codec,
            **extra,
        )
        return protocol.encode_body(header, vector), int(vector.size) * 2

    # ---- loop ------------------------------------------------------------

    def _one_draw(self, step: int, *, first: bool, averaged):
        """One DUST estimate, with the timing and probe bookkeeping ``run`` needs."""
        started = time.perf_counter()
        grads, stats = self.gradient(step=step)
        stats["seconds"] = round(time.perf_counter() - started, 4)
        if stats["seconds"] > 0:
            stats["tokensPerSecond"] = round(float(stats["tokens"]) / stats["seconds"], 1)
        if self.config.probe_cosine and first:
            idx, tgt = self.batch()
            stats["cosine"] = probe_cosine(self.model, self.dust_cfg, idx, tgt,
                                           base_seed=stats["baseSeed"], step=step)
        if self.config.local_steps > 1:
            # The estimate is applied locally so the next estimate is measured at
            # the moved point; the coordinator sees only the last one.
            self.apply_locally(grads)
        if self.config.draws > 1:
            stats["averagedDraws"] = self.config.draws
        return grads, stats

    def run(self, rounds: int, log=print) -> dict[str, Any]:
        summary: dict[str, Any] = {"schema": "crowd-v15-client-run-1", "rounds": [], "dryRun": self.config.dry_run}
        for _ in range(rounds):
            manifest = self.sync()
            self.round_id = int(manifest.get("round", 0)) if "round" in manifest else self.round_id
            averaged = None
            for step in range(self.config.local_steps):
                for draw in range(self.config.draws):
                    grads, stats = self._one_draw(step if self.config.draws == 1 else draw,
                                                  first=(draw == 0), averaged=averaged)
                    averaged = grads if averaged is None else {
                        name: averaged[name] + grads[name] for name in averaged}
                grads = averaged if self.config.draws == 1 else {
                    name: value / float(self.config.draws) for name, value in averaged.items()}
            if self.config.dry_run or self.config.offline:
                receipt = {"accepted": None, "dryRun": True}
            else:
                receipt = self.submit(grads, stats)
            entry = {
                "roundId": self.round_id,
                "stats": stats,
                "receipt": receipt,
                "gradNorm": stats.get("gradNorm"),
            }
            summary["rounds"].append(entry)
            log(json.dumps({"round": self.round_id, "gradNorm": stats.get("gradNorm"),
                            "tokens": stats.get("tokens"), "receipt": receipt}, sort_keys=True))
            self.round_id += 1
        summary["clientId"] = self.client_id
        return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crowd-v1.5 DUST client")
    p.add_argument("--server", default=os.getenv("CROWDGPT_V15_SERVER", "http://127.0.0.1:8000"))
    p.add_argument("--client-id", default="")
    p.add_argument("--preset", default="tiny")
    p.add_argument("--task", default="identity")
    p.add_argument("--data", default="")
    p.add_argument("--data-dtype", default="uint32")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--draw-chunk", type=int, default=8)
    p.add_argument("--sigma", type=float, default=1e-3)
    p.add_argument("--sigma-mode", default="absolute", choices=["absolute", "relative"])
    p.add_argument("--sigma-rel", type=float, default=0.01)
    p.add_argument("--noise-device", default="cpu", choices=["cpu", "device"],
                   help="cpu noise can be audited by the coordinator; device noise is faster")
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--codec", default="ngd1r")
    p.add_argument("--device", default="cpu", help="cpu, cuda (incl. ROCm), mps or auto")
    p.add_argument("--compute-dtype", default="float32")
    p.add_argument("--local-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--offline", action="store_true",
                   help="train and measure without contacting a server")
    p.add_argument("--probe-cosine", action="store_true")
    p.add_argument("--error-feedback", action="store_true",
                   help="carry the quantization error of the upload across rounds")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ClientConfig(
        server=args.server, client_id=args.client_id, preset=args.preset, task=args.task,
        data=args.data, data_dtype=args.data_dtype, batch=args.batch, seq_len=args.seq_len,
        population=args.population, draw_chunk=args.draw_chunk, sigma=args.sigma,
        sigma_mode=args.sigma_mode, sigma_rel=args.sigma_rel, noise_device=args.noise_device,
        gamma=args.gamma,
        lr=args.lr, momentum=args.momentum, grad_clip=args.grad_clip, codec=args.codec,
        device=args.device, compute_dtype=args.compute_dtype, local_steps=args.local_steps,
        seed=args.seed, timeout=args.timeout, dry_run=args.dry_run, offline=args.offline,
        probe_cosine=args.probe_cosine, error_feedback=args.error_feedback,
    )
    client = DustClient(config)
    summary = client.run(args.rounds)
    text = json.dumps(summary, indent=2, sort_keys=True, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
