"""Round control for a v1.5 coordinator.

The coordinator is the only component that decides *whether a step happened*.
Clients contribute gradient estimates; this module decides which of them count,
how they combine, and when the model moves.

Two routes, deliberately not merged:

``dust``
    A zeroth-order estimate.  It is unbiased in expectation but noisy per
    submission, so it is weighted by the evidence behind it (``tokens``), and it
    is rejected outright when the estimator record is impossible (no tokens, no
    population, non-positive sigma).  A single DUST client must not be able to
    claim a full step's worth of descent from a two-token probe.

``backprop``
    An exact gradient.  Same container, same averaging, but no evidence
    weighting: a backprop client that processed ten tokens still reports a real
    gradient.  This route exists so a v3.5 client keeps working after the
    coordinator moves to v1.5.

The two are counted separately and never averaged together.  Mixing them would
make the effective step size depend on the client mix, which is exactly the kind
of number nobody can reproduce.
"""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import audit, protocol
from .plain import check_plain_parameter_names
from .store import ModelStore


class Rejected(Exception):
    """A submission the coordinator refuses, with a stable machine-readable reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason if not detail else f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class CoordConfig:
    lr: float = 1e-3
    momentum: float = 0.0
    # DiLoCo-style outer optimizer. ``outer_lr`` is the *pseudo-gradient*
    # learning rate: DiLoCo's local-drift pseudo-gradient has a different scale
    # from a minibatch gradient, so it needs its own step size rather than
    # reusing ``lr``. 0 keeps the plain ``lr * g`` path untouched.
    outer_lr: float = 0.0
    outer_momentum: float = 0.0
    grad_clip: float = 0.0
    min_submissions: int = 2
    min_tokens: int = 1
    max_abs_grad: float = 100.0
    max_body_bytes: int = 1 << 30
    max_round_seconds: float = 3600.0
    routes: tuple[str, ...] = protocol.ROUTES
    # Re-score DUST submissions before accepting them (``audit.rescore``).
    # 0 is off; ``k`` re-scores ``k`` perturbed layers picked from a salt the
    # clients never see, which costs about ``k / n_layers`` of a client step.
    audit_layers: int = 0
    audit_threshold: float = 0.5
    audit_device: str = "cpu"
    # "" re-scores in the client's declared dtype; "float32" overrides it.
    audit_dtype: str = ""
    # Reject what cannot be re-scored (accelerator noise, local steps, ...)
    # instead of accepting it unaudited.
    audit_require: bool = False
    audit_salt: str = field(default_factory=lambda: secrets.token_hex(16), repr=False)

    def __post_init__(self) -> None:
        if not (self.lr > 0):
            raise ValueError("lr must be positive")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if self.outer_lr < 0:
            raise ValueError("outer_lr must not be negative")
        if not (0.0 <= self.outer_momentum < 1.0):
            raise ValueError("outer_momentum must be in [0, 1)")
        if self.min_submissions < 1:
            raise ValueError("min_submissions must be at least 1")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must not be negative")
        if self.audit_layers < 0:
            raise ValueError("audit_layers must not be negative")
        if not (-1.0 <= self.audit_threshold <= 1.0):
            raise ValueError("audit_threshold must be a cosine in [-1, 1]")


@dataclass
class Round:
    round_id: int
    base_sha256: str
    started_at: float
    submissions: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def accepted_count(self, route: str) -> int:
        return sum(1 for s in self.submissions.values() if s["route"] == route)


class Coordinator:
    def __init__(self, store: ModelStore, config: CoordConfig | None = None,
                 model_meta: dict[str, Any] | None = None):
        self.store = store
        self.config = config or CoordConfig()
        self.model_meta = dict(model_meta or {})
        self._momentum: dict[str, np.ndarray] = {}
        self._outer_momentum: dict[str, np.ndarray] = {}
        self._pending_vectors: dict[str, np.ndarray] = {}
        self._state = self._load_state()

    # ---- lifecycle -------------------------------------------------------

    def bootstrap(self, tensors: dict[str, np.ndarray]) -> dict[str, Any]:
        """Install the model a coordinator serves before the first round."""
        check_plain_parameter_names(tensors)
        digest = self.store.save_weights(tensors)
        self._momentum = {}
        self._outer_momentum = {}
        self._state = {
            "schema": "crowd-v15-coordinator-1",
            "round": 0,
            "baseSha256": digest,
            "startedAt": time.time(),
            "history": [],
            "routes": list(self.config.routes),
        }
        self._persist()
        return self.status()

    def _load_state(self) -> dict[str, Any]:
        state = self.store.load_status()
        if not state:
            return {"round": 0, "baseSha256": "", "startedAt": time.time(),
                    "history": [], "routes": list(self.config.routes)}
        if state.get("pending"):
            # Accepted-but-unapplied submissions live in memory only; a restart
            # drops them rather than applying a step nobody can audit. Clients
            # see the round number unchanged and resubmit.
            state["pending"] = {}
            state["seenThisRound"] = {}
            state["acceptedThisRound"] = 0
            state["acceptedByRoute"] = {}
        return state

    def _persist(self) -> None:
        status = self.status()
        self.store.save_status(status)

    @property
    def round_id(self) -> int:
        return int(self._state.get("round", 0))

    @property
    def base_sha256(self) -> str:
        return str(self._state.get("baseSha256", ""))

    def status(self) -> dict[str, Any]:
        accepted = int(self._state.get("acceptedThisRound", 0))
        by_route = self._state.get("acceptedByRoute", {})
        return {
            "schema": "crowd-v15-status-1",
            "round": self.round_id,
            "baseSha256": self.base_sha256,
            "routes": list(self.config.routes),
            "minSubmissions": self.config.min_submissions,
            "lr": self.config.lr,
            "momentum": self.config.momentum,
            "outerLr": self.config.outer_lr,
            "outerMomentum": self.config.outer_momentum,
            "gradClip": self.config.grad_clip,
            "acceptedThisRound": accepted,
            "acceptedByRoute": by_route,
            "pending": sorted(self._state.get("pending") or {}),
            "lastRound": (self._state.get("history") or [None])[-1],
            "model": {
                "name": self.model_meta.get("name", "crowd-v1.5-dust"),
                "route": self.model_meta.get("route", "dust"),
                "params": self.model_meta.get("params"),
            },
        }

    # ---- submissions -----------------------------------------------------

    def submit(self, body: bytes) -> dict[str, Any]:
        """Decode, validate and record one gradient estimate. Returns the receipt."""
        weights = self.store.load_weights()
        total = int(sum(int(np.asarray(t).size) for t in weights.values()))
        header, vector = protocol.decode_body(
            body, max_params=total, max_body_bytes=self.config.max_body_bytes
        )
        self._check_against_model(header, weights)
        record = self._accept(header, vector, weights)
        receipt = {
            "accepted": True,
            "roundId": header["roundId"],
            "route": header["route"],
            "clientId": header["clientId"],
            "params": header["params"],
            "bytes": len(body),
            "acceptedThisRound": int(self._state.get("acceptedThisRound", 0)),
            "neededThisRound": max(0, self.config.min_submissions - int(self._state.get("acceptedThisRound", 0))),
        }
        if record.get("audit") is not None:
            receipt["audit"] = record["audit"]
        finalized = self.finalize()
        if finalized is not None:
            receipt["finalizedRound"] = finalized["round"]
        return receipt

    def _check_against_model(self, header: dict[str, Any], weights: dict[str, np.ndarray]) -> None:
        names = sorted(weights)
        if header["names"] != names:
            missing = sorted(set(names) - set(header["names"]))
            extra = sorted(set(header["names"]) - set(names))
            raise Rejected("parameter_set_mismatch",
                           f"missing={missing[:3]} extra={extra[:3]}")
        sizes = [int(np.asarray(weights[n]).size) for n in names]
        if [int(s) for s in header["sizes"]] != sizes:
            raise Rejected("parameter_size_mismatch", "shapes do not match the served model")

    def _audit(self, header: dict[str, Any], vector: np.ndarray,
               weights: dict[str, np.ndarray] | None) -> dict[str, Any] | None:
        """Re-score a DUST submission when auditing is on; raise if it fails."""
        if self.config.audit_layers <= 0 or header["route"] != "dust" or weights is None:
            return None
        result = audit.rescore(weights, header, vector, layers=self.config.audit_layers,
                               salt=self.config.audit_salt,
                               threshold=self.config.audit_threshold,
                               device=self.config.audit_device,
                               dtype=self.config.audit_dtype or None)
        if result["verdict"] == "fail":
            raise Rejected("audit_failed", str(result.get("reason", "re-score did not match")))
        if result["verdict"] == "skipped" and self.config.audit_require:
            raise Rejected("unauditable", str(result.get("reason")))
        keep = ("verdict", "cosine", "headCosine", "layersChecked", "layersTotal", "seconds", "reason")
        return {k: result[k] for k in keep if result.get(k) is not None}

    def _accept(self, header: dict[str, Any], vector: np.ndarray,
                weights: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        route = header["route"]
        if route not in self.config.routes:
            raise Rejected("route_disabled", f"{route} is not open on this coordinator")
        if int(header["roundId"]) != self.round_id:
            raise Rejected("stale_round",
                           f"submission is for round {header['roundId']}, current is {self.round_id}")
        if header["baseSha256"] != self.base_sha256:
            raise Rejected("base_weights_changed",
                           "gradient was measured from different weights")
        pending = self._state.get("pending") or {}
        if pending:
            existing = {info["route"] for info in pending.values()}
            if route not in existing:
                raise Rejected(
                    "mixed_routes",
                    f"round already holds {sorted(existing)} submissions; "
                    "open a new round before switching route",
                )
        client = header["clientId"]
        if client in self._state.setdefault("seenThisRound", {}):
            raise Rejected("duplicate_client", "one submission per client per round")
        estimate = protocol.estimator_fields(header)
        if route == "dust":
            if estimate["tokens"] < self.config.min_tokens:
                raise Rejected("insufficient_evidence",
                               f"{estimate['tokens']:.0f} tokens < {self.config.min_tokens}")
            if estimate["population"] < 1 or estimate["sigma"] <= 0:
                raise Rejected("invalid_estimator_record",
                               "population and sigma must be positive for a DUST estimate")
        # A DUST estimate's absolute scale is an artefact of ``sigma`` and the
        # token count: a 64-token draw at sigma=1e-3 comes back with a norm near
        # 1.5e5 on a 1B model, so an absolute peak bound rejects every honest
        # submission. Normalise to a unit direction and keep the norm as a
        # declared magnitude the round applies through ``max_abs_grad``.
        norm = float(np.linalg.norm(vector)) if vector.size else 0.0
        peak = float(np.abs(vector).max()) if vector.size else 0.0
        if not math.isfinite(norm) or not math.isfinite(peak):
            raise Rejected("gradient_out_of_range", "estimate contains a non-finite value")
        direction = vector if norm == 0.0 else (vector / norm).astype(np.float32)
        audited = self._audit(header, vector, weights)

        # Evidence weighting: a DUST estimate carries the token count it averaged
        # over; a backprop gradient is exact and gets unit weight.
        weight = 1.0
        if route == "dust":
            weight = max(1.0, estimate["tokens"])
        self._state.setdefault("pending", {})[client] = {
            "route": route,
            "weight": weight,
            "tokens": estimate["tokens"],
            "population": estimate["population"],
            "sigma": estimate["sigma"],
            "gamma": estimate["gamma"],
            "norm": norm,
            "peak": peak,
        }
        self._pending_vectors[client] = direction
        self._state.setdefault("seenThisRound", {})[client] = int(time.time())
        accepted = int(self._state.get("acceptedThisRound", 0)) + 1
        by_route = dict(self._state.get("acceptedByRoute") or {})
        by_route[route] = int(by_route.get(route, 0)) + 1
        self._state["acceptedThisRound"] = accepted
        self._state["acceptedByRoute"] = by_route
        self._persist()
        self.store.append_submission({
            "at": int(time.time()), "roundId": int(header["roundId"]), "route": route,
            "clientId": client, "params": int(header["params"]),
            "tokens": estimate["tokens"], "population": estimate["population"],
            "sigma": estimate["sigma"], "weight": weight,
            # Whether the client carried its quantization error forward. It does
            # not change acceptance, but a round's audit trail should say which
            # clients were compensating.
            "errorFeedback": bool(header.get("errorFeedback")),
            "audit": audited,
        })
        return {"route": route, "clientId": client, "weight": weight, "audit": audited}

    # ---- finalize --------------------------------------------------------

    def can_finalize(self) -> bool:
        accepted = int(self._state.get("acceptedThisRound", 0))
        return accepted >= self.config.min_submissions

    def finalize(self, *, force: bool = False) -> dict[str, Any] | None:
        """Apply one round if a quorum has formed. Returns the round record or None."""
        if not force and not self.can_finalize():
            return None
        pending = self._state.get("pending") or {}
        vectors = getattr(self, "_pending_vectors", {})
        if not pending or not vectors:
            return None
        routes = {info["route"] for info in pending.values()}
        # Directions are unit vectors, so plain averaging keeps one client from
        # dominating on scale alone; the round's magnitude is the mean declared
        # norm, capped by ``max_abs_grad`` (and by ``grad_clip`` when set).
        count = len(pending)
        combined = None
        for client in pending:
            contribution = vectors[client].astype(np.float64) / count
            combined = contribution if combined is None else combined + contribution
        declared = float(sum(float(i.get("norm", 0.0)) for i in pending.values()) / count)
        magnitude = min(declared, float(self.config.max_abs_grad))
        combined = (combined * magnitude).astype(np.float32)

        norm = float(np.linalg.norm(combined))
        clipped = magnitude < declared
        if self.config.grad_clip > 0 and norm > self.config.grad_clip:
            combined = (combined * (self.config.grad_clip / norm)).astype(np.float32)
            clipped = True
            norm = float(np.linalg.norm(combined))

        weights = self.store.load_weights()
        names = sorted(weights)
        update = protocol.unflatten(combined, names, [int(np.asarray(weights[n]).size) for n in names])
        outer = float(self.config.outer_lr) > 0.0
        for name in names:
            grad = update[name].reshape(np.asarray(weights[name]).shape)
            if outer:
                # The aggregated estimate is a pseudo-gradient, not a minibatch
                # gradient: it is the average of *directions* with a declared
                # magnitude, so it does not shrink as the model approaches an
                # optimum. Nesterov momentum is what DiLoCo's scaling work finds
                # stable at this scale (lr 0.7, momentum 0.9).
                if name not in self._outer_momentum:
                    self._outer_momentum[name] = np.zeros_like(grad)
                buf = self._outer_momentum[name]
                buf *= float(self.config.outer_momentum)
                buf += grad
                step = buf * float(self.config.outer_momentum) + grad
                weights[name] = (np.asarray(weights[name], dtype=np.float32)
                                 - float(self.config.outer_lr) * step)
                continue
            if name not in self._momentum:
                self._momentum[name] = np.zeros_like(grad)
            if self.config.momentum:
                self._momentum[name] = self._momentum[name] * self.config.momentum + grad
                grad = self._momentum[name]
            weights[name] = (np.asarray(weights[name], dtype=np.float32) - self.config.lr * grad)

        digest = self.store.save_weights(weights)
        previous_round = self.round_id
        record = {
            "round": previous_round,
            "finishedAt": int(time.time()),
            "baseSha256": self.base_sha256,
            "newSha256": digest,
            "clients": len(pending),
            "byRoute": {r: sum(1 for i in pending.values() if i["route"] == r) for r in sorted(routes)},
            "route": sorted(routes)[0] if len(routes) == 1 else "mixed",
            "gradNorm": round(norm, 6),
            "declaredNorm": round(declared, 6),
            "clipped": clipped,
            "lr": self.config.lr,
            "momentum": self.config.momentum,
            "outerLr": self.config.outer_lr,
            "outerMomentum": self.config.outer_momentum,
            "tokens": int(sum(float(i["tokens"]) for i in pending.values())),
        }
        self._state = {
            "schema": "crowd-v15-coordinator-1",
            "round": previous_round + 1,
            "baseSha256": digest,
            "startedAt": time.time(),
            "history": (self._state.get("history") or [])[-49:] + [record],
            "routes": list(self.config.routes),
            "acceptedThisRound": 0,
            "acceptedByRoute": {},
            "pending": {},
            "seenThisRound": {},
        }
        self._pending_vectors = {}
        self._persist()
        return record


__all__ = ["CoordConfig", "Coordinator", "Rejected", "Round"]
