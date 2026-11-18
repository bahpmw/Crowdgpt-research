"""HTTP surface for a Crowd-v1.5 coordinator.

Routes are separated by *paradigm*, not by version, because the two training
methods cannot share a step:

``POST /submit/dust``      a zeroth-order gradient estimate from a DUST client
``POST /submit/backprop``  an exact gradient from a v3.5 client

Both take the same container; the route in the URL and the route in the header
must agree, so a client cannot re-label a noisy estimate as an exact gradient to
skip the evidence checks.

Read-only endpoints are unauthenticated on purpose -- a client needs the model
before it can train, and there is nothing secret about public weights that are
already published.  Administration (bootstrap, finalize, closing a route) needs
the admin secret, read from a header: a secret in a query string ends up in
proxy logs, browser history and referrers.
"""

from __future__ import annotations

import os
import secrets
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .. import protocol
from ..coordinator import CoordConfig, Coordinator, Rejected
from ..store import ModelStore, StoreError

ADMIN_SECRET = os.getenv("CROWDGPT_ADMIN_SECRET", "")


def _admin_ok(request: Request) -> bool:
    if not ADMIN_SECRET:
        return False  # fail closed: no secret configured means no admin access
    supplied = ""
    for header in ("x-admin-secret", "X-Admin-Secret"):
        value = request.headers.get(header)
        if value:
            supplied = value.strip()
            break
    if not supplied:
        auth = (request.headers.get("authorization") or "").strip()
        if auth[:7].lower() == "bearer ":
            supplied = auth[7:].strip()
    if not supplied:
        return False
    return secrets.compare_digest(supplied, ADMIN_SECRET)


def _require_admin(request: Request) -> None:
    if not ADMIN_SECRET:
        raise HTTPException(status_code=503, detail="admin secret is not configured")
    if not _admin_ok(request):
        raise HTTPException(status_code=403, detail="invalid admin secret")


def _header_route(body: bytes) -> str:
    """Peek at the declared route before decoding 100 MB of container."""
    import json
    import struct

    if len(body) < 8:
        raise protocol.ProtocolError("body too short for a header length")
    (length,) = struct.unpack_from("<I", body, 0)
    if length == 0 or length > protocol.MAX_HEADER_BYTES or len(body) < 4 + length:
        raise protocol.ProtocolError("implausible header length")
    try:
        header = json.loads(body[4:4 + length].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise protocol.ProtocolError(f"header is not JSON: {exc}") from exc
    if not isinstance(header, dict) or header.get("route") not in protocol.ROUTES:
        raise protocol.ProtocolError(f"unknown route {header.get('route') if isinstance(header, dict) else None!r}")
    return str(header["route"])


async def _body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail=f"body exceeds {limit} bytes")
    payload = await request.body()
    if len(payload) > limit:
        raise HTTPException(status_code=413, detail=f"body exceeds {limit} bytes")
    if not payload:
        raise HTTPException(status_code=400, detail="empty body")
    return payload


def _byte_range(value: str, size: int) -> tuple[int, int]:
    """One ``bytes=a-b`` / ``bytes=a-`` / ``bytes=-n`` range as ``[start, end)``."""
    unit, _, spec = value.strip().partition("=")
    if unit.strip().lower() != "bytes" or "," in spec or "-" not in spec:
        raise HTTPException(status_code=416, detail="one bytes range is supported")
    first, _, last = spec.strip().partition("-")
    try:
        if first:
            start = int(first)
            end = min(size, int(last) + 1) if last else size
        else:
            start, end = max(0, size - int(last)), size
    except ValueError:
        raise HTTPException(status_code=416, detail="malformed range") from None
    if not (0 <= start < end <= size):
        raise HTTPException(status_code=416, detail=f"range not satisfiable for {size} bytes",
                            headers={"Content-Range": f"bytes */{size}"})
    return start, end


def config_from_env() -> CoordConfig:
    """Coordinator settings the factory reads when no config is passed in."""
    return CoordConfig(
        audit_layers=int(os.getenv("CROWDGPT_AUDIT_LAYERS", "0")),
        audit_threshold=float(os.getenv("CROWDGPT_AUDIT_THRESHOLD", "0.5")),
        audit_device=os.getenv("CROWDGPT_AUDIT_DEVICE", "cpu"),
        audit_dtype=os.getenv("CROWDGPT_AUDIT_DTYPE", ""),
        audit_require=os.getenv("CROWDGPT_AUDIT_REQUIRE", "") not in ("", "0", "false"),
    )


def create_app(root: str | os.PathLike[str] | None = None,
               config: CoordConfig | None = None,
               model_meta: dict[str, Any] | None = None) -> FastAPI:
    store = ModelStore(root or os.getenv("CROWDGPT_V15_ROOT", "v15_state"))
    coordinator = Coordinator(store, config or config_from_env(), model_meta)

    app = FastAPI(title="Crowd-v1.5 coordinator", version="1.5.0")
    app.state.store = store
    app.state.coordinator = coordinator

    def submit(route: str, body: bytes) -> dict[str, Any]:
        """Check the URL/header agreement, then hand the body to the coordinator."""
        try:
            declared = _header_route(body)
        except protocol.ProtocolError as exc:
            raise HTTPException(status_code=400, detail={"reason": "malformed_body", "detail": str(exc)})
        if declared != route:
            raise HTTPException(
                status_code=400,
                detail={"reason": "route_mismatch",
                        "detail": f"header says {declared}, url says {route}"},
            )
        try:
            return coordinator.submit(body)
        except Rejected as exc:
            raise HTTPException(status_code=409, detail={"reason": exc.reason, "detail": exc.detail})
        except protocol.ProtocolError as exc:
            raise HTTPException(status_code=400, detail={"reason": "malformed_body", "detail": str(exc)})

    @app.get("/")
    def index() -> dict[str, Any]:
        return {
            "service": "crowd-v1.5",
            "routes": list(coordinator.config.routes),
            "status": "/status",
            "manifest": "/model/manifest",
            "weights": "/model/weights",
            "submit": {r: f"/submit/{r}" for r in coordinator.config.routes},
        }

    @app.get("/status")
    def status() -> dict[str, Any]:
        body = coordinator.status()
        try:
            body["model"].update(
                {k: v for k, v in store.manifest(coordinator.model_meta).items()
                 if k in ("name", "params", "baseSha256", "bytes", "route")}
            )
        except StoreError:
            body["model"]["bootstrap"] = "no weights installed yet"
        return body

    @app.get("/model/manifest")
    def manifest() -> dict[str, Any]:
        try:
            return store.manifest(coordinator.model_meta)
        except StoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.get("/model/weights")
    def weights(request: Request) -> Response:
        headers = {"X-Model-Sha256": coordinator.base_sha256, "Accept-Ranges": "bytes"}
        try:
            wanted = request.headers.get("range")
            if not wanted:
                return Response(content=store.weights_bytes(),
                                media_type="application/octet-stream", headers=headers)
            size = store.weights_size()
            start, end = _byte_range(wanted, size)
            payload = store.weights_range(start, end)
        except StoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        headers["Content-Range"] = f"bytes {start}-{end - 1}/{size}"
        return Response(content=payload, status_code=206,
                        media_type="application/octet-stream", headers=headers)

    @app.post("/submit/dust")
    async def submit_dust(request: Request) -> JSONResponse:
        return JSONResponse(submit("dust", await _body(request, coordinator.config.max_body_bytes)))

    @app.post("/submit/backprop")
    async def submit_backprop(request: Request) -> JSONResponse:
        return JSONResponse(submit("backprop", await _body(request, coordinator.config.max_body_bytes)))

    @app.get("/admin/round")
    def admin_round(request: Request) -> dict[str, Any]:
        _require_admin(request)
        state = coordinator.status()
        state["pending"] = sorted((coordinator._state.get("pending") or {}))
        state["canFinalize"] = coordinator.can_finalize()
        return state

    @app.post("/admin/finalize")
    def admin_finalize(request: Request, force: bool = False) -> dict[str, Any]:
        _require_admin(request)
        try:
            record = coordinator.finalize(force=force)
        except Rejected as exc:
            raise HTTPException(status_code=409, detail={"reason": exc.reason, "detail": exc.detail})
        if record is None:
            return {"finalized": False, "reason": "quorum not reached",
                    "acceptedThisRound": coordinator.status()["acceptedThisRound"]}
        return {"finalized": True, "round": record}

    @app.post("/admin/bootstrap")
    async def admin_bootstrap(request: Request) -> dict[str, Any]:
        """Install the initial weights (a safetensors file in the request body)."""
        _require_admin(request)
        payload = await _body(request, coordinator.config.max_body_bytes)
        import io

        from safetensors.numpy import load as load_buffer

        try:
            tensors = {k: np.asarray(v, dtype=np.float32) for k, v in load_buffer(payload).items()}
        except Exception as exc:  # noqa: BLE001 - the library raises several types
            raise HTTPException(status_code=400, detail=f"not a safetensors file: {exc}")
        if not tensors:
            raise HTTPException(status_code=400, detail="no tensors in the body")
        try:
            return coordinator.bootstrap(tensors)
        except ValueError as exc:  # not a plain transformer
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/admin/routes")
    def admin_routes(request: Request, dust: bool | None = None, backprop: bool | None = None) -> dict[str, Any]:
        """Open or close a submission route without restarting."""
        _require_admin(request)
        routes = set(coordinator.config.routes)
        if dust is not None:
            routes = (routes | {"dust"}) if dust else (routes - {"dust"})
        if backprop is not None:
            routes = (routes | {"backprop"}) if backprop else (routes - {"backprop"})
        coordinator.config.routes = tuple(r for r in protocol.ROUTES if r in routes)
        coordinator._persist()
        return {"routes": list(coordinator.config.routes)}

    return app


# No module-level ``app``: creating one at import time would make a directory in
# whatever the CWD happens to be.  Run it as a factory instead:
#     uvicorn --factory crowd_v15.server.app:create_app
__all__ = ["ADMIN_SECRET", "create_app"]
