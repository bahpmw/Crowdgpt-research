"""HTTP surface of an orchestrator (FastAPI). All state lives in ``Node``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Path, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse

from .node import Node, Reject, parse
from .paths import MVP
from .schemas import KID, DataEnvelope, EntriesBody, Envelope

log = logging.getLogger("crowd_v2.api")
MAX_REQUEST = 100 * 1024 * 1024


def create_app(node: Node) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(node.run())
        task.add_done_callback(_crashed)
        yield
        task.cancel()
        await node.peers.close()

    app = FastAPI(title="CrowdGPT v2 orchestrator", lifespan=lifespan)

    @app.middleware("http")
    async def limit_size(request: Request, call_next):
        if int(request.headers.get("content-length") or 0) > MAX_REQUEST:
            return JSONResponse({"v": 1, "error": "request too large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(Reject)
    async def rejected(_: Request, exc: Reject) -> JSONResponse:
        return JSONResponse({"v": 1, "error": exc.reason}, status_code=exc.code)

    @app.post("/v2/register")
    async def register(env: Envelope) -> dict:
        return node.register(env.model_dump())

    @app.get("/v2/round")
    async def round_info() -> dict:
        return node.round_info()

    @app.post("/v2/work")
    async def work(env: Envelope) -> dict:
        return node.work(env.model_dump())

    @app.get("/v2/blob/{sha}")
    async def blob(sha: str = Path(pattern=r"^[0-9a-f]{64}$")) -> Response:
        data = node.blob(sha)
        if data is None:
            raise Reject(404, "unknown blob")
        return Response(data, media_type="application/octet-stream")

    @app.post("/v2/commit")
    async def commit(env: Envelope) -> dict:
        return node.commit(env.model_dump())

    @app.post("/v2/upload")
    async def upload(env: DataEnvelope) -> dict:
        d = env.model_dump()
        return node.upload({k: d[k] for k in ("v", "pub", "body", "sig")}, d["data"])

    @app.get("/v2/audit/{kid}")
    async def audit_requests(kid: str = Path(pattern=KID)) -> dict:
        return node.audit_requests(kid)

    @app.post("/v2/reveal")
    async def reveal(env: DataEnvelope) -> dict:
        d = env.model_dump()
        return node.reveal({k: d[k] for k in ("v", "pub", "body", "sig")}, d["data"])

    @app.get("/v2/record")
    async def record(since: int = Query(0, ge=0)) -> dict:
        return {"v": 1, "entries": node.chain.record.entries[since:]}

    @app.get("/v2/status")
    async def status() -> dict:
        return node.status()

    @app.get("/v2/events")
    async def events(request: Request) -> StreamingResponse:
        q: asyncio.Queue = asyncio.Queue()
        node.subscribers.add(q)

        async def stream() -> AsyncIterator[str]:
            try:
                for ev in list(node.events)[-30:]:
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                while not await request.is_disconnected():
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=10.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
            finally:
                node.subscribers.discard(q)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/health")
    async def health() -> dict:
        st = node.chain.state
        rs = node.rs
        return {"v": 1, "status": "ok", "orchestrator": node.kid, "round": st.round,
                "open": {"round": rs.r, "beacon": rs.beacon, "commits": sorted(rs.commits)} if rs else None,
                "subsystems": {"record": {"entries": node.chain.record.next_n},
                               "round_loop": node.rs.phase if node.rs else "starting",
                               "peers": {k: "up" if v else "down" for k, v in node.peers.up.items()},
                               "weights_modules": len(st.weights)}}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return node.metrics()

    @app.post("/v2/peer/submission")
    async def peer_submission(env: Envelope) -> dict:
        return node.on_forward(env.model_dump())

    @app.post("/v2/peer/verdict")
    async def peer_verdict(env: Envelope) -> dict:
        return node.on_verdict(env.model_dump())

    @app.post("/v2/peer/propose")
    async def peer_propose(env: Envelope) -> JSONResponse:
        kid, body = node.peer_auth(env.model_dump())
        b = parse(EntriesBody, body)
        code, out = await node.chain.on_propose(b.entries, kid)
        return JSONResponse({"v": 1, **out}, status_code=code)

    @app.post("/v2/peer/cosign")
    async def peer_cosign(env: Envelope) -> JSONResponse:
        kid, body = node.peer_auth(env.model_dump())
        b = parse(EntriesBody, body)
        code, out = await node.chain.on_final(b.entries, kid)
        return JSONResponse({"v": 1, **out}, status_code=code)

    @app.get("/dashboard")
    async def dashboard() -> FileResponse:
        return FileResponse(MVP / "dashboard" / "index.html")

    return app


def _crashed(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("round loop crashed", exc_info=task.exception())
