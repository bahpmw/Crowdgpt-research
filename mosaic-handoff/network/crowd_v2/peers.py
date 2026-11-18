"""Signed orchestrator-to-orchestrator HTTP and liveness tracking."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx

from .crypto import Identity

log = logging.getLogger("crowd_v2.peers")


class Peers:
    def __init__(self, ident: Identity, urls: dict[str, str],
                 on_change: Callable[[str, bool], None]):
        self.ident, self.urls, self.on_change = ident, urls, on_change
        self.up: dict[str, bool] = {kid: False for kid in urls}
        self.rounds: dict[str, int] = {kid: 0 for kid in urls}
        self.open: dict[str, dict | None] = {kid: None for kid in urls}
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=2.0))
        self._tasks: set[asyncio.Task] = set()

    def mark(self, kid: str, alive: bool) -> None:
        if self.up.get(kid) != alive:
            self.up[kid] = alive
            self.on_change(kid, alive)

    async def post(self, kid: str, path: str, body: dict, timeout: float = 30.0) -> httpx.Response | None:
        try:
            resp = await self.http.post(self.urls[kid] + path, json=self.ident.envelope(body),
                                        timeout=timeout)
        except httpx.HTTPError as exc:
            # A dead or slow peer is expected during chaos; the caller treats None as "no answer".
            log.info("peer post failed", extra={"f": {"peer": kid, "path": path, "err": type(exc).__name__}})
            self.mark(kid, False)
            return None
        self.mark(kid, True)
        return resp

    async def get(self, kid: str, path: str, timeout: float = 30.0) -> httpx.Response | None:
        try:
            resp = await self.http.get(self.urls[kid] + path, timeout=timeout)
        except httpx.HTTPError as exc:
            log.info("peer get failed", extra={"f": {"peer": kid, "path": path, "err": type(exc).__name__}})
            self.mark(kid, False)
            return None
        self.mark(kid, True)
        return resp

    def broadcast(self, path: str, body: dict) -> None:
        """Fire-and-forget to every peer; failures only flip liveness."""
        for kid in self.urls:
            task = asyncio.create_task(self.post(kid, path, body))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def heartbeat(self) -> None:
        for kid in self.urls:
            resp = await self.get(kid, "/health", timeout=2.0)
            if resp is not None and resp.status_code == 200:
                health = resp.json()
                self.rounds[kid] = int(health.get("round", 0))
                self.open[kid] = health.get("open")
            elif resp is not None:
                self.mark(kid, False)

    def live(self) -> list[str]:
        return [kid for kid, alive in self.up.items() if alive]

    async def close(self) -> None:
        await self.http.aclose()
