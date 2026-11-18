"""Structured JSON logging (no IPs, no secrets): ``log.info(msg, extra={"f": {...}})``."""

from __future__ import annotations

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def __init__(self, node: str):
        super().__init__()
        self.node = node

    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": round(time.time(), 3), "level": record.levelname, "node": self.node,
               "logger": record.name, "msg": record.getMessage()}
        out.update(getattr(record, "f", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup(node: str, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(node))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
