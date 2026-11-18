"""Run one orchestrator: ``python -m crowd_v2.orchestrator --config node.json``.

The config file is a JSON ``NodeConfig``: port, key file, genesis and initial
weights files, data directory, peers (url + public key) and device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import uvicorn

from . import logs
from .api import create_app
from .node import Node, NodeConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = NodeConfig.model_validate(json.loads(a.config.read_text()))
    node = Node(cfg)
    logs.setup(f"orch-{node.kid}")
    uvicorn.run(create_app(node), host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
