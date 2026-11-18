"""The one place that knows where the v1.5 model code and token files live.

mvp: every node reads the same local token files; in v2 these become
content-addressed shards fetched from peers.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V15 = REPO / "archived" / "crowd-v1.5"
DATA_DIR = V15 / "runs" / "data" / "tok"
MVP = Path(__file__).resolve().parents[1]

if str(V15) not in sys.path:
    sys.path.insert(0, str(V15))
