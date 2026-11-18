"""The one place that knows where the v1.5 model code and token files live.

Handoff layout: crowd_v15/ sits two levels up; token files default to
data/tok there and can be pointed elsewhere with CROWD_TOKENS.

mvp: every node reads the same local token files; in v2 these become
content-addressed shards fetched from peers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

V15 = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CROWD_TOKENS", V15 / "data" / "tok"))
MVP = Path(__file__).resolve().parents[1]

if str(V15) not in sys.path:
    sys.path.insert(0, str(V15))
