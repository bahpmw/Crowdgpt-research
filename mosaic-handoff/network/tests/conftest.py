import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)

from crowd_v2.nets import Hyper, Tokens  # noqa: E402

TINY = Hyper(preset="tiny", H=4, batch=4)


@pytest.fixture(scope="session")
def tokens() -> Tokens:
    return Tokens(TINY.vocab, TINY.seq)
