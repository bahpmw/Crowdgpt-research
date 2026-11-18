"""Model, MOSAIC module split, token data and held-out loss.

The GPT comes from ``crowd_v15.model`` with the head untied (cloned from the
embedding, as in the lab's mosaic.py). Modules for ``K`` cells: blocks are cut
into ``K`` contiguous groups; module 0 also owns ``tok_emb``; module k < K-1
owns its local head; the last module owns ``norm_f`` and ``head``.

mvp: token ids are taken modulo the preset vocabulary, so the ``tiny`` preset
(vocab 256) can train on the 4096-token files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import paths
from crowd_v15.model import GPT, Block, ModelConfig, RMSNorm  # noqa: E402  (path set by paths)


@dataclass(frozen=True)
class Hyper:
    preset: str = "tiny"
    K: int = 2
    H: int = 30
    batch: int = 8
    seq: int = 64
    lr: float = 3e-3
    topk: int = 64
    beta: float = 0.95
    alpha: float = 1.0
    gate_tol: float = 0.25
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def vocab(self) -> int:
        return ModelConfig.preset(self.preset).vocab_size


class LocalHead(nn.Module):
    """SOLO local head: one Block + RMSNorm + learned scale, read out through W_bar."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg)])
        self.norm = RMSNorm(cfg.d_model)
        self.tau = nn.Parameter(torch.ones(()))
        self.apply(GPT._init_weights)

    def forward(self, h: torch.Tensor, mask: torch.Tensor, wbar: torch.Tensor) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            h = blk(h, mask, None, f"head.{i}")
        return self.tau * (self.norm(h) @ wbar.t())


class Net(nn.Module):
    def __init__(self, hp: Hyper):
        super().__init__()
        cfg = ModelConfig.preset(hp.preset)
        self.model = GPT(cfg)
        self.model.head.weight = nn.Parameter(self.model.tok_emb.weight.detach().clone())
        self.heads = nn.ModuleList([LocalHead(cfg) for _ in range(hp.K - 1)])
        self.K = hp.K
        self.split = [list(map(int, c)) for c in torch.arange(cfg.n_layers).tensor_split(hp.K)]

    def module_names(self, k: int) -> list[str]:
        names = list(dict(self.named_parameters()))
        pre = tuple(f"model.blocks.{i}." for i in self.split[k])
        keep = [n for n in names if n.startswith(pre)]
        keep += ["model.tok_emb.weight"] if k == 0 else []
        keep += ([n for n in names if n.startswith(f"heads.{k}.")] if k < self.K - 1
                 else ["model.norm_f.weight", "model.head.weight"])
        return keep

    def local_loss(self, k: int, idx: torch.Tensor, tgt: torch.Tensor,
                   wbar: torch.Tensor) -> torch.Tensor:
        """Earlier modules run forward only; module k learns from its own local loss."""
        m = self.model
        t = idx.shape[1]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=idx.device), 1)[None, None]
        with torch.set_grad_enabled(k == 0):
            h = m.tok_emb(idx)
            for i in range(self.split[k][0]):
                h = m.blocks[i](h, mask, None, f"blocks.{i}")
        for i in self.split[k]:
            h = m.blocks[i](h, mask, None, f"blocks.{i}")
        logits = (self.heads[k](h, mask, wbar) if k < self.K - 1
                  else m.norm_f(h) @ m.head.weight.t())
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))


def init_weights(hp: Hyper) -> dict[str, torch.Tensor]:
    torch.manual_seed(hp.seed)
    return {n: p.detach().clone() for n, p in Net(hp).named_parameters()}


def load_net(hp: Hyper, weights: dict[str, torch.Tensor], device: str) -> Net:
    net = Net(hp)
    net.load_state_dict(weights, strict=True)
    return net.to(device)


def module_names(hp: Hyper) -> list[list[str]]:
    net = Net(hp)
    return [net.module_names(k) for k in range(hp.K)]


class Tokens:
    def __init__(self, vocab: int, seq: int):
        self.train = np.memmap(paths.DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(paths.DATA_DIR / "val.bin", dtype=np.uint16, mode="r")
        self.vocab, self.seq = vocab, seq

    def _windows(self, arr: np.ndarray, starts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.stack([arr[s:s + self.seq + 1] for s in starts]).astype(np.int64) % self.vocab
        t = torch.from_numpy(x)
        return t[:, :-1], t[:, 1:]

    def batches(self, seed: int, steps: int, batch: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Training batches nobody chooses: windows drawn from a public seed."""
        rng = np.random.default_rng(seed)
        hi = len(self.train) - self.seq - 1
        return [self._windows(self.train, rng.integers(0, hi, size=batch)) for _ in range(steps)]

    def val_batch(self, n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        starts = np.random.default_rng(12345).integers(0, len(self.val) - self.seq - 1, size=n)
        return self._windows(self.val, starts)


@torch.no_grad()
def eval_loss(hp: Hyper, weights: dict[str, torch.Tensor], tokens: Tokens) -> float:
    """Held-out loss of the full model (untied head) on the fixed val windows, on cpu."""
    net = load_net(hp, weights, "cpu").eval()
    idx, tgt = tokens.val_batch()
    total = 0.0
    for i, j in zip(idx.chunk(4), tgt.chunk(4)):
        logits = net.model(i)
        total += float(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), j.reshape(-1)))
    return total / 4
