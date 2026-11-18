"""Token sources for the DUST trainer.

The maintainer's scripts default to ``torch.randint`` targets, which have no
learnable signal — a loss curve from that is a smoke test, not a training run.
The default here is an induction task that *is* learnable at small scale, so a
loss curve proves the estimator descends rather than only that it runs.
"""

from __future__ import annotations

import numpy as np
import torch

_DTYPES = {"uint16": np.uint16, "uint32": np.uint32, "int32": np.int32,
           "int64": np.int64, "uint8": np.uint8}


def induction_batch(batch: int, seq_len: int, vocab: int, device: torch.device,
                    generator: torch.Generator, period: int | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy/induction task: the second half repeats the first.

    Predicting position ``t`` requires copying the symbol seen ``seq_len // 2``
    steps earlier, which a small attention model can learn, so loss falls for a
    real reason.
    """
    period = max(2, period or seq_len // 2 + 1)
    if period > seq_len:
        raise ValueError(f"period {period} cannot exceed seq_len {seq_len}")
    symbols = torch.randint(0, vocab, (batch, period), generator=generator)
    # Repeat the block: position t + period always holds the token at t, so each
    # prediction is a copy from exactly one period back. Works for odd lengths.
    reps = (seq_len + 1 + period - 1) // period
    seq = symbols.repeat(1, reps)[:, :seq_len + 1]
    return seq[:, :-1].to(device), seq[:, 1:].to(device)


def random_batch(batch: int, seq_len: int, vocab: int, device: torch.device,
                 generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Parity with the reference scripts: independent uniform ids, no signal."""
    idx = torch.randint(0, vocab, (batch, seq_len), generator=generator)
    tgt = torch.randint(0, vocab, (batch, seq_len), generator=generator)
    return idx.to(device), tgt.to(device)


def identity_batch(batch: int, seq_len: int, vocab: int, device: torch.device,
                   generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict the current token. Trivial, and therefore a clean descent check."""
    idx = torch.randint(0, vocab, (batch, seq_len), generator=generator)
    return idx.to(device), idx.clone().to(device)


def load_tokens(path: str, dtype: str = "uint32") -> torch.Tensor:
    """Load a flat token file. ``dtype`` must match the file's element width."""
    if dtype not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(_DTYPES)}")
    if path.endswith(".npy"):
        array = np.load(path)
    else:
        array = np.fromfile(path, dtype=_DTYPES[dtype])
    if array.size < 2:
        raise ValueError(f"{path}: need at least two tokens, found {array.size}")
    return torch.from_numpy(array.astype(np.int64))


def token_batch(tokens: torch.Tensor, batch: int, seq_len: int, cursor: int,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Next ``seq_len + 1`` tokens, wrapping at the end of the file."""
    need = seq_len + 1
    if tokens.numel() < need:
        raise ValueError(f"need {need} tokens per window, file holds {tokens.numel()}")
    if cursor + need > tokens.numel():
        cursor = 0
    window = tokens[cursor:cursor + need]
    cursor += need
    idx = window[:-1].unsqueeze(0).repeat(batch, 1).to(device)
    tgt = window[1:].unsqueeze(0).repeat(batch, 1).to(device)
    return idx, tgt, cursor


def make_source(task: str, *, batch: int, seq_len: int, vocab: int,
                device: torch.device, generator: torch.Generator,
                tokens: torch.Tensor | None = None):
    """Return ``next_batch() -> (idx, tgt)`` for the requested task."""
    if task == "induction":
        return lambda: induction_batch(batch, seq_len, vocab, device, generator)

    if task == "random":
        return lambda: random_batch(batch, seq_len, vocab, device, generator)

    if task == "identity":
        return lambda: identity_batch(batch, seq_len, vocab, device, generator)

    if task == "copy":
        if tokens is None:
            raise ValueError("task 'copy' needs a token file")
        state = {"cursor": 0}

        def _next():
            idx, tgt, state["cursor"] = token_batch(
                tokens, batch, seq_len, state["cursor"], device
            )
            return idx, tgt

        return _next

    raise ValueError(f"unknown task {task!r}")
