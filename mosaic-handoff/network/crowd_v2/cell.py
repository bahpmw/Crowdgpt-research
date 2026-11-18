"""One training cell, shared by the client (to train) and the auditor (to replay).

A cell is fully determined by public inputs: round weights, module ``k``, the
beacon-derived data seed, the hyper-parameters and the contributor's error
feedback (EF) buffer. AdamW is fresh every round so the cell can be replayed.

Wire format (one of the two the spec allows): the compressed message is sent
decompressed as dense bf16 safetensors, and ``wire_bytes`` reports the honest
SparseLoCo size (12-bit index + 2-bit code per kept value, one fp32 scale per
64x64 chunk, 1-D tensors dense bf16).
mvp: the dense bf16 transport is far bigger than ``wire_bytes`` (10m preset, module 0:
~10.5 MB raw, ~14 MB as base64 JSON, vs 153 KB by the formula).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .nets import Hyper, Tokens, load_net

CHUNK = 64


def compress(acc: torch.Tensor, k: int) -> torch.Tensor:
    """Chunk Top-k of a 2-D tensor plus 2-bit quantisation, returned dense (lab sparseloco.py)."""
    r, c = acc.shape
    if r % CHUNK or c % CHUNK:
        raise ValueError(f"2-D shape {tuple(acc.shape)} not a multiple of {CHUNK}")
    blocks = acc.reshape(r // CHUNK, CHUNK, c // CHUNK, CHUNK).permute(0, 2, 1, 3)
    blocks = blocks.reshape(-1, CHUNK * CHUNK)
    mag, idx = blocks.abs().topk(k, dim=1)
    s = mag.mean(1, keepdim=True)
    q = torch.where(mag > s, 1.5 * s, 0.5 * s) * blocks.gather(1, idx).sign()
    out = torch.zeros_like(blocks).scatter_(1, idx, q)
    return out.reshape(r // CHUNK, c // CHUNK, CHUNK, CHUNK).permute(0, 2, 1, 3).reshape(r, c)


def wire_bytes(shapes: dict[str, tuple[int, ...]], topk: int) -> int:
    n2 = sum(s[0] * s[1] for s in shapes.values() if len(s) == 2)
    n1 = sum(int(torch.Size(s).numel()) for s in shapes.values() if len(s) != 2)
    return int(n2 // (CHUNK * CHUNK) * (topk * 14 + 32) / 8 + 2 * n1)


@dataclass
class CellOut:
    msg: dict[str, torch.Tensor]      # bf16, cpu: what goes on the wire
    ef: dict[str, torch.Tensor]       # fp32, cpu: EF buffer after this cell
    tokens: int
    loss: float


def run_cell(hp: Hyper, weights: dict[str, torch.Tensor], k: int, seed: int,
             ef: dict[str, torch.Tensor], tokens: Tokens, device: str = "cpu") -> CellOut:
    net = load_net(hp, weights, device)
    names = net.module_names(k)
    named = dict(net.named_parameters())
    for n, p in named.items():
        p.requires_grad_(n in names)
    params = [named[n] for n in names]
    wbar = weights["model.head.weight"].to(device)
    opt = torch.optim.AdamW(params, lr=hp.lr, betas=(0.9, 0.95), weight_decay=0.0)
    loss = torch.zeros(())
    n_tok = 0
    for idx, tgt in tokens.batches(seed, hp.H, hp.batch):
        idx, tgt = idx.to(device), tgt.to(device)
        loss = net.local_loss(k, idx, tgt, wbar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        n_tok += idx.numel()
    msg, ef_out = {}, {}
    with torch.no_grad():
        for n in names:
            d = weights[n].float() - named[n].detach().to("cpu", torch.float32)
            if d.dim() == 2:
                acc = ef.get(n, torch.zeros_like(d)) * hp.beta + d
                msg[n] = compress(acc, hp.topk).bfloat16()
                ef_out[n] = acc - msg[n].float()
            else:
                msg[n] = d.bfloat16()
    return CellOut(msg=msg, ef=ef_out, tokens=n_tok, loss=float(loss.detach()))
