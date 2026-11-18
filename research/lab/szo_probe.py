"""probe.py with an MPS-safe cosine (probe.block_cosine calls .double() before
.cpu(), which MPS rejects). Same CLI as probe.py."""

import torch

import probe


def block_cosine(est: dict, ref: dict) -> float:
    keys = [k for k in est if k in ref and k.startswith("blocks.") and ref[k].dim() == 2]
    a = torch.cat([est[k].flatten().cpu().double() for k in keys])
    b = torch.cat([ref[k].flatten().cpu().double() for k in keys])
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


if __name__ == "__main__":
    probe.block_cosine = block_cosine
    probe.main()
