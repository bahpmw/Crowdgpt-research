"""Causality check for lab model variants: changing token j must not change logits before j.

Usage: python leak_check.py gdn_hybrid pattern=GGGGGG
"""

import importlib
import sys

import torch

from harness import build_model, parse_cfg


def main() -> None:
    method, cfg = sys.argv[1], parse_cfg(sys.argv[2:])
    cfg.setdefault("tokens", 1)
    dev = torch.device("cpu")
    model = build_model("10m", 0, dev)

    class _Data:
        batch, seq = 1, 256

    importlib.import_module(f"methods.{method}").Method(model, _Data, cfg, dev)
    model.eval()
    torch.manual_seed(0)
    idx = torch.randint(0, model.cfg.vocab_size, (1, 256))
    worst = 0.0
    with torch.no_grad():
        base = model(idx)
        base = base[0] if isinstance(base, tuple) else base
        for j in (1, 37, 64, 100, 128, 200, 255):
            alt = idx.clone()
            alt[0, j] = (alt[0, j] + 1) % model.cfg.vocab_size
            out = model(alt)
            out = out[0] if isinstance(out, tuple) else out
            diff = (out[0, :j] - base[0, :j]).abs().max().item()
            later = (out[0, j:] - base[0, j:]).abs().max().item()
            worst = max(worst, diff)
            print(f"changed pos {j}: earlier positions {diff:.3e}, this and later {later:.3e}")
    print("LEAK" if worst > 1e-4 else "causal", f"(worst {worst:.3e})")


if __name__ == "__main__":
    main()
