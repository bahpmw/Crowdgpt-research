"""Per-volunteer, per-round bytes for full-replica DiLoCo/SparseLoCo, SOLO pipeline
crews, MOSAIC and MOSAIC+ReLoRA, for the crowd_v15 presets.

Uses the byte formulas of methods/sparseloco.py and methods/mosaic.py (chunk
Top-64 of 64x64 + 2-bit + fp32 scale = 116 bytes per 4096 values; 1-D tensors
bf16) and mosaic's FLOP model (block 16d^2 + 2Td MACs/token, readout Vd).
Prints Markdown. Usage: python bandwidth.py [W] (peers, default 8).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowd_v15.model import ModelConfig  # noqa: E402

H, CHUNK_BYTES, CHUNK = 30, (64 * 14 + 32) / 8, 4096
SETTINGS = {"real": (2048, 8), "lab": (128, 8)}
PRESETS = ("10m", "100m", "300m", "1b")


def mb(x):
    return f"{x / 1e6:,.1f}" if x >= 1e7 else f"{x / 1e6:.3f}"


def modules(c, K):
    """[(n2, n1, lora_mats)] per MOSAIC module (untied head, 1-block local heads)."""
    d, V = c.d_model, c.vocab_size
    sizes = [len(s) for s in torch.arange(c.n_layers).tensor_split(K)]
    mats = [(3 * d, d), (d, d), (4 * d, d), (4 * d, d), (d, 4 * d)]
    out = []
    for k, nb in enumerate(sizes):
        last = k == K - 1
        blocks = nb + (0 if last else 1)
        n2 = blocks * 16 * d * d + (V * d if k == 0 else 0) + (V * d if last else 0)
        n1 = blocks * 2 * d + (d if last else d + 1)
        out.append((n2, n1, mats * blocks, nb))
    return out


def comp(n2, n1):
    return n2 / CHUNK * CHUNK_BYTES + 2 * n1


def main():
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"Per volunteer per round; H={H} local steps, W={W} peers; MB = 1e6 bytes.\n")
    for name, (seq, batch) in SETTINGS.items():
        print(f"### {name} setting: seq {seq} x batch {batch} per step\n")
        print("| preset | params | K | method | download MB | upload MB |")
        print("|---|---|---|---|---|---|")
        for p in PRESETS:
            c = ModelConfig.preset(p)
            d, V, L = c.d_model, c.vocab_size, c.n_layers
            P = c.n_params
            n1 = L * 2 * d + d
            sl_up = comp(P - n1, n1)
            print(f"| {p} | {P / 1e6:.0f}M | 1 | DiLoCo dense bf16 | {mb(2 * P)} | {mb(2 * P)} |")
            print(f"| {p} | | 1 | SparseLoCo | {mb(min(W * sl_up, 2 * P))} | {mb(sl_up)} |")
            for K in (2, 4):
                act = H * seq * batch * d * 2
                mods = modules(c, K)
                msgs = [comp(n2, a) for n2, a, _, _ in mods]
                full = [2 * (n2 + a) for n2, a, _, _ in mods]
                sync_dn = max(min(W // K * m, f) for m, f in zip(msgs, full))
                mid = "both" if K > 2 else "one side"
                print(f"| {p} | | {K} | SOLO crew, raw acts (worst stage: {mid}) "
                      f"| {mb(act + sync_dn)} | {mb(act + max(msgs))} |")
                print(f"| {p} | | {K} | SOLO crew, ResBM 64x acts "
                      f"| {mb(act / 64 + sync_dn)} | {mb(act / 64 + max(msgs))} |")
                # MOSAIC worker k needs modules 0..k without earlier local heads,
                # plus W_bar (V x d) when k < K-1.
                dn_full, dn_delta = [], []
                for k in range(K):
                    need = sum(16 * d * d * mods[j][3] + 2 * d * mods[j][3] for j in range(k))
                    need += mods[k][0] + mods[k][1] + (V * d if k < K - 1 else 0)
                    need += V * d if k > 0 else 0
                    dn_full.append(2 * need)
                    delta = sum(min(W // K * msgs[j], full[j]) for j in range(k + 1))
                    delta += 0 if k == K - 1 else min(W // K * V * d / CHUNK * CHUNK_BYTES, 2 * V * d)
                    dn_delta.append(min(delta, 2 * need))
                print(f"| {p} | | {K} | MOSAIC, bf16 weights (mean / worst k) "
                      f"| {mb(sum(dn_full) / K)} / {mb(max(dn_full))} | {mb(sum(msgs) / K)} / {mb(max(msgs))} |")
                print(f"| {p} | | {K} | MOSAIC, compressed deltas (mean / worst k) "
                      f"| {mb(sum(dn_delta) / K)} / {mb(max(dn_delta))} | {mb(sum(msgs) / K)} / {mb(max(msgs))} |")
                for r in (32, 128):
                    fac = []
                    for n2, a, mats, _ in mods:
                        lin = sum(m * n for m, n in mats)
                        fac.append(2 * r * sum(m + n for m, n in mats) + comp(n2 - lin, a))
                    print(f"| {p} | | {K} | MOSAIC+ReLoRA r{r}, factors up (mean / worst k) "
                          f"| = MOSAIC | {mb(sum(fac) / K)} / {mb(max(fac))} |")
                print(f"| {p} | | {K} | MOSAIC+ReLoRA, compressed product up | = MOSAIC | = MOSAIC |")
        print()
    print("### Compute per volunteer per inner step (forward-equivalents; full backprop step = 3.00)\n")
    print("| preset | setting | K | worker k=0..K-1 | of which redundant prefix forward | crowd total / BP |")
    print("|---|---|---|---|---|---|")
    for p in PRESETS:
        c = ModelConfig.preset(p)
        d, V, L = c.d_model, c.vocab_size, c.n_layers
        for name, (seq, _) in SETTINGS.items():
            b, r = 16 * d * d + 2 * seq * d, V * d
            full = L * b + r
            for K in (2, 4):
                split = [list(map(int, s)) for s in torch.arange(L).tensor_split(K)]
                cs, red = [], []
                for k in range(K):
                    last = k == K - 1
                    fwd = (split[k][-1] + 1) * b + (r if last else b + r)
                    bwd = len(split[k]) * b + (r if last else b + 0.5 * r)
                    cs.append((fwd + 2 * bwd) / full)
                    red.append(split[k][0] * b / full)
                print(f"| {p} | {name} | {K} | {' / '.join(f'{x:.2f}' for x in cs)} "
                      f"| {' / '.join(f'{x:.2f}' for x in red)} | {sum(cs) / K / 3:.2f} |")


if __name__ == "__main__":
    main()
