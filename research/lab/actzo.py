"""Activation-space ZO engine on crowd_v15.model.GPT (forward-only).

Every estimator here perturbs one layer's *output* H = X W^T by some dH, scores
each sequence by its loss change, and maps credit back to W through the clean
input X. They differ only in how dH is drawn and how credit is mapped:

  np    DUST-like node perturbation: dH = eps*phi, phi iid N(0,1) per token.
        G_b = (dL_b/eps) phi_b^T X_b                                (unbiased)
  ntwp  node-through-weight (2611.08712 Alg.2, per-sample): one shared weight
        probe R (k x q) per probe; dH_b = eps * X_b R^T.
        G_b = k (dL_b/eps) dH_b^T X_b / ||dH_b||^2
        Unbiased for a single token (the paper's linear-layer case); with T
        tokens per sequence it carries the per-sample C_b = X_b^T X_b bias,
        exactly like the paper's conv layers.
  szo1  SZO-1 (2611.08712 App. I): Psi_b = G_b X_b / ||X_b||_F, G_b ~ N(0,1/k)
        of shape k x T; dH_b = eps X_b Psi_b^T.
        G_b-hat = k (dL_b/eps) Psi_b / ||Psi_b||_F^2  (Alg. 1 with the realised
        budget c = ||Psi||^2; the factor k makes it unbiased for one token, like
        ntwp, so the kinds share a learning-rate scale). Psi is never
        materialised: dH uses the T x T Gram X_b X_b^T and the readout
        contracts G_b with X_b directly.

Credit is per sequence (sequences are independent under causal attention).
Probes are one-sided against a shared clean pass (paper convention). Every
probe's dH is rescaled to unit RMS so the finite-difference step is eps (times
the layer output RMS when ``rel``); all three readouts are invariant to that
rescale. Probes of one layer run in batched forwards of ``chunk_rows`` rows.
The tied head/embedding gets the exact softmax gradient, as in this repo's
DUST, so arms differ only in the blocks. RMSNorm weights get no estimate.
``aware`` reweights credit by the clean-pass slope^2 on the MLP up/gate outputs
(2611.08712 App. G): node-space (np, ntwp) multiplies the node credit per
token/unit; weight-space (szo1) weights each per-sample row j of Psi_b by the
slope^2 averaged over tokens with the probe's own (dH)^2 as weights. Slopes are
normalised to mean 1 per layer (SiLU-gated slopes have no natural scale).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from crowd_v15.model import Perturb


def per_seq_loss(logits, tgt):
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1),
                         reduction="none")
    return ce.view(tgt.shape).mean(1)


def layer_weights(model) -> dict[str, torch.Tensor]:
    out = {}
    for i, blk in enumerate(model.blocks):
        p = f"blocks.{i}"
        out[f"{p}.attn.qkv"] = blk.attn.qkv.weight
        out[f"{p}.attn.o"] = blk.attn.o.weight
        out[f"{p}.mlp.up"] = blk.mlp.up.weight
        out[f"{p}.mlp.gate"] = blk.mlp.gate.weight
        out[f"{p}.mlp.down"] = blk.mlp.down.weight
    return out


@torch.no_grad()
def head_grad(model, h, tgt):
    """Exact grad of mean CE w.r.t. the (tied) head weight, and the loss."""
    logits = h @ model.head.weight.t()
    p = torch.softmax(logits.float(), -1)
    p.scatter_add_(-1, tgt.unsqueeze(-1), -torch.ones_like(tgt, dtype=p.dtype).unsqueeze(-1))
    n = tgt.numel()
    return (p.reshape(-1, p.size(-1)).t() @ h.reshape(-1, h.size(-1)).float()) / n


def _slope_weight(model, name, X):
    """Clean-pass slope^2 of the SwiGLU product w.r.t. an up/gate output, mean 1."""
    blk = model.blocks[int(name.split(".")[1])].mlp
    g = X @ blk.gate.weight.float().t()
    if name.endswith("up"):
        slope = F.silu(g)
    else:
        sg = torch.sigmoid(g)
        slope = sg * (1 + g * (1 - sg)) * (X @ blk.up.weight.float().t())
    w = slope.pow(2)
    return w / w.mean()


@torch.no_grad()
def estimate(model, idx, tgt, *, kind="np", n=4, eps=1e-2, rel=True, aware=False,
             layers=None, gen=None, chunk_rows=256):
    """Return (grads keyed like model.named_parameters, forwards, stats).

    ``forwards`` counts full-model passes over the B sequences (one clean pass
    plus one per probe per layer).
    """
    if kind not in ("np", "ntwp", "szo1"):
        raise ValueError(kind)
    B, T = idx.shape
    st = Perturb(record={})
    L0 = per_seq_loss(model(idx, st), tgt)                       # (B,)
    rec = st.record
    W = layer_weights(model)
    grads = {"tok_emb.weight": head_grad(model, rec["__head__"], tgt)}
    forwards = 1
    dev = idx.device
    per = max(1, chunk_rows // B)                                # probes per forward
    for name in layers or (["tok_emb"] + list(W)):
        emb = name == "tok_emb"
        if emb:
            X, k = None, model.cfg.d_model
            ref = model.tok_emb(idx).float()
        else:
            X = rec[name].float()                                # (B,T,q)
            k = W[name].shape[0]
            ref = X @ W[name].float().t()
        scale = float(eps * (ref.pow(2).mean().sqrt() if rel else 1.0))
        w = (_slope_weight(model, name, X)
             if aware and not emb and name.endswith(("mlp.up", "mlp.gate")) else None)
        mode = "np" if emb else kind
        if mode == "szo1":
            xn = X.pow(2).sum(dim=(1, 2)).sqrt()                 # (B,)
            K = X @ X.transpose(1, 2)                            # (B,T,T) Gram
        acc = torch.zeros(model.cfg.vocab_size if emb else k, model.cfg.d_model if emb
                          else X.shape[-1], device=dev)
        for s in range(0, n, per):
            c = min(per, n - s)
            if mode == "np":
                dH = torch.randn(c, B, T, k, device=dev, generator=gen)
            elif mode == "ntwp":
                R = torch.randn(c, k, X.shape[-1], device=dev, generator=gen)
                dH = torch.einsum("btq,ckq->cbtk", X, R)
                dH = dH / dH.pow(2).mean(dim=(2, 3), keepdim=True).sqrt()
            else:
                Gb = torch.randn(c, B, k, T, device=dev, generator=gen) / k ** 0.5
                Gb = Gb / xn.view(1, B, 1, 1)                    # Psi = Gb X
                dH = torch.einsum("bst,cbkt->cbsk", K, Gb)       # X Psi^T
                norm = dH.pow(2).mean(dim=(2, 3), keepdim=True).sqrt()
                dH, Gb = dH / norm, Gb / norm.transpose(2, 3)
            st2 = Perturb(enabled=True, sigma=scale, sign=1, only=name,
                          noise={name: dH.reshape(c * B, T, k)})
            out = model(idx.repeat(c, 1), st2)
            forwards += c
            dL = (per_seq_loss(out, tgt.repeat(c, 1)).view(c, B) - L0.view(1, B)) / scale
            if mode == "szo1":
                psi2 = (torch.einsum("cbkt,bts->cbks", Gb, K) * Gb).sum(dim=(2, 3))
                coef = (k * dL / psi2).view(c, B, 1, 1).expand(c, B, k, 1)
                if w is not None:
                    p = dH.pow(2)
                    coef = ((w.unsqueeze(0) * p).sum(2) / p.sum(2)).unsqueeze(-1) * coef
                acc += torch.einsum("cbkt,btq->kq", coef * Gb, X)
                continue
            credit = dL.view(c, B, 1, 1) * dH                    # (c,B,T,k)
            if w is not None:
                credit = credit * w.unsqueeze(0)
            if emb:
                acc.index_add_(0, idx.reshape(-1), credit.sum(0).reshape(B * T, k))
            else:
                # ntwp: k/||dH_b||^2 = k/(T k) with unit-RMS dH
                acc += torch.einsum("cbtk,btq->kq", credit, X) / (T if mode == "ntwp" else 1)
        if emb:
            grads["tok_emb.weight"] += acc / (n * B)
        else:
            grads[f"{name}.weight"] = acc / (n * B)
    return grads, forwards, {"loss": float(L0.mean())}


def backprop(model, idx, tgt):
    model.zero_grad(set_to_none=True)
    logits = model(idx)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()
    seen, out = set(), {}
    for n_, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen or p.grad is None:
            continue
        seen.add(id(p))
        out[n_] = p.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    return out


def cosine(est: dict, ref: dict, skip=("tok_emb.weight",)):
    keys = [k for k in est if k in ref and k not in skip]
    a = torch.cat([est[k].flatten().double() for k in keys])
    b = torch.cat([ref[k].flatten().double() for k in keys])
    return float(a @ b / (a.norm() * b.norm() + 1e-30))
