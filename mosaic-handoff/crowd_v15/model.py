"""A small GPT whose hidden linears can be perturbed in the forward pass.

Pre-norm blocks, fused-QKV attention, SwiGLU MLP, tied embeddings. The only
unusual part is that every hidden projection goes through :func:`_apply_lin`,
which can add ``sign * sigma * noise`` to the layer output and can record the
layer input for the DUST outer product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .plain import check_plain_parameter_names


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    n_layers: int = 20
    d_model: int = 2048
    n_heads: int = 16
    context_length: int = 2048
    mlp_ratio: float = 4.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    @property
    def hidden(self) -> int:
        return int(self.mlp_ratio * self.d_model)

    @property
    def n_params(self) -> int:
        """Analytic parameter count, for printing before the model is built.

        Includes the per-block RMSNorm weights and the final norm. An earlier
        version left them out, which under-reported every preset by ~0.006%
        (63k on 1B) -- small, but a count that disagrees with
        ``GPT(cfg).n_params()`` is a count nobody should quote.
        """
        return param_report(self)["total"]

    @classmethod
    def preset(cls, name: str) -> "ModelConfig":
        if name == "tiny":      # CPU smoke tests
            return cls(vocab_size=256, n_layers=4, d_model=128, n_heads=4, context_length=64)
        if name == "10m":
            return cls(vocab_size=4096, n_layers=6, d_model=256, n_heads=8, context_length=512)
        if name == "300m":
            # The size the author asked to test convergence on: 18 layers x
            # 1024 wide is 306,222,080 parameters including the tied embedding,
            # and a float32 baton is only 1.2 GB on the Hub.
            return cls(vocab_size=4096, n_layers=18, d_model=1024, n_heads=16,
                       context_length=2048)
        if name == "100m":
            return cls(vocab_size=4096, n_layers=12, d_model=768, n_heads=12, context_length=1024)
        if name == "1b":
            # A literal 1B: 15 layers x 2048 wide is 1,015,085,056 parameters
            # including the tied embedding. See ``param_report`` for the count.
            return cls(vocab_size=4096, n_layers=15, d_model=2048, n_heads=16)
        if name == "1b-deep":
            # The sizing the maintainer's train_dust.py calls "1b" (20 layers,
            # 1,350,565,888 parameters). Kept so earlier numbers stay comparable.
            return cls(vocab_size=4096, n_layers=20, d_model=2048, n_heads=16)
        if name == "2b":
            return cls(vocab_size=4096, n_layers=32, d_model=2304, n_heads=18)
        raise ValueError(f"unknown preset {name!r}")


PLAIN_MODULE_TYPES = (
    "GPT", "ModuleList", "Block", "Attention", "MLP", "RMSNorm", "Embedding", "Linear",
)


def assert_plain(model: torch.nn.Module) -> dict[str, int]:
    """Verify the module tree is a plain transformer and return a parameter breakdown."""
    check_plain_parameter_names(dict(model.named_parameters(remove_duplicate=False)))
    unexpected = sorted({
        type(m).__name__ for m in model.modules()
        if type(m).__name__ not in PLAIN_MODULE_TYPES
    })
    if unexpected:
        raise ValueError(f"unexpected module types in a plain transformer: {unexpected}")

    def count(predicate) -> int:
        return sum(
            int(p.numel()) for n, p in canonical_params(model).items() if predicate(n)
        )

    return {
        "embedding": count(lambda n: n.startswith(("tok_emb", "head"))),
        "attention": count(lambda n: ".attn." in n),
        "mlp": count(lambda n: ".mlp." in n),
        "norms": count(lambda n: ".norm" in n or n.startswith("norm_f")),
        "total": int(sum(p.numel() for p in canonical_params(model).values())),
    }


def param_report(cfg: "ModelConfig") -> dict[str, int]:
    """Parameter arithmetic for a preset without allocating it."""
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    per_block = (
        4 * cfg.d_model ** 2
        + 3 * cfg.d_model * int(cfg.mlp_ratio * cfg.d_model)
        + 2 * cfg.d_model  # two RMSNorm weights per block
    )
    total = cfg.vocab_size * cfg.d_model + cfg.n_layers * per_block + cfg.d_model + head
    return {"embedding": cfg.vocab_size * cfg.d_model, "blocks": cfg.n_layers * per_block,
            "finalNorm": cfg.d_model, "untiedHead": head, "total": total}


def canonical_params(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """``name -> tensor`` with tied weights listed once, under their first name.

    With ``tie_embeddings`` the embedding and the head are the same storage, so a
    plain ``named_parameters()`` dict silently drops one of the two names and a
    wire format built from it would disagree with the one built from the other.
    First registration order wins (``tok_emb.weight`` before ``head.weight``),
    which is also what :func:`crowd_v15.dust.dust_gradients` uses as its
    canonical key, so the client's upload layout matches the served weights.
    """
    seen: set[int] = set()
    canonical: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        if id(param) in seen:
            continue
        seen.add(id(param))
        canonical[name] = param
    return canonical


@dataclass
class Perturb:
    """Per-forward perturbation state.

    ``noise[name]`` must already have the shape of that layer's output so the
    same tensors can be reused by the ``+a`` and ``-a`` passes. ``record``, when
    a dict, collects the *clean* layer inputs for the outer product.
    """

    enabled: bool = False
    sigma: float | dict[str, float] = 0.0
    sign: int = 0
    noise: dict[str, torch.Tensor] = field(default_factory=dict)
    record: dict[str, torch.Tensor] | None = None
    only: str | None = None
    # When a dict, the clean pass stores each layer's output RMS here so the
    # noise can be scaled relative to the activation it perturbs. A single
    # absolute sigma is wrong across widths: at ``10m`` the same 1e-3 is a
    # different fraction of the signal than at ``1b``, and below the compute
    # dtype's resolution the perturbation rounds away entirely.
    scales: dict[str, torch.Tensor] | None = None


def layer_sigma(st: Perturb, name: str) -> float:
    """Absolute noise scale for one layer, from a float or a per-layer dict."""
    value = st.sigma
    if isinstance(value, dict):
        return float(value.get(name, 0.0) or 0.0)
    return float(value)


def _apply_lin(x: torch.Tensor, weight: torch.Tensor, name: str,
               st: Perturb | None) -> torch.Tensor:
    if st is not None and st.record is not None:
        st.record[name] = x
    y = x @ weight.t()
    if st is None:
        return y
    if st.scales is not None:
        # Cheap diagnostic: one scalar per layer per clean pass. Kept on the
        # device so the caller can resolve a whole dict without a sync per layer.
        st.scales[name] = y.detach().float().pow(2).mean().sqrt()
    if not st.enabled:
        return y
    if st.only is not None and st.only != name:
        return y
    noise = st.noise.get(name)
    if noise is None:
        return y
    sigma = layer_sigma(st, name)
    if sigma == 0.0:
        return y
    return y + st.sign * sigma * noise


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, st: Perturb | None,
                prefix: str) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = _apply_lin(x, self.qkv.weight, f"{prefix}.qkv", st)
        q, k, v = qkv.split(self.d, dim=-1)
        shape = (b, t, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask, float("-inf"))
        y = torch.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).reshape(b, t, self.d)
        return _apply_lin(y, self.o.weight, f"{prefix}.o", st)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.hidden, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.hidden, bias=False)
        self.down = nn.Linear(cfg.hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, st: Perturb | None, prefix: str) -> torch.Tensor:
        up = _apply_lin(x, self.up.weight, f"{prefix}.up", st)
        gate = _apply_lin(x, self.gate.weight, f"{prefix}.gate", st)
        return _apply_lin(F.silu(gate) * up, self.down.weight, f"{prefix}.down", st)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, st: Perturb | None,
                prefix: str) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask, st, f"{prefix}.attn")
        return x + self.mlp(self.norm2(x), st, f"{prefix}.mlp")


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def noise_shapes(self) -> dict[str, int]:
        """Perturbed layer name -> output width. Order is deterministic."""
        # The embedding is a layer too: its input is one-hot, so its outer
        # product is a scatter-add into the touched rows. With tied embeddings
        # that term is required for the shared tensor's gradient to be complete.
        shapes: dict[str, int] = {"tok_emb": self.cfg.d_model}
        for i in range(self.cfg.n_layers):
            p = f"blocks.{i}"
            shapes[f"{p}.attn.qkv"] = 3 * self.cfg.d_model
            shapes[f"{p}.attn.o"] = self.cfg.d_model
            shapes[f"{p}.mlp.up"] = self.cfg.hidden
            shapes[f"{p}.mlp.gate"] = self.cfg.hidden
            shapes[f"{p}.mlp.down"] = self.cfg.d_model
        return shapes

    def forward(self, idx: torch.Tensor, st: Perturb | None = None) -> torch.Tensor:
        """Return logits ``(B, T, vocab)``. ``st`` records inputs when it has a dict."""
        b, t = idx.shape
        if t > self.cfg.context_length:
            raise ValueError(f"sequence {t} exceeds context {self.cfg.context_length}")
        mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=idx.device), diagonal=1
        )[None, None]
        x = self.tok_emb(idx)
        if st is not None and st.record is not None:
            st.record["tok_emb"] = idx
        if st is not None:
            if st.scales is not None:
                st.scales["tok_emb"] = x.detach().float().pow(2).mean().sqrt()
            if st.enabled and (st.only is None or st.only == "tok_emb"):
                sigma = layer_sigma(st, "tok_emb")
                if sigma:
                    x = x + st.sign * sigma * st.noise["tok_emb"]
        for i, block in enumerate(self.blocks):
            x = block(x, mask, st, f"blocks.{i}")
        h = self.norm_f(x)
        if st is not None and st.record is not None:
            st.record["__head__"] = h
        return h @ self.head.weight.t()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
