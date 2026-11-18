"""SparseLoCo as run in Covenant-72B (arXiv 2603.08163): W simulated peers each run
H local AdamW steps from the global weights, then send a chunked Top-k + 2-bit
pseudo-gradient with error feedback; the outer step is plain SGD with alpha=1.

One harness ``step()`` = one outer round = W*H inner steps. ``model`` holds the
global weights; peers are deep-copied replicas with persistent AdamW state.

Faithful: H=30, inner AdamW betas (0.9, 0.95), clip 1; Delta = theta_global -
theta_local; acc = beta*e + Delta (beta=0.95); Delta_hat = Q(TopK_chunk(acc));
e <- acc - Delta_hat; theta -= alpha * mean(Delta_hat), no outer momentum.
Top-k keeps k=64 of every 64x64 block of each 2-D tensor (incl. the tied
embedding/head). Bytes: 12-bit index + 2-bit code per kept value.

Simplified / assumed (paper leaves these open or they do not fit 120 s):
- 2-bit quantiser: per 64x64 chunk scale s = mean|kept|; code = sign + (|v| > s);
  dequantised to +-0.5s / +-1.5s. One fp32 scale per chunk is counted in bytes.
- 1-D tensors (RMSNorm gains, 13 x 256 values) go dense in bf16 with no EF.
- Inner lr constant 3e-3, wd 0 (matches the harness adam baseline), no warmup.
- ``topk=0``: dense fp32 Delta, no EF/quantisation = plain DiLoCo with outer SGD;
  ``outer=nesterov``: DiLoCo outer SGD-Nesterov (olr 0.7, mom 0.9).
- ``gauntlet=1`` (Gauntlet-lite): LossScore_r = L(theta) - L(theta - alpha*Delta_hat_r)
  on one fresh validator batch (``vbatch`` 8 windows) per round; peers with
  score <= 0 are dropped (``gscale`` multiplies alpha in the probe step). No
  OpenSkill history, no assigned-vs-unassigned check. ``mednorm=1`` rescales each
  kept message to the median kept norm (the paper's normalisation, form assumed).
- ``byz=f``: peers 0..f-1 replace their message by -c*Delta_hat (attack=flip, c=5)
  or random values on the same support with c*||Delta_hat|| norm (attack=noise, c=1).
- ``down``: object-store model, every peer fetches every peer's message
  (dense arm: every peer fetches the fp32 average).
- ``simTime``: synchronous rounds cost H / min(speeds) (speeds default all 1).
"""

import copy

import torch

from crowd import assign, corrupt, local_adamw, norm
from harness import Cost, loss_fn
from zo import gen, trainable

CHUNK = 64  # 64x64 blocks = 4096 values per chunk


def compress(acc: torch.Tensor, k: int) -> torch.Tensor:
    """Chunk Top-k of a 2-D tensor plus 2-bit quantisation, returned dense."""
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


class Method:
    def __init__(self, model, data, cfg, device):
        self.model, self.data, self.device = model, data, device
        self.W = int(cfg.get("workers", 4))
        self.H = int(cfg.get("H", 30))
        self.k = int(cfg.get("topk", 64))
        self.beta = float(cfg.get("beta", 0.95))
        self.alpha = float(cfg.get("alpha", 1.0))
        self.outer = cfg.get("outer", "sgd")
        self.olr = float(cfg.get("olr", 0.7))
        self.mom = float(cfg.get("mom", 0.9))
        self.noniid = bool(cfg.get("noniid", 0))
        self.gauntlet = bool(cfg.get("gauntlet", 0))
        self.mednorm = bool(cfg.get("mednorm", 0))
        self.vbatch = int(cfg.get("vbatch", 8))
        self.gscale = float(cfg.get("gscale", 1.0))
        self.byz = int(cfg.get("byz", 0))
        self.attack = cfg.get("attack", "flip")
        self.c = float(cfg.get("c", 5.0 if self.attack == "flip" else 1.0))
        speeds = cfg.get("speeds", [1.0] * self.W)
        self.round_time = self.H / min(speeds)
        self.clip = float(cfg.get("clip", 1.0))
        self.G = trainable(model)
        self.reps = [copy.deepcopy(model) for _ in range(self.W)]
        self.rp = [trainable(r) for r in self.reps]
        self.opts = [torch.optim.AdamW(r.parameters(), lr=float(cfg.get("lr", 3e-3)),
                                       betas=(0.9, 0.95), weight_decay=float(cfg.get("wd", 0.0)))
                     for r in self.reps]
        self.ef = [{n: torch.zeros_like(p) for n, p in self.G.items() if p.dim() == 2}
                   for _ in range(self.W)] if self.k else None
        self.buf = {n: torch.zeros_like(p) for n, p in self.G.items()}
        n2 = sum(p.numel() for p in self.G.values() if p.dim() == 2)
        n1 = sum(p.numel() for p in self.G.values() if p.dim() == 1)
        self.n_params = n1 + n2
        chunks = n2 // (CHUNK * CHUNK)
        self.msg_bytes = (chunks * (self.k * 14 + 32) / 8 + 2 * n1) if self.k else 4 * self.n_params
        self.sim = 0.0

    @torch.no_grad()
    def _message(self, w: int) -> dict:
        delta = {n: self.G[n] - self.rp[w][n] for n in self.G}
        if not self.k:
            return delta
        msg = {}
        for n, d in delta.items():
            if d.dim() == 2:
                acc = self.ef[w][n].mul_(self.beta).add_(d)
                msg[n] = compress(acc, self.k)
                acc.sub_(msg[n])
            else:
                msg[n] = d.bfloat16().float()
        return msg

    def step(self, it):
        msgs, tokens, inner = [], 0, 0.0
        for w in range(self.W):
            assign(self.rp[w], self.G)
            t, loss = local_adamw(self.reps[w], self.opts[w], self.data, self.H, clip=self.clip,
                                  worker=w, workers=self.W, noniid=self.noniid)
            tokens += t
            inner += loss / self.W
            msg = self._message(w)
            if w < self.byz:
                msg = corrupt(msg, self.attack, self.c, gen(it * 1000 + w, self.device))
            msgs.append(msg)
        val_fwd = 0
        with torch.no_grad():
            keep = list(range(self.W))
            if self.gauntlet:
                idx, tgt = self.data.next(batch=self.vbatch)
                base = float(loss_fn(self.model, idx, tgt))
                keep = []
                for w, m in enumerate(msgs):
                    for n, p in self.G.items():
                        p.sub_(m[n], alpha=self.alpha * self.gscale)
                    if base - float(loss_fn(self.model, idx, tgt)) > 0:
                        keep.append(w)
                    for n, p in self.G.items():
                        p.add_(m[n], alpha=self.alpha * self.gscale)
                val_fwd = (self.W + 1) * idx.numel()
            norms = [float(norm(msgs[w])) for w in keep]
            if self.mednorm and keep:
                med = sorted(norms)[len(norms) // 2]
                msgs = [{n: v * (med / (norms[keep.index(w)] + 1e-12)) for n, v in m.items()}
                        if w in keep else m for w, m in enumerate(msgs)]
            if keep:
                for n, p in self.G.items():
                    avg = sum(msgs[w][n] for w in keep) / len(keep)
                    if self.outer == "nesterov":
                        b = self.buf[n].mul_(self.mom).add_(avg)
                        p.sub_(avg + self.mom * b, alpha=self.olr)
                    else:
                        p.sub_(avg, alpha=self.alpha)
        self.sim += self.round_time
        up = self.W * self.msg_bytes
        down = self.W * (up if self.k else 4 * self.n_params)
        extra = {"simTime": self.sim, "innerLoss": round(inner, 4), "kept": len(keep),
                 "msgNorm": round(sum(norms) / max(len(norms), 1), 4),
                 "bitsPerParam": round(8 * self.msg_bytes / self.n_params, 4)}
        return Cost(tokens=tokens, fwd=tokens + val_fwd, bwd=tokens, up=up, down=down, extra=extra)
