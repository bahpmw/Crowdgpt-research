"""Error feedback for a biased upload codec.

The ``ngd1``/``ngd1r`` containers keep one sign bit (plus an optional residual
sign) per value and one fp16 scale per 1024-value block.  That is a *biased*
compressor: a value whose magnitude sits well below its block scale decodes far
from where it started, and the error does not average out over rounds because
the same blocks are quantized the same way every time.

Seide et al. (Interspeech 2014) showed that one bit per value costs almost
nothing *if the quantization error is carried forward*: quantize ``g + r``
instead of ``g``, then keep ``r' = (g + r) - Q(g + r)`` for the next round.
Karimireddy et al. (ICML 2019) proved the property we rely on -- with error
feedback the compression factor appears only as a second-order term, so the
method converges to the same point instead of a neighbourhood of it -- and also
showed that sign-based compressors can actively diverge without it.

What this does *not* buy: the coordinator averages unit directions and re-applies
a capped magnitude (see ``Coordinator.finalize``), so feedback improves the
*direction* of each client's contribution, not the applied step size.  A client
whose blocks are all well scaled gains nothing measurable; the win is concentrated
where compression is most aggressive.
"""

from __future__ import annotations

import numpy as np

from . import codec

FRAME_PIECE = 4_000_000


def quantize(vector, codec_name: str):
    """Return exactly what the coordinator will decode from ``vector``.

    Mirrors ``protocol.encode_body`` frame for frame, so a caller cannot end up
    measuring a different quantizer than the one on the wire.
    """

    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if codec_name == "dense":
        return codec.decode_dense(codec.encode_dense(vector), vector.size)
    if codec_name not in ("ngd1", "ngd1r"):
        raise ValueError(f"unknown codec {codec_name!r}")
    pieces = [
        vector[i : i + FRAME_PIECE] for i in range(0, vector.size, FRAME_PIECE)
    ] or [np.zeros(0, dtype=np.float32)]
    blob = codec.pack_frames(pieces, residual=(codec_name == "ngd1r"))
    frames = list(
        codec.iter_frames(
            blob,
            max_total_params=int(vector.size),
            expect_residual=(codec_name == "ngd1r"),
        )
    )
    decoded = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    if decoded.size != vector.size:
        raise ValueError(f"decoded {decoded.size} values, expected {vector.size}")
    return decoded.astype(np.float32, copy=False)


class ErrorFeedback:
    """Carry the quantization error of the previous upload into the next one."""

    def __init__(self, codec_name: str = "ngd1r"):
        if codec_name not in codec.CODECS:
            raise ValueError(f"unknown codec {codec_name!r}")
        self.codec_name = codec_name
        self.residual = None
        self.rounds = 0

    def reset(self) -> None:
        self.residual = None
        self.rounds = 0

    def encode(self, vector):
        """Return the vector to upload plus stats describing the correction."""

        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self.residual is None or self.residual.size != vector.size:
            # A changed parameter count (new preset, new client) invalidates the
            # carried error; starting from zero is the only sound option.
            self.residual = np.zeros_like(vector)
        effective = vector + self.residual
        sent = quantize(effective, self.codec_name)
        residual = effective - sent
        self.residual = residual.astype(np.float32, copy=False)
        self.rounds += 1
        scale = float(np.linalg.norm(vector)) or 1.0
        return sent, {
            "rounds": self.rounds,
            "codec": self.codec_name,
            "residualNorm": float(np.linalg.norm(residual)),
            "residualRelative": float(np.linalg.norm(residual)) / scale,
            "correctionNorm": float(np.linalg.norm(effective - vector)),
        }
