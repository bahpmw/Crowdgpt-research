"""What "a plain transformer" means here, as a rejection rule.

v1 carried a 350M sparse engram table with its own optimizer and its own delta
format.  v1.5 does not, and not only for simplicity: a zeroth-order update
estimates a *scalar* per population member, and applying that to rows the forward
pass never touched is meaningless.  A lookup table that is only sparsely exercised
therefore degrades the estimator in a way the loss curve will not show until much
later.

So the coordinator refuses a checkpoint that carries one instead of training
around it.  The check is name-based and lives in a torch-free module, because the
coordinator must not need a 2 GB deep-learning stack to validate a weight file.
"""

from __future__ import annotations

FORBIDDEN_PARAMETER_MARKERS = ("engram", "moe", "expert", "router", "lora", "adapter")


def check_plain_parameter_names(names) -> None:
    """Raise ``ValueError`` if any parameter name belongs to a non-plain component."""
    offenders = sorted(
        str(name) for name in names
        if any(marker in str(name).lower() for marker in FORBIDDEN_PARAMETER_MARKERS)
    )
    if offenders:
        raise ValueError(
            "v1.5 serves a plain transformer only; refusing extra parameter tables: "
            + ", ".join(offenders[:5])
            + ("" if len(offenders) <= 5 else f" (+{len(offenders) - 5} more)")
        )


__all__ = ["FORBIDDEN_PARAMETER_MARKERS", "check_plain_parameter_names"]
