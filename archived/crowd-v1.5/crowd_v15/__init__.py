"""Crowd-v1.5: DUST pretraining for the CrowdGPT client/server line.

The v1 stack trained by backprop and shipped dense weight deltas.  v1.5 replaces
the training method: a client perturbs activations, evaluates a whole population
of virtual members inside one forward pass, and uploads a weight-shaped estimate
of the descent direction.  No backward pass, no optimizer state, no gradient
buffer -- which is what makes a 1B model trainable on a client that could not
backprop it.

Modules
-------
``model``        the GPT the coordinator serves (RMSNorm, fused-QKV, SwiGLU)
``dust``         the estimator, the optimizer and the alignment probe
``tasks``        data sources: identity, induction, copy, random
``codec``        NGD1/NGD1R gradient containers (same layout as v1)
``protocol``     header + body framing and the guards around both
``store``        atomic on-disk weights, status and the audit log
``coordinator``  which submissions count, how they combine, when the model moves
``server.app``   the HTTP surface: separate /submit/dust and /submit/backprop
``client``       a DUST client that trains, verifies the served weights, uploads
"""

__version__ = "1.5.0"

__all__ = ["__version__"]
