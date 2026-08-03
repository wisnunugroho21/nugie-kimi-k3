"""Attention Residuals (AttnRes) — §2.2 of the Kimi K3 report.

THE IDEA IN ONE SENTENCE: do to *depth* what the Transformer did to *time*.

A standard residual stream is a recurrence over layers, `h_{l+1} = h_l + f_l(h_l)`.
Every layer's output is folded in with weight exactly 1 and never re-selected, so
by layer 90 the token embedding is one summand among ninety — the same
information bottleneck an RNN has over time. The Transformer's answer to that
bottleneck over time was attention. AttnRes applies the identical answer over
depth: each layer *attends* over the outputs of all preceding layers and picks
what it needs, with data-dependent weights.

FULL FORM (Eqs. 8-9)
--------------------
Layer `l` owns a single learned vector `w_l in R^d` — its "pseudo-query". Keys
and values are the preceding layer outputs (and the token embedding, as source
index 0):

    k_i = v_i = h_1              if i = 0      (the embedding)
                f_i(h_i)         if 1 <= i <= l-1

    alpha_{i->l} = softmax_i( w_l . RMSNorm(k_i) )
    h_l          = sum_i alpha_{i->l} v_i

The RMSNorm on the keys is load-bearing: without it a layer whose outputs happen
to have large magnitude would dominate every downstream softmax purely on scale.
Note the parameter cost of the whole mechanism is one d-vector per layer.

BLOCK FORM (Eq. 10) — what K3 actually uses
-------------------------------------------
The full form needs every layer output kept alive: O(L*d) memory, and under
pipeline parallelism O(L*d) of cross-stage traffic. So the L layers are cut into
N blocks of S layers, and each block is *summed* down to one representation:

    b_n = sum_{j in block n} f_j(h_j),        b_0 = the token embedding
    b_n^i = the partial sum over the first i layers of block n

Attention then runs over block representations only. For the i-th layer of
block n the value set is

    V = [b_0, ..., b_{n-1}]              if i = 1   (block just started)
        [b_0, ..., b_{n-1}, b_n^{i-1}]   if i >= 2  (plus what this block has
                                                     accumulated so far)

So: coarse, attention-weighted access *across* blocks; ordinary summation
*within* a block. Memory drops from O(L*d) to O(N*d). It also bounds the
inference-time state, which lets the parallel inter-block term be merged with
the sequential intra-block partial sum by online softmax.

K3 uses N = 8 blocks of 12 layers over 93 layers (the last block is partial), or
9 sources counting the embedding. The final output layer attends over all of
them.

ONE NOTE ON GRANULARITY. Eq. 10 indexes the value set by *layer*, while Fig. 2
draws a separate (alpha, w) on every module — the KDA/MLA mixer and the MoE
mixer each have their own. Both readings are honoured here: a layer computes its
value set V once, and its attention module and its FFN module each attend over
that same V with their own pseudo-query. The layer's contribution to the running
block sum is the sum of the two module outputs, i.e. `f_i(h_i)` in Eq. 10.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .layers import F32, rms_norm


class DepthAttention(nnx.Module):
    """One pseudo-query `w` (Eqs. 8-9). This is the entire AttnRes parameter set.

    `w` is initialised to ZERO, which makes the softmax uniform: at step 0 every
    module reads the plain average of the sources available to it. That is a
    sane, well-scaled starting point (it cannot blow up the way a random query
    over unnormalised sources could), and the module learns its selectivity from
    there.
    """

    def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        self.w = nnx.Param(jnp.zeros((d_model,), dtype=F32))

    def __call__(self, values: jax.Array) -> jax.Array:
        """values: [S, B, T, d] — the S sources of Eq. 10. Returns [B, T, d].

        The softmax runs over S, per token, and is shared across the whole
        hidden vector: the mechanism chooses *which depths* to read, not which
        channels (that is the FFN's and the output gates' job).
        """
        keys = rms_norm(values)  # prevents scale from deciding the weights
        logits = jnp.einsum("sbtd,d->bts", keys.astype(F32), self.w[...])
        alpha = jax.nn.softmax(logits, axis=-1)  # [B, T, S]
        return jnp.einsum("bts,sbtd->btd", alpha, values.astype(F32))


class AttnResStream:
    """The bookkeeping of Eq. 10, as a small mutable helper (not an nnx.Module).

    Holds the completed block representations `b_0 .. b_{n-1}` plus the running
    partial sum `b_n^i` of the block currently being built. It replaces the
    single `h` that a normal transformer threads through its layers.

    A consequence worth knowing: the very first layer's value set is `[b_0]`
    alone, and a softmax over one source is identically 1 regardless of the
    query. So layer 0's pseudo-queries are mathematically inert and receive
    exactly zero gradient for the whole of training — 2*d dead parameters out of
    2.8T. That falls out of Eq. 10 and is not a defect, but an optimiser or a
    test that assumes every parameter moves will notice.
    """

    def __init__(self, embedding: jax.Array):
        # b_0 is the token embedding, so every layer can always reach the input
        # directly no matter how deep it sits.
        self.sources: list[jax.Array] = [embedding]
        self.partial: jax.Array | None = None

    def values(self) -> jax.Array:
        """The value matrix V of Eq. 10, stacked as [S, B, T, d]."""
        vals = self.sources if self.partial is None else self.sources + [self.partial]
        return jnp.stack(vals, axis=0)

    def accumulate(self, layer_output: jax.Array) -> None:
        """Add `f_i(h_i)` to the current block's partial sum."""
        self.partial = layer_output if self.partial is None else self.partial + layer_output

    def close_block(self) -> None:
        """Seal `b_n` and start the next block."""
        assert self.partial is not None, "closing an empty AttnRes block"
        self.sources.append(self.partial)
        self.partial = None

    def finish(self) -> None:
        """Seal a trailing partial block (K3's 93 layers leave one)."""
        if self.partial is not None:
            self.close_block()

    @property
    def num_sources(self) -> int:
        return len(self.sources) + (self.partial is not None)
