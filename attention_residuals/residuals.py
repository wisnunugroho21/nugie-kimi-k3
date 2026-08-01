"""
Attention Residuals (AttnRes) — Kimi K3 §2.2, in JAX / Flax NNX.

THE IDEA IN ONE PARAGRAPH
------------------------
A standard transformer accumulates depth with `x = x + f_l(x)`: every layer's
input is ONE state that has compressed everything before it. The paper points
out that this is exactly the RNN bottleneck, just along depth instead of time —
and that the Transformer already solved that problem along time by replacing the
recurrent carry with ATTENTION. AttnRes applies the same fix to depth: instead of
inheriting a single accumulated stream, each module ATTENDS over the token
embedding and the outputs of all preceding layers, with data-dependent weights,
and reads a weighted combination of them (K3 §2.2).

  Standard residual :  h_l = h_{l-1} + f_{l-1}(h_{l-1})   (uniform accumulation)
  AttnRes           :  h_l = Σ_i α_{i→l} · v_i            (selective retrieval)

Note there is NO additive residual left. The read is a softmax mixture (the α's
sum to 1) over sources that always include the token embedding b_0, so the
network can still recover "pass the input through" behaviour — it just has to
choose it, per token, per layer.

FULL AttnRes (Eqs. 8-9)
-----------------------
Each layer l owns a learnable *pseudo-query* q_l = w_l ∈ R^d (a plain parameter
vector — there is no query projection, because the "sequence" being attended
over is the list of layers, not tokens). The keys and values are the same
objects, namely the layer outputs:

    k_i = v_i = { h_1            i = 0     (the token embedding)          Eq. 8
                { f_i(h_i)       1 ≤ i ≤ l-1  (the OUTPUT of layer i)

    φ(q, k)   = exp( qᵀ RMSNorm(k) )                                      Eq. 9
    α_{i→l}   = φ(q_l, k_i) / Σ_{j<l} φ(q_l, k_j)
    h_l       = Σ_{i<l} α_{i→l} · v_i

Two details worth pausing on:
  * RMSNorm is applied to the KEY only. Without it, a layer whose outputs happen
    to have large magnitude would win every dot product and dominate the weights
    regardless of content; normalizing makes the scores about DIRECTION. The
    VALUE stays un-normalized (Eq. 9 mixes v_i, not RMSNorm(v_i)), so the actual
    magnitudes of the layer outputs are preserved in the mixture.
  * v_i is the layer's OWN OUTPUT f_i(h_i) — its delta contribution — not a
    running sum. The running sums only appear in the block variant below.

BLOCK AttnRes (Eq. 10) — what K3 actually uses
----------------------------------------------
Full AttnRes must keep every layer output alive: O(L·d) memory per token, and
cross-stage traffic under pipeline parallelism. So K3 partitions the L layers
into N blocks of S = L/N layers and attends over BLOCK-level representations:

    b_n     = Σ_{j ∈ B_n} f_j(h_j)      the sum of block n's layer outputs
    b_n^i   = the partial sum over the first i layers of block n
    b_0     = h_1                       the token embedding is always a source

    V = [b_0, b_1, …, b_{n-1}]              for the FIRST module of block n
    V = [b_0, b_1, …, b_{n-1}, b_n^{i-1}]   for later modules of block n     Eq. 10

with the same keys/weights as Eqs. 8-9. So a module sees: every FINISHED block
as its own separate source, plus one running partial sum for the block it is
currently inside — inter-block retrieval is selective, intra-block accumulation
is the ordinary uniform sum. Memory drops from O(Ld) to O(Nd). The paper reports
N ≈ 8 recovers most of the benefit; K3 uses 8 blocks of 12 layers (9 sources in
total once b_0 is counted).

The final output layer then aggregates all N block representations — i.e. one
more read, with its own pseudo-query, over the closed blocks (see
`AttnResBus.close_block` + a final `AttnResQuery` in the model).

WHY THIS COSTS ALMOST NOTHING
-----------------------------
Depth is small (L < 100), so the depth-attention is O(L²d) arithmetic against
the O(L·d²) of the layers themselves. The report quotes ≈ 4% added training cost
and ≈ 2% added inference cost.

STREAMING
---------
AttnRes attends across LAYERS, not across tokens, so all of its state lives
inside a single forward pass. There is no cache to carry between decode steps —
which is why `AttnResBus` is a plain Python object and not part of GDN2Cache /
MLACache.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

F32 = jnp.float32


def _rms_normalize(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm WITHOUT a learnable gain, as used inside the AttnRes kernel φ.

    Eq. 9 writes φ(q, k) = exp(qᵀ RMSNorm(k)) and specifies no gain vector; a
    per-channel gain here would in any case be redundant with the learnable
    pseudo-query q it is dotted against (the two multiply channel-wise before
    the sum, so q can absorb it). Keeping it gain-free also means the score is
    purely about the DIRECTION of each source, which is the stated purpose.
    """
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


class AttnResQuery(nnx.Module):
    """One module's learnable pseudo-query w_l (Eq. 8), plus the read it performs.

    Every module that consumes a hidden state — each token mixer AND each channel
    mixer, per Fig. 2, where an (α, w) pair hangs off every KDA / Gated MLA /
    Stable LatentMoE box — owns one of these.

    Initialized to ZERO, which makes every score qᵀRMSNorm(k) = 0 and hence the
    initial α uniform over all available sources: at step 0 a module reads the
    plain average of the embedding and the preceding blocks. That is the neutral
    starting point (no source is preferred a priori); training then sharpens the
    α's. The paper does not prescribe an init.
    """

    def __init__(self, d_model: int):
        self.w = nnx.Param(jnp.zeros((d_model,), F32))

    def __call__(self, bus: "AttnResBus") -> jax.Array:
        """Read the bus with this module's pseudo-query -> h_l: [B, L, d_model]."""
        return bus.read(self.w[...])


class AttnResBus:
    """The set of depth-sources visible right now, and the block bookkeeping.

    NOT an nnx.Module: it holds intermediate ACTIVATIONS for one forward pass
    (the b_n's), never parameters, and it is rebuilt from scratch on every call.
    The learnable part of AttnRes lives entirely in the per-module
    `AttnResQuery` vectors.

    Usage inside a model's forward (see KimiK3.__call__):

        bus = AttnResBus(embedding, block_size=cfg.attnres_block_size)
        for layer in layers:
            h = layer.q_attn(bus)          # Eq. 9/10 read
            f = layer.token_mixer(norm(h))
            bus.add(f)                     # contribute to the open block's sum
            ...same for the channel mixer...
            bus.end_layer()                # closes the block every `block_size` layers
        bus.close_block()                  # flush a trailing partial block
        x = final_query(bus)               # "the final output layer aggregates all N blocks"
    """

    def __init__(self, embedding: jax.Array, block_size: int):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size

        # Closed block representations [b_0, b_1, …]. b_0 = h_1, the token
        # embedding (Eq. 10: "we set b_0 = h_1 so the token embedding is always
        # included as a source").
        self.blocks: list[jax.Array] = [embedding]

        # b_n^i, the running partial sum of the block currently open. None means
        # the block is still empty — which is exactly the i = 1 case of Eq. 10,
        # where the value matrix contains only the closed blocks.
        self.partial: jax.Array | None = None

        self._layers_in_block = 0

    # ---- the read (Eqs. 9 / 10) ------------------------------------------- #
    def read(self, q: jax.Array) -> jax.Array:
        """Depth-attention with pseudo-query `q` [d_model] -> [B, L, d_model].

        Sources are the closed blocks plus (if non-empty) the open block's
        partial sum — precisely the value matrix V of Eq. 10.
        """
        sources = self.blocks if self.partial is None else [*self.blocks, self.partial]

        # First module of the network: b_0 is the only source, and a softmax over
        # one entry is 1. Short-circuit — this is not an approximation, it is the
        # same value without building a length-1 attention.
        # (Consequence worth knowing: the very first module's pseudo-query is
        # then structurally unused and gets an exactly zero gradient. That is the
        # correct answer — with one source there is nothing to choose — not a
        # wiring bug.)
        if len(sources) == 1:
            return sources[0]

        V = jnp.stack(sources, axis=-2)  # [B, L, S, d]  values v_i (un-normalized)

        # φ(q, k) = exp(qᵀ RMSNorm(k)); the softmax below supplies the exp and the
        # normalization of Eq. 9 in one numerically stable step (it subtracts the
        # row max, which cancels in the ratio). Scores in fp32: S is small but the
        # exponential is not forgiving in bf16.
        scores = jnp.einsum("...sd,d->...s", _rms_normalize(V.astype(F32)), q.astype(F32))
        alpha = jax.nn.softmax(scores, axis=-1)  # α_{i→l}: [B, L, S]

        # h_l = Σ_i α_{i→l} · v_i — mixing the RAW values, per Eq. 9.
        h = jnp.einsum("...s,...sd->...d", alpha, V.astype(F32))
        return h.astype(V.dtype)

    # ---- the write side ---------------------------------------------------- #
    def add(self, f_out: jax.Array) -> None:
        """Accumulate a module's output f_l(h_l) into the open block's sum b_n."""
        self.partial = f_out if self.partial is None else self.partial + f_out

    def end_layer(self) -> None:
        """Mark one decoder layer done; close the block every `block_size` layers."""
        self._layers_in_block += 1
        if self._layers_in_block >= self.block_size:
            self.close_block()

    def close_block(self) -> None:
        """Freeze the open block into a new source b_n. Idempotent when empty, so
        it is safe to call once more at the end to flush a trailing partial block
        (K3's 93 layers / 12 per block leave exactly such a partial block)."""
        if self.partial is not None:
            self.blocks.append(self.partial)
            self.partial = None
        self._layers_in_block = 0

    @property
    def num_sources(self) -> int:
        """How many b's a read would currently see (diagnostics / tests)."""
        return len(self.blocks) + (0 if self.partial is None else 1)
