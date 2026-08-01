"""
Attention Residuals (AttnRes) in JAX / Flax NNX.

TWO PAPERS, ONE METHOD
----------------------
  * "Attention Residuals" (arXiv:2603.15031) — the ORIGINAL. Method in §3.1
    (Full) and §3.2 (Block), ablations in §5.3. Equation numbers 2-6 below.
  * "Kimi K3" (arXiv:2607.24653) §2.2 — the DEPLOYMENT. Equation numbers 8-10.

The math is identical: K3 Eqs. 8/9 are the original's Eqs. 2/3/4, and K3's
Eq. 10 is its Eqs. 5/6. Both numberings are cited below (as "Eq. 2/8" etc.) so
either paper can be read alongside this file. What K3 actually changed:

  1. It uses ONLY Block AttnRes. The original offers both, and Full is the
     better model (§5.3: 1.737 vs 1.746 validation loss on the 16-layer
     ablation; scaling fits 1.865·C^-0.057 vs 1.870·C^-0.058). Full loses on
     systems grounds alone — O(Ld) memory AND cross-stage traffic under pipeline
     parallelism — and the original explicitly names it as the thing to return
     to "as future hardware alleviates memory capacity constraints".
  2. Much coarser blocks. See the BLOCK SIZING note below.
  3. Scale + infrastructure: the original validates on a scaling sweep plus one
     48B/3B-active Kimi Linear model; K3 runs it at 2.8T/104B-active over 93
     layers, and adds the production machinery (checkpointed AttnRes,
     cache-based pipeline communication, sequence-parallel prefill kernels,
     side-stream inter-block decode) in its §5.

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

FULL AttnRes (Eqs. 2-4 / 8-9)
----------------------------
Each layer l owns a learnable *pseudo-query* q_l = w_l ∈ R^d (a plain parameter
vector — there is no query projection, because the "sequence" being attended
over is the list of layers, not tokens). The keys and values are the same
objects, namely the layer outputs:

    k_i = v_i = { h_1            i = 0     (the token embedding)        Eq. 3/8
                { f_i(h_i)       1 ≤ i ≤ l-1  (the OUTPUT of layer i)

    φ(q, k)   = exp( qᵀ RMSNorm(k) )                                    Eq. 2/9
    α_{i→l}   = φ(q_l, k_i) / Σ_{j<l} φ(q_l, k_j)
    h_l       = Σ_{i<l} α_{i→l} · v_i                                   Eq. 4/9

Two details worth pausing on:
  * RMSNorm is applied to the KEY only. Without it, a layer whose outputs happen
    to have large magnitude would win every dot product and dominate the weights
    regardless of content; normalizing makes the scores about DIRECTION. The
    VALUE stays un-normalized (the mixture is over v_i, not RMSNorm(v_i)), so the
    actual magnitudes of the layer outputs are preserved.
  * v_i is the layer's OWN OUTPUT f_i(h_i) — its delta contribution — not a
    running sum. The running sums only appear in the block variant below.

BLOCK AttnRes (Eqs. 5-6 / 10) — what K3 actually uses
-----------------------------------------------------
Full AttnRes must keep every layer output alive: O(L·d) memory per token, and
cross-stage traffic under pipeline parallelism. Block AttnRes partitions the L
layers into N blocks of S = L/N layers and attends over BLOCK-level
representations:

    b_n     = Σ_{j ∈ B_n} f_j(h_j)      the sum of block n's layer outputs Eq. 5
    b_n^i   = the partial sum over the first i layers of block n
    b_0     = h_1                       the token embedding is always a source

    V = [b_0, b_1, …, b_{n-1}]              for the FIRST module of block n
    V = [b_0, b_1, …, b_{n-1}, b_n^{i-1}]   for later modules of block n  Eq. 6/10

with the same keys/weights as above. So a module sees: every FINISHED block as
its own separate source, plus one running partial sum for the block it is
currently inside — inter-block retrieval is selective, intra-block accumulation
is the ordinary uniform sum. Memory and cross-stage communication drop from
O(Ld) to O(Nd).

N interpolates between two degenerate cases, both worth knowing as sanity
checks: N = L recovers Full AttnRes exactly, and N = 1 reduces to STANDARD
residual connections with the embedding isolated as b_0. (In this implementation
`block_size` counts decoder LAYERS, so block_size = 1 stops one step short of
Full — a layer's token-mixer and channel-mixer outputs still land in the same
block sum. Reaching true Full would mean closing after every `add`.)

The final output layer then aggregates all N block representations — i.e. one
more read, with its own pseudo-query, over the closed blocks (see
`AttnResBus.close_block` + a final `AttnResQuery` in the model).

BLOCK SIZING — and a 2× trap when porting from the papers
----------------------------------------------------------
The original sweeps the block size S and finds S = 2, 4, 8 all land near 1.746
loss, degrading toward baseline at S = 16, 32. But its actual recommendation is
stated in terms of the block COUNT: "we fix the number of blocks to ≈8 for
infrastructure efficiency", since N is what sets cross-stage communication. K3
follows that rule — 93 layers into 8 blocks of 12, with a partial final block
and 9 sources once b_0 is counted.

The trap: the two papers COUNT LAYERS DIFFERENTLY.
  * The original counts MODULES. Its 48B run is "27 Transformer blocks (54
    layers)", and its pseudocode says outright: "block_size counts ATTN + MLP;
    each transformer layer has 2".
  * K3 counts ATTENTION LAYERS — its 93 = 69 KDA + 24 MLA, each paired with a
    LatentMoE. So K3's 12-layer block is 24 modules, ~3× coarser than anything
    in the original's sweep.
This module (and `KimiK3Config.attnres_block_size`) uses K3's convention:
`block_size` counts DECODER LAYERS, each contributing two modules to the sum.

WHY THIS COSTS ALMOST NOTHING
-----------------------------
Depth is small (L < 100), so the depth-attention is O(L²d) arithmetic against
the O(L·d²) of the layers themselves — and Block cuts that to O(N²d). K3 quotes
≈ 4% added training cost and ≈ 2% added inference cost.

DESIGN CHOICES, AND THE ABLATIONS BEHIND THEM (original §5.3)
--------------------------------------------------------------
Every one of these is a fork in the road that was measured. Losses below come
from the paper's Table 4 ablation model, where the PreNorm baseline scores
1.766, Full AttnRes 1.737, and Block AttnRes 1.746 — component ablations are
against Full unless noted.

  * softmax, not sigmoid (1.737 -> 1.741). The paper credits softmax's
    "competitive normalization, which forces sharper selection among sources" —
    depth-mixing benefits from the sources having to compete for one budget.
  * ONE shared query per module, not multi-head — measured on Block, 1.746 ->
    1.752 with H = 16. Per-head depth attention, letting different channel
    groups pick different source layers, actively HURTS: "when a layer's output
    is relevant, it is relevant as a whole."
  * A learned query, not an input-dependent one. Projecting the query from the
    hidden state is slightly better (1.731) but was rejected — it adds a d×d
    projection per layer and forces sequential memory access during decoding.
    Keeping w_l a free parameter is also what makes the two-phase batched
    schedule possible: attention weights for a whole block of layers can be
    computed before any of their outputs exist.
  * Content-dependent weighting is the whole point. Replacing q and k with
    learned input-INDEPENDENT scalars costs 1.749 vs 1.737; DenseFormer, which
    does exactly that, scores 1.767 — no gain over baseline at all.
  * RMSNorm on the keys matters MORE for Block than for Full (removing it: 1.743
    Full, 1.750 Block), because block sums accumulate over many layers and so
    develop larger magnitude differences than individual outputs do.

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

    Eq. 2/9 writes φ(q, k) = exp(qᵀ RMSNorm(k)) and specifies no gain vector; a
    per-channel gain here would in any case be redundant with the learnable
    pseudo-query q it is dotted against (the two multiply channel-wise before
    the sum, so q can absorb it). Keeping it gain-free also means the score is
    purely about the DIRECTION of each source, which is the stated purpose.

    Not optional: the original ablates it away and loses 1.737 -> 1.743 on Full
    AttnRes and 1.746 -> 1.750 on Block. Block suffers more because its sources
    are SUMS over many layers, so their magnitudes diverge further than
    individual layer outputs do — and an unnormalized dot product would then let
    the largest block win the softmax on magnitude rather than on content.
    """
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


class AttnResQuery(nnx.Module):
    """One module's learnable pseudo-query w_l (Eq. 3/8), plus the read it performs.

    ONE PER MODULE, NOT PER LAYER. Each token mixer AND each channel mixer owns
    its own query and reads the bus independently — they can, and do, attend to
    different sources. K3's Fig. 2 shows this by hanging an (α, w) pair off every
    KDA / Gated MLA / Stable LatentMoE box; the original's pseudocode makes it
    explicit by carrying separate `attn_res_proj`/`attn_res_norm` and
    `mlp_res_proj`/`mlp_res_norm` per transformer layer. (The original's prose
    saying "one RMSNorm and one pseudo-query vector w_l per layer" is consistent
    with this — "layer" there means module, which is why it counts its 27
    transformer blocks as 54 layers.)

    ZERO-INIT IS REQUIRED, not merely tidy. The original (§5) states it flatly:
    "Crucially, all pseudo-query vectors must be initialized to zero." Every
    score qᵀRMSNorm(k) is then 0, so the initial α is uniform and AttnRes starts
    life as an equal-weight average over the embedding and the preceding blocks —
    training only then sharpens it. Their stated reason is stability: starting
    from the equal-weight average "prevents training volatility, as we validated
    empirically". Any other init instead locks in an arbitrary depth mixture
    before the layer outputs mean anything.
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
        """Mark one decoder layer done; close the block every `block_size` layers.

        TIMING, vs the original's pseudocode. It closes the block at the START of
        the boundary layer — after that layer's token-mixer read, before the mixer
        runs — whereas this closes at the END of the preceding layer. The two are
        EQUIVALENT, not merely similar: in their ordering the boundary layer's
        read sees [b_0 … b_{n-1}] + partial where partial is already the complete
        block sum, i.e. the same set {b_0 … b_n} this ordering supplies as closed
        blocks. Only the bookkeeping differs, and closing at the end keeps
        `read` free of any is-this-a-boundary special case.
        """
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
