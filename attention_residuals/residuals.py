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
more read, with its own pseudo-query, over the closed blocks plus the trailing
partial one (see `KimiK3._head`, which owns an extra `AttnResReader`).

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

HOW THIS FILE IS ORGANIZED
--------------------------
It follows the original's Fig. 2 pseudocode line for line:

  * `block_attn_res(blocks, partial_block, proj, norm)` is the paper's function
    of the same name and signature — stack, norm, score, softmax, mix.
  * `AttnResReader` bundles the (proj, norm) pair one module owns, i.e. the
    paper's `attn_res_proj`/`attn_res_norm` and `mlp_res_proj`/`mlp_res_norm`.
  * The state is the paper's pair `(blocks, hidden_states)`, threaded EXPLICITLY
    through each layer's forward and returned as `(blocks, partial_block)` —
    there is no mutable container. `DecoderLayer.__call__` in kimi_k3_gdn2.py
    mirrors the pseudocode's `forward` statement for statement, including the
    block-boundary check sitting BETWEEN the token mixer's read and the mixer
    itself.

Two consequences of that ordering worth internalizing:
  * `blocks` starts EMPTY. b_0 = h_1 enters as the very first boundary append,
    because layer 0 satisfies `layer_idx % block_size == 0` and its incoming
    partial is the token embedding. That is the whole mechanism by which "the
    token embedding is always included as a source" holds.
  * `partial_block` is never None at a read. It is set to None only between the
    boundary append and the token mixer's output, and both reads happen outside
    that window — which is why `block_attn_res` needs no empty-partial case.

STREAMING
---------
AttnRes attends across LAYERS, not across tokens, so all of its state lives
inside a single forward pass. There is no cache to carry between decode steps —
which is why `blocks`/`partial_block` are ordinary values threaded through the
call and not part of GDN2Cache / MLACache.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

F32 = jnp.float32


class RMSNorm(nnx.Module):
    """The `norm: RMSNorm` argument of Fig. 2 — applied to the KEYS inside φ.

    Defined locally rather than imported so this package stands alone as a
    readable transcription of the paper.

    It carries a LEARNABLE GAIN, because the pseudocode types it as an RMSNorm
    module and §5 counts "one RMSNorm and one pseudo-query vector per layer"
    among the parameters AttnRes adds. Note the gain is mathematically
    absorbable: g ⊙ RMSNorm(k) dotted with w equals RMSNorm(k) dotted with
    g ⊙ w, and there is only ONE query per reader, so the gain adds no
    expressive power — it changes the parameterization (and hence the optimizer
    geometry), not the function class.

    Why it is here at all: without it, a source whose magnitude happens to be
    large wins every dot product regardless of content. Ablating it costs
    1.737 -> 1.743 on Full AttnRes and 1.746 -> 1.750 on Block; Block suffers
    more because its sources are SUMS over many layers, whose magnitudes diverge
    further than individual layer outputs do.

    EXPECT ZERO GRADIENT ON THE GAIN AT STEP 0 — this is not a bug, it is the two
    paper-mandated choices interacting. The gain reaches the loss only through
    `logits = einsum(w, norm(V))`, so ∂logits/∂gain = w ⊙ k̂; with the required
    zero-init on w, that is identically zero. The gain starts moving as soon as
    w does (measured here: all gains zero at init, all but the first receiving
    gradient two steps later). It is another sign the gain is doing no work the
    query could not do — see the absorption argument above.
    """

    def __init__(self, dim: int, *, eps: float = 1e-6):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,), F32))

    def __call__(self, x: jax.Array) -> jax.Array:
        xf = x.astype(F32)
        rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return xf * rms * self.weight[...]


def block_attn_res(
    blocks: list[jax.Array],
    partial_block: jax.Array,
    proj: nnx.Linear,
    norm: RMSNorm,
) -> jax.Array:
    """Inter-block attention: attend over block reps + partial sum. (Fig. 2)

    Args:
      blocks:        N tensors of shape [B, T, D] — completed block
                     representations for each previous block.
      partial_block: [B, T, D] — the intra-block partial sum b_n^i.
      proj:          the pseudo-query w_l, stored as a d->1 Linear so it can be
                     read off as `proj.kernel.squeeze()` (the paper stores it the
                     same way, as `proj.weight.squeeze()`).
      norm:          RMSNorm applied to the keys.

    Returns h_l: [B, T, D] — Eq. 4/9's Σ_i α_{i→l} · v_i.

    The value matrix is exactly Eq. 6/10: the closed blocks, plus the open
    block's running partial sum as one extra source. Keys and values are the
    SAME tensors (Eq. 3/8), the difference being that only the key side is
    normalized — so the scores compare directions while the mixture preserves
    the sources' true magnitudes.

    `blocks` may be EMPTY, which happens exactly once: the very first module of
    the network, before layer 0's boundary has closed the embedding into b_0.
    The stack is then length 1 and the softmax over it is identically 1, so this
    returns `partial_block` unchanged — correct (with one source there is nothing
    to select) and requiring no special case. Its one visible consequence is that
    that reader's query and gain are structurally unused and keep an exactly zero
    gradient forever; every other reader trains normally.
    """
    V = jnp.stack([*blocks, partial_block])  # [N+1, B, T, D]
    K = norm(V)  # keys only — values stay raw

    # φ(q, k) = exp(qᵀ RMSNorm(k)), normalized over the N+1 sources. softmax
    # supplies both the exp and the denominator of Eq. 2/9 in one stable step
    # (it subtracts the max, which cancels in the ratio). fp32 throughout: the
    # source count is tiny but the exponential is not forgiving in bf16.
    logits = jnp.einsum("d, n b t d -> n b t", proj.kernel[...].squeeze(-1).astype(F32), K)
    h = jnp.einsum("n b t, n b t d -> b t d", jax.nn.softmax(logits, axis=0), V.astype(F32))
    return h.astype(partial_block.dtype)


class AttnResReader(nnx.Module):
    """The (proj, norm) pair belonging to ONE module — Fig. 2's `attn_res_proj` /
    `attn_res_norm` (and the `mlp_res_*` twin).

    ONE PER MODULE, NOT PER LAYER. Each token mixer AND each channel mixer owns
    its own reader and attends independently — they can, and do, land on
    different sources. K3's Fig. 2 shows this by hanging an (α, w) pair off every
    KDA / Gated MLA / Stable LatentMoE box. (The original's prose saying "one
    RMSNorm and one pseudo-query vector w_l per layer" agrees: "layer" there
    means module, which is why it counts 27 transformer blocks as 54 layers.)

    ZERO-INIT IS REQUIRED, not merely tidy. The original (§5) states it flatly:
    "Crucially, all pseudo-query vectors must be initialized to zero." Every
    score qᵀRMSNorm(k) is then 0, so the initial α is uniform and AttnRes starts
    life as an equal-weight average over the embedding and the preceding blocks —
    training only then sharpens it. Their stated reason is stability: starting
    from the equal-weight average "prevents training volatility, as we validated
    empirically". Any other init instead locks in an arbitrary depth mixture
    before the layer outputs mean anything.
    """

    def __init__(self, d_model: int, *, eps: float = 1e-6, rngs: nnx.Rngs):
        # w_l as a d->1 projection, matching the paper's storage. `rngs` is
        # accepted for call-site parity with the other nnx modules but never
        # consumed: the kernel init is deterministically zero.
        self.proj = nnx.Linear(
            d_model,
            1,
            use_bias=False,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.norm = RMSNorm(d_model, eps=eps)

    def __call__(
        self, blocks: list[jax.Array], partial_block: jax.Array
    ) -> jax.Array:
        return block_attn_res(blocks, partial_block, self.proj, self.norm)

