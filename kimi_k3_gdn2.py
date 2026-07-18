"""
Kimi K3 recreation — the top-level decoder-only language model, in JAX / Flax NNX.

This project is our FIRST ATTEMPT at recreating the architecture of Kimi K3
(Moonshot AI, July 2026 — https://www.kimi.com/blog/kimi-k3). K3's backbone is
the hybrid linear-attention transformer of "Kimi Linear: An Expressive,
Efficient Attention Architecture", extended with the K3 architectural features.
The K3 technical report ships with the model weights; until it lands, each
feature below is annotated against its best public source.

WHAT KIMI K3 IS, AND WHAT THIS MODEL IMPLEMENTS
-----------------------------------------------
A hybrid linear-attention transformer with depth-wise attention residuals.
Most layers use a cheap, O(L) linear-attention token mixer (K3/Kimi Linear:
"Kimi Delta Attention", KDA); a minority use ordinary softmax full attention
(Multi-head Latent Attention, MLA). The two interleave at a fixed **3:1 ratio**
— K3's architecture diagram shows the same 3× KDA : 1× MLA cell as Kimi Linear,
which found this recovers full-attention quality at a fraction of the KV-cache
and compute cost.

Feature-by-feature status of this recreation:

  • ATTENTION RESIDUALS (AttnRes) — implemented faithfully from the paper
    (arXiv:2603.15031, Kimi Team); the Block variant, K3's backbone. See below.
  • GATED MLA — implemented: a head-wise sigmoid output gate on the MLA layers
    (the Gated Attention lineage, arXiv:2505.06708; K3: "Gated MLA improves
    attention selectivity"). The MLA layers are NoPE — the linear-attention
    layers carry position implicitly through their recurrence, so the
    full-attention layers need no positional encoding at all.
    See multi_latent_attention/attention.py.
  • LatentMoE — implemented, and REQUIRED: every layer's channel mixer (FFN) is
    a DeepSeek-V3 / Moonlight-style MoE whose routed experts run in a shared
    low-rank latent with α-scaled expert count/top-k at iso-cost
    (arXiv:2601.18089 — the design K3's "Stable LatentMoE" builds on).
    See multi_latent_attention/latent_moe.py.
  • KIMI DELTA ATTENTION — deliberately SUBSTITUTED with **Gated DeltaNet-2**
    ("Decoupling Erase and Write in Linear Attention", arXiv:2605.22791). Both
    are gated-delta-rule linear attentions with fine-grained (channel-wise)
    gating; GDN-2's twist is a separate erase gate `b` and write gate `w`
    instead of the single `beta` that KDA/GDN share. This is this recreation's
    one intentional departure from K3. See gated_deltanet_2/layer.py.
  • NOT (yet) recreated, pending the K3 technical report: the "Stable" additions
    and Quantile Balancing of K3's Stable LatentMoE, the Per-Head Muon
    optimizer, and K3's multimodal / 1M-context / MXFP4 training machinery.

BLOCK STRUCTURE — ATTENTION RESIDUALS (arXiv:2603.15031; K3's backbone)
-----------------------------------------------------------------------
The standard pre-norm residual stream accumulates every sub-layer output with
fixed unit weights, so hidden-state magnitude grows O(depth) and early layers
get diluted. Attention Residuals replaces that fixed accumulation with SOFTMAX
ATTENTION OVER DEPTH: the input to each sub-layer is a learned, input-dependent
mixture of preceding representations,

    h_l = Σ_i softmax_i( w_l · RMSNorm(v_i) ) · v_i

where the v_i are depth-wise sources and w_l is a per-sub-layer learned
pseudo-query (a single d_model vector — the mechanism's whole parameter cost).

We implement the paper's scalable BLOCK variant (its Fig. 2 pseudocode), the
form K3's diagram labels "Block Attention Residuals": layers are grouped into
blocks; inside a block, sub-layer outputs accumulate into a plain partial sum;
the depth-attention sources are [token embedding, completed block sums, current
partial sum]. With AttnRes the per-sub-layer update becomes

    h       = AttnRes(blocks, partial)        # depth-wise softmax mixture
    out     = SubLayer(RMSNorm(h))            # GDN-2 / MLA / MoE, pre-norm as before
    partial = partial + out                   # intra-block accumulation
    (at each block boundary: blocks.append(partial); partial = None)

and a final AttnRes op aggregates all sources before the output norm/LM head.
AttnRes mixes DEPTH, never positions, so the streaming caches are untouched.
Set attn_res=False to recover the plain pre-norm residual stream.

MODEL = Embed -> [DecoderLayer] * n_layers (AttnRes backbone) -> RMSNorm -> LM head.

TWO FORWARD MODES
-----------------
  • Training / full sequence:  model(input_ids)  — parallel, GDN-2 via its chunkwise
    core, MLA via a full causal-attention matrix.
  • Streaming / inference:     model.step(ids, caches) and model.generate(...)  —
    reuses per-layer state across calls so each new token is O(1) work for the GDN-2
    layers (fixed-size recurrent state) and O(context) for the few MLA layers (growing
    latent cache). See GatedDeltaNet2.step / GroupedQueryLatentAttention.step.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Reuse the building blocks already implemented and verified in this repo.
from gated_deltanet_2.layer import GatedDeltaNet2, GDN2Cache, RMSNorm
from multi_latent_attention.attention import GroupedQueryLatentAttention, MLACache
from multi_latent_attention.latent_moe import LatentMoE

# App. D.5: Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale = gain² =
# 2^{-5}) for the embedding and LM head, replacing Flax NNX's defaults. The (small)
# embedding scale this produces is fine — RMSNorm renormalizes the residual stream.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


# --------------------------------------------------------------------------- #
#  Configuration
#
#  Defaults are deliberately TINY so the whole model trains on a laptop CPU.
#  Reference numbers quoted in the comments come from the Kimi Linear paper's
#  1.3B / 48B-A3B configs (K3's own hyperparameters await its technical report);
#  only the *ratios* and structure matter for understanding — scale up by
#  raising the dims/layers.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class KimiK3Config:
    vocab_size: int = 256  # Kimi Linear: 160k; tiny here (byte-level demo)
    d_model: int = 256  # model width  (Kimi Linear 1.3B: 2048)
    n_layers: int = 8  # depth        (Kimi Linear 1.3B: 27)

    # --- Hybrid schedule: which layers are FULL attention (MLA) vs linear (GDN-2) ---
    # full_attn_period = 4 places one MLA layer every 4th layer (indices 3, 7, ...),
    # i.e. a 3:1 linear:full ratio — Kimi Linear's hybrid recipe (its Sec. 3.2),
    # kept by K3 (the K3 architecture diagram shows the same 3×/1× cell).
    full_attn_period: int = 4

    # --- Block Attention Residuals (arXiv:2603.15031) — K3's backbone ---
    # attn_res_layers_per_block counts TRANSFORMER layers (attn+MoE pairs) per
    # block. The AttnRes paper's 48B run used 3 (27 layers -> 9 blocks) and
    # recommends ~8 blocks overall; 2 gives this tiny 8-layer demo 4 blocks +
    # the embedding = 5 depth-wise sources. attn_res=False restores the plain
    # pre-norm residual stream (bitwise-identical math to a no-AttnRes model).
    attn_res: bool = True
    attn_res_layers_per_block: int = 2

    # --- GDN-2 token mixer (this recreation's KDA stand-in) — see gated_deltanet_2/layer.py ---
    gdn_num_heads: int = 4  # H key/query heads   (Kimi Linear 1.3B: 16)
    gdn_head_k_dim: int = 64  # d_k                 (Kimi Linear: 128)
    gdn_head_v_dim: int = 64  # d_v                 (Kimi Linear: 128)
    gdn_num_v_heads: int | None = None  # H_v for GQA value heads; None -> = num_heads
    gdn_chunk_size: int = 64  # chunkwise block size C (GDN-2 paper App. C.2: 64).
    #   NOTE: the GDN-2 chunkwise core requires every fed sequence length to be a
    #   multiple of this C (it reshapes L into L/C chunks). Keep seq_len % C == 0.
    gdn_conv_size: int = 4  # short-conv kernel width
    gdn_expanded_erase: bool = True  # erase gate in [0,2] (neg-eigenvalue variant)
    gdn_core: str = "centered"  # which GDN-2 chunkwise core computes each head (paper: "faithful")

    # --- MLA full-attention layers (NoPE) — see multi_latent_attention/attention.py ---
    mla_num_q_heads: int = 8  # query heads
    mla_num_kv_heads: int = 2  # KV/latent heads (GQA); q_heads must be a multiple
    mla_head_dim: int = 64  # per-head latent (rank) width
    # Kimi K3 "Gated MLA": head-wise sigmoid(W_g x) ⊙ o output gate on the
    # attention output, before the absorbed out-projection (the Gated Attention
    # lineage, arXiv:2505.06708). False = the plain (pre-K3) MLA.
    mla_gated: bool = True
    # Declared context cap: checked against the training seq_len and used as the
    # default size of the preallocated MLA latent cache in init_cache/generate.
    # (The MLA causal mask itself is built on the fly from the actual length.)
    max_seq_len: int = 512

    # --- Channel mixer (FFN): LatentMoE (arXiv:2601.18089) ---
    # The routed experts run in a shared low-rank latent of width moe_d_latent
    # (see multi_latent_attention/latent_moe.py). Following the LatentMoE
    # paper's iso-cost recipe at compression α = d_model/moe_d_latent = 4, BOTH
    # the expert count and top-k scale by α versus a full-width MoE (here:
    # 2-of-8 full-width -> 8-of-32 latent; K3 itself: K2's 8-of-384 -> K3's
    # 16-of-896): per-expert cost shrinks by α, so the total and active expert
    # cost is unchanged while accuracy improves.
    moe_d_latent: int = 64  # shared expert-latent width ℓ (= d_model/4)
    moe_d_ff: int = 512  # per-expert hidden width, inside the latent (shared expert: full-width)
    moe_n_routed: int = 32  # number of routed experts E (α-scaled; K3: 896)
    moe_n_shared: int = 1  # always-on shared experts (always full-width)
    moe_top_k: int = 8  # experts activated per token (α-scaled; K3: 16)
    # Group-limited routing (DeepSeek-V3 / Kimi K2 "node-limited"): experts split
    # into moe_n_groups groups; each token draws its top-k only from its
    # moe_topk_groups best groups (at scale: bounds all-to-all traffic).
    # Constraints: moe_n_routed % moe_n_groups == 0 and
    # moe_top_k <= moe_topk_groups * group size. Set moe_n_groups = 1 to disable.
    moe_n_groups: int = 4
    moe_topk_groups: int = 2

    rms_eps: float = 1e-5

    # --- Mixed precision ---
    # Matmul (compute) dtype for the projection Linears + MoE expert GEMMs. Master
    # weights are ALWAYS stored fp32 (param_dtype), and the numerically sensitive
    # parts stay fp32 regardless: the GDN-2 chunkwise core, RMSNorm, the router
    # scores, the AttnRes depth-softmax, and the loss. Set "bfloat16" on an H200;
    # "float32" disables mixed precision. Read from YAML as a string; use
    # `.cdtype` for the resolved dtype.
    compute_dtype: str = "float32"

    @property
    def cdtype(self) -> jnp.dtype:
        return jnp.dtype(self.compute_dtype)


# --------------------------------------------------------------------------- #
#  Attention Residuals operation (arXiv:2603.15031, Eq. 2-4 / Fig. 2).
#
#  One per SUB-layer (each attn and each MoE gets its own), plus one final
#  aggregator before the LM head. Parameters per op: a single pseudo-query
#  w ∈ R^d and an RMSNorm — the paper's entire per-layer cost.
# --------------------------------------------------------------------------- #
class AttnResOp(nnx.Module):
    """h = Σ_i softmax_i( w · RMSNorm(v_i) ) · v_i over depth-wise sources v_i.

    The pseudo-query MUST be zero-initialized: the initial depth-attention is
    then uniform over sources, reducing AttnRes to an equal-weight average at
    the start of training (the paper §5 found nonzero init causes volatility).
    The RMSNorm on the KEYS stops large-magnitude sources (e.g. block sums that
    accumulated over many layers) from monopolizing the softmax; the VALUES are
    mixed un-normalized (Eq. 3: k_i = norm'd, v_i = raw layer/block outputs).
    """

    def __init__(self, d_model: int, *, eps: float = 1e-5, rngs: nnx.Rngs):
        self.query = nnx.Param(jnp.zeros((d_model,), jnp.float32))  # w_l (§5: zero)
        self.norm = RMSNorm(d_model, eps=eps, rngs=rngs)

    def __call__(
        self, blocks: list[jax.Array], partial: jax.Array | None
    ) -> jax.Array:
        """blocks: completed block reps [b_0(embedding), b_1, ...], each [B, L, d];
        partial: the intra-block partial sum b_n^i, or None at a block start
        (Eq. 6: the first layer of a block sees only the completed blocks).
        Depth-wise only — every position attends over its own stack of sources,
        so this is position-independent and needs no sequence cache."""
        sources = blocks + ([partial] if partial is not None else [])
        V = jnp.stack(sources)  # [N, B, L, d]
        K = self.norm(V.astype(jnp.float32))  # RMSNorm'ed keys, fp32
        logits = jnp.einsum("d,nbld->nbl", self.query[...], K)
        alpha = jax.nn.softmax(logits, axis=0)  # depth-wise attention weights
        return jnp.einsum("nbl,nbld->bld", alpha, V.astype(jnp.float32))


# --------------------------------------------------------------------------- #
#  One decoder block: token mixer + channel mixer, threaded through the Block
#  AttnRes backbone (or the plain pre-norm residual stream when attn_res=False).
#
#  The ONLY thing that varies across layers is the token mixer: GDN-2 (linear) on
#  most layers, MLA (full attention) on the 3:1 schedule. The channel mixer is a
#  LatentMoE on every layer — as in K3 (and Kimi Linear before it), the hybrid
#  is in the *attention*, not the FFN.
# --------------------------------------------------------------------------- #
class DecoderLayer(nnx.Module):
    def __init__(self, cfg: KimiK3Config, layer_idx: int, *, rngs: nnx.Rngs):
        # 3:1 schedule: this layer is full-attention iff it is the last of its period.
        self.is_full_attn = (layer_idx + 1) % cfg.full_attn_period == 0

        # Block AttnRes boundary: this layer OPENS a new block — the previous
        # block's partial sum is sealed into `blocks` before its token mixer runs.
        # Layer 0 opens the first block trivially (partial is still None there).
        self.attn_res = cfg.attn_res
        self.starts_new_block = layer_idx % cfg.attn_res_layers_per_block == 0

        if cfg.attn_res:
            # One AttnRes op per sub-layer (paper Fig. 2: the pre-attn and the
            # pre-MLP mixes each have their own pseudo-query and key-norm).
            self.attn_res_mixer = AttnResOp(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)
            self.mlp_res_mixer = AttnResOp(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)

        # Pre-norm before the token mixer (Fig. 2). RMSNorm reused from the GDN-2 layer.
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)

        if self.is_full_attn:
            # Full attention: NoPE Multi-head Latent Attention (absorbed/GQA
            # form), with the K3 sigmoid output gate when mla_gated is set.
            self.token_mixer = GroupedQueryLatentAttention(
                embed_dim=cfg.d_model,
                num_q_heads=cfg.mla_num_q_heads,
                num_kv_heads=cfg.mla_num_kv_heads,
                head_dim=cfg.mla_head_dim,
                compute_dtype=cfg.cdtype,
                gated=cfg.mla_gated,
                rngs=rngs,
            )
        else:
            # Linear attention: Gated DeltaNet-2 (this recreation's stand-in
            # for K3's Kimi Delta Attention — see the module docstring).
            self.token_mixer = GatedDeltaNet2(
                d_model=cfg.d_model,
                num_heads=cfg.gdn_num_heads,
                head_k_dim=cfg.gdn_head_k_dim,
                head_v_dim=cfg.gdn_head_v_dim,
                num_v_heads=cfg.gdn_num_v_heads,
                chunk_size=cfg.gdn_chunk_size,
                conv_size=cfg.gdn_conv_size,
                expanded_erase=cfg.gdn_expanded_erase,
                compute_dtype=cfg.cdtype,
                core=cfg.gdn_core,
                rngs=rngs,
            )

        # Pre-norm before the channel mixer.
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)

        # Channel mixer: LatentMoE (routed experts in a shared low-rank latent,
        # arXiv:2601.18089). Shares GroupedGemmMoE's routing machinery and aux
        # contract, so the training loop (aux_loss, update_router_bias) is
        # unchanged from the full-width MoE this repo used before.
        self.channel_mixer = LatentMoE(
            d_model=cfg.d_model,
            d_latent=cfg.moe_d_latent,
            d_ff=cfg.moe_d_ff,
            n_routed=cfg.moe_n_routed,
            n_shared=cfg.moe_n_shared,
            top_k=cfg.moe_top_k,
            n_groups=cfg.moe_n_groups,
            topk_groups=cfg.moe_topk_groups,
            compute_dtype=cfg.cdtype,
            rngs=rngs,
        )

    # -- shared per-sub-layer plumbing ------------------------------------- #
    def _mix_in(
        self, mixer_name: str, blocks: list[jax.Array], partial: jax.Array | None
    ) -> jax.Array:
        """Sub-layer input: the AttnRes depth-attention over sources, or — when
        attn_res=False — their plain sum, which reconstructs the pre-norm
        residual stream exactly (x = embedding + Σ all sub-layer outputs =
        Σ completed blocks + partial)."""
        if self.attn_res:
            mixer: AttnResOp = getattr(self, mixer_name)
            return mixer(blocks, partial)
        acc = blocks[0]
        for b in blocks[1:]:
            acc = acc + b
        return acc if partial is None else acc + partial

    def _seal_block(
        self, blocks: list[jax.Array], partial: jax.Array | None
    ) -> tuple[list[jax.Array], jax.Array | None]:
        """At a block boundary, seal the finished block AFTER its last partial
        sum fed the pre-attn depth-attention (the paper's Eq. 6 ordering)."""
        if self.starts_new_block and partial is not None:
            return blocks + [partial], None
        return blocks, partial

    def __call__(
        self, blocks: list[jax.Array], partial: jax.Array | None
    ) -> tuple[list[jax.Array], jax.Array, dict[str, jax.Array]]:
        """One transformer layer on the Block AttnRes backbone (paper Fig. 2).

        blocks:  [b_0(embedding), b_1, ...], completed block reps, each [B, L, d].
        partial: intra-block partial sum entering this layer (None at a block start).
        Returns (blocks, partial, aux); `aux` carries the MoE load-balancing
        diagnostics the training loop needs (aux loss + per-expert token counts
        for the router-bias update), unchanged from before.
        """
        # --- token mixing (AttnRes input, pre-norm sub-layer) ---
        h = self._mix_in("attn_res_mixer", blocks, partial)
        blocks, partial = self._seal_block(blocks, partial)

        out = self.token_mixer(self.norm1(h))
        partial = out if partial is None else partial + out

        # --- channel mixing (AttnRes input, pre-norm sub-layer) ---
        h = self._mix_in("mlp_res_mixer", blocks, partial)
        m, aux = self.channel_mixer(self.norm2(h))
        partial = partial + m
        return blocks, partial, aux

    def init_cache(self, batch_size: int, max_len: int, dtype=jnp.float32):
        """Per-layer streaming cache: a GDN2Cache (linear layer) or MLACache (MLA).
        AttnRes needs NO cache — it mixes depth, not positions."""
        return self.token_mixer.init_cache(batch_size, max_len, dtype)

    def step(
        self,
        blocks: list[jax.Array],
        partial: jax.Array | None,
        cache: GDN2Cache | MLACache,
    ) -> tuple[list[jax.Array], jax.Array, GDN2Cache | MLACache]:
        """Streaming forward for one layer: identical depth-wise math to
        __call__, with the token mixer running against its cache. Only the token
        mixer is stateful; the MoE is position-wise and AttnRes is depth-wise,
        so neither needs a cache."""
        h = self._mix_in("attn_res_mixer", blocks, partial)
        blocks, partial = self._seal_block(blocks, partial)

        hn = self.norm1(h)
        if isinstance(cache, GDN2Cache) and isinstance(
            self.token_mixer, GatedDeltaNet2
        ):
            # GDN-2: fixed-size recurrent state (O(1) per token).
            out, new_cache = self.token_mixer.step(hn, cache)
        elif isinstance(cache, MLACache) and isinstance(
            self.token_mixer, GroupedQueryLatentAttention
        ):
            # MLA: growing latent cache (O(context) per token).
            out, new_cache = self.token_mixer.step(hn, cache)
        else:
            raise ValueError(
                f"Cache type {type(cache)} does not match token mixer {type(self.token_mixer)}"
            )
        partial = out if partial is None else partial + out

        h = self._mix_in("mlp_res_mixer", blocks, partial)
        m, _ = self.channel_mixer(self.norm2(h))
        partial = partial + m
        return blocks, partial, new_cache


# --------------------------------------------------------------------------- #
#  The full model.
# --------------------------------------------------------------------------- #
class KimiK3(nnx.Module):
    """Decoder-only Kimi K3-style LM: Block AttnRes backbone over a 3:1
    GDN-2 (KDA stand-in) : gated-MLA hybrid, with a LatentMoE channel mixer
    on every layer."""

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        # Token embedding table.
        self.embed = nnx.Embed(
            cfg.vocab_size, cfg.d_model, embedding_init=_XAVIER, rngs=rngs
        )

        # Stack of decoder blocks. NOTE: in Flax NNX a plain Python list of submodules
        # is not tracked as state — it must be wrapped in nnx.List(...).
        self.layers = nnx.List(
            [DecoderLayer(cfg, i, rngs=rngs) for i in range(cfg.n_layers)]
        )

        # Final AttnRes aggregation over all block representations (paper §3.2:
        # "the final output layer aggregates all N block representations"),
        # then the final pre-head norm + untied LM head (Moonlight/DeepSeek do
        # not tie weights; to tie, drop lm_head and use
        # `x @ self.embed.embedding.value.T` instead).
        if cfg.attn_res:
            self.final_res_mixer = AttnResOp(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)
        self.lm_head = nnx.Linear(
            cfg.d_model,
            cfg.vocab_size,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=cfg.cdtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, input_ids: jax.Array) -> tuple[jax.Array, dict[str, ArrayLike]]:
        """input_ids: int[B, L] -> (logits[B, L, vocab], aux).

        aux is ALWAYS returned (callers that don't need it just unpack `logits, _ =`):
            aux = {"aux_loss":   scalar, the MoE load-balancing loss summed over layers,
                   "group_sizes": int[n_layers, E], per-expert token counts per layer}.
        The training loop uses aux_loss (added to the CE loss) and group_sizes (to nudge
        each MoE layer's router bias); eval/inference paths simply ignore it.
        """
        aux_loss: ArrayLike = 0.0
        group_sizes: list[
            ArrayLike
        ] = []  # one [E] vector per MoE layer, in layer order

        # Block AttnRes state: blocks[0] is ALWAYS the token embedding (the
        # paper's b_0), and the intra-block partial sum starts empty.
        emb = self.embed(input_ids)  # [B, L, d_model]
        blocks: list[jax.Array] = [emb]
        partial: jax.Array | None = None

        for layer in self.layers:
            blocks, partial, aux = layer(blocks, partial)

            aux_loss = aux_loss + aux["aux_loss"]
            group_sizes.append(aux["group_sizes"])

        x = self._final_mix(blocks, partial)
        x = self.norm_f(x)
        # Upcast logits to fp32 for a numerically stable softmax/cross-entropy under
        # bf16 compute (the lm_head matmul itself still runs in cfg.compute_dtype).
        logits = self.lm_head(x).astype(jnp.float32)  # [B, L, vocab]

        return logits, {"aux_loss": aux_loss, "group_sizes": jnp.stack(group_sizes)}

    def _final_mix(
        self, blocks: list[jax.Array], partial: jax.Array | None
    ) -> jax.Array:
        """Pre-head aggregation: the final AttnRes op over every depth-wise
        source, or their plain sum (= the classic residual stream) without it."""
        if self.cfg.attn_res:
            return self.final_res_mixer(blocks, partial)
        acc = blocks[0]
        for b in blocks[1:]:
            acc = acc + b
        return acc if partial is None else acc + partial

    # ----------------------------------------------------------------------- #
    #  Streaming / inference.  Each layer carries its own cache (GDN-2: fixed-size
    #  recurrent state + conv state; MLA: growing latent cache).  Reusing them makes
    #  generation O(1) per token for the linear layers instead of re-reading history.
    #  AttnRes carries NOTHING across steps — depth-wise mixing is recomputed for
    #  each new position from that position's own layer outputs.
    # ----------------------------------------------------------------------- #
    def init_cache(
        self, batch_size: int, max_len: int | None = None, dtype=jnp.float32
    ) -> list:
        """Streaming caches for every layer. `max_len` (default cfg.max_seq_len) sizes
        the MLA latent buffers; GDN-2 layers ignore it (their state is fixed-size)."""
        max_len = max_len or self.cfg.max_seq_len
        return [layer.init_cache(batch_size, max_len, dtype) for layer in self.layers]

    def step(self, input_ids: jax.Array, caches: list) -> tuple[jax.Array, list]:
        """One streaming step. input_ids: int[B, L] (L = prompt length on prefill, or
        1 per decoded token). Returns (logits[B, L, vocab], new_caches)."""
        new_caches = []

        emb = self.embed(input_ids)
        blocks: list[jax.Array] = [emb]
        partial: jax.Array | None = None

        for layer, cache in zip(self.layers, caches):
            blocks, partial, new_cache = layer.step(blocks, partial, cache)
            new_caches.append(new_cache)

        x = self._final_mix(blocks, partial)
        x = self.norm_f(x)
        return self.lm_head(x).astype(jnp.float32), new_caches

    def generate(
        self, prompt_ids: jax.Array, max_new_tokens: int, max_len: int | None = None
    ) -> jax.Array:
        """Greedy autoregressive decode that REUSES each layer's state across steps.
        prompt_ids: int[B, P]. Returns the continuation int[B, max_new_tokens].

        Prefill consumes the whole prompt in one step (filling every layer's cache) —
        the GDN-2 layers push all whole chunks of the prompt through their PARALLEL
        chunkwise core and only the ragged tail through the recurrence, so prefill
        cost scales with P/chunk_size sequential steps, not P. Each decode step then
        feeds back ONE token and carries the caches forward — the GDN-2 layers via
        their fixed-size recurrent state, the MLA layers via the growing latent
        cache. The decode loop runs through `_decode_step`, a module-level nnx.jit
        function: it compiles once per (batch size, cache length) and every further
        token — across generate() calls too — reuses the trace."""
        B, P = prompt_ids.shape
        # Default the cache length to the config's declared context cap when the
        # request fits inside it: a FIXED cache shape lets _decode_step reuse its
        # compiled trace across generate() calls with different prompt lengths
        # (e.g. a chat loop) instead of recompiling for every P + max_new_tokens.
        max_len = max_len or max(self.cfg.max_seq_len, P + max_new_tokens)

        caches = self.init_cache(B, max_len)
        logits, caches = self.step(prompt_ids, caches)  # prefill the prompt
        next_tok = jnp.argmax(logits[:, -1:], axis=-1)  # [B, 1] greedy
        outs = [next_tok]

        for _ in range(max_new_tokens - 1):
            next_tok, caches = _decode_step(self, next_tok, caches)
            outs.append(next_tok)

        return jnp.concatenate(outs, axis=1)  # [B, max_new_tokens]


# --------------------------------------------------------------------------- #
#  Jitted greedy decode step, shared by every generate() call.
#
#  During decoding everything is shape-constant — the weights, the fixed-size
#  GDN-2 states, the preallocated MLA latent buffers (position is a TRACED int32,
#  so advancing it never retraces), and L=1 — so this compiles ONCE per (batch
#  size, cache length) and each further token replays the compiled trace.
#  Module-level on purpose: nnx.jit keys its compilation cache on the function
#  object, so a wrapper created inside generate() would recompile every call.
# --------------------------------------------------------------------------- #
@nnx.jit
def _decode_step(
    model: KimiK3, tok: jax.Array, caches: list
) -> tuple[jax.Array, list]:
    """One greedy decode step: tok int[B, 1] -> (next greedy token int[B, 1], caches)."""
    logits, caches = model.step(tok, caches)
    return jnp.argmax(logits[:, -1:], axis=-1), caches


def count_params(model: nnx.Module) -> int:
    """Total number of trainable parameters (sum of nnx.Param leaf sizes)."""
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))))
