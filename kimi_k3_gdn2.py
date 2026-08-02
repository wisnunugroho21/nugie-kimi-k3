"""
Kimi K3 (GDN-2 variant) — the top-level decoder-only language model, in JAX /
Flax NNX. ANNOTATED against "Kimi K3: Open Frontier Intelligence"
(arXiv:2607.24653), §2 Model Architecture.

WHAT KIMI K3 IS (paper §2, Fig. 2)
----------------------------------
K3's stated design principle is to scale INFORMATION FLOW along three axes, and
each axis gets one architectural mechanism:

  SEQUENCE (§2.1)  Hybrid Attention. Three linear-attention layers followed by
                   one Gated MLA layer per block — a 3:1 ratio — plus one extra
                   Gated MLA at the very end of the backbone, so the final layer
                   always performs global attention. Linear layers carry position
                   implicitly through their recurrence, so the MLA layers are
                   NoPE and do pure global content matching.
  DEPTH (§2.2)     Attention Residuals. Each module ATTENDS over the token
                   embedding and preceding block representations instead of
                   inheriting one accumulated residual stream. See
                   attention_residuals/residuals.py — this is the change that
                   most alters the shape of the model code.
  WIDTH (§2.3)     Stable LatentMoE. Routed experts live in a latent of width
                   ℓ = 0.5·d so a token can activate many of them cheaply; shared
                   experts stay full width. Plus the three stabilizers: RMSNorm
                   before the up-projection, SiTU-GLU, Quantile Balancing.
                   See multi_latent_attention/moe.py.

THIS FILE'S ONE DELIBERATE SUBSTITUTION
---------------------------------------
K3's linear-attention layer is **Kimi Delta Attention** (KDA). We keep this
project's **Gated DeltaNet-2** ("Decoupling Erase and Write in Linear Attention",
arXiv:2605.22791) in its place. Both are gated-delta-rule linear attentions with
channel-wise decay; the difference is that KDA writes with a single scalar β_t
(K3 Eq. 1) whereas GDN-2 decouples it into a per-channel erase gate b and write
gate w. Everything else of K3 is kept as in the paper — and the two K3 changes
that are orthogonal to the β-vs-(b,w) choice are adopted into the GDN-2 layer:

  * LOWER-BOUNDED DECAY (K3 Eq. 5): g = g_min·Sigmoid(exp(A)z), g_min = -5,
    replacing GDN-2 / Kimi Linear's unbounded g = -exp(A)·softplus(z).
  * FULL-RANK OUTPUT GATE (K3 Eq. 6), replacing Kimi Linear's low-rank gate.

See gated_deltanet_2/layer.py for both.

BLOCK STRUCTURE — NOT the usual pre-norm residual
--------------------------------------------------
The familiar `x = x + Mixer(Norm(x))` is gone. Under AttnRes each module reads
its own input by ATTENDING over the depth sources, and its output is accumulated
into the current block's running sum rather than into its own input:

    h = attn_res(blocks, partial);  partial += TokenMixer(  RMSNorm(h) )
    h = mlp_res( blocks, partial);  partial += ChannelMixer(RMSNorm(h) )

with `(blocks, partial)` threaded through the layers — `DecoderLayer.__call__`
transcribes the AttnRes paper's Fig. 2 pseudocode statement for statement,
including the block-boundary check that sits between the first read and the
token mixer. There is no additive residual because the read itself is a softmax
mixture over sources that always include the token embedding: the network can
choose a residual-like read, it just is not hardwired to one.

MODEL = Embed -> [DecoderLayer] * n_layers -> AttnRes read -> RMSNorm -> LM head.

TWO FORWARD MODES
-----------------
  • Training / full sequence:  model(input_ids)  — parallel; GDN-2 via its
    chunkwise core, Gated MLA via a full causal-attention matrix.
  • Streaming / inference:     model.step(ids, caches) and model.generate(...) —
    reuses per-layer state across calls so each new token is O(1) work for the
    GDN-2 layers (fixed-size recurrent state) and O(context) for the few MLA
    layers (growing latent cache). AttnRes needs no cache: it attends across
    LAYERS, not across tokens, so all of its state is internal to one forward.

DELIBERATELY OUT OF SCOPE
-------------------------
K3 is a native multimodal model trained with a specific optimizer; this file is
the language backbone only. Not implemented here: MoonViT-V2 and the vision
projector (§2.4), Per-Head Muon (§2.5 — an optimizer, not an architecture), and
the MTP / EAGLE-3 draft layer (Table 1) used for speculative decoding.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Reuse the building blocks already implemented and verified in this repo.
from attention_residuals import AttnResReader
from gated_deltanet_2.layer import GatedDeltaNet2, GDN2Cache, RMSNorm
from multi_latent_attention.attention import GatedMultiLatentAttention, MLACache
from multi_latent_attention.moe import DenseFFN, StableLatentMoE

# Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale = gain² = 2^{-5})
# for the embedding and LM head, replacing Flax NNX's defaults. The (small)
# embedding scale this produces is fine — RMSNorm renormalizes the stream.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


# --------------------------------------------------------------------------- #
#  Configuration
#
#  Defaults are deliberately TINY so the whole model runs on a laptop CPU. The
#  paper's 2.8T-A104B numbers are quoted in comments for reference (Table 1);
#  only the RATIOS and the structure matter for understanding — scale up by
#  raising the dims/blocks.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class KimiK3Config:
    vocab_size: int = 256  # paper: 160k; tiny here (byte-level demo)
    d_model: int = 256  # model width d   (paper: 7168)

    # --- Hybrid attention schedule (§2.1) ---
    # Each block is (full_attn_period - 1) linear layers + 1 Gated MLA layer,
    # i.e. the paper's 3:1 ratio at the default period of 4. `n_blocks` blocks
    # give 4·n_blocks layers; ONE extra Gated MLA is appended so the final layer
    # is always global attention ("An additional Gated MLA layer is placed at the
    # end of the backbone"). Paper: 23 blocks -> 69 KDA + 23 MLA, +1 = 93 layers.
    n_blocks: int = 2  # -> n_layers = 4*2 + 1 = 9
    full_attn_period: int = 4

    # --- Attention Residuals (§2.2) ---
    # DECODER LAYERS per AttnRes block — a SEPARATE, coarser partition than the
    # 4-layer hybrid block above (K3: 12 layers = 3 hybrid blocks).
    #
    # The real design rule is the block COUNT, not this size: the AttnRes paper
    # (arXiv:2603.15031 §5.3) sweeps block size and finds a broad plateau, then
    # fixes "the number of blocks to ≈8 for infrastructure efficiency" — N is
    # what sets cross-stage communication under pipeline parallelism. K3 obeys
    # that at 93 layers / 12 = 8 blocks (the last partial), 9 sources with the
    # embedding. So when scaling this config up, pick attnres_block_size ≈
    # n_layers / 8 rather than holding it fixed.
    #
    # CAUTION when reading the AttnRes paper directly: it counts MODULES, not
    # decoder layers ("27 Transformer blocks (54 layers)"; its pseudocode notes
    # "block_size counts ATTN + MLP"). This field follows K3's convention, so a
    # value copied straight out of that paper's sweep would be 2× too small.
    #
    # Two extreme settings are useful for testing. block_size = n_layers gives
    # N = 1: every read is then a 2-way softmax between the embedding and one
    # accumulated stream — the paper's "standard residual connections with the
    # embedding isolated as b_0". block_size = 1 is the finest granularity this
    # implementation offers, one source per decoder layer; it is NOT quite Full
    # AttnRes, since a layer's two module outputs still get summed together.
    attnres_block_size: int = 3

    # --- GDN-2 token mixer (the KDA replacement) — gated_deltanet_2/layer.py ---
    gdn_num_heads: int = 4  # H key/query heads   (paper: 96 attention heads)
    gdn_head_k_dim: int = 64  # d_k
    gdn_head_v_dim: int = 64  # d_v
    gdn_num_v_heads: int | None = None  # H_v for GQA value heads; None -> = num_heads
    gdn_chunk_size: int = 64  # chunkwise block size C
    #   NOTE: the GDN-2 chunkwise core requires every fed sequence length to be a
    #   multiple of this C (it reshapes L into L/C chunks). Keep seq_len % C == 0.
    gdn_conv_size: int = 4  # short-conv kernel width
    gdn_expanded_erase: bool = False  # erase gate in [0,2] (neg-eigenvalue variant)
    gdn_core: str = "centered"  # which GDN-2 chunkwise core computes each head
    # K3 Eq. 5's lower-bounded decay; "softplus" reverts to GDN-2 / Kimi Linear.
    gdn_decay_mode: str = "bounded_sigmoid"
    gdn_decay_min: float = -5.0  # g_min; K3 fixes -5

    # --- Gated MLA layers (NoPE) — multi_latent_attention/attention.py ---
    # MLA has ONE shared latent per token: `kv_lora_rank` is the whole cache
    # width, and EVERY head reconstructs its own key and value from all of it
    # (see attention.py — slicing it per head would be GQA, not MLA). Paper
    # Table 1 gives H = 96; widths below are scaled to this demo model, keeping
    # kv_lora_rank well under H*(qk+v) so the cache actually compresses.
    mla_num_heads: int = 8  # H attention heads (paper: 96)
    mla_kv_lora_rank: int = 64  # width of the shared latent c_t — the cache
    mla_qk_head_dim: int = 32  # per-head key/query width
    mla_v_head_dim: int = 32  # per-head value width
    # Query down-projection rank; queries are never cached, so this only trades
    # parameters. None uses a direct projection.
    mla_q_lora_rank: int | None = 64
    # Declared context cap: used as the default size of the preallocated MLA
    # latent cache in init_cache/generate. (The causal mask itself is built on
    # the fly from the actual length.) Paper: 1M tokens — NoPE means there is no
    # positional parameter to retune when extending it.
    max_seq_len: int = 512

    # --- Channel mixer: Stable LatentMoE (§2.3) ---
    moe_latent_dim: int | None = None  # ℓ; None -> d_model // 2, the paper's 0.5×
    moe_d_ff: int = 128  # per-ROUTED-expert hidden width, in latent space (paper: 3072)
    moe_d_ff_shared: int | None = None  # per-SHARED-expert hidden width; None -> moe_d_ff
    moe_n_routed: int = 8  # routed experts E     (paper: 896)
    moe_n_shared: int = 2  # shared experts N_s   (paper: 2)
    moe_top_k: int = 2  # experts activated per token (paper: 16)
    moe_beta1: float = 4.0  # SiTU-GLU gate-branch soft cap  (paper: 4)
    moe_beta2: float = 25.0  # SiTU-GLU up-branch soft cap    (paper: 25)
    # Optional group-limited ("node-limited") routing from DeepSeek-V3 / Kimi K2.
    # K3 does not describe it — Quantile Balancing is its answer to routing at
    # scale — so it is OFF by default. Set moe_n_groups > 1 to enable.
    moe_n_groups: int = 1
    moe_topk_groups: int = 1
    # Table 1, "Number of Dense Layers: 1": the first layer's channel mixer is a
    # plain dense FFN, not MoE (see DenseFFN for why).
    n_dense_layers: int = 1

    rms_eps: float = 1e-5

    # --- Mixed precision ---
    # Matmul (compute) dtype for the projection Linears + MoE expert GEMMs.
    # Master weights are ALWAYS stored fp32 (param_dtype), and the numerically
    # sensitive parts stay fp32 regardless: the GDN-2 chunkwise core, every
    # RMSNorm, the AttnRes and attention softmaxes, the router, and the loss.
    # Set "bfloat16" on an H200; "float32" disables mixed precision.
    compute_dtype: str = "float32"

    @property
    def cdtype(self) -> jnp.dtype:
        return jnp.dtype(self.compute_dtype)

    @property
    def n_layers(self) -> int:
        """4·n_blocks hybrid layers + the trailing global-attention layer (§2.1)."""
        return self.full_attn_period * self.n_blocks + 1

    @property
    def latent_dim(self) -> int:
        """ℓ, the routed-expert width (§2.3). Paper: 3584 = 0.5 × 7168."""
        if self.moe_latent_dim is not None:
            return self.moe_latent_dim
        return self.d_model // 2


# --------------------------------------------------------------------------- #
#  One decoder layer: token mixer + channel mixer, each preceded by its own
#  AttnRes read and its own pre-norm.
#
#  What varies across layers:
#    * the TOKEN mixer — Gated MLA on the 3:1 schedule (and on the final layer),
#      GDN-2 everywhere else;
#    * the CHANNEL mixer — a dense FFN on the first `n_dense_layers` layers,
#      Stable LatentMoE afterwards.
# --------------------------------------------------------------------------- #
class DecoderLayer(nnx.Module):
    def __init__(self, cfg: KimiK3Config, layer_idx: int, *, rngs: nnx.Rngs):
        # §2.1: full attention on the last layer of every hybrid block, and on
        # the extra trailing layer that guarantees a global final layer.
        is_block_end = (layer_idx + 1) % cfg.full_attn_period == 0
        is_last = layer_idx == cfg.n_layers - 1
        self.is_full_attn = is_block_end or is_last

        # AttnRes readers (§2.2). Fig. 2 hangs an (α, w) pair off every module,
        # so the token mixer and the channel mixer each get one — they read the
        # depth sources independently and can attend to different ones. Named
        # after the pseudocode's attn_res_* / mlp_res_* pairs.
        self.attn_res = AttnResReader(cfg.d_model, rngs=rngs)
        self.mlp_res = AttnResReader(cfg.d_model, rngs=rngs)

        # Block boundary, decided once at construction so the forward has no
        # traced branch. The pseudocode writes this as
        #     layer_number % (block_size // 2) == 0
        # where its block_size counts MODULES; attnres_block_size already counts
        # decoder layers, so the //2 is folded in. Note layer 0 is a boundary:
        # that is what appends the token embedding as b_0.
        self.starts_new_block = layer_idx % cfg.attnres_block_size == 0

        # Pre-norm before the token mixer (the "Norm" boxes inside the modules of
        # Fig. 2). RMSNorm reused from the GDN-2 layer.
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)

        if self.is_full_attn:
            # Full attention: NoPE Gated MLA (absorbed/GQA form), K3 §2.1.2.
            self.token_mixer = GatedMultiLatentAttention(
                embed_dim=cfg.d_model,
                num_heads=cfg.mla_num_heads,
                kv_lora_rank=cfg.mla_kv_lora_rank,
                qk_head_dim=cfg.mla_qk_head_dim,
                v_head_dim=cfg.mla_v_head_dim,
                q_lora_rank=cfg.mla_q_lora_rank,
                rms_eps=cfg.rms_eps,
                compute_dtype=cfg.cdtype,
                output_gate=True,  # K3 Eq. 7
                rngs=rngs,
            )
        else:
            # Linear attention: Gated DeltaNet-2 (the KDA substitute), carrying
            # K3's lower-bounded decay (Eq. 5) and full-rank output gate (Eq. 6).
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
                decay_mode=cfg.gdn_decay_mode,
                decay_min=cfg.gdn_decay_min,
                output_gate_rank=None,  # K3 Eq. 6: full-rank W_g
                rngs=rngs,
            )

        # Pre-norm before the channel mixer.
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)

        # Channel mixer: dense on the first layer(s), Stable LatentMoE after.
        self.is_moe = layer_idx >= cfg.n_dense_layers
        if self.is_moe:
            self.channel_mixer = StableLatentMoE(
                d_model=cfg.d_model,
                latent_dim=cfg.latent_dim,
                d_ff=cfg.moe_d_ff,
                d_ff_shared=cfg.moe_d_ff_shared,
                n_routed=cfg.moe_n_routed,
                n_shared=cfg.moe_n_shared,
                top_k=cfg.moe_top_k,
                n_groups=cfg.moe_n_groups,
                topk_groups=cfg.moe_topk_groups,
                beta1=cfg.moe_beta1,
                beta2=cfg.moe_beta2,
                rms_eps=cfg.rms_eps,
                compute_dtype=cfg.cdtype,
                rngs=rngs,
            )
        else:
            self.channel_mixer = DenseFFN(
                d_model=cfg.d_model,
                # Match the MoE layer's activated width so the dense layer is
                # comparable in size: top_k routed experts + the shared ones.
                d_ff=cfg.moe_d_ff * (cfg.moe_top_k + cfg.moe_n_shared),
                beta1=cfg.moe_beta1,
                beta2=cfg.moe_beta2,
                compute_dtype=cfg.cdtype,
                rngs=rngs,
            )

    def __call__(
        self, blocks: list[jax.Array], hidden_states: jax.Array
    ) -> tuple[list[jax.Array], jax.Array, dict[str, jax.Array]]:
        """One layer, following the AttnRes paper's Fig. 2 `forward` statement for
        statement.

        blocks:        the completed block representations [b_0, …, b_{n-1}].
        hidden_states: the incoming intra-block partial sum b_n^i.
        Returns (blocks, partial_block, aux) — aux is the channel mixer's MoE
        diagnostics, empty for a dense layer.

        Note what is NOT here: no `x = x + ...` against a residual stream. Each
        module's input is an attention read over the depth sources, and its
        output is accumulated into the OPEN BLOCK's sum, not into its own input.
        """
        partial_block = hidden_states

        # apply block attnres before the token mixer
        h = self.attn_res(blocks, partial_block)

        # if reaches block boundary, start new block. This sits BETWEEN the read
        # and the mixer, exactly as in Fig. 2: the read above still sees the
        # finished block as `partial_block`, and closing it here just moves it
        # into `blocks` before the new block starts accumulating. On layer 0 the
        # partial being closed IS the token embedding — that is how b_0 = h_1
        # becomes a permanent source.
        if self.starts_new_block:
            blocks = [*blocks, partial_block]
            partial_block = None

        # token mixer (KDA in the paper; GDN-2 or Gated MLA here)
        f = self.token_mixer(self.norm1(h))
        partial_block = partial_block + f if partial_block is not None else f

        # apply block attnres before the channel mixer — a FRESH read, which now
        # also sees the token mixer's contribution inside the open block
        h = self.mlp_res(blocks, partial_block)

        # channel mixer (Stable LatentMoE, or a dense FFN on the first layers)
        m, aux = self.channel_mixer(self.norm2(h))
        partial_block = partial_block + m

        return blocks, partial_block, aux

    def init_cache(self, batch_size: int, max_len: int, dtype=None):
        """Per-layer streaming cache: a GDN2Cache (linear layer) or MLACache (MLA).

        `dtype=None` lets each token mixer pick its own — which is what keeps
        decode numerically identical to training under bf16. Both mixers default
        their buffers to compute_dtype and keep the parts that must stay fp32
        (GDN-2's recurrent state) fp32 regardless."""
        return self.token_mixer.init_cache(batch_size, max_len, dtype)

    def step(
        self,
        blocks: list[jax.Array],
        hidden_states: jax.Array,
        cache: GDN2Cache | MLACache,
    ) -> tuple[list[jax.Array], jax.Array, GDN2Cache | MLACache]:
        """Streaming forward for one layer. Identical AttnRes bookkeeping to
        __call__; only the token mixer call differs (it threads a cache). The
        channel mixer is position-wise and AttnRes state lives inside this single
        forward pass, so neither needs one."""
        partial_block = hidden_states

        h = self.norm1(self.attn_res(blocks, partial_block))

        if self.starts_new_block:
            blocks = [*blocks, partial_block]
            partial_block = None

        if isinstance(cache, GDN2Cache) and isinstance(self.token_mixer, GatedDeltaNet2):
            # GDN-2: fixed-size recurrent state (O(1) per token).
            f, new_cache = self.token_mixer.step(h, cache)
        elif isinstance(cache, MLACache) and isinstance(
            self.token_mixer, GatedMultiLatentAttention
        ):
            # Gated MLA: growing latent cache (O(context) per token).
            f, new_cache = self.token_mixer.step(h, cache)
        else:
            raise ValueError(
                f"Cache type {type(cache)} does not match token mixer {type(self.token_mixer)}"
            )
        partial_block = partial_block + f if partial_block is not None else f

        m, _ = self.channel_mixer(self.norm2(self.mlp_res(blocks, partial_block)))
        partial_block = partial_block + m

        return blocks, partial_block, new_cache


# --------------------------------------------------------------------------- #
#  The full model.
# --------------------------------------------------------------------------- #
class KimiK3(nnx.Module):
    """Decoder-only Kimi K3 LM with a GDN-2 linear-attention backbone."""

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        # Token embedding table. Under AttnRes this is also b_0, a first-class
        # source every module can attend back to at any depth (Eq. 10).
        self.embed = nnx.Embed(
            cfg.vocab_size, cfg.d_model, embedding_init=_XAVIER, rngs=rngs
        )

        # Stack of decoder layers. NOTE: in Flax NNX a plain Python list of
        # submodules is not tracked as state — it must be wrapped in nnx.List.
        self.layers = nnx.List(
            [DecoderLayer(cfg, i, rngs=rngs) for i in range(cfg.n_layers)]
        )
        # Which rows of the stacked aux arrays belong to which layer (dense
        # layers produce none), so a training loop can map diagnostics back.
        self.moe_layer_indices = [
            i for i in range(cfg.n_layers) if self.layers[i].is_moe
        ]

        # §2.2: "The final output layer then aggregates all N block
        # representations" — one last AttnRes read, with its own pseudo-query,
        # over the closed blocks plus the trailing partial one.
        self.out_res = AttnResReader(cfg.d_model, rngs=rngs)

        # Final pre-head norm + untied LM head (the DeepSeek/Moonshot line does
        # not tie weights; to tie, drop lm_head and use x @ embed.embedding.T).
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.lm_head = nnx.Linear(
            cfg.d_model,
            cfg.vocab_size,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=cfg.cdtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    # ----------------------------------------------------------------------- #
    def _head(self, blocks: list[jax.Array], partial_block: jax.Array) -> jax.Array:
        """Shared tail of __call__/step: the final aggregation read + pre-head norm.

        `partial_block` is the last block, still open — K3's 93 layers over
        12-layer blocks leave exactly such a trailing partial. Passing it as the
        partial argument of the read is what makes it the Nth representation, so
        no explicit flush is needed."""
        return self.norm_f(self.out_res(blocks, partial_block))

    def __call__(self, input_ids: jax.Array) -> tuple[jax.Array, dict[str, ArrayLike]]:
        """input_ids: int[B, L] -> (logits[B, L, vocab], aux).

        aux is ALWAYS returned (callers that don't need it just unpack
        `logits, _ =`):
            aux_loss    scalar — 0.0 under K3's aux-loss-free routing, unless a
                        StableLatentMoE was given aux_alpha > 0.
            group_sizes int[n_moe_layers, E] — realized per-expert token counts.
            qb_bias     float[n_moe_layers, E] — the NEXT router bias from
                        Quantile Balancing (Eq. 14), to install AFTER this step
                        via `apply_quantile_balancing`.
        Rows are ordered by `self.moe_layer_indices`. Eval/inference paths simply
        ignore all of it.
        """
        aux_loss: ArrayLike = 0.0
        group_sizes: list[ArrayLike] = []
        qb_bias: list[ArrayLike] = []

        # AttnRes state, threaded through the layers as in Fig. 2's `forward`.
        # `blocks` starts EMPTY: the embedding enters as b_0 when layer 0 hits
        # its block boundary and closes the partial it was handed (Eq. 10).
        blocks: list[jax.Array] = []
        h = self.embed(input_ids)  # the initial partial block

        for layer in self.layers:
            blocks, h, aux = layer(blocks, h)
            if aux:  # dense layers return {}
                aux_loss = aux_loss + aux["aux_loss"]
                group_sizes.append(aux["group_sizes"])
                qb_bias.append(aux["qb_bias"])

        x = self._head(blocks, h)
        # Upcast logits to fp32 for a numerically stable softmax/cross-entropy
        # under bf16 compute (the lm_head matmul itself runs in compute_dtype).
        logits = self.lm_head(x).astype(jnp.float32)  # [B, L, vocab]

        return logits, {
            "aux_loss": aux_loss,
            "group_sizes": jnp.stack(group_sizes),
            "qb_bias": jnp.stack(qb_bias),
        }

    # ----------------------------------------------------------------------- #
    #  Streaming / inference.  Each layer carries its own cache (GDN-2:
    #  fixed-size recurrent state + conv state; MLA: growing latent cache).
    #  Reusing them makes generation O(1) per token for the linear layers instead
    #  of re-reading history.  AttnRes adds no cache — see its module docstring.
    # ----------------------------------------------------------------------- #
    def init_cache(
        self, batch_size: int, max_len: int | None = None, dtype=None
    ) -> list:
        """Streaming caches for every layer. `max_len` (default cfg.max_seq_len)
        sizes the MLA latent buffers; GDN-2 layers ignore it (fixed-size state).

        `dtype` defaults to None, meaning EACH LAYER PICKS ITS OWN — the MLA
        latent buffer and the GDN-2 short-conv buffers take compute_dtype, while
        the GDN-2 recurrent state stays fp32 regardless. Do not pass fp32 here to
        "be safe" under bf16: both mixers read their caches back into the forward
        path, so a wider buffer promotes the arithmetic downstream of it and
        decode silently stops matching training (see
        GatedMultiLatentAttention.init_cache for the full account)."""
        max_len = max_len or self.cfg.max_seq_len
        return [layer.init_cache(batch_size, max_len, dtype) for layer in self.layers]

    def step(self, input_ids: jax.Array, caches: list) -> tuple[jax.Array, list]:
        """One streaming step. input_ids: int[B, L] (L = prompt length on prefill,
        or 1 per decoded token). Returns (logits[B, L, vocab], new_caches)."""
        new_caches = []

        blocks: list[jax.Array] = []
        h = self.embed(input_ids)
        for layer, cache in zip(self.layers, caches):
            blocks, h, new_cache = layer.step(blocks, h, cache)
            new_caches.append(new_cache)

        x = self._head(blocks, h)
        return self.lm_head(x).astype(jnp.float32), new_caches

    def generate(
        self,
        prompt_ids: jax.Array,
        max_new_tokens: int,
        max_len: int | None = None,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_id: int | None = None,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Autoregressive decode that REUSES each layer's state across steps.
        prompt_ids: int[B, P]. Returns the continuation int[B, T] with
        T <= max_new_tokens (shorter only when `eos_id` ends every row early).

        SELECTION. Greedy by default (temperature == 0). temperature > 0 samples
        from softmax(logits / temperature), truncated to the nucleus: the
        smallest set of tokens whose probability mass reaches `top_p` (top_p=1.0
        disables the truncation). `key` seeds the sampler (defaults to
        PRNGKey(0) — pass your own for varied samples). With `eos_id` set, rows
        that emitted it keep emitting eos_id (padding), and the loop stops early
        once every row is done (costs one small host sync per token).

        Prefill consumes the whole prompt in one JITTED step (filling every
        layer's cache) — the GDN-2 layers push all whole chunks of the prompt
        through their PARALLEL chunkwise core and only the ragged tail through
        the recurrence, so prefill cost scales with P/chunk_size sequential
        steps, not P. It compiles once per (batch size, P, cache length). Each
        decode step then feeds back ONE token and carries the caches forward —
        the GDN-2 layers via their fixed-size recurrent state, the MLA layers via
        the growing latent cache — through the same module-level nnx.jit step,
        which compiles once per (batch size, cache length) and every further
        token — across generate() calls too — reuses the trace."""
        B, P = prompt_ids.shape
        # Default the cache length to the config's declared context cap when the
        # request fits inside it: a FIXED cache shape lets the decode step reuse
        # its compiled trace across generate() calls with different prompt
        # lengths (e.g. a chat loop) instead of recompiling for every
        # P + max_new_tokens.
        max_len = max_len or max(self.cfg.max_seq_len, P + max_new_tokens)
        if P + max_new_tokens > max_len:
            # An undersized MLA cache would not error: dynamic_update_slice CLAMPS
            # out-of-bounds start indices, silently overwriting the last slot.
            raise ValueError(
                f"prompt ({P}) + max_new_tokens ({max_new_tokens}) exceeds "
                f"max_len ({max_len}); the MLA latent cache would overflow.")

        greedy = temperature <= 0.0
        if not greedy and key is None:
            key = jax.random.PRNGKey(0)

        def advance(tok, caches):
            nonlocal key
            if greedy:
                return _decode_step(self, tok, caches)
            key, sub = jax.random.split(key)
            return _sample_step(self, tok, caches, sub, temperature, top_p)

        caches = self.init_cache(B, max_len)
        next_tok, caches = advance(prompt_ids, caches)  # jitted prefill
        outs = [next_tok]
        done = (next_tok == eos_id) if eos_id is not None else None

        for _ in range(max_new_tokens - 1):
            if done is not None and bool(jnp.all(done)):
                break  # every row has emitted eos_id
            next_tok, caches = advance(next_tok, caches)
            if done is not None:
                next_tok = jnp.where(done, eos_id, next_tok)  # pad finished rows
                done = done | (next_tok == eos_id)
            outs.append(next_tok)

        return jnp.concatenate(outs, axis=1)  # [B, T<=max_new_tokens]


# --------------------------------------------------------------------------- #
#  Jitted decode steps, shared by every generate() call (prefill included).
#
#  During decoding everything is shape-constant — the weights, the fixed-size
#  GDN-2 states, the preallocated MLA latent buffers (position is a TRACED int32,
#  so advancing it never retraces), and L=1 — so these compile ONCE per (batch
#  size, input length, cache length) and each further token replays the compiled
#  trace. Module-level on purpose: nnx.jit keys its compilation cache on the
#  function object, so a wrapper created inside generate() would recompile every
#  call.
# --------------------------------------------------------------------------- #
@nnx.jit
def _decode_step(
    model: KimiK3, tok: jax.Array, caches: list
) -> tuple[jax.Array, list]:
    """One greedy step: tok int[B, L] -> (next greedy token int[B, 1], caches).
    L is the prompt length on prefill, 1 on every decode step."""
    logits, caches = model.step(tok, caches)
    return jnp.argmax(logits[:, -1:], axis=-1), caches


@nnx.jit
def _sample_step(
    model: KimiK3, tok: jax.Array, caches: list,
    key: jax.Array, temperature: jax.Array, top_p: jax.Array,
) -> tuple[jax.Array, list]:
    """One sampling step: softmax(logits / temperature) truncated to the top-p
    nucleus. temperature/top_p are traced scalars, so changing them between
    generate() calls reuses the compiled trace."""
    logits, caches = model.step(tok, caches)
    logits = logits[:, -1] / temperature  # [B, vocab], already fp32

    # Nucleus (top-p) filter: sort descending, find the logit where the
    # cumulative probability first reaches top_p, and mask everything below it.
    # top_p = 1.0 keeps every token (the cutoff lands on the smallest logit).
    sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
    cum = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
    cut_idx = jnp.minimum(
        jnp.sum(cum < top_p, axis=-1, keepdims=True), logits.shape[-1] - 1)
    cutoff = jnp.take_along_axis(sorted_logits, cut_idx, axis=-1)  # [B, 1]
    logits = jnp.where(logits < cutoff, -jnp.inf, logits)

    return jax.random.categorical(key, logits, axis=-1)[:, None], caches


# --------------------------------------------------------------------------- #
def apply_quantile_balancing(model: KimiK3, aux: dict) -> None:
    """Install the Quantile Balancing biases produced by a forward pass (§2.3.3).

    Call this in the training loop AFTER the optimizer step:

        logits, aux = model(batch)
        ... loss / grads / optimizer.update(...) ...
        apply_quantile_balancing(model, aux)

    The timing is load-bearing, not cosmetic: "the update takes effect only in
    the next step, i.e. a batch is never routed with a bias derived from itself"
    (§2.3.3). The bias is an nnx.Variable, not an nnx.Param, so this assignment
    stays outside the gradient by construction. Stop calling it at inference —
    the paper freezes the final bias there.
    """
    for row, layer_idx in enumerate(model.moe_layer_indices):
        model.layers[layer_idx].channel_mixer.router_bias[...] = aux["qb_bias"][row]


def count_params(model: nnx.Module) -> int:
    """Total number of trainable parameters (sum of nnx.Param leaf sizes)."""
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))))
