"""The Kimi K3 model — assembling §2 into one network.

    "The Kimi K3 architecture is designed to scale information flow along three
     complementary dimensions: sequence length, network depth, and model width."

That sentence is the whole design, and each dimension has exactly one mechanism:

    SEQUENCE  Hybrid Attention (§2.1) — 3 KDA layers then 1 Gated MLA layer,
              repeated, plus a trailing MLA so the last layer is always global.
              69 KDA + 24 MLA = 93 layers.                        kda.py, mla.py
    DEPTH     Attention Residuals (§2.2) — layers attend over block
              representations instead of accumulating a residual stream.
                                                                   attn_res.py
    WIDTH     Stable LatentMoE (§2.3) — every attention layer is paired with a
              sparse channel mixer, 16 of 896 routed experts active.     moe.py

Plus a native vision pathway at the input (§2.4, vision.py) and Per-Head Muon
(§2.5, muon.py).

HOW A LAYER READS ITS INPUT
---------------------------
This is the one structural thing that differs from a normal transformer, so it
is worth stating plainly. There is no `h = h + f(h)` here. Instead the model
carries an `AttnResStream` of block representations, and each layer:

    1. takes the value set V of Eq. 10 (all sealed blocks, plus the current
       block's running partial sum),
    2. its attention module and its FFN module EACH attend over that same V with
       their own learned pseudo-query, producing their own input,
    3. their two outputs are summed into `f_i(h_i)` and added to the running
       partial sum.

Every 12 layers the partial sum is sealed as a new block representation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .attn_res import AttnResStream, DepthAttention
from .config import KimiK3Config
from .kda import KimiDeltaAttention
from .layers import F32
from .mla import GatedMLA
from .moe import RouterStats, SiTUFFN, StableLatentMoE, quantile_balancing_update
from .vision import VisionTower


class DecoderLayer(nnx.Module):
    """One K3 layer: a token mixer and a channel mixer, both fed by AttnRes.

    The token mixer is KDA or Gated MLA depending on position (§2.1); the channel
    mixer is Stable LatentMoE, except in the single leading dense layer.
    """

    def __init__(self, cfg: KimiK3Config, layer_idx: int, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_mla = cfg.is_mla_layer(layer_idx)
        self.is_dense = cfg.is_dense_layer(layer_idx)
        d = cfg.hidden_size

        # --- token mixer. Each module owns a pseudo-query (Fig. 2's per-module
        # alpha/w) and a pre-norm; AttnRes supplies the input, the norm sets its
        # scale before it hits the projections.
        self.attn_query = DepthAttention(d, rngs=rngs)
        self.attn_norm = nnx.RMSNorm(d, epsilon=cfg.rms_norm_eps, rngs=rngs)
        self.attn = GatedMLA(cfg, rngs=rngs) if self.is_mla else KimiDeltaAttention(cfg, rngs=rngs)

        # --- channel mixer.
        self.ffn_query = DepthAttention(d, rngs=rngs)
        self.ffn_norm = nnx.RMSNorm(d, epsilon=cfg.rms_norm_eps, rngs=rngs)
        self.ffn = (
            SiTUFFN(d, cfg.dense_ffn_hidden, cfg, rngs=rngs)
            if self.is_dense
            else StableLatentMoE(cfg, rngs=rngs)
        )

    def __call__(self, values: jax.Array, mask=None) -> tuple[jax.Array, RouterStats | None]:
        """values: [S, B, T, d] — the value set V of Eq. 10, fixed for this layer.

        Returns `f_i(h_i)`, the layer's contribution to its block's sum.
        """
        a_in = self.attn_norm(self.attn_query(values))
        attn_out = self.attn(a_in, mask) if self.is_mla else self.attn(a_in)

        f_in = self.ffn_norm(self.ffn_query(values))
        if self.is_dense:
            ffn_out, stats = self.ffn(f_in), None
        else:
            ffn_out, stats = self.ffn(f_in)

        return attn_out + ffn_out, stats


class MTPLayer(nnx.Module):
    """The multi-token-prediction layer ([Table 1] "Number of MTP Layers: 1").

    Trained to predict token t+2 from the backbone's state at t, which densifies
    the training signal. Its real payoff comes later: §4.1.4 fine-tunes this
    exact pre-trained layer into an EAGLE-3 speculative-decoding draft model,
    "as the draft model of EAGLE-3 comprises a single decoder layer whose
    structure matches the MTP layer".

    Structure follows DeepSeek-V3's: normalise the backbone's hidden state and
    the embedding of the next token, concatenate, project back to d, and run one
    decoder layer. It uses a plain residual rather than AttnRes — AttnRes is a
    backbone-depth mechanism and would be degenerate over a single layer.
    """

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        d = cfg.hidden_size
        self.norm_hidden = nnx.RMSNorm(d, epsilon=cfg.rms_norm_eps, rngs=rngs)
        self.norm_embed = nnx.RMSNorm(d, epsilon=cfg.rms_norm_eps, rngs=rngs)
        self.merge = nnx.Linear(2 * d, d, use_bias=False, rngs=rngs)
        # A global-attention layer: a draft head benefits from full context.
        self.attn = GatedMLA(cfg, rngs=rngs)
        self.attn_norm = nnx.RMSNorm(d, epsilon=cfg.rms_norm_eps, rngs=rngs)
        self.ffn = StableLatentMoE(cfg, rngs=rngs)
        self.ffn_norm = nnx.RMSNorm(d, epsilon=cfg.rms_norm_eps, rngs=rngs)

    def __call__(self, hidden: jax.Array, next_embed: jax.Array, mask=None):
        h = self.merge(
            jnp.concatenate([self.norm_hidden(hidden), self.norm_embed(next_embed)], axis=-1)
        )
        h = h + self.attn(self.attn_norm(h), mask)
        out, stats = self.ffn(self.ffn_norm(h))
        return h + out, stats


class Eagle3Fusion(nnx.Module):
    """§4.1.4: the draft model's low/mid/high feature fusion.

    "The draft input fuses low-, mid-, and high-level features of the target
    model, taken from the outputs of the 1st, 4th, and final AttnRes blocks.
    These features are concatenated and projected to the hidden size by a
    bias-free matrix W_E3, initialized as [0 0 I]."

    That initialisation is the interesting part: at step 0 the fused vector
    equals the high-level feature exactly — the input the MTP layer was
    pre-trained on — so fine-tuning starts from the pre-trained behaviour and
    *learns* to bring in the shallower features rather than being perturbed by
    them from the outset.
    """

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        d = cfg.hidden_size
        init = jnp.concatenate([jnp.zeros((2 * d, d), F32), jnp.eye(d, dtype=F32)], axis=0)
        self.w = nnx.Param(init)  # [3d, d], the [0 0 I] block structure

    def __call__(self, low: jax.Array, mid: jax.Array, high: jax.Array) -> jax.Array:
        return jnp.concatenate([low, mid, high], axis=-1) @ self.w[...]


class KimiK3(nnx.Module):
    """The full model: embedding -> 93 AttnRes layers -> output aggregation -> logits."""

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs, with_vision: bool = True):
        self.cfg = cfg
        d = cfg.hidden_size

        self.embed = nnx.Embed(cfg.vocab_size, d, rngs=rngs)
        self.layers = nnx.List([DecoderLayer(cfg, i, rngs=rngs) for i in range(cfg.num_layers)])

        # §2.2: "The final output layer then aggregates all N block
        # representations." One more pseudo-query, over the sealed blocks.
        self.output_query = DepthAttention(d, rngs=rngs)
        self.final_norm = nnx.RMSNorm(d, epsilon=cfg.rms_norm_eps, rngs=rngs)
        self.lm_head = nnx.Linear(d, cfg.vocab_size, use_bias=False, rngs=rngs)

        self.mtp = MTPLayer(cfg, rngs=rngs) if cfg.num_mtp_layers else None
        self.vision = VisionTower(cfg, rngs=rngs) if with_vision else None

    # ------------------------------------------------------------------
    def embed_inputs(
        self,
        input_ids: jax.Array,
        pixels: jax.Array | None = None,
        image_token_id: int | None = None,
    ) -> jax.Array:
        """Token embeddings, with visual tokens spliced in where requested.

        §2.4: "text, images, and videos are processed by a single shared backbone
        within one context, with no post-hoc modality-alignment stage". The
        entire multimodal interface is this splice — after it, nothing downstream
        knows or cares which tokens came from pixels.
        """
        h = self.embed(input_ids)
        if pixels is None:
            return h
        assert self.vision is not None and image_token_id is not None
        vis = self.vision(pixels)  # [B, Nv, d]

        # Write the visual tokens into the placeholder slots, in order.
        slots = input_ids == image_token_id  # [B, T]
        rank = jnp.cumsum(slots, axis=-1) - 1  # which visual token each slot takes
        gathered = jnp.take_along_axis(vis, jnp.clip(rank, 0)[..., None], axis=1)
        return jnp.where(slots[..., None], gathered, h)

    def backbone(self, h1: jax.Array, mask=None):
        """Run the 93 layers over the AttnRes stream. Returns (hidden, blocks, stats)."""
        stream = AttnResStream(h1)  # b_0 = the token embedding
        stats: list[RouterStats] = []

        for i, layer in enumerate(self.layers):
            # Eq. 10: the value set is computed ONCE per layer; both of the
            # layer's modules attend over it with their own pseudo-query.
            layer_out, layer_stats = layer(stream.values(), mask)
            stream.accumulate(layer_out)
            if layer_stats is not None:
                stats.append(layer_stats)
            # Seal b_n every `attn_res_block_size` layers.
            if (i + 1) % self.cfg.attn_res_block_size == 0:
                stream.close_block()
        stream.finish()  # K3's 93 layers leave a partial final block

        blocks = jnp.stack(stream.sources, axis=0)  # [N+1, B, T, d]
        hidden = self.final_norm(self.output_query(blocks))
        return hidden, blocks, stats

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        pixels: jax.Array | None = None,
        image_token_id: int | None = None,
        mask: jax.Array | None = None,
        return_mtp: bool = False,
    ):
        """input_ids: [B, T] -> dict with `logits`, `router_stats`, and friends."""
        h1 = self.embed_inputs(input_ids, pixels, image_token_id)
        hidden, blocks, stats = self.backbone(h1, mask)
        out = {
            "logits": self.lm_head(hidden),
            "hidden": hidden,
            "blocks": blocks,  # the AttnRes sources; EAGLE-3 reads 1st/4th/last
            "router_stats": stats,
        }
        if return_mtp and self.mtp is not None:
            # Predict token t+2: pair the state at t with the embedding at t+1.
            next_embed = jnp.concatenate(
                [self.embed(input_ids)[:, 1:], jnp.zeros_like(h1[:, :1])], axis=1
            )
            mtp_hidden, mtp_stats = self.mtp(hidden, next_embed, mask)
            out["mtp_logits"] = self.lm_head(mtp_hidden)
            stats.append(mtp_stats)  # so QB balances the MTP layer's router too
        return out


# ===========================================================================
# Training helpers
# ===========================================================================


def causal_lm_loss(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Mean next-token cross-entropy. `logits[:, t]` predicts `labels[:, t+1]`."""
    logits, targets = logits[:, :-1], labels[:, 1:]
    ll = jnp.take_along_axis(jax.nn.log_softmax(logits.astype(F32), -1), targets[..., None], -1)
    nll = -ll[..., 0]
    if mask is None:
        return nll.mean()
    m = mask[:, 1:].astype(F32)
    return (nll * m).sum() / jnp.clip(m.sum(), 1.0)


def mtp_loss(mtp_logits: jax.Array, labels: jax.Array) -> jax.Array:
    """The MTP objective: `mtp_logits[:, t]` predicts `labels[:, t+2]`."""
    logits, targets = mtp_logits[:, :-2], labels[:, 2:]
    ll = jnp.take_along_axis(jax.nn.log_softmax(logits.astype(F32), -1), targets[..., None], -1)
    return -ll[..., 0].mean()


def apply_quantile_balancing(model: KimiK3, stats: list[RouterStats]) -> None:
    """Run the QB update (Eq. 14) on every MoE layer, in place.

    Call this AFTER the optimiser step, so the new bias takes effect on the next
    batch: §2.3.3 requires that "a batch is never routed with a bias derived from
    itself". The bias is not a gradient-trained parameter — it is recomputed from
    scratch each step — which is why it lives in a `RouterBias` variable that the
    optimiser filters out.
    """
    # Order matches how `KimiK3.__call__` collects them: backbone layers first,
    # then the MTP layer if it ran.
    moe_layers = [layer.ffn for layer in model.layers if not layer.is_dense]
    if model.mtp is not None:
        moe_layers.append(model.mtp.ffn)
    assert len(stats) <= len(moe_layers), f"{len(stats)} stats vs {len(moe_layers)} MoE layers"
    for layer, s in zip(moe_layers, stats):
        layer.router_bias[...] = quantile_balancing_update(s, layer.top_k)
