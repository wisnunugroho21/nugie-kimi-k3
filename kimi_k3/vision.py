"""MoonViT-V2 and the vision projector — §2.4 "Native Vision" of the Kimi K3 report.

K3 is natively multimodal: text, images and video go through ONE backbone in ONE
context, with no post-hoc alignment stage. The vision pathway's job is therefore
narrow — turn pixels into tokens that live in the same embedding space as text —
and then it gets out of the way.

WHAT §2.4 SPECIFIES
-------------------
  * A 27-layer, ~401M-parameter ViT, RMSNorm, and NO bias terms anywhere in the
    linear or attention projections. The report attributes the bias removal to
    optimisation stability under from-scratch training.
  * Trained from scratch with next-token prediction — not initialised from a
    contrastively pre-trained encoder such as SigLIP. This is a deliberate
    reversal of prior practice (including K2.5's own). The reason given is
    stability: a SigLIP-initialised tower shows persistently higher gradient
    norms with frequent spikes when jointly optimised with the LLM, while the
    from-scratch tower stays smooth (Fig. 6). It also lets the language-modelling
    objective shape the visual features directly, rather than a contrastive loss
    that favours global semantics over fine-grained text and structure. The
    finding is that contrastive pre-training turns out to be *unnecessary* at
    scale — the from-scratch tower matches it on vision evals.
  * Images and video share parameters completely. Attention is FACTORISED into
    an intra-frame spatial pass and an inter-frame temporal pass, and temporal
    pooling compresses tokens along time.
  * A 2x2 pixel-shuffle before projection cuts visual tokens 4x, which is what
    keeps a 3584x3584 image affordable inside a 1M-token context.
  * A lightweight MLP projector maps the result into the LLM embedding space.

Widths (1152 hidden, 4304 MLP) are not stated in the report; they are the
SigLIP-So400M shape, which is what reproduces "27 layers / 401M / patch 14".

NOT MODELLED HERE: native-resolution sequence packing (real MoonViT packs
variable-sized images into one batch with block-diagonal attention). This file
takes a fixed [B, F, H, W, C] tensor so the factorised attention stays legible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .config import KimiK3Config
from .layers import F32


def rope_2d(x: jax.Array, h: int, w: int) -> jax.Array:
    """Axial 2-D rotary embedding over a grid of patches.

    Half the head channels encode the row index, half the column index. This is
    what lets a ViT accept native resolutions: position is relative and computed
    from the grid, so there is no fixed-length learned position table to
    interpolate when the image size changes. (Inherited from MoonViT; the K3
    report does not restate the vision tower's positional scheme.)

    x: [B, heads, h*w, dim] -> same shape.
    """
    B, H, N, D = x.shape
    q = D // 4  # a quarter of the channels per (axis, sin/cos) group
    freqs = 1.0 / (10_000 ** (jnp.arange(q, dtype=F32) / q))
    rows = jnp.repeat(jnp.arange(h, dtype=F32), w)[:, None] * freqs[None, :]  # [N, q]
    cols = jnp.tile(jnp.arange(w, dtype=F32), h)[:, None] * freqs[None, :]  # [N, q]
    angles = jnp.concatenate([rows, cols], axis=-1)  # [N, 2q] = [N, D/2]
    cos, sin = jnp.cos(angles)[None, None], jnp.sin(angles)[None, None]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class VisionAttention(nnx.Module):
    """Bias-free multi-head attention, used for both the spatial and temporal pass."""

    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nnx.Linear(dim, 3 * dim, use_bias=False, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, *, grid: tuple[int, int] | None = None) -> jax.Array:
        """x: [B, N, dim]. `grid=(h, w)` applies 2-D RoPE (the spatial pass);
        omit it for the temporal pass, where N indexes frames."""
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))  # [B,H,N,Dh]
        if grid is not None:
            q, k = rope_2d(q, *grid), rope_2d(k, *grid)
        # Bidirectional — a vision encoder has no causality to respect.
        attn = jax.nn.softmax(jnp.einsum("bhnd,bhmd->bhnm", q, k) * self.scale, axis=-1)
        o = jnp.einsum("bhnm,bhmd->bhnd", attn, v).transpose(0, 2, 1, 3)
        return self.proj(o.reshape(B, N, -1))


class MoonViTBlock(nnx.Module):
    """One MoonViT-V2 layer: factorised spatial + temporal attention, then an MLP.

    FACTORISED ATTENTION is the whole reason images and video can share one set
    of weights. Full 3-D attention over F*h*w patches costs O((F*h*w)^2); the
    factorisation runs
        - spatial: attend within each frame, across h*w patches (F copies), then
        - temporal: attend across the F frames, at each spatial location,
    for O(F*(hw)^2 + hw*F^2). An image is simply the F=1 case, where the temporal
    pass is a no-op — same parameters, no special-casing.
    """

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        d, eps = cfg.vision_hidden, cfg.rms_norm_eps
        self.norm_spatial = nnx.RMSNorm(d, epsilon=eps, use_scale=True, rngs=rngs)
        self.spatial = VisionAttention(d, cfg.vision_heads, rngs=rngs)
        self.norm_temporal = nnx.RMSNorm(d, epsilon=eps, use_scale=True, rngs=rngs)
        self.temporal = VisionAttention(d, cfg.vision_heads, rngs=rngs)
        self.norm_mlp = nnx.RMSNorm(d, epsilon=eps, use_scale=True, rngs=rngs)
        self.fc1 = nnx.Linear(d, cfg.vision_mlp_hidden, use_bias=False, rngs=rngs)
        self.fc2 = nnx.Linear(cfg.vision_mlp_hidden, d, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, frames: int, grid: tuple[int, int]) -> jax.Array:
        """x: [B, F*h*w, d] with frame-major ordering."""
        B, N, d = x.shape
        hw = grid[0] * grid[1]

        # --- spatial: [B, F*hw, d] -> [B*F, hw, d], attend, and fold back.
        s = self.norm_spatial(x).reshape(B * frames, hw, d)
        x = x + self.spatial(s, grid=grid).reshape(B, N, d)

        # --- temporal: regroup so the sequence axis is the frame axis.
        if frames > 1:
            t = self.norm_temporal(x).reshape(B, frames, hw, d)
            t = t.transpose(0, 2, 1, 3).reshape(B * hw, frames, d)
            t = self.temporal(t).reshape(B, hw, frames, d).transpose(0, 2, 1, 3)
            x = x + t.reshape(B, N, d)

        return x + self.fc2(jax.nn.gelu(self.fc1(self.norm_mlp(x))))


class MoonViTV2(nnx.Module):
    """The vision encoder: patchify -> 27 factorised blocks -> temporal pool."""

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        p = cfg.vision_patch_size
        # Patchify = a stride-p conv, i.e. a linear map on each p*p*3 patch.
        self.patch_embed = nnx.Conv(
            3, cfg.vision_hidden, kernel_size=(p, p), strides=(p, p), padding="VALID",
            use_bias=False, rngs=rngs,
        )
        self.blocks = nnx.List([MoonViTBlock(cfg, rngs=rngs) for _ in range(cfg.vision_layers)])
        self.norm = nnx.RMSNorm(cfg.vision_hidden, epsilon=cfg.rms_norm_eps, rngs=rngs)

    def __call__(self, pixels: jax.Array) -> tuple[jax.Array, tuple[int, int]]:
        """pixels: [B, F, H, W, 3] (an image is F=1). Returns ([B, N, d], grid)."""
        B, F_, H, W, _ = pixels.shape
        x = self.patch_embed(pixels.reshape(B * F_, H, W, 3))  # [B*F, h, w, d]
        _, gh, gw, d = x.shape
        x = x.reshape(B, F_ * gh * gw, d)

        for block in self.blocks:
            x = block(x, F_, (gh, gw))
        x = self.norm(x)

        # §2.4: temporal pooling compresses tokens along the time dimension.
        pool = self.cfg.vision_temporal_pool
        if F_ > 1 and pool > 1:
            keep = (F_ // pool) * pool
            x = x.reshape(B, F_, gh * gw, d)[:, :keep]
            x = x.reshape(B, keep // pool, pool, gh * gw, d).mean(axis=2)
            x = x.reshape(B, -1, d)
        return x, (gh, gw)


class VisionProjector(nnx.Module):
    """2x2 pixel shuffle + a lightweight MLP into the LLM embedding space.

    PIXEL SHUFFLE trades resolution for channels: a 2x2 neighbourhood of patch
    tokens is concatenated into one token of 4x the width. Four tokens become
    one, so the LLM sees 4x fewer visual tokens with no information discarded —
    it is a reshape, not a pooling. That 4x is what makes 3584x3584 inputs fit
    the 1M-token budget.
    """

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        s = cfg.vision_pixel_shuffle
        d_in = cfg.vision_hidden * s * s
        self.norm = nnx.RMSNorm(d_in, epsilon=cfg.rms_norm_eps, rngs=rngs)
        self.fc1 = nnx.Linear(d_in, cfg.hidden_size, use_bias=False, rngs=rngs)
        self.fc2 = nnx.Linear(cfg.hidden_size, cfg.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, grid: tuple[int, int]) -> jax.Array:
        """x: [B, N, d_vis] -> [B, N/s^2, d_model]."""
        B, N, d = x.shape
        s, (gh, gw) = self.cfg.vision_pixel_shuffle, grid
        frames = N // (gh * gw)
        x = x.reshape(B, frames, gh // s, s, gw // s, s, d)
        # group the s*s neighbourhood into the channel axis
        x = x.transpose(0, 1, 2, 4, 3, 5, 6).reshape(B, frames * (gh // s) * (gw // s), s * s * d)
        return self.fc2(jax.nn.gelu(self.fc1(self.norm(x))))


class VisionTower(nnx.Module):
    """MoonViT-V2 + projector. Output tokens are spliced straight into the LLM's
    embedding sequence — that splice is the entire "multimodal interface"."""

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        self.encoder = MoonViTV2(cfg, rngs=rngs)
        self.projector = VisionProjector(cfg, rngs=rngs)

    def __call__(self, pixels: jax.Array) -> jax.Array:
        features, grid = self.encoder(pixels)
        return self.projector(features, grid)
