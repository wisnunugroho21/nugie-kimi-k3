"""Gated NoPE Multi-head Latent Attention — §2.1.2 of the Kimi K3 report.

This is the *global* token mixer: 24 of K3's 93 layers, one at the end of every
3-KDA block plus one extra at the very top of the backbone, so the final layer
always performs full attention. The KDA layers give cheap position-aware local
mixing; these layers give unrestricted content-based access to every token.

THREE THINGS ARE GOING ON.

1. MLA (from DeepSeek-V2, kept from K2/K2.5). Each token is compressed into a
   single low-dimensional latent `c_t = W_c x_t` (rank 512 against 96 heads x
   128 dims). Only `c_t` is cached at decode time; the per-head keys and values
   are reconstructed from it through learned up-projections. The KV cache is
   therefore ~48x smaller than storing full K and V, which is what makes these
   layers affordable at a 1M-token context.

2. NoPE — no positional encoding at all. K2 used RoPE here; K3 follows Kimi
   Linear and drops it. The reasoning: the interleaved KDA layers already carry
   position implicitly through their decay/recurrence, so the global layers are
   free to be purely content-addressed. The practical payoff is that extending
   the context window needs no positional surgery whatsoever — no RoPE base
   retuning, no YaRN interpolation. K3 extrapolates from 64K training to 1M
   inference with the same weights.

   NoPE also has a nice side effect: with no rotation sitting between them, the
   key up-projection folds algebraically into the query projection, so decode
   can attend directly against the cached latent and never materialise K at all
   (see `step`).

3. GATED (Eq. 7) — an input-dependent, channel-wise, FULL-RANK output gate:

       y_t = W_o [ Sigmoid(W_g x_t) * otilde_t ]

   Same gate K3 puts on KDA (Eq. 6), and the same "Gated Attention" mechanism
   K2 adopted: it lets each token decide, per channel, how much of what global
   attention retrieved actually enters the residual stream. Empirically this is
   what suppresses attention sinks and massive activations.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from .config import KimiK3Config
from .layers import F32

_NEG_INF = -1e30


class MLACache(NamedTuple):
    """Decode cache. We store ONLY the compressed latent, one vector per token.

    Unlike the KDA layers' fixed-size state, this grows linearly with context —
    these are the layers that pay the long-context memory cost, which is exactly
    why the hybrid keeps them to 1 in 4.
    """

    latent: jax.Array  # [B, max_len, kv_lora_rank]
    pos: jax.Array  # scalar int32: how many slots are filled


class GatedMLA(nnx.Module):
    """Multi-head Latent Attention with NoPE and a full-rank output gate."""

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        d, H, Dh = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        self.H, self.Dh = H, Dh
        self.scale = Dh**-0.5

        # --- queries: d -> q_lora_rank -> H*Dh.
        # The query path is factored purely to save parameters (96*128 = 12288
        # is larger than d itself); it has nothing to do with the KV cache.
        self.q_down = nnx.Linear(d, cfg.q_lora_rank, use_bias=False, rngs=rngs)
        self.q_norm = nnx.RMSNorm(cfg.q_lora_rank, epsilon=cfg.rms_norm_eps, rngs=rngs)
        self.q_up = nnx.Linear(cfg.q_lora_rank, H * Dh, use_bias=False, rngs=rngs)

        # --- keys/values: the MLA latent. `kv_down` is W_c; `kv_up` reconstructs
        # both K and V from it (hence the 2*Dh output width).
        self.kv_down = nnx.Linear(d, cfg.kv_lora_rank, use_bias=False, rngs=rngs)
        self.kv_norm = nnx.RMSNorm(cfg.kv_lora_rank, epsilon=cfg.rms_norm_eps, rngs=rngs)
        self.kv_up = nnx.Linear(cfg.kv_lora_rank, H * 2 * Dh, use_bias=False, rngs=rngs)

        # --- Eq. 7: full-rank gate + output projection.
        self.gate_proj = nnx.Linear(d, H * Dh, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(H * Dh, d, use_bias=False, rngs=rngs)

    # ------------------------------------------------------------------
    def _queries(self, x: jax.Array) -> jax.Array:
        B, T, _ = x.shape
        q = self.q_up(self.q_norm(self.q_down(x)))
        return q.reshape(B, T, self.H, self.Dh).transpose(0, 2, 1, 3)  # [B,H,T,Dh]

    def _keys_values(self, latent: jax.Array):
        B, T, _ = latent.shape
        kv = self.kv_up(latent).reshape(B, T, self.H, 2 * self.Dh)
        k, v = jnp.split(kv, 2, axis=-1)
        return k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)  # [B,H,T,Dh] each

    def _output(self, o: jax.Array, x: jax.Array) -> jax.Array:
        """Eq. 7."""
        B, _, T, _ = o.shape
        o = o.transpose(0, 2, 1, 3).reshape(B, T, self.H * self.Dh)
        return self.out_proj(jax.nn.sigmoid(self.gate_proj(x)) * o)

    # ------------------------------------------------------------------
    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """Training / prefill. x: [B, T, d] -> [B, T, d].

        `mask` is an optional [B, T] boolean of valid positions (padding).
        Causality is applied unconditionally.
        """
        B, T, _ = x.shape
        q = self._queries(x)
        latent = self.kv_norm(self.kv_down(x))
        k, v = self._keys_values(latent)

        # §2.1.2 keeps the attention output in FP32 during training, to avoid the
        # biased rounding error flash-attention accumulates in low precision.
        # Here that just means: do the softmax and the weighted sum in fp32.
        logits = jnp.einsum("bhtd,bhsd->bhts", q.astype(F32), k.astype(F32)) * self.scale

        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        if mask is not None:
            causal = causal[None] & mask[:, None, :]  # [B,T,S]
            causal = causal[:, None]  # [B,1,T,S]
        logits = jnp.where(causal, logits, _NEG_INF)

        attn = jax.nn.softmax(logits, axis=-1)
        o = jnp.einsum("bhts,bhsd->bhtd", attn, v.astype(F32))
        return self._output(o, x)

    # ------------------------------------------------------------------
    def init_cache(self, batch: int, max_len: int) -> MLACache:
        return MLACache(
            jnp.zeros((batch, max_len, self.cfg.kv_lora_rank), dtype=F32),
            jnp.array(0, dtype=jnp.int32),
        )

    def step(self, x: jax.Array, cache: MLACache) -> tuple[jax.Array, MLACache]:
        """One decode token. x: [B, 1, d].

        Note what is *not* here: no RoPE, so no relative-position bookkeeping and
        no cache rewriting. We append one latent vector and attend over the
        prefix. (A production kernel would go further and absorb `kv_up` into
        `q_up` and `out_proj`, attending against the raw latent so K and V are
        never materialised — an exact algebraic rewrite that only NoPE permits.)
        """
        B = x.shape[0]
        latent = self.kv_norm(self.kv_down(x))  # [B,1,r]
        buf = jax.lax.dynamic_update_slice(cache.latent, latent.astype(F32), (0, cache.pos, 0))
        pos = cache.pos + 1

        q = self._queries(x)  # [B,H,1,Dh]
        k, v = self._keys_values(buf)  # [B,H,max_len,Dh]
        logits = jnp.einsum("bhtd,bhsd->bhts", q.astype(F32), k.astype(F32)) * self.scale
        valid = jnp.arange(buf.shape[1]) < pos
        logits = jnp.where(valid[None, None, None, :], logits, _NEG_INF)
        o = jnp.einsum("bhts,bhsd->bhtd", jax.nn.softmax(logits, axis=-1), v.astype(F32))
        return self._output(o, x), MLACache(buf, pos)
