"""NoPE Gated Multi-head Latent Attention (MLA) for the Kimi K3 backbone.

Kimi K3 places one Gated MLA layer after every three linear-attention layers and
adds one final Gated MLA layer at the end of the backbone.  This implementation
follows the released K3 factorization rather than treating the compressed KV
latent as both the key and the value:

  * queries use a low-rank projection with RMSNorm (W_QA, W_QB);
  * each token is compressed to one shared KV latent c_t (W_DKV);
  * RMSNorm(c_t) is up-projected into distinct per-head content keys and values;
  * an additional shared key component is concatenated to every head, but no
    rotary transform is applied -- K3's MLA layers are NoPE;
  * K3 Eq. 7 gates the full per-head value output before the output projection.

The streaming path caches only the compressed KV representation plus the small
shared key component.  Per-head keys and values are reconstructed from that
cache when attention is evaluated.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx


_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")
F32 = jnp.float32


class RMSNorm(nnx.Module):
    """RMSNorm used between the two low-rank MLA projections."""

    def __init__(self, dim: int, *, eps: float = 1e-6):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,), F32))

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        xf = x.astype(F32)
        xf = xf * jax.lax.rsqrt(
            jnp.mean(xf * xf, axis=-1, keepdims=True) + self.eps
        )
        return (xf * self.weight[...]).astype(dtype)


class MLACache(NamedTuple):
    """Compressed streaming cache for one MLA layer.

    ``l_kv`` stores ``[c_kv, k_shared]`` for every cached token.  ``c_kv`` is the
    shared low-rank KV latent and ``k_shared`` is K3's direct shared key component
    (named the RoPE component in inherited MLA configurations, although K3 applies
    no positional rotation to it).
    """

    l_kv: jax.Array  # [B, max_len, kv_lora_rank + qk_rope_head_dim]
    pos: jax.Array  # scalar int32: number of filled positions


class GatedMultiLatentAttention(nnx.Module):
    """Kimi K3 NoPE Gated MLA, matching the released model factorization.

    Args:
        embed_dim: model width.
        num_heads: number of query/key/value heads.
        q_lora_rank: query compression rank.  ``None`` uses a direct query
            projection, matching the compatibility path in the released model.
        kv_lora_rank: shared compressed KV width.
        qk_nope_head_dim: per-head content-key/query width reconstructed from the
            compressed latent.
        qk_rope_head_dim: per-head direct shared-key/query width.  The inherited
            configuration name is retained for checkpoint compatibility, but K3
            applies no RoPE and therefore treats this as content.
        v_head_dim: per-head value width.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rngs: nnx.Rngs,
        compute_dtype: jnp.dtype = F32,
        output_gate: bool = True,
        rms_eps: float = 1e-6,
    ):
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if q_lora_rank is not None and q_lora_rank <= 0:
            raise ValueError(f"q_lora_rank must be positive or None, got {q_lora_rank}")
        for name, value in (
            ("kv_lora_rank", kv_lora_rank),
            ("qk_nope_head_dim", qk_nope_head_dim),
            ("v_head_dim", v_head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if qk_rope_head_dim < 0:
            raise ValueError(
                f"qk_rope_head_dim must be non-negative, got {qk_rope_head_dim}"
            )

        self.compute_dtype = compute_dtype
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.cache_width = kv_lora_rank + qk_rope_head_dim

        q_width = num_heads * self.q_head_dim
        if q_lora_rank is not None:
            self.q_a_proj = nnx.Linear(
                embed_dim,
                q_lora_rank,
                use_bias=False,
                kernel_init=_XAVIER,
                dtype=compute_dtype,
                param_dtype=F32,
                rngs=rngs,
            )
            self.q_a_norm = RMSNorm(q_lora_rank, eps=rms_eps)
            self.q_b_proj = nnx.Linear(
                q_lora_rank,
                q_width,
                use_bias=False,
                kernel_init=_XAVIER,
                dtype=compute_dtype,
                param_dtype=F32,
                rngs=rngs,
            )
            self.q_proj = None
        else:
            self.q_a_proj = None
            self.q_a_norm = None
            self.q_b_proj = None
            self.q_proj = nnx.Linear(
                embed_dim,
                q_width,
                use_bias=False,
                kernel_init=_XAVIER,
                dtype=compute_dtype,
                param_dtype=F32,
                rngs=rngs,
            )

        # W_DKV produces the shared compressed latent and the direct shared key.
        self.kv_a_proj = nnx.Linear(
            embed_dim,
            self.cache_width,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.kv_a_norm = RMSNorm(kv_lora_rank, eps=rms_eps)

        # W_UK and W_UV reconstruct distinct per-head content keys and values.
        self.kv_b_proj = nnx.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

        value_width = num_heads * v_head_dim
        self.gate_proj = (
            nnx.Linear(
                embed_dim,
                value_width,
                use_bias=False,
                kernel_init=_XAVIER,
                dtype=compute_dtype,
                param_dtype=F32,
                rngs=rngs,
            )
            if output_gate
            else None
        )
        self.o_proj = nnx.Linear(
            value_width,
            embed_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

    def _project_q(self, x: jax.Array) -> jax.Array:
        """Return queries as [B, H, L, q_head_dim]."""
        B, L, _ = x.shape
        if self.q_lora_rank is not None:
            assert self.q_a_proj is not None
            assert self.q_a_norm is not None
            assert self.q_b_proj is not None
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        else:
            assert self.q_proj is not None
            q = self.q_proj(x)
        return q.reshape(B, L, self.num_heads, self.q_head_dim).swapaxes(1, 2)

    def _project_kv(self, compressed: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Reconstruct per-head keys and values from cached compressed KV data."""
        B, L, _ = compressed.shape
        c_kv = compressed[..., : self.kv_lora_rank]
        k_shared = compressed[..., self.kv_lora_rank :]

        kv = self.kv_b_proj(self.kv_a_norm(c_kv))
        kv = kv.reshape(
            B,
            L,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).swapaxes(1, 2)
        k_content, values = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)

        # One direct key vector is shared by all heads.  No rotary transform is
        # applied: these dimensions are content-only in K3's NoPE MLA.
        k_shared = k_shared[:, None, :, :]
        k_shared = jnp.broadcast_to(
            k_shared, (B, self.num_heads, L, self.qk_rope_head_dim)
        )
        keys = jnp.concatenate((k_content, k_shared), axis=-1)
        return keys, values

    def _gated_out(self, o_tilde: jax.Array, x: jax.Array) -> jax.Array:
        """K3 Eq. 7, with the gate in full H*v_head_dim value space."""
        if self.gate_proj is not None:
            o_tilde = jax.nn.sigmoid(self.gate_proj(x)) * o_tilde
        return self.o_proj(o_tilde)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Full causal attention.  x: [B, L, d] -> [B, L, d]."""
        B, L, _ = x.shape
        queries = self._project_q(x)
        compressed = self.kv_a_proj(x)
        keys, values = self._project_kv(compressed)

        logits = jnp.einsum("bhqd,bhkd->bhqk", queries, keys).astype(F32)
        logits = logits / jnp.sqrt(self.q_head_dim)
        causal_mask = jnp.tril(jnp.ones((L, L), dtype=bool))
        logits = jnp.where(causal_mask[None, None], logits, -jnp.inf)

        probs = jax.nn.softmax(logits, axis=-1).astype(values.dtype)
        weighted = jnp.einsum("bhqk,bhkd->bhqd", probs, values)
        weighted = weighted.swapaxes(1, 2).reshape(
            B, L, self.num_heads * self.v_head_dim
        )
        return self._gated_out(weighted, x)

    def init_cache(self, batch_size: int, max_len: int, dtype=None) -> MLACache:
        """Allocate the compressed ``[c_kv, k_shared]`` streaming cache."""
        return MLACache(
            l_kv=jnp.zeros(
                (batch_size, max_len, self.cache_width),
                dtype or self.compute_dtype,
            ),
            pos=jnp.array(0, jnp.int32),
        )

    def step(self, x: jax.Array, cache: MLACache) -> tuple[jax.Array, MLACache]:
        """Cached prefill or decode using the same projections as ``__call__``."""
        B, L, _ = x.shape
        max_len = cache.l_kv.shape[1]
        new_pos = cache.pos + L

        queries = self._project_q(x)
        compressed_new = self.kv_a_proj(x)
        compressed = jax.lax.dynamic_update_slice(
            cache.l_kv,
            compressed_new.astype(cache.l_kv.dtype),
            (0, cache.pos, 0),
        )
        keys, values = self._project_kv(compressed)

        logits = jnp.einsum("bhqd,bhkd->bhqk", queries, keys).astype(F32)
        logits = logits / jnp.sqrt(self.q_head_dim)
        q_pos = cache.pos + jnp.arange(L)
        k_pos = jnp.arange(max_len)
        mask = k_pos[None, :] <= q_pos[:, None]
        logits = jnp.where(mask[None, None], logits, -jnp.inf)

        probs = jax.nn.softmax(logits, axis=-1).astype(values.dtype)
        weighted = jnp.einsum("bhqk,bhkd->bhqd", probs, values)
        weighted = weighted.swapaxes(1, 2).reshape(
            B, L, self.num_heads * self.v_head_dim
        )
        return self._gated_out(weighted, x), MLACache(compressed, new_pos)


# Compatibility alias for callers that imported the previous class name.  The
# implementation is no longer grouped-query attention.
GroupedQueryLatentAttention = GatedMultiLatentAttention
