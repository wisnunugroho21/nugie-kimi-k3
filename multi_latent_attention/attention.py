"""Gated NoPE Multi-head Latent Attention (MLA) — the FULL-attention token mixer
of our Kimi K3 recreation.

In the K3 hybrid (inherited from Kimi Linear), 1 of every 4 layers is ordinary
softmax attention; this module is that layer:

  * MLA (DeepSeek-V2 lineage): keys/values live in a small shared low-rank LATENT,
    so the decode-time cache stores one latent vector per position instead of full
    K and V — the whole point of MLA is that tiny KV cache.
  * NoPE — NO positional encoding of any kind. The linear-attention layers (KDA
    in K3; GDN-2 here) already encode position implicitly through their
    recurrence, so the hybrid drops RoPE from its full-attention layers entirely
    (Kimi Linear Sec. 3.3, "NoPE"; K3 keeps the recipe).
  * Written in the ABSORBED form (see the class docstring): with no RoPE in the
    way, the K/V up-projections fold into the neighboring matrices exactly, so the
    latent itself serves as both K and V and never gets up-projected at runtime.
  * GATED MLA (Kimi K3: "Gated MLA improves attention selectivity"): a head-wise
    sigmoid output gate applied to the attention output right before the final
    (absorbed) output projection,

        o_gated = sigmoid(W_g x) ⊙ o

    This is the selectivity gate of the "Gated Attention" lineage
    (arXiv:2505.06708 — the input-conditioned, head-specific elementwise output
    gate Kimi K2 adopted, which suppresses attention sinks and massive
    activations). K3's exact gate formulation awaits its technical report; this
    is the lineage-faithful reading of the blog's description. Behind the
    `gated` flag (the model config enables it by default via mla_gated).

Two paths, same math: `__call__` for full-sequence training (causal-masked matrix
attention) and `step` for streaming decode (append the new latent to a preallocated
cache, attend over it).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

# App. D.5: Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale = gain² =
# 2^{-5}), replacing Flax NNX's default Linear kernel init. Biases stay at zero.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")

F32 = jnp.float32


class MLACache(NamedTuple):
    """Streaming KV cache for the MLA layer. Thanks to MLA we cache only the small
    COMPRESSED latent `l_kv` (one latent serves as BOTH K and V — see below), in a
    preallocated [B, max_len, Hkv*Dh] buffer written at position `pos`. Unlike GDN-2's
    fixed-size state, this GROWS with context: these full-attention layers are exactly
    the ones that pay the long-context KV-cache cost in the hybrid (3:1 keeps them few)."""

    l_kv: jax.Array  # [B, max_len, num_kv_heads*head_dim]  preallocated latent buffer
    pos: jax.Array  # scalar int32: number of filled positions so far


class GroupedQueryLatentAttention(nnx.Module):
    """Grouped-Query attention over a low-rank KV *latent*, in MLA "absorbed" form.

    This is NoPE (no rotary embeddings) Multi-head Latent Attention written in its
    matrix-absorbed form, fused with GQA-style KV-head sharing. Each of the three
    projections folds together two of the usual MLA matrices:

        w_q_uk : W_Q  . W_UK   -> queries are produced *directly* in the
                                  compressed K space, so they can dot against the
                                  latent without an explicit key up-projection.
        w_dkv  : W_DKV          -> down-projects x to the shared KV latent (c_kv).
        w_uv_o : W_UV . W_O     -> up-projects the value latent and applies the
                                  output projection in a single matmul.

    Key consequence: because there is no RoPE, W_UK and W_UV can be absorbed away
    *exactly*, and in the compressed latent space the keys and the values are the
    same tensor. That is why a single `l_kv` plays the role of BOTH K and V below.

    Note: `head_dim` here is the per-head latent (rank) dimension, not a
    conventional attention head width.
    """

    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rngs: nnx.Rngs,
        compute_dtype: jnp.dtype = F32,
        gated: bool = False,
    ):
        dims = {
            "embed_dim": embed_dim,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        invalid = [name for name, value in dims.items() if value <= 0]
        if invalid:
            raise ValueError(f"Attention dimensions must be positive: {', '.join(invalid)}")
        # Matmul dtype for the projections (bf16 on H200); the QK^T / softmax / AV
        # core is upcast to fp32 below regardless, for a stable attention distribution.
        self.compute_dtype = compute_dtype
        # GQA constraint: every KV (latent) head must serve a whole number of
        # query heads, so that `repeat` below tiles the latent evenly.
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # How many query heads share each KV/latent head (the GQA group size).
        self.group_size = num_q_heads // num_kv_heads

        d_q = num_q_heads * head_dim  # total width of the query projection
        d_kv = num_kv_heads * head_dim  # total width of the (shared) KV latent

        # W_Q . W_UK absorbed: x -> queries already living in the latent K space.
        self.w_q_uk = nnx.Linear(
            embed_dim,
            d_q,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

        # W_DKV: x -> low-rank KV latent c_kv (one latent per KV head).
        self.w_dkv = nnx.Linear(
            embed_dim,
            d_kv,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

        # W_UV . W_O absorbed: value-latent -> up-projected, output-projected.
        self.w_uv_o = nnx.Linear(
            d_q,
            embed_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

        # Gated MLA (Kimi K3): sigmoid output gate — sigmoid(W_g x) ⊙ o —
        # applied to the per-head attention output BEFORE w_uv_o (the G1
        # "before out-projection" position of the Gated Attention paper,
        # head-specific and elementwise over the d_q gate width). With the
        # Xavier gain-2^{-2.5} init the gate starts near sigmoid(0) = 0.5: a
        # uniform half-open gate, so the init-time behavior is a benignly
        # rescaled ungated MLA.
        self.gated = gated
        if gated:
            self.w_gate = nnx.Linear(
                embed_dim,
                d_q,
                use_bias=False,
                kernel_init=_XAVIER,
                dtype=compute_dtype,
                param_dtype=F32,
                rngs=rngs,
            )

    def _output_gate(self, o: jax.Array, x: jax.Array) -> jax.Array:
        """sigmoid(W_g x) ⊙ o. o: [B, L, Hq*Dh] attention output, x: [B, L,
        embed_dim] block input. The sigmoid is taken in fp32 so training and
        decode resolve it identically, then returns to o's dtype for the
        (possibly bf16) w_uv_o matmul."""
        g = jax.nn.sigmoid(self.w_gate(x).astype(F32))
        return (g * o.astype(F32)).astype(o.dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, T, embed_dim)
        batch_size, seq_length, _ = x.shape

        # --- Queries (already in the compressed K space via the absorbed W_UK) ---
        q_latent = self.w_q_uk(x)  # (B, T, num_q_heads * head_dim)

        # Split the flat projection into per-head latent vectors.
        q_reshaped = q_latent.reshape(
            batch_size, seq_length, self.num_q_heads, self.head_dim
        )  # (B, T, Hq, Dh)

        # Move the head axis next to batch for batched matmuls: (B, Hq, T, Dh)
        q_heads = q_reshaped.swapaxes(1, 2)

        # --- Shared KV latent (serves as both keys and values) ---
        l_kv = self.w_dkv(x)  # (B, T, num_kv_heads * head_dim)

        l_kv_reshaped = l_kv.reshape(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        )  # (B, T, Hkv, Dh)

        l_kv_heads = l_kv_reshaped.swapaxes(1, 2)  # (B, Hkv, T, Dh)

        # GQA tiling: repeat each latent head `group_size` times so it lines up
        # with the query heads. `repeat` interleaves, so KV head i feeds query
        # heads [i*group_size : (i+1)*group_size]. Result: (B, Hq, T, Dh).
        # (This materializes the full Hq KV stack; broadcasting would save memory
        # but materializing keeps the einsums simple.)
        l_kv_repeated = l_kv_heads.repeat(self.group_size, axis=1)

        # --- Attention scores: Q . K^T, contracting the latent feature dim `d` ---
        # 'd' is shared (contracted); 'k' indexes key/latent positions (kept).
        qk_t = jnp.einsum("bhqd, bhkd -> bhqk", q_heads, l_kv_repeated)  # (B, Hq, T, T)

        # Scale by sqrt of the latent per-head dim. Upcast to fp32 so the masking,
        # softmax max/exp/sum are stable even when the projections ran in bf16.
        scaled_logits = qk_t.astype(F32) / jnp.sqrt(self.head_dim)

        # Causal mask (True = keep): future positions -> -inf so they vanish under
        # softmax. Built at trace time from the actual sequence length — under jit
        # this is a compile-time constant (folded by XLA), so nothing is stored in
        # the module state or in checkpoints. Safe to use -inf because the diagonal
        # is always kept (no fully-masked rows -> the softmax cannot NaN).
        causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
        scaled_logits = jnp.where(causal_mask[None, None], scaled_logits, -jnp.inf)

        # Softmax over the key axis -> per-query attention distribution (fp32), then
        # back to the compute dtype for the (bf16) weighted-sum matmul below.
        a = jax.nn.softmax(scaled_logits, axis=-1).astype(l_kv_repeated.dtype)  # (B, Hq, T, T)

        # --- Weighted sum of value-latents ---
        # 'k' is shared between the weights and the value positions, so it is the
        # contracted axis (the actual attention sum); 'd' is the kept feature dim.
        # Because keys and values are the same latent, l_kv_repeated reappears here.
        weighted_heads = jnp.einsum("bhqk, bhkd -> bhqd", a, l_kv_repeated)  # (B, Hq, T, Dh)

        # Move head axis back and flatten heads: (B, T, Hq, Dh) -> (B, T, Hq*Dh)
        weighted_reshaped = weighted_heads.swapaxes(1, 2)
        weighted_latents = weighted_reshaped.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )

        # Gated MLA (K3): sigmoid output gate before the absorbed out-projection.
        if self.gated:
            weighted_latents = self._output_gate(weighted_latents, x)

        # Absorbed W_UV . W_O: up-project the value latent and output-project.
        output = self.w_uv_o(weighted_latents)  # (B, T, embed_dim)

        return output

    # ----------------------------------------------------------------------- #
    #  Streaming / inference.  Same softmax attention, but the KV latents of past
    #  positions are read from a preallocated cache instead of recomputed, and the
    #  new positions are written into it.  Use it for prefill (L = prompt length)
    #  and per-token decode (L = 1) alike.
    # ----------------------------------------------------------------------- #
    def init_cache(self, batch_size: int, max_len: int, dtype=None) -> MLACache:
        """Initialize the streaming KV cache for a given batch size and max length.
        The cache is a zero-filled preallocated buffer of shape [B, max_len, Hkv*Dh]
        plus a position counter. dtype defaults to the layer's compute dtype so
        full-sequence and streaming attention use the same projection precision."""
        if batch_size <= 0 or max_len <= 0:
            raise ValueError("batch_size and max_len must be positive")
        d_kv = self.num_kv_heads * self.head_dim
        return MLACache(
            l_kv=jnp.zeros(
                (batch_size, max_len, d_kv),
                self.compute_dtype if dtype is None else dtype,
            ),
            pos=jnp.array(0, jnp.int32),
        )

    def step(self, x: jax.Array, cache: MLACache) -> tuple[jax.Array, MLACache]:
        """Process a new chunk of input x, updating the cache and returning the output.
        x: [B, L, embed_dim]  cache: MLACache with l_kv: [B, max_len, Hkv*Dh], pos: scalar int32"""
        B, L, _ = x.shape
        max_len = cache.l_kv.shape[1]
        if L == 0:
            raise ValueError("x must contain at least one position")
        if L > max_len:
            raise ValueError(f"Input chunk length ({L}) exceeds cache capacity ({max_len})")
        if cache.l_kv.shape[0] != B:
            raise ValueError("x batch size does not match the MLA cache")
        expected_width = self.num_kv_heads * self.head_dim
        if cache.l_kv.shape[2] != expected_width:
            raise ValueError(f"MLA cache width must be {expected_width}, got {cache.l_kv.shape[2]}")
        # Outside a transformed/JIT context the position is concrete, so fail
        # loudly instead of relying on dynamic_update_slice's clamping behavior.
        # generate() performs the same capacity check before entering its jitted
        # decode loop.
        try:
            pos = int(cache.pos)
        except jax.errors.ConcretizationTypeError:
            # A traced position cannot be converted to a Python int. Capacity is
            # validated by generate() before its jitted decode loop instead.
            pass
        else:
            if pos + L > max_len:
                raise ValueError(
                    f"MLA cache capacity exceeded: pos={pos}, chunk={L}, max_len={max_len}"
                )
        new_pos = cache.pos + L

        # Queries for the new positions (already in the compressed K space via W_UK).
        q_heads = (
            self.w_q_uk(x).reshape(B, L, self.num_q_heads, self.head_dim).swapaxes(1, 2)
        )  # (B, Hq, L, Dh)

        # New latents -> write them into the cache buffer at the current position.
        l_new = self.w_dkv(x)  # (B, L, Hkv*Dh)
        l_kv = jax.lax.dynamic_update_slice(
            cache.l_kv, l_new.astype(cache.l_kv.dtype), (0, cache.pos, 0)
        )

        # --- Shared KV latent (serves as both keys and values) ---
        l_kv_heads = l_kv.reshape(B, max_len, self.num_kv_heads, self.head_dim).swapaxes(
            1, 2
        )  # (B, Hkv, max_len, Dh)
        l_kv_rep = l_kv_heads.repeat(self.group_size, axis=1)  # (B, Hq, max_len, Dh)

        # Scores: the L new queries attend over all max_len cached slots. Upcast to
        # fp32 for the mask/softmax exactly as the training __call__ path does — the
        # projections may run in bf16, but the attention distribution must be built
        # in fp32 so prefill/decode stay numerically consistent with training.
        logits = jnp.einsum("bhqd, bhkd -> bhqk", q_heads, l_kv_rep).astype(F32) / jnp.sqrt(
            self.head_dim
        )

        # Causal mask offset by the cache position: query i sits at absolute position
        # pos+i and may attend to slot j iff j <= pos+i.  This also masks the not-yet-
        # filled slots (j >= pos+L > pos+i), so no separate validity mask is needed.
        q_pos = cache.pos + jnp.arange(L)  # (L,)
        k_pos = jnp.arange(max_len)  # (max_len,)
        mask = k_pos[None, :] <= q_pos[:, None]  # (L, max_len)
        logits = jnp.where(mask[None, None], logits, -jnp.inf)

        # Softmax over the key axis -> per-query attention distribution (fp32), then
        # back to the compute dtype for the (bf16) weighted-sum matmul below.
        a = jax.nn.softmax(logits, axis=-1).astype(l_kv_rep.dtype)

        # Weighted sum of value-latents: the same latent serves as both K and V.
        weighted = jnp.einsum("bhqk, bhkd -> bhqd", a, l_kv_rep)  # (B, Hq, L, Dh)
        weighted = weighted.swapaxes(1, 2).reshape(B, L, self.num_q_heads * self.head_dim)

        # Gated MLA (K3): same sigmoid gate as the training path. The gate
        # depends only on the CURRENT positions' x — never on past positions —
        # so streaming needs no extra cache and prefill/decode match training.
        if self.gated:
            weighted = self._output_gate(weighted, x)

        output = self.w_uv_o(weighted)  # (B, L, embed_dim)
        return output, MLACache(l_kv, new_pos)
