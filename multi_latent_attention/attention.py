"""NoPE Gated Multi-head Latent Attention (MLA) — the FULL-attention token mixer.

In the Kimi K3 hybrid (§2.1) each block is 3 linear-attention layers followed by
1 Gated MLA layer, plus one extra Gated MLA at the end of the backbone. This
module is that layer, implementing what K3 §2.1.2 specifies.

WHAT MLA IS — AND THE TRAP
--------------------------
The paper's definition: "MLA compresses the key-value representation of each
token into a low-dimensional latent vector c_t = W_c x_t. Instead of caching
full head-specific keys and values, MLA caches c_t and reconstructs the content
keys and values through learned up-projections during attention computation."

Two words there carry the whole design and are easy to lose:

  * SHARED — there is ONE latent per token, not one per head. Every head sees
    all of it.
  * HEAD-SPECIFIC, RECONSTRUCTED — each head owns learned up-projections W_UK^h
    and W_UV^h that build ITS key and ITS value out of that whole latent.

The cache therefore shrinks to `kv_lora_rank` numbers per token while each head
still gets a full-rank, individually-learned key and value.

The trap: slicing the latent into per-head blocks also shrinks the cache, and
looks superficially similar — but it forces W_UK/W_UV block-diagonal and is
grouped-query attention over a low-rank cache, not MLA. It pays MLA's memory and
gets GQA's expressiveness. `_project_kv` below does the real thing: `kv_b_proj`
maps the FULL latent to every head's key and value.

K3's THREE SPECIFIC CHOICES (§2.1.2)
------------------------------------
  1. NoPE. No positional encoding of any kind on queries or keys. The linear
     layers carry position through their recurrence, leaving these layers to do
     pure global content matching. The paper notes the payoff: nothing
     positional to retune when extending context — no RoPE base rescaling, no
     YaRN — so it extrapolates to 1M directly.
  2. Gated output, Eq. 7:   y_t = W_o[ Sigmoid(W_g x_t) ⊙ õ_t ]
     where õ_t is the UNGATED MLA OUTPUT — the concatenated per-head value
     outputs, AFTER W_UV and BEFORE W_o. W_g is full rank and reads x_t, so the
     gate is input-dependent and channel-wise. New in K3; Kimi Linear's MLA was
     ungated.
  3. FP32 attention output. "To correct the biased rounding error that arises in
     flash attention ... we keep the attention output in FP32 during training."
     So the probs·V accumulation runs in fp32 here even under bf16 compute.

Two paths, same math: `__call__` for full-sequence training, `step` for
streaming prefill/decode against a preallocated latent cache.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

# App. D.5: Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale =
# gain² = 2^{-5}), replacing Flax NNX's default Linear kernel init.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")

F32 = jnp.float32


class RMSNorm(nnx.Module):
    """RMSNorm applied to a low-rank latent before its up-projection.

    Used on both the query LoRA and the KV latent — the DeepSeek-V2/V3
    arrangement K3 says it "retains". Without it the up-projection sees an input
    whose scale drifts with the down-projection.
    """

    def __init__(self, dim: int, *, eps: float = 1e-6):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,), F32))

    def __call__(self, x: jax.Array) -> jax.Array:
        xf = x.astype(F32)
        xf = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (xf * self.weight[...]).astype(x.dtype)


class MLACache(NamedTuple):
    """Streaming cache for one MLA layer: the compressed latent, nothing else.

    This is the headline property — one `kv_lora_rank`-wide vector per token
    instead of H·(qk_head_dim + v_head_dim) for full per-head K and V. Per-head
    keys and values are never stored; they are rebuilt from this on each step.
    Unlike the linear layers' fixed-size recurrent state this still GROWS with
    context, which is exactly why the hybrid keeps these layers to 1 in 4.
    """

    c_kv: jax.Array  # [B, max_len, kv_lora_rank]  preallocated latent buffer
    pos: jax.Array  # scalar int32: number of filled positions so far


class GatedMultiLatentAttention(nnx.Module):
    """Kimi K3 §2.1.2: NoPE Multi-head Latent Attention with a full-rank output gate.

    Args:
        embed_dim:    model width d.
        num_heads:    attention heads H (paper Table 1: 96).
        kv_lora_rank: width of the shared compressed latent c_t — the ONLY thing
                      cached. Every head reads all of it.
        qk_head_dim:  per-head key/query width, reconstructed from the latent.
        v_head_dim:   per-head value width, reconstructed from the latent.
        q_lora_rank:  rank of the query down-projection; None applies a direct
                      query projection. Queries are never cached, so this is
                      purely a parameter-count choice.
        output_gate:  K3 Eq. 7's gate. True is the K3 behaviour; False gives
                      Kimi Linear's ungated MLA for comparison.

    NOTE ON RoPE: there is deliberately no `qk_rope_head_dim` here. In the
    DeepSeek lineage a separate key component exists ONLY because RoPE cannot be
    folded through an up-projection, so those dimensions must bypass the latent.
    K3 is NoPE (§2.1.2), so that exemption does not apply and the paper describes
    no such path — every key dimension comes from the shared latent.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_head_dim: int,
        v_head_dim: int,
        rngs: nnx.Rngs,
        q_lora_rank: int | None = None,
        compute_dtype: jnp.dtype = F32,
        output_gate: bool = True,
        rms_eps: float = 1e-6,
    ):
        for name, value in (
            ("num_heads", num_heads),
            ("kv_lora_rank", kv_lora_rank),
            ("qk_head_dim", qk_head_dim),
            ("v_head_dim", v_head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if q_lora_rank is not None and q_lora_rank <= 0:
            raise ValueError(f"q_lora_rank must be positive or None, got {q_lora_rank}")

        self.compute_dtype = compute_dtype
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank

        q_width = num_heads * qk_head_dim
        v_width = num_heads * v_head_dim

        # ---- Queries. Optionally low-rank: down -> RMSNorm -> up. ------------
        if q_lora_rank is None:
            self.q_a_proj = None
            self.q_a_norm = None
        else:
            self.q_a_proj = nnx.Linear(
                embed_dim, q_lora_rank, use_bias=False, kernel_init=_XAVIER,
                dtype=compute_dtype, param_dtype=F32, rngs=rngs,
            )
            self.q_a_norm = RMSNorm(q_lora_rank, eps=rms_eps)
        self.q_proj = nnx.Linear(
            embed_dim if q_lora_rank is None else q_lora_rank,
            q_width, use_bias=False, kernel_init=_XAVIER,
            dtype=compute_dtype, param_dtype=F32, rngs=rngs,
        )

        # ---- W_DKV: x -> the shared compressed latent c_t. The only thing that
        # ever enters the cache. ----------------------------------------------
        self.kv_a_proj = nnx.Linear(
            embed_dim, kv_lora_rank, use_bias=False, kernel_init=_XAVIER,
            dtype=compute_dtype, param_dtype=F32, rngs=rngs,
        )
        self.kv_a_norm = RMSNorm(kv_lora_rank, eps=rms_eps)

        # ---- W_UK and W_UV, fused into one projection OF THE FULL LATENT.
        # Output is laid out head-major as [head, qk_head_dim | v_head_dim], so
        # head h's key and value are each a learned function of every latent
        # dimension. This is §2.1.2's "reconstructs the content keys and values
        # through learned up-projections", and it is what makes this MLA rather
        # than GQA over a low-rank cache. --------------------------------------
        self.kv_b_proj = nnx.Linear(
            kv_lora_rank, num_heads * (qk_head_dim + v_head_dim),
            use_bias=False, kernel_init=_XAVIER,
            dtype=compute_dtype, param_dtype=F32, rngs=rngs,
        )

        # ---- Eq. 7: the full-rank, input-dependent output gate, and W_o. -----
        self.gate_proj = (
            nnx.Linear(
                embed_dim, v_width, use_bias=False, kernel_init=_XAVIER,
                dtype=compute_dtype, param_dtype=F32, rngs=rngs,
            )
            if output_gate
            else None
        )
        self.o_proj = nnx.Linear(
            v_width, embed_dim, use_bias=False, kernel_init=_XAVIER,
            dtype=compute_dtype, param_dtype=F32, rngs=rngs,
        )

    # ----------------------------------------------------------------------- #
    def _project_q(self, x: jax.Array) -> jax.Array:
        """x: [B, L, d] -> queries [B, H, L, qk_head_dim]. NoPE: nothing rotates."""
        B, L, _ = x.shape
        q = x if self.q_a_proj is None else self.q_a_norm(self.q_a_proj(x))
        return self.q_proj(q).reshape(
            B, L, self.num_heads, self.qk_head_dim
        ).swapaxes(1, 2)

    def _project_kv(self, c_kv: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Reconstruct per-head keys and values from the SHARED latent.

        c_kv: [B, K, kv_lora_rank] -> (keys [B,H,K,qk], values [B,H,K,v]).

        `kv_b_proj` consumes the whole latent, so every head's key and value
        depend on every latent dimension — the property that distinguishes MLA
        from slicing the latent per head. W_UK and W_UV are also distinct halves
        of that projection, so a head's key and its value are different learned
        functions of c_t, not the same tensor.
        """
        B, K, _ = c_kv.shape
        kv = self.kv_b_proj(self.kv_a_norm(c_kv))
        kv = kv.reshape(
            B, K, self.num_heads, self.qk_head_dim + self.v_head_dim
        ).swapaxes(1, 2)
        return jnp.split(kv, [self.qk_head_dim], axis=-1)

    def _attend(
        self, queries: jax.Array, keys: jax.Array, values: jax.Array, mask: jax.Array
    ) -> jax.Array:
        """Scores -> mask -> softmax -> weighted values. Returns [B, L, H*v].

        The softmax runs in fp32 (stable max/exp/sum even when the projections
        ran in bf16), and — per §2.1.2 — so does the probs·V accumulation that
        produces the attention output. The -inf mask is safe because every query
        keeps at least its own position, so no row is fully masked and the
        softmax cannot NaN.
        """
        B = queries.shape[0]
        L = queries.shape[2]
        logits = jnp.einsum("bhqd,bhkd->bhqk", queries, keys).astype(F32)
        logits = logits / jnp.sqrt(self.qk_head_dim)
        logits = jnp.where(mask, logits, -jnp.inf)
        probs = jax.nn.softmax(logits, axis=-1)

        # FP32 attention output (§2.1.2), not a downcast back to compute_dtype.
        o_tilde = jnp.einsum("bhqk,bhkd->bhqd", probs, values.astype(F32))
        return o_tilde.swapaxes(1, 2).reshape(
            B, L, self.num_heads * self.v_head_dim
        )

    def _gated_out(self, o_tilde: jax.Array, x: jax.Array) -> jax.Array:
        """K3 Eq. 7:  y_t = W_o[ Sigmoid(W_g x_t) ⊙ õ_t ].

        `o_tilde` is the ungated MLA output — per-head value outputs AFTER W_UV,
        concatenated, [B, L, H*v_head_dim] — precisely Eq. 7's õ_t. It arrives in
        fp32 and the gate is evaluated there too (a bf16 sigmoid would cost ~1e-2
        on a factor multiplying every output channel, and GDN-2's identical K3
        gate is computed in fp32); only then is it cast down for the W_o matmul.
        """
        if self.gate_proj is not None:
            o_tilde = jax.nn.sigmoid(self.gate_proj(x).astype(F32)) * o_tilde
        return self.o_proj(o_tilde.astype(self.compute_dtype))

    # ----------------------------------------------------------------------- #
    def __call__(self, x: jax.Array) -> jax.Array:
        """Full causal attention over a sequence. x: [B, L, d] -> [B, L, d].

        The training path, written the way §2.1.2 describes it: compress to the
        latent, reconstruct per-head keys and values from it, attend.
        """
        L = x.shape[1]
        queries = self._project_q(x)
        keys, values = self._project_kv(self.kv_a_proj(x))
        causal = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None]
        return self._gated_out(self._attend(queries, keys, values, causal), x)

    # ----------------------------------------------------------------------- #
    def init_cache(self, batch_size: int, max_len: int, dtype=None) -> MLACache:
        """Preallocate the latent cache.

        The buffer dtype DEFAULTS TO compute_dtype and that default matters — do
        not widen it to fp32 "to be safe". `step` reads the latent back out and
        JAX propagates the wider type through everything downstream, so an fp32
        buffer silently runs decode's whole attention core in fp32 while training
        runs it in bf16, and the two paths stop agreeing. It would also double
        the KV cache, the one thing MLA exists to shrink.
        """
        return MLACache(
            c_kv=jnp.zeros(
                (batch_size, max_len, self.kv_lora_rank),
                dtype or self.compute_dtype,
            ),
            pos=jnp.array(0, jnp.int32),
        )

    def step(self, x: jax.Array, cache: MLACache) -> tuple[jax.Array, MLACache]:
        """Cached prefill or decode. Same projections and same math as `__call__`.

        COST NOTE. The cache is a preallocated [B, max_len, kv_lora_rank] buffer
        and the causal mask is applied after the scores, so the reconstruction
        below runs over the FULL capacity every step — empty slots included. That
        is O(max_len · kv_lora_rank · H · (qk+v)) per step.

        Because K3 is NoPE, W_UK and W_UV can instead be absorbed EXACTLY into
        the query and the output ((q·W_UK)·c^T = q·(W_UK·c)^T and
        probs·(c·W_UV) = (probs·c)·W_UV), which removes the H·(qk+v) factor from
        everything that touches the cache. That is the right optimization for
        long-context decode and it changes no result in exact arithmetic — but it
        reassociates the sums, so it does not reproduce `__call__` bit for bit
        (measured ~1 ULP per layer in bf16). The reconstruction form is kept here
        because §2.1.2 describes reconstruction and because train/decode
        agreement is worth more at this scale than decode throughput.
        """
        B, L, _ = x.shape
        max_len = cache.c_kv.shape[1]
        if L > max_len:
            raise ValueError(f"input length {L} exceeds cache capacity {max_len}")
        new_pos = cache.pos + L

        queries = self._project_q(x)
        c_new = self.kv_a_proj(x)
        c_kv = jax.lax.dynamic_update_slice(
            cache.c_kv, c_new.astype(cache.c_kv.dtype), (0, cache.pos, 0)
        )
        keys, values = self._project_kv(c_kv)

        # Causal mask offset by the cache position: query i sits at absolute
        # position pos+i and may read slot j iff j <= pos+i. This also masks the
        # not-yet-filled slots, so no separate validity mask is needed.
        q_pos = cache.pos + jnp.arange(L)
        k_pos = jnp.arange(max_len)
        mask = (k_pos[None, :] <= q_pos[:, None])[None, None]

        o_tilde = self._attend(queries, keys, values, mask)
        return self._gated_out(o_tilde, x), MLACache(c_kv, new_pos)
