"""Kimi Delta Attention — §2.1.1 of the Kimi K3 report.

KDA is the *linear* token mixer: 69 of K3's 93 layers use it. Instead of an
attention matrix it keeps a fixed-size associative memory

    S_t  in  R^{d_k x d_v}

that maps key directions to value rows, and reads it with the query,
`o_t = S_t^T q_t`. Cost is O(T) in sequence length and the state does not grow,
which is what makes a 1M-token context affordable.

THE RECURRENCE (Eq. 1)
----------------------
    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

Read it as three edits applied to the memory at every token:

  1. FORGET  `Diag(alpha_t) S_{t-1}` — a *channel-wise* decay. alpha_t is a
     vector in (0,1)^{d_k}, one retention factor per key channel, so the model
     can hold some channels indefinitely while flushing others. This is the
     "channel-wise forget gate" that separates KDA from the plain delta rule.
  2. ERASE   `-beta_t k_t k_t^T (...)` — removes whatever the memory currently
     returns along direction k_t.
  3. WRITE   `+beta_t k_t v_t^T` — writes the new association.

Steps 2 and 3 together are the delta rule: they replace the old value at k_t
with the new one rather than accumulating, with beta_t in (0,1) controlling how
much of the replacement happens. Equivalently, one rank-1 error-correcting step

    S_t = S_decayed + beta_t k_t (v_t - S_decayed^T k_t)^T

which is how the code below writes it.

WHAT K3 CHANGED VERSUS KIMI LINEAR
----------------------------------
  * Eq. 5 — the log-decay is now LOWER-BOUNDED by a scaled sigmoid instead of
    an unbounded negative-softplus. This is a numerics change with a big
    performance payoff; see `_decay` and `kda_chunkwise` for why.
  * Eq. 6 — the output gate is a full-rank input-dependent projection instead
    of a low-rank one.

This file gives two mathematically identical implementations:
  * `kda_recurrent` — Eq. 1 literally, one token at a time. Slow, obviously
    correct, and doubles as the decode step. Used as the test oracle.
  * `kda_chunkwise` — Eqs. 3-4, the form you actually train with: sequential
    across chunks, matmul-parallel within a chunk.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from .config import KimiK3Config
from .layers import F32, RMSNormGated, ShortConv, l2_norm, swish

# ===========================================================================
# The recurrence, two ways
# ===========================================================================


def kda_recurrent(
    q: jax.Array,  # [B, H, T, Dk]
    k: jax.Array,  # [B, H, T, Dk]
    v: jax.Array,  # [B, H, T, Dv]
    alpha: jax.Array,  # [B, H, T, Dk]  channel-wise retention, in (0,1)
    beta: jax.Array,  # [B, H, T]       delta-rule write strength, in (0,1)
    state: jax.Array | None = None,  # [B, H, Dk, Dv]
) -> tuple[jax.Array, jax.Array]:
    """Eq. 1, evaluated token by token. Returns (outputs [B,H,T,Dv], final state).

    This is the definition. Everything else in this file must agree with it.
    """
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    if state is None:
        state = jnp.zeros((B, H, Dk, Dv), dtype=F32)

    def step(S, xs):
        q_t, k_t, v_t, a_t, b_t = xs  # each [B, H, ...]
        S = a_t[..., None] * S  # 1. forget: Diag(alpha_t) S_{t-1}
        pred = jnp.einsum("bhd,bhdv->bhv", k_t, S)  # what memory returns for k_t
        err = v_t - pred  # 2+3. delta rule: correct the error
        S = S + b_t[..., None, None] * k_t[..., :, None] * err[..., None, :]
        o = jnp.einsum("bhd,bhdv->bhv", q_t, S)  # read: o_t = S_t^T q_t
        return S, o

    seq = (
        q.transpose(2, 0, 1, 3).astype(F32),
        k.transpose(2, 0, 1, 3).astype(F32),
        v.transpose(2, 0, 1, 3).astype(F32),
        alpha.transpose(2, 0, 1, 3).astype(F32),
        beta.transpose(2, 0, 1).astype(F32),
    )
    final_state, out = jax.lax.scan(step, state, seq)
    return out.transpose(1, 2, 0, 3), final_state  # [B, H, T, Dv]


def kda_chunkwise(
    q: jax.Array,  # [B, H, T, Dk]
    k: jax.Array,  # [B, H, T, Dk]
    v: jax.Array,  # [B, H, T, Dv]
    alpha: jax.Array,  # [B, H, T, Dk]
    beta: jax.Array,  # [B, H, T]
    state: jax.Array | None = None,  # [B, H, Dk, Dv]
    chunk_size: int = 16,
) -> tuple[jax.Array, jax.Array]:
    """Eqs. 3-4: recurrent across chunks, parallel (matmuls) inside a chunk.

    WHY THIS WORKS.  Fold the cumulative decay into the state by substituting
    `S_i = Diag(gamma_i) P_i`, where `gamma_i = prod_{r<=i} alpha_r` is the
    cumulative retention from the chunk's start (Eq. 3). The decay disappears
    from the recurrence and leaves a pure delta rule on P:

        P_i = P_{i-1} + beta_i * u_i * (v_i - P_{i-1}^T w_i)^T
        with  u_i = k_i / gamma_i     (key rescaled by *reciprocal* decay)
              w_i = k_i * gamma_i

    Unrolling that gives `P_i = P_0 + sum_{r<=i} u_r vtilde_r^T` for some
    "pseudo-values" vtilde, and reading it out with `o_i = P_i^T (gamma_i * q_i)`
    reproduces Eq. 4 exactly:

        A   = Tril[ (Q * Gamma) (K / Gamma)^T ]           intra-chunk weights
        O   = (Gamma * Q) S_chunk      +      A Vtilde
              \\_____ inter-chunk _____/    \\_ intra-chunk _/

    The diagonal of the Tril mask is KEPT: `o_i` reads the state *after* token
    i's own write.

    THE UT TRANSFORM.  vtilde is defined recursively (token r's write depends on
    what tokens < r already wrote), but the recursion is linear and lower
    triangular, so it is one triangular solve. Writing M_{rs} = beta_r (w_r . u_s)
    for s < r:

        (I + M) Vtilde = diag(beta) V - diag(beta) W S_chunk
        =>  Vtilde = U - W_ut S_chunk        <- the paper's U[t], W[t]

    NUMERICS — and why Eq. 5's lower bound matters.  The `K / Gamma` term
    rescales keys by the *reciprocal* cumulative decay. Gamma is a product of
    numbers in (0,1), so 1/Gamma grows without bound and overflows: this is the
    exact bottleneck §2.1.1 describes. K3 bounds the per-step log-decay below by
    g_min = -5, so over a 16-token chunk the cumulative log-decay lies in
    (-80, 0) and 1/Gamma <= e^80, comfortably inside the BF16/FP32 exponent
    range. That is why `chunk_size` defaults to 16, and why K3 can run *every*
    tile — diagonal included — as a dense matmul, where Kimi Linear needed an
    explicit position-pair computation on the diagonal tiles.

    (A production kernel uses a larger primary chunk subdivided into 16-token
    secondary tiles, and can also center the exponents by G_C/2 to double the
    safe range. Both are optimisations of exactly this algebra.)
    """
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    C = chunk_size

    # Pad the sequence to a whole number of chunks. Padding tokens are made
    # inert: alpha=1 (no decay), beta=0 (no write), q=k=v=0 (no read/write).
    pad = (-T) % C
    if pad:
        zeros = lambda x, w: jnp.pad(x, ((0, 0), (0, 0), (0, w), (0, 0)))
        q, k, v = zeros(q, pad), zeros(k, pad), zeros(v, pad)
        alpha = jnp.pad(alpha, ((0, 0), (0, 0), (0, pad), (0, 0)), constant_values=1.0)
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad)))
    N = (T + pad) // C  # number of chunks

    to_chunks = lambda x, d: x.reshape(B, H, N, C, d).astype(F32)
    q, k, v = to_chunks(q, Dk), to_chunks(k, Dk), to_chunks(v, Dv)
    alpha = to_chunks(alpha, Dk)
    beta = beta.reshape(B, H, N, C).astype(F32)

    # Eq. 3: cumulative decay within each chunk, in log space.
    #   G[..., r, :] = sum_{i<=r} log alpha_i          (inclusive)
    #   Gamma        = exp(G)
    log_alpha = jnp.log(alpha)
    G = jnp.cumsum(log_alpha, axis=-2)  # [B,H,N,C,Dk], in (-5C, 0]
    Gamma = jnp.exp(G)
    inv_Gamma = jnp.exp(-G)  # <= e^{5C}: the term Eq. 5 bounds
    G_last = G[..., -1:, :]  # cumulative decay across the whole chunk

    q_g = q * Gamma  # Q * Gamma
    k_ig = k * inv_Gamma  # K / Gamma      (== u_r)
    k_g = k * Gamma  # K * Gamma      (== w_r)

    # --- UT transform -----------------------------------------------------
    # M_{rs} = beta_r (w_r . u_s) for s < r, strictly lower triangular.
    strict_lower = jnp.tril(jnp.ones((C, C), dtype=bool), -1)
    M = jnp.where(strict_lower, jnp.einsum("...rd,...sd->...rs", k_g, k_ig), 0.0)
    M = M * beta[..., :, None]
    eye = jnp.eye(C, dtype=F32)

    if state is None:
        state = jnp.zeros((B, H, Dk, Dv), dtype=F32)

    def scan_chunk(S, xs):
        q_g_t, k_ig_t, k_g_t, v_t, beta_t, M_t, Gl_t = xs

        # Vtilde = (I + M)^{-1} [ diag(beta) V - diag(beta) (K*Gamma) S ]
        rhs = beta_t[..., :, None] * (v_t - jnp.einsum("...cd,...dv->...cv", k_g_t, S))
        v_tilde = jax.scipy.linalg.solve_triangular(eye + M_t, rhs, lower=True, unit_diagonal=True)

        # Eq. 4.
        A = jnp.tril(jnp.einsum("...rd,...sd->...rs", q_g_t, k_ig_t))  # diagonal kept
        o = jnp.einsum("...cd,...dv->...cv", q_g_t, S) + A @ v_tilde

        # Carry the state to the next chunk:
        #   S_next = Diag(gamma_C) (S + U^T Vtilde)
        # Folding gamma_C *inside* the sum scales each key by
        # exp(G_C - G_r) = gamma^{r+1 -> C} <= 1, so the cross-chunk update
        # never touches the large reciprocal factor.
        gamma_C = jnp.exp(Gl_t)  # [..., 1, Dk]
        S_next = gamma_C[..., 0, :, None] * S + jnp.einsum(
            "...cd,...cv->...dv", k_ig_t * gamma_C, v_tilde
        )
        return S_next, o

    seq = tuple(
        jnp.moveaxis(x, 2, 0)
        for x in (q_g, k_ig, k_g, v, beta, M, jnp.broadcast_to(G_last, (B, H, N, 1, Dk)))
    )
    final_state, out = jax.lax.scan(scan_chunk, state, seq)  # out: [N,B,H,C,Dv]
    out = jnp.moveaxis(out, 0, 2).reshape(B, H, N * C, Dv)
    return out[:, :, :T], final_state


# ===========================================================================
# The module: Eq. 2 (parameterisation), Eq. 5 (decay), Eq. 6 (output gate)
# ===========================================================================


class KDACache(NamedTuple):
    """Decode-time state. Fixed size — this is the point of linear attention.

    `state` is the S_t of Eq. 1; `conv` holds the last `kernel_size-1` inputs of
    each ShortConv. Neither grows with context length, in contrast to the MLA
    layers' KV cache.
    """

    state: jax.Array  # [B, H, Dk, Dv]
    conv_q: jax.Array
    conv_k: jax.Array
    conv_v: jax.Array


class KimiDeltaAttention(nnx.Module):
    """KDA layer: Eq. 2 -> Eq. 5 -> the recurrence -> Eq. 6."""

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        d, H, Dh = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        self.H, self.Dh = H, Dh
        inner = H * Dh

        # --- Eq. 2: q, k = L2Norm(Swish(ShortConv(W x))); v = Swish(ShortConv(W x))
        self.q_proj = nnx.Linear(d, inner, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(d, inner, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(d, inner, use_bias=False, rngs=rngs)
        self.q_conv = ShortConv(inner, cfg.kda_conv_size, rngs=rngs)
        self.k_conv = ShortConv(inner, cfg.kda_conv_size, rngs=rngs)
        self.v_conv = ShortConv(inner, cfg.kda_conv_size, rngs=rngs)

        # --- Eq. 2: beta_t = Sigmoid(W_beta x_t), one scalar per head.
        self.beta_proj = nnx.Linear(d, H, use_bias=False, rngs=rngs)

        # --- Eq. 2: z_t = W_up W_down x_t + b_alpha, a *low-rank* projection to a
        # per-key-channel decay logit, plus a head-specific bias. Low rank because
        # d_model -> num_heads*head_dim at full rank would cost as much as the
        # q/k/v projections combined, for what is only a gating signal.
        self.alpha_down = nnx.Linear(d, cfg.kda_alpha_rank, use_bias=False, rngs=rngs)
        self.alpha_up = nnx.Linear(cfg.kda_alpha_rank, inner, use_bias=False, rngs=rngs)
        # b_alpha init: following Mamba-2 / GDN practice, start with very little
        # forgetting. With A_h = 0, alpha = exp(g_min * sigmoid(z)), so a bias in
        # [-8.5, -3.8] puts the initial retention in roughly [0.90, 0.999] — the
        # memory holds on to almost everything and learns to forget.
        key = rngs.params()
        self.alpha_bias = nnx.Param(jax.random.uniform(key, (H, Dh), minval=-8.5, maxval=-3.8, dtype=F32))
        # A_h, the learnable per-head log-scale of Eq. 5, initialised to 0.
        self.log_scale = nnx.Param(jnp.zeros((H,), dtype=F32))

        # --- Eq. 6: y = W_o [ Sigmoid(W_g x) * RMSNorm(o) ]
        self.o_norm = RMSNormGated(Dh, H, eps=cfg.rms_norm_eps, rngs=rngs)
        # FULL-RANK gate — this is one of K3's two changes to KDA. Kimi Linear
        # used a low-rank factorisation here.
        self.gate_proj = nnx.Linear(d, inner, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(inner, d, use_bias=False, rngs=rngs)

    # ------------------------------------------------------------------
    def _decay(self, x: jax.Array) -> jax.Array:
        """Eq. 5 — the lower-bounded decay. Returns alpha in (e^{g_min}, 1)."""
        B, T, _ = x.shape
        z = self.alpha_up(self.alpha_down(x)).reshape(B, T, self.H, self.Dh)
        z = z + self.alpha_bias[...]  # head-specific bias b_alpha^h
        # g = g_min * Sigmoid(e^{A_h} z)  in (g_min, 0)
        #
        # Compare Kimi Linear: g = -e^{A} Softplus(z) in (-inf, 0). Both are
        # negative and monotone in z, but the sigmoid saturates at g_min instead
        # of running off to -inf. Bounding it is what keeps 1/Gamma finite in
        # `kda_chunkwise`, and hence what lets every tile be a dense matmul.
        scale = jnp.exp(self.log_scale[...])[None, None, :, None]  # e^{A_h}
        g = self.cfg.kda_g_min * jax.nn.sigmoid(scale * z.astype(F32))
        return jnp.exp(g)  # alpha

    def _inputs(self, x: jax.Array, cache: KDACache | None):
        """Eq. 2: project -> ShortConv -> Swish -> (L2Norm for q, k)."""
        B, T, _ = x.shape
        cq, ck, cv = (cache.conv_q, cache.conv_k, cache.conv_v) if cache else (None, None, None)
        q, cq = self.q_conv(self.q_proj(x), cq)
        k, ck = self.k_conv(self.k_proj(x), ck)
        v, cv = self.v_conv(self.v_proj(x), cv)

        split = lambda t: t.reshape(B, T, self.H, self.Dh).transpose(0, 2, 1, 3)  # [B,H,T,Dh]
        q = l2_norm(split(swish(q)))
        k = l2_norm(split(swish(k)))
        v = split(swish(v))  # v is NOT L2-normalised: it carries magnitude
        beta = jax.nn.sigmoid(self.beta_proj(x)).transpose(0, 2, 1)  # [B,H,T]
        alpha = self._decay(x).transpose(0, 2, 1, 3)  # [B,H,T,Dh]
        return q, k, v, alpha, beta, (cq, ck, cv)

    def _output(self, o: jax.Array, x: jax.Array) -> jax.Array:
        """Eq. 6: head-wise RMSNorm, then a full-rank input-dependent gate."""
        B, _, T, _ = o.shape
        o = self.o_norm(o.transpose(0, 2, 1, 3))  # [B,T,H,Dh]
        gate = jax.nn.sigmoid(self.gate_proj(x)).reshape(B, T, self.H, self.Dh)
        return self.out_proj((gate * o).reshape(B, T, self.H * self.Dh))

    # ------------------------------------------------------------------
    def __call__(self, x: jax.Array, *, use_chunkwise: bool = True) -> jax.Array:
        """Training / prefill forward pass. x: [B, T, d] -> [B, T, d]."""
        q, k, v, alpha, beta, _ = self._inputs(x, None)
        core = kda_chunkwise if use_chunkwise else kda_recurrent
        kwargs = {"chunk_size": self.cfg.kda_chunk_size} if use_chunkwise else {}
        o, _ = core(q, k, v, alpha, beta, None, **kwargs)
        return self._output(o, x)

    def init_cache(self, batch: int) -> KDACache:
        pad = self.cfg.kda_conv_size - 1
        inner = self.H * self.Dh
        z = lambda: jnp.zeros((batch, pad, inner), dtype=F32)
        return KDACache(jnp.zeros((batch, self.H, self.Dh, self.Dh), dtype=F32), z(), z(), z())

    def step(self, x: jax.Array, cache: KDACache) -> tuple[jax.Array, KDACache]:
        """One decode token. x: [B, 1, d]. O(1) memory in context length."""
        q, k, v, alpha, beta, (cq, ck, cv) = self._inputs(x, cache)
        o, state = kda_recurrent(q, k, v, alpha, beta, cache.state)
        return self._output(o, x), KDACache(state, cq, ck, cv)
