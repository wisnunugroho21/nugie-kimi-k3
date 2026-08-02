"""Small building blocks shared across the Kimi K3 modules.

Nothing here is novel on its own; these are the primitives the paper's equations
are written in terms of. Keeping them in one file means the interesting modules
(kda.py, mla.py, attn_res.py, moe.py) contain only the ideas from the paper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

F32 = jnp.float32


def rms_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Parameter-free RMSNorm: x / sqrt(mean(x^2) + eps).

    Used where the paper applies RMSNorm as a *scale controller* rather than as
    a learned transform — notably inside Attention Residuals (§2.2, Eq. 9),
    where its whole job is to stop a layer with large-magnitude outputs from
    dominating the depth-attention weights. Computed in fp32 regardless of the
    activation dtype, because that is exactly the quantity we cannot afford to
    lose precision on.
    """
    x32 = x.astype(F32)
    return (x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps)).astype(x.dtype)


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Unit-L2 normalisation along the last axis (the "L2Norm" of Eq. 2).

    KDA applies this to queries and keys. Unit-norm keys are what keep the
    delta-rule update `(I - beta k k^T)` a well-behaved (contractive) operator:
    with ||k|| = 1 and beta in (0,1), `I - beta k k^T` has eigenvalues 1 and
    1 - beta, so repeated application never amplifies the state.
    """
    x32 = x.astype(F32)
    return (x32 * jax.lax.rsqrt(jnp.sum(jnp.square(x32), axis=-1, keepdims=True) + eps)).astype(x.dtype)


def swish(x: jax.Array) -> jax.Array:
    """Swish / SiLU: x * sigmoid(x). The activation inside KDA's q/k/v path."""
    return x * jax.nn.sigmoid(x)


def softcap(x: jax.Array, beta: float) -> jax.Array:
    """The smooth cap `beta * tanh(x / beta)` of §2.3.2.

    Two properties the paper leans on (Appendix B):
      * near the origin it is the identity, `beta*tanh(z/beta) = z + O(z^3/beta^2)`
        (Eq. 18), so it does not disturb the small-activation regime;
      * it is bounded by `beta`, and unlike a hard clamp it keeps a nonzero
        gradient everywhere, which the authors found trains better.
    """
    return beta * jnp.tanh(x / beta)


class SiTUGLU(nnx.Module):
    """Sigmoid Tanh Unit GLU (§2.3.2, Eq. 12) — K3's replacement for SwiGLU.

        SiTU-GLU(x) = [ b1*tanh(W_g x / b1) * sigmoid(W_g x) ] * [ b2*tanh(W_u x / b2) ]

    The motivation is numerical, not representational. In SwiGLU both factors
    (`Swish(W_g x)` and `W_u x`) are unbounded, so two coincident large
    coordinates multiply into an activation outlier — a real overflow risk once
    you are training a 2.8T model with low-precision arithmetic. SiTU-GLU
    soft-caps the *linear* factor of the Swish gate and the up branch
    separately, while keeping the sigmoid factor untouched (the sigmoid already
    kills the negative tail). Every output coordinate is then bounded:

        |SiTU-GLU(x)| <= b1 * b2 = 4 * 25 = 100          (Appendix B, Eq. 19)

    and as b1, b2 -> inf it recovers SwiGLU exactly.

    This class is the *activation with its two input projections*; the caller
    supplies the down-projection. That mirrors how the paper writes Eq. 12.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        *,
        beta_gate: float,
        beta_up: float,
        rngs: nnx.Rngs,
    ):
        self.beta_gate = beta_gate
        self.beta_up = beta_up
        self.gate = nnx.Linear(d_in, d_hidden, use_bias=False, rngs=rngs)
        self.up = nnx.Linear(d_in, d_hidden, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        g = self.gate(x)
        u = self.up(x)
        # Note `sigmoid(g)` uses the *uncapped* pre-activation: the cap is applied
        # only to the linear factor of Swish, as written in Eq. 12.
        gate = softcap(g, self.beta_gate) * jax.nn.sigmoid(g)
        return gate * softcap(u, self.beta_up)


class ShortConv(nnx.Module):
    """Causal depthwise 1-D convolution over the token axis ("ShortConv", Eq. 2).

    KDA passes q, k and v through this before the activation. It is a cheap way
    to give each linear-attention head a short, hard-wired local receptive
    field, so the recurrent state does not have to spend capacity on
    "what did the previous few tokens say". Depthwise = one small kernel per
    channel, so the cost is `d * kernel_size` parameters and O(L*d) work.

    Causality is enforced by left-padding with `kernel_size - 1` zeros and
    taking a VALID convolution, so position t only ever sees positions <= t.
    """

    def __init__(self, features: int, kernel_size: int, *, rngs: nnx.Rngs):
        self.kernel_size = kernel_size
        self.features = features
        # [kernel_size, features]; initialised so the conv starts near identity
        # on the current token (last tap = 1, history taps = 0).
        w = jnp.zeros((kernel_size, features), dtype=F32).at[-1].set(1.0)
        self.weight = nnx.Param(w)
        self.bias = nnx.Param(jnp.zeros((features,), dtype=F32))

    def __call__(self, x: jax.Array, cache: jax.Array | None = None):
        """x: [B, T, features] -> [B, T, features].

        If `cache` ([B, kernel_size-1, features], the previous tokens) is given,
        it is prepended instead of zero padding and the updated cache is
        returned alongside the output — that is all a decode step needs.
        """
        pad = self.kernel_size - 1
        if cache is None:
            padded = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))
        else:
            padded = jnp.concatenate([cache, x], axis=1)
        new_cache = padded[:, padded.shape[1] - pad :] if pad > 0 else padded[:, :0]

        # Depthwise conv as an explicit tap-and-sum: T*kernel_size shifted adds.
        # Written this way (rather than via lax.conv) because it makes the
        # causal indexing obvious, and kernel_size is 4.
        T = x.shape[1]
        out = jnp.zeros_like(x, dtype=F32)
        for i in range(self.kernel_size):
            # tap i looks back (kernel_size-1-i) tokens
            out += padded[:, i : i + T].astype(F32) * self.weight[...][i]
        out = out + self.bias[...]
        return out.astype(x.dtype), new_cache


class RMSNormGated(nnx.Module):
    """Learned RMSNorm applied per attention head.

    KDA normalises its recurrent read-out head-wise before the output gate
    (§2.1.1, Eq. 6). Doing it per head rather than across the flattened
    `num_heads * head_dim` vector keeps one head's scale from setting another's.
    """

    def __init__(self, head_dim: int, num_heads: int, *, eps: float = 1e-6, rngs: nnx.Rngs):
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((num_heads, head_dim), dtype=F32))

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: [..., num_heads, head_dim]"""
        return (rms_norm(x, self.eps).astype(F32) * self.scale[...]).astype(x.dtype)
