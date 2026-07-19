"""Verification of the GatedDeltaNet2 layer (gated_deltanet_2/layer.py).

Covers the invariants the layer docstrings promise:
  * the ShortConv single-token decode fast path equals the general conv,
  * the folded GQA recurrence (value width G·d_v) equals the paper's
    repeat-the-key-side-tensors formulation,
  * streaming `step` (prefill + token-by-token decode, crossing the
    chunkwise/recurrent split point) equals the full-sequence `__call__`,
  * `initial_state` / `return_state` segment chaining matches one long call
    when the conv left-context caveat is sidestepped.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from gated_deltanet_2.core import chunkwise_gated_delta_rule_2
from gated_deltanet_2.layer import GatedDeltaNet2, ShortConv

D_MODEL, HEADS, DK, DV, CHUNK = 64, 2, 16, 16, 16
B, L = 2, 80  # L = 5 chunks; step() below also exercises ragged tails


def make_layer(num_v_heads=None, chunk_size=CHUNK, core="centered"):
    return GatedDeltaNet2(
        d_model=D_MODEL, num_heads=HEADS, head_k_dim=DK, head_v_dim=DV,
        num_v_heads=num_v_heads, chunk_size=chunk_size, conv_size=4,
        expanded_erase=True, core=core, rngs=nnx.Rngs(0),
    )


def test_shortconv_decode_fast_path():
    """step() with L=1 (the einsum fast path) equals the general conv."""
    conv = ShortConv(8, kernel_size=4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 1, 8))
    state = jax.random.normal(jax.random.PRNGKey(2), (B, 3, 8))
    y_fast, s_fast = conv.step(x, state)
    y_ref, s_ref = conv._apply(x, state)
    np.testing.assert_allclose(y_fast, y_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(s_fast, s_ref, rtol=1e-6, atol=1e-6)


def test_gqa_folding_equals_repeat_formulation():
    """The grouped recurrence (one call per key head at value width G·d_v)
    equals App. C.1's 'repeat the key-side tensors across the group' reading,
    run per value head with the un-grouped v/w."""
    G = 2
    layer = make_layer(num_v_heads=HEADS * G)
    x = jax.random.normal(jax.random.PRNGKey(3), (B, L, D_MODEL))
    out_folded = layer(x)

    # Rebuild the projected tensors, then run the REPEAT formulation manually.
    q, k, v, g, b, w, _ = layer._project(x, conv_states=None)
    # Un-group v, w: [B, H, L, G*dv] -> [B, Hv, L, dv] with the G group members
    # of key head h at value heads h*G..h*G+G-1 (matching _split_v's layout).
    def ungroup(t):
        return (t.reshape(B, HEADS, L, G, DV).transpose(0, 1, 3, 2, 4)
                 .reshape(B, HEADS * G, L, DV))
    v_r, w_r = ungroup(v), ungroup(w)
    rep = lambda t: jnp.repeat(t, G, axis=1)  # noqa: E731  key-side repeat
    S0 = jnp.zeros((B, HEADS * G, DK, DV))
    o_rep, _ = chunkwise_gated_delta_rule_2(
        rep(q), rep(k), v_r, rep(g), rep(b), w_r, S0,
        chunk_size=CHUNK, core=layer.core)
    # Regroup the per-value-head outputs into the layout _output expects.
    o_grouped = (o_rep.reshape(B, HEADS, G, L, DV).transpose(0, 1, 3, 2, 4)
                      .reshape(B, HEADS, L, G * DV))
    out_repeat = layer._output(o_grouped, x)
    np.testing.assert_allclose(out_folded, out_repeat, rtol=2e-4, atol=2e-4)


def test_step_matches_full_call():
    """Prefill + per-token decode through step() reproduces __call__, including
    a ragged prefill (not a multiple of chunk_size) that crosses the
    chunkwise/recurrent split point."""
    layer = make_layer()
    x = jax.random.normal(jax.random.PRNGKey(4), (B, L, D_MODEL))
    full = layer(x)

    for prefill in (CHUNK * 2, CHUNK * 2 + 5):  # aligned and ragged prefixes
        cache = layer.init_cache(B)
        out_prefill, cache = layer.step(x[:, :prefill], cache)
        outs = [out_prefill]
        for t in range(prefill, L):
            o, cache = layer.step(x[:, t : t + 1], cache)
            outs.append(o)
        streamed = jnp.concatenate(outs, axis=1)
        np.testing.assert_allclose(streamed, full, rtol=2e-4, atol=2e-4)


def test_initial_state_segment_chaining():
    """__call__ with return_state / initial_state chains segments exactly,
    provided the conv left context is warm (here: recompute segment 2's conv
    inputs from the full sequence via step(), whose cache carries both)."""
    layer = make_layer()
    x = jax.random.normal(jax.random.PRNGKey(5), (B, 2 * L, D_MODEL))
    full = layer(x)

    out1, S = layer(x[:, :L], return_state=True)
    # Streaming from a cache seeded with segment 1's state AND conv context
    # must continue exactly; __call__(initial_state=S) alone would differ in
    # the first conv_size-1 tokens (the documented zero-padding caveat).
    cache = layer.init_cache(B)
    _, cache = layer.step(x[:, :L], cache)
    np.testing.assert_allclose(cache.recurrent_state, S, rtol=2e-4, atol=2e-4)
    out2, _ = layer.step(x[:, L:], cache)
    np.testing.assert_allclose(
        jnp.concatenate([out1, out2], axis=1), full, rtol=2e-4, atol=2e-4)
