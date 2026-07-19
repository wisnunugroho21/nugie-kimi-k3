"""Verification of the Gated Delta Rule-2 recurrence cores (gated_deltanet_2/core.py).

Every chunkwise core is checked against TWO references:
  1. `_recurrent_single` — the token-by-token JAX scan of Eq. 9/29, and
  2. an independent float64 NumPy implementation of the same recurrence
     (written directly from the (I − k eᵀ) Diag(α) S + k zᵀ form of Eq. 10/29,
     sharing no code with core.py).

Decay strengths are chosen per core: the "faithful" and "stacked_rhs" cores keep
the paper's literal exp(-G) factor and are only exact while the per-chunk
cumulative log-decay |G_C| stays under ~88; "centered" doubles that; "pairwise"
and "subchunking" have no range limit and are additionally tested under decay
strong enough to overflow the literal factorizations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gated_deltanet_2.core import (
    _recurrent_single,
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)

L, DK, DV, CHUNK = 128, 16, 24, 32


def ref_recurrence_f64(q, k, v, g, b, w, S0):
    """Independent float64 oracle: S_t = (I − k_t e_tᵀ) Diag(α_t) S_{t−1} + k_t z_tᵀ,
    o_t = S_tᵀ q_t. One (L, ...) head; no code shared with core.py."""
    q, k, v, g, b, w, S = (np.asarray(a, np.float64) for a in (q, k, v, g, b, w, S0))
    out = np.zeros((len(q), v.shape[-1]))
    for t in range(len(q)):
        S = np.exp(g[t])[:, None] * S  # Diag(α) S
        e = b[t] * k[t]
        S = S - np.outer(k[t], S.T @ e)  # (I − k eᵀ) applied to the decayed state
        S = S + np.outer(k[t], w[t] * v[t])  # + k zᵀ
        out[t] = S.T @ q[t]
    return out, S


def make_inputs(key, decay_strength: float, batched: bool = False):
    """Random head inputs with g ≤ 0 scaled so per-chunk |G_C| ≈ CHUNK*decay_strength/2.

    q and k are L2-normalized exactly as the layer front-end does (Sec. 3.5):
    with ‖k‖ = 1 and b ∈ [0, 2] the erase factor (I − b k kᵀ) has eigenvalues in
    [−1, 1], keeping the recurrence stable. Unnormalized keys make the delta rule
    explosive over long sequences — an fp32-overflow artifact of the INPUTS, not
    a property of the cores under test."""
    ks = jax.random.split(key, 7)
    shape = (2, 3) if batched else ()  # [B, H] leading axes for the public API
    q, k = (jax.random.normal(ks[i], (*shape, L, DK)) for i in (0, 1))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v, w = (jax.random.normal(ks[2], (*shape, L, DV)),
            jax.nn.sigmoid(jax.random.normal(ks[3], (*shape, L, DV))))
    b = 2.0 * jax.nn.sigmoid(jax.random.normal(ks[4], (*shape, L, DK)))
    g = -decay_strength * jax.random.uniform(ks[5], (*shape, L, DK))
    S0 = 0.1 * jax.random.normal(ks[6], (*shape, DK, DV))
    return q, k, v, g, b, w, S0


@pytest.mark.parametrize("decay_strength", [0.05, 1.0])
def test_recurrent_matches_f64_oracle(decay_strength):
    q, k, v, g, b, w, S0 = make_inputs(jax.random.PRNGKey(0), decay_strength)
    o, S = _recurrent_single(q, k, v, g, b, w, S0)
    o_ref, S_ref = ref_recurrence_f64(q, k, v, g, b, w, S0)
    np.testing.assert_allclose(o, o_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(S, S_ref, rtol=1e-4, atol=1e-4)


# Per-core decay strengths kept inside each core's documented safe range.
# mean(|g|) = strength/2, so per-chunk |G_C| ≈ CHUNK * strength / 2.
CORE_DECAYS = {
    "faithful": [0.05, 1.0, 4.0],       # |G_C| up to ~64  (< 88 limit)
    "stacked_rhs": [0.05, 1.0, 4.0],    # same literal factors, same limit
    "centered": [0.05, 1.0, 9.0],       # |G_C| up to ~144 (< 176 limit)
    "pairwise": [0.05, 1.0, 9.0, 30.0],     # no limit; 30 -> |G_C| ~ 480
    "subchunking": [0.05, 1.0, 9.0, 30.0],  # no limit
}


@pytest.mark.parametrize(
    "core,decay_strength",
    [(c, d) for c, decays in CORE_DECAYS.items() for d in decays],
)
def test_chunkwise_cores_match_oracles(core, decay_strength):
    q, k, v, g, b, w, S0 = make_inputs(jax.random.PRNGKey(1), decay_strength)
    o, S = chunkwise_gated_delta_rule_2(
        q[None, None], k[None, None], v[None, None], g[None, None],
        b[None, None], w[None, None], S0[None, None],
        chunk_size=CHUNK, core=core, sub_chunk_size=8,
    )
    o, S = o[0, 0], S[0, 0]

    o_rec, S_rec = _recurrent_single(q, k, v, g, b, w, S0)
    o_ref, S_ref = ref_recurrence_f64(q, k, v, g, b, w, S0)

    np.testing.assert_allclose(o, o_rec, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(o, o_ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(S, S_ref, rtol=2e-3, atol=2e-3)


def test_faithful_overflows_where_pairwise_does_not():
    """The documented fp32 range limits are real: with per-chunk |G_C| >> 88 the
    literal exp(-G) factor overflows (NaN/inf) while pairwise stays finite."""
    q, k, v, g, b, w, S0 = make_inputs(jax.random.PRNGKey(2), 30.0)
    args = lambda core: chunkwise_gated_delta_rule_2(  # noqa: E731
        q[None, None], k[None, None], v[None, None], g[None, None],
        b[None, None], w[None, None], S0[None, None], chunk_size=CHUNK, core=core)
    o_faithful, _ = args("faithful")
    o_pairwise, _ = args("pairwise")
    assert not bool(jnp.isfinite(o_faithful).all())
    assert bool(jnp.isfinite(o_pairwise).all())


def test_batched_entry_points_match_single():
    """The [B, H] vmapped public entry points equal per-head calls."""
    q, k, v, g, b, w, S0 = make_inputs(jax.random.PRNGKey(3), 1.0, batched=True)
    o_b, S_b = recurrent_gated_delta_rule_2(q, k, v, g, b, w, S0)
    o_c, S_c = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, chunk_size=CHUNK)
    for bi in range(q.shape[0]):
        for h in range(q.shape[1]):
            o1, S1 = _recurrent_single(
                q[bi, h], k[bi, h], v[bi, h], g[bi, h], b[bi, h], w[bi, h], S0[bi, h])
            np.testing.assert_allclose(o_b[bi, h], o1, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(o_c[bi, h], o1, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(S_b[bi, h], S1, rtol=1e-5, atol=1e-5)


def test_chunk_size_must_divide_length():
    q, k, v, g, b, w, S0 = make_inputs(jax.random.PRNGKey(4), 1.0)
    with pytest.raises(ValueError, match="chunk_size"):
        chunkwise_gated_delta_rule_2(
            q[None, None], k[None, None], v[None, None], g[None, None],
            b[None, None], w[None, None], S0[None, None], chunk_size=48)


def test_unknown_core_rejected():
    q, k, v, g, b, w, S0 = make_inputs(jax.random.PRNGKey(5), 1.0)
    with pytest.raises(ValueError, match="core"):
        chunkwise_gated_delta_rule_2(
            q[None, None], k[None, None], v[None, None], g[None, None],
            b[None, None], w[None, None], S0[None, None], core="nope")
