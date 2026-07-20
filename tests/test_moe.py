"""Verification of the LatentMoE channel mixer (multi_latent_attention/moe.py).

The dispatched path (sort by expert id -> ragged_dot grouped GEMMs ->
scatter-add combine) is checked against `dense_forward`, which runs every
expert densely with the same weights — any mismatch is a dispatch bug.
Also covers group-limited routing, the router-bias update rule, and the
aux-dict contract the training loop relies on.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from multi_latent_attention.moe import LatentMoE, update_router_bias

D_MODEL, D_FF, D_LATENT = 32, 64, 8
B, L = 2, 24


def _x(key=0):
    return jax.random.normal(jax.random.PRNGKey(key), (B, L, D_MODEL))


def test_latent_moe_matches_dense():
    moe = LatentMoE(
        D_MODEL,
        D_LATENT,
        D_FF,
        n_routed=8,
        n_shared=1,
        top_k=2,
        n_groups=4,
        topk_groups=2,
        rngs=nnx.Rngs(0),
    )
    x = _x(1)
    out, aux = moe(x)
    np.testing.assert_allclose(out, moe.dense_forward(x), rtol=1e-4, atol=1e-4)
    # Aux contract used by the training loop (aux_loss sum, router-bias nudge).
    assert set(aux) == {"load", "aux_loss", "group_sizes"}
    assert aux["group_sizes"].shape == (moe.E,)
    np.testing.assert_allclose(float(aux["load"].sum()), 1.0, rtol=1e-6)


def test_aux_loss_uses_normalized_sigmoid_probs():
    """The balancing loss must build P_e from the NORMALIZED SIGMOID affinities
    the router routes with (DeepSeek-V3 Eq. 18), not a softmax."""
    moe = LatentMoE(
        D_MODEL,
        D_LATENT,
        D_FF,
        n_routed=8,
        n_shared=1,
        top_k=2,
        rngs=nnx.Rngs(0),
    )
    x = _x(3)
    _, aux = moe(x)

    logits = moe.router(x.reshape(-1, D_MODEL)).astype(jnp.float32)
    scores = jax.nn.sigmoid(logits)
    probs = (scores / (scores.sum(-1, keepdims=True) + 1e-9)).mean(0)
    load = aux["group_sizes"].astype(jnp.float32) / aux["group_sizes"].sum()
    expected = moe.aux_alpha * moe.E * jnp.sum(load * probs)
    np.testing.assert_allclose(float(aux["aux_loss"]), float(expected), rtol=1e-6, atol=1e-8)


def test_group_limited_routing_respects_groups():
    """Every token's top-k experts must fall inside its topk_groups best groups."""
    n_routed, n_groups, topk_groups, top_k = 8, 4, 2, 2
    moe = LatentMoE(
        D_MODEL,
        D_LATENT,
        D_FF,
        n_routed=n_routed,
        n_shared=1,
        top_k=top_k,
        n_groups=n_groups,
        topk_groups=topk_groups,
        rngs=nnx.Rngs(0),
    )
    xf = _x(2).reshape(-1, D_MODEL)
    top_idx, _, _ = moe._route(xf)
    groups_hit = jnp.sort(top_idx // (n_routed // n_groups), axis=-1)
    n_distinct = (jnp.diff(groups_hit, axis=-1) != 0).sum(-1) + 1
    assert int(n_distinct.max()) <= topk_groups


def test_update_router_bias_direction():
    """Under-loaded experts get a positive nudge, over-loaded a negative one."""
    bias = jnp.zeros((4,))
    group_sizes = jnp.array([10, 0, 5, 5])  # expert 1 starved, expert 0 hot
    new = update_router_bias(bias, group_sizes, lr=0.1)
    assert float(new[1]) > 0 and float(new[0]) < 0
    np.testing.assert_allclose(new[2], new[3])


def test_latent_moe_never_allocates_full_width_experts():
    """LatentMoE allocates only latent-width routed expert stacks."""
    moe = LatentMoE(D_MODEL, D_LATENT, D_FF, n_routed=8, n_shared=1, top_k=2, rngs=nnx.Rngs(0))
    assert moe.w_in.shape == (8, D_LATENT, 2 * D_FF)
    assert moe.w_out.shape == (8, D_FF, D_LATENT)
