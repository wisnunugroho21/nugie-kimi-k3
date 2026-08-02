"""Tests for the Kimi K3 implementation.

These check the properties the paper's equations actually assert, rather than
just that tensors have the right shape:

  * the chunkwise KDA form (Eqs. 3-4) equals the recurrence it is derived from
    (Eq. 1), and both are causal;
  * the streaming decode paths reproduce the full forward pass exactly;
  * SiTU-GLU respects its stated bound (Appendix B, Eq. 19);
  * Quantile Balancing (Eq. 14) actually drives expert loads to the target;
  * the layer composition and parameter counts match Table 1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import kimi_k3 as k3
from kimi_k3.kda import KimiDeltaAttention, kda_chunkwise, kda_recurrent
from kimi_k3.mla import GatedMLA
from kimi_k3.moe import StableLatentMoE, quantile_balancing_update, quantile_balancing_update_histogram
from kimi_k3.muon import orthogonalize, per_head_muon


@pytest.fixture(scope="module")
def cfg():
    return k3.KimiK3Config.tiny()


def _kda_inputs(seed=0, B=2, H=3, T=37, D=16):
    ks = jax.random.split(jax.random.PRNGKey(seed), 5)
    unit = lambda k: (lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True))(
        jax.random.normal(k, (B, H, T, D))
    )
    return (
        unit(ks[0]),  # q
        unit(ks[1]),  # k
        jax.random.normal(ks[2], (B, H, T, D)),  # v
        jnp.exp(-5.0 * jax.nn.sigmoid(jax.random.normal(ks[3], (B, H, T, D)))),  # Eq. 5 range
        jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, T))),  # beta
    )


# ---------------------------------------------------------------- §2.1.1 KDA


@pytest.mark.parametrize("chunk", [4, 8, 16])
def test_kda_chunkwise_equals_recurrence(chunk):
    """Eqs. 3-4 are an exact rewrite of Eq. 1, not an approximation."""
    q, k, v, alpha, beta = _kda_inputs()
    o_ref, s_ref = kda_recurrent(q, k, v, alpha, beta)
    o_chunk, s_chunk = kda_chunkwise(q, k, v, alpha, beta, chunk_size=chunk)
    assert jnp.allclose(o_chunk, o_ref, atol=1e-5)
    assert jnp.allclose(s_chunk, s_ref, atol=1e-5)


def test_kda_chunkwise_handles_ragged_length():
    """T need not be a multiple of the chunk size; padding must stay inert."""
    q, k, v, alpha, beta = _kda_inputs(T=37)
    o_ref, _ = kda_recurrent(q, k, v, alpha, beta)
    o, _ = kda_chunkwise(q, k, v, alpha, beta, chunk_size=16)
    assert o.shape == o_ref.shape
    assert jnp.allclose(o, o_ref, atol=1e-5)


def test_kda_decay_is_lower_bounded(cfg):
    """Eq. 5: alpha in (e^{g_min}, 1), which is what keeps 1/Gamma finite.

    The interval is open in the algebra but closed in floating point — a large
    enough logit saturates `sigmoid` to exactly 1.0 — so the assertion is `>=`.
    That is precisely the guarantee the chunkwise form needs: alpha never goes
    *below* e^{g_min}, so 1/Gamma over a 16-token chunk never exceeds e^80.
    """
    kda = KimiDeltaAttention(cfg, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, cfg.hidden_size)) * 10.0
    alpha = kda._decay(x)
    assert float(alpha.min()) >= float(jnp.exp(jnp.array(cfg.kda_g_min)))
    assert float(alpha.max()) <= 1.0


def test_kda_streaming_matches_forward(cfg):
    kda = KimiDeltaAttention(cfg, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 12, cfg.hidden_size)) * 0.5
    full = kda(x)
    cache = kda.init_cache(2)
    steps = []
    for t in range(x.shape[1]):
        y, cache = kda.step(x[:, t : t + 1], cache)
        steps.append(y)
    assert jnp.allclose(jnp.concatenate(steps, axis=1), full, atol=1e-5)


def test_kda_is_causal(cfg):
    kda = KimiDeltaAttention(cfg, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 12, cfg.hidden_size)) * 0.5
    assert jnp.allclose(kda(x)[:, :8], kda(x.at[:, 8:].add(100.0))[:, :8], atol=1e-4)


# ---------------------------------------------------------------- §2.1.2 MLA


def test_mla_streaming_matches_forward(cfg):
    mla = GatedMLA(cfg, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 12, cfg.hidden_size)) * 0.5
    full = mla(x)
    cache = mla.init_cache(2, 12)
    steps = []
    for t in range(x.shape[1]):
        y, cache = mla.step(x[:, t : t + 1], cache)
        steps.append(y)
    assert jnp.allclose(jnp.concatenate(steps, axis=1), full, atol=1e-5)


def test_mla_is_causal(cfg):
    mla = GatedMLA(cfg, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 12, cfg.hidden_size)) * 0.5
    assert jnp.allclose(mla(x)[:, :8], mla(x.at[:, 8:].add(100.0))[:, :8], atol=1e-4)


def test_mla_caches_only_the_latent(cfg):
    """The point of MLA: cache width is kv_lora_rank, not num_heads*head_dim."""
    mla = GatedMLA(cfg, rngs=nnx.Rngs(0))
    cache = mla.init_cache(2, 64)
    assert cache.latent.shape == (2, 64, cfg.kv_lora_rank)
    assert cfg.kv_lora_rank < cfg.num_heads * cfg.head_dim


# ----------------------------------------------------------- §2.3.2 SiTU-GLU


def test_situ_glu_is_bounded(cfg):
    """Appendix B, Eq. 19: |SiTU-GLU(x)| <= beta1*beta2 = 100."""
    act = k3.SiTUGLU(32, 64, beta_gate=cfg.situ_beta_gate, beta_up=cfg.situ_beta_up, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (128, 32)) * 1_000.0  # extreme on purpose
    y = act(x)
    bound = cfg.situ_beta_gate * cfg.situ_beta_up
    assert float(jnp.abs(y).max()) <= bound + 1e-3
    assert jnp.isfinite(y).all()


def test_softcap_matches_identity_near_origin():
    """Eq. 18: beta*tanh(z/beta) = z + O(z^3/beta^2)."""
    z = jnp.linspace(-0.05, 0.05, 41)
    assert jnp.allclose(k3.softcap(z, 25.0), z, atol=1e-6)


# --------------------------------------------------- §2.3.3 Quantile Balancing


@pytest.mark.parametrize("update", [quantile_balancing_update, quantile_balancing_update_histogram])
def test_quantile_balancing_reduces_imbalance(cfg, update):
    """One QB update should move loads decisively toward the target m*k/n."""
    moe = StableLatentMoE(cfg, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 32, cfg.hidden_size))
    _, stats = moe(x)

    def loads(bias):
        s = jax.nn.sigmoid(moe.router(x.reshape(-1, cfg.hidden_size)))
        idx = jax.lax.top_k(s + bias, cfg.num_experts_per_token)[1]
        return jnp.bincount(idx.reshape(-1), length=cfg.num_routed_experts).astype(float)

    before = loads(moe.router_bias[...])
    after = loads(update(stats, cfg.num_experts_per_token))
    assert float(after.std()) < float(before.std())
    assert float(after.sum()) == float(before.sum())  # every token still gets k experts


def test_quantile_balancing_bias_is_zero_mean(cfg):
    """Eq. 14's second line: a common offset cannot change any Top-k decision."""
    moe = StableLatentMoE(cfg, rngs=nnx.Rngs(0))
    _, stats = moe(jax.random.normal(jax.random.PRNGKey(1), (4, 32, cfg.hidden_size)))
    assert abs(float(quantile_balancing_update(stats, cfg.num_experts_per_token).mean())) < 1e-5


def test_router_bias_excluded_from_gate_weights(cfg):
    """Eq. 13: b regulates dispatch only; the mixture weights p come from raw s."""
    moe = StableLatentMoE(cfg, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, cfg.hidden_size))
    y0, _ = moe(x)
    # A uniform shift of the bias leaves both selection and weights untouched.
    moe.router_bias[...] = moe.router_bias[...] + 3.0
    assert jnp.allclose(moe(x)[0], y0, atol=1e-6)


def test_moe_dispatch_is_topk_exact(cfg):
    moe = StableLatentMoE(cfg, rngs=nnx.Rngs(0))
    _, stats = moe(jax.random.normal(jax.random.PRNGKey(1), (2, 16, cfg.hidden_size)))
    assert int(stats.load.sum()) == 2 * 16 * cfg.num_experts_per_token


# ------------------------------------------------------- §2.2 AttnRes / model


def test_first_layer_sees_only_the_embedding(cfg):
    """Eq. 10 with n=1, i=1: V = [b_0], so the softmax is trivially 1."""
    stream = k3.AttnResStream(jnp.ones((2, 4, cfg.hidden_size)))
    assert stream.num_sources == 1
    query = k3.DepthAttention(cfg.hidden_size, rngs=nnx.Rngs(0))
    assert jnp.allclose(query(stream.values()), jnp.ones((2, 4, cfg.hidden_size)))


def test_attn_res_block_bookkeeping(cfg):
    """Sources grow by one per sealed block; K3's 93 layers leave a partial one."""
    stream = k3.AttnResStream(jnp.zeros((1, 2, cfg.hidden_size)))
    for i in range(cfg.num_layers):
        stream.accumulate(jnp.full((1, 2, cfg.hidden_size), float(i)))
        if (i + 1) % cfg.attn_res_block_size == 0:
            stream.close_block()
    stream.finish()
    assert len(stream.sources) == cfg.num_attn_res_blocks + 1  # +1 for the embedding


def test_model_forward_and_gradients(cfg):
    model = k3.KimiK3(cfg, rngs=nnx.Rngs(0), with_vision=False)
    ids = jax.random.randint(jax.random.PRNGKey(1), (2, 24), 0, cfg.vocab_size)
    gdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss(p):
        out = nnx.merge(gdef, p, rest)(ids, return_mtp=True)
        return k3.causal_lm_loss(out["logits"], ids) + 0.3 * k3.mtp_loss(out["mtp_logits"], ids)

    value, grads = jax.value_and_grad(loss)(params)
    assert jnp.isfinite(value)
    flat = jax.tree_util.tree_flatten_with_path(grads)[0]
    assert flat and all(jnp.isfinite(g).all() for _, g in flat)

    # Every parameter should receive gradient except the two documented in
    # `test_layer_zero_pseudo_queries_are_inert` below.
    dead = [jax.tree_util.keystr(p) for p, g in flat if float(jnp.abs(g).sum()) == 0]
    assert all("['layers'][0]" in name and "_query" in name for name in dead), dead
    assert len(dead) == 2


def test_layer_zero_pseudo_queries_are_inert():
    """A real property of Block AttnRes, worth stating explicitly.

    Layer 0 sits at the start of the first block, so by Eq. 10 its value set is
    `[b_0]` — the token embedding alone. A softmax over a single source is
    identically 1 no matter what the pseudo-query is, so layer 0's two `w`
    vectors cannot affect the output and receive exactly zero gradient forever.

    They are dead weight by construction: 2*d parameters out of 2.8T. Nothing to
    fix, but an optimiser that assumes every parameter moves would be surprised.
    """
    cfg = k3.KimiK3Config.tiny()
    query = k3.DepthAttention(cfg.hidden_size, rngs=nnx.Rngs(0))
    single_source = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 3, cfg.hidden_size))

    def out_sum(w):
        logits = jnp.einsum("sbtd,d->bts", single_source, w)
        return jnp.einsum("bts,sbtd->btd", jax.nn.softmax(logits, -1), single_source).sum()

    assert float(jnp.abs(jax.grad(out_sum)(query.w[...])).sum()) == 0.0


def test_model_is_causal(cfg):
    model = k3.KimiK3(cfg, rngs=nnx.Rngs(0), with_vision=False)
    ids = jax.random.randint(jax.random.PRNGKey(1), (1, 16), 0, cfg.vocab_size)
    other = ids.at[:, 10:].set(0)
    assert jnp.allclose(model(ids)["logits"][:, :10], model(other)["logits"][:, :10], atol=1e-4)


def test_quantile_balancing_applied_to_every_moe_layer(cfg):
    model = k3.KimiK3(cfg, rngs=nnx.Rngs(0), with_vision=False)
    ids = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, cfg.vocab_size)
    out = model(ids, return_mtp=True)
    k3.apply_quantile_balancing(model, out["router_stats"])
    biases = [l.ffn.router_bias[...] for l in model.layers if not l.is_dense]
    assert all(float(jnp.abs(b).sum()) > 0 for b in biases)


# ------------------------------------------------------------ §2.4 / §2.5


def test_vision_tokens_land_in_the_llm_space(cfg):
    model = k3.KimiK3(cfg, rngs=nnx.Rngs(0), with_vision=True)
    pixels = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 32, 32, 3))
    tokens = model.vision(pixels)
    grid = 32 // cfg.vision_patch_size  # patches per side
    s, pool = cfg.vision_pixel_shuffle, cfg.vision_temporal_pool
    assert tokens.shape == (2, (4 // pool) * (grid // s) ** 2, cfg.hidden_size)


def test_vision_splice_replaces_placeholder_tokens(cfg):
    model = k3.KimiK3(cfg, rngs=nnx.Rngs(0), with_vision=True)
    ids = jnp.array([[5, 5, 7, 7, 5, 5]])  # token 7 marks an image slot
    pixels = jax.random.normal(jax.random.PRNGKey(1), (1, 1, 16, 16, 3))
    spliced = model.embed_inputs(ids, pixels, image_token_id=7)
    text_only = model.embed(ids)
    assert jnp.allclose(spliced[:, [0, 1, 4, 5]], text_only[:, [0, 1, 4, 5]])
    assert not jnp.allclose(spliced[:, 2:4], text_only[:, 2:4])


def test_newton_schulz_orthogonalises():
    m = jax.random.normal(jax.random.PRNGKey(0), (32, 64))
    sv = jnp.linalg.svd(k3.newton_schulz(m), compute_uv=False)
    # The quintic iteration is tuned for speed, not precision: it lands the
    # singular values near 1 rather than exactly at 1.
    assert 0.5 < float(sv.min()) and float(sv.max()) < 1.5
    assert float(sv.max() - sv.min()) < float(
        (lambda s: s.max() - s.min())(jnp.linalg.svd(m, compute_uv=False))
    )


def test_per_head_orthogonalisation_is_independent_per_head():
    """§2.5: each head's block is normalised on its own, so a head with a huge
    gradient scale cannot swamp the others."""
    key = jax.random.PRNGKey(0)
    m = jax.random.normal(key, (32, 4 * 8))
    m = m.reshape(32, 4, 8).at[:, 0].multiply(1000.0).reshape(32, 32)  # head 0 dominates
    per_head = orthogonalize(m, 4).reshape(32, 4, 8)
    norms = jnp.linalg.norm(per_head, axis=(0, 2))
    assert float(norms.max() / norms.min()) < 1.5  # scales equalised

    whole = orthogonalize(m, 0).reshape(32, 4, 8)
    whole_norms = jnp.linalg.norm(whole, axis=(0, 2))
    assert float(whole_norms.max() / whole_norms.min()) > 2.0  # still dominated


def test_optimizer_step_runs(cfg):
    import optax

    model = k3.KimiK3(cfg, rngs=nnx.Rngs(0), with_vision=False)
    ids = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, cfg.vocab_size)
    gdef, params, rest = nnx.split(model, nnx.Param, ...)
    tx = k3.kimi_k3_optimizer(k3.cosine_schedule_with_warmup(1e-3, 100), cfg.num_heads, params)
    state = tx.init(params)

    def loss(p):
        return k3.causal_lm_loss(nnx.merge(gdef, p, rest)(ids)["logits"], ids)

    before = loss(params)
    for _ in range(3):
        updates, state = tx.update(jax.grad(loss)(params), state, params)
        params = optax.apply_updates(params, updates)
    assert float(loss(params)) < float(before)


def test_muon_returns_gradient_direction_updates():
    """`per_head_muon` must point ALONG the gradient, like `scale_by_adam`.

    The descent negation belongs to `scale_by_learning_rate` (flip_sign=True).
    If this transformation negated as well, the two would cancel and every
    matrix parameter would ascend. Asserting the raw direction here is what
    pins the convention down; a loss-goes-down test does not, because the
    AdamW branch alone is enough to drag the loss down for a few steps.
    """
    grads = {"w": jax.random.normal(jax.random.PRNGKey(0), (16, 8))}
    tx = per_head_muon(momentum=0.0, nesterov=False)
    update, _ = tx.update(grads, tx.init(grads), grads)
    # <UV^T, USV^T> = tr(S) > 0, so this is strictly positive whenever correct.
    assert float(jnp.vdot(update["w"], grads["w"])) > 0


def test_optimizer_chain_descends_on_a_convex_problem():
    """End-to-end sign check on loss = 0.5*||W||^2, whose minimum is at W = 0.

    The regression this guards against inverted the whole Muon branch: the same
    setup went 6.21 -> 7.19 (ascent) before the fix and 6.21 -> 5.20 after, so
    the direction assertion and a plain "the norm shrank" both separate them
    decisively.
    """
    import optax

    params = {"w": jnp.eye(4) * 3.0 + 0.1}
    loss = lambda p: 0.5 * jnp.sum(p["w"] ** 2)
    start = float(jnp.linalg.norm(params["w"]))
    tx = k3.kimi_k3_optimizer(0.1, num_heads=4, params=params, weight_decay=0.0)
    state = tx.init(params)

    first, _ = tx.update(jax.grad(loss)(params), state, params)
    assert float(jnp.vdot(first["w"], jax.grad(loss)(params)["w"])) < 0, "update must oppose the gradient"

    for _ in range(15):
        updates, state = tx.update(jax.grad(loss)(params), state, params)
        params = optax.apply_updates(params, updates)
    assert float(jnp.linalg.norm(params["w"])) < 0.95 * start


# ----------------------------------------------------------------- Table 1


def test_layer_composition_matches_table_1():
    """93 layers = 69 KDA + 24 MLA, and the last layer is global."""
    cfg = k3.KimiK3Config()
    mla = [i for i in range(cfg.num_layers) if cfg.is_mla_layer(i)]
    assert len(mla) == 24
    assert cfg.num_layers - len(mla) == 69
    assert mla[-1] == cfg.num_layers - 1
    assert cfg.num_attn_res_blocks == 8


def test_parameter_counts_match_table_1():
    """The inferred head dims must reproduce the report's headline counts."""
    c = k3.KimiK3Config()
    d, hd = c.hidden_size, c.num_heads * c.head_dim
    lat, dff = c.moe_latent_size, c.moe_expert_hidden
    n_mla = sum(c.is_mla_layer(i) for i in range(c.num_layers))
    n_kda = c.num_layers - n_mla
    n_moe = c.num_layers - c.num_dense_layers

    kda = 3 * d * hd + hd * d + d * hd
    mla = (
        d * c.q_lora_rank + c.q_lora_rank * hd + d * c.kv_lora_rank
        + c.kv_lora_rank * c.num_heads * 2 * c.head_dim + hd * d + d * hd
    )
    shared = c.num_shared_experts * 3 * d * dff + 2 * d * lat
    act = n_kda * kda + n_mla * mla
    act += n_moe * (c.num_experts_per_token * 3 * lat * dff + shared) + 2 * c.vocab_size * d
    total = n_kda * kda + n_mla * mla
    total += n_moe * (c.num_routed_experts * 3 * lat * dff + shared) + 2 * c.vocab_size * d

    assert 100e9 < act < 108e9, f"activated {act/1e9:.1f}B, paper says 104.2B"
    assert 2.7e12 < total < 2.85e12, f"total {total/1e12:.2f}T, paper says 2.78T"
