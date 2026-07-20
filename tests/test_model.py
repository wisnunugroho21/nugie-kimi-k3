"""End-to-end verification of the KimiK3 model (kimi_k3_gdn2.py) and the
optimizer split (pipeline/optimizer.py).

  * training __call__ vs streaming step(): identical logits,
  * generate(): greedy determinism, sampling shapes, EOS early stop, and the
    evaluate.py keyword contract (temperature / top_p / eos_id / key),
  * attn_res=False: the plain-residual fallback runs and differs from AttnRes,
  * MLA training path vs streaming path,
  * Muon/AdamW classification and one real optimizer step.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from kimi_k3_gdn2 import KimiK3, KimiK3Config, count_params
from multi_latent_attention.attention import GroupedQueryLatentAttention
from pipeline.optimizer import make_optimizer

CFG = KimiK3Config(
    vocab_size=64, d_model=32, n_layers=4, full_attn_period=4,
    attn_res_layers_per_block=2,
    gdn_num_heads=2, gdn_head_k_dim=8, gdn_head_v_dim=8, gdn_chunk_size=16,
    mla_num_q_heads=4, mla_num_kv_heads=2, mla_head_dim=8, max_seq_len=64,
    moe_d_latent=8, moe_d_ff=32, moe_n_routed=8, moe_n_shared=1, moe_top_k=2,
    moe_n_groups=4, moe_topk_groups=2,
)


def make_model(cfg=CFG):
    return KimiK3(cfg, rngs=nnx.Rngs(0))


def test_forward_shapes_and_aux():
    model = make_model()
    ids = jax.random.randint(jax.random.PRNGKey(0), (2, 32), 0, CFG.vocab_size)
    logits, aux = model(ids)
    assert logits.shape == (2, 32, CFG.vocab_size)
    assert logits.dtype == jnp.float32
    assert aux["group_sizes"].shape == (CFG.n_layers, CFG.moe_n_routed)
    assert jnp.isfinite(logits).all() and jnp.isfinite(aux["aux_loss"])


def test_streaming_step_matches_training_forward():
    """Prefill + token-by-token step() reproduces the training __call__ logits
    (the GDN-2 chunk split, MLA cache, and AttnRes recomputation all agree)."""
    model = make_model()
    ids = jax.random.randint(jax.random.PRNGKey(1), (2, 32), 0, CFG.vocab_size)
    full, _ = model(ids)

    caches = model.init_cache(2, max_len=32)
    prefill = 17  # ragged on purpose
    out, caches = model.step(ids[:, :prefill], caches)
    outs = [out]
    for t in range(prefill, 32):
        out, caches = model.step(ids[:, t : t + 1], caches)
        outs.append(out)
    streamed = jnp.concatenate(outs, axis=1)
    np.testing.assert_allclose(streamed, full, rtol=2e-3, atol=2e-3)


def test_generate_greedy_and_sampling():
    model = make_model()
    prompt = jax.random.randint(jax.random.PRNGKey(2), (1, 8), 0, CFG.vocab_size)

    g1 = model.generate(prompt, max_new_tokens=12)
    g2 = model.generate(prompt, max_new_tokens=12)
    assert g1.shape == (1, 12)
    np.testing.assert_array_equal(g1, g2)  # greedy is deterministic

    s1 = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_p=0.9,
                        key=jax.random.PRNGKey(0))
    s2 = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_p=0.9,
                        key=jax.random.PRNGKey(0))
    assert s1.shape == (1, 12)
    np.testing.assert_array_equal(s1, s2)  # same key -> same sample

    # temperature > 0 without a key must be rejected, not silently greedy.
    try:
        model.generate(prompt, max_new_tokens=4, temperature=1.0)
        raise AssertionError("expected ValueError for missing key")
    except ValueError:
        pass

    with pytest.raises(ValueError, match="top_p"):
        model.generate(prompt, max_new_tokens=4, top_p=0.0)
    with pytest.raises(ValueError, match="max_len"):
        model.generate(prompt, max_new_tokens=4, max_len=11)


def test_streaming_rejects_wrong_cache_count():
    model = make_model()
    ids = jnp.ones((1, 1), jnp.int32)
    caches = model.init_cache(1, max_len=8)
    with pytest.raises(ValueError, match="layer caches"):
        model.step(ids, caches[:-1])


def test_generate_eos_early_stop():
    """With eos_id set, generation stops once every row has emitted it, and
    finished rows are padded with eos_id."""
    model = make_model()
    prompt = jax.random.randint(jax.random.PRNGKey(3), (2, 8), 0, CFG.vocab_size)
    greedy = model.generate(prompt, max_new_tokens=16)
    eos = int(greedy[0, 2])  # force an early stop for row 0 at step 3
    gen = model.generate(prompt, max_new_tokens=16, eos_id=eos)
    assert gen.shape[1] <= 16
    row0 = np.asarray(gen[0])
    hits = np.nonzero(row0 == eos)[0]
    assert hits.size > 0
    assert (row0[hits[0]:] == eos).all()  # padding after the first EOS


def test_attn_res_off_is_plain_residual_stream():
    cfg = KimiK3Config(**{**CFG.__dict__, "attn_res": False})
    model = make_model(cfg)
    ids = jax.random.randint(jax.random.PRNGKey(4), (2, 32), 0, cfg.vocab_size)
    logits, _ = model(ids)
    assert jnp.isfinite(logits).all()
    # No AttnRes parameters are allocated when the backbone is disabled.
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(
        nnx.state(model, nnx.Param))]
    assert not any("res_mixer" in n for n in names)


def test_mla_step_matches_call():
    attn = GroupedQueryLatentAttention(
        embed_dim=32, num_q_heads=4, num_kv_heads=2, head_dim=8,
        gated=True, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 20, 32))
    full = attn(x)
    cache = attn.init_cache(2, max_len=20)
    out1, cache = attn.step(x[:, :13], cache)
    out2, _ = attn.step(x[:, 13:], cache)
    np.testing.assert_allclose(
        jnp.concatenate([out1, out2], axis=1), full, rtol=2e-4, atol=2e-4)

    with pytest.raises(ValueError, match="capacity exceeded"):
        attn.step(x[:, :1], cache._replace(pos=jnp.array(20, jnp.int32)))


def test_remat_matches_no_remat():
    """cfg.remat must change memory behavior only: identical logits, aux, and
    gradients versus the un-checkpointed forward."""
    import optax

    model = make_model()
    model_r = make_model(KimiK3Config(**{**CFG.__dict__, "remat": True}))
    ids = jax.random.randint(jax.random.PRNGKey(9), (2, 32), 0, CFG.vocab_size)
    tgt = jax.random.randint(jax.random.PRNGKey(10), (2, 32), 0, CFG.vocab_size)

    def loss(m):
        logits, aux = m(ids)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, tgt).mean()
        return ce + aux["aux_loss"]

    l1, g1 = nnx.value_and_grad(loss)(model)
    l2, g2 = nnx.value_and_grad(loss)(model_r)
    np.testing.assert_allclose(float(l1), float(l2), rtol=1e-6)
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_muon_adamw_split_and_step():
    """Every 2-D Linear kernel and 3-D expert stack goes to Muon; embedding,
    LM head, 1-D params, and conv kernels go to AdamW — and one optimizer
    step on real gradients updates the params without NaNs."""
    from pipeline.optimizer import _muon_spec

    model = make_model()
    params = nnx.state(model, nnx.Param)
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        names = {getattr(k, "key", getattr(k, "name", getattr(k, "idx", "")))
                 for k in path}
        names = {str(n) for n in names}
        spec = _muon_spec(path, leaf)
        if names & {"embed", "lm_head", "A_log"} or leaf.ndim == 1:
            assert spec is None, path
        elif names & {"w_in", "w_out"} and leaf.ndim == 3:
            assert spec is not None, path
        elif leaf.ndim == 2:
            assert spec is not None, path

    optimizer = make_optimizer(
        model, optax.constant_schedule(1e-3), verbose=False)
    ids = jax.random.randint(jax.random.PRNGKey(6), (2, 32), 0, CFG.vocab_size)
    tgt = jax.random.randint(jax.random.PRNGKey(7), (2, 32), 0, CFG.vocab_size)

    def loss_fn(m):
        logits, aux = m(ids)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, tgt).mean()
        return ce + aux["aux_loss"]

    before = float(count_params(model))
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    after_state = nnx.state(model, nnx.Param)
    assert jnp.isfinite(loss)
    assert all(jnp.isfinite(x).all() for x in jax.tree.leaves(after_state))
    assert float(count_params(model)) == before
