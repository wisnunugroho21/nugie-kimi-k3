"""A guided tour of the Kimi K3 implementation. Run: `python demo.py`.

Builds a tiny K3 (same structure as the 2.8T model, ~4M parameters), walks
through each architectural component from §2 of the report, and trains it for a
few steps with Per-Head Muon and Quantile Balancing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import kimi_k3 as k3
from kimi_k3.kda import kda_chunkwise, kda_recurrent
from kimi_k3.moe import quantile_balancing_update


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * len(title))


def main() -> None:
    cfg = k3.KimiK3Config.tiny()
    real = k3.KimiK3Config()

    # ---------------------------------------------------------------------
    rule("Table 1 — the real Kimi K3")
    mla_layers = [i for i in range(real.num_layers) if real.is_mla_layer(i)]
    print(f"  {real.num_layers} layers = {real.num_layers - len(mla_layers)} KDA "
          f"+ {len(mla_layers)} Gated MLA   (paper: 69 + 24)")
    print(f"  hidden {real.hidden_size}, {real.num_heads} heads, vocab {real.vocab_size:,}")
    print(f"  MoE: {real.num_experts_per_token} of {real.num_routed_experts} routed experts "
          f"+ {real.num_shared_experts} shared, latent width {real.moe_latent_size}")
    print(f"  AttnRes: {real.num_attn_res_blocks} blocks of {real.attn_res_block_size} layers")
    print(f"\n  Demo model below uses the same structure at {cfg.hidden_size} hidden / "
          f"{cfg.num_layers} layers.")

    # ---------------------------------------------------------------------
    rule("§2.1.1  Kimi Delta Attention — chunkwise form == the recurrence")
    ks = jax.random.split(jax.random.PRNGKey(0), 5)
    unit = lambda k, s: (lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True))(
        jax.random.normal(k, s)
    )
    shape = (2, 3, 40, 16)
    q, kk = unit(ks[0], shape), unit(ks[1], shape)
    v = jax.random.normal(ks[2], shape)
    alpha = jnp.exp(-5.0 * jax.nn.sigmoid(jax.random.normal(ks[3], shape)))  # Eq. 5
    beta = jax.nn.sigmoid(jax.random.normal(ks[4], shape[:3]))

    o_ref, _ = kda_recurrent(q, kk, v, alpha, beta)  # Eq. 1, token by token
    o_chunk, _ = kda_chunkwise(q, kk, v, alpha, beta, chunk_size=16)  # Eqs. 3-4
    print(f"  max |chunkwise - recurrent| = {jnp.abs(o_chunk - o_ref).max():.2e}  (exact rewrite)")
    print(f"  decay range: alpha in [{alpha.min():.4f}, {alpha.max():.4f}]; "
          f"Eq. 5 floor e^-5 = {jnp.exp(-5.0):.4f}")
    print(f"  -> 1/Gamma over a 16-token chunk stays below e^80 ~ {jnp.exp(80.0):.1e}, "
          "inside the BF16 range")

    # ---------------------------------------------------------------------
    rule("§2.3.2  SiTU-GLU is bounded where SwiGLU is not")
    x = jnp.array([[-50.0, -1.0, 0.05, 1.0, 50.0, 500.0]])
    g = k3.softcap(x, cfg.situ_beta_gate) * jax.nn.sigmoid(x)
    u = k3.softcap(x, cfg.situ_beta_up)
    swiglu = (x * jax.nn.sigmoid(x)) * x
    print(f"  input     {[f'{v:9.2f}' for v in x[0]]}")
    print(f"  SwiGLU    {[f'{v:9.2f}' for v in swiglu[0]]}")
    print(f"  SiTU-GLU  {[f'{v:9.2f}' for v in (g * u)[0]]}")
    print(f"  bound beta1*beta2 = {cfg.situ_beta_gate * cfg.situ_beta_up:.0f} "
          "(Appendix B, Eq. 19); near 0 the two agree")

    # ---------------------------------------------------------------------
    rule("§2.2  Attention Residuals — what each layer can see")
    stream = k3.AttnResStream(jnp.zeros((1, 1, cfg.hidden_size)))
    seen = []
    for i in range(cfg.num_layers):
        seen.append(stream.num_sources)
        stream.accumulate(jnp.zeros((1, 1, cfg.hidden_size)))
        if (i + 1) % cfg.attn_res_block_size == 0:
            stream.close_block()
    stream.finish()
    print(f"  sources visible to layer 0..{cfg.num_layers - 1}: {seen}")
    print(f"  (1 = the embedding alone; grows as blocks of {cfg.attn_res_block_size} are sealed)")
    print(f"  final aggregation over {len(stream.sources)} block representations")

    # ---------------------------------------------------------------------
    rule("§2.3.3  Quantile Balancing — one update, not a slow nudge")
    moe = k3.StableLatentMoE(cfg, rngs=nnx.Rngs(0))
    xb = jax.random.normal(jax.random.PRNGKey(3), (8, 32, cfg.hidden_size))
    _, stats = moe(xb)

    def loads(bias):
        s = jax.nn.sigmoid(moe.router(xb.reshape(-1, cfg.hidden_size)))
        idx = jax.lax.top_k(s + bias, cfg.num_experts_per_token)[1]
        return jnp.bincount(idx.reshape(-1), length=cfg.num_routed_experts)

    before = loads(moe.router_bias[...])
    after = loads(quantile_balancing_update(stats, cfg.num_experts_per_token))
    target = 8 * 32 * cfg.num_experts_per_token / cfg.num_routed_experts
    print(f"  target load per expert: {target:.0f}")
    print(f"  before QB: {before}  (std {before.astype(float).std():.2f})")
    print(f"  after  QB: {after}  (std {after.astype(float).std():.2f})")

    # ---------------------------------------------------------------------
    rule("§2.4  Native vision — pixels become ordinary tokens")
    model = k3.KimiK3(cfg, rngs=nnx.Rngs(0), with_vision=True)
    pixels = jax.random.normal(jax.random.PRNGKey(4), (1, 4, 32, 32, 3))
    vis = model.vision(pixels)
    patches = 4 * (32 // cfg.vision_patch_size) ** 2
    print(f"  4 frames of 32x32 -> {patches} patches -> {vis.shape[1]} tokens "
          f"(temporal pool /{cfg.vision_temporal_pool}, pixel shuffle /{cfg.vision_pixel_shuffle**2})")
    print(f"  output width {vis.shape[-1]} == d_model: they splice straight into the token stream")

    # ---------------------------------------------------------------------
    rule("Forward pass")
    ids = jax.random.randint(jax.random.PRNGKey(5), (2, 32), 0, cfg.vocab_size)
    out = model(ids, return_mtp=True)
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"  parameters      {n_params / 1e6:.2f}M")
    print(f"  logits          {out['logits'].shape}")
    print(f"  MTP logits      {out['mtp_logits'].shape}   (predicts token t+2)")
    print(f"  AttnRes blocks  {out['blocks'].shape}   [N+1, B, T, d]")
    print(f"  MoE layers      {len(out['router_stats'])}")

    # ---------------------------------------------------------------------
    rule("§2.5  Training with Per-Head Muon + Quantile Balancing")
    model = k3.KimiK3(cfg, rngs=nnx.Rngs(0), with_vision=False)
    gdef, params, rest = nnx.split(model, nnx.Param, ...)
    steps = 8
    tx = k3.kimi_k3_optimizer(k3.cosine_schedule_with_warmup(3e-3, steps), cfg.num_heads, params)
    opt_state = tx.init(params)

    def loss_fn(p):
        out = nnx.merge(gdef, p, rest)(ids, return_mtp=True)
        loss = k3.causal_lm_loss(out["logits"], ids) + 0.3 * k3.mtp_loss(out["mtp_logits"], ids)
        return loss, out["router_stats"]

    for step in range(steps):
        (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # QB runs AFTER the step: §2.3.3 requires that a batch is never routed
        # with a bias derived from itself.
        model = nnx.merge(gdef, params, rest)
        k3.apply_quantile_balancing(model, stats)
        _, params, rest = nnx.split(model, nnx.Param, ...)

        imbalance = jnp.stack([s.load for s in stats]).astype(float).std(axis=1).mean()
        print(f"  step {step}   loss {loss:.4f}   mean expert-load std {imbalance:.3f}")

    print("\nEverything above is §2 of arXiv:2607.24653. See README.md for the file map.\n")


if __name__ == "__main__":
    main()
