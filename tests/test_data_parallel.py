"""Smoke-test the real shard_map training path on two forced CPU devices."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from kimi_k3_gdn2 import KimiK3Config
from pipeline.config import DataConfig, ExperimentConfig, TrainConfig
from pipeline.train import build_model, build_optimizer, make_train_step


@pytest.mark.multi_device
def test_two_device_train_step_keeps_replicas_in_sync():
    if jax.device_count() < 2:
        pytest.skip("requires two JAX devices; set XLA_FLAGS before importing JAX")

    cfg = ExperimentConfig(
        model=KimiK3Config(
            vocab_size=32,
            d_model=32,
            n_layers=4,
            attn_res_layers_per_block=2,
            gdn_num_heads=2,
            gdn_head_k_dim=8,
            gdn_head_v_dim=8,
            gdn_chunk_size=4,
            gdn_conv_size=2,
            mla_num_q_heads=4,
            mla_num_kv_heads=1,
            mla_head_dim=8,
            max_seq_len=16,
            moe_d_latent=8,
            moe_d_ff=16,
            moe_n_routed=4,
            moe_n_shared=1,
            moe_top_k=2,
            moe_n_groups=2,
            moe_topk_groups=1,
        ),
        data=DataConfig(source="synthetic", tokenizer="byte", seq_len=8),
        train=TrainConfig(batch_size=2, max_steps=2, warmup_steps=0),
    )
    cfg.validate()

    devices = np.asarray(jax.devices()[:2])
    mesh = Mesh(devices, ("data",))
    replicated = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("data"))

    model = build_model(cfg, nnx.Rngs(0))
    optimizer = build_optimizer(model, cfg)
    for obj in (model, optimizer):
        nnx.update(obj, jax.device_put(nnx.state(obj), replicated))

    input_ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8) % cfg.model.vocab_size
    batch = {
        "input_ids": input_ids,
        "target_ids": (input_ids + 1) % cfg.model.vocab_size,
    }
    batch = jax.device_put(batch, data_sharding)

    before = model.embed.embedding[...].copy()
    total, ce, aux = make_train_step(mesh)(model, optimizer, batch, cfg.train.router_bias_lr)
    jax.block_until_ready(total)

    assert bool(jnp.isfinite(total) & jnp.isfinite(ce) & jnp.isfinite(aux))
    assert not bool(jnp.array_equal(before, model.embed.embedding[...]))
    for layer in model.layers:
        shards = layer.channel_mixer.router_bias.get_value().addressable_shards
        np.testing.assert_array_equal(np.asarray(shards[0].data), np.asarray(shards[1].data))
