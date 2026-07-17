"""
Training pipeline for the Kimi K3 recreation: Grain data, Optax optimizer,
Orbax checkpointing.

WHAT ONE TRAINING STEP DOES
---------------------------
    batch = train_ds[step]                       # Grain: pure function of (seed, step)
    loss  = CE(logits, labels) + Σ_layers aux_loss   # MoE Switch-style aux loss
    grads -> optax.adamw (warmup-cosine LR, global-norm clip, weight decay
             masked to >=2-D params: norms/biases/decays/pseudo-queries exempt)
    router_bias[e] += lr_bias * sign(fair_share - load[e])   # per MoE layer,
             OUTSIDE the gradient: DeepSeek-V3 aux-loss-free load balancing
             (see multi_latent_attention/moe.py::update_router_bias)

Because the Grain batch at any step is a pure function of (seed, step), resume
needs no data-iterator state: Orbax restores (model, optimizer) and training
continues at latest_step + 1 reading `train_ds[step]` — bit-identical to a run
that never stopped.

CHECKPOINT LAYOUT (Orbax CheckpointManager, StandardSave)
---------------------------------------------------------
    <ckpt_dir>/<step>/state/   nnx.to_pure_dict of nnx.state((model, optimizer))
The model config + data meta travel as JSON next to the checkpoints so that
evaluate.py can rebuild the exact model without importing this file's CLI args.

USAGE
-----
    python codeparrot_data.py --out data                       # once
    python train.py --data data --steps 2000                   # train
    python train.py --data data --steps 4000                   # resumes + extends
    python evaluate.py --ckpt-dir checkpoints                  # eval + sample
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from codeparrot_data import load_meta, train_dataset, val_dataset
from kimi_k3_gdn2 import KimiK3, KimiK3Config, count_params
from multi_latent_attention.moe import update_router_bias


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TrainConfig:
    data_dir: str = "data"
    ckpt_dir: str = "checkpoints"
    seq_len: int = 256  # must be a multiple of the GDN-2 chunk size (64)
    batch_size: int = 8
    steps: int = 2000
    # Optax schedule/optimizer
    lr: float = 3e-4
    warmup_steps: int = 100
    min_lr_frac: float = 0.1  # cosine floor = lr * min_lr_frac
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    b1: float = 0.9
    b2: float = 0.95
    # MoE router-bias balancing (outside the gradient)
    router_bias_lr: float = 1e-3
    # Cadences
    log_every: int = 10
    eval_every: int = 250
    eval_batches: int = 20
    ckpt_every: int = 250
    ckpt_keep: int = 3
    seed: int = 0


def build_model(vocab_size: int, seq_len: int, seed: int) -> KimiK3:
    """The K3-recreation model at its (tiny) defaults, sized for the data.

    max_seq_len is set to the training seq_len: it only sizes the MLA decode
    cache default, and evaluate.py generates within this window."""
    cfg = KimiK3Config(vocab_size=vocab_size, max_seq_len=seq_len)
    assert seq_len % cfg.gdn_chunk_size == 0, (
        f"seq_len must be a multiple of gdn_chunk_size={cfg.gdn_chunk_size}"
    )
    return KimiK3(cfg, rngs=nnx.Rngs(seed))


def build_optimizer(model: KimiK3, tc: TrainConfig) -> nnx.Optimizer:
    """AdamW with warmup-cosine schedule and global-norm clipping.

    Weight decay is masked to >=2-D parameters: RMSNorm gains, biases, the
    GDN-2 decay params (A_log/dt_bias), and the AttnRes pseudo-queries are
    1-D and must not be decayed (the pseudo-queries in particular are
    zero-initialized — decay would fight their growth from zero).
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=tc.lr,
        warmup_steps=tc.warmup_steps,
        decay_steps=max(tc.steps, tc.warmup_steps + 1),
        end_value=tc.lr * tc.min_lr_frac,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(tc.grad_clip),
        optax.adamw(
            schedule,
            b1=tc.b1,
            b2=tc.b2,
            weight_decay=tc.weight_decay,
            mask=lambda params: jax.tree.map(lambda p: jnp.ndim(p) >= 2, params),
        ),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# --------------------------------------------------------------------------- #
#  Steps (jitted)
# --------------------------------------------------------------------------- #
def _ce(logits: jax.Array, labels: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(logp, labels[..., None], axis=-1).mean()


@nnx.jit
def train_step(
    model: KimiK3,
    optimizer: nnx.Optimizer,
    inputs: jax.Array,
    labels: jax.Array,
    router_bias_lr: float,
) -> dict[str, jax.Array]:
    """One optimization step; mutates model + optimizer in place (nnx)."""

    def loss_fn(model: KimiK3):
        logits, aux = model(inputs)
        ce = _ce(logits, labels)
        return ce + aux["aux_loss"], (ce, aux)

    (loss, (ce, aux)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)

    # Aux-loss-free load balancing, OUTSIDE the gradient: nudge each MoE
    # layer's selection bias toward uniform expert load (DeepSeek-V3 style).
    for i, layer in enumerate(model.layers):
        moe = layer.channel_mixer
        moe.router_bias.value = update_router_bias(
            moe.router_bias[...], aux["group_sizes"][i], lr=router_bias_lr
        )

    return {"loss": loss, "ce": ce, "aux_loss": aux["aux_loss"]}


@nnx.jit
def eval_step(model: KimiK3, inputs: jax.Array, labels: jax.Array) -> jax.Array:
    """Cross-entropy on one batch (no aux loss — it's a training regularizer)."""
    logits, _ = model(inputs)
    return _ce(logits, labels)


def evaluate(model: KimiK3, val_ds, n_batches: int) -> float:
    """Mean CE over the first n_batches of the deterministic val set."""
    total, n = 0.0, 0
    for i in range(min(n_batches, len(val_ds))):
        b = val_ds[i]
        total += float(eval_step(model, jnp.asarray(b["inputs"]), jnp.asarray(b["labels"])))
        n += 1
    return total / max(n, 1)


# --------------------------------------------------------------------------- #
#  Orbax checkpointing of the (model, optimizer) pair
# --------------------------------------------------------------------------- #
def make_ckpt_manager(ckpt_dir: str, tc: TrainConfig) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        pathlib.Path(ckpt_dir).absolute(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=tc.ckpt_keep,
            save_interval_steps=tc.ckpt_every,
        ),
    )


def save_ckpt(mngr, step: int, model, optimizer) -> None:
    state = nnx.state((model, optimizer))
    mngr.save(step, args=ocp.args.StandardSave(nnx.to_pure_dict(state)))


def restore_ckpt(mngr, step: int, model, optimizer) -> None:
    """Restore IN PLACE into freshly built (model, optimizer)."""
    state = nnx.state((model, optimizer))
    pure = mngr.restore(step, args=ocp.args.StandardRestore(nnx.to_pure_dict(state)))
    nnx.replace_by_pure_dict(state, pure)
    nnx.update((model, optimizer), state)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Train the Kimi K3 recreation on CodeParrot")
    for f in dataclasses.fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        p.add_argument(flag, type=type(f.default), default=f.default)
    tc = TrainConfig(**vars(p.parse_args()))

    meta = load_meta(tc.data_dir)
    print(f"data: {meta['dataset']} | vocab {meta['vocab_size']} "
          f"| {meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens")

    model = build_model(meta["vocab_size"], tc.seq_len, tc.seed)
    optimizer = build_optimizer(model, tc)
    print(f"model: {count_params(model):,} params "
          f"({model.cfg.n_layers} layers, d_model {model.cfg.d_model})")

    train_ds = train_dataset(tc.data_dir, tc.seq_len, tc.batch_size, tc.seed)
    val_ds = val_dataset(tc.data_dir, tc.seq_len, tc.batch_size)

    mngr = make_ckpt_manager(tc.ckpt_dir, tc)
    start = 0
    if mngr.latest_step() is not None:
        start = mngr.latest_step() + 1
        restore_ckpt(mngr, mngr.latest_step(), model, optimizer)
        print(f"resumed from checkpoint step {mngr.latest_step()}")

    # Sidecar metadata so evaluate.py can rebuild the exact model.
    ckpt_dir = pathlib.Path(tc.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "run.json").write_text(json.dumps({
        "model_config": dataclasses.asdict(model.cfg),
        "data_meta": meta,
        "train_config": dataclasses.asdict(tc),
    }, indent=2))

    tokens_per_step = tc.batch_size * tc.seq_len
    t0, tok0 = time.perf_counter(), 0
    for step in range(start, tc.steps):
        b = train_ds[step]  # Grain: deterministic batch for this global step
        m = train_step(
            model, optimizer,
            jnp.asarray(b["inputs"]), jnp.asarray(b["labels"]),
            tc.router_bias_lr,
        )
        tok0 += tokens_per_step

        if step % tc.log_every == 0 or step == tc.steps - 1:
            dt = time.perf_counter() - t0
            ce = float(m["ce"])
            print(f"step {step:6d} | loss {float(m['loss']):.4f} | ce {ce:.4f} "
                  f"| ppl {jnp.exp(ce):.1f} | aux {float(m['aux_loss']):.4f} "
                  f"| {tok0 / max(dt, 1e-9):,.0f} tok/s")
            t0, tok0 = time.perf_counter(), 0

        if tc.eval_every and step and step % tc.eval_every == 0:
            vce = evaluate(model, val_ds, tc.eval_batches)
            print(f"step {step:6d} | VAL ce {vce:.4f} | VAL ppl {jnp.exp(vce):.1f}")

        if step % tc.ckpt_every == 0 or step == tc.steps - 1:
            save_ckpt(mngr, step, model, optimizer)

    mngr.wait_until_finished()
    vce = evaluate(model, val_ds, tc.eval_batches)
    print(f"final | VAL ce {vce:.4f} | VAL ppl {jnp.exp(vce):.1f}")
    print(f"checkpoints in {ckpt_dir.absolute()}")


if __name__ == "__main__":
    main()
