"""
Training pipeline for the Kimi K3 recreation: Grain data, Optax optimizer,
Orbax checkpointing.

WHAT ONE TRAINING STEP DOES
---------------------------
    batch = train_ds[step]                       # Grain: pure function of (seed, step)
    loss  = CE(logits, labels) + Σ_layers aux_loss   # MoE Switch-style aux loss
    grads -> Muon + AdamW (optax.contrib.muon; warmup-cosine LR, global-norm
             clip): Newton-Schulz-orthogonalized momentum for the MATRIX
             parameters — per-expert for the stacked MoE weights — and AdamW
             for embeddings/LM head/1-D params, the Moonlight / Kimi K2 recipe
             (see build_optimizer)
    router_bias[e] += lr_bias * sign(fair_share - load[e])   # per MoE layer,
             OUTSIDE the gradient: DeepSeek-V3 aux-loss-free load balancing
             (see multi_latent_attention/moe.py::update_router_bias)

Because the Grain batch at any step is a pure function of (seed, step), resume
needs no data-iterator state: Orbax restores (model, optimizer) and training
continues at latest_step + 1 reading `train_ds[step]` — bit-identical to a run
that never stopped.

MULTI-GPU (e.g. Kaggle's 2x T4): plain DATA PARALLELISM via GSPMD — parameters
are replicated across all local devices and each global batch is split along
its batch axis (`setup_data_parallel`); the jitted step then runs partitioned
automatically, with XLA inserting the gradient all-reduce. batch_size is the
GLOBAL batch and must be divisible by jax.device_count(). On one device this
degenerates to a trivial 1-device mesh (a no-op).

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
import os
import pathlib
import shutil
import subprocess
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import optax.contrib
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding, PartitionSpec
from optax.contrib import MuonDimensionNumbers

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
    # Optax schedule/optimizer (Muon + AdamW; one lr serves both — see
    # build_optimizer's consistent-RMS note)
    lr: float = 3e-4
    warmup_steps: int = 100
    min_lr_frac: float = 0.1  # cosine floor = lr * min_lr_frac
    weight_decay: float = 0.1  # on the Muon (matrix) side; AdamW side undecayed
    grad_clip: float = 1.0
    muon_beta: float = 0.95  # Muon momentum decay
    b1: float = 0.9  # AdamW side (embeddings / LM head / 1-D params)
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


def _muon_dim_numbers(params):
    """Per-leaf Muon/AdamW partition for optax.contrib.muon.

    A MuonDimensionNumbers leaf routes the parameter to Muon (declaring which
    axes form the matrix — every other axis is vmapped as a batch axis);
    None routes it to the AdamW side. The split follows the Muon/Moonlight
    recipe, which Kimi K2 scaled up:

      Muon (0, 1):  every plain 2-D kernel — attention/MoE projections,
                    routers, low-rank factors, shared-expert matrices.
      Muon (1, 2):  the stacked MoE expert weights [E, in, out] — the expert
                    axis 0 becomes a batch axis, so each expert's matrix is
                    orthogonalized INDEPENDENTLY.
      AdamW:        embeddings and the LM head (2-D, but excluded by the Muon
                    authors — they are lookup tables, not linear maps), every
                    1-D parameter (norm gains, biases, A_log/dt_bias, the
                    zero-initialized AttnRes pseudo-queries), and the [W, 1, C]
                    depthwise short-conv kernels.
    """
    def rule(path, p):
        name = "/".join(str(getattr(k, "key", k)) for k in path)
        if "embed" in name or "lm_head" in name:
            return None
        if p.ndim == 2:
            return MuonDimensionNumbers()  # matrix as-is: (reduction 0, output 1)
        if p.ndim == 3 and "channel_mixer" in name and "conv" not in name:
            return MuonDimensionNumbers(1, 2)  # [E, in, out]: per-expert
        return None

    return jax.tree_util.tree_map_with_path(rule, params)


def build_optimizer(model: KimiK3, tc: TrainConfig) -> nnx.Optimizer:
    """Muon + AdamW (the Moonlight / Kimi K2 optimizer recipe) with a
    warmup-cosine schedule and global-norm clipping.

    Muon orthogonalizes each matrix parameter's momentum with a Newton-Schulz
    iteration; non-matrix parameters fall through to AdamW (partition in
    `_muon_dim_numbers`). consistent_rms=0.2 applies Moonlight's update-RMS
    matching (arXiv:2502.16982): Muon updates are rescaled by
    0.2·sqrt(max(fan_in, fan_out)) so they land at AdamW-like RMS — which is
    what lets ONE learning rate (and schedule) drive both partitions.

    Weight decay: applied on the Muon side (all matrices). The AdamW side is
    left undecayed — it holds exactly the parameters the previous AdamW-only
    setup masked out of decay, plus the embeddings/LM head (which the Muon
    reference setup also leaves undecayed).
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
        optax.contrib.muon(
            learning_rate=schedule,
            beta=tc.muon_beta,
            weight_decay=tc.weight_decay,
            consistent_rms=0.2,  # Moonlight RMS matching -> shared lr
            adam_b1=tc.b1,
            adam_b2=tc.b2,
            muon_weight_dimension_numbers=_muon_dim_numbers,
        ),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def gpu_utilization() -> str:
    """Per-GPU utilization/memory sampled from nvidia-smi, for the step log.

    This is the ground truth for "is the second GPU actually computing":
    utilization.gpu is the fraction of the last sample window in which a kernel
    was executing on that device. Returns '' on machines without nvidia-smi.
    Under healthy data parallelism every GPU shows similar, nonzero numbers.
    """
    if shutil.which("nvidia-smi") is None:
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        parts = []
        for line in out.strip().splitlines():
            idx, util, mem = (x.strip() for x in line.split(","))
            parts.append(f"{idx}:{util}%/{int(mem):,}MiB")
        return " | gpu " + " ".join(parts) if parts else ""
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
#  Data parallelism (GSPMD): replicate params, shard the batch axis
# --------------------------------------------------------------------------- #
def setup_data_parallel(model, optimizer) -> NamedSharding:
    """Replicate (model, optimizer) across all local devices; return the batch
    sharding. GSPMD then partitions the jitted train/eval steps automatically:
    each device computes its batch shard, XLA all-reduces the gradients. With
    one device this is a trivial no-op mesh."""
    # AxisType.Auto: classic GSPMD propagation. (The default explicit
    # sharding-in-types mode rejects the embedding gather over sharded token
    # indices as ambiguous; Auto lets XLA resolve it.)
    mesh = jax.make_mesh(
        (jax.device_count(),), ("data",), axis_types=(jax.sharding.AxisType.Auto,)
    )
    state = nnx.state((model, optimizer))
    state = jax.device_put(state, NamedSharding(mesh, PartitionSpec()))
    nnx.update((model, optimizer), state)
    return NamedSharding(mesh, PartitionSpec("data"))


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


def evaluate(
    model: KimiK3, val_ds, n_batches: int, sharding: NamedSharding | None = None
) -> float:
    """Mean CE over the first n_batches of the deterministic val set.
    `sharding` (from setup_data_parallel) splits eval batches across devices."""
    total, n = 0.0, 0
    for i in range(min(n_batches, len(val_ds))):
        b = val_ds[i]
        x, y = jnp.asarray(b["inputs"]), jnp.asarray(b["labels"])
        if sharding is not None:
            x, y = jax.device_put((x, y), sharding)
        total += float(eval_step(model, x, y))
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


def save_ckpt(mngr: ocp.CheckpointManager, step: int, model, optimizer, *, force: bool = False) -> None:
    """force=True bypasses save_interval_steps — used for the final step, which
    otherwise gets skipped whenever (steps-1) isn't a multiple of ckpt_every."""
    state = nnx.state((model, optimizer))
    mngr.save(step, args=ocp.args.StandardSave(nnx.to_pure_dict(state)), force=force)
    mngr.wait_until_finished()


def restore_ckpt(mngr: ocp.CheckpointManager, step: int, model, optimizer) -> None:
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

    # Data parallelism (no-op on a single device). AFTER restore, so the
    # restored state is what gets replicated.
    n_dev = jax.device_count()
    assert tc.batch_size % n_dev == 0, (
        f"batch_size ({tc.batch_size}) must be divisible by device count ({n_dev})"
    )
    batch_sharding = setup_data_parallel(model, optimizer)
    print(f"devices: {n_dev} x {jax.devices()[0].platform.upper()} "
          f"| per-device batch {tc.batch_size // n_dev} "
          f"| jax {jax.__version__} "
          f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    for d in jax.devices():
        print(f"  {d.id}: {d.device_kind}")
    if n_dev == 1 and jax.devices()[0].platform == "gpu":
        # The common multi-GPU failure mode is silent: JAX enumerates ONE CUDA
        # device (stale CUDA_VISIBLE_DEVICES, broken jax[cuda12] install, ...)
        # and trains happily on it, while other monitors still show memory
        # reserved on every physical GPU. Say so explicitly.
        print("WARNING: JAX sees only ONE GPU. If this machine has more "
              "(e.g. Kaggle T4 x2), check CUDA_VISIBLE_DEVICES and the "
              "jax[cuda12] install — training will use just this device.")

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
        inputs, labels = jax.device_put(
            (jnp.asarray(b["inputs"]), jnp.asarray(b["labels"])), batch_sharding
        )
        m = train_step(model, optimizer, inputs, labels, tc.router_bias_lr)
        tok0 += tokens_per_step

        if step == start:
            # One-time proof of where the work actually lives: the batch shard
            # placement and each device's live memory. Under data parallelism
            # every device must appear below with a similar bytes-in-use.
            devs = sorted({d.id for d in inputs.sharding.device_set})
            print(f"batch sharded over devices {devs} "
                  f"({inputs.sharding.shard_shape(inputs.shape)[0]} rows each)")
            for d in jax.local_devices():
                stats = d.memory_stats() or {}
                if "bytes_in_use" in stats:
                    print(f"  device {d.id} in use: "
                          f"{stats['bytes_in_use'] / 2**20:,.0f} MiB")

        if step % tc.log_every == 0 or step == tc.steps - 1:
            # NOTE: float(...) blocks on the step, so the utilization sampled
            # right after reflects the steady-state training just executed.
            dt = time.perf_counter() - t0
            ce = float(m["ce"])
            print(f"step {step:6d} | loss {float(m['loss']):.4f} | ce {ce:.4f} "
                  f"| ppl {jnp.exp(ce):.1f} | aux {float(m['aux_loss']):.4f} "
                  f"| {tok0 / max(dt, 1e-9):,.0f} tok/s{gpu_utilization()}")
            t0, tok0 = time.perf_counter(), 0

        if tc.eval_every and step and step % tc.eval_every == 0:
            vce = evaluate(model, val_ds, tc.eval_batches, batch_sharding)
            print(f"step {step:6d} | VAL ce {vce:.4f} | VAL ppl {jnp.exp(vce):.1f}")

        if step % tc.ckpt_every == 0 or step == tc.steps - 1:
            save_ckpt(mngr, step, model, optimizer, force=step == tc.steps - 1)

    mngr.wait_until_finished()
    vce = evaluate(model, val_ds, tc.eval_batches, batch_sharding)
    print(f"final | VAL ce {vce:.4f} | VAL ppl {jnp.exp(vce):.1f}")
    print(f"checkpoints in {ckpt_dir.absolute()}")


if __name__ == "__main__":
    main()
