"""
Stage 3: the training loop.

Ties everything together:
    Grain batches  ->  KimiK3 (GDN-2)  ->  CE + MoE aux loss  ->  Muon/AdamW (Optax)
    ->  aux-loss-free router-bias nudge  ->  Orbax checkpoint.

LOSS
    Next-token cross-entropy on the shifted targets, PLUS the summed MoE
    load-balancing aux loss the model already returns. The router-bias update
    (DeepSeek-V3 aux-loss-free balancing) is a NON-gradient step applied after each
    optimizer update, nudging each layer's per-expert selection bias toward uniform
    load using that step's realized `group_sizes`.

OPTIMIZER
    The Moonlight Muon/AdamW split (pipeline/optimizer.py): hidden weight matrices
    get Muon (orthogonalized momentum with consistent-RMS scaling), while the
    embedding, LM head, biases, norm gains, and the GDN-2 A_log / dt_bias decay
    parameters get AdamW. One linear warmup + cosine decay schedule and global-norm
    gradient clipping cover both sides; weight decay touches only the Muon matrices.

MIXED PRECISION
    Governed entirely by model.compute_dtype (fp32 by default; set "bfloat16" on a
    GPU). Master weights stay fp32; logits/loss are fp32 for a stable softmax.

MULTI-GPU (DATA PARALLEL)
    Auto-detected from jax.device_count(): parameters + optimizer state are
    REPLICATED across all visible devices, and each global batch is SHARDED along its
    leading axis (so every GPU processes batch_size/n_devices examples). This is pure
    GSPMD data parallelism — no code path branches on device count; a single device is
    just the degenerate replicate-over-1 case. batch_size must be divisible by the
    number of devices (checked at startup). Used by the 2x-T4 (Kaggle) config.

Run:  python -m pipeline.train --config configs/tiny.yaml [--resume]
"""

from __future__ import annotations

import argparse
import math
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# `shard_map` became a top-level `jax.shard_map` in recent JAX; fall back to its
# experimental location on older releases.
import inspect

try:
    from jax import shard_map as _shard_map_impl
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map as _shard_map_impl

# Disable shard_map's varying-axis (a.k.a. replication) type checker: the GDN-2 token
# mixer runs an internal `lax.scan` whose carry is varying over the "data" axis, which
# the checker rejects (scan-vma). We own out_specs correctness explicitly via the
# pmean/psum collectives below, so turning the check off is safe. The kwarg was renamed
# check_rep -> check_vma across JAX versions; pick whichever this build exposes.
_CHECK_KW = ("check_vma" if "check_vma" in inspect.signature(_shard_map_impl).parameters
             else "check_rep")


def shard_map(f, **kwargs):
    return _shard_map_impl(f, **kwargs, **{_CHECK_KW: False})


from kimi_k3_gdn2 import KimiK3, count_params
from multi_latent_attention.moe import update_router_bias
from pipeline import data as data_mod
from pipeline.checkpointing import CheckpointManager
from pipeline.config import ExperimentConfig
from pipeline.optimizer import make_optimizer


# --------------------------------------------------------------------------- #
#  Model / optimizer construction (shared with evaluate.py).
# --------------------------------------------------------------------------- #
def build_model(cfg: ExperimentConfig, rngs: nnx.Rngs) -> KimiK3:
    return KimiK3(cfg.model, rngs=rngs)


def build_schedule(tc) -> optax.Schedule:
    """Linear warmup to `lr`, then cosine decay to `min_lr` over the run."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=tc.lr,
        warmup_steps=tc.warmup_steps,
        decay_steps=max(tc.max_steps, tc.warmup_steps + 1),
        end_value=tc.min_lr,
    )


def build_optimizer(model: KimiK3, cfg: ExperimentConfig) -> nnx.Optimizer:
    """Global-norm clip -> Muon (hidden weight matrices) / AdamW (everything else),
    the Moonlight recipe (see pipeline/optimizer.py). Muon's consistent-RMS scaling
    keeps the same LR scale as the old plain-AdamW setup, so tc.lr is unchanged; weight
    decay now touches only the Muon-side matrices (not embed/head/norms/biases)."""
    tc = cfg.train
    return make_optimizer(
        model,
        build_schedule(tc),
        weight_decay=tc.weight_decay,
        clip_norm=tc.grad_clip,
        adam_b1=tc.beta1,
        adam_b2=tc.beta2,
        eps=tc.eps,
    )


# --------------------------------------------------------------------------- #
#  Loss.
# --------------------------------------------------------------------------- #
def loss_fn(model: KimiK3, batch: dict[str, jax.Array]):
    """Returns (total_loss, (ce_loss, aux_loss, group_sizes)). total = CE + MoE aux."""
    logits, aux = model(batch["input_ids"])  # logits fp32 [B,L,V]
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["target_ids"]).mean()
    total = ce + aux["aux_loss"]
    return total, (ce, aux["aux_loss"], aux["group_sizes"])


# Filter used to split the model into (graphdef, params, everything-else) for the
# functional split/merge inside shard_map. `nnx.Param` are the differentiated weights;
# the `...` catch-all keeps the rest (RMSNorm state, the MoE router_bias Variable, ...).
_PARAM_FILTER = (nnx.Param, ...)


def make_train_step(mesh: Mesh):
    """Build the data-parallel train step bound to `mesh` (single axis "data").

    The forward + backward runs inside `shard_map`, so each device processes ONLY its
    shard of the batch and the MoE's token dispatch (argsort / ragged_dot / scatter-add
    over the B*L token axis, see multi_latent_attention/moe.py) stays DEVICE-LOCAL.
    That global permutation is unshardable along the batch axis, so plain GSPMD auto-
    partitioning all-gathers every token onto one device — silently collapsing data
    parallelism and OOMing large batches. Manual sharding keeps the experts replicated
    and routes each device's own tokens instead.

    Collectives across the "data" axis re-sync the replicas: gradients are averaged
    (pmean) and the per-expert token counts are summed (psum) so every replica applies
    an identical optimizer step AND identical router-bias nudge, keeping the replicated
    params/router_bias bit-for-bit in sync.
    """

    @nnx.jit
    def train_step(model: KimiK3, optimizer: nnx.Optimizer,
                   batch: dict[str, jax.Array], router_bias_lr: float):
        graphdef, params, rest = nnx.split(model, *_PARAM_FILTER)

        def _dp(params, rest, batch):
            # Per-device forward/backward over this shard's local tokens.
            def _loss(p):
                return loss_fn(nnx.merge(graphdef, p, rest), batch)
            (total, (ce, aux_loss, group_sizes)), grads = jax.value_and_grad(
                _loss, has_aux=True)(params)
            # Re-sync replicas: average grads/loss, SUM the per-expert counts so the
            # router-bias update sees the global batch's load (not one shard's).
            grads = jax.lax.pmean(grads, "data")
            total = jax.lax.pmean(total, "data")
            ce = jax.lax.pmean(ce, "data")
            aux_loss = jax.lax.pmean(aux_loss, "data")
            group_sizes = jax.lax.psum(group_sizes, "data")
            return grads, total, ce, aux_loss, group_sizes

        grads, total, ce, aux_loss, group_sizes = shard_map(
            _dp, mesh=mesh,
            in_specs=(P(), P(), P("data")),      # params/rest replicated; batch sharded
            out_specs=(P(), P(), P(), P(), P()),  # all outputs already replica-reduced
        )(params, rest, batch)

        # Grads/counts are identical across replicas now, so these replicated updates
        # keep params + router_bias in sync on every device.
        optimizer.update(model, grads)
        for i, layer in enumerate(model.layers):
            moe = layer.channel_mixer
            moe.router_bias.set_value(
                update_router_bias(
                    moe.router_bias.get_value(), group_sizes[i], router_bias_lr)
            )

        return total, ce, aux_loss

    return train_step


def make_eval_step(mesh: Mesh):
    """Build the data-parallel eval step bound to `mesh` (mirrors make_train_step).
    Runs the forward under shard_map for the same reason (device-local MoE dispatch,
    no all-gather / OOM), then psums the CE sum and token count to global totals."""

    @nnx.jit
    def eval_step(model: KimiK3, batch: dict[str, jax.Array]):
        graphdef, params, rest = nnx.split(model, *_PARAM_FILTER)

        def _fwd(params, rest, batch):
            logits, _ = nnx.merge(graphdef, params, rest)(batch["input_ids"])
            tok_ce = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch["target_ids"])  # [B_local, L]
            ce_sum = jax.lax.psum(tok_ce.sum(), "data")
            n = jax.lax.psum(jnp.array(tok_ce.size, jnp.float32), "data")
            return ce_sum, n

        return shard_map(
            _fwd, mesh=mesh,
            in_specs=(P(), P(), P("data")),
            out_specs=(P(), P()),
        )(params, rest, batch)

    return eval_step


def evaluate_loss(model: KimiK3, eval_step, val_iter, steps: int,
                  shard=None) -> dict[str, float]:
    """Mean CE / perplexity over `steps` val batches. Sets the model to eval mode so
    any train-only behavior is disabled (harmless here; good hygiene). `shard`, if
    given, places each batch on the data-parallel sharding used by the params."""
    model.eval()
    tot_ce, tot_tok = 0.0, 0.0
    for _ in range(steps):
        batch = _to_jax(next(val_iter))
        if shard is not None:
            batch = shard(batch)
        ce_sum, n = eval_step(model, batch)
        tot_ce += float(ce_sum)
        tot_tok += float(n)
    model.train()
    mean_ce = tot_ce / max(tot_tok, 1.0)
    return {"val_loss": mean_ce, "val_ppl": math.exp(mean_ce)}


def _to_jax(batch: dict) -> dict[str, jax.Array]:
    return {k: jnp.asarray(v) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
#  Training loop.
# --------------------------------------------------------------------------- #
def train(cfg: ExperimentConfig, resume: bool = False) -> None:
    tc = cfg.train
    print(f"JAX devices: {jax.devices()}")

    # --- data ---
    meta = data_mod.load_meta(cfg.data.data_dir)
    if meta["vocab_size"] != cfg.model.vocab_size:
        raise ValueError(
            f"model.vocab_size ({cfg.model.vocab_size}) != tokenized vocab "
            f"({meta['vocab_size']}). Fix the YAML so they match.")
    train_iter = data_mod.make_loader(
        cfg.data.data_dir, "train", cfg.data.seq_len, tc.batch_size,
        shuffle=True, repeat=True, seed=cfg.data.shuffle_buffer_seed,
        num_workers=cfg.data.num_workers)

    def make_val_iter():
        # Rebuilt for EVERY in-training eval so each one measures the same
        # leading eval_steps batches of the val split. A single shared
        # repeating iterator would give every eval a different window,
        # adding data noise to the val-loss curve that isn't model noise.
        return data_mod.make_loader(
            cfg.data.data_dir, "val", cfg.data.seq_len, tc.batch_size,
            shuffle=False, repeat=True, seed=0, num_workers=0)

    # --- model + optimizer ---
    # Keep the Rngs object around (not just the seed): it is checkpointed alongside
    # the model/optimizer so a resumed run continues the same random stream.
    rngs = nnx.Rngs(tc.seed)
    model = build_model(cfg, rngs)
    optimizer = build_optimizer(model, cfg)
    print(f"Model params: {count_params(model):,}  "
          f"(compute_dtype={cfg.model.compute_dtype}, seq_len={cfg.data.seq_len})")

    # --- data-parallel sharding (works for 1 device too) ---
    devices = jax.devices()
    n_dev = len(devices)
    if tc.batch_size % n_dev != 0:
        raise ValueError(
            f"batch_size ({tc.batch_size}) must be divisible by the number of "
            f"devices ({n_dev}) for data-parallel training.")
    # Manual (shard_map) data parallelism: one "data" axis over all devices. train/eval
    # steps run the model forward INSIDE shard_map so the MoE's global token dispatch
    # stays device-local (GSPMD auto-partitioning would all-gather it onto one GPU —
    # collapsing the second device and OOMing the batch). See make_train_step.
    mesh = Mesh(np.asarray(devices), ("data",))
    data_shard = NamedSharding(mesh, P("data"))  # split batch across devices
    repl_shard = NamedSharding(mesh, P())        # replicate params/opt state
    # Replicate the model params, the Adam optimizer state, and the rng stream across
    # all devices. (In this Flax version nnx.Optimizer does NOT nest the model, so the
    # two must be replicated explicitly.) This also fixes the sharding of the abstract
    # targets used on checkpoint restore below.
    for obj in (model, optimizer, rngs):
        nnx.update(obj, jax.device_put(nnx.state(obj), repl_shard))
    shard_batch = lambda b: jax.device_put(b, data_shard)  # noqa: E731
    train_step = make_train_step(mesh)
    eval_step = make_eval_step(mesh)
    if n_dev > 1:
        print(f"Data-parallel over {n_dev} devices "
              f"({tc.batch_size // n_dev} examples/device).")

    # --- checkpoint manager (+ optional resume) ---
    ckpt = CheckpointManager(tc.ckpt_dir, keep=tc.keep_checkpoints)
    start_step = 0
    if resume and ckpt.latest_step() is not None:
        restored_step, train_iter = ckpt.restore(
            model=model, optimizer=optimizer, rngs=rngs, train_iterator=train_iter)
        start_step = restored_step + 1
        print(f"Resumed from step {restored_step}")

    # --- loop ---
    schedule = build_schedule(tc)
    t0 = time.time()
    tokens_per_step = tc.batch_size * cfg.data.seq_len
    running_ce = None  # device-side accumulator, host-synced only at log_every
    for step in range(start_step, tc.max_steps):
        batch = shard_batch(_to_jax(next(train_iter)))
        total, ce, aux_loss = train_step(model, optimizer, batch, tc.router_bias_lr)
        # Keep ce on device: calling float(ce) every step would block the host
        # on that step's result and defeat JAX's async dispatch (the device
        # could no longer run ahead while the host prepares the next batch).
        running_ce = ce if running_ce is None else running_ce + ce

        if (step + 1) % tc.log_every == 0:
            mean_ce = float(running_ce) / tc.log_every  # the one host sync
            dt = time.time() - t0
            tok_s = tokens_per_step * tc.log_every / dt
            lr = float(schedule(step))
            print(f"step {step + 1:>7}/{tc.max_steps} | loss {mean_ce:6.4f} | "
                  f"ppl {math.exp(mean_ce):8.2f} | aux {float(aux_loss):.4f} "
                  f"| lr {lr:.2e} | {tok_s:,.0f} tok/s", flush=True)
            running_ce, t0 = None, time.time()

        if (step + 1) % tc.eval_every == 0:
            m = evaluate_loss(model, eval_step, make_val_iter(), tc.eval_steps,
                              shard=shard_batch)
            print(f"  [eval] step {step + 1} | val_loss {m['val_loss']:.4f} | "
                  f"val_ppl {m['val_ppl']:.2f}", flush=True)
            t0 = time.time()  # don't count eval time against tok/s

        if (step + 1) % tc.save_every == 0:
            ckpt.save(step, model=model, optimizer=optimizer, rngs=rngs,
                      train_iterator=train_iter)
            # Block until the async save commits BEFORE training resumes. Orbax pins the
            # full fp32 optimizer state (params + Adam m/v) on-device until the save
            # finishes; letting the next steps run concurrently stacks their working set
            # on top of it and OOMs small GPUs (e.g. a 15 GB T4). Waiting here serializes
            # save vs. train — a few idle seconds every save_every steps, no memory spike.
            ckpt.wait_until_finished()
            print(f"  [ckpt] saved step {step}", flush=True)
            t0 = time.time()  # don't count save time against tok/s

    # Final checkpoint — unless the loop's last in-loop save already covered this
    # step (max_steps a multiple of save_every), where re-saving the same step
    # would raise StepAlreadyExistsError in Orbax.
    if ckpt.latest_step() != tc.max_steps - 1:
        ckpt.save(tc.max_steps - 1, model=model, optimizer=optimizer, rngs=rngs,
                  train_iterator=train_iter)
        ckpt.wait_until_finished()
    print(f"Training complete. Final checkpoint at step {tc.max_steps - 1}.")
    ckpt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Kimi-K3-GDN2 on CodeParrot.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint in train.ckpt_dir.")
    args = ap.parse_args()
    train(ExperimentConfig.load(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
