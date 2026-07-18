"""
Stage 4: evaluation & sampling from a trained checkpoint.

Restores the latest (or a chosen) Orbax checkpoint and does two things:

  1. `--eval`     full-corpus validation loss + perplexity over the held-out split
                  (or a capped number of batches via --max-batches).
  2. `--generate` autoregressive code completion from a text prompt, using the
                  model's streaming `generate` (GDN-2 fixed-size state + growing MLA
                  latent cache), decoded back to text with the training tokenizer.

Run:
    python -m pipeline.evaluate --config configs/tiny.yaml --eval
    python -m pipeline.evaluate --config configs/tiny.yaml --generate \
        --prompt "def fibonacci(n):" --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from pipeline import data as data_mod
from pipeline.checkpointing import CheckpointManager
from pipeline.config import ExperimentConfig
from pipeline.tokenizer import build_tokenizer
from pipeline.train import build_model, make_eval_step


def load_trained(cfg: ExperimentConfig, step: int | None = None):
    """Rebuild the model skeleton and restore its checkpointed weights.
    Returns (model, restored_step). Evaluation only needs the `model` item of the
    composite checkpoint; the optimizer/rngs/train_iterator items are left untouched."""
    model = build_model(cfg, nnx.Rngs(cfg.train.seed))
    ckpt = CheckpointManager(cfg.train.ckpt_dir, keep=cfg.train.keep_checkpoints)
    restored_step, _ = ckpt.restore(step, model=model)
    ckpt.close()
    model.eval()
    return model, restored_step


# --------------------------------------------------------------------------- #
def run_eval(cfg: ExperimentConfig, step: int | None, max_batches: int | None) -> None:
    model, restored = load_trained(cfg, step)
    print(f"Restored step {restored}. Evaluating validation split...")

    total_windows = data_mod.num_windows(cfg.data.data_dir, "val", cfg.data.seq_len)
    n_batches = total_windows // cfg.train.batch_size
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)
    if n_batches == 0:
        raise ValueError("Not enough validation data for a single batch.")

    val_iter = data_mod.make_loader(
        cfg.data.data_dir, "val", cfg.data.seq_len, cfg.train.batch_size,
        shuffle=False, repeat=False, seed=0, num_workers=0)

    # Data-parallel eval matching train.py: replicate the params, shard each batch, and
    # run the forward inside shard_map so the MoE dispatch stays device-local (no all-
    # gather / OOM on multi-GPU). Degenerates to a plain single-device pass on 1 GPU.
    devices = jax.devices()
    n_dev = len(devices)
    if cfg.train.batch_size % n_dev != 0:
        raise ValueError(
            f"batch_size ({cfg.train.batch_size}) must be divisible by the number of "
            f"devices ({n_dev}).")
    mesh = Mesh(np.asarray(devices), ("data",))
    nnx.update(model, jax.device_put(nnx.state(model), NamedSharding(mesh, P())))
    data_shard = NamedSharding(mesh, P("data"))
    eval_step = make_eval_step(mesh)

    tot_ce, tot_tok = 0.0, 0.0
    for i in range(n_batches):
        batch = {k: jnp.asarray(v) for k, v in next(val_iter).items()}
        ce_sum, n = eval_step(model, jax.device_put(batch, data_shard))
        tot_ce += float(ce_sum)
        tot_tok += float(n)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_batches} batches...", flush=True)

    mean_ce = tot_ce / max(tot_tok, 1.0)
    print("\n=== Validation ===")
    print(f"  tokens     : {int(tot_tok):,}")
    print(f"  cross-ent  : {mean_ce:.4f} nats")
    print(f"  perplexity : {math.exp(mean_ce):.2f}")
    print(f"  bits/token : {mean_ce / math.log(2):.4f}")


# --------------------------------------------------------------------------- #
def run_generate(cfg: ExperimentConfig, step: int | None, prompt: str,
                 max_new_tokens: int, temperature: float = 0.0,
                 top_p: float = 1.0, seed: int = 0) -> None:
    model, restored = load_trained(cfg, step)
    meta = data_mod.load_meta(cfg.data.data_dir)
    tokenizer = build_tokenizer(
        meta.get("tokenizer", cfg.data.tokenizer), meta.get("tokenizer_name",
                                                            cfg.data.tokenizer_name))
    print(f"Restored step {restored}. Generating...\n")

    ids = tokenizer.encode(prompt) or [tokenizer.eos_id]
    prompt_ids = jnp.asarray(ids, jnp.int32)[None, :]  # [1, P]

    # The MLA latent cache spans the whole prompt+continuation. Truncate the request
    # to the model's declared context cap rather than clamping max_len below the
    # request — an undersized cache would make dynamic_update_slice silently
    # overwrite the last slot instead of erroring.
    budget = cfg.model.max_seq_len - len(ids)
    if budget <= 0:
        raise ValueError(
            f"Prompt ({len(ids)} tokens) already fills model.max_seq_len "
            f"({cfg.model.max_seq_len}); nothing can be generated.")
    if max_new_tokens > budget:
        print(f"WARNING: prompt ({len(ids)}) + max_new_tokens ({max_new_tokens}) "
              f"exceeds model.max_seq_len ({cfg.model.max_seq_len}); "
              f"truncating to {budget} new tokens.")
        max_new_tokens = budget
    max_len = len(ids) + max_new_tokens
    # Stop at the training EOS (the packed-stream document separator): once the
    # model closes the "document", further tokens would start an unrelated one.
    gen = model.generate(
        prompt_ids, max_new_tokens=max_new_tokens, max_len=max_len,
        temperature=temperature, top_p=top_p,
        eos_id=int(meta["eos_id"]), key=jax.random.PRNGKey(seed))
    continuation = tokenizer.decode(gen[0].tolist())

    print("=== Prompt ===")
    print(prompt)
    print("\n=== Continuation ===")
    print(continuation)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate / sample from a checkpoint.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--step", type=int, default=None,
                    help="Checkpoint step to load (default: latest).")
    ap.add_argument("--eval", action="store_true", help="Compute validation loss/ppl.")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Cap the number of val batches in --eval.")
    ap.add_argument("--generate", action="store_true", help="Sample a completion.")
    ap.add_argument("--prompt", default="def hello_world():\n",
                    help="Prompt text for --generate.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy decode; > 0 samples from "
                         "softmax(logits / temperature).")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="Nucleus sampling cutoff in (0, 1]; only used when "
                         "--temperature > 0. 1.0 disables the truncation.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Sampling seed for --temperature > 0.")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    if not (args.eval or args.generate):
        args.eval = True  # default action
    if args.eval:
        run_eval(cfg, args.step, args.max_batches)
    if args.generate:
        run_generate(cfg, args.step, args.prompt, args.max_new_tokens,
                     temperature=args.temperature, top_p=args.top_p,
                     seed=args.seed)


if __name__ == "__main__":
    main()
