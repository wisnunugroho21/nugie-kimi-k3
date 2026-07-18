"""
Evaluation pipeline for the Kimi K3 recreation.

Loads the latest (or a chosen) Orbax checkpoint written by train.py, rebuilds
the exact model from the run.json sidecar, and reports:

  * validation cross-entropy and perplexity over the packed CodeParrot val set
    (deterministic Grain batches — the same numbers every run), and
  * optionally, a greedy code sample continued from --prompt, exercising the
    model's streaming decode path (GDN-2 recurrent state + MLA latent cache).

USAGE
-----
    python evaluate.py --ckpt-dir checkpoints                  # full val set
    python evaluate.py --ckpt-dir checkpoints --max-batches 50
    python evaluate.py --ckpt-dir checkpoints --prompt "def add(a, b):" \
                       --sample-tokens 64
"""

from __future__ import annotations

import argparse
import json
import pathlib

import flax.nnx as nnx
import jax.numpy as jnp

from codeparrot_data import val_dataset
from kimi_k3_gdn2 import KimiK3, KimiK3Config, count_params
from train import (
    TrainConfig,
    build_optimizer,
    eval_step,
    make_ckpt_manager,
    restore_ckpt,
)


def load_run(ckpt_dir: str) -> tuple[KimiK3, nnx.Optimizer, dict, TrainConfig, int]:
    """Rebuild (model, optimizer) from run.json and restore the checkpoint.

    The optimizer is rebuilt only because the checkpoint stores the
    (model, optimizer) pair as one tree; its state is restored and discarded.
    Returns (model, optimizer, data_meta, train_config, restored_step).
    """
    run = json.loads((pathlib.Path(ckpt_dir) / "run.json").read_text())
    tc = TrainConfig(**run["train_config"])
    model = KimiK3(KimiK3Config(**run["model_config"]), rngs=nnx.Rngs(tc.seed))
    optimizer = build_optimizer(model, tc)

    mngr = make_ckpt_manager(ckpt_dir, tc)
    step = mngr.latest_step()
    if step is None:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    restore_ckpt(mngr, step, model, optimizer)
    return model, optimizer, run["data_meta"], tc, step


def evaluate_full(model: KimiK3, tc: TrainConfig, max_batches: int | None) -> float:
    """Mean CE over the (deterministic, unshuffled) val set."""
    val_ds = val_dataset(tc.data_dir, tc.seq_len, tc.batch_size)
    n = len(val_ds) if max_batches is None else min(max_batches, len(val_ds))
    total = 0.0
    for i in range(n):
        b = val_ds[i]
        total += float(
            eval_step(model, jnp.asarray(b["inputs"]), jnp.asarray(b["labels"]))
        )
    return total / max(n, 1)


def sample(model: KimiK3, meta: dict, prompt: str, n_tokens: int) -> str:
    """Greedy continuation of `prompt` through the streaming decode path."""
    from transformers import AutoTokenizer  # heavy import, sampling-only

    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    ids = jnp.asarray([tok(prompt).input_ids], jnp.int32)
    out = model.generate(ids, max_new_tokens=n_tokens)
    return prompt + tok.decode(list(out[0]))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the Kimi K3 recreation")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap on val batches (default: full val set)")
    p.add_argument("--prompt", default=None,
                   help="if set, greedy-decode a code sample from this prompt")
    p.add_argument("--sample-tokens", type=int, default=64)
    a = p.parse_args()

    model, _, meta, tc, step = load_run(a.ckpt_dir)
    print(f"restored step {step} | {count_params(model):,} params "
          f"| val set: {meta['val_tokens']:,} tokens of {meta['dataset']}")

    ce = evaluate_full(model, tc, a.max_batches)
    print(f"val cross-entropy {ce:.4f} | perplexity {jnp.exp(ce):.2f} "
          f"| bits/token {ce / jnp.log(2):.3f}")

    if a.prompt is not None:
        print("--- sample " + "-" * 50)
        print(sample(model, meta, a.prompt, a.sample_tokens))
        print("-" * 61)


if __name__ == "__main__":
    main()
