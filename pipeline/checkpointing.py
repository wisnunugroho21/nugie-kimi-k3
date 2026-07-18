"""
Orbax checkpointing for the whole training state.

WHAT WE CHECKPOINT
    A single Orbax *composite* checkpoint with five named items, saved/restored
    together at each step:
        model           the KimiLinear array state (params + MoE router_bias + norm
                        scales) from `nnx.state(model)`.
        optimizer       the Optax optimizer state + nnx step counter from
                        `nnx.state(optimizer)`. In this Flax version `nnx.Optimizer`
                        does NOT nest the model, so the model weights above are saved
                        exactly once — the two items are disjoint.
        rngs            the `nnx.Rngs` stream state (key + counter) so resumed runs
                        continue the same random sequence.
        step            the training-loop step index (plain int, JSON).
        train_iterator  the Grain input iterator's position, via Grain's Orbax
                        handler, so resuming picks up the exact same data stream.
    `nnx.split`/`nnx.state` separate the static graph definition from the array
    state; only the array state is written. On restore we rebuild identical
    skeletons, hand each one's live state as the abstract target, and `nnx.update`
    the restored arrays back in place.

WHY A MANAGER
    Orbax's CheckpointManager gives us step-numbered checkpoints, automatic pruning
    to the last `keep` (train.keep_checkpoints), and latest-step discovery for
    resuming — all we add on top is the nnx / Grain split/merge glue.
"""

from __future__ import annotations

import os

import flax.nnx as nnx
import grain
import orbax.checkpoint as ocp


class CheckpointManager:
    def __init__(self, ckpt_dir: str, keep: int = 3):
        # Orbax requires an absolute path for its atomic-rename commits.
        self.directory = os.path.abspath(ckpt_dir)
        options = ocp.CheckpointManagerOptions(max_to_keep=keep, create=True)
        self.mgr = ocp.CheckpointManager(self.directory, options=options)

    def save(self, step: int, *, model: nnx.Module, optimizer: nnx.Optimizer,
             rngs: nnx.Rngs, train_iterator) -> None:
        """Persist model/optimizer/rngs/step/train_iterator at `step` (async; commits
        in the background — call `wait_until_finished` before relying on it on disk)."""
        self.mgr.save(step, args=ocp.args.Composite(
            model=ocp.args.StandardSave(nnx.state(model)),
            optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
            rngs=ocp.args.StandardSave(nnx.state(rngs)),
            step=ocp.args.JsonSave(step),
            train_iterator=grain.checkpoint.CheckpointSave(train_iterator),
        ))

    def restore(self, step: int | None = None, *, model: nnx.Module | None = None,
                optimizer: nnx.Optimizer | None = None, rngs: nnx.Rngs | None = None,
                train_iterator=None):
        """Restore in place the objects that are passed (each is optional — evaluation
        only needs `model`, resuming needs all of them). Defaults to the latest
        checkpoint. Returns `(step, train_iterator)`, where `train_iterator` is the
        restored Grain iterator (or None if one wasn't requested)."""
        step = self.mgr.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"No checkpoints found in {self.directory}")

        # Build the composite restore request from only the items we were handed. Each
        # live object's current state is the abstract target: it fixes the pytree
        # structure, dtypes and (device) sharding the restored arrays are placed into.
        items = {"step": ocp.args.JsonRestore()}
        if model is not None:
            items["model"] = ocp.args.StandardRestore(nnx.state(model))
        if optimizer is not None:
            items["optimizer"] = ocp.args.StandardRestore(nnx.state(optimizer))
        if rngs is not None:
            items["rngs"] = ocp.args.StandardRestore(nnx.state(rngs))
        if train_iterator is not None:
            items["train_iterator"] = grain.checkpoint.CheckpointRestore(train_iterator)

        restored = self.mgr.restore(step, args=ocp.args.Composite(**items))

        if model is not None:
            nnx.update(model, restored["model"])
        if optimizer is not None:
            nnx.update(optimizer, restored["optimizer"])
        if rngs is not None:
            nnx.update(rngs, restored["rngs"])
        return restored["step"], restored.get("train_iterator")

    def latest_step(self) -> int | None:
        return self.mgr.latest_step()

    def wait_until_finished(self) -> None:
        """Block until all pending async saves have committed (call before exit)."""
        self.mgr.wait_until_finished()

    def close(self) -> None:
        self.mgr.close()
