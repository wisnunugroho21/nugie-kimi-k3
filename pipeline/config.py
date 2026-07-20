"""
Experiment configuration for the Kimi-Linear-GDN2 code LM.

One YAML file drives the whole pipeline (prepare_data -> train -> evaluate). It is
split into three logical groups, all flat in the YAML for convenience:

  * `model:`  -> a KimiK3Config (the architecture, from kimi_k3_gdn2.py).
  * `data:`   -> where the tokenized CodeParrot memmaps live + how to build them.
  * `train:`  -> optimizer / schedule / checkpoint / logging knobs.

`ExperimentConfig.load(path)` reads the YAML, fills defaults, and returns a fully
typed object. Cross-field and range constraints are validated up front so
misconfigurations fail loudly instead of deep inside a jitted step.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from kimi_k3_gdn2 import KimiK3Config


# --------------------------------------------------------------------------- #
#  Data pipeline config.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class DataConfig:
    """Prepared-data, tokenizer, corpus-source, and Grain loader settings."""

    # Where prepare_data.py writes {train,val}.bin + meta.json, and where the
    # loader reads them back from.
    data_dir: str = "data/codeparrot"

    # --- Source (used only by prepare_data.py) ---
    #   "huggingface": stream the real CodeParrot corpus from the HF Hub.
    #   "synthetic":   generate random tokens locally (no network) so the whole
    #                  pipeline can be exercised end-to-end offline / in CI.
    source: str = "huggingface"
    hf_dataset: str = "codeparrot/codeparrot-train-v2-near-dedup"
    hf_train_split: str = "train"
    # When this equals hf_train_split, validation is carved from the head of that
    # stream. A different value loads a native validation split independently.
    hf_val_split: str = "train"
    text_field: str = "content"

    # Cap how much we pull (the full corpus is >1TB). None means no cap, except
    # num_val_docs must be finite when validation is carved from the train split.
    num_train_docs: int | None = 20000
    num_val_docs: int | None = 500
    synthetic_train_tokens: int = 1_000_000
    synthetic_val_tokens: int = 50_000

    # --- Tokenizer ---
    #   "codeparrot": the pretrained BPE tokenizer (vocab 32768), matches the model.
    #   "byte":       raw UTF-8 bytes (vocab 256), zero-download, good for smoke tests.
    tokenizer: str = "codeparrot"
    tokenizer_name: str = "codeparrot/codeparrot"

    # --- Loader ---
    seq_len: int = 256  # tokens per training window (must divide evenly, see below)
    shuffle_buffer_seed: int = 0
    num_workers: int = 0  # requested Grain reader threads (0 uses the minimum of 1)


# --------------------------------------------------------------------------- #
#  Training config.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TrainConfig:
    """Optimization, checkpointing, logging, and validation-loop settings."""

    batch_size: int = 16
    max_steps: int = 10000

    # --- Optimizer (Muon with an AdamW implementation for fallback leaves) ---
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Aux-loss-free load-balancing: step size for the per-expert router-bias nudge
    # (DeepSeek-V3 style; applied outside the gradient each step).
    router_bias_lr: float = 1e-3

    # --- Checkpointing (Orbax) ---
    ckpt_dir: str = "checkpoints/codeparrot"
    save_every: int = 1000
    keep_checkpoints: int = 3

    # --- Logging / eval-during-train ---
    log_every: int = 20
    eval_every: int = 1000
    eval_steps: int = 50  # number of val batches per in-training eval
    seed: int = 0


@dataclasses.dataclass
class ExperimentConfig:
    """Complete model/data/training configuration shared by every pipeline stage."""

    model: KimiK3Config = dataclasses.field(default_factory=KimiK3Config)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)

    # ----------------------------------------------------------------------- #
    def validate(self) -> None:
        """Fail fast on invalid pipeline ranges and cross-field constraints."""
        C = self.model.gdn_chunk_size
        L = self.data.seq_len
        if L <= 0:
            raise ValueError("data.seq_len must be positive")
        if L % C != 0:
            raise ValueError(
                f"data.seq_len ({L}) must be a multiple of the GDN-2 chunk_size "
                f"({C}); the chunkwise core reshapes L into L/C chunks."
            )
        if L > self.model.max_seq_len:
            raise ValueError(
                f"data.seq_len ({L}) exceeds model.max_seq_len "
                f"({self.model.max_seq_len})."
            )
        if self.data.source not in {"huggingface", "synthetic"}:
            raise ValueError("data.source must be 'huggingface' or 'synthetic'")
        valid_tokenizers = {"byte", "synthetic", "codeparrot", "hf", "huggingface"}
        if self.data.tokenizer not in valid_tokenizers:
            raise ValueError(f"Unknown data.tokenizer: {self.data.tokenizer!r}")
        if self.data.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")
        for name in ("num_train_docs", "num_val_docs"):
            value = getattr(self.data, name)
            if value is not None and value <= 0:
                raise ValueError(f"data.{name} must be positive or None")
        if (
            self.data.source == "huggingface"
            and self.data.hf_val_split == self.data.hf_train_split
            and self.data.num_val_docs is None
        ):
            raise ValueError(
                "data.num_val_docs must be finite when validation is carved from "
                "the training split"
            )
        if self.data.source == "synthetic":
            if self.data.synthetic_train_tokens <= L:
                raise ValueError("data.synthetic_train_tokens must exceed data.seq_len")
            if self.data.synthetic_val_tokens <= L:
                raise ValueError("data.synthetic_val_tokens must exceed data.seq_len")

        tc = self.train
        positive = {
            "batch_size": tc.batch_size,
            "max_steps": tc.max_steps,
            "lr": tc.lr,
            "eps": tc.eps,
            "grad_clip": tc.grad_clip,
            "save_every": tc.save_every,
            "keep_checkpoints": tc.keep_checkpoints,
            "log_every": tc.log_every,
            "eval_every": tc.eval_every,
            "eval_steps": tc.eval_steps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Training values must be positive: {', '.join(invalid)}")
        if not 0 <= tc.min_lr <= tc.lr:
            raise ValueError("train.min_lr must be between 0 and train.lr")
        if not 0 <= tc.warmup_steps <= tc.max_steps:
            raise ValueError("train.warmup_steps must be between 0 and train.max_steps")
        if tc.weight_decay < 0 or tc.router_bias_lr < 0:
            raise ValueError("train.weight_decay and router_bias_lr must be non-negative")
        if not 0 <= tc.beta1 < 1 or not 0 <= tc.beta2 < 1:
            raise ValueError("train.beta1 and beta2 must be in [0, 1)")

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("The experiment YAML must contain a top-level mapping")
        unknown_groups = set(raw) - {"model", "data", "train"}
        if unknown_groups:
            raise ValueError(f"Unknown top-level YAML keys: {sorted(unknown_groups)}")
        cfg = cls(
            model=_build(KimiK3Config, raw.get("model", {})),
            data=_build(DataConfig, raw.get("data", {})),
            train=_build(TrainConfig, raw.get("train", {})),
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": dataclasses.asdict(self.model),
            "data": dataclasses.asdict(self.data),
            "train": dataclasses.asdict(self.train),
        }


def _build(dc_type: type, values: dict[str, Any]):
    """Instantiate a dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(values, dict):
        raise ValueError(f"{dc_type.__name__} configuration must be a mapping")
    fields = {f.name for f in dataclasses.fields(dc_type)}
    unknown = set(values) - fields
    if unknown:
        raise ValueError(f"Unknown {dc_type.__name__} keys in YAML: {sorted(unknown)}")
    return dc_type(**values)
