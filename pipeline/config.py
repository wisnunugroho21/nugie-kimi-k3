"""
Experiment configuration for the Kimi-Linear-GDN2 code LM.

One YAML file drives the whole pipeline (prepare_data -> train -> evaluate). It is
split into three logical groups, all flat in the YAML for convenience:

  * `model:`  -> a KimiLinearConfig (the architecture, from kimi_linear_gdn2.py).
  * `data:`   -> where the tokenized CodeParrot memmaps live + how to build them.
  * `train:`  -> optimizer / schedule / checkpoint / logging knobs.

`ExperimentConfig.load(path)` reads the YAML, fills defaults, and returns a fully
typed object. `seq_len` is validated against the two hard model constraints so
misconfigurations fail loudly at startup instead of deep inside a jitted step.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from kimi_linear_gdn2 import KimiLinearConfig


# --------------------------------------------------------------------------- #
#  Data pipeline config.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class DataConfig:
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
    hf_val_split: str = "train"  # codeparrot-clean has no val split; we carve one out
    text_field: str = "content"

    # Cap how much we pull (the full corpus is >1TB). None => no cap.
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
    num_workers: int = 0  # Grain prefetch threads (0 = read inline)


# --------------------------------------------------------------------------- #
#  Training config.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 10000

    # --- Optimizer (AdamW) ---
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
    model: KimiLinearConfig = dataclasses.field(default_factory=KimiLinearConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)

    # ----------------------------------------------------------------------- #
    def validate(self) -> None:
        """Fail fast on the two hard model constraints for a training window."""
        C = self.model.gdn_chunk_size
        L = self.data.seq_len
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

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls(
            model=_build(KimiLinearConfig, raw.get("model", {})),
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
    """Instantiate a dataclass from a dict, ignoring unknown keys with a clear error."""
    fields = {f.name for f in dataclasses.fields(dc_type)}
    unknown = set(values) - fields
    if unknown:
        raise ValueError(f"Unknown {dc_type.__name__} keys in YAML: {sorted(unknown)}")
    return dc_type(**values)
