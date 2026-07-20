"""
Stage 2: the Grain input pipeline.

Reads the packed memmaps written by prepare_data.py and serves fixed-length,
next-token training windows.

RANDOM-ACCESS SOURCE
    `PackedTokenSource` memory-maps train.bin/val.bin and, for index i, returns the
    contiguous window tokens[i*seq_len : i*seq_len + seq_len + 1] split into
        input_ids  = window[:-1]   (length seq_len)
        target_ids = window[1:]    (length seq_len, shifted by one)
    Non-overlapping stride = seq_len, so every token is a target exactly once per
    epoch. Because it is true random access (__len__ + __getitem__), Grain can
    shuffle globally and prefetch with reader threads.

GRAIN PIPELINE
    MapDataset.source(src).shuffle(seed).repeat().batch(B).to_iter_dataset(...)
    Batching stacks the per-example dicts into {input_ids:[B,L], target_ids:[B,L]}
    int32 arrays ready for the model. `repeat()` makes training draw an unbounded
    stream so the loop is driven purely by train.max_steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import grain
import numpy as np


def load_meta(data_dir: str | Path) -> dict:
    """Load the metadata written beside the packed train/validation binaries."""
    return json.loads((Path(data_dir) / "meta.json").read_text())


class PackedTokenSource:
    """Grain RandomAccessDataSource over one packed .bin memmap.

    Implements the __len__/__getitem__ protocol Grain expects. The memmap is opened
    lazily on first access, avoiding an unnecessary file mapping in the source's
    constructor and keeping the source straightforward to serialize."""

    def __init__(self, bin_path: str | Path, seq_len: int, dtype: str):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.bin_path = str(bin_path)
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "u":
            raise ValueError(f"Packed token dtype must be unsigned integer, got {self.dtype}")
        n_bytes = Path(self.bin_path).stat().st_size
        if n_bytes % self.dtype.itemsize:
            raise ValueError(
                f"{self.bin_path} size ({n_bytes} bytes) is not aligned to dtype {self.dtype}"
            )
        n_tokens = n_bytes // self.dtype.itemsize
        # Number of non-overlapping (seq_len+1)-token windows we can slice. Need one
        # extra token for the shifted target, hence (n_tokens - 1).
        self._len = max(0, (n_tokens - 1) // seq_len)
        if self._len == 0:
            raise ValueError(
                f"{self.bin_path} has {n_tokens} tokens, too few for one window of "
                f"seq_len={seq_len}. Prepare more data or lower data.seq_len."
            )
        self._mmap: np.memmap | None = None

    def _data(self) -> np.memmap:
        if self._mmap is None:  # reopen inside the worker that first touches it
            self._mmap = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        return self._mmap

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < self._len:
            raise IndexError(index)
        start = index * self.seq_len
        window = np.asarray(self._data()[start : start + self.seq_len + 1], dtype=np.int32)
        return {"input_ids": window[:-1], "target_ids": window[1:]}


def make_loader(
    data_dir: str | Path,
    split: str,
    seq_len: int,
    batch_size: int,
    *,
    shuffle: bool,
    repeat: bool,
    seed: int = 0,
    num_workers: int = 0,
) -> grain.DatasetIterator:
    """Build a Grain iterator yielding {input_ids, target_ids} int32 batches.

    split: "train" or "val". shuffle/repeat are typically True for train, False for
    val. num_workers controls Grain's reader-thread count; zero uses one thread."""
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    meta = load_meta(data_dir)
    bin_path = Path(data_dir) / f"{split}.bin"
    source = PackedTokenSource(bin_path, seq_len, meta["dtype"])
    # Repeating happens before batching, so a tiny repeating source can still
    # produce full batches. A finite source cannot: drop_remainder would yield none.
    if not repeat and len(source) < batch_size:
        raise ValueError(
            f"{split} split has {len(source)} windows, fewer than batch_size={batch_size}"
        )

    ds = grain.MapDataset.source(source)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if repeat:
        ds = ds.repeat()  # unbounded; the training loop bounds itself by max_steps
    ds = ds.batch(batch_size, drop_remainder=True)

    read_options = grain.ReadOptions(
        num_threads=max(1, num_workers), prefetch_buffer_size=4 * max(1, num_workers)
    )
    return ds.to_iter_dataset(read_options=read_options).__iter__()


def num_windows(data_dir: str | Path, split: str, seq_len: int) -> int:
    """How many non-overlapping windows the split holds (for sizing a full-val pass)."""
    meta = load_meta(data_dir)
    return len(PackedTokenSource(Path(data_dir) / f"{split}.bin", seq_len, meta["dtype"]))
