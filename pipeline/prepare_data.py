"""
Stage 1 of the pipeline: turn CodeParrot into packed token memmaps.

WHAT THIS PRODUCES (in cfg.data.data_dir)
    train.bin   flat little-endian array of token ids (uint16 or uint32)
    val.bin     same, held-out
    meta.json   {dtype, vocab_size, eos_id, tokenizer, n_train_tokens, n_val_tokens}

WHY PACKED MEMMAP
    The GDN-2 / Grain training loop wants fixed-length windows sampled by random
    access. Rather than re-tokenize on the fly, we tokenize ONCE, concatenate every
    document (separated by the tokenizer's EOS id so the model sees boundaries) into
    one long stream, and dump it to disk. The loader then memory-maps the file and
    slices contiguous (seq_len+1) windows — O(1) per example, zero re-tokenization.

SOURCES
    source="huggingface": streams `codeparrot/codeparrot-train-v2-near-dedup` from the Hub. The
        corpus has only a train split, so we carve validation out of the head of the
        stream (first num_val_docs) and take training docs after it.
    source="synthetic": writes random token ids locally — no network, no tokenizer
        download — so the full prepare->train->eval path can run offline / in CI.

Run:  python -m pipeline.prepare_data --config configs/tiny.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.config import ExperimentConfig
from pipeline.tokenizer import build_tokenizer

# uint16 covers vocab <= 65536 (codeparrot 32768, byte 256); half the disk of uint32.
_FLUSH_TOKENS = 1_000_000  # write to disk in ~1M-token chunks to bound memory


def _dtype_for(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16 if vocab_size <= 2**16 else np.uint32)


def _write_stream(path: Path, token_iter, dtype: np.dtype) -> int:
    """Consume an iterator of 1-D token-id arrays, appending to `path`. Returns the
    total number of tokens written. Buffers up to ~_FLUSH_TOKENS before each write so
    we never hold the whole corpus in memory."""
    total = 0
    buf: list[np.ndarray] = []
    buffered = 0
    with open(path, "wb") as f:
        for chunk in token_iter:
            arr = np.asarray(chunk, dtype=dtype)
            buf.append(arr)
            buffered += arr.size
            if buffered >= _FLUSH_TOKENS:
                np.concatenate(buf).tofile(f)
                total += buffered
                buf, buffered = [], 0
        if buf:
            np.concatenate(buf).tofile(f)
            total += buffered
    return total


# --------------------------------------------------------------------------- #
#  Token producers.
# --------------------------------------------------------------------------- #
def _hf_doc_tokens(docs, tokenizer, text_field: str, dtype: np.dtype):
    """Yield one token-id array per document, with an EOS appended as a separator."""
    eos = tokenizer.eos_id
    for i, doc in enumerate(docs):
        text = doc[text_field]
        if not text:
            continue
        ids = tokenizer.encode(text)
        ids.append(eos)
        yield np.asarray(ids, dtype=dtype)
        if (i + 1) % 1000 == 0:
            print(f"  tokenized {i + 1} docs...", flush=True)


def _synthetic_tokens(
    n_tokens: int, vocab_size: int, eos_id: int, dtype: np.dtype, seed: int
):
    """Yield random token ids in ~200-token 'documents' ended by EOS. Purely for
    exercising the pipeline without a network or tokenizer download."""
    rng = np.random.default_rng(seed)
    produced = 0
    while produced < n_tokens:
        n = min(int(rng.integers(50, 250)), n_tokens - produced)
        ids = rng.integers(1, vocab_size, size=n, dtype=dtype)
        ids[-1] = eos_id
        produced += n
        yield ids


# --------------------------------------------------------------------------- #
def prepare(cfg: ExperimentConfig) -> None:
    data_dir = Path(cfg.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_bin, val_bin = data_dir / "train.bin", data_dir / "val.bin"

    if cfg.data.source == "synthetic":
        vocab_size, eos_id, tok_name = cfg.model.vocab_size, 0, "synthetic"
        dtype = _dtype_for(vocab_size)
        print(f"[synthetic] vocab={vocab_size} dtype={dtype}")
        n_val = _write_stream(
            val_bin,
            _synthetic_tokens(
                cfg.data.synthetic_val_tokens, vocab_size, eos_id, dtype, seed=1
            ),
            dtype,
        )
        n_train = _write_stream(
            train_bin,
            _synthetic_tokens(
                cfg.data.synthetic_train_tokens, vocab_size, eos_id, dtype, seed=2
            ),
            dtype,
        )
    elif cfg.data.source == "huggingface":
        from datasets import load_dataset

        tokenizer = build_tokenizer(cfg.data.tokenizer, cfg.data.tokenizer_name)
        vocab_size, eos_id, tok_name = (
            tokenizer.vocab_size,
            tokenizer.eos_id,
            cfg.data.tokenizer,
        )
        dtype = _dtype_for(vocab_size)
        print(
            f"[huggingface] {cfg.data.hf_dataset} tok={cfg.data.tokenizer} "
            f"vocab={vocab_size} dtype={dtype}"
        )

        stream = load_dataset(
            cfg.data.hf_dataset, split=cfg.data.hf_train_split, streaming=True
        )

        # Val = head of the stream; train = the docs after it (disjoint).
        val_docs = stream.take(cfg.data.num_val_docs) if cfg.data.num_val_docs else []
        print("Tokenizing validation split...")
        n_val = _write_stream(
            val_bin,
            _hf_doc_tokens(val_docs, tokenizer, cfg.data.text_field, dtype),
            dtype,
        )

        train_stream = stream.skip(cfg.data.num_val_docs or 0)
        if cfg.data.num_train_docs:
            train_stream = train_stream.take(cfg.data.num_train_docs)
        print("Tokenizing training split...")
        n_train = _write_stream(
            train_bin,
            _hf_doc_tokens(train_stream, tokenizer, cfg.data.text_field, dtype),
            dtype,
        )
    else:
        raise ValueError(f"Unknown data.source: {cfg.data.source!r}")

    if vocab_size != cfg.model.vocab_size:
        print(
            f"WARNING: tokenizer vocab {vocab_size} != model.vocab_size "
            f"{cfg.model.vocab_size}. Set model.vocab_size = {vocab_size} in the YAML."
        )

    meta = {
        "dtype": np.dtype(dtype).name,
        "vocab_size": int(vocab_size),
        "eos_id": int(eos_id),
        "tokenizer": tok_name,
        "tokenizer_name": cfg.data.tokenizer_name,
        "n_train_tokens": int(n_train),
        "n_val_tokens": int(n_val),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. train={n_train:,} tokens  val={n_val:,} tokens  -> {data_dir}")
    print(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tokenize CodeParrot into packed memmaps.")
    ap.add_argument("--config", required=True, help="Path to the experiment YAML.")
    args = ap.parse_args()
    prepare(ExperimentConfig.load(args.config))


if __name__ == "__main__":
    main()
