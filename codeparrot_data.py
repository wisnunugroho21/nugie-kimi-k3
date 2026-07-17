"""
CodeParrot data pipeline for the Kimi K3 recreation — prepare + Grain loading.

DATASET
-------
CodeParrot (codeparrot/codeparrot-clean on the Hugging Face Hub): ~50 GB of
deduplicated Python source files scraped from GitHub. We STREAM it — no full
download — tokenize documents with the official `codeparrot/codeparrot` BPE
tokenizer (GPT-2-style, vocab 32768, trained on this exact corpus), join them
with the EOS token, and PACK the token stream into flat binary files:

    <out_dir>/train.bin   uint16 tokens, exactly --train-tokens of them
    <out_dir>/val.bin     uint16 tokens, from DISJOINT documents (taken first)
    <out_dir>/meta.json   tokenizer name, vocab size, token counts

uint16 suffices because vocab 32768 < 2^16. Packing (concatenate-then-slice)
is the standard LM pretraining layout: no padding, every position supervised,
and a fixed seq_len just slices windows off the stream.

GRAIN LOADING
-------------
`PackedSource` wraps a .bin memmap as a grain.RandomAccessDataSource of
NON-OVERLAPPING seq_len windows: record i is tokens[i*L : i*L+L+1], split into
(inputs, labels) shifted by one. `train_dataset` builds the grain.MapDataset
chain  source -> shuffle(seed) -> repeat -> batch  and returns it WITHOUT
wrapping it in an iterator: MapDataset is randomly accessible, so the training
loop reads `ds[step]` — the batch at any global step is a pure function of
(seed, step), which makes checkpoint resume trivial (no iterator state to save;
grain reshuffles each epoch internally from the same seed). `val_dataset` is
the same source unshuffled and unrepeated for deterministic evaluation.

USAGE
-----
    python codeparrot_data.py --out data --train-tokens 2000000 --val-tokens 200000
"""

from __future__ import annotations

import argparse
import json
import pathlib

import grain
import numpy as np

TOKENIZER_NAME = "codeparrot/codeparrot"


# --------------------------------------------------------------------------- #
#  Preparation: stream CodeParrot -> tokenize -> pack -> .bin memmaps
# --------------------------------------------------------------------------- #
def prepare(out_dir: str, train_tokens: int, val_tokens: int) -> None:
    """Stream codeparrot/codeparrot-clean until the token budgets are filled.

    Validation documents are taken FIRST, so train/val never share a document
    (the stream is deterministic, making this split reproducible). Documents
    are joined by the tokenizer's EOS so the model learns file boundaries.
    """
    from datasets import load_dataset  # imported here: heavy, prepare-only
    from transformers import AutoTokenizer

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    assert tok.vocab_size <= np.iinfo(np.uint16).max + 1, "uint16 too small"
    eos = tok.eos_token_id

    stream = iter(
        load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    )

    def fill(path: pathlib.Path, budget: int) -> int:
        """Tokenize documents off the shared stream into `path` until `budget`
        tokens are written. Returns the number of source documents consumed."""
        buf = np.memmap(path, dtype=np.uint16, mode="w+", shape=(budget,))
        pos, docs = 0, 0
        while pos < budget:
            text = next(stream)["content"]
            ids = tok(text).input_ids + [eos]
            take = min(len(ids), budget - pos)
            buf[pos : pos + take] = np.asarray(ids[:take], np.uint16)
            pos += take
            docs += 1
            if docs % 200 == 0:
                print(f"  {path.name}: {pos:,}/{budget:,} tokens ({docs} docs)")
        buf.flush()
        return docs

    print(f"packing {val_tokens:,} validation tokens ...")
    val_docs = fill(out / "val.bin", val_tokens)
    print(f"packing {train_tokens:,} training tokens ...")
    train_docs = fill(out / "train.bin", train_tokens)

    meta = {
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": tok.vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "dataset": "codeparrot/codeparrot-clean",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done. meta: {meta}")


def load_meta(data_dir: str) -> dict:
    return json.loads((pathlib.Path(data_dir) / "meta.json").read_text())


# --------------------------------------------------------------------------- #
#  Grain data source + dataset builders
# --------------------------------------------------------------------------- #
class PackedSource(grain.sources.RandomAccessDataSource):
    """Non-overlapping seq_len windows over a packed .bin token file.

    Record i covers tokens [i*L, i*L + L + 1): L inputs and L labels shifted by
    one. The +1 overlap between inputs and labels is read from the same window,
    so records never share supervised positions. Reads go through np.memmap —
    the OS page cache does the buffering, nothing is loaded eagerly.
    """

    def __init__(self, bin_path: str | pathlib.Path, seq_len: int):
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self._n = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        L = self.seq_len
        window = np.asarray(self.tokens[i * L : i * L + L + 1], dtype=np.int32)
        return {"inputs": window[:-1], "labels": window[1:]}


def train_dataset(
    data_dir: str, seq_len: int, batch_size: int, seed: int
) -> grain.MapDataset:
    """Infinite, per-epoch-reshuffled, batched MapDataset over train.bin.

    Index it with the GLOBAL STEP (`ds[step]`): batch content is a pure
    function of (seed, step), so resuming from an Orbax checkpoint only needs
    the step counter — no data-iterator state.
    """
    src = PackedSource(pathlib.Path(data_dir) / "train.bin", seq_len)
    return (
        grain.MapDataset.source(src)
        .shuffle(seed=seed)  # reshuffled every epoch from this seed
        .repeat()  # infinite epochs
        .batch(batch_size, drop_remainder=True)
    )


def val_dataset(data_dir: str, seq_len: int, batch_size: int) -> grain.MapDataset:
    """Deterministic (unshuffled, single-epoch) batches over val.bin."""
    src = PackedSource(pathlib.Path(data_dir) / "val.bin", seq_len)
    return grain.MapDataset.source(src).batch(batch_size, drop_remainder=True)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare packed CodeParrot data")
    p.add_argument("--out", default="data", help="output directory")
    p.add_argument("--train-tokens", type=int, default=2_000_000)
    p.add_argument("--val-tokens", type=int, default=200_000)
    a = p.parse_args()
    prepare(a.out, a.train_tokens, a.val_tokens)
