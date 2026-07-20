"""
Tokenizer abstraction for the code LM.

Two backends behind one tiny interface (`encode`, `decode`, `vocab_size`, `eos_id`):

  * ByteTokenizer   — raw UTF-8 bytes, vocab 256, no dependencies, no download.
                      Perfect for smoke tests and matches the model's byte-level
                      default (KimiK3Config.vocab_size = 256).
  * HFTokenizer     — the pretrained CodeParrot BPE tokenizer (vocab 32768) via
                      `transformers.AutoTokenizer`. This is what you want for real
                      code generation.

Both expose an EOS id used by prepare_data.py to separate documents in the packed
token stream, so the model learns document boundaries.
"""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    """Minimal tokenizer contract consumed by data preparation and generation."""

    vocab_size: int
    eos_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """UTF-8 byte tokenizer. eos_id reuses byte 0 (NUL) as a document separator —
    NUL effectively never occurs in source code, so it is a practical boundary
    marker. Decoding omits it, matching skip-special-token behavior."""

    vocab_size = 256
    eos_id = 0

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(b for b in ids if 0 <= b < 256 and b != self.eos_id).decode(
            "utf-8", errors="replace"
        )


class HFTokenizer:
    """Wraps a pretrained HuggingFace tokenizer (default: codeparrot/codeparrot)."""

    def __init__(self, name: str = "codeparrot/codeparrot"):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(name)
        # Some code tokenizers use BOS as their only document-boundary token. Never
        # co-opt an ordinary vocabulary item: that would corrupt training targets
        # and cause generation to stop on a normal token.
        eos = self._tok.eos_token_id
        if eos is None:
            eos = self._tok.bos_token_id
        # len(tok) counts added special tokens; .vocab_size excludes them. Ids up
        # to len(tok)-1 can appear in encoded text, so the model's embedding (and
        # meta.json's vocab check) must be sized by len(tok).
        self.vocab_size = len(self._tok)
        if eos is None:
            raise ValueError(f"Tokenizer {name!r} defines neither eos_token_id nor bos_token_id")
        self.eos_id = int(eos)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


def build_tokenizer(kind: str, name: str = "codeparrot/codeparrot") -> Tokenizer:
    """Construct the configured byte or Hugging Face tokenizer backend."""
    if kind in ("byte", "synthetic"):
        # Synthetic data has no real tokenizer; decode its ids as raw bytes so the
        # generation path still runs (output is gibberish, as expected offline).
        return ByteTokenizer()
    if kind in ("codeparrot", "hf", "huggingface"):
        return HFTokenizer(name)
    raise ValueError(f"Unknown tokenizer kind: {kind!r} (expected 'byte' or 'codeparrot')")
