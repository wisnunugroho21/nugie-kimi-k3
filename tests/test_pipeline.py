"""Small regression tests for configuration, tokenization, and packed data I/O."""

import json

import numpy as np
import pytest

from pipeline.config import ExperimentConfig
from pipeline.data import PackedTokenSource
from pipeline.prepare_data import _dtype_for
from pipeline.tokenizer import ByteTokenizer


def test_byte_tokenizer_hides_document_separator():
    tok = ByteTokenizer()
    assert tok.decode([ord("a"), tok.eos_id, ord("b")]) == "ab"


def test_packed_source_bounds_and_little_endian_dtype(tmp_path):
    dtype = _dtype_for(256)
    assert dtype.str == "<u2"
    path = tmp_path / "train.bin"
    np.arange(17, dtype=dtype).tofile(path)
    source = PackedTokenSource(path, seq_len=8, dtype=dtype.str)
    assert len(source) == 2
    np.testing.assert_array_equal(source[0]["target_ids"], np.arange(1, 9))
    with pytest.raises(IndexError):
        _ = source[-1]


def test_config_rejects_unknown_top_level_key(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model: {}\nunknown: true\n")
    with pytest.raises(ValueError, match="top-level"):
        ExperimentConfig.load(path)


def test_config_dict_is_json_serializable():
    # Guards the checkpoint/run-metadata contract after adding config validation.
    json.dumps(ExperimentConfig().to_dict())
