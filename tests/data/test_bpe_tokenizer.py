"""Tests for the pure-Python BPE tokenizer (src/data/bpe_tokenizer.py)."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.data.bpe_tokenizer import BPEConfig, BPETokenizer, get_byte_pairs, merge_vocab

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SHORT_TEXTS = ["aababc" * 10, "bcbcbc" * 10]


@pytest.fixture
def trained_tok() -> BPETokenizer:
    """A BPETokenizer trained on SHORT_TEXTS with a modest vocab target."""
    cfg = BPEConfig(vocab_size=300, min_frequency=2)
    tok = BPETokenizer(cfg)
    tok.train(SHORT_TEXTS)
    return tok


# ---------------------------------------------------------------------------
# 1. BPEConfig defaults
# ---------------------------------------------------------------------------


def test_bpe_config_defaults():
    cfg = BPEConfig()
    assert cfg.vocab_size == 1000
    assert cfg.min_frequency == 2
    assert cfg.special_tokens == ["<pad>", "<unk>", "<s>", "</s>"]


# ---------------------------------------------------------------------------
# 2. get_byte_pairs returns dict with correct counts
# ---------------------------------------------------------------------------


def test_get_byte_pairs_correct_counts():
    word_tokens = [("a", "b", "c"), ("a", "b", "a")]
    pairs = get_byte_pairs(word_tokens)
    assert isinstance(pairs, dict)
    # ("a", "b") appears twice (once per word)
    assert pairs[("a", "b")] == 2
    # ("b", "c") appears once
    assert pairs[("b", "c")] == 1
    # ("b", "a") appears once
    assert pairs[("b", "a")] == 1


# ---------------------------------------------------------------------------
# 3. get_byte_pairs empty input -> empty dict
# ---------------------------------------------------------------------------


def test_get_byte_pairs_empty():
    assert get_byte_pairs([]) == {}


# ---------------------------------------------------------------------------
# 4. merge_vocab merges target pair
# ---------------------------------------------------------------------------


def test_merge_vocab_merges_pair():
    vocab = {("a", "b", "c"): 3, ("a", "b", "d"): 2}
    new_vocab = merge_vocab(vocab, ("a", "b"))
    # Both words should now start with merged "ab"
    for symbols in new_vocab:
        assert "ab" in symbols
        assert "a" not in symbols or symbols.index("a") != 0 or len(symbols) < 2


def test_merge_vocab_merged_symbol_present():
    vocab = {("x", "y", "z"): 5}
    new_vocab = merge_vocab(vocab, ("x", "y"))
    assert ("xy", "z") in new_vocab
    assert new_vocab[("xy", "z")] == 5


# ---------------------------------------------------------------------------
# 5. merge_vocab leaves other pairs unchanged
# ---------------------------------------------------------------------------


def test_merge_vocab_leaves_others_unchanged():
    vocab = {("a", "b"): 3, ("c", "d"): 7}
    new_vocab = merge_vocab(vocab, ("a", "b"))
    # ("c", "d") should be untouched
    assert ("c", "d") in new_vocab
    assert new_vocab[("c", "d")] == 7


# ---------------------------------------------------------------------------
# 6. BPETokenizer.train increases vocab beyond 256
# ---------------------------------------------------------------------------


def test_train_increases_vocab(trained_tok: BPETokenizer):
    assert trained_tok.vocab_size > 256


# ---------------------------------------------------------------------------
# 7. BPETokenizer.encode returns list of ints
# ---------------------------------------------------------------------------


def test_encode_returns_list_of_ints(trained_tok: BPETokenizer):
    ids = trained_tok.encode("abc")
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


# ---------------------------------------------------------------------------
# 8. encode then decode roundtrips ASCII text
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip(trained_tok: BPETokenizer):
    text = "aababc"
    decoded = trained_tok.decode(trained_tok.encode(text))
    assert decoded == text


# ---------------------------------------------------------------------------
# 9. BPETokenizer.vocab_size matches trained size
# ---------------------------------------------------------------------------


def test_vocab_size_property(trained_tok: BPETokenizer):
    vs = trained_tok.vocab_size
    assert isinstance(vs, int)
    assert vs > 256
    # Must be consistent with internal vocab dict
    assert vs == len(trained_tok._token_to_id)


# ---------------------------------------------------------------------------
# 10. encode output IDs all within [0, vocab_size)
# ---------------------------------------------------------------------------


def test_encode_ids_within_range(trained_tok: BPETokenizer):
    ids = trained_tok.encode("bcbcbc")
    vs = trained_tok.vocab_size
    assert all(0 <= i < vs for i in ids)


# ---------------------------------------------------------------------------
# 11. train with small vocab_size respects limit
# ---------------------------------------------------------------------------


def test_small_vocab_size_respected():
    cfg = BPEConfig(vocab_size=270, min_frequency=1)
    tok = BPETokenizer(cfg)
    tok.train(["ab" * 5])
    # vocab should not exceed the configured vocab_size by more than the
    # number of special tokens (they are added before merge loop)
    assert tok.vocab_size <= cfg.vocab_size + len(cfg.special_tokens)


# ---------------------------------------------------------------------------
# 12. special tokens in vocab
# ---------------------------------------------------------------------------


def test_special_tokens_in_vocab(trained_tok: BPETokenizer):
    for sp in trained_tok.config.special_tokens:
        assert sp in trained_tok._token_to_id


# ---------------------------------------------------------------------------
# 13. save then load preserves encode behavior
# ---------------------------------------------------------------------------


def test_save_load_preserves_encode(trained_tok: BPETokenizer):
    text = "aababc"
    original_ids = trained_tok.encode(text)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        trained_tok.save(tmp_path)
        loaded = BPETokenizer.load(tmp_path)
        loaded_ids = loaded.encode(text)
        assert loaded_ids == original_ids
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 14. BPETokenizer on repeated text learns useful merges (common pair merged)
# ---------------------------------------------------------------------------


def test_repeated_text_learns_merges():
    """The pair ('a','b') must appear as a merge when 'ab' is very frequent."""
    cfg = BPEConfig(vocab_size=300, min_frequency=2)
    tok = BPETokenizer(cfg)
    tok.train(["ab" * 50])
    # ('a', 'b') should be among the first merges learned
    assert len(tok._merges) > 0
    merged_pair = tok._merges[0]
    # The first merge should involve 'a' and 'b' (most frequent pair)
    assert merged_pair == (chr(ord("a")), chr(ord("b")))
