"""Tests for byte-level BPE tokenizer."""

import pytest

from src.data.byte_tokenizer import (
    BPETokenizer,
    TokenizerConfig,
    count_pairs,
    get_byte_vocab,
    merge_pair,
    train_tokenizer,
)

TRAINING_CORPUS = ["hello world " * 20, "foo bar baz " * 20]


@pytest.fixture
def trained_tokenizer() -> BPETokenizer:
    config = TokenizerConfig(vocab_size=300, min_frequency=2, max_merges=100)
    tok = BPETokenizer(config)
    tok.train(TRAINING_CORPUS)
    return tok


# 1. TokenizerConfig defaults
def test_config_defaults():
    cfg = TokenizerConfig()
    assert cfg.vocab_size == 256
    assert cfg.min_frequency == 2
    assert cfg.max_merges == 1000
    assert cfg.special_tokens == ["<pad>", "<bos>", "<eos>", "<unk>"]


# 2. get_byte_vocab returns 256 entries
def test_get_byte_vocab_size():
    vocab = get_byte_vocab()
    assert len(vocab) == 256
    assert vocab[0] == b"\x00"
    assert vocab[255] == b"\xff"
    assert vocab[65] == b"A"


# 3. count_pairs counts correctly on simple input
def test_count_pairs():
    seqs = [[1, 2, 3, 1, 2]]
    counts = count_pairs(seqs)
    assert counts[(1, 2)] == 2
    assert counts[(2, 3)] == 1
    assert counts[(3, 1)] == 1


# 4. merge_pair replaces correct pairs
def test_merge_pair():
    seqs = [[1, 2, 3, 1, 2, 4]]
    result = merge_pair(seqs, (1, 2), 99)
    assert result == [[99, 3, 99, 4]]


# 5. BPETokenizer.train learns merges (vocab grows)
def test_train_learns_merges(trained_tokenizer: BPETokenizer):
    assert len(trained_tokenizer.merges) > 0
    assert trained_tokenizer.vocab_size > 256


# 6. BPETokenizer.encode returns list of ints
def test_encode_returns_ints(trained_tokenizer: BPETokenizer):
    ids = trained_tokenizer.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


# 7. Round-trip for ASCII text
def test_round_trip_ascii(trained_tokenizer: BPETokenizer):
    text = "hello world"
    assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text


# 8. Encode compresses repeated patterns
def test_encode_compresses(trained_tokenizer: BPETokenizer):
    text = "hello world " * 5
    ids = trained_tokenizer.encode(text)
    byte_len = len(text.encode("utf-8"))
    assert len(ids) < byte_len


# 9. Handles empty string
def test_empty_string(trained_tokenizer: BPETokenizer):
    assert trained_tokenizer.encode("") == []
    assert trained_tokenizer.decode([]) == ""


# 10. encode_batch returns correct number of sequences
def test_encode_batch(trained_tokenizer: BPETokenizer):
    texts = ["hello", "world", "foo bar"]
    batch = trained_tokenizer.encode_batch(texts)
    assert len(batch) == 3
    assert all(isinstance(seq, list) for seq in batch)


# 11. Special tokens are assigned valid ids
def test_special_tokens_assigned(trained_tokenizer: BPETokenizer):
    assert len(trained_tokenizer.special_tokens) == 4
    for name, tid in trained_tokenizer.special_tokens.items():
        assert isinstance(tid, int)
        assert tid >= 256


# 12. get_special_token_id works / raises
def test_get_special_token_id(trained_tokenizer: BPETokenizer):
    pad_id = trained_tokenizer.get_special_token_id("<pad>")
    assert isinstance(pad_id, int)
    with pytest.raises(KeyError):
        trained_tokenizer.get_special_token_id("<nonexistent>")


# 13. train_tokenizer convenience function
def test_train_tokenizer_convenience():
    cfg = TokenizerConfig(vocab_size=280, min_frequency=2, max_merges=50)
    tok = train_tokenizer(TRAINING_CORPUS, cfg)
    assert isinstance(tok, BPETokenizer)
    assert len(tok.merges) > 0


# 14. Handles non-ASCII (UTF-8 multi-byte) text
def test_non_ascii_text(trained_tokenizer: BPETokenizer):
    text = "café résumé 日本語"
    ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(ids)
    assert decoded == text
