"""15 tests for src/data/bpe_tokenizer_v2.py

Uses only tiny corpora so the suite is fast.
"""

from __future__ import annotations

import os
import tempfile

from src.data.bpe_tokenizer_v2 import (
    BPEMergeRule,
    BPETokenizer,
    BPETrainer,
    BPEVocabulary,
    ByteVocabulary,
)

# ---------------------------------------------------------------------------
# Shared tiny corpus
# ---------------------------------------------------------------------------
TINY_CORPUS = [
    "aaabdaaabac",
    "aaaa",
    "aababc",
    "hello world hello world hello",
]

REPEAT_CORPUS = ["abababababababababababababab"] * 10


# ---------------------------------------------------------------------------
# 1. ByteVocabulary.to_bytes / from_bytes  — ASCII roundtrip is identity
# ---------------------------------------------------------------------------
def test_byte_vocabulary_ascii_roundtrip():
    bv = ByteVocabulary()
    text = "Hello, World!"
    assert bv.from_bytes(bv.to_bytes(text)) == text


# ---------------------------------------------------------------------------
# 2. ByteVocabulary.to_bytes — length equals UTF-8 byte count
# ---------------------------------------------------------------------------
def test_byte_vocabulary_length_equals_utf8_bytes():
    bv = ByteVocabulary()
    text = "café"  # 'é' is 2 UTF-8 bytes
    assert len(bv.to_bytes(text)) == len(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# 3. BPEMergeRule repr contains pair and merged_token
# ---------------------------------------------------------------------------
def test_merge_rule_repr():
    rule = BPEMergeRule(pair=(97, 98), merged_token=256, frequency=5)
    r = repr(rule)
    assert "(97, 98)" in r
    assert "256" in r


# ---------------------------------------------------------------------------
# 4. BPETrainer.train returns list of BPEMergeRule, length ≤ vocab_size - 256
# ---------------------------------------------------------------------------
def test_trainer_returns_merge_rules():
    trainer = BPETrainer(vocab_size=270)
    rules = trainer.train(TINY_CORPUS)
    assert isinstance(rules, list)
    assert all(isinstance(r, BPEMergeRule) for r in rules)
    assert len(rules) <= 270 - 256


# ---------------------------------------------------------------------------
# 5. BPETrainer._count_pairs — correct counts for a known sequence
# ---------------------------------------------------------------------------
def test_count_pairs_known_sequence():
    trainer = BPETrainer(vocab_size=300)
    seqs = [[1, 2, 1, 2, 3]]
    counts = trainer._count_pairs(seqs)
    assert counts[(1, 2)] == 2
    assert counts[(2, 1)] == 1
    assert counts[(2, 3)] == 1


# ---------------------------------------------------------------------------
# 6. BPETrainer._merge — correctly replaces pair with new token
# ---------------------------------------------------------------------------
def test_merge_replaces_pair():
    trainer = BPETrainer(vocab_size=300)
    seqs = [[1, 2, 1, 2, 3]]
    merged = trainer._merge(seqs, (1, 2), 99)
    assert merged == [[99, 99, 3]]


# ---------------------------------------------------------------------------
# 7. BPETokenizer.encode — returns list of ints, all valid token ids
# ---------------------------------------------------------------------------
def test_tokenizer_encode_returns_valid_ids():
    tok = BPETokenizer(vocab_size=280)
    tok.train(TINY_CORPUS)
    ids = tok.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    # All ids must be in the vocabulary
    assert tok._vocabulary is not None
    valid_ids = set(tok._vocabulary._id_to_bytes.keys())
    for i in ids:
        assert i in valid_ids


# ---------------------------------------------------------------------------
# 8. BPETokenizer.decode — encode then decode is identity for ASCII text
# ---------------------------------------------------------------------------
def test_tokenizer_encode_decode_roundtrip():
    tok = BPETokenizer(vocab_size=300)
    tok.train(TINY_CORPUS)
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text


# ---------------------------------------------------------------------------
# 9. Frequent bigrams get merged — common pairs have a merge rule
# ---------------------------------------------------------------------------
def test_frequent_bigrams_get_merged():
    # "ab" repeats heavily in REPEAT_CORPUS
    tok = BPETokenizer(vocab_size=270)
    tok.train(REPEAT_CORPUS)
    ab_pair = (ord("a"), ord("b"))
    merged_pairs = {rule.pair for rule in tok._merge_rules}
    assert ab_pair in merged_pairs, "Expected (a,b) to be merged from high-freq corpus"


# ---------------------------------------------------------------------------
# 10. compression_ratio ≥ 1.0 — BPE never expands
# ---------------------------------------------------------------------------
def test_compression_ratio_at_least_one():
    tok = BPETokenizer(vocab_size=300)
    tok.train(REPEAT_CORPUS)
    ratio = tok.compression_ratio("abababababababab")
    assert ratio >= 1.0


# ---------------------------------------------------------------------------
# 11. BPEVocabulary.vocab_size = 256 + number of merge rules
# ---------------------------------------------------------------------------
def test_bpe_vocabulary_vocab_size():
    trainer = BPETrainer(vocab_size=270)
    rules = trainer.train(TINY_CORPUS)
    vocab = BPEVocabulary(rules)
    assert vocab.vocab_size() == 256 + len(rules)


# ---------------------------------------------------------------------------
# 12. BPEVocabulary.decode_token — valid bytes for every token in vocab
# ---------------------------------------------------------------------------
def test_bpe_vocabulary_decode_token_valid():
    trainer = BPETrainer(vocab_size=270)
    rules = trainer.train(TINY_CORPUS)
    vocab = BPEVocabulary(rules)
    for token_id in vocab._id_to_bytes:
        b = vocab.decode_token(token_id)
        assert isinstance(b, bytes)
        assert len(b) >= 1


# ---------------------------------------------------------------------------
# 13. save_vocab + load_vocab — encode(text) same before and after save/load
# ---------------------------------------------------------------------------
def test_save_load_vocab_preserves_encoding():
    tok = BPETokenizer(vocab_size=290)
    tok.train(TINY_CORPUS)
    text = "aaabdaaabac"
    ids_before = tok.encode(text)

    tok2 = BPETokenizer(vocab_size=290)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        path = fh.name
    try:
        tok.save_vocab(path)
        tok2.load_vocab(path)
        ids_after = tok2.encode(text)
    finally:
        os.unlink(path)

    assert ids_before == ids_after


# ---------------------------------------------------------------------------
# 14. Empty string — encode returns [], decode([]) returns ""
# ---------------------------------------------------------------------------
def test_empty_string_encode_decode():
    tok = BPETokenizer(vocab_size=300)
    tok.train(TINY_CORPUS)
    assert tok.encode("") == []
    assert tok.decode([]) == ""


# ---------------------------------------------------------------------------
# 15. Larger corpus → more merges available (up to vocab_size limit)
# ---------------------------------------------------------------------------
def test_larger_corpus_enables_more_merges():
    small_corpus = ["ab"]
    large_corpus = ["abababababababab" * 20, "cdcdcdcdcdcdcdcd" * 20, "ababcdabcdab" * 10]
    vs = 270  # room for 14 merges

    tok_small = BPETokenizer(vocab_size=vs)
    tok_small.train(small_corpus)

    tok_large = BPETokenizer(vocab_size=vs)
    tok_large.train(large_corpus)

    assert len(tok_large._merge_rules) >= len(tok_small._merge_rules)
