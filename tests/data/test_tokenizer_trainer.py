"""Unit tests for BPE tokenizer trainer (pure stdlib)."""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from src.data.tokenizer_trainer import (
    BPEConfig,
    BPETokenizer,
    BPETrainer,
)


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick",
    "hello world hello there",
    "banana banana banana apple",
    "pack my box with five dozen liquor jugs",
]


def _make_tok(cfg: BPEConfig, texts=None):
    trainer = BPETrainer(cfg)
    tok = trainer.train(texts if texts is not None else CORPUS)
    return trainer, tok


def test_train_vocab_size_capped() -> None:
    cfg = BPEConfig(vocab_size=300, special_tokens=["<pad>", "<eos>"])
    _, tok = _make_tok(cfg)
    assert len(tok["vocab"]) <= 300
    # must include 256 base bytes + 2 specials minimum
    assert len(tok["vocab"]) >= 256 + 2


def test_special_tokens_in_vocab() -> None:
    cfg = BPEConfig(vocab_size=300, special_tokens=["<pad>", "<eos>", "<bos>"])
    _, tok = _make_tok(cfg)
    for s in ["<pad>", "<eos>", "<bos>"]:
        assert s in tok["vocab"]


def test_encode_decode_roundtrip_byte_level() -> None:
    cfg = BPEConfig(vocab_size=400, special_tokens=["<pad>"])
    _, tok = _make_tok(cfg)
    bt = BPETokenizer(
        tok["vocab"], tok["merges"],
        byte_level=True,
        special_tokens=tok["special_tokens"],
        pretokenize_regex=tok["pretokenize_regex"],
    )
    text = "the quick brown fox"
    ids = bt.encode(text)
    out = bt.decode(ids)
    # pretokenize strips whitespace; tokens should be preserved
    assert "quick" in out and "brown" in out and "fox" in out


def test_oov_bytes_byte_level_fallback() -> None:
    cfg = BPEConfig(vocab_size=260, special_tokens=[])
    _, tok = _make_tok(cfg)
    bt = BPETokenizer(
        tok["vocab"], tok["merges"], byte_level=True,
        pretokenize_regex=tok["pretokenize_regex"],
    )
    # emoji / rare unicode not seen in corpus -- must still encode
    ids = bt.encode("zzz_\u2603_zzz")  # snowman
    assert len(ids) > 0
    out = bt.decode(ids)
    assert "\u2603" in out


def test_merges_nonempty() -> None:
    cfg = BPEConfig(vocab_size=320, special_tokens=[])
    _, tok = _make_tok(cfg)
    assert len(tok["merges"]) > 0
    assert all(isinstance(p, tuple) and len(p) == 2 for p in tok["merges"])


def test_save_load_roundtrip(tmp_path) -> None:
    cfg = BPEConfig(vocab_size=320, special_tokens=["<pad>"])
    trainer, tok = _make_tok(cfg)
    p = tmp_path / "tok.json"
    trainer.save(tok, str(p))
    loaded = trainer.load(str(p))
    assert loaded["vocab"] == tok["vocab"]
    assert loaded["merges"] == tok["merges"]
    assert loaded["special_tokens"] == tok["special_tokens"]


def test_determinism_same_corpus() -> None:
    cfg = BPEConfig(vocab_size=320, special_tokens=["<pad>"])
    _, t1 = _make_tok(cfg)
    _, t2 = _make_tok(cfg)
    assert t1["merges"] == t2["merges"]
    assert t1["vocab"] == t2["vocab"]


def test_empty_corpus_raises() -> None:
    cfg = BPEConfig(vocab_size=300)
    trainer = BPETrainer(cfg)
    with pytest.raises(ValueError):
        trainer.train([])
    with pytest.raises(ValueError):
        trainer.train(["", ""])


def test_vocab_size_too_small_raises() -> None:
    cfg = BPEConfig(vocab_size=100, special_tokens=["<pad>"], byte_level=True)
    trainer = BPETrainer(cfg)
    with pytest.raises(ValueError):
        trainer.train(CORPUS)


def test_pretokenize_regex_respected() -> None:
    # regex that only matches alphabetic runs -- punctuation/digits dropped
    cfg = BPEConfig(
        vocab_size=300, special_tokens=[],
        pretokenize_regex=r"[a-z]+",
    )
    trainer, tok = _make_tok(cfg, texts=["hello123world foo!bar"])
    bt = BPETokenizer(
        tok["vocab"], tok["merges"], byte_level=True,
        pretokenize_regex=tok["pretokenize_regex"],
    )
    ids = bt.encode("hello123world foo!bar")
    out = bt.decode(ids)
    # digits and punctuation should have been dropped by pretokenization
    assert "123" not in out
    assert "!" not in out
    assert "hello" in out


def test_large_corpus_trains_fast() -> None:
    # generate ~10k whitespace tokens
    words = ["foo", "bar", "baz", "qux", "quux", "corge", "grault"]
    text = " ".join(words * 1500)  # 10500 tokens
    cfg = BPEConfig(vocab_size=400, special_tokens=[])
    trainer = BPETrainer(cfg)
    t0 = time.time()
    tok = trainer.train([text])
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"training took {elapsed:.2f}s"
    assert len(tok["vocab"]) <= 400


def test_special_token_single_id() -> None:
    cfg = BPEConfig(vocab_size=320, special_tokens=["<pad>", "<eos>"])
    _, tok = _make_tok(cfg)
    bt = BPETokenizer(
        tok["vocab"], tok["merges"], byte_level=True,
        special_tokens=tok["special_tokens"],
        pretokenize_regex=tok["pretokenize_regex"],
    )
    ids = bt.encode("<pad>")
    assert ids == [tok["vocab"]["<pad>"]]
    ids2 = bt.encode("hello <eos>")
    assert tok["vocab"]["<eos>"] in ids2


def test_special_pattern_not_split_as_bytes() -> None:
    # special appears inline -- must be a single id, not broken into bytes
    cfg = BPEConfig(vocab_size=320, special_tokens=["<EOS>"])
    _, tok = _make_tok(cfg)
    bt = BPETokenizer(
        tok["vocab"], tok["merges"], byte_level=True,
        special_tokens=tok["special_tokens"],
        pretokenize_regex=tok["pretokenize_regex"],
    )
    ids = bt.encode("foo<EOS>bar")
    assert tok["vocab"]["<EOS>"] in ids
    # exactly one occurrence of the special-id
    assert ids.count(tok["vocab"]["<EOS>"]) == 1


def test_decode_out_of_vocab_raises() -> None:
    cfg = BPEConfig(vocab_size=320, special_tokens=[])
    _, tok = _make_tok(cfg)
    bt = BPETokenizer(
        tok["vocab"], tok["merges"], byte_level=True,
        pretokenize_regex=tok["pretokenize_regex"],
    )
    bad_id = max(tok["vocab"].values()) + 9999
    with pytest.raises(KeyError):
        bt.decode([bad_id])
