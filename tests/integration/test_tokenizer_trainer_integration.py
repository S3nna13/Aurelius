"""Integration tests: BPE tokenizer trainer is exposed via src.data."""

from __future__ import annotations

import src.data as data_pkg
from src.data import (
    BPEConfig,
    BPETokenizer,
    BPETrainer,
)


def test_surface_exposed() -> None:
    assert hasattr(data_pkg, "BPEConfig")
    assert hasattr(data_pkg, "BPETrainer")
    assert hasattr(data_pkg, "BPETokenizer")


def test_prior_entries_intact() -> None:
    # prior exports from dataset_config and fim_transform remain importable
    from src.data import (
        AURELIUS_MIX,
        FIM_MIDDLE,
        FIM_PREFIX,
        FIM_SUFFIX,
        FIMConfig,
        fim_transform,
    )

    assert AURELIUS_MIX is not None
    assert "fim_prefix" in FIM_PREFIX
    assert "fim_suffix" in FIM_SUFFIX
    assert "fim_middle" in FIM_MIDDLE
    assert FIMConfig is not None
    assert callable(fim_transform)


def test_train_and_encode_decode_small_corpus() -> None:
    cfg = BPEConfig(vocab_size=320, special_tokens=["<pad>", "<eos>"])
    trainer = BPETrainer(cfg)
    tok = trainer.train(
        [
            "the quick brown fox",
            "the slow green turtle",
            "hello world hello",
        ]
    )
    assert "<pad>" in tok["vocab"]
    assert len(tok["merges"]) > 0

    bt = BPETokenizer(
        tok["vocab"],
        tok["merges"],
        byte_level=True,
        special_tokens=tok["special_tokens"],
        pretokenize_regex=tok["pretokenize_regex"],
    )
    ids = bt.encode("the quick brown fox")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    out = bt.decode(ids)
    assert "quick" in out and "fox" in out


def test_save_load_via_trainer(tmp_path) -> None:
    cfg = BPEConfig(vocab_size=300, special_tokens=["<pad>"])
    trainer = BPETrainer(cfg)
    tok = trainer.train(["alpha beta gamma delta", "alpha beta alpha"])
    path = tmp_path / "bpe.json"
    trainer.save(tok, str(path))
    loaded = trainer.load(str(path))
    assert loaded["vocab"] == tok["vocab"]
    assert loaded["merges"] == tok["merges"]
