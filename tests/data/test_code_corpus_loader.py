"""Unit tests for src.data.code_corpus_loader."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import pytest
import torch

from src.data.code_corpus_loader import (
    CodeChunk,
    CodeCorpusLoader,
    FIM_MIDDLE,
    FIM_PREFIX,
    FIM_SUFFIX,
)


def _char_tokenizer(text: str) -> list[int]:
    # Simple deterministic tokenizer: ord() per char, mod 1000.
    return [ord(c) % 1000 for c in text]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _populate(tmp_path: Path, n: int = 3) -> list[Path]:
    files = []
    for i in range(n):
        p = tmp_path / f"mod_{i}.py"
        _write(p, f"def fn_{i}():\n    return {i}\n")
        files.append(p)
    return files


# --------------------------------------------------------------------- walk

def test_walk_yields_one_chunk_per_file(tmp_path: Path) -> None:
    _populate(tmp_path, n=4)
    loader = CodeCorpusLoader(tokenizer=_char_tokenizer)
    chunks = list(loader.walk(str(tmp_path)))
    assert len(chunks) == 4
    for c in chunks:
        assert isinstance(c, CodeChunk)
        assert c.file_path.endswith(".py")
        assert c.language == "python"
        assert len(c.tokens) > 0


def test_extension_filter(tmp_path: Path) -> None:
    _write(tmp_path / "a.py", "x = 1\n")
    _write(tmp_path / "b.md", "# readme\n")
    _write(tmp_path / "c.txt", "hello\n")
    loader = CodeCorpusLoader(tokenizer=_char_tokenizer, extensions=(".py",))
    chunks = list(loader.walk(str(tmp_path)))
    assert len(chunks) == 1
    assert chunks[0].file_path.endswith("a.py")


def test_language_detection(tmp_path: Path) -> None:
    loader = CodeCorpusLoader(tokenizer=_char_tokenizer)
    assert loader.detect_language("foo/bar.py") == "python"
    assert loader.detect_language("foo/bar.rs") == "rust"
    assert loader.detect_language("foo/bar.go") == "go"
    assert loader.detect_language("foo/bar.unknown") == "unknown"


# ------------------------------------------------------------------ packing

def test_pack_iter_yields_fixed_length(tmp_path: Path) -> None:
    # Make files big enough to produce multiple chunks of size 64.
    for i in range(3):
        _write(tmp_path / f"f{i}.py", "x" * 200)
    loader = CodeCorpusLoader(
        tokenizer=_char_tokenizer, chunk_size=64, file_sep_token_id=7
    )
    seqs = list(loader.pack_iter(str(tmp_path)))
    assert len(seqs) >= 1
    for s in seqs:
        assert isinstance(s, torch.Tensor)
        assert s.dtype == torch.long
        assert s.shape == (64,)


def test_file_sep_inserted_between_files(tmp_path: Path) -> None:
    _write(tmp_path / "a.py", "AAAA")
    _write(tmp_path / "b.py", "BBBB")
    loader = CodeCorpusLoader(
        tokenizer=_char_tokenizer, chunk_size=9, file_sep_token_id=999,
    )
    seqs = list(loader.pack_iter(str(tmp_path)))
    assert len(seqs) == 1
    # Two 4-char files + 1 separator = 9 tokens.
    assert 999 in seqs[0].tolist()


def test_chunk_size_tiny_works(tmp_path: Path) -> None:
    _write(tmp_path / "a.py", "a" * 500)
    loader = CodeCorpusLoader(tokenizer=_char_tokenizer, chunk_size=128)
    seqs = list(loader.pack_iter(str(tmp_path)))
    assert all(s.shape == (128,) for s in seqs)
    assert len(seqs) >= 3


# --------------------------------------------------------------------- FIM

def test_fim_rate_zero_never_augments(tmp_path: Path) -> None:
    _write(tmp_path / "a.py", "hello world 12345")
    loader = CodeCorpusLoader(
        tokenizer=lambda s: list(s.encode()),
        apply_fim=True,
        fim_rate=0.0,
        rng=random.Random(0),
    )
    for _ in range(5):
        chunks = list(loader.walk(str(tmp_path)))
        # Re-read text from tokens: since tokenizer is bytes, decode.
        decoded = bytes(chunks[0].tokens).decode("utf-8", errors="replace")
        assert FIM_PREFIX not in decoded


def test_fim_rate_one_always_augments(tmp_path: Path) -> None:
    _write(tmp_path / "a.py", "hello world 12345")
    loader = CodeCorpusLoader(
        tokenizer=lambda s: list(s.encode()),
        apply_fim=True,
        fim_rate=1.0,
        rng=random.Random(0),
    )
    for _ in range(5):
        chunks = list(loader.walk(str(tmp_path)))
        decoded = bytes(chunks[0].tokens).decode("utf-8", errors="replace")
        assert FIM_PREFIX in decoded
        assert FIM_SUFFIX in decoded
        assert FIM_MIDDLE in decoded


def test_fim_rate_half_approx(tmp_path: Path) -> None:
    _write(tmp_path / "a.py", "hello world " * 10)
    hits = 0
    trials = 20
    rng = random.Random(42)
    loader = CodeCorpusLoader(
        tokenizer=lambda s: list(s.encode()),
        apply_fim=True,
        fim_rate=0.5,
        rng=rng,
    )
    for _ in range(trials):
        chunks = list(loader.walk(str(tmp_path)))
        decoded = bytes(chunks[0].tokens).decode("utf-8", errors="replace")
        if FIM_PREFIX in decoded:
            hits += 1
    # Allow wide margin: 3..17 hits out of 20.
    assert 3 <= hits <= 17, f"got {hits}/{trials}"


def test_fim_determinism_seeded(tmp_path: Path) -> None:
    _write(tmp_path / "a.py", "hello world " * 8)

    def run():
        loader = CodeCorpusLoader(
            tokenizer=lambda s: list(s.encode()),
            apply_fim=True,
            fim_rate=0.5,
            rng=random.Random(1234),
        )
        return [list(loader.walk(str(tmp_path)))[0].tokens for _ in range(5)]

    assert run() == run()


# ------------------------------------------------------------------- misc

def test_empty_repo_yields_nothing(tmp_path: Path) -> None:
    loader = CodeCorpusLoader(tokenizer=_char_tokenizer)
    assert list(loader.walk(str(tmp_path))) == []
    assert list(loader.pack_iter(str(tmp_path))) == []


def test_binary_file_skipped(tmp_path: Path) -> None:
    _write(tmp_path / "ok.py", "print('hi')\n")
    # Write a file named .py but containing NUL bytes (fake binary).
    (tmp_path / "bad.py").write_bytes(b"\x00\x01\x02binary\x00data\x00")
    loader = CodeCorpusLoader(tokenizer=_char_tokenizer)
    chunks = list(loader.walk(str(tmp_path)))
    assert len(chunks) == 1
    assert chunks[0].file_path.endswith("ok.py")


def test_custom_tokenizer_honored(tmp_path: Path) -> None:
    _write(tmp_path / "a.py", "xyz")
    sentinel = [42, 42, 42, 42, 42]
    loader = CodeCorpusLoader(tokenizer=lambda _t: sentinel)
    chunks = list(loader.walk(str(tmp_path)))
    assert chunks[0].tokens == sentinel


def test_50_files_pack_fast(tmp_path: Path) -> None:
    for i in range(50):
        _write(tmp_path / f"f{i}.py", f"# file {i}\n" + ("x = 1\n" * 20))
    loader = CodeCorpusLoader(
        tokenizer=_char_tokenizer, chunk_size=256, file_sep_token_id=3,
    )
    t0 = time.perf_counter()
    seqs = list(loader.pack_iter(str(tmp_path)))
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"pack_iter took {elapsed:.3f}s"
    assert all(s.shape == (256,) for s in seqs)
