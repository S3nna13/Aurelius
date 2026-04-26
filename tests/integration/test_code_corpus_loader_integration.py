"""Integration test: CodeCorpusLoader end-to-end over a small tmp repo.

Checks:
- Module is importable from src.data (existing entries intact).
- End-to-end walk + pack_iter produces valid packed tensors.
- Multi-language (.py, .rs, .go) files are traversed.
- Existing src.data exports (e.g. FIM_PREFIX, AURELIUS_MIX) still import.
"""

from __future__ import annotations

from pathlib import Path

import torch


def test_existing_data_exports_intact() -> None:
    # Existing symbols must still work after our additive module exists.
    from src.data import AURELIUS_MIX, fim_transform  # noqa: F401
    from src.data import FIM_PREFIX as PKG_FIM_PREFIX

    assert isinstance(PKG_FIM_PREFIX, str)


def test_loader_importable_from_src_data() -> None:
    # Direct module import path is the canonical one; package-level
    # re-export is optional.
    from src.data.code_corpus_loader import (
        FIM_PREFIX,
        CodeChunk,
        CodeCorpusLoader,
    )

    assert CodeChunk is not None
    assert CodeCorpusLoader is not None
    assert isinstance(FIM_PREFIX, str)


def test_end_to_end_small_repo(tmp_path: Path) -> None:
    from src.data.code_corpus_loader import CodeCorpusLoader

    # Build a small polyglot repo.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text(
        "def add(a, b):\n    return a + b\n" * 10,
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "helper.py").write_text(
        "from pkg.core import add\n\ndef double(x):\n    return add(x, x)\n" * 5,
        encoding="utf-8",
    )
    (tmp_path / "lib.rs").write_text(
        'fn main() { println!("hello"); }\n' * 8,
        encoding="utf-8",
    )
    (tmp_path / "svc.go").write_text(
        "package main\nfunc main() {}\n" * 8,
        encoding="utf-8",
    )
    # Decoy: not-a-source file.
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")

    loader = CodeCorpusLoader(
        tokenizer=lambda s: [ord(c) % 500 for c in s],
        chunk_size=128,
        file_sep_token_id=11,
    )

    chunks = list(loader.walk(str(tmp_path)))
    langs = {c.language for c in chunks}
    assert "python" in langs
    assert "rust" in langs
    assert "go" in langs
    # README.md should be excluded.
    assert all(not c.file_path.endswith(".md") for c in chunks)

    seqs = list(loader.pack_iter(str(tmp_path)))
    assert len(seqs) >= 1
    for s in seqs:
        assert isinstance(s, torch.Tensor)
        assert s.shape == (128,)
        assert s.dtype == torch.long
    # file_sep should appear somewhere in the packed stream.
    flat = torch.cat(seqs).tolist()
    assert 11 in flat
