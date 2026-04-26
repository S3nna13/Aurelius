"""Integration tests for InstructionPrefixEmbedder via ``src.retrieval``."""

from __future__ import annotations

import pytest
import torch

import src.retrieval as retrieval
from src.retrieval import (
    EMBEDDING_REGISTRY,
    INSTRUCTION_PREFIXES,
    DenseEmbedder,
    EmbedderConfig,
    InstructionPrefixEmbedder,
)


def _tokenize(text: str) -> list[int]:
    # word-level hash tokenizer in [1, 31]; keeps sequences short so
    # prefixed inputs fit within the tiny max_seq_len.
    return [((sum(ord(c) for c in w) % 31) + 1) for w in text.split()]


def _build() -> InstructionPrefixEmbedder:
    torch.manual_seed(0)
    cfg = EmbedderConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        max_seq_len=64,
        dropout=0.0,
        pad_token_id=0,
        embed_dim=16,
    )
    emb = DenseEmbedder(cfg)
    emb.train(False)
    return InstructionPrefixEmbedder(emb, _tokenize)


def test_surface_exposed_from_package() -> None:
    assert hasattr(retrieval, "InstructionPrefixEmbedder")
    assert hasattr(retrieval, "INSTRUCTION_PREFIXES")
    assert retrieval.InstructionPrefixEmbedder is InstructionPrefixEmbedder
    assert retrieval.INSTRUCTION_PREFIXES is INSTRUCTION_PREFIXES


def test_prior_embedding_registry_entries_intact() -> None:
    # The dense embedder registration from dense_embedding_trainer must
    # survive the new module's import side-effects.
    assert "dense" in EMBEDDING_REGISTRY
    assert EMBEDDING_REGISTRY["dense"] is DenseEmbedder


def test_end_to_end_similarity_ranking_is_deterministic() -> None:
    wrapper = _build()
    passages = ["alpha beta", "gamma delta", "epsilon"]
    sims = wrapper.similarity(
        "alpha beta",
        passages,
        query_task="query",
        passage_task="passage",
    )
    assert len(sims) == 3
    assert all(isinstance(s, float) for s in sims)
    # Re-running must give the exact same scores (deterministic, eval mode).
    sims2 = wrapper.similarity(
        "alpha beta",
        passages,
        query_task="query",
        passage_task="passage",
    )
    assert sims == pytest.approx(sims2, abs=1e-7)


def test_end_to_end_self_match_peaks_at_one() -> None:
    wrapper = _build()
    q = "needle"
    passages = ["haystack one", "needle", "haystack three"]
    sims = wrapper.similarity(q, passages, query_task="passage", passage_task="passage")
    # Same text + same task => cosine sim == 1.0.
    assert sims[1] == pytest.approx(1.0, abs=1e-5)
    assert sims[1] >= max(sims[0], sims[2])


def test_all_six_tasks_encodable() -> None:
    wrapper = _build()
    for task in INSTRUCTION_PREFIXES:
        out = wrapper.encode("text", task=task)
        assert out.shape == (16,)
        assert torch.isfinite(out).all()
