"""Integration tests for dense embedding trainer registry wiring."""

from __future__ import annotations

import subprocess
import sys

import torch

from src import retrieval


def test_embedding_registry_has_dense() -> None:
    assert "dense" in retrieval.EMBEDDING_REGISTRY
    assert retrieval.EMBEDDING_REGISTRY["dense"] is retrieval.DenseEmbedder


def test_instantiate_tiny_embedder_from_registry() -> None:
    cls = retrieval.EMBEDDING_REGISTRY["dense"]
    cfg = retrieval.EmbedderConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        max_seq_len=8,
        embed_dim=16,
    )
    emb = cls(cfg)
    ids = torch.randint(1, cfg.vocab_size, (2, 8))
    out = emb(ids)
    assert out.shape == (2, cfg.embed_dim)


def test_pairwise_similarity_scores_between_two_text_pairs() -> None:
    cfg = retrieval.EmbedderConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        max_seq_len=8,
        embed_dim=16,
    )
    torch.manual_seed(0)
    emb = retrieval.DenseEmbedder(cfg)
    emb.train(False)
    # Two "text pairs" encoded as fake token-id sequences.
    pair_a_query = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]])
    pair_a_doc = torch.tensor([[1, 2, 3, 5, 0, 0, 0, 0]])
    pair_b_query = torch.tensor([[10, 11, 12, 13, 0, 0, 0, 0]])
    pair_b_doc = torch.tensor([[10, 11, 12, 14, 0, 0, 0, 0]])
    with torch.no_grad():
        qa = emb(pair_a_query)
        da = emb(pair_a_doc)
        qb = emb(pair_b_query)
        db = emb(pair_b_doc)
    score_a = float((qa * da).sum(dim=-1).item())
    score_b = float((qb * db).sum(dim=-1).item())
    # Both scores are cosine similarities of unit vectors, so in [-1, 1].
    for s in (score_a, score_b):
        assert -1.0 - 1e-5 <= s <= 1.0 + 1e-5


def test_existing_retrieval_entries_intact() -> None:
    assert "bm25" in retrieval.RETRIEVER_REGISTRY
    assert "hybrid_rrf" in retrieval.RETRIEVER_REGISTRY
    assert "cross_encoder" in retrieval.RERANKER_REGISTRY
    assert retrieval.RETRIEVER_REGISTRY["bm25"] is retrieval.BM25Retriever
    assert retrieval.RETRIEVER_REGISTRY["hybrid_rrf"] is retrieval.HybridRetriever


def test_hermetic_import_does_not_pull_src_model() -> None:
    code = (
        "import sys\n"
        "import src.retrieval  # noqa: F401\n"
        "bad = [m for m in sys.modules if m == 'src.model' "
        "or m.startswith('src.model.')]\n"
        "assert not bad, f'src.model leaked into retrieval import: {bad}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert result.returncode == 0, (
        f"hermetic import failed:\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
    )
    assert "OK" in result.stdout
