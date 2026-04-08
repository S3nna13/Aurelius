"""Tests for the RAG pipeline (rag_pipeline.py)."""

from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.rag_pipeline import (
    RAGConfig,
    Document,
    Chunk,
    RetrievalResult,
    DocumentChunker,
    DenseRetriever,
    BM25Retriever,
    RAGPipeline,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(0)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


def _make_documents(n: int = 3, length: int = 400) -> list[Document]:
    """Create n simple documents each with roughly length characters of text."""
    docs = []
    for i in range(n):
        text = f"Document {i}: " + ("word " * (length // 5))
        docs.append(Document(doc_id=f"doc_{i}", title=f"Title {i}", text=text))
    return docs


def _simple_embed_fn(dim: int = 64):
    """Return a deterministic embed_fn based on character hashing."""
    def embed(text: str) -> torch.Tensor:
        torch.manual_seed(abs(hash(text)) % (2 ** 31))
        return torch.randn(dim)
    return embed


# ---------------------------------------------------------------------------
# DocumentChunker tests
# ---------------------------------------------------------------------------

def test_chunk_document_basic():
    """Single doc with chunk_size=100 should produce multiple chunks."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "word " * 60  # 300 characters
    doc = Document(doc_id="d0", title="T", text=text)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.doc_id == "d0"
        assert chunk.text  # non-empty


def test_chunk_overlap():
    """Consecutive chunks should have overlapping character ranges."""
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=20)
    text = "abcde " * 30  # 180 characters, word-friendly
    doc = Document(doc_id="d0", title="T", text=text)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    # chunk[1].start_char should be less than chunk[0].end_char (overlap)
    assert chunks[1].start_char < chunks[0].end_char


def test_chunk_corpus_flat():
    """3 documents should produce a flat list containing chunks from all documents."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    docs = _make_documents(n=3, length=400)
    chunks = chunker.chunk_corpus(docs)
    assert len(chunks) > 0
    doc_ids_seen = {c.doc_id for c in chunks}
    assert doc_ids_seen == {"doc_0", "doc_1", "doc_2"}
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


# ---------------------------------------------------------------------------
# DenseRetriever tests
# ---------------------------------------------------------------------------

def test_dense_retriever_index_and_retrieve():
    """After indexing, retrieve should return a RetrievalResult with top_k chunks."""
    embed_fn = _simple_embed_fn(dim=64)
    retriever = DenseRetriever(embed_fn=embed_fn, top_k=3)

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    docs = _make_documents(n=3, length=400)
    chunks = chunker.chunk_corpus(docs)
    retriever.index(chunks)

    result = retriever.retrieve("some query text")
    assert isinstance(result, RetrievalResult)
    assert len(result.retrieved_chunks) == 3
    assert len(result.scores) == 3
    assert all(isinstance(c, Chunk) for c in result.retrieved_chunks)


def test_dense_retriever_scores_decreasing():
    """Scores returned by retrieve() should be in descending order."""
    embed_fn = _simple_embed_fn(dim=64)
    retriever = DenseRetriever(embed_fn=embed_fn, top_k=5)

    chunker = DocumentChunker(chunk_size=80, chunk_overlap=10)
    docs = _make_documents(n=3, length=400)
    chunks = chunker.chunk_corpus(docs)
    retriever.index(chunks)

    result = retriever.retrieve("test query")
    scores = result.scores
    assert len(scores) >= 2
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Scores not descending: {scores}"


# ---------------------------------------------------------------------------
# BM25Retriever tests
# ---------------------------------------------------------------------------

def test_bm25_index_and_retrieve():
    """BM25 retrieve should return a RetrievalResult with chunks and scores."""
    retriever = BM25Retriever(top_k=2)

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    docs = _make_documents(n=3, length=400)
    chunks = chunker.chunk_corpus(docs)
    retriever.index(chunks)

    result = retriever.retrieve("word document")
    assert isinstance(result, RetrievalResult)
    assert len(result.retrieved_chunks) == 2
    assert len(result.scores) == 2
    assert all(isinstance(c, Chunk) for c in result.retrieved_chunks)


def test_bm25_relevant_chunk_ranked_high():
    """A chunk containing the exact query words should be ranked highest."""
    retriever = BM25Retriever(top_k=3)

    chunks = [
        Chunk(chunk_id="c0", doc_id="d0", text="the cat sat on the mat", start_char=0, end_char=22),
        Chunk(chunk_id="c1", doc_id="d0", text="quantum physics entanglement superposition", start_char=22, end_char=63),
        Chunk(chunk_id="c2", doc_id="d0", text="the dog barked loudly in the park", start_char=63, end_char=96),
    ]
    retriever.index(chunks)

    result = retriever.retrieve("cat sat mat")
    assert result.retrieved_chunks[0].chunk_id == "c0"


# ---------------------------------------------------------------------------
# RAGPipeline tests
# ---------------------------------------------------------------------------

def _make_pipeline(model, cfg: AureliusConfig, rag_config: RAGConfig = None) -> RAGPipeline:
    """Build a minimal RAGPipeline around the small test model."""
    def tokenizer_encode(text: str) -> torch.Tensor:
        ids = [min(b, cfg.vocab_size - 1) for b in text.encode("utf-8")]
        ids = ids[: cfg.max_seq_len]
        if not ids:
            ids = [0]
        return torch.tensor(ids, dtype=torch.long)

    def tokenizer_decode(token_ids: torch.Tensor) -> str:
        return "".join(chr(t.item()) for t in token_ids.flatten())

    embed_fn = _simple_embed_fn(dim=cfg.d_model)
    retriever = DenseRetriever(embed_fn=embed_fn, top_k=2)
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    config = rag_config if rag_config is not None else RAGConfig(top_k=2, max_context_tokens=64)

    return RAGPipeline(
        model=model,
        tokenizer_encode=tokenizer_encode,
        tokenizer_decode=tokenizer_decode,
        retriever=retriever,
        chunker=chunker,
        config=config,
    )


def test_rag_pipeline_ingest_returns_count(small_model, small_cfg):
    """ingest() should return a positive chunk count."""
    pipeline = _make_pipeline(small_model, small_cfg)
    docs = _make_documents(n=2, length=300)
    count = pipeline.ingest(docs)
    assert count > 0


def test_rag_pipeline_query_returns_dict(small_model, small_cfg):
    """query() should return a dict with the required keys."""
    pipeline = _make_pipeline(small_model, small_cfg)
    docs = _make_documents(n=2, length=300)
    pipeline.ingest(docs)

    result = pipeline.query("What is document 0 about?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "retrieved_chunks" in result
    assert "scores" in result
    assert isinstance(result["retrieved_chunks"], list)
    assert isinstance(result["scores"], list)


# ---------------------------------------------------------------------------
# DenseRetriever.update() test
# ---------------------------------------------------------------------------

def test_dense_retriever_update():
    """update() should add new chunks to the existing index."""
    embed_fn = _simple_embed_fn(dim=64)
    retriever = DenseRetriever(embed_fn=embed_fn, top_k=3)

    initial_chunks = [
        Chunk(chunk_id="c0", doc_id="d0", text="hello world", start_char=0, end_char=11),
        Chunk(chunk_id="c1", doc_id="d0", text="foo bar baz", start_char=11, end_char=22),
    ]
    retriever.index(initial_chunks)
    assert len(retriever._chunks) == 2

    new_chunks = [
        Chunk(chunk_id="c2", doc_id="d1", text="new document chunk", start_char=0, end_char=18),
    ]
    retriever.update(new_chunks)
    assert len(retriever._chunks) == 3
    assert retriever._embeddings is not None
    assert retriever._embeddings.shape[0] == 3

    result = retriever.retrieve("new document", top_k=3)
    assert len(result.retrieved_chunks) == 3
