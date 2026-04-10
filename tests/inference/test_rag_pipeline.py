"""Tests for the RAG pipeline (rag_pipeline.py) — original 12 + 16 dense-RAG tests."""

from __future__ import annotations

import torch
import pytest

from src.inference.rag_pipeline import (
    RAGConfig,
    chunk_text,
    BM25Index,
    reciprocal_rank_fusion,
    DenseRetriever,
    RAGPipeline,
    # Dense RAG additions
    DenseRAGConfig,
    DocumentStore,
    QueryEncoder,
    build_augmented_input,
    DenseRAGPipeline,
    DenseRAGTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared tiny model fixture
# ---------------------------------------------------------------------------

def _tiny_model() -> AureliusTransformer:
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


EMBED_DIM = 64
VOCAB_SIZE = 256


# ---------------------------------------------------------------------------
# RAGConfig tests
# ---------------------------------------------------------------------------

def test_ragconfig_defaults():
    """RAGConfig should have the specified default values."""
    cfg = RAGConfig()
    assert cfg.chunk_size == 256
    assert cfg.chunk_overlap == 32
    assert cfg.n_retrieve == 5
    assert cfg.n_rerank == 3
    assert cfg.fusion_alpha == 0.5
    assert cfg.max_context_len == 1024


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------

def test_chunk_text_basic():
    """chunk_text should split text into multiple word-level chunks."""
    text = " ".join([f"word{i}" for i in range(100)])  # 100 words
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk.split()) <= 20


def test_chunk_text_overlap_creates_shared_words():
    """Consecutive chunks should share words (overlap > 0)."""
    text = " ".join([f"w{i}" for i in range(50)])
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert len(chunks) >= 2
    # The last 3 words of chunk[0] should appear at the start of chunk[1]
    words0 = chunks[0].split()
    words1 = chunks[1].split()
    shared = words0[-3:]
    assert words1[:3] == shared, (
        f"Expected overlap words {shared} at start of chunk[1], got {words1[:3]}"
    )


def test_chunk_text_empty_returns_empty_list():
    """chunk_text on empty string should return []."""
    assert chunk_text("", chunk_size=10, overlap=2) == []


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------

def test_bm25_search_returns_list_of_tuples():
    """BM25Index.search should return a list of (int, float) tuples."""
    index = BM25Index()
    docs = ["the quick brown fox", "jumped over the lazy dog", "hello world"]
    index.index(docs)
    results = index.search("fox", top_k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    for item in results:
        assert isinstance(item, tuple)
        assert len(item) == 2
        idx, score = item
        assert isinstance(idx, int)
        assert isinstance(score, float)


def test_bm25_search_exact_query_term_high_score():
    """Document containing the exact query term should rank first."""
    index = BM25Index()
    docs = [
        "machine learning is a subfield of artificial intelligence",
        "the cat sat on the mat in the garden",
        "neural networks are used in deep learning machine tasks",
    ]
    index.index(docs)
    results = index.search("cat sat mat", top_k=3)
    # The second document (index 1) contains all query terms
    top_idx = results[0][0]
    assert top_idx == 1, f"Expected doc 1 at top, got {top_idx}"


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion tests
# ---------------------------------------------------------------------------

def test_reciprocal_rank_fusion_merges_correctly():
    """RRF should promote docs that appear high in multiple lists."""
    list1 = [0, 1, 2, 3]
    list2 = [1, 0, 3, 2]
    fused = reciprocal_rank_fusion([list1, list2])
    # Doc 0 is rank-1 in list1 and rank-2 in list2
    # Doc 1 is rank-2 in list1 and rank-1 in list2
    # Both should be near the top; all input docs should be present
    assert set(fused) == {0, 1, 2, 3}
    # The top two should be 0 and 1 (highest combined RRF scores)
    assert set(fused[:2]) == {0, 1}


def test_reciprocal_rank_fusion_single_list():
    """RRF on a single ranked list should preserve the original order."""
    ranking = [3, 1, 4, 0, 5]
    fused = reciprocal_rank_fusion([ranking])
    # All docs should appear; order should match the single list (rank 1 → highest score)
    assert set(fused) == set(ranking)
    assert fused[0] == 3  # rank-1 item gets the highest RRF score


# ---------------------------------------------------------------------------
# DenseRetriever tests
# ---------------------------------------------------------------------------

def test_dense_retriever_encode_output_shape():
    """DenseRetriever.encode should return a (N, embed_dim) tensor."""
    retriever = DenseRetriever(embed_dim=64)
    texts = ["hello world", "foo bar baz", "test document"]
    embeddings = retriever.encode(texts)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (3, 64)


# ---------------------------------------------------------------------------
# RAGPipeline tests
# ---------------------------------------------------------------------------

def test_rag_pipeline_index_documents_runs_without_error():
    """index_documents should complete without raising an exception."""
    cfg = RAGConfig(chunk_size=10, chunk_overlap=2, n_retrieve=2, n_rerank=2)
    pipeline = RAGPipeline(cfg)
    docs = [
        "The quick brown fox jumped over the lazy dog near the river bank.",
        "Machine learning models require large amounts of training data to generalise.",
        "Python is a popular programming language used in data science and AI research.",
    ]
    # Should not raise
    pipeline.index_documents(docs)


def test_rag_pipeline_retrieve_returns_list_of_strings():
    """retrieve should return a list of strings (chunk texts)."""
    cfg = RAGConfig(chunk_size=10, chunk_overlap=2, n_retrieve=3, n_rerank=2)
    pipeline = RAGPipeline(cfg)
    docs = [
        "The quick brown fox jumped over the lazy dog near the river bank.",
        "Machine learning models require large amounts of training data.",
        "Python is a popular programming language used in data science.",
    ]
    pipeline.index_documents(docs)
    results = pipeline.retrieve("machine learning python")
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


def test_rag_pipeline_format_context_contains_query():
    """format_context should include the query in the returned string."""
    cfg = RAGConfig()
    pipeline = RAGPipeline(cfg)
    query = "what is machine learning?"
    chunks = ["Machine learning is a subset of AI.", "It uses data to train models."]
    context = pipeline.format_context(query, chunks)
    assert query in context
    assert "Context:" in context
    assert "Query:" in context


# ===========================================================================
# Dense RAG tests (16 tests)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. DenseRAGConfig defaults
# ---------------------------------------------------------------------------

def test_dense_ragconfig_defaults():
    """DenseRAGConfig should expose the correct default field values."""
    cfg = DenseRAGConfig()
    assert cfg.n_docs == 3
    assert cfg.max_doc_len == 64
    assert cfg.max_answer_len == 32
    assert cfg.score_method == "dot"
    assert cfg.prepend_docs is True


# ---------------------------------------------------------------------------
# 2. DocumentStore starts empty
# ---------------------------------------------------------------------------

def test_document_store_starts_empty():
    store = DocumentStore(embed_dim=EMBED_DIM)
    assert len(store) == 0


# ---------------------------------------------------------------------------
# 3. DocumentStore.add increases length
# ---------------------------------------------------------------------------

def test_document_store_add_increases_length():
    store = DocumentStore(embed_dim=EMBED_DIM)
    doc_ids = torch.randint(0, VOCAB_SIZE, (10,))
    emb = torch.randn(EMBED_DIM)
    store.add(doc_ids, emb)
    assert len(store) == 1
    store.add(doc_ids, emb)
    assert len(store) == 2


# ---------------------------------------------------------------------------
# 4. DocumentStore.add_batch adds multiple docs
# ---------------------------------------------------------------------------

def test_document_store_add_batch():
    store = DocumentStore(embed_dim=EMBED_DIM)
    n = 5
    doc_ids_list = [torch.randint(0, VOCAB_SIZE, (8,)) for _ in range(n)]
    embeddings = torch.randn(n, EMBED_DIM)
    store.add_batch(doc_ids_list, embeddings)
    assert len(store) == n


# ---------------------------------------------------------------------------
# 5. DocumentStore.retrieve returns n_docs results
# ---------------------------------------------------------------------------

def test_document_store_retrieve_returns_n_docs():
    store = DocumentStore(embed_dim=EMBED_DIM)
    for _ in range(10):
        store.add(torch.randint(0, VOCAB_SIZE, (5,)), torch.randn(EMBED_DIM))
    query_emb = torch.randn(EMBED_DIM)
    results = store.retrieve(query_emb, n_docs=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# 6. DocumentStore.retrieve results sorted by score descending
# ---------------------------------------------------------------------------

def test_document_store_retrieve_sorted_descending():
    store = DocumentStore(embed_dim=EMBED_DIM)
    for _ in range(8):
        store.add(torch.randint(0, VOCAB_SIZE, (5,)), torch.randn(EMBED_DIM))
    query_emb = torch.randn(EMBED_DIM)
    results = store.retrieve(query_emb, n_docs=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by score descending."


# ---------------------------------------------------------------------------
# 7. DocumentStore.retrieve cosine vs dot both work
# ---------------------------------------------------------------------------

def test_document_store_retrieve_cosine_and_dot():
    store = DocumentStore(embed_dim=EMBED_DIM)
    for _ in range(4):
        store.add(torch.randint(0, VOCAB_SIZE, (5,)), torch.randn(EMBED_DIM))
    query_emb = torch.randn(EMBED_DIM)

    dot_results = store.retrieve(query_emb, n_docs=2, method="dot")
    cos_results = store.retrieve(query_emb, n_docs=2, method="cosine")

    assert len(dot_results) == 2
    assert len(cos_results) == 2
    # Both return (Tensor, float) tuples
    for doc_ids, score in dot_results + cos_results:
        assert isinstance(doc_ids, torch.Tensor)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 8. QueryEncoder output shape is (B, embed_dim)
# ---------------------------------------------------------------------------

def test_query_encoder_output_shape():
    enc = QueryEncoder(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    input_ids = torch.randint(0, VOCAB_SIZE, (3, 12))  # batch=3, seq=12
    out = enc(input_ids)
    assert out.shape == (3, EMBED_DIM), f"Expected (3, {EMBED_DIM}), got {out.shape}"


# ---------------------------------------------------------------------------
# 9. QueryEncoder is differentiable
# ---------------------------------------------------------------------------

def test_query_encoder_is_differentiable():
    enc = QueryEncoder(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
    out = enc(input_ids)
    loss = out.sum()
    loss.backward()  # must not raise
    assert enc.embed.weight.grad is not None


# ---------------------------------------------------------------------------
# 10. build_augmented_input output shape is (max_total_len,)
# ---------------------------------------------------------------------------

def test_build_augmented_input_output_shape():
    query_ids = torch.randint(0, VOCAB_SIZE, (10,))
    doc1 = torch.randint(0, VOCAB_SIZE, (8,))
    doc2 = torch.randint(0, VOCAB_SIZE, (6,))
    out = build_augmented_input(query_ids, [doc1, doc2], max_total_len=32)
    assert out.shape == (32,), f"Expected shape (32,), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. build_augmented_input truncates correctly when too long
# ---------------------------------------------------------------------------

def test_build_augmented_input_truncates():
    """When concat > max_total_len, only the tail (query end) is kept."""
    query_ids = torch.arange(10)               # 0..9
    doc_ids = torch.arange(100, 150)           # long doc: 50 tokens
    max_len = 20
    out = build_augmented_input(query_ids, [doc_ids], max_total_len=max_len)
    assert out.shape == (max_len,)
    # The last tokens of the combined sequence are the query_ids tail
    # Combined is [doc(50), query(10)] = 60 tokens; keep last 20
    combined = torch.cat([doc_ids, query_ids])  # 60
    expected = combined[-max_len:]
    assert torch.equal(out, expected), "Truncation did not keep the tail of the sequence."


# ---------------------------------------------------------------------------
# 12. DenseRAGPipeline.retrieve returns list of (Tensor, float) tuples
# ---------------------------------------------------------------------------

def test_dense_ragpipeline_retrieve_returns_correct_types():
    model = _tiny_model()
    enc = QueryEncoder(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    store = DocumentStore(embed_dim=EMBED_DIM)
    for _ in range(4):
        store.add(torch.randint(0, VOCAB_SIZE, (8,)), torch.randn(EMBED_DIM))

    cfg = DenseRAGConfig(n_docs=2, score_method="dot")
    pipeline = DenseRAGPipeline(model, enc, store, cfg)

    query_ids = torch.randint(0, VOCAB_SIZE, (5,))
    results = pipeline.retrieve(query_ids)
    assert isinstance(results, list)
    assert len(results) == 2
    for doc_ids, score in results:
        assert isinstance(doc_ids, torch.Tensor)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 13. DenseRAGPipeline.generate returns (Tensor, list)
# ---------------------------------------------------------------------------

def test_dense_ragpipeline_generate_return_types():
    model = _tiny_model()
    enc = QueryEncoder(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    store = DocumentStore(embed_dim=EMBED_DIM)
    for _ in range(3):
        store.add(torch.randint(0, VOCAB_SIZE, (6,)), torch.randn(EMBED_DIM))

    cfg = DenseRAGConfig(n_docs=2)
    pipeline = DenseRAGPipeline(model, enc, store, cfg)

    query_ids = torch.randint(0, VOCAB_SIZE, (4,))
    next_token, retrieved = pipeline.generate(query_ids)

    assert isinstance(next_token, torch.Tensor)
    assert next_token.shape == (1,)
    assert isinstance(retrieved, list)


# ---------------------------------------------------------------------------
# 14. DenseRAGPipeline.score_answer returns float
# ---------------------------------------------------------------------------

def test_dense_ragpipeline_score_answer_returns_float():
    model = _tiny_model()
    enc = QueryEncoder(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    store = DocumentStore(embed_dim=EMBED_DIM)
    for _ in range(3):
        store.add(torch.randint(0, VOCAB_SIZE, (6,)), torch.randn(EMBED_DIM))

    cfg = DenseRAGConfig(n_docs=2)
    pipeline = DenseRAGPipeline(model, enc, store, cfg)

    query_ids = torch.randint(0, VOCAB_SIZE, (5,))
    answer_ids = torch.randint(0, VOCAB_SIZE, (3,))
    score = pipeline.score_answer(query_ids, answer_ids)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 15. DenseRAGTrainer.train_step returns dict with correct keys
# ---------------------------------------------------------------------------

def test_dense_ragtrainer_train_step_keys():
    model = _tiny_model()
    enc = QueryEncoder(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    store = DocumentStore(embed_dim=EMBED_DIM)
    for _ in range(3):
        store.add(torch.randint(0, VOCAB_SIZE, (6,)), torch.randn(EMBED_DIM))

    cfg = DenseRAGConfig(n_docs=2)
    pipeline = DenseRAGPipeline(model, enc, store, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = DenseRAGTrainer(pipeline, optimizer)

    query_ids = torch.randint(0, VOCAB_SIZE, (5,))
    answer_ids = torch.randint(0, VOCAB_SIZE, (4,))
    result = trainer.train_step(query_ids, answer_ids)

    assert isinstance(result, dict)
    assert "loss" in result
    assert "n_docs_retrieved" in result
    assert "mean_doc_score" in result


# ---------------------------------------------------------------------------
# 16. DenseRAGTrainer.train_step loss is finite
# ---------------------------------------------------------------------------

def test_dense_ragtrainer_train_step_loss_finite():
    model = _tiny_model()
    enc = QueryEncoder(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    store = DocumentStore(embed_dim=EMBED_DIM)
    for _ in range(3):
        store.add(torch.randint(0, VOCAB_SIZE, (6,)), torch.randn(EMBED_DIM))

    cfg = DenseRAGConfig(n_docs=2)
    pipeline = DenseRAGPipeline(model, enc, store, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = DenseRAGTrainer(pipeline, optimizer)

    query_ids = torch.randint(0, VOCAB_SIZE, (5,))
    answer_ids = torch.randint(0, VOCAB_SIZE, (4,))
    result = trainer.train_step(query_ids, answer_ids)

    import math
    assert math.isfinite(result["loss"]), f"Loss is not finite: {result['loss']}"
