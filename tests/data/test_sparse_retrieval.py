"""
Tests for src/data/sparse_retrieval.py
"""

import pytest
from src.data.sparse_retrieval import (
    BM25Config,
    BM25Index,
    HybridRetriever,
    RAGContext,
    TF_IDF_Index,
    build_rag_context,
    tokenize_bm25,
)

# ---------------------------------------------------------------------------
# Sample corpus used across many tests
# ---------------------------------------------------------------------------

CORPUS = [
    "The cat sat on the mat.",
    "Dogs are loyal and friendly animals.",
    "Python is a great programming language.",
    "Machine learning enables computers to learn from data.",
    "The quick brown fox jumps over the lazy dog.",
]


# ---------------------------------------------------------------------------
# 1. BM25Config defaults
# ---------------------------------------------------------------------------

def test_bm25config_defaults():
    cfg = BM25Config()
    assert cfg.k1 == 1.5
    assert cfg.b == 0.75
    assert cfg.epsilon == 0.25


# ---------------------------------------------------------------------------
# 2. tokenize_bm25 lowercases and splits correctly
# ---------------------------------------------------------------------------

def test_tokenize_bm25_lowercase_and_split():
    tokens = tokenize_bm25("Hello, World! How are you?")
    assert "hello" in tokens
    assert "world" in tokens
    assert "how" in tokens
    assert "are" in tokens
    assert "you" in tokens


# ---------------------------------------------------------------------------
# 3. tokenize_bm25 filters empty strings
# ---------------------------------------------------------------------------

def test_tokenize_bm25_no_empty_strings():
    tokens = tokenize_bm25("  hello   world  ")
    assert "" not in tokens
    assert len(tokens) >= 2


def test_tokenize_bm25_punctuation_only_produces_no_empties():
    tokens = tokenize_bm25("!!! ,,, ???")
    assert "" not in tokens


# ---------------------------------------------------------------------------
# 4. BM25Index.build computes correct avgdl
# ---------------------------------------------------------------------------

def test_bm25index_build_avgdl():
    docs = ["one two three", "four five"]
    index = BM25Index()
    index.build(docs)
    expected_avgdl = (3 + 2) / 2
    assert abs(index._avgdl - expected_avgdl) < 1e-9


# ---------------------------------------------------------------------------
# 5. BM25Index.build IDF is positive for known terms
# ---------------------------------------------------------------------------

def test_bm25index_build_idf_positive():
    index = BM25Index()
    index.build(CORPUS)
    # "cat" appears in the first document — IDF must be >= epsilon
    assert index._idf.get("cat", 0) > 0
    assert index._idf.get("python", 0) > 0


# ---------------------------------------------------------------------------
# 6. BM25Index.score — exact match scores higher than no match
# ---------------------------------------------------------------------------

def test_bm25index_score_match_vs_no_match():
    docs = ["machine learning is great", "the weather is sunny today"]
    index = BM25Index()
    index.build(docs)
    score_match = index.score("machine learning", 0)
    score_no_match = index.score("machine learning", 1)
    assert score_match > score_no_match


# ---------------------------------------------------------------------------
# 7. BM25Index.search returns correct count (top_k)
# ---------------------------------------------------------------------------

def test_bm25index_search_top_k_count():
    index = BM25Index()
    index.build(CORPUS)
    results = index.search("cat", top_k=3)
    assert len(results) == 3


def test_bm25index_search_top_k_at_most_n_docs():
    index = BM25Index()
    index.build(CORPUS)
    results = index.search("cat", top_k=100)
    assert len(results) == len(CORPUS)


# ---------------------------------------------------------------------------
# 8. BM25Index.search — most relevant document ranks first
# ---------------------------------------------------------------------------

def test_bm25index_search_relevant_first():
    docs = [
        "Python programming language tutorial",
        "Cooking recipes for beginners",
        "Python is widely used in data science",
    ]
    index = BM25Index()
    index.build(docs)
    results = index.search("Python programming")
    top_doc = results[0][1]
    assert "Python" in top_doc or "python" in top_doc.lower()


# ---------------------------------------------------------------------------
# 9. BM25Index.__len__
# ---------------------------------------------------------------------------

def test_bm25index_len():
    index = BM25Index()
    index.build(CORPUS)
    assert len(index) == len(CORPUS)


def test_bm25index_len_empty():
    index = BM25Index()
    index.build([])
    assert len(index) == 0


# ---------------------------------------------------------------------------
# 10. TF_IDF_Index.search returns relevant results
# ---------------------------------------------------------------------------

def test_tfidf_search_returns_results():
    index = TF_IDF_Index()
    index.build(CORPUS)
    results = index.search("programming language", top_k=3)
    assert len(results) == 3
    # All scores should be non-negative
    for score, _ in results:
        assert score >= 0.0


# ---------------------------------------------------------------------------
# 11. TF_IDF_Index exact match ranks first
# ---------------------------------------------------------------------------

def test_tfidf_exact_match_ranks_first():
    docs = [
        "machine learning deep neural networks",
        "baking bread at home is fun",
        "machine learning is transforming industries",
    ]
    index = TF_IDF_Index()
    index.build(docs)
    results = index.search("machine learning")
    # Top result should be one of the ML docs
    assert "machine" in results[0][1].lower() or "learning" in results[0][1].lower()


# ---------------------------------------------------------------------------
# 12. HybridRetriever.build works without error
# ---------------------------------------------------------------------------

def test_hybrid_retriever_build_no_error():
    retriever = HybridRetriever()
    retriever.build(CORPUS)  # should not raise
    assert len(retriever._documents) == len(CORPUS)


# ---------------------------------------------------------------------------
# 13. HybridRetriever.search returns top_k results
# ---------------------------------------------------------------------------

def test_hybrid_retriever_search_top_k():
    retriever = HybridRetriever()
    retriever.build(CORPUS)
    results = retriever.search("cat mat", top_k=2)
    assert len(results) == 2


def test_hybrid_retriever_search_scores_in_range():
    retriever = HybridRetriever()
    retriever.build(CORPUS)
    results = retriever.search("Python data", top_k=5)
    for score, _ in results:
        assert 0.0 <= score <= 1.0 + 1e-9  # weights sum to 1.0


# ---------------------------------------------------------------------------
# 14. build_rag_context returns string containing query
# ---------------------------------------------------------------------------

def test_build_rag_context_contains_query():
    index = BM25Index()
    index.build(CORPUS)
    prompt = build_rag_context("cat on the mat", index)
    assert "cat on the mat" in prompt


def test_build_rag_context_contains_context_header():
    index = BM25Index()
    index.build(CORPUS)
    prompt = build_rag_context("cat", index)
    assert "Context:" in prompt


# ---------------------------------------------------------------------------
# 15. RAGContext fields accessible
# ---------------------------------------------------------------------------

def test_ragcontext_fields():
    ctx = RAGContext(
        query="What is BM25?",
        retrieved_docs=["BM25 is a ranking function.", "Used in information retrieval."],
        scores=[0.9, 0.7],
    )
    assert ctx.query == "What is BM25?"
    assert len(ctx.retrieved_docs) == 2
    assert ctx.scores[0] == pytest.approx(0.9)
    assert ctx.scores[1] == pytest.approx(0.7)
