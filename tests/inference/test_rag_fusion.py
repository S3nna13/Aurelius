"""Tests for src/inference/rag_fusion.py — RAG-Fusion with multi-query and RRF."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.rag_fusion import (
    RAGFusionConfig,
    Document,
    compute_query_doc_similarity,
    reciprocal_rank_fusion,
    generate_query_variations,
    MockRetriever,
    RAGFusionPipeline,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
    )
    torch.manual_seed(0)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def simple_encode(text: str) -> list:
    """Byte-level tokeniser: each character -> its ASCII value (clamped to [0, 255])."""
    return [min(ord(c), 255) for c in text]


def simple_decode(ids: list) -> str:
    """Inverse of simple_encode."""
    return "".join(chr(min(max(i, 0), 127)) for i in ids)


def _sample_docs(n: int = 8) -> list:
    topics = [
        "machine learning algorithms",
        "deep neural networks training",
        "natural language processing models",
        "transformer architecture attention",
        "gradient descent optimization",
        "convolutional neural networks vision",
        "reinforcement learning agents",
        "generative adversarial networks",
    ]
    return [
        Document(doc_id=str(i), text=topics[i % len(topics)])
        for i in range(n)
    ]


@pytest.fixture
def docs():
    return _sample_docs()


@pytest.fixture
def retriever(docs):
    return MockRetriever(docs)


@pytest.fixture
def pipeline(small_model, retriever):
    cfg = RAGFusionConfig(n_queries=2, top_k_per_query=3, final_top_k=5)
    return RAGFusionPipeline(small_model, retriever, cfg, simple_encode, simple_decode)


# ---------------------------------------------------------------------------
# 1. RAGFusionConfig defaults
# ---------------------------------------------------------------------------

def test_rag_fusion_config_defaults():
    cfg = RAGFusionConfig()
    assert cfg.n_queries == 4
    assert cfg.top_k_per_query == 5
    assert cfg.final_top_k == 10
    assert cfg.rrf_k == 60
    assert cfg.use_hyde is False


# ---------------------------------------------------------------------------
# 2. Document fields
# ---------------------------------------------------------------------------

def test_document_fields():
    doc = Document(doc_id="abc", text="hello world", score=0.9, source="wiki")
    assert doc.doc_id == "abc"
    assert doc.text == "hello world"
    assert doc.score == 0.9
    assert doc.source == "wiki"


def test_document_defaults():
    doc = Document(doc_id="x", text="test")
    assert doc.score == 0.0
    assert doc.source == ""


# ---------------------------------------------------------------------------
# 3. compute_query_doc_similarity — identical strings → 1.0
# ---------------------------------------------------------------------------

def test_similarity_identical():
    s = "machine learning algorithms"
    assert compute_query_doc_similarity(s, s) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. compute_query_doc_similarity — unrelated strings → 0.0
# ---------------------------------------------------------------------------

def test_similarity_unrelated():
    # Strings chosen so they share no character trigrams
    result = compute_query_doc_similarity("aaa", "zzz")
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. compute_query_doc_similarity — partial overlap → (0, 1)
# ---------------------------------------------------------------------------

def test_similarity_partial_overlap():
    q = "machine learning"
    d = "machine vision systems"
    sim = compute_query_doc_similarity(q, d)
    assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# 6. reciprocal_rank_fusion — returns correct count
# ---------------------------------------------------------------------------

def test_rrf_returns_correct_count():
    list1 = [Document(doc_id=str(i), text=f"doc {i}") for i in range(5)]
    list2 = [Document(doc_id=str(i), text=f"doc {i}") for i in range(3, 8)]
    fused = reciprocal_rank_fusion([list1, list2])
    # 8 unique doc_ids (0-7)
    assert len(fused) == 8


# ---------------------------------------------------------------------------
# 7. reciprocal_rank_fusion — deduplicates documents
# ---------------------------------------------------------------------------

def test_rrf_deduplicates():
    doc = Document(doc_id="shared", text="shared doc")
    list1 = [doc, Document(doc_id="a", text="doc a")]
    list2 = [doc, Document(doc_id="b", text="doc b")]
    fused = reciprocal_rank_fusion([list1, list2])
    ids = [d.doc_id for d in fused]
    assert ids.count("shared") == 1


# ---------------------------------------------------------------------------
# 8. reciprocal_rank_fusion — doc in more lists → higher score
# ---------------------------------------------------------------------------

def test_rrf_more_lists_higher_score():
    shared = Document(doc_id="shared", text="shared")
    unique = Document(doc_id="unique", text="unique")

    list1 = [shared, unique]
    list2 = [shared]
    list3 = [shared]

    fused = reciprocal_rank_fusion([list1, list2, list3])
    scores = {d.doc_id: d.score for d in fused}
    assert scores["shared"] > scores["unique"]


# ---------------------------------------------------------------------------
# 9. MockRetriever.retrieve — returns top_k docs
# ---------------------------------------------------------------------------

def test_mock_retriever_top_k(docs):
    r = MockRetriever(docs)
    results = r.retrieve("neural networks", top_k=3)
    assert len(results) == 3


def test_mock_retriever_top_k_capped(docs):
    r = MockRetriever(docs)
    results = r.retrieve("anything", top_k=100)
    assert len(results) == len(docs)


# ---------------------------------------------------------------------------
# 10. MockRetriever.retrieve — scores documents
# ---------------------------------------------------------------------------

def test_mock_retriever_scores_set(docs):
    r = MockRetriever(docs)
    results = r.retrieve("machine learning", top_k=5)
    for doc in results:
        assert isinstance(doc.score, float)
        assert 0.0 <= doc.score <= 1.0


def test_mock_retriever_scores_sorted(docs):
    r = MockRetriever(docs)
    results = r.retrieve("deep learning neural", top_k=5)
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


# ---------------------------------------------------------------------------
# 11. RAGFusionPipeline.retrieve_and_fuse — returns list of Document
# ---------------------------------------------------------------------------

def test_pipeline_retrieve_and_fuse_returns_documents(pipeline):
    results = pipeline.retrieve_and_fuse("machine learning")
    assert isinstance(results, list)
    assert len(results) > 0
    for doc in results:
        assert isinstance(doc, Document)


def test_pipeline_retrieve_and_fuse_respects_final_top_k(small_model, docs):
    cfg = RAGFusionConfig(n_queries=2, top_k_per_query=3, final_top_k=2)
    r = MockRetriever(docs)
    p = RAGFusionPipeline(small_model, r, cfg, simple_encode, simple_decode)
    results = p.retrieve_and_fuse("neural networks")
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# 12. RAGFusionPipeline.build_context — contains doc text
# ---------------------------------------------------------------------------

def test_pipeline_build_context_contains_text():
    docs_list = [
        Document(doc_id="1", text="first document"),
        Document(doc_id="2", text="second document"),
    ]
    cfg = RAGFusionConfig()
    # retriever and model not needed for build_context
    r = MockRetriever([])
    p = RAGFusionPipeline(None, r, cfg, simple_encode, simple_decode)
    ctx = p.build_context(docs_list)
    assert "first document" in ctx
    assert "second document" in ctx


def test_pipeline_build_context_format():
    docs_list = [Document(doc_id="1", text="alpha"), Document(doc_id="2", text="beta")]
    cfg = RAGFusionConfig()
    r = MockRetriever([])
    p = RAGFusionPipeline(None, r, cfg, simple_encode, simple_decode)
    ctx = p.build_context(docs_list)
    assert "Document 1:" in ctx
    assert "Document 2:" in ctx


# ---------------------------------------------------------------------------
# 13. RAGFusionPipeline.generate_answer — returns string
# ---------------------------------------------------------------------------

def test_pipeline_generate_answer_returns_string(pipeline):
    answer = pipeline.generate_answer("What is ML?", "Machine learning is a field of AI.")
    assert isinstance(answer, str)


def test_pipeline_generate_answer_nonempty(pipeline):
    answer = pipeline.generate_answer("What is AI?", "AI stands for artificial intelligence.", max_new_tokens=8)
    assert len(answer) > 0


# ---------------------------------------------------------------------------
# 14. RAGFusionPipeline.run — returns required keys
# ---------------------------------------------------------------------------

def test_pipeline_run_returns_required_keys(pipeline):
    result = pipeline.run("neural network training")
    assert "query" in result
    assert "answer" in result
    assert "docs_retrieved" in result
    assert "queries_used" in result


def test_pipeline_run_query_preserved(pipeline):
    query = "transformer architecture"
    result = pipeline.run(query)
    assert result["query"] == query


def test_pipeline_run_docs_retrieved_is_int(pipeline):
    result = pipeline.run("deep learning")
    assert isinstance(result["docs_retrieved"], int)
    assert result["docs_retrieved"] >= 0


def test_pipeline_run_queries_used_is_list(pipeline):
    result = pipeline.run("gradient descent")
    assert isinstance(result["queries_used"], list)
    assert len(result["queries_used"]) > 0


# ---------------------------------------------------------------------------
# 15. generate_query_variations — returns list of strings
# ---------------------------------------------------------------------------

def test_generate_query_variations_returns_list(small_model):
    variations = generate_query_variations(
        small_model,
        "deep learning",
        n_queries=3,
        tokenizer_encode=simple_encode,
        tokenizer_decode=simple_decode,
        max_new_tokens=16,
    )
    assert isinstance(variations, list)
    assert len(variations) == 3


def test_generate_query_variations_all_strings(small_model):
    variations = generate_query_variations(
        small_model,
        "neural networks",
        n_queries=4,
        tokenizer_encode=simple_encode,
        tokenizer_decode=simple_decode,
        max_new_tokens=16,
    )
    for v in variations:
        assert isinstance(v, str)


def test_generate_query_variations_exact_count(small_model):
    for n in [1, 2, 4]:
        variations = generate_query_variations(
            small_model,
            "machine learning algorithms",
            n_queries=n,
            tokenizer_encode=simple_encode,
            tokenizer_decode=simple_decode,
            max_new_tokens=8,
        )
        assert len(variations) == n
