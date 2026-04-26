"""Tests for src/inference/speculative_rag.py

Covers DocumentRetriever, ContextBuilder, SpeculativeRAGDecoder, and
the full generate loop — 12 tests total.

Tiny config: vocab_size=256, n_draft_tokens=4, top_k_docs=2, batch=1.
"""

from __future__ import annotations

import torch

from src.inference.speculative_rag import (
    ContextBuilder,
    Document,
    DocumentRetriever,
    MockDraftModel,
    MockTargetModel,
    SpeculativeRAGConfig,
    SpeculativeRAGDecoder,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
N_DRAFT = 4
TOP_K = 2


def _make_corpus() -> list[Document]:
    return [
        Document(text="the cat sat on the mat", doc_id="d0"),
        Document(text="dogs love to run and play fetch", doc_id="d1"),
        Document(text="the sun rises in the east every morning", doc_id="d2"),
        Document(text="python is a popular programming language", doc_id="d3"),
        Document(text="neural networks learn from data", doc_id="d4"),
    ]


def _make_decoder(rerank: bool = False) -> SpeculativeRAGDecoder:
    torch.manual_seed(42)
    draft = MockDraftModel(vocab_size=VOCAB_SIZE, hidden=16)
    target = MockTargetModel(vocab_size=VOCAB_SIZE, hidden=32)
    retriever = DocumentRetriever(_make_corpus())
    cfg = SpeculativeRAGConfig(
        n_draft_tokens=N_DRAFT,
        top_k_docs=TOP_K,
        draft_temperature=1.0,
        verify_temperature=1.0,
        rerank=rerank,
    )
    return SpeculativeRAGDecoder(draft, target, retriever, cfg)


def _query_ids(length: int = 6) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (length,))


# ---------------------------------------------------------------------------
# Test 1 — DocumentRetriever.retrieve returns exactly top_k docs
# ---------------------------------------------------------------------------


def test_retriever_returns_top_k():
    retriever = DocumentRetriever(_make_corpus())
    results = retriever.retrieve("the cat", top_k=TOP_K)
    assert len(results) == TOP_K, f"Expected {TOP_K} docs, got {len(results)}"


# ---------------------------------------------------------------------------
# Test 2 — Retrieved docs are sorted by score descending
# ---------------------------------------------------------------------------


def test_retriever_sorted_descending():
    retriever = DocumentRetriever(_make_corpus())
    results = retriever.retrieve("the cat sat", top_k=3)
    scores = [d.score for d in results]
    assert scores == sorted(scores, reverse=True), f"Docs not sorted by score descending: {scores}"


# ---------------------------------------------------------------------------
# Test 3 — add_documents increases corpus size
# ---------------------------------------------------------------------------


def test_add_documents_increases_corpus():
    retriever = DocumentRetriever(_make_corpus())
    original_len = len(retriever)
    retriever.add_documents(
        [
            Document(text="extra document about cats", doc_id="e0"),
            Document(text="another extra document", doc_id="e1"),
        ]
    )
    assert len(retriever) == original_len + 2, (
        f"Expected corpus size {original_len + 2}, got {len(retriever)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — ContextBuilder.build output length <= max_len
# ---------------------------------------------------------------------------


def test_context_builder_respects_max_len():
    builder = ContextBuilder()
    max_len = 20
    query_ids = torch.randint(0, VOCAB_SIZE, (8,))
    doc_ids = [torch.randint(0, VOCAB_SIZE, (10,)) for _ in range(3)]
    ctx = builder.build(query_ids, doc_ids, max_len=max_len)
    assert ctx.shape[0] <= max_len, f"Context length {ctx.shape[0]} exceeds max_len {max_len}"


# ---------------------------------------------------------------------------
# Test 5 — ContextBuilder truncates when docs exceed max_len
# ---------------------------------------------------------------------------


def test_context_builder_truncates():
    builder = ContextBuilder()
    max_len = 10
    query_ids = torch.arange(6, dtype=torch.long)  # 6 tokens
    doc_ids = [torch.arange(20, dtype=torch.long)]  # 20 tokens -> total 26 > 10
    ctx = builder.build(query_ids, doc_ids, max_len=max_len)
    assert ctx.shape[0] == max_len, f"Expected truncated length {max_len}, got {ctx.shape[0]}"


# ---------------------------------------------------------------------------
# Test 6 — draft_with_context returns (ids, logits) of correct shapes
# ---------------------------------------------------------------------------


def test_draft_with_context_shapes():
    decoder = _make_decoder()
    query_ids = _query_ids(6)
    context_ids = torch.randint(0, VOCAB_SIZE, (20,))
    draft_ids, draft_logits = decoder.draft_with_context(query_ids, context_ids)

    assert draft_ids.shape == (1, N_DRAFT), f"draft_ids shape {draft_ids.shape} != (1, {N_DRAFT})"
    assert draft_logits.shape == (1, N_DRAFT, VOCAB_SIZE), (
        f"draft_logits shape {draft_logits.shape} != (1, {N_DRAFT}, {VOCAB_SIZE})"
    )


# ---------------------------------------------------------------------------
# Test 7 — verify_with_context returns logits of correct shape
# ---------------------------------------------------------------------------


def test_verify_with_context_shape():
    decoder = _make_decoder()
    T_c = 20
    context_ids = torch.randint(0, VOCAB_SIZE, (T_c,))
    draft_ids = torch.randint(0, VOCAB_SIZE, (1, N_DRAFT))
    target_logits = decoder.verify_with_context(context_ids, draft_ids)

    expected_T = T_c + N_DRAFT
    assert target_logits.shape == (1, expected_T, VOCAB_SIZE), (
        f"target_logits shape {target_logits.shape} != (1, {expected_T}, {VOCAB_SIZE})"
    )


# ---------------------------------------------------------------------------
# Test 8 — generate returns an output_ids tensor
# ---------------------------------------------------------------------------


def test_generate_returns_tensor():
    decoder = _make_decoder()
    query_ids = _query_ids(6)
    output_ids, stats = decoder.generate(query_ids, max_new_tokens=8)
    assert isinstance(output_ids, torch.Tensor), (
        f"output_ids should be a Tensor, got {type(output_ids)}"
    )
    assert output_ids.dtype == torch.long, (
        f"output_ids dtype should be long, got {output_ids.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 9 — stats dict has 'n_accepted' and 'n_rounds' keys
# ---------------------------------------------------------------------------


def test_generate_stats_keys():
    decoder = _make_decoder()
    query_ids = _query_ids(6)
    _, stats = decoder.generate(query_ids, max_new_tokens=8)
    assert "n_accepted" in stats, "'n_accepted' missing from stats"
    assert "n_rounds" in stats, "'n_rounds' missing from stats"


# ---------------------------------------------------------------------------
# Test 10 — n_accepted is between 0 and n_draft_tokens * n_rounds
# ---------------------------------------------------------------------------


def test_n_accepted_in_valid_range():
    decoder = _make_decoder()
    query_ids = _query_ids(6)
    _, stats = decoder.generate(query_ids, max_new_tokens=8)
    n_accepted = stats["n_accepted"]
    n_rounds = stats["n_rounds"]
    max_possible = N_DRAFT * n_rounds
    assert 0 <= n_accepted <= max_possible, f"n_accepted={n_accepted} not in [0, {max_possible}]"


# ---------------------------------------------------------------------------
# Test 11 — Different queries retrieve different docs (when corpus varies)
# ---------------------------------------------------------------------------


def test_different_queries_retrieve_different_docs():
    corpus = [
        Document(text="cats and kittens", doc_id="cat0"),
        Document(text="cats love fish", doc_id="cat1"),
        Document(text="machine learning and deep neural nets", doc_id="ml0"),
        Document(text="gradient descent optimization", doc_id="ml1"),
    ]
    retriever = DocumentRetriever(corpus)

    results_a = retriever.retrieve("cats kittens fish", top_k=2)
    results_b = retriever.retrieve("machine learning gradient", top_k=2)

    ids_a = {d.doc_id for d in results_a}
    ids_b = {d.doc_id for d in results_b}

    # The two queries should not retrieve identical sets
    assert ids_a != ids_b, f"Expected different doc sets; both returned {ids_a}"


# ---------------------------------------------------------------------------
# Test 12 — Empty corpus returns empty list from retrieve
# ---------------------------------------------------------------------------


def test_empty_corpus_returns_empty():
    retriever = DocumentRetriever([])
    results = retriever.retrieve("any query at all", top_k=3)
    assert results == [], f"Expected empty list from empty corpus, got {results}"
