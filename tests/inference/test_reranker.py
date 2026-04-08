import pytest
import torch
from src.inference.reranker import (
    RerankConfig,
    CrossEncoderScorer,
    DocumentReranker,
    listwise_rerank_loss,
    pairwise_rerank_loss,
    RerankTrainer,
)

D_MODEL = 32
VOCAB_SIZE = 256
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scorer() -> CrossEncoderScorer:
    torch.manual_seed(0)
    return CrossEncoderScorer(d_model=D_MODEL, vocab_size=VOCAB_SIZE)


def _make_query() -> torch.Tensor:
    """Returns (1, 8) query token ids."""
    return torch.randint(0, VOCAB_SIZE, (1, 8))


def _make_doc() -> torch.Tensor:
    """Returns (1, 16) document token ids."""
    return torch.randint(0, VOCAB_SIZE, (1, 16))


def _make_docs(n: int = 5) -> list[torch.Tensor]:
    return [torch.randint(0, VOCAB_SIZE, (1, 16)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rerank_config_defaults():
    cfg = RerankConfig()
    assert cfg.max_length == 256
    assert cfg.batch_size == 8
    assert cfg.score_aggregation == "max"


def test_cross_encoder_scorer_output_shape():
    scorer = _make_scorer()
    query = _make_query()
    doc = _make_doc()
    out = scorer(query, doc)
    assert out.shape == (1, 1)


def test_cross_encoder_scorer_differentiable():
    scorer = _make_scorer()
    query = _make_query()
    doc = _make_doc()
    out = scorer(query, doc)
    loss = out.sum()
    loss.backward()  # should not raise


def test_document_reranker_score_count():
    scorer = _make_scorer()
    config = RerankConfig()
    reranker = DocumentReranker(scorer, config)
    query = _make_query()
    docs = _make_docs(5)
    scores = reranker.score_documents(query, docs)
    assert len(scores) == 5


def test_document_reranker_score_shape():
    scorer = _make_scorer()
    config = RerankConfig()
    reranker = DocumentReranker(scorer, config)
    query = _make_query()
    docs = _make_docs(5)
    scores = reranker.score_documents(query, docs)
    assert scores.shape == (5,)


def test_document_reranker_rerank_sorted():
    scorer = _make_scorer()
    config = RerankConfig()
    reranker = DocumentReranker(scorer, config)
    query = _make_query()
    docs = _make_docs(5)
    _, sorted_scores = reranker.rerank(query, docs)
    # Scores should be in descending order
    for i in range(len(sorted_scores) - 1):
        assert sorted_scores[i].item() >= sorted_scores[i + 1].item()


def test_document_reranker_top_k():
    scorer = _make_scorer()
    config = RerankConfig()
    reranker = DocumentReranker(scorer, config)
    query = _make_query()
    docs = _make_docs(5)
    sorted_docs, sorted_scores = reranker.rerank(query, docs, top_k=2)
    assert len(sorted_docs) == 2
    assert sorted_scores.shape == (2,)


def test_rerank_texts_returns_tuples():
    scorer = _make_scorer()
    config = RerankConfig()
    reranker = DocumentReranker(scorer, config)

    def simple_tokenize(text: str) -> list[int]:
        return [ord(c) % VOCAB_SIZE for c in text[:16]] or [0]

    results = reranker.rerank_texts(
        query="hello world",
        documents=["first document", "second document", "third document"],
        tokenize_fn=simple_tokenize,
    )
    assert len(results) == 3
    for item in results:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], float)


def test_listwise_rerank_loss_shape():
    scores = torch.tensor([1.0, 0.5, 0.2])
    labels = torch.tensor([1.0, 0.0, 0.0])
    loss = listwise_rerank_loss(scores, labels)
    assert loss.shape == ()  # scalar


def test_pairwise_rerank_loss_positive_pos_neg():
    """When pos_score >> neg_score, pairwise loss should be 0."""
    pos_scores = torch.tensor([10.0])
    neg_scores = torch.tensor([-10.0])
    loss = pairwise_rerank_loss(pos_scores, neg_scores)
    assert loss.item() == pytest.approx(0.0)


def test_rerank_trainer_step_keys():
    torch.manual_seed(0)
    scorer = _make_scorer()
    optimizer = torch.optim.Adam(scorer.parameters(), lr=1e-3)
    config = RerankConfig()
    trainer = RerankTrainer(scorer, optimizer, config)

    query = _make_query()
    pos_doc = _make_doc()
    neg_doc = _make_doc()

    result = trainer.train_step(query, pos_doc, neg_doc)
    assert "loss" in result
    assert "pos_score" in result
    assert "neg_score" in result
    assert isinstance(result["loss"], float)
    assert isinstance(result["pos_score"], float)
    assert isinstance(result["neg_score"], float)
