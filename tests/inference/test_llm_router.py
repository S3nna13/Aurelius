"""Tests for src/inference/llm_router.py — LLM Router / Model Cascade."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference.llm_router import (
    RouteDecision,
    RouterConfig,
    LinearRouter,
    QueryComplexityEstimator,
    CascadeRouter,
    RouterTrainer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCAB = 64
D_MODEL = 16
B = 2
T = 8


# ---------------------------------------------------------------------------
# Mock model: returns (None, logits, None) with shape (B, T, vocab_size)
# ---------------------------------------------------------------------------

class MockModel(nn.Module):
    """Minimal mock LM: returns fixed random logits for testing."""

    def __init__(self, vocab_size: int = VOCAB, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.proj = nn.Linear(1, vocab_size)   # just needs params
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        labels=None,
    ):
        B, T = input_ids.shape
        # Generate deterministic logits from input ids
        # Use a simple linear map: embed token ids and project
        x = input_ids.float().unsqueeze(-1)          # (B, T, 1)
        logits = self.proj(x)                         # (B, T, vocab_size)
        return None, logits, None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router():
    torch.manual_seed(0)
    return LinearRouter(vocab_size=VOCAB, d_model=D_MODEL, n_models=2)


@pytest.fixture
def input_ids():
    torch.manual_seed(7)
    return torch.randint(0, VOCAB, (B, T))


@pytest.fixture
def two_models():
    return {
        "small": MockModel(vocab_size=VOCAB, seed=1),
        "large": MockModel(vocab_size=VOCAB, seed=2),
    }


@pytest.fixture
def cascade_router(two_models):
    cfg = RouterConfig(n_models=2, model_names=["small", "large"], confidence_threshold=0.7)
    return CascadeRouter(models=two_models, config=cfg)


# ---------------------------------------------------------------------------
# 1. LinearRouter instantiates with n_models=2
# ---------------------------------------------------------------------------

def test_linear_router_instantiates(router):
    assert isinstance(router, LinearRouter)
    assert router.n_models == 2


# ---------------------------------------------------------------------------
# 2. LinearRouter forward returns shape (B, n_models)
# ---------------------------------------------------------------------------

def test_linear_router_forward_shape(router, input_ids):
    logits = router(input_ids)
    assert logits.shape == (B, 2)


# ---------------------------------------------------------------------------
# 3. LinearRouter.route returns model_indices shape (B,) and confidences (B,)
# ---------------------------------------------------------------------------

def test_linear_router_route_shapes(router, input_ids):
    indices, confs = router.route(input_ids)
    assert indices.shape == (B,)
    assert confs.shape == (B,)


# ---------------------------------------------------------------------------
# 4. LinearRouter confidences in [0, 1]
# ---------------------------------------------------------------------------

def test_linear_router_confidences_range(router, input_ids):
    _, confs = router.route(input_ids)
    assert (confs >= 0.0).all(), "All confidences should be >= 0"
    assert (confs <= 1.0).all(), "All confidences should be <= 1"


# ---------------------------------------------------------------------------
# 5. QueryComplexityEstimator.extract_features returns dict with feature keys
# ---------------------------------------------------------------------------

def test_complexity_estimator_feature_keys():
    est = QueryComplexityEstimator()
    feats = est.extract_features([1, 2, 3, 4, 5])
    expected_keys = {"token_count", "unique_token_ratio", "question_mark_count",
                     "token_id_variance", "clause_depth"}
    assert expected_keys.issubset(feats.keys())


# ---------------------------------------------------------------------------
# 6. QueryComplexityEstimator.estimate_complexity returns float in [0, 1]
# ---------------------------------------------------------------------------

def test_complexity_estimator_score_range():
    est = QueryComplexityEstimator()
    score = est.estimate_complexity([1, 2, 3, 4, 5, 6, 7, 8])
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 7. Long sequence > short sequence (usually higher complexity)
# ---------------------------------------------------------------------------

def test_complexity_long_vs_short():
    est = QueryComplexityEstimator()
    short_ids = list(range(4))
    long_ids = list(range(100))
    score_short = est.estimate_complexity(short_ids)
    score_long = est.estimate_complexity(long_ids)
    # Long sequence should have higher complexity due to length_score
    assert score_long > score_short, (
        f"Expected long ({score_long:.4f}) > short ({score_short:.4f})"
    )


# ---------------------------------------------------------------------------
# 8. QueryComplexityEstimator.should_use_large_model returns bool
# ---------------------------------------------------------------------------

def test_complexity_should_use_large_model():
    est = QueryComplexityEstimator(complexity_threshold=0.5)
    result = est.should_use_large_model([1, 2, 3, 4, 5])
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 9. CascadeRouter instantiates with 2 models
# ---------------------------------------------------------------------------

def test_cascade_router_instantiates(cascade_router):
    assert isinstance(cascade_router, CascadeRouter)
    assert len(cascade_router.models) == 2


# ---------------------------------------------------------------------------
# 10. CascadeRouter.route_only returns list of RouteDecision objects
# ---------------------------------------------------------------------------

def test_cascade_router_route_only_returns_decisions(cascade_router, input_ids):
    decisions = cascade_router.route_only(input_ids)
    assert isinstance(decisions, list)
    assert len(decisions) == B
    for d in decisions:
        assert isinstance(d, RouteDecision)


# ---------------------------------------------------------------------------
# 11. RouteDecision.model_id is one of the registered model names
# ---------------------------------------------------------------------------

def test_cascade_router_decision_model_id(cascade_router, input_ids):
    decisions = cascade_router.route_only(input_ids)
    valid_names = set(cascade_router.models.keys())
    for d in decisions:
        assert d.model_id in valid_names, f"'{d.model_id}' not in {valid_names}"


# ---------------------------------------------------------------------------
# 12. RouteDecision.confidence in [0, 1]
# ---------------------------------------------------------------------------

def test_cascade_router_decision_confidence_range(cascade_router, input_ids):
    decisions = cascade_router.route_only(input_ids)
    for d in decisions:
        assert 0.0 <= d.confidence <= 1.0, f"confidence {d.confidence} out of range"


# ---------------------------------------------------------------------------
# 13. CascadeRouter.decode returns output_ids of correct shape
# ---------------------------------------------------------------------------

def test_cascade_router_decode_output_shape(cascade_router, input_ids):
    max_new = 5
    output_ids, decisions = cascade_router.decode(input_ids, max_new_tokens=max_new)
    # output should be (B, T + max_new_tokens)
    assert output_ids.shape == (B, T + max_new)
    assert len(decisions) == B


# ---------------------------------------------------------------------------
# 14. RouterTrainer.train_step returns float loss
# ---------------------------------------------------------------------------

def test_router_trainer_train_step(input_ids):
    torch.manual_seed(42)
    router = LinearRouter(vocab_size=VOCAB, d_model=D_MODEL, n_models=2)
    trainer = RouterTrainer(router, lr=1e-3)
    labels = torch.randint(0, 2, (B,))
    loss = trainer.train_step(input_ids, labels)
    assert isinstance(loss, float)
    assert loss >= 0.0


# ---------------------------------------------------------------------------
# 15. RouterTrainer.evaluate returns accuracy in [0, 1]
# ---------------------------------------------------------------------------

def test_router_trainer_evaluate(input_ids):
    torch.manual_seed(42)
    router = LinearRouter(vocab_size=VOCAB, d_model=D_MODEL, n_models=2)
    trainer = RouterTrainer(router, lr=1e-3)
    labels = torch.randint(0, 2, (B,))
    acc = trainer.evaluate(input_ids, labels)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# 16. RouterTrainer 5 steps reduces training loss
# ---------------------------------------------------------------------------

def test_router_trainer_loss_decreases():
    """Training on a fixed batch for 5 steps should reduce the loss."""
    torch.manual_seed(0)
    router = LinearRouter(vocab_size=VOCAB, d_model=D_MODEL, n_models=2)
    trainer = RouterTrainer(router, lr=1e-2)

    # Fixed batch: all label 0 — learnable signal
    ids = torch.randint(0, VOCAB, (B, T))
    labels = torch.zeros(B, dtype=torch.long)

    first_loss = trainer.train_step(ids, labels)
    for _ in range(4):
        last_loss = trainer.train_step(ids, labels)

    assert last_loss < first_loss, (
        f"Expected loss to decrease: {first_loss:.4f} -> {last_loss:.4f}"
    )
