"""Tests for RLAIF (Reinforcement Learning from AI Feedback) module."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.alignment.rlaif import (
    ConstitutionalPrinciple,
    RLAIFCritique,
    RLAIFReviser,
    RLAIFPipeline,
    compute_improvement_score,
    PRINCIPLES,
)


def _make_tiny_model() -> nn.Module:
    """Build a tiny AureliusTransformer (vocab=64) for fast tests."""
    from src.model.config import AureliusConfig
    from src.model.transformer import AureliusTransformer

    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=32,
        n_heads=2,
        n_kv_heads=2,
        head_dim=16,
        d_ff=64,
        vocab_size=64,
        max_seq_len=128,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model() -> nn.Module:
    return _make_tiny_model()


@pytest.fixture(scope="module")
def tiny_optimizer(tiny_model) -> torch.optim.Optimizer:
    return torch.optim.SGD(tiny_model.parameters(), lr=1e-4)


@pytest.fixture
def prompt_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 64, (1, 4))


@pytest.fixture
def principle() -> ConstitutionalPrinciple:
    return ConstitutionalPrinciple(PRINCIPLES[0])


def test_constitutional_principle_defaults():
    """ConstitutionalPrinciple should auto-generate non-empty critique and revision prompts."""
    p = ConstitutionalPrinciple("Be helpful.")
    assert isinstance(p.critique_prompt, str) and len(p.critique_prompt) > 0
    assert isinstance(p.revision_prompt, str) and len(p.revision_prompt) > 0


def test_rlaif_critique_returns_ids(tiny_model, prompt_ids, principle):
    """RLAIFCritique.critique() should return a tensor."""
    critiquer = RLAIFCritique(tiny_model, max_critique_tokens=8)
    response_ids = torch.randint(0, 64, (1, 6))
    critique_ids = critiquer.critique(prompt_ids, response_ids, principle)
    assert isinstance(critique_ids, torch.Tensor)


def test_score_response_range(tiny_model, prompt_ids, principle):
    """RLAIFCritique.score_response() should return a float in [0, 1]."""
    critiquer = RLAIFCritique(tiny_model, max_critique_tokens=8)
    response_ids = torch.randint(0, 64, (1, 6))
    score = critiquer.score_response(prompt_ids, response_ids, principle)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_reviser_returns_ids(tiny_model, prompt_ids, principle):
    """RLAIFReviser.revise() should return a tensor."""
    reviser = RLAIFReviser(tiny_model, max_revision_tokens=8)
    response_ids = torch.randint(0, 64, (1, 6))
    critique_ids = torch.randint(0, 64, (1, 4))
    revised_ids = reviser.revise(prompt_ids, response_ids, critique_ids, principle)
    assert isinstance(revised_ids, torch.Tensor)


def test_critique_revision_loop_keys(tiny_model, tiny_optimizer, prompt_ids):
    """run_critique_revision_loop() should return dict with required keys."""
    principles = [ConstitutionalPrinciple(PRINCIPLES[0])]
    pipeline = RLAIFPipeline(
        tiny_model,
        tiny_optimizer,
        principles=principles,
        max_response_tokens=8,
    )
    pipeline.critiquer.max_critique_tokens = 8
    pipeline.reviser.max_revision_tokens = 8
    result = pipeline.run_critique_revision_loop(prompt_ids)
    assert "initial_ids" in result
    assert "revised_ids" in result
    assert "scores" in result


def test_scores_length(tiny_model, tiny_optimizer, prompt_ids):
    """scores list should have length equal to number of principles."""
    n_principles = 2
    principles = [ConstitutionalPrinciple(PRINCIPLES[i]) for i in range(n_principles)]
    pipeline = RLAIFPipeline(
        tiny_model,
        tiny_optimizer,
        principles=principles,
        max_response_tokens=8,
    )
    pipeline.critiquer.max_critique_tokens = 8
    pipeline.reviser.max_revision_tokens = 8
    result = pipeline.run_critique_revision_loop(prompt_ids)
    assert len(result["scores"]) == n_principles


def test_sft_step_returns_loss(tiny_model, tiny_optimizer, prompt_ids):
    """sft_step() should return a float."""
    pipeline = RLAIFPipeline(tiny_model, tiny_optimizer, max_response_tokens=8)
    revised_ids = torch.randint(0, 64, (1, 6))
    loss = pipeline.sft_step(prompt_ids, revised_ids)
    assert isinstance(loss, float)


def test_run_returns_loss(tiny_model, tiny_optimizer, prompt_ids):
    """run() should return a float training loss."""
    principles = [ConstitutionalPrinciple(PRINCIPLES[0])]
    pipeline = RLAIFPipeline(
        tiny_model,
        tiny_optimizer,
        principles=principles,
        max_response_tokens=8,
    )
    pipeline.critiquer.max_critique_tokens = 8
    pipeline.reviser.max_revision_tokens = 8
    loss = pipeline.run(prompt_ids)
    assert isinstance(loss, float)


def test_improvement_score_positive():
    """compute_improvement_score should be positive when revised_score > initial_score."""
    improvement = compute_improvement_score(initial_score=0.3, revised_score=0.7)
    assert improvement > 0.0


def test_improvement_score_negative():
    """compute_improvement_score should be negative when revised_score < initial_score."""
    improvement = compute_improvement_score(initial_score=0.8, revised_score=0.2)
    assert improvement < 0.0
