"""Tests for src/alignment/constitutional_ai.py — RLAIF self-critique loop."""
from __future__ import annotations

import copy

import pytest
import torch

from src.alignment.constitutional_ai import (
    ConstitutionalAIConfig,
    Principle,
    default_principles,
    score_principle_compliance,
    compute_critique_loss,
    SelfCritiqueBuffer,
    ConstitutionalAITrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Tiny model factory
# ---------------------------------------------------------------------------

def make_tiny_model() -> AureliusTransformer:
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


def make_trainer() -> ConstitutionalAITrainer:
    policy = make_tiny_model()
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    config = ConstitutionalAIConfig()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    return ConstitutionalAITrainer(policy=policy, ref_model=ref, config=config, optimizer=optimizer)


# ---------------------------------------------------------------------------
# 1. ConstitutionalAIConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ConstitutionalAIConfig()
    assert cfg.n_principles == 4
    assert cfg.n_critique_rounds == 2
    assert cfg.sft_loss_coeff == 1.0
    assert cfg.kl_coeff == 0.1
    assert cfg.max_seq_len == 128


# ---------------------------------------------------------------------------
# 2. default_principles returns list of length 4
# ---------------------------------------------------------------------------

def test_default_principles_length():
    principles = default_principles()
    assert isinstance(principles, list)
    assert len(principles) == 4


# ---------------------------------------------------------------------------
# 3. Principle dataclass fields
# ---------------------------------------------------------------------------

def test_principle_fields():
    p = Principle(
        name="test",
        description="Test description",
        critique_prompt="Is it good?",
        revision_prompt="Make it better.",
    )
    assert p.name == "test"
    assert p.description == "Test description"
    assert p.critique_prompt == "Is it good?"
    assert p.revision_prompt == "Make it better."


# ---------------------------------------------------------------------------
# 4. score_principle_compliance returns scalar tensor
# ---------------------------------------------------------------------------

def test_score_principle_compliance_scalar():
    B, T, V = 2, 10, 256
    logits = torch.randn(B, T, V)
    token_ids = torch.randint(0, V, (B, T))
    score = score_principle_compliance(logits, token_ids, 0, 4)
    assert isinstance(score, torch.Tensor)
    assert score.numel() == 1


# ---------------------------------------------------------------------------
# 5. score_principle_compliance output is finite
# ---------------------------------------------------------------------------

def test_score_principle_compliance_finite():
    B, T, V = 2, 10, 256
    logits = torch.randn(B, T, V)
    token_ids = torch.randint(0, V, (B, T))
    for i in range(4):
        score = score_principle_compliance(logits, token_ids, i, 4)
        assert torch.isfinite(score), f"Score for principle {i} is not finite"


# ---------------------------------------------------------------------------
# 6. compute_critique_loss returns (Tensor, dict)
# ---------------------------------------------------------------------------

def test_compute_critique_loss_return_types():
    B, T, V = 2, 10, 256
    policy_logits = torch.randn(B, T, V, requires_grad=True)
    ref_logits = torch.randn(B, T, V)
    target_ids = torch.randint(0, V, (B, T))
    principle_scores = torch.randn(B, 4)
    config = ConstitutionalAIConfig()

    result = compute_critique_loss(policy_logits, ref_logits, target_ids, principle_scores, config)
    assert isinstance(result, tuple)
    loss, metrics = result
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# 7. compute_critique_loss dict has required keys
# ---------------------------------------------------------------------------

def test_compute_critique_loss_dict_keys():
    B, T, V = 2, 10, 256
    policy_logits = torch.randn(B, T, V, requires_grad=True)
    ref_logits = torch.randn(B, T, V)
    target_ids = torch.randint(0, V, (B, T))
    principle_scores = torch.randn(B, 4)
    config = ConstitutionalAIConfig()

    _, metrics = compute_critique_loss(policy_logits, ref_logits, target_ids, principle_scores, config)
    for key in ("sft_loss", "kl_loss", "principle_reward", "total_loss"):
        assert key in metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 8. compute_critique_loss loss is scalar and finite
# ---------------------------------------------------------------------------

def test_compute_critique_loss_scalar_finite():
    B, T, V = 2, 10, 256
    policy_logits = torch.randn(B, T, V, requires_grad=True)
    ref_logits = torch.randn(B, T, V)
    target_ids = torch.randint(0, V, (B, T))
    principle_scores = torch.randn(B, 4)
    config = ConstitutionalAIConfig()

    loss, metrics = compute_critique_loss(policy_logits, ref_logits, target_ids, principle_scores, config)
    assert loss.numel() == 1
    assert torch.isfinite(loss)
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


# ---------------------------------------------------------------------------
# 9. compute_critique_loss kl_loss >= 0
# ---------------------------------------------------------------------------

def test_compute_critique_loss_kl_nonneg():
    B, T, V = 2, 10, 256
    policy_logits = torch.randn(B, T, V, requires_grad=True)
    ref_logits = torch.randn(B, T, V)
    target_ids = torch.randint(0, V, (B, T))
    principle_scores = torch.randn(B, 4)
    config = ConstitutionalAIConfig()

    _, metrics = compute_critique_loss(policy_logits, ref_logits, target_ids, principle_scores, config)
    assert metrics["kl_loss"] >= 0.0


# ---------------------------------------------------------------------------
# 10. SelfCritiqueBuffer starts empty
# ---------------------------------------------------------------------------

def test_buffer_starts_empty():
    buf = SelfCritiqueBuffer()
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# 11. SelfCritiqueBuffer.add increases length
# ---------------------------------------------------------------------------

def test_buffer_add_increases_length():
    buf = SelfCritiqueBuffer()
    prompt = torch.randint(0, 256, (1, 8))
    response = torch.randint(0, 256, (1, 8))
    revised = torch.randint(0, 256, (1, 8))
    scores = torch.randn(1, 4)

    buf.add(prompt, response, revised, scores)
    assert len(buf) == 1

    buf.add(prompt, response, revised, scores)
    assert len(buf) == 2


# ---------------------------------------------------------------------------
# 12. SelfCritiqueBuffer.sample returns None when buffer < batch_size
# ---------------------------------------------------------------------------

def test_buffer_sample_returns_none_when_small():
    buf = SelfCritiqueBuffer()
    prompt = torch.randint(0, 256, (1, 8))
    response = torch.randint(0, 256, (1, 8))
    revised = torch.randint(0, 256, (1, 8))
    scores = torch.randn(1, 4)

    buf.add(prompt, response, revised, scores)
    # buffer has 1 item, batch_size=4 should return None
    result = buf.sample(batch_size=4)
    assert result is None


# ---------------------------------------------------------------------------
# 13. SelfCritiqueBuffer.sample returns 4-tuple when buffer is full enough
# ---------------------------------------------------------------------------

def test_buffer_sample_returns_tuple():
    buf = SelfCritiqueBuffer()
    for _ in range(5):
        prompt = torch.randint(0, 256, (1, 8))
        response = torch.randint(0, 256, (1, 8))
        revised = torch.randint(0, 256, (1, 8))
        scores = torch.randn(1, 4)
        buf.add(prompt, response, revised, scores)

    result = buf.sample(batch_size=3)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 4
    for t in result:
        assert isinstance(t, torch.Tensor)


# ---------------------------------------------------------------------------
# 14. ConstitutionalAITrainer.critique_and_revise returns (Tensor, Tensor)
# ---------------------------------------------------------------------------

def test_critique_and_revise_return_types():
    trainer = make_trainer()
    prompt_ids = torch.randint(0, 256, (2, 10))
    response_ids = torch.randint(0, 256, (2, 10))

    revised_ids, scores = trainer.critique_and_revise(prompt_ids, response_ids)
    assert isinstance(revised_ids, torch.Tensor)
    assert isinstance(scores, torch.Tensor)
    assert revised_ids.shape == response_ids.shape
    assert scores.shape == (2, trainer.config.n_principles)


# ---------------------------------------------------------------------------
# 15. ConstitutionalAITrainer.train_step returns dict with correct keys
# ---------------------------------------------------------------------------

def test_train_step_returns_correct_keys():
    trainer = make_trainer()
    prompt_ids = torch.randint(0, 256, (2, 10))
    response_ids = torch.randint(0, 256, (2, 10))

    metrics = trainer.train_step(prompt_ids, response_ids)
    assert isinstance(metrics, dict)
    for key in ("sft_loss", "kl_loss", "principle_reward", "total_loss"):
        assert key in metrics, f"Missing key: {key}"
        assert isinstance(metrics[key], float)
