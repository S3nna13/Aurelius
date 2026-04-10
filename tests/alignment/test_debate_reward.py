"""Tests for debate-based reward modeling module."""

from __future__ import annotations

import math

import pytest
import torch

from src.alignment.debate_reward import (
    DebateConfig,
    ArgumentGenerator,
    JudgeModel,
    debate_reward,
    DebateRewardTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

DEBATE_CFG = DebateConfig(
    n_rounds=1,
    max_argument_tokens=3,
    judge_temperature=0.5,
    reward_scale=1.0,
)


def make_model() -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(CONFIG)


def make_question_ids(seq_len: int = 4) -> torch.Tensor:
    """Return (1, seq_len) context ids."""
    return torch.randint(1, 256, (1, seq_len))


def make_arg_gen(model=None) -> ArgumentGenerator:
    if model is None:
        model = make_model()
    return ArgumentGenerator(model, DEBATE_CFG)


def make_judge(model=None) -> JudgeModel:
    if model is None:
        model = make_model()
    return JudgeModel(model, DEBATE_CFG)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_debate_config_defaults():
    """DebateConfig has correct default values."""
    cfg = DebateConfig()
    assert cfg.n_rounds == 2
    assert cfg.max_argument_tokens == 16
    assert cfg.judge_temperature == pytest.approx(0.5)
    assert cfg.reward_scale == pytest.approx(1.0)


def test_argument_generator_shape():
    """generate_argument returns tensor of shape (1, max_argument_tokens)."""
    gen = make_arg_gen()
    q = make_question_ids()
    result = gen.generate_argument(q, "pro")
    assert result.shape == (1, DEBATE_CFG.max_argument_tokens)


def test_argument_generator_pro_vs_con_differ():
    """generate_argument produces different outputs for 'pro' vs 'con'."""
    torch.manual_seed(0)
    model = make_model()
    gen = ArgumentGenerator(model, DebateConfig(n_rounds=1, max_argument_tokens=3, judge_temperature=0.5))
    q = make_question_ids()
    # Run multiple trials; they should differ at least once (with temperature)
    any_diff = False
    for seed in range(5):
        torch.manual_seed(seed)
        pro = gen.generate_argument(q, "pro")
        con = gen.generate_argument(q, "con")
        if not torch.equal(pro, con):
            any_diff = True
            break
    assert any_diff, "pro and con arguments should differ (different position prefix)"


def test_judge_score_argument_returns_float():
    """score_argument returns a Python float."""
    judge = make_judge()
    arg_ids = torch.randint(0, 256, (1, 3))
    score = judge.score_argument(arg_ids)
    assert isinstance(score, float)


def test_judge_compare_sums_to_one():
    """compare() returns (a, b) where a + b ≈ 1.0."""
    judge = make_judge()
    ids_a = torch.randint(0, 256, (1, 3))
    ids_b = torch.randint(0, 256, (1, 3))
    score_a, score_b = judge.compare(ids_a, ids_b)
    assert score_a + score_b == pytest.approx(1.0, abs=1e-5)


def test_judge_compare_symmetric_same_input():
    """compare() with identical inputs returns (0.5, 0.5)."""
    judge = make_judge()
    ids = torch.randint(0, 256, (1, 3))
    score_a, score_b = judge.compare(ids, ids)
    assert score_a == pytest.approx(0.5, abs=1e-5)
    assert score_b == pytest.approx(0.5, abs=1e-5)


def test_debate_reward_required_keys():
    """debate_reward returns dict with required keys."""
    model = make_model()
    gen = ArgumentGenerator(model, DEBATE_CFG)
    judge = JudgeModel(model, DEBATE_CFG)
    q = make_question_ids()
    result = debate_reward(gen, judge, q, n_rounds=1)
    assert "pro_score" in result
    assert "con_score" in result
    assert "winner" in result
    assert "rounds" in result


def test_debate_reward_winner_is_valid():
    """debate_reward winner is 'pro' or 'con'."""
    model = make_model()
    gen = ArgumentGenerator(model, DEBATE_CFG)
    judge = JudgeModel(model, DEBATE_CFG)
    q = make_question_ids()
    result = debate_reward(gen, judge, q, n_rounds=1)
    assert result["winner"] in ("pro", "con")


def test_debate_reward_rounds_field():
    """debate_reward rounds field equals n_rounds argument."""
    model = make_model()
    gen = ArgumentGenerator(model, DEBATE_CFG)
    judge = JudgeModel(model, DEBATE_CFG)
    q = make_question_ids()
    result = debate_reward(gen, judge, q, n_rounds=1)
    assert result["rounds"] == 1


def test_trainer_train_step_required_keys():
    """train_step returns dict with required keys."""
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = DebateRewardTrainer(model, DEBATE_CFG, optimizer)
    chosen = torch.randint(0, 256, (1, 4))
    rejected = torch.randint(0, 256, (1, 4))
    result = trainer.train_step(chosen, rejected)
    assert "loss" in result
    assert "chosen_score" in result
    assert "rejected_score" in result


def test_trainer_train_step_loss_finite():
    """train_step loss is a finite float."""
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = DebateRewardTrainer(model, DEBATE_CFG, optimizer)
    chosen = torch.randint(0, 256, (1, 4))
    rejected = torch.randint(0, 256, (1, 4))
    result = trainer.train_step(chosen, rejected)
    assert isinstance(result["loss"], float)
    assert math.isfinite(result["loss"])


def test_trainer_train_step_scores_are_floats():
    """train_step chosen_score and rejected_score are floats."""
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = DebateRewardTrainer(model, DEBATE_CFG, optimizer)
    chosen = torch.randint(0, 256, (1, 4))
    rejected = torch.randint(0, 256, (1, 4))
    result = trainer.train_step(chosen, rejected)
    assert isinstance(result["chosen_score"], float)
    assert isinstance(result["rejected_score"], float)


def test_judge_different_sequences_different_scores():
    """score_argument produces different scores for different sequences."""
    model = make_model()
    judge = JudgeModel(model, DEBATE_CFG)
    torch.manual_seed(1)
    ids_a = torch.randint(0, 128, (1, 4))
    ids_b = torch.randint(128, 256, (1, 4))
    score_a = judge.score_argument(ids_a)
    score_b = judge.score_argument(ids_b)
    # Different token sequences should generally produce different log-probs
    assert score_a != pytest.approx(score_b, abs=1e-6)


def test_trainer_train_step_updates_params():
    """train_step performs a gradient update (parameters change)."""
    torch.manual_seed(99)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    trainer = DebateRewardTrainer(model, DEBATE_CFG, optimizer)
    chosen = torch.randint(0, 256, (1, 4))
    rejected = torch.randint(0, 256, (1, 4))

    # Snapshot parameters before step.
    params_before = [p.clone().detach() for p in model.parameters()]

    trainer.train_step(chosen, rejected)

    # At least one parameter tensor should have changed.
    params_after = list(model.parameters())
    any_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(params_before, params_after)
    )
    assert any_changed, "Parameters did not change after train_step"
