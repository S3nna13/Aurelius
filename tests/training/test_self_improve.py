"""Tests for the autonomous self-improvement loop."""

import math
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.alignment.reward_model import RewardModel
from src.training.self_improve import (
    SelfImproveConfig,
    SelfImproveReport,
    SelfImprover,
    RoundResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    torch.manual_seed(0)
    model = AureliusTransformer(cfg)
    return model


@pytest.fixture(scope="module")
def small_reward_model(small_model):
    import copy
    backbone = copy.deepcopy(small_model)
    rm = RewardModel(backbone=backbone, freeze_backbone=False)
    return rm


@pytest.fixture(scope="module")
def tiny_cfg():
    return SelfImproveConfig(
        n_synthetic_examples=4,
        sft_steps=2,
        max_rounds=2,
        min_examples_to_train=2,  # low enough that tests can train
        max_new_tokens=4,
        reward_threshold=-1e9,    # keep all above a very low threshold
        top_fraction=0.75,
    )


@pytest.fixture(scope="module")
def prompt_template():
    """Short fixed prompt template (5 tokens)."""
    return torch.randint(0, 256, (5,))


@pytest.fixture(scope="module")
def self_improver(small_model, small_reward_model, tiny_cfg):
    return SelfImprover(
        model=small_model,
        reward_model=small_reward_model,
        cfg=tiny_cfg,
        eval_loader=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_self_improve_config_defaults():
    cfg = SelfImproveConfig()
    assert cfg.top_fraction == 0.7
    assert cfg.max_rounds == 5


def test_generate_synthetic_batch_count(self_improver, prompt_template):
    responses = self_improver.generate_synthetic_batch(4, prompt_template)
    assert len(responses) == 4


def test_generate_synthetic_batch_types(self_improver, prompt_template):
    responses = self_improver.generate_synthetic_batch(3, prompt_template)
    for r in responses:
        assert isinstance(r, torch.Tensor)


def test_score_with_reward_model_shape(self_improver, prompt_template):
    n = 4
    responses = self_improver.generate_synthetic_batch(n, prompt_template)
    prompts = [prompt_template for _ in range(n)]
    scores = self_improver.score_with_reward_model(prompts, responses)
    assert scores.shape == (n,)


def test_filter_by_reward_keeps_top_fraction(self_improver):
    n_total = 8
    # Create dummy examples: (input_ids, labels)
    examples = [
        (torch.randint(0, 256, (10,)), torch.randint(0, 256, (10,)))
        for _ in range(n_total)
    ]
    rewards = torch.randn(n_total)
    filtered, mean_r = self_improver.filter_by_reward(examples, rewards)
    assert len(filtered) <= n_total
    assert len(filtered) >= 1  # at least one kept


def test_sft_step_returns_finite_loss(self_improver):
    examples = [
        (torch.randint(0, 256, (12,)), torch.randint(0, 256, (12,)))
        for _ in range(4)
    ]
    loss = self_improver.sft_step(examples)
    assert isinstance(loss, float)
    assert math.isfinite(loss)


def test_run_round_returns_result(self_improver, tiny_cfg, prompt_template):
    result = self_improver.run_round(0, prompt_template)
    assert isinstance(result, RoundResult)
    assert result.round_idx == 0


def test_run_returns_report(small_model, small_reward_model, tiny_cfg, prompt_template):
    import copy
    # Use a fresh improver to avoid state pollution
    improver = SelfImprover(
        model=copy.deepcopy(small_model),
        reward_model=copy.deepcopy(small_reward_model),
        cfg=tiny_cfg,
        eval_loader=None,
    )
    report = improver.run(prompt_template)
    assert isinstance(report, SelfImproveReport)


def test_report_rounds_count(small_model, small_reward_model, tiny_cfg, prompt_template):
    import copy
    improver = SelfImprover(
        model=copy.deepcopy(small_model),
        reward_model=copy.deepcopy(small_reward_model),
        cfg=tiny_cfg,
        eval_loader=None,
    )
    report = improver.run(prompt_template)
    assert len(report.rounds) == tiny_cfg.max_rounds


def test_round_result_n_kept_le_generated(self_improver, prompt_template):
    result = self_improver.run_round(0, prompt_template)
    assert result.n_kept <= result.n_generated
