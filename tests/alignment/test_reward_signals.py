import torch
import torch.nn as nn
from unittest.mock import MagicMock
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.reward_signals import RewardSignalBundle

def make_bundle(d_model=16, B=2):
    cfg = PRAXISConfig(d_model=d_model, n_principles=4, mc_dropout_n=5)

    # Mock all sub-reward components
    prime_mock    = MagicMock(return_value=(torch.randn(B, 6), {}))
    critique_mock = MagicMock(return_value=torch.rand(B, 4))
    hier_mock     = MagicMock(return_value=torch.rand(B))
    mc_mock       = MagicMock()
    mc_mock.predict_with_uncertainty = MagicMock(
        return_value=(torch.rand(B), torch.rand(B) * 0.1 + 0.01)
    )

    return RewardSignalBundle(cfg, prime_mock, critique_mock, hier_mock, mc_mock)

def test_bundle_returns_six_signals():
    B, T, D = 2, 6, 16
    bundle = make_bundle(d_model=D, B=B)
    hidden = torch.randn(B, T, D)
    log_probs = torch.randn(B, T)
    ref_log_probs = torch.randn(B, T)
    outcome_rewards = torch.rand(B)
    mask = torch.ones(B, T, dtype=torch.bool)

    signals = bundle.compute(hidden, log_probs, ref_log_probs, outcome_rewards, mask)
    assert len(signals) == 6, f"expected 6 signals, got {len(signals)}"
    for name, (mean, std) in signals.items():
        assert mean.shape == (B,), f"{name} mean shape wrong: {mean.shape}"
        assert std.shape  == (B,), f"{name} std shape wrong: {std.shape}"