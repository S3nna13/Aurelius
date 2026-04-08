"""Tests for RewardEnsemble — multiple reward models for more reliable RLHF scoring."""
import math
import torch
import pytest
from unittest.mock import MagicMock

from src.alignment.reward_ensemble import (
    EnsembleConfig,
    RewardEnsemble,
    filter_by_uncertainty,
)
from src.alignment.reward_model import RewardModel
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg():
    return AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )


@pytest.fixture
def tiny_backbone(tiny_cfg):
    torch.manual_seed(42)
    return AureliusTransformer(tiny_cfg)


@pytest.fixture
def tiny_reward_model(tiny_backbone):
    return RewardModel(tiny_backbone)


def make_reward_model(tiny_cfg, seed=0):
    torch.manual_seed(seed)
    backbone = AureliusTransformer(tiny_cfg)
    return RewardModel(backbone)


@pytest.fixture
def three_member_ensemble(tiny_cfg):
    members = [make_reward_model(tiny_cfg, seed=i) for i in range(3)]
    return RewardEnsemble(members)


# ---------------------------------------------------------------------------
# 1. test_ensemble_score_shape
# ---------------------------------------------------------------------------

def test_ensemble_score_shape(three_member_ensemble):
    """score() must return a (B,) tensor for a batch of 2."""
    ids = torch.randint(0, 256, (2, 16))
    scores = three_member_ensemble.score(ids)
    assert scores.shape == (2,), f"Expected (2,), got {scores.shape}"
    assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# 2. test_ensemble_mean_aggregation
# ---------------------------------------------------------------------------

def test_ensemble_mean_aggregation():
    """Ensemble mean must equal arithmetic mean of member scores."""
    m1 = MagicMock(spec=RewardModel)
    m2 = MagicMock(spec=RewardModel)
    m3 = MagicMock(spec=RewardModel)
    m1.forward.return_value = torch.tensor([1.0, 2.0])
    m2.forward.return_value = torch.tensor([3.0, 4.0])
    m3.forward.return_value = torch.tensor([5.0, 6.0])
    m1.training = False
    m2.training = False
    m3.training = False

    ensemble = RewardEnsemble([m1, m2, m3])
    ids = torch.randint(0, 10, (2, 8))
    scores = ensemble.score(ids)  # default mode = "mean"

    expected = torch.tensor([(1 + 3 + 5) / 3, (2 + 4 + 6) / 3])
    assert torch.allclose(scores, expected, atol=1e-5), \
        f"Expected {expected}, got {scores}"


# ---------------------------------------------------------------------------
# 3. test_ensemble_min_aggregation
# ---------------------------------------------------------------------------

def test_ensemble_min_aggregation():
    """Ensemble 'min' mode must return the minimum across members."""
    m1 = MagicMock(spec=RewardModel)
    m2 = MagicMock(spec=RewardModel)
    m1.forward.return_value = torch.tensor([1.0, 5.0])
    m2.forward.return_value = torch.tensor([3.0, 2.0])
    m1.training = False
    m2.training = False

    ensemble = RewardEnsemble([m1, m2])
    ids = torch.randint(0, 10, (2, 8))
    scores = ensemble.score(ids, mode="min")

    expected = torch.tensor([1.0, 2.0])
    assert torch.allclose(scores, expected, atol=1e-5), \
        f"Expected {expected}, got {scores}"


# ---------------------------------------------------------------------------
# 4. test_ensemble_product_aggregation
# ---------------------------------------------------------------------------

def test_ensemble_product_aggregation():
    """Ensemble 'product' mode must return geometric mean of member scores."""
    m1 = MagicMock(spec=RewardModel)
    m2 = MagicMock(spec=RewardModel)
    # Use positive values so geometric mean is well-defined
    m1.forward.return_value = torch.tensor([2.0, 4.0])
    m2.forward.return_value = torch.tensor([8.0, 1.0])
    m1.training = False
    m2.training = False

    ensemble = RewardEnsemble([m1, m2])
    ids = torch.randint(0, 10, (2, 8))
    scores = ensemble.score(ids, mode="product")

    # Geometric mean: sqrt(2*8)=4, sqrt(4*1)=2
    expected = torch.tensor([4.0, 2.0])
    assert torch.allclose(scores, expected, atol=1e-4), \
        f"Expected {expected}, got {scores}"


# ---------------------------------------------------------------------------
# 5. test_ensemble_uncertainty_single_member
# ---------------------------------------------------------------------------

def test_ensemble_uncertainty_single_member(tiny_cfg):
    """std must be 0 (or very close) when only one member is present."""
    member = make_reward_model(tiny_cfg, seed=0)
    ensemble = RewardEnsemble([member])
    ids = torch.randint(0, 256, (3, 16))
    mean, std = ensemble.score_with_uncertainty(ids)
    assert std.shape == (3,)
    assert torch.allclose(std, torch.zeros(3), atol=1e-6), \
        f"Expected std=0 for single member, got {std}"


# ---------------------------------------------------------------------------
# 6. test_ensemble_uncertainty_multiple_members
# ---------------------------------------------------------------------------

def test_ensemble_uncertainty_multiple_members(tiny_cfg):
    """std must be > 0 when members are initialised differently (they disagree)."""
    members = [make_reward_model(tiny_cfg, seed=i * 10) for i in range(3)]
    ensemble = RewardEnsemble(members)
    ids = torch.randint(0, 256, (2, 16))
    mean, std = ensemble.score_with_uncertainty(ids)
    assert std.shape == (2,)
    # With different random inits the members will disagree -> std > 0
    assert (std > 0).any(), f"Expected some std > 0, got {std}"


# ---------------------------------------------------------------------------
# 7. test_filter_by_uncertainty
# ---------------------------------------------------------------------------

def test_filter_by_uncertainty(tiny_cfg):
    """filter_by_uncertainty must return (safe_mask, scores) of correct shape/type."""
    members = [make_reward_model(tiny_cfg, seed=i) for i in range(3)]
    ensemble = RewardEnsemble(members)
    ids = torch.randint(0, 256, (4, 16))

    safe_mask, scores = filter_by_uncertainty(ensemble, ids, threshold=1.0)

    assert safe_mask.shape == (4,), f"safe_mask shape wrong: {safe_mask.shape}"
    assert scores.shape == (4,), f"scores shape wrong: {scores.shape}"
    assert safe_mask.dtype == torch.bool, f"safe_mask must be bool, got {safe_mask.dtype}"
    assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# 8. test_ensemble_module_list
# ---------------------------------------------------------------------------

def test_ensemble_module_list(tiny_cfg):
    """Members must be in nn.ModuleList so parameters() works."""
    members = [make_reward_model(tiny_cfg, seed=i) for i in range(2)]
    ensemble = RewardEnsemble(members)

    params = list(ensemble.parameters())
    assert len(params) > 0, "ensemble.parameters() must be non-empty"

    # Each member's reward_head.weight must appear in the ensemble's params.
    # Use identity check (is) to avoid ambiguous Tensor.__eq__ comparisons.
    for i, m in enumerate(members):
        found = any(p is m.reward_head.weight for p in params)
        assert found, f"Member {i} reward_head.weight not found in ensemble parameters()"


# ---------------------------------------------------------------------------
# 9. test_ensemble_train_eval_mode
# ---------------------------------------------------------------------------

def test_ensemble_train_eval_mode(tiny_cfg):
    """ensemble.train() / .eval() must propagate to all members."""
    members = [make_reward_model(tiny_cfg, seed=i) for i in range(3)]
    ensemble = RewardEnsemble(members)

    ensemble.eval()
    for i, m in enumerate(members):
        assert not m.training, f"Member {i} should be in eval mode"

    ensemble.train()
    for i, m in enumerate(members):
        assert m.training, f"Member {i} should be in train mode"


# ---------------------------------------------------------------------------
# 10. test_ensemble_score_with_real_reward_model
# ---------------------------------------------------------------------------

def test_ensemble_score_with_real_reward_model():
    """Integration: real tiny RewardModels + AureliusConfig produce finite scores."""
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    members = []
    for seed in range(3):
        torch.manual_seed(seed)
        backbone = AureliusTransformer(cfg)
        members.append(RewardModel(backbone))

    ensemble = RewardEnsemble(members)
    ids = torch.randint(0, 256, (2, 16))

    # score() with all three modes
    for mode in ("mean", "min", "product"):
        s = ensemble.score(ids, mode=mode)
        assert s.shape == (2,), f"[{mode}] shape wrong: {s.shape}"
        assert torch.isfinite(s).all(), f"[{mode}] non-finite scores: {s}"

    # score_with_uncertainty
    mean, std = ensemble.score_with_uncertainty(ids)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
