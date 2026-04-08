import pytest
import torch
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.alignment.process_reward import PRMConfig, ProcessRewardModel, prm_loss


@pytest.fixture
def small_backbone():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    torch.manual_seed(0)
    return AureliusTransformer(cfg)


@pytest.fixture
def prm(small_backbone):
    cfg = PRMConfig(d_model=64, step_token_id=10, freeze_backbone=True)
    return ProcessRewardModel(small_backbone, cfg)


def test_prm_output_keys(prm):
    input_ids = torch.tensor([[1, 2, 10, 3, 4, 10, 5]])  # 2 steps at pos 2, 5
    out = prm(input_ids)
    assert "step_scores" in out
    assert "step_counts" in out
    assert "hidden_states" in out


def test_prm_step_count(prm):
    input_ids = torch.tensor([[1, 2, 10, 3, 4, 10, 5]])  # step_token=10 at pos 2, 5
    out = prm(input_ids)
    assert out["step_counts"][0].item() == 2


def test_prm_step_scores_shape(prm):
    input_ids = torch.tensor([[1, 10, 2, 10, 3, 10]])  # 3 steps
    out = prm(input_ids)
    assert out["step_scores"].shape[0] == 1
    assert out["step_scores"].shape[1] >= 3


def test_prm_hidden_states_shape(prm):
    input_ids = torch.randint(0, 256, (2, 8))
    out = prm(input_ids)
    assert out["hidden_states"].shape == (2, 8, 64)


def test_prm_backbone_frozen(prm):
    for p in prm.backbone.parameters():
        assert not p.requires_grad


def test_prm_scorer_not_frozen(prm):
    assert prm.step_scorer.weight.requires_grad


def test_prm_loss_scalar():
    B, max_steps = 2, 4
    scores = torch.randn(B, max_steps)
    labels = torch.randint(0, 2, (B, max_steps)).float()
    mask = torch.ones(B, max_steps, dtype=torch.bool)
    loss = prm_loss(scores, labels, mask)
    assert loss.ndim == 0
    import math

    assert math.isfinite(loss.item())


def test_prm_loss_partial_mask():
    scores = torch.randn(2, 4)
    labels = torch.zeros(2, 4)
    mask = torch.tensor(
        [[True, True, False, False], [True, False, False, False]]
    )
    loss = prm_loss(scores, labels, mask)
    assert loss.ndim == 0
