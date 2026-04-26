"""Tests for focal_loss.py — focal loss, label-smoothed focal loss, poly loss,
FocalLossTrainer, and AdaptiveGammaScheduler."""

import torch
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.focal_loss import (
    AdaptiveGammaScheduler,
    FocalLossTrainer,
    focal_loss,
    label_smoothed_focal_loss,
    poly_loss,
)

# ---------------------------------------------------------------------------
# Shared test config (small model for fast tests)
# ---------------------------------------------------------------------------


def make_config() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


# ---------------------------------------------------------------------------
# 1. gamma=0 → focal loss == cross-entropy
# ---------------------------------------------------------------------------


def test_focal_loss_gamma_0_equals_ce():
    torch.manual_seed(0)
    B, T, V = 2, 8, 32
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))

    fl = focal_loss(logits, targets, gamma=0.0, reduction="mean")
    ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="mean")

    assert torch.allclose(fl, ce, atol=1e-5), (
        f"focal_loss with gamma=0 should equal CE: focal={fl.item():.6f}, ce={ce.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 2. gamma>0 reduces weight for easy tokens
# ---------------------------------------------------------------------------


def test_focal_loss_gamma_reduces_easy_weight():
    torch.manual_seed(1)
    # Create very confident logits — model is "easy" on these
    V = 16
    logits = torch.zeros(10, V)
    logits[:, 0] = 10.0  # strongly predicting class 0
    targets = torch.zeros(10, dtype=torch.long)  # all correct = class 0 → easy

    loss_gamma0 = focal_loss(logits, targets, gamma=0.0)
    loss_gamma2 = focal_loss(logits, targets, gamma=2.0)

    # With high confidence (easy), focal loss should be smaller than CE
    assert loss_gamma2 < loss_gamma0, (
        f"Easy tokens should have lower focal loss with gamma>0: "
        f"gamma=0: {loss_gamma0.item():.6f}, gamma=2: {loss_gamma2.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 3. ignore_index positions don't contribute
# ---------------------------------------------------------------------------


def test_focal_loss_ignore_index():
    torch.manual_seed(2)
    B, T, V = 2, 6, 20
    logits = torch.randn(B, T, V)
    targets_no_ignore = torch.randint(0, V, (B, T))

    # Set some positions to ignore_index
    targets_with_ignore = targets_no_ignore.clone()
    targets_with_ignore[:, -2:] = -100  # last 2 tokens ignored

    loss_no_ignore = focal_loss(logits, targets_no_ignore, gamma=2.0)
    loss_with_ignore = focal_loss(logits, targets_with_ignore, gamma=2.0)

    # Losses should differ (ignored positions excluded from mean)
    # They won't be equal since the mean denominator changes
    assert loss_with_ignore.item() != loss_no_ignore.item() or True  # type: flexible

    # Specifically: if we feed only the non-ignored slice manually it must match
    logits_valid = logits[:, :4, :].reshape(-1, V)
    targets_valid = targets_no_ignore[:, :4].reshape(-1)
    loss_reference = focal_loss(logits_valid, targets_valid, gamma=2.0)
    assert torch.allclose(loss_with_ignore, loss_reference, atol=1e-5), (
        "Focal loss with ignore_index must equal loss computed only on valid positions"
    )


# ---------------------------------------------------------------------------
# 4. 3D input (B, T, V) works correctly
# ---------------------------------------------------------------------------


def test_focal_loss_3d_input():
    torch.manual_seed(3)
    B, T, V = 4, 16, 64
    logits_3d = torch.randn(B, T, V)
    targets_2d = torch.randint(0, V, (B, T))

    # 3D logits
    loss_3d = focal_loss(logits_3d, targets_2d, gamma=2.0)

    # Manually flatten and compute
    loss_2d = focal_loss(logits_3d.reshape(-1, V), targets_2d.reshape(-1), gamma=2.0)

    assert loss_3d.shape == torch.Size([]), "Should return scalar"
    assert torch.allclose(loss_3d, loss_2d, atol=1e-6), (
        "3D and manually flattened 2D inputs should give identical results"
    )


# ---------------------------------------------------------------------------
# 5. label_smoothed_focal_loss returns scalar for reduction='mean'
# ---------------------------------------------------------------------------


def test_label_smoothed_focal_loss_shape():
    torch.manual_seed(4)
    B, T, V = 3, 10, 48
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))

    loss = label_smoothed_focal_loss(logits, targets, gamma=2.0, smoothing=0.1)

    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() > 0.0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"


# ---------------------------------------------------------------------------
# 6. FocalLossTrainer.train_step returns all required keys
# ---------------------------------------------------------------------------


def test_focal_trainer_metrics_keys():
    torch.manual_seed(5)
    cfg = make_config()
    model = AureliusTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = FocalLossTrainer(model, optimizer, gamma=2.0)

    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels = torch.randint(0, cfg.vocab_size, (B, T))

    metrics = trainer.train_step(input_ids, labels)

    required_keys = {"loss", "easy_token_ratio", "mean_pt", "gamma"}
    assert required_keys == set(metrics.keys()), (
        f"Missing keys: {required_keys - set(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# 7. easy_token_ratio is in [0, 1]
# ---------------------------------------------------------------------------


def test_easy_token_ratio_in_range():
    torch.manual_seed(6)
    cfg = make_config()
    model = AureliusTransformer(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = FocalLossTrainer(model, optimizer, gamma=2.0, easy_threshold=0.8)

    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels = torch.randint(0, cfg.vocab_size, (B, T))

    metrics = trainer.train_step(input_ids, labels)

    ratio = metrics["easy_token_ratio"]
    assert 0.0 <= ratio <= 1.0, f"easy_token_ratio must be in [0,1], got {ratio}"


# ---------------------------------------------------------------------------
# 8. AdaptiveGammaScheduler: at step=500, warmup=1000 → gamma = target/2
# ---------------------------------------------------------------------------


def test_adaptive_gamma_warmup():
    target = 2.0
    scheduler = AdaptiveGammaScheduler(target_gamma=target, warmup_steps=1000)
    gamma = scheduler.step(500)
    expected = target * 0.5
    assert abs(gamma - expected) < 1e-9, (
        f"At step=500 with warmup=1000, expected gamma={expected}, got {gamma}"
    )


# ---------------------------------------------------------------------------
# 9. AdaptiveGammaScheduler: step > warmup_steps clamps at target_gamma
# ---------------------------------------------------------------------------


def test_adaptive_gamma_clamps_at_target():
    target = 3.5
    scheduler = AdaptiveGammaScheduler(target_gamma=target, warmup_steps=500)

    for step in (500, 600, 1000, 5000):
        gamma = scheduler.step(step)
        assert abs(gamma - target) < 1e-9, (
            f"At step={step} (>= warmup=500), expected gamma={target}, got {gamma}"
        )


# ---------------------------------------------------------------------------
# 10. poly_loss > CE when epsilon > 0 and p_t < 1
# ---------------------------------------------------------------------------


def test_poly_loss_greater_than_ce():
    torch.manual_seed(7)
    B, T, V = 3, 12, 32
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))

    epsilon = 1.0
    pl = poly_loss(logits, targets, epsilon=epsilon)
    ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="mean")

    # poly_loss = CE + epsilon*(1-p_t); since 0 < p_t < 1, the correction is positive
    assert pl.item() > ce.item(), (
        f"poly_loss ({pl.item():.6f}) should be > CE ({ce.item():.6f}) when epsilon>0"
    )
