"""Tests for ORPO and IPO loss modules (Hong et al. 2024, Azar et al. 2024)."""

from __future__ import annotations

import torch

from src.alignment.orpo import IPOLoss, ORPOLoss, ORPOTrainer
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tiny_model() -> AureliusTransformer:
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def _make_orpo_inputs(batch_size: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (chosen_logps, rejected_logps, sft_loss) for testing."""
    torch.manual_seed(42)
    # Mean log probs are in (-inf, 0). Use values in (-2, -0.1) for stability.
    chosen_logps = -torch.rand(batch_size) * 0.5 - 0.1  # roughly (-0.6, -0.1)
    rejected_logps = -torch.rand(batch_size) * 0.5 - 0.5  # roughly (-1.0, -0.5)
    sft_loss = torch.tensor(1.5)
    return chosen_logps, rejected_logps, sft_loss


def _make_ipo_inputs(
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    policy_chosen = -torch.rand(batch_size) - 0.1
    policy_rejected = -torch.rand(batch_size) - 0.5
    ref_chosen = -torch.rand(batch_size) - 0.2
    ref_rejected = -torch.rand(batch_size) - 0.4
    return policy_chosen, policy_rejected, ref_chosen, ref_rejected


# ---------------------------------------------------------------------------
# ORPOLoss tests
# ---------------------------------------------------------------------------


def test_orpo_loss_scalar():
    """forward() must return (loss, dict) where loss is a scalar tensor."""
    loss_fn = ORPOLoss(lambda_=0.1)
    chosen, rejected, sft = _make_orpo_inputs()
    result = loss_fn(chosen, rejected, sft)

    assert isinstance(result, tuple) and len(result) == 2
    loss, metrics = result
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "loss must be a scalar"
    assert isinstance(metrics, dict)


def test_orpo_loss_positive():
    """ORPO loss must be strictly positive."""
    loss_fn = ORPOLoss(lambda_=0.1)
    chosen, rejected, sft = _make_orpo_inputs()
    loss, _ = loss_fn(chosen, rejected, sft)
    assert loss.item() > 0


def test_orpo_metrics_keys():
    """Metrics dict must contain all required keys."""
    loss_fn = ORPOLoss(lambda_=0.1)
    chosen, rejected, sft = _make_orpo_inputs()
    _, metrics = loss_fn(chosen, rejected, sft)

    required = {"sft_loss", "or_loss", "log_odds_ratio", "or_reward_margin"}
    assert required.issubset(metrics.keys()), f"Missing keys: {required - metrics.keys()}"


def test_log_odds_range():
    """log_odds must return finite values for valid (negative) log probs."""
    loss_fn = ORPOLoss()
    log_probs = torch.linspace(-5.0, -0.01, steps=50)
    result = loss_fn.log_odds(log_probs)
    assert torch.isfinite(result).all(), "log_odds returned non-finite values"


def test_orpo_lambda_effect():
    """Higher lambda_ should increase the contribution from the OR loss."""
    chosen, rejected, sft = _make_orpo_inputs()

    loss_low, metrics_low = ORPOLoss(lambda_=0.01)(chosen, rejected, sft)
    loss_high, metrics_high = ORPOLoss(lambda_=1.0)(chosen, rejected, sft)

    # OR component scaled by lambda; higher lambda → higher total loss when or_loss > 0
    assert loss_high.item() > loss_low.item(), (
        "Higher lambda_ should produce higher total loss when OR loss is positive"
    )


# ---------------------------------------------------------------------------
# IPOLoss tests
# ---------------------------------------------------------------------------


def test_ipo_loss_scalar():
    """IPOLoss.forward() must return (loss, dict) where loss is a scalar."""
    loss_fn = IPOLoss(tau=0.1, beta=0.1)
    pc, pr, rc, rr = _make_ipo_inputs()
    result = loss_fn(pc, pr, rc, rr)

    assert isinstance(result, tuple) and len(result) == 2
    loss, metrics = result
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert isinstance(metrics, dict)


def test_ipo_loss_positive():
    """IPO uses squared loss — must be >= 0."""
    loss_fn = IPOLoss(tau=0.1, beta=0.1)
    pc, pr, rc, rr = _make_ipo_inputs()
    loss, _ = loss_fn(pc, pr, rc, rr)
    assert loss.item() >= 0


def test_ipo_metrics_keys():
    """Metrics dict must contain all required keys."""
    loss_fn = IPOLoss(tau=0.1)
    pc, pr, rc, rr = _make_ipo_inputs()
    _, metrics = loss_fn(pc, pr, rc, rr)

    required = {"h_chosen", "h_rejected", "ipo_margin", "loss"}
    assert required.issubset(metrics.keys()), f"Missing keys: {required - metrics.keys()}"


def test_ipo_tau_effect():
    """Higher tau shifts the minimizer (target 1/(2*tau) gets smaller → loss changes)."""
    # With tau_high the target 1/(2*tau) is small → easier to satisfy with typical h values.
    batch_size = 16
    torch.manual_seed(99)
    # Construct inputs where h_chosen - h_rejected is moderate (~0.5).
    pc = torch.full((batch_size,), -0.3)
    pr = torch.full((batch_size,), -0.8)
    rc = torch.zeros(batch_size)
    rr = torch.zeros(batch_size)
    # h_chosen - h_rejected = -0.3 - (-0.8) = 0.5

    loss_low_tau, _ = IPOLoss(tau=0.1)(pc, pr, rc, rr)  # target = 5.0  → big error
    loss_high_tau, _ = IPOLoss(tau=1.0)(pc, pr, rc, rr)  # target = 0.5  → near zero

    assert loss_high_tau.item() < loss_low_tau.item(), (
        "Higher tau (smaller target) should produce lower loss when margin ≈ target"
    )


def test_ipo_unique_minimizer():
    """When h_chosen - h_rejected == 1/(2*tau), IPO loss should be 0."""
    tau = 0.5
    target = 1.0 / (2.0 * tau)  # 1.0

    batch_size = 8
    # Craft inputs so h_chosen - h_rejected == target exactly.
    # h = policy - ref; set ref=0 and policy values s.t. diff = target.
    pc = torch.full((batch_size,), -0.5)
    pr = torch.full((batch_size,), -0.5 - target)  # so diff = target
    rc = torch.zeros(batch_size)
    rr = torch.zeros(batch_size)

    loss_fn = IPOLoss(tau=tau)
    loss, _ = loss_fn(pc, pr, rc, rr)

    assert loss.item() < 1e-6, f"IPO loss should be ~0 at the unique minimizer, got {loss.item()}"


# ---------------------------------------------------------------------------
# ORPOTrainer tests
# ---------------------------------------------------------------------------


def test_orpo_trainer_step_keys():
    """train_step must return a dict that includes a 'loss' key."""
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = ORPOTrainer(model=model, optimizer=optimizer, lambda_=0.1)

    torch.manual_seed(5)
    prompt_ids = torch.randint(0, 256, (8,))
    chosen_ids = torch.randint(0, 256, (8,))
    rejected_ids = torch.randint(0, 256, (8,))

    metrics = trainer.train_step(prompt_ids, chosen_ids, rejected_ids)

    assert isinstance(metrics, dict)
    assert "loss" in metrics, f"'loss' key missing from metrics: {list(metrics.keys())}"
    assert torch.isfinite(metrics["loss"])


def test_orpo_no_ref_model():
    """ORPOTrainer must not have a ref_model attribute (no reference model needed)."""
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = ORPOTrainer(model=model, optimizer=optimizer)

    assert not hasattr(trainer, "ref_model"), (
        "ORPOTrainer should have no ref_model attribute — ORPO requires no reference model"
    )
