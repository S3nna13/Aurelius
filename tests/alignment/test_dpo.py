"""Tests for native DPO implementation."""
import copy
import math
import pytest
import torch
import torch.nn.functional as F
from src.alignment.dpo import (
    compute_log_probs,
    dpo_loss,
    DPOConfig,
    DPOTrainer,
    compute_reward_margin,
    _dpo_loss_from_logps,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
    )
    return AureliusTransformer(cfg)


@pytest.fixture
def ref_model(small_model):
    ref = copy.deepcopy(small_model)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _make_batch(batch_size=2, seq_len=16, vocab_size=256):
    torch.manual_seed(1)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Mask: first 8 tokens are prompt (0), last 8 are response (1)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    mask[:, seq_len // 2:] = 1
    return input_ids, mask


# ---------------------------------------------------------------------------
# Legacy tests (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def test_compute_log_probs_shape(small_model):
    """compute_log_probs must return (B,) shaped tensor."""
    ids, mask = _make_batch()
    lp = compute_log_probs(small_model, ids, mask)
    assert lp.shape == (2,)


def test_compute_log_probs_finite(small_model):
    """Log probs must be finite and negative."""
    ids, mask = _make_batch()
    lp = compute_log_probs(small_model, ids, mask)
    assert torch.isfinite(lp).all()
    assert (lp < 0).all()


def test_dpo_loss_scalar(small_model):
    """dpo_loss must return a scalar finite positive tensor."""
    chosen_ids, chosen_mask = _make_batch()
    rejected_ids, rejected_mask = _make_batch(batch_size=2)
    # Perturb rejected to be different
    rejected_ids = (rejected_ids + 1) % 256

    reference = copy.deepcopy(small_model)
    for p in reference.parameters():
        p.requires_grad_(False)

    loss = dpo_loss(small_model, reference, chosen_ids, rejected_ids, chosen_mask, rejected_mask)
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)
    assert loss > 0


def test_dpo_loss_identical_pair(small_model):
    """When chosen == rejected, DPO loss should be close to log(2)."""
    ids, mask = _make_batch()

    reference = copy.deepcopy(small_model)
    for p in reference.parameters():
        p.requires_grad_(False)

    loss = dpo_loss(small_model, reference, ids, ids, mask, mask)
    assert abs(loss.item() - math.log(2)) < 0.1


def test_dpo_loss_backward(small_model):
    """dpo_loss must produce gradients through policy parameters."""
    chosen_ids, chosen_mask = _make_batch()
    rejected_ids = (chosen_ids + 1) % 256
    rejected_mask = chosen_mask.clone()

    reference = copy.deepcopy(small_model)
    for p in reference.parameters():
        p.requires_grad_(False)

    loss = dpo_loss(small_model, reference, chosen_ids, rejected_ids, chosen_mask, rejected_mask)
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in small_model.parameters()
    )
    assert has_grad, "No gradients flowed through policy"


def test_reference_frozen(small_model):
    """Reference model must not accumulate gradients."""
    chosen_ids, chosen_mask = _make_batch()
    rejected_ids = (chosen_ids + 1) % 256
    rejected_mask = chosen_mask.clone()

    reference = copy.deepcopy(small_model)
    for p in reference.parameters():
        p.requires_grad_(False)

    loss = dpo_loss(small_model, reference, chosen_ids, rejected_ids, chosen_mask, rejected_mask)
    loss.backward()

    for p in reference.parameters():
        assert p.grad is None, "Reference model should not have gradients"


# ---------------------------------------------------------------------------
# DPOConfig tests
# ---------------------------------------------------------------------------

def test_dpo_config_defaults():
    """DPOConfig should have correct default values."""
    cfg = DPOConfig()
    assert cfg.beta == 0.1
    assert cfg.label_smoothing == 0.0
    assert cfg.reference_free is False
    assert cfg.loss_type == "sigmoid"


def test_dpo_config_custom():
    """DPOConfig should accept custom values."""
    cfg = DPOConfig(beta=0.5, label_smoothing=0.1, reference_free=True, loss_type="hinge")
    assert cfg.beta == 0.5
    assert cfg.label_smoothing == 0.1
    assert cfg.reference_free is True
    assert cfg.loss_type == "hinge"


# ---------------------------------------------------------------------------
# compute_log_probs tests
# ---------------------------------------------------------------------------

def test_compute_log_probs_returns_batch_tensor(small_model):
    """compute_log_probs returns a 1D tensor with one entry per batch item."""
    ids, mask = _make_batch(batch_size=4, seq_len=16)
    lp = compute_log_probs(small_model, ids, mask)
    assert lp.ndim == 1
    assert lp.shape[0] == 4


def test_compute_log_probs_full_mask_vs_partial(small_model):
    """Partial mask yields smaller absolute log prob than full response mask."""
    ids, full_mask = _make_batch(batch_size=1, seq_len=16)
    partial_mask = full_mask.clone()
    # Zero out the second half of the response mask
    partial_mask[:, 12:] = 0

    lp_full = compute_log_probs(small_model, ids, full_mask)
    lp_partial = compute_log_probs(small_model, ids, partial_mask)

    # Full mask covers more tokens, so sum of log probs should be more negative
    assert lp_partial.abs() < lp_full.abs(), (
        "Partial mask should give smaller absolute log prob than full mask"
    )


def test_compute_log_probs_zero_mask_gives_zero(small_model):
    """A zero response mask should yield zero log probs (no response tokens)."""
    ids, _ = _make_batch(batch_size=2, seq_len=16)
    zero_mask = torch.zeros(2, 16, dtype=torch.long)
    lp = compute_log_probs(small_model, ids, zero_mask)
    assert torch.allclose(lp, torch.zeros_like(lp))


# ---------------------------------------------------------------------------
# dpo_loss (new logps-based functional API) tests
# ---------------------------------------------------------------------------

def test_dpo_loss_sigmoid_preferred_gt_rejected_lower_loss():
    """Sigmoid DPO loss: when chosen margin > rejected margin, loss is lower."""
    cfg = DPOConfig(beta=0.1, loss_type="sigmoid")
    # Case 1: clear preference signal
    loss_good, _ = _dpo_loss_from_logps(
        torch.tensor([-1.0]),   # policy_chosen
        torch.tensor([-5.0]),   # policy_rejected
        torch.tensor([-2.0]),   # ref_chosen
        torch.tensor([-2.0]),   # ref_rejected
        cfg,
    )
    # Case 2: reversed preference signal
    loss_bad, _ = _dpo_loss_from_logps(
        torch.tensor([-5.0]),
        torch.tensor([-1.0]),
        torch.tensor([-2.0]),
        torch.tensor([-2.0]),
        cfg,
    )
    assert loss_good < loss_bad, "Clear preference should give lower sigmoid DPO loss"


def test_dpo_loss_hinge_margin_satisfied_gives_zero():
    """Hinge DPO loss: when margin is satisfied, loss should be zero."""
    cfg = DPOConfig(beta=0.1, loss_type="hinge")
    # beta * (chosen_diff - rejected_diff) = 0.1 * (4 - (-4)) * 0.1 wait let me be explicit
    # chosen_diff = policy_chosen - ref_chosen = -1 - (-3) = 2
    # rejected_diff = policy_rejected - ref_rejected = -3 - (-3) = 0
    # margin = beta * (chosen_diff - rejected_diff) = 0.1 * 2 = 0.2 > 1 is false
    # Need margin = beta*(chosen_diff - rejected_diff) >= 1
    # So with beta=0.1, need (chosen_diff - rejected_diff) >= 10
    loss, _ = _dpo_loss_from_logps(
        torch.tensor([-1.0]),   # policy_chosen
        torch.tensor([-20.0]),  # policy_rejected
        torch.tensor([-1.0]),   # ref_chosen (chosen_diff = 0)
        torch.tensor([-9.0]),   # ref_rejected (rejected_diff = -11)
        # margin = 0.1 * (0 - (-11)) = 1.1 > 1, loss = max(0, 1-1.1) = 0
        cfg,
    )
    assert loss.item() == 0.0, f"Hinge loss should be zero when margin >= 1, got {loss.item()}"


def test_dpo_loss_hinge_margin_unsatisfied_positive():
    """Hinge DPO loss: when margin < 1, loss should be positive."""
    cfg = DPOConfig(beta=0.1, loss_type="hinge")
    # margin = beta * (chosen_diff - rejected_diff) = 0.1 * 0 = 0 < 1 → loss = 1
    loss, _ = _dpo_loss_from_logps(
        torch.tensor([-2.0]),
        torch.tensor([-2.0]),
        torch.tensor([-2.0]),
        torch.tensor([-2.0]),
        cfg,
    )
    assert loss.item() > 0.0


def test_dpo_loss_ipo_returns_scalar():
    """IPO loss should return a scalar (0-dim) tensor."""
    cfg = DPOConfig(beta=0.1, loss_type="ipo")
    loss, _ = _dpo_loss_from_logps(
        torch.tensor([-1.0, -2.0]),
        torch.tensor([-3.0, -4.0]),
        torch.tensor([-1.5, -2.5]),
        torch.tensor([-3.5, -4.5]),
        cfg,
    )
    assert loss.ndim == 0, "IPO loss should be a scalar"
    assert torch.isfinite(loss)


def test_dpo_loss_all_three_types_work():
    """All three loss types should run without errors."""
    chosen = torch.tensor([-1.0, -2.0])
    rejected = torch.tensor([-3.0, -4.0])
    ref_chosen = torch.tensor([-1.5, -2.5])
    ref_rejected = torch.tensor([-3.5, -4.5])

    for loss_type in ["sigmoid", "hinge", "ipo"]:
        cfg = DPOConfig(beta=0.1, loss_type=loss_type)
        loss, metrics = _dpo_loss_from_logps(chosen, rejected, ref_chosen, ref_rejected, cfg)
        assert torch.isfinite(loss), f"{loss_type} loss is not finite"
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "reward_margin" in metrics


def test_dpo_loss_label_smoothing_effect():
    """Label smoothing should increase the sigmoid DPO loss."""
    chosen = torch.tensor([-1.0])
    rejected = torch.tensor([-5.0])
    ref_chosen = torch.tensor([-2.0])
    ref_rejected = torch.tensor([-2.0])

    cfg_no_ls = DPOConfig(beta=0.1, label_smoothing=0.0, loss_type="sigmoid")
    cfg_ls = DPOConfig(beta=0.1, label_smoothing=0.2, loss_type="sigmoid")

    loss_no_ls, _ = _dpo_loss_from_logps(chosen, rejected, ref_chosen, ref_rejected, cfg_no_ls)
    loss_ls, _ = _dpo_loss_from_logps(chosen, rejected, ref_chosen, ref_rejected, cfg_ls)

    # With positive margin, label smoothing penalises strong confidence → higher loss
    assert loss_ls > loss_no_ls, "Label smoothing should increase sigmoid DPO loss"


def test_dpo_loss_invalid_loss_type():
    """An unsupported loss_type should raise ValueError."""
    cfg = DPOConfig(loss_type="unknown")
    with pytest.raises(ValueError, match="Unknown loss_type"):
        _dpo_loss_from_logps(
            torch.tensor([-1.0]),
            torch.tensor([-2.0]),
            torch.tensor([-1.5]),
            torch.tensor([-2.5]),
            cfg,
        )


def test_dpo_loss_reference_free_mode():
    """reference_free=True should ignore reference log probs."""
    cfg_ref = DPOConfig(beta=0.1, reference_free=False, loss_type="sigmoid")
    cfg_free = DPOConfig(beta=0.1, reference_free=True, loss_type="sigmoid")

    chosen = torch.tensor([-1.0])
    rejected = torch.tensor([-3.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])

    loss_ref, _ = _dpo_loss_from_logps(chosen, rejected, ref_chosen, ref_rejected, cfg_ref)
    loss_free, _ = _dpo_loss_from_logps(chosen, rejected, ref_chosen, ref_rejected, cfg_free)

    # With zero references the results should be equal
    assert abs(loss_ref.item() - loss_free.item()) < 1e-5


# ---------------------------------------------------------------------------
# DPOTrainer tests
# ---------------------------------------------------------------------------

def test_dpo_trainer_train_step_returns_required_keys(small_model, ref_model):
    """DPOTrainer.train_step must return dict with required keys."""
    cfg = DPOConfig(beta=0.1, loss_type="sigmoid")
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
    trainer = DPOTrainer(small_model, ref_model, cfg, optimizer)

    chosen_ids, response_mask = _make_batch(batch_size=2, seq_len=16)
    rejected_ids = (chosen_ids + 1) % 256

    result = trainer.train_step(chosen_ids, rejected_ids, response_mask)

    assert "loss" in result, "train_step must return 'loss'"
    assert "chosen_rewards" in result, "train_step must return 'chosen_rewards'"
    assert "rejected_rewards" in result, "train_step must return 'rejected_rewards'"
    assert "reward_margin" in result, "train_step must return 'reward_margin'"


def test_dpo_trainer_train_step_loss_is_finite(small_model, ref_model):
    """DPOTrainer.train_step loss must be finite."""
    cfg = DPOConfig(beta=0.1, loss_type="sigmoid")
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
    trainer = DPOTrainer(small_model, ref_model, cfg, optimizer)

    chosen_ids, response_mask = _make_batch(batch_size=2, seq_len=16)
    rejected_ids = (chosen_ids + 3) % 256

    result = trainer.train_step(chosen_ids, rejected_ids, response_mask)
    assert math.isfinite(result["loss"]), f"Loss should be finite, got {result['loss']}"


def test_dpo_trainer_ref_model_frozen_after_init(small_model, ref_model):
    """DPOTrainer should freeze reference model parameters."""
    cfg = DPOConfig()
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
    trainer = DPOTrainer(small_model, ref_model, cfg, optimizer)

    for p in trainer.ref_model.parameters():
        assert not p.requires_grad, "Reference model params should be frozen"


def test_dpo_trainer_hinge_loss_type(small_model, ref_model):
    """DPOTrainer should work with hinge loss type."""
    cfg = DPOConfig(beta=0.1, loss_type="hinge")
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
    trainer = DPOTrainer(small_model, ref_model, cfg, optimizer)

    chosen_ids, response_mask = _make_batch(batch_size=2, seq_len=16)
    rejected_ids = (chosen_ids + 1) % 256

    result = trainer.train_step(chosen_ids, rejected_ids, response_mask)
    assert math.isfinite(result["loss"])
    assert result["loss"] >= 0.0, "Hinge loss is non-negative"


def test_dpo_trainer_ipo_loss_type(small_model, ref_model):
    """DPOTrainer should work with IPO loss type."""
    cfg = DPOConfig(beta=0.1, loss_type="ipo")
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
    trainer = DPOTrainer(small_model, ref_model, cfg, optimizer)

    chosen_ids, response_mask = _make_batch(batch_size=2, seq_len=16)
    rejected_ids = (chosen_ids + 1) % 256

    result = trainer.train_step(chosen_ids, rejected_ids, response_mask)
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# compute_reward_margin tests
# ---------------------------------------------------------------------------

def test_compute_reward_margin_positive_when_chosen_better():
    """compute_reward_margin should be positive when chosen_logps > rejected_logps."""
    chosen = torch.tensor([-1.0, -2.0])
    rejected = torch.tensor([-3.0, -4.0])
    margin = compute_reward_margin(chosen, rejected, beta=0.1)
    assert (margin > 0).all(), "Reward margin should be positive when chosen > rejected"


def test_compute_reward_margin_negative_when_rejected_better():
    """compute_reward_margin should be negative when chosen_logps < rejected_logps."""
    chosen = torch.tensor([-4.0, -5.0])
    rejected = torch.tensor([-1.0, -2.0])
    margin = compute_reward_margin(chosen, rejected, beta=0.1)
    assert (margin < 0).all(), "Reward margin should be negative when chosen < rejected"


def test_compute_reward_margin_zero_when_equal():
    """compute_reward_margin should be zero when chosen_logps == rejected_logps."""
    logps = torch.tensor([-2.0, -3.0])
    margin = compute_reward_margin(logps, logps, beta=0.5)
    assert torch.allclose(margin, torch.zeros_like(margin))


def test_compute_reward_margin_scales_with_beta():
    """compute_reward_margin should scale linearly with beta."""
    chosen = torch.tensor([-1.0])
    rejected = torch.tensor([-3.0])
    margin_small = compute_reward_margin(chosen, rejected, beta=0.1)
    margin_large = compute_reward_margin(chosen, rejected, beta=1.0)
    assert abs(margin_large.item() / margin_small.item() - 10.0) < 1e-5
