"""Tests for native DPO implementation."""
import copy
import math
import pytest
import torch
import torch.nn.functional as F
from src.alignment.dpo import compute_log_probs, dpo_loss
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def _make_batch(batch_size=2, seq_len=16, vocab_size=256):
    torch.manual_seed(1)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Mask: first 8 tokens are prompt (0), last 8 are response (1)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    mask[:, seq_len // 2:] = 1
    return input_ids, mask


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
