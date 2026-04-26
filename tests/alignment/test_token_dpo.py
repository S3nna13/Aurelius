"""Tests for src/alignment/token_dpo.py

Uses a tiny 2-layer transformer (vocab_size=256, d_model=64) to verify
Token-level DPO behaviour end-to-end without any HuggingFace dependency.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.token_dpo import (
    TokenDPOConfig,
    TokenDPOTrainer,
)

# ---------------------------------------------------------------------------
# Tiny 2-layer transformer
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
SEQ_LEN = 16
BATCH = 2


class _TinyTransformer(nn.Module):
    """Minimal causal transformer: embedding -> 2x TransformerEncoderLayer -> lm_head."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        nhead: int = N_HEADS,
        num_layers: int = N_LAYERS,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns (B, T, vocab_size) logits."""
        x = self.embed(input_ids)  # (B, T, D)
        x = self.transformer(x)  # (B, T, D)
        return self.lm_head(x)  # (B, T, V)


def _make_model() -> _TinyTransformer:
    torch.manual_seed(0)
    return _TinyTransformer()


def _make_trainer(
    loss_type: str = "sigmoid",
    token_weight_fn=None,
    normalize_weights: bool = True,
) -> TokenDPOTrainer:
    policy = _make_model()
    ref = copy.deepcopy(policy)
    config = TokenDPOConfig(
        beta=0.1,
        loss_type=loss_type,
        normalize_weights=normalize_weights,
    )
    return TokenDPOTrainer(
        policy=policy,
        ref_policy=ref,
        config=config,
        token_weight_fn=token_weight_fn,
    )


def _random_ids(b: int = BATCH, t: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (b, t))


def _random_labels(b: int = BATCH, t: int = SEQ_LEN, mask_last: int = 0) -> torch.Tensor:
    labels = torch.randint(0, VOCAB_SIZE, (b, t))
    if mask_last > 0:
        labels[:, -mask_last:] = -100
    return labels


# ---------------------------------------------------------------------------
# Test 1: TokenDPOConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = TokenDPOConfig()
    assert cfg.beta == 0.1
    assert cfg.normalize_weights is True
    assert cfg.weight_temperature == 1.0
    assert cfg.min_weight == 0.0
    assert cfg.loss_type == "sigmoid"


# ---------------------------------------------------------------------------
# Test 2: compute_token_log_probs output shape
# ---------------------------------------------------------------------------


def test_compute_token_log_probs_shape():
    trainer = _make_trainer()
    ids = _random_ids()
    labels = _random_labels()
    log_probs = trainer.compute_token_log_probs(trainer.policy, ids, labels)
    assert log_probs.shape == (BATCH, SEQ_LEN), (
        f"Expected ({BATCH}, {SEQ_LEN}), got {log_probs.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: Masked positions (-100 labels) don't contribute to log probs
# ---------------------------------------------------------------------------


def test_masked_positions_are_zero():
    trainer = _make_trainer()
    ids = _random_ids()

    # Build two label tensors: one where last 4 tokens are valid, one where they are masked
    labels_full = _random_labels(mask_last=0)
    labels_masked = labels_full.clone()
    labels_masked[:, -4:] = -100

    lp_full = trainer.compute_token_log_probs(trainer.policy, ids, labels_full)
    lp_masked = trainer.compute_token_log_probs(trainer.policy, ids, labels_masked)

    # Masked positions should be 0.0 in the masked version
    assert torch.all(lp_masked[:, -4:] == 0.0), "Masked positions should produce 0.0 log probs"

    # Full and masked should differ at those positions (log probs non-zero for valid labels)
    assert not torch.allclose(lp_full[:, -4:], lp_masked[:, -4:]), (
        "Masked and unmasked log probs should differ"
    )


# ---------------------------------------------------------------------------
# Test 4: compute_tdpo_weights sums to 1.0 per sequence
# ---------------------------------------------------------------------------


def test_tdpo_weights_sum_to_one():
    trainer = _make_trainer()
    torch.manual_seed(1)
    advantages = torch.randn(BATCH, SEQ_LEN)
    weights = trainer.compute_tdpo_weights(advantages, temperature=1.0)

    # Each row should sum to 1
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(BATCH), atol=1e-5), (
        f"Weight row sums should be 1.0, got {row_sums}"
    )


# ---------------------------------------------------------------------------
# Test 5: Higher advantage tokens get higher weights
# ---------------------------------------------------------------------------


def test_higher_advantage_higher_weight():
    trainer = _make_trainer()
    advantages = torch.zeros(1, SEQ_LEN)
    # Make position 0 have the highest advantage
    advantages[0, 0] = 10.0
    # Make position 1 have the lowest
    advantages[0, 1] = -10.0

    weights = trainer.compute_tdpo_weights(advantages, temperature=1.0)
    assert weights[0, 0] > weights[0, 1], (
        f"Higher advantage position should have higher weight; "
        f"pos 0: {weights[0, 0].item():.4f}, pos 1: {weights[0, 1].item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: compute_token_dpo_loss returns scalar loss
# ---------------------------------------------------------------------------


def test_compute_token_dpo_loss_scalar():
    trainer = _make_trainer()
    torch.manual_seed(42)
    B, T = BATCH, SEQ_LEN
    pc = torch.randn(B, T)
    pr = torch.randn(B, T)
    rc = torch.randn(B, T)
    rr = torch.randn(B, T)
    loss, _ = trainer.compute_token_dpo_loss(pc, pr, rc, rr)
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss should be finite"


# ---------------------------------------------------------------------------
# Test 7: Metrics dict has required keys
# ---------------------------------------------------------------------------


def test_metrics_dict_required_keys():
    trainer = _make_trainer()
    B, T = BATCH, SEQ_LEN
    pc = torch.randn(B, T)
    pr = torch.randn(B, T)
    rc = torch.randn(B, T)
    rr = torch.randn(B, T)
    _, metrics = trainer.compute_token_dpo_loss(pc, pr, rc, rr)
    required = {"loss", "chosen_rewards", "rejected_rewards", "accuracy"}
    assert required.issubset(metrics.keys()), f"Missing keys: {required - set(metrics.keys())}"


# ---------------------------------------------------------------------------
# Test 8: Uniform weights == standard DPO loss (up to normalization)
# ---------------------------------------------------------------------------


def test_uniform_weights_match_standard_dpo():
    """With uniform weights, token-DPO should match sequence-level DPO (scaled by 1/T)."""
    trainer = _make_trainer(normalize_weights=True)
    B, T = BATCH, SEQ_LEN

    torch.manual_seed(7)
    pc = torch.randn(B, T)
    pr = torch.randn(B, T)
    rc = torch.randn(B, T)
    rr = torch.randn(B, T)

    # Uniform weights: each weight = 1/T, normalized stays 1/T
    uniform_w = torch.ones(B, T) / T
    loss_weighted, _ = trainer.compute_token_dpo_loss(pc, pr, rc, rr, uniform_w, uniform_w)

    # Manual standard DPO using mean of log probs (equivalent to uniform-weighted sum)
    pi_c = pc.mean(dim=-1)
    pi_r = pr.mean(dim=-1)
    ref_c = rc.mean(dim=-1)
    ref_r = rr.mean(dim=-1)
    reward_diff = trainer.config.beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss_standard = -F.logsigmoid(reward_diff).mean()

    assert torch.allclose(loss_weighted, loss_standard, atol=1e-5), (
        f"Uniform token-DPO loss ({loss_weighted.item():.6f}) should match "
        f"standard DPO loss ({loss_standard.item():.6f})"
    )


# ---------------------------------------------------------------------------
# Test 9: train_step returns all required keys
# ---------------------------------------------------------------------------


def test_train_step_returns_required_keys():
    trainer = _make_trainer()
    chosen_ids = _random_ids()
    rejected_ids = _random_ids()
    chosen_labels = _random_labels()
    rejected_labels = _random_labels()

    metrics = trainer.train_step(chosen_ids, rejected_ids, chosen_labels, rejected_labels)

    required = {"loss", "chosen_rewards", "rejected_rewards", "accuracy"}
    assert required.issubset(metrics.keys()), f"Missing keys: {required - set(metrics.keys())}"
    # All float values should be finite
    for key in required:
        assert isinstance(metrics[key], float), f"{key} should be a float"
        assert torch.isfinite(torch.tensor(metrics[key])), f"{key} should be finite"


# ---------------------------------------------------------------------------
# Test 10: Gradient flows through token DPO loss
# ---------------------------------------------------------------------------


def test_gradient_flows_through_loss():
    trainer = _make_trainer()

    # Verify policy parameters have no grad before backward
    for p in trainer.policy.parameters():
        assert p.grad is None

    chosen_ids = _random_ids()
    rejected_ids = _random_ids()
    chosen_labels = _random_labels()
    rejected_labels = _random_labels()

    metrics = trainer.train_step(chosen_ids, rejected_ids, chosen_labels, rejected_labels)
    loss_tensor = metrics["_loss_tensor"]

    loss_tensor.backward()

    # At least some policy parameters should have gradients now
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in trainer.policy.parameters()
    )
    assert has_grad, "No gradients found after backward() — gradient flow broken"


# ---------------------------------------------------------------------------
# Test 11: loss_type='ipo' produces different loss than 'sigmoid'
# ---------------------------------------------------------------------------


def test_ipo_loss_differs_from_sigmoid():
    torch.manual_seed(99)
    B, T = BATCH, SEQ_LEN
    pc = torch.randn(B, T)
    pr = torch.randn(B, T)
    rc = torch.randn(B, T)
    rr = torch.randn(B, T)
    w = torch.ones(B, T) / T

    trainer_sigmoid = _make_trainer(loss_type="sigmoid")
    trainer_ipo = _make_trainer(loss_type="ipo")

    loss_sig, _ = trainer_sigmoid.compute_token_dpo_loss(pc, pr, rc, rr, w, w)
    loss_ipo, _ = trainer_ipo.compute_token_dpo_loss(pc, pr, rc, rr, w, w)

    assert not torch.allclose(loss_sig, loss_ipo), (
        f"Sigmoid loss ({loss_sig.item():.6f}) and IPO loss ({loss_ipo.item():.6f}) "
        "should differ for the same inputs"
    )
    assert torch.isfinite(loss_sig), "Sigmoid loss should be finite"
    assert torch.isfinite(loss_ipo), "IPO loss should be finite"
