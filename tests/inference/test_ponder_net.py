"""Tests for PonderNet adaptive computation module (Banino et al. 2021)."""

from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.ponder_net import (
    PonderConfig,
    HaltingUnit,
    PonderNet,
    geometric_prior,
    ponder_loss,
)

# ---------------------------------------------------------------------------
# Shared tiny config / fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
)

PONDER_CFG = PonderConfig(
    d_model=64,
    max_steps=4,
    halt_threshold=0.9,
    lambda_p=0.01,
    p_geometric=0.2,
)


@pytest.fixture
def base_model() -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(TINY_CFG)


@pytest.fixture
def ponder_net(base_model: AureliusTransformer) -> PonderNet:
    torch.manual_seed(42)
    return PonderNet(base_model, PONDER_CFG)


# ---------------------------------------------------------------------------
# 1. HaltingUnit output shape is (B,)
# ---------------------------------------------------------------------------

def test_halting_unit_output_shape():
    torch.manual_seed(0)
    hu = HaltingUnit(d_model=64)
    hidden = torch.randn(3, 8, 64)  # (B=3, T=8, d_model=64)
    out = hu(hidden)
    assert out.shape == (3,), f"Expected (3,) but got {out.shape}"


# ---------------------------------------------------------------------------
# 2. HaltingUnit output in (0, 1) — sigmoid activation
# ---------------------------------------------------------------------------

def test_halting_unit_output_range():
    torch.manual_seed(1)
    hu = HaltingUnit(d_model=64)
    hidden = torch.randn(4, 10, 64)
    out = hu(hidden)
    assert (out > 0).all(), "All halting probs must be > 0"
    assert (out < 1).all(), "All halting probs must be < 1"


# ---------------------------------------------------------------------------
# 3. PonderNet forward returns (Tensor, dict)
# ---------------------------------------------------------------------------

def test_ponder_net_forward_return_types(ponder_net: PonderNet):
    torch.manual_seed(2)
    ids = torch.randint(0, 256, (2, 8))
    result = ponder_net(ids)
    assert isinstance(result, tuple), "PonderNet.forward must return a tuple"
    assert len(result) == 2, "PonderNet.forward must return (logits, info_dict)"
    logits, info = result
    assert isinstance(logits, torch.Tensor), "First element must be a Tensor"
    assert isinstance(info, dict), "Second element must be a dict"


# ---------------------------------------------------------------------------
# 4. weighted_logits shape is (B, T, vocab)
# ---------------------------------------------------------------------------

def test_ponder_net_logits_shape(ponder_net: PonderNet):
    torch.manual_seed(3)
    B, T = 2, 8
    ids = torch.randint(0, 256, (B, T))
    logits, _ = ponder_net(ids)
    assert logits.shape == (B, T, 256), f"Expected ({B}, {T}, 256) but got {logits.shape}"


# ---------------------------------------------------------------------------
# 5. info_dict has required keys: n_steps, halt_probs, step_weights
# ---------------------------------------------------------------------------

def test_ponder_net_info_dict_keys(ponder_net: PonderNet):
    torch.manual_seed(4)
    ids = torch.randint(0, 256, (2, 6))
    _, info = ponder_net(ids)
    assert "n_steps" in info, "info_dict must contain 'n_steps'"
    assert "halt_probs" in info, "info_dict must contain 'halt_probs'"
    assert "step_weights" in info, "info_dict must contain 'step_weights'"


# ---------------------------------------------------------------------------
# 6. n_steps <= config.max_steps
# ---------------------------------------------------------------------------

def test_ponder_net_n_steps_bounded(ponder_net: PonderNet):
    torch.manual_seed(5)
    ids = torch.randint(0, 256, (2, 6))
    _, info = ponder_net(ids)
    assert info["n_steps"] <= PONDER_CFG.max_steps, (
        f"n_steps ({info['n_steps']}) must be <= max_steps ({PONDER_CFG.max_steps})"
    )
    assert info["n_steps"] >= 1, "n_steps must be at least 1"


# ---------------------------------------------------------------------------
# 7. step_weights sum to ~1 per batch item
# ---------------------------------------------------------------------------

def test_ponder_net_step_weights_sum_to_one(ponder_net: PonderNet):
    torch.manual_seed(6)
    ids = torch.randint(0, 256, (3, 8))
    _, info = ponder_net(ids)
    step_weights = info["step_weights"]  # (B, n_steps)
    weight_sums = step_weights.sum(dim=1)  # (B,)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), (
        f"step_weights must sum to 1 per batch item, got {weight_sums}"
    )


# ---------------------------------------------------------------------------
# 8. adaptive_generate returns tensor of length max_new_tokens
# ---------------------------------------------------------------------------

def test_adaptive_generate_output_length(ponder_net: PonderNet):
    torch.manual_seed(7)
    prompt = torch.randint(0, 256, (4,))
    max_new = 5
    generated, _ = ponder_net.adaptive_generate(prompt, max_new_tokens=max_new)
    assert generated.shape == (max_new,), (
        f"Expected generated tensor of length {max_new}, got {generated.shape}"
    )


# ---------------------------------------------------------------------------
# 9. steps_per_token length equals max_new_tokens
# ---------------------------------------------------------------------------

def test_adaptive_generate_steps_per_token_length(ponder_net: PonderNet):
    torch.manual_seed(8)
    prompt = torch.randint(0, 256, (4,))
    max_new = 6
    _, steps_per_token = ponder_net.adaptive_generate(prompt, max_new_tokens=max_new)
    assert len(steps_per_token) == max_new, (
        f"Expected steps_per_token of length {max_new}, got {len(steps_per_token)}"
    )


# ---------------------------------------------------------------------------
# 10. geometric_prior sums to ~1
# ---------------------------------------------------------------------------

def test_geometric_prior_sums_to_one():
    prior = geometric_prior(n_steps=8, p=0.2)
    assert abs(prior.sum().item() - 1.0) < 1e-5, (
        f"geometric_prior must sum to 1, got {prior.sum().item()}"
    )


# ---------------------------------------------------------------------------
# 11. geometric_prior is monotonically decreasing for p > 0
# ---------------------------------------------------------------------------

def test_geometric_prior_monotonically_decreasing():
    prior = geometric_prior(n_steps=8, p=0.2)
    diffs = prior[1:] - prior[:-1]
    assert (diffs <= 0).all(), (
        f"geometric_prior must be monotonically non-increasing, got {prior.tolist()}"
    )


# ---------------------------------------------------------------------------
# 12. ponder_loss returns (Tensor, dict) with nll/kl_reg/total keys
# ---------------------------------------------------------------------------

def test_ponder_loss_return_structure(ponder_net: PonderNet):
    torch.manual_seed(9)
    B, T = 2, 8
    ids = torch.randint(0, 256, (B, T))
    weighted_logits, info = ponder_net(ids)
    halt_probs = info["halt_probs"]

    loss, loss_dict = ponder_loss(
        weighted_logits,
        ids,
        halt_probs,
        lambda_p=PONDER_CFG.lambda_p,
        p_geometric=PONDER_CFG.p_geometric,
    )

    assert isinstance(loss, torch.Tensor), "ponder_loss must return a Tensor as first element"
    assert isinstance(loss_dict, dict), "ponder_loss must return a dict as second element"
    assert "nll" in loss_dict, "loss_dict must contain 'nll'"
    assert "kl_reg" in loss_dict, "loss_dict must contain 'kl_reg'"
    assert "total" in loss_dict, "loss_dict must contain 'total'"
    assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"


# ---------------------------------------------------------------------------
# 13. ponder_loss total matches nll + lambda_p * kl_reg
# ---------------------------------------------------------------------------

def test_ponder_loss_total_consistency(ponder_net: PonderNet):
    torch.manual_seed(10)
    B, T = 2, 8
    ids = torch.randint(0, 256, (B, T))
    weighted_logits, info = ponder_net(ids)
    halt_probs = info["halt_probs"]
    lp = 0.05

    loss, loss_dict = ponder_loss(weighted_logits, ids, halt_probs, lambda_p=lp)

    expected = loss_dict["nll"] + lp * loss_dict["kl_reg"]
    assert abs(loss_dict["total"] - expected) < 1e-5, (
        f"total ({loss_dict['total']:.6f}) != nll + lambda_p * kl_reg ({expected:.6f})"
    )


# ---------------------------------------------------------------------------
# 14. halt_probs shape is (B, n_steps)
# ---------------------------------------------------------------------------

def test_ponder_net_halt_probs_shape(ponder_net: PonderNet):
    torch.manual_seed(11)
    B, T = 3, 6
    ids = torch.randint(0, 256, (B, T))
    _, info = ponder_net(ids)
    halt_probs = info["halt_probs"]
    n_steps = info["n_steps"]
    assert halt_probs.shape == (B, n_steps), (
        f"halt_probs shape should be ({B}, {n_steps}), got {halt_probs.shape}"
    )


# ---------------------------------------------------------------------------
# 15. steps_per_token values are within [1, max_steps]
# ---------------------------------------------------------------------------

def test_adaptive_generate_steps_in_range(ponder_net: PonderNet):
    torch.manual_seed(12)
    prompt = torch.randint(0, 256, (4,))
    _, steps_per_token = ponder_net.adaptive_generate(prompt, max_new_tokens=5)
    for s in steps_per_token:
        assert 1 <= s <= PONDER_CFG.max_steps, (
            f"steps_per_token value {s} out of range [1, {PONDER_CFG.max_steps}]"
        )
