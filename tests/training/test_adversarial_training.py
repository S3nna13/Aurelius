"""Tests for adversarial training module (FreeLB + FGSM + robustness scoring)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.adversarial_training import (
    AdvConfig,
    FreeLBTrainer,
    compute_robustness_score,
    fgsm_step,
    forward_with_embeds,
    project_perturbation,
)

# ---------------------------------------------------------------------------
# Tiny model config for fast tests
# ---------------------------------------------------------------------------
CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)
BATCH = 2
SEQ = 8


def make_model() -> AureliusTransformer:
    return AureliusTransformer(CFG)


def make_input_ids() -> torch.Tensor:
    return torch.randint(0, CFG.vocab_size, (BATCH, SEQ))


# ---------------------------------------------------------------------------
# 1. AdvConfig defaults
# ---------------------------------------------------------------------------
def test_adv_config_defaults():
    cfg = AdvConfig()
    assert cfg.epsilon == 0.01
    assert cfg.alpha == 0.005
    assert cfg.n_adv_steps == 3
    assert cfg.norm_type == "l2"


# ---------------------------------------------------------------------------
# 2. project_perturbation — L2: output norm <= epsilon
# ---------------------------------------------------------------------------
def test_project_perturbation_l2_norm():
    torch.manual_seed(0)
    delta = torch.randn(BATCH, SEQ, 64) * 10.0  # large perturbation
    epsilon = 0.5
    projected = project_perturbation(delta, epsilon, norm_type="l2")
    flat = projected.view(BATCH, -1)
    norms = flat.norm(dim=-1)
    assert (norms <= epsilon + 1e-5).all(), f"L2 norms {norms} exceed epsilon {epsilon}"


# ---------------------------------------------------------------------------
# 3. project_perturbation — Linf: all values in [-epsilon, epsilon]
# ---------------------------------------------------------------------------
def test_project_perturbation_linf_bounds():
    torch.manual_seed(1)
    delta = torch.randn(BATCH, SEQ, 64) * 5.0
    epsilon = 0.3
    projected = project_perturbation(delta, epsilon, norm_type="linf")
    assert projected.abs().max().item() <= epsilon + 1e-6


# ---------------------------------------------------------------------------
# 4. project_perturbation — zero delta returns zero
# ---------------------------------------------------------------------------
def test_project_perturbation_zero_delta():
    delta = torch.zeros(BATCH, SEQ, 64)
    for norm_type in ("l2", "linf"):
        projected = project_perturbation(delta, 0.1, norm_type=norm_type)
        assert projected.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# 5. fgsm_step — returns same shape as embed_input
# ---------------------------------------------------------------------------
def test_fgsm_step_shape():
    model = make_model()
    input_ids = make_input_ids()
    embeds = model.embed(input_ids).detach().requires_grad_(True)
    _, logits = forward_with_embeds(model, embeds)
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.reshape(-1, CFG.vocab_size),
        shift_labels.reshape(-1),
    )
    delta = fgsm_step(embeds, loss, alpha=0.01, norm_type="l2")
    assert delta.shape == embeds.shape


# ---------------------------------------------------------------------------
# 6. fgsm_step — Linf: values are ±alpha (sign of grad)
# ---------------------------------------------------------------------------
def test_fgsm_step_linf_values():
    model = make_model()
    input_ids = make_input_ids()
    embeds = model.embed(input_ids).detach().requires_grad_(True)
    _, logits = forward_with_embeds(model, embeds)
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.reshape(-1, CFG.vocab_size),
        shift_labels.reshape(-1),
    )
    alpha = 0.007
    delta = fgsm_step(embeds, loss, alpha=alpha, norm_type="linf")
    # Max absolute value must equal alpha (sign * alpha for non-zero grad)
    assert delta.abs().max().item() == pytest.approx(alpha, abs=1e-6)
    # Every non-zero element must be exactly ±alpha (sign output scaled by alpha)
    nonzero_vals = delta[delta != 0].abs()
    assert (nonzero_vals - alpha).abs().max().item() < 1e-6, (
        "Linf fgsm_step: non-zero elements should all have magnitude alpha"
    )


# ---------------------------------------------------------------------------
# 7. forward_with_embeds — logits shape (B, T, V)
# ---------------------------------------------------------------------------
def test_forward_with_embeds_shape():
    model = make_model()
    input_ids = make_input_ids()
    embeds = model.embed(input_ids)
    loss_ph, logits = forward_with_embeds(model, embeds)
    assert loss_ph is None
    assert logits.shape == (BATCH, SEQ, CFG.vocab_size)


# ---------------------------------------------------------------------------
# 8. forward_with_embeds — differentiable through embeds
# ---------------------------------------------------------------------------
def test_forward_with_embeds_differentiable():
    model = make_model()
    input_ids = make_input_ids()
    embeds = model.embed(input_ids).detach().requires_grad_(True)
    _, logits = forward_with_embeds(model, embeds)
    loss = logits.sum()
    loss.backward()
    assert embeds.grad is not None
    assert embeds.grad.shape == embeds.shape


# ---------------------------------------------------------------------------
# 9. FreeLBTrainer.train_step — returns required keys
# ---------------------------------------------------------------------------
def test_freelb_train_step_keys():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = AdvConfig(n_adv_steps=2)
    trainer = FreeLBTrainer(model, cfg, optimizer)
    input_ids = make_input_ids()
    result = trainer.train_step(input_ids)
    assert "loss" in result
    assert "adv_loss" in result


# ---------------------------------------------------------------------------
# 10. FreeLBTrainer.train_step — loss is finite
# ---------------------------------------------------------------------------
def test_freelb_train_step_loss_finite():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = AdvConfig(n_adv_steps=2)
    trainer = FreeLBTrainer(model, cfg, optimizer)
    input_ids = make_input_ids()
    result = trainer.train_step(input_ids)
    assert isinstance(result["loss"], float)
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert torch.isfinite(torch.tensor(result["adv_loss"]))


# ---------------------------------------------------------------------------
# 11. train_step n_adv_steps=1 works
# ---------------------------------------------------------------------------
def test_freelb_train_step_single_step():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = AdvConfig(n_adv_steps=1)
    trainer = FreeLBTrainer(model, cfg, optimizer)
    input_ids = make_input_ids()
    result = trainer.train_step(input_ids)
    assert "loss" in result and "adv_loss" in result
    assert torch.isfinite(torch.tensor(result["loss"]))


# ---------------------------------------------------------------------------
# 12. compute_robustness_score — returns float in [0, 1]
# ---------------------------------------------------------------------------
def test_robustness_score_range():
    torch.manual_seed(42)
    model = make_model()
    input_ids = make_input_ids()
    score = compute_robustness_score(model, input_ids, n_trials=3, epsilon=0.01)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 13. compute_robustness_score — same model, n_trials=2 returns stable result
# ---------------------------------------------------------------------------
def test_robustness_score_stable():
    torch.manual_seed(99)
    model = make_model()
    input_ids = make_input_ids()
    # Run twice with same seed; results should both be valid floats in range
    torch.manual_seed(7)
    s1 = compute_robustness_score(model, input_ids, n_trials=2, epsilon=0.005)
    torch.manual_seed(7)
    s2 = compute_robustness_score(model, input_ids, n_trials=2, epsilon=0.005)
    # Identical seed => identical score
    assert abs(s1 - s2) < 1e-5


# ---------------------------------------------------------------------------
# 14. Larger epsilon => lower (or equal) robustness score
# ---------------------------------------------------------------------------
def test_robustness_score_larger_epsilon():
    torch.manual_seed(123)
    model = make_model()
    input_ids = make_input_ids()

    # Use many trials and a fixed seed to get stable estimates
    scores_small = []
    scores_large = []
    for seed in range(5):
        torch.manual_seed(seed)
        scores_small.append(compute_robustness_score(model, input_ids, n_trials=4, epsilon=1e-4))
        torch.manual_seed(seed)
        scores_large.append(compute_robustness_score(model, input_ids, n_trials=4, epsilon=1.0))

    mean_small = sum(scores_small) / len(scores_small)
    mean_large = sum(scores_large) / len(scores_large)
    assert mean_small >= mean_large, (
        f"Expected small epsilon ({mean_small:.4f}) >= large epsilon ({mean_large:.4f})"
    )
