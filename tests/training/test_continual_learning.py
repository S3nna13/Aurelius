"""
Tests for src/training/continual_learning.py

Tiny configs: D=8, VOCAB=16, B=4
Model: nn.Linear(D, VOCAB)
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.continual_learning import (
    CLConfig,
    ContinualTrainer,
    EWCRegularizer,
    ExperienceReplayBuffer,
    clone_model,
    compute_distillation_loss,
    compute_fisher_information,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

D = 8
VOCAB = 16
B = 4


def make_model() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(D, VOCAB)


def make_xy():
    x = torch.randn(B, D)
    y = torch.randint(0, VOCAB, (B,))
    return x, y


def ce_loss(x, y):
    model = make_model()
    return F.cross_entropy(model(x), y)


def make_simple_loss(model):
    def loss_fn(x, y):
        return F.cross_entropy(model(x), y)

    return loss_fn


def make_fisher(model) -> dict:
    """Quick fisher using a single batch."""
    x, y = make_xy()
    loader = [(x, y)]
    loss_fn = make_simple_loss(model)
    return compute_fisher_information(model, loader, loss_fn, n_samples=1)


def ref_params_from(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# ---------------------------------------------------------------------------
# 1. CLConfig defaults
# ---------------------------------------------------------------------------


def test_clconfig_defaults():
    cfg = CLConfig()
    assert cfg.ewc_lambda == 5000.0
    assert cfg.replay_buffer_size == 1000
    assert cfg.n_replay_per_step == 16
    assert cfg.distill_alpha == 0.5
    assert cfg.method == "ewc"


# ---------------------------------------------------------------------------
# 2. EWCRegularizer.penalty is a scalar and finite
# ---------------------------------------------------------------------------


def test_ewc_penalty_is_scalar_finite():
    model = make_model()
    fisher = make_fisher(model)
    ref = ref_params_from(model)
    ewc = EWCRegularizer(model, fisher, ref, lam=5000.0)
    penalty = ewc.penalty(model)
    assert penalty.dim() == 0, "penalty must be scalar"
    assert torch.isfinite(penalty), "penalty must be finite"


# ---------------------------------------------------------------------------
# 3. EWC penalty is zero when params unchanged
# ---------------------------------------------------------------------------


def test_ewc_penalty_zero_when_unchanged():
    model = make_model()
    fisher = make_fisher(model)
    ref = ref_params_from(model)
    ewc = EWCRegularizer(model, fisher, ref, lam=5000.0)
    penalty = ewc.penalty(model)
    assert penalty.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. EWC penalty > 0 when params changed
# ---------------------------------------------------------------------------


def test_ewc_penalty_positive_when_params_changed():
    model = make_model()
    fisher = make_fisher(model)
    ref = ref_params_from(model)
    ewc = EWCRegularizer(model, fisher, ref, lam=5000.0)

    # Perturb parameters
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 0.5)

    penalty = ewc.penalty(model)
    assert penalty.item() > 0.0, "penalty must be positive after param change"


# ---------------------------------------------------------------------------
# 5. ExperienceReplayBuffer add / len
# ---------------------------------------------------------------------------


def test_replay_buffer_add_len():
    buf = ExperienceReplayBuffer(max_size=100)
    assert len(buf) == 0
    x, y = make_xy()
    buf.add(x, y)
    assert len(buf) == B


# ---------------------------------------------------------------------------
# 6. Replay buffer FIFO eviction at max_size
# ---------------------------------------------------------------------------


def test_replay_buffer_fifo_eviction():
    buf = ExperienceReplayBuffer(max_size=B)
    x1 = torch.zeros(B, D)
    y1 = torch.zeros(B, dtype=torch.long)
    buf.add(x1, y1)

    assert len(buf) == B

    x2 = torch.ones(B, D) * 99.0
    y2 = torch.ones(B, dtype=torch.long)
    buf.add(x2, y2)

    # Buffer should still be capped at max_size
    assert len(buf) == B

    # All samples in buffer should be the newest ones (value 99.0)
    xs, _ = buf.sample(B)
    assert (xs == 99.0).all(), "Buffer should contain only newest samples after eviction"


# ---------------------------------------------------------------------------
# 7. sample returns <= n items
# ---------------------------------------------------------------------------


def test_replay_buffer_sample_at_most_n():
    buf = ExperienceReplayBuffer(max_size=100)
    x, y = make_xy()
    buf.add(x, y)  # adds B=4 items

    xs, ys = buf.sample(2)
    assert xs.shape[0] == 2
    assert ys.shape[0] == 2

    # Requesting more than available returns all
    xs_all, ys_all = buf.sample(100)
    assert xs_all.shape[0] == B


# ---------------------------------------------------------------------------
# 8. clone_model produces same parameter values
# ---------------------------------------------------------------------------


def test_clone_model_same_params():
    model = make_model()
    cloned = clone_model(model)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), cloned.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2), f"Param {n1} differs after cloning"


# ---------------------------------------------------------------------------
# 9. clone_model is an independent copy
# ---------------------------------------------------------------------------


def test_clone_model_independence():
    model = make_model()
    cloned = clone_model(model)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 10.0)

    for (_, p_orig), (_, p_clone) in zip(model.named_parameters(), cloned.named_parameters()):
        assert not torch.allclose(p_orig, p_clone), (
            "Clone should not be affected by original mutation"
        )


# ---------------------------------------------------------------------------
# 10. compute_distillation_loss is scalar
# ---------------------------------------------------------------------------


def test_distillation_loss_is_scalar():
    student = torch.randn(B, VOCAB)
    teacher = torch.randn(B, VOCAB)
    loss = compute_distillation_loss(student, teacher)
    assert loss.dim() == 0, "distillation loss must be scalar"
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 11. distillation_loss non-negative
# ---------------------------------------------------------------------------


def test_distillation_loss_non_negative():
    student = torch.randn(B, VOCAB)
    teacher = torch.randn(B, VOCAB)
    loss = compute_distillation_loss(student, teacher)
    assert loss.item() >= 0.0, "KL divergence must be non-negative"


# ---------------------------------------------------------------------------
# 12. ContinualTrainer.train_step returns correct keys
# ---------------------------------------------------------------------------


def test_trainer_train_step_keys():
    model = make_model()
    cfg = CLConfig(method="none")
    trainer = ContinualTrainer(model, cfg)
    x, y = make_xy()
    result = trainer.train_step(x, y, make_simple_loss(model))
    assert set(result.keys()) == {"loss", "task_loss", "reg_loss"}


# ---------------------------------------------------------------------------
# 13. train_step loss is finite
# ---------------------------------------------------------------------------


def test_trainer_train_step_loss_finite():
    model = make_model()
    cfg = CLConfig(method="none")
    trainer = ContinualTrainer(model, cfg)
    x, y = make_xy()
    result = trainer.train_step(x, y, make_simple_loss(model))
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["task_loss"])
    assert math.isfinite(result["reg_loss"])


# ---------------------------------------------------------------------------
# 14. register_task stores params
# ---------------------------------------------------------------------------


def test_register_task_stores_params():
    model = make_model()
    cfg = CLConfig(method="ewc")
    trainer = ContinualTrainer(model, cfg)

    assert trainer._ref_params is None
    fisher = make_fisher(model)
    trainer.register_task(task_id=0, fisher=fisher)

    assert trainer._ref_params is not None
    for name, param in model.named_parameters():
        assert name in trainer._ref_params
        assert torch.allclose(trainer._ref_params[name], param.detach())


# ---------------------------------------------------------------------------
# 15. get_forgetting is non-negative
# ---------------------------------------------------------------------------


def test_get_forgetting_non_negative():
    model = make_model()
    cfg = CLConfig(method="none")
    trainer = ContinualTrainer(model, cfg)

    ref = ref_params_from(model)
    forgetting = trainer.get_forgetting(model, ref)
    assert forgetting >= 0.0

    # Perturb and check forgetting increases
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))

    forgetting_after = trainer.get_forgetting(model, ref)
    assert forgetting_after > forgetting
