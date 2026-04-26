"""Tests for context distillation implementation (Askell et al. 2021)."""

import pytest
import torch

from src.alignment.context_distillation import (
    ContextDistillationConfig,
    ContextDistillationTrainer,
    compute_student_logits,
    compute_teacher_logits,
    context_distillation_loss,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=4,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(42)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def system_prompt_ids():
    torch.manual_seed(0)
    return torch.randint(1, 256, (5,))


@pytest.fixture
def message_ids():
    torch.manual_seed(1)
    return torch.randint(1, 256, (8,))


@pytest.fixture
def response_ids():
    torch.manual_seed(2)
    return torch.randint(1, 256, (6,))


@pytest.fixture
def cd_cfg():
    return ContextDistillationConfig(
        temperature=2.0,
        kl_weight=1.0,
        ce_weight=0.0,
        lr=1e-5,
        freeze_except_last_n=4,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_teacher_logits_shape(small_model, system_prompt_ids, message_ids, response_ids):
    """compute_teacher_logits must return (S_resp, V)."""
    logits = compute_teacher_logits(small_model, system_prompt_ids, message_ids, response_ids)
    V = small_model.config.vocab_size
    S_resp = response_ids.shape[0]
    assert logits.shape == (S_resp, V), f"Expected ({S_resp}, {V}), got {logits.shape}"


def test_compute_student_logits_shape(small_model, message_ids, response_ids):
    """compute_student_logits must return (S_resp, V)."""
    logits = compute_student_logits(small_model, message_ids, response_ids)
    V = small_model.config.vocab_size
    S_resp = response_ids.shape[0]
    assert logits.shape == (S_resp, V), f"Expected ({S_resp}, {V}), got {logits.shape}"


def test_teacher_student_logits_differ(small_model, system_prompt_ids, message_ids, response_ids):
    """Teacher and student logits must differ due to different context."""
    teacher = compute_teacher_logits(small_model, system_prompt_ids, message_ids, response_ids)
    student = compute_student_logits(small_model, message_ids, response_ids)
    # They come from different sequence positions/contexts, so should differ
    assert not torch.allclose(teacher, student), (
        "Teacher and student logits should differ (different context lengths)"
    )


def test_context_distillation_loss_scalar(
    small_model, system_prompt_ids, message_ids, response_ids, cd_cfg
):
    """context_distillation_loss must return a scalar tensor."""
    teacher_logits = compute_teacher_logits(
        small_model, system_prompt_ids, message_ids, response_ids
    )
    student_logits = compute_student_logits(small_model, message_ids, response_ids)
    loss, _ = context_distillation_loss(student_logits, teacher_logits, response_ids, cd_cfg)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


def test_context_distillation_loss_finite(
    small_model, system_prompt_ids, message_ids, response_ids, cd_cfg
):
    """context_distillation_loss must be finite."""
    teacher_logits = compute_teacher_logits(
        small_model, system_prompt_ids, message_ids, response_ids
    )
    student_logits = compute_student_logits(small_model, message_ids, response_ids)
    loss, _ = context_distillation_loss(student_logits, teacher_logits, response_ids, cd_cfg)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_context_distillation_loss_metrics(
    small_model, system_prompt_ids, message_ids, response_ids, cd_cfg
):
    """context_distillation_loss must return a dict with kl, ce, total keys."""
    teacher_logits = compute_teacher_logits(
        small_model, system_prompt_ids, message_ids, response_ids
    )
    student_logits = compute_student_logits(small_model, message_ids, response_ids)
    _, metrics = context_distillation_loss(student_logits, teacher_logits, response_ids, cd_cfg)
    assert "kl" in metrics, "Missing 'kl' key in metrics"
    assert "ce" in metrics, "Missing 'ce' key in metrics"
    assert "total" in metrics, "Missing 'total' key in metrics"
    assert isinstance(metrics["kl"], float)
    assert isinstance(metrics["ce"], float)
    assert isinstance(metrics["total"], float)


def test_train_step_returns_metrics(
    small_model, system_prompt_ids, message_ids, response_ids, cd_cfg
):
    """train_step() must return finite metrics with kl, ce, total keys."""
    trainer = ContextDistillationTrainer(small_model, system_prompt_ids, cd_cfg)
    metrics = trainer.train_step(message_ids, response_ids)
    assert "kl" in metrics
    assert "ce" in metrics
    assert "total" in metrics
    for key, val in metrics.items():
        assert isinstance(val, float), f"metrics[{key!r}] should be float, got {type(val)}"
        assert torch.isfinite(torch.tensor(val)), f"metrics[{key!r}] = {val} is not finite"


def test_train_step_updates_weights(
    small_model, system_prompt_ids, message_ids, response_ids, cd_cfg
):
    """At least one weight must change after a train_step."""
    # Capture a snapshot of the last layer's parameters before training
    last_layer = list(small_model.layers)[-1]
    params_before = {name: p.clone().detach() for name, p in last_layer.named_parameters()}

    trainer = ContextDistillationTrainer(small_model, system_prompt_ids, cd_cfg)
    trainer.train_step(message_ids, response_ids)

    changed = False
    for name, p in last_layer.named_parameters():
        if not torch.allclose(params_before[name], p.detach()):
            changed = True
            break

    assert changed, "No weights changed after train_step — optimizer may not be working"


def test_freeze_layers_config(small_model, system_prompt_ids):
    """With freeze_except_last_n=1, only the last layer + lm_head should be in optimizer."""
    cfg = ContextDistillationConfig(freeze_except_last_n=1, lr=1e-5)
    trainer = ContextDistillationTrainer(small_model, system_prompt_ids, cfg)

    # Collect all parameters in optimizer param groups
    optimizer_param_ids = set()
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            optimizer_param_ids.add(id(p))

    n_layers = len(list(small_model.layers))
    # Layers 0..n_layers-2 should NOT be in optimizer
    for i, layer in enumerate(small_model.layers):
        if i < n_layers - 1:
            for p in layer.parameters():
                assert id(p) not in optimizer_param_ids, (
                    f"Layer {i} parameter is in optimizer but should be frozen"
                )
    # Last layer SHOULD be in optimizer
    last_layer = list(small_model.layers)[-1]
    last_layer_params = list(last_layer.parameters())
    assert len(last_layer_params) > 0
    for p in last_layer_params:
        assert id(p) in optimizer_param_ids, (
            "Last layer parameter should be in optimizer but is not"
        )


def test_config_defaults():
    """ContextDistillationConfig must have temperature=2.0 and kl_weight=1.0 by default."""
    cfg = ContextDistillationConfig()
    assert cfg.temperature == 2.0, f"Expected temperature=2.0, got {cfg.temperature}"
    assert cfg.kl_weight == 1.0, f"Expected kl_weight=1.0, got {cfg.kl_weight}"
