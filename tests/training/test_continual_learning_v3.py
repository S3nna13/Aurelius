"""Tests for continual_learning_v3.py

Covers:
  - FisherInformationEstimator
  - EWCRegularizer
  - OnlineEWC
  - PackNetManager
  - ProgressiveNNColumn
  - ContinualTrainer
  - ContinualConfig

Uses a tiny inline model — no other Aurelius files imported.
Dimensions: d_model=16, n_tasks=2, seq_len=4, batch=2, n_steps=3.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn

from src.training.continual_learning_v3 import (
    ContinualConfig,
    ContinualTrainer,
    EWCRegularizer,
    FisherInformationEstimator,
    OnlineEWC,
    PackNetManager,
    ProgressiveNNColumn,
)

# ---------------------------------------------------------------------------
# Shared constants and tiny model
# ---------------------------------------------------------------------------

D_MODEL = 16
N_TASKS = 2
SEQ_LEN = 4
BATCH = 2
N_STEPS = 3


class TinyModel(nn.Module):
    """Simple two-layer linear model for testing.

    Accepts [B, T, D] and returns [B, T, D].
    """

    def __init__(self, d: int = D_MODEL) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _make_model() -> TinyModel:
    torch.manual_seed(0)
    return TinyModel(D_MODEL)


def _make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a random (inputs, targets) pair with shape [BATCH, SEQ_LEN, D_MODEL]."""
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    y = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    return x, y


def _dataloader_fn() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yields a fixed number of batches."""
    for _ in range(5):
        yield _make_batch()


def _data_fn():
    """Zero-argument callable returning batches for ContinualTrainer."""
    return _dataloader_fn()


# ---------------------------------------------------------------------------
# FisherInformationEstimator tests
# ---------------------------------------------------------------------------


def test_fisher_returns_dict_matching_param_names():
    """compute_diagonal_fisher keys match model.named_parameters() names."""
    model = _make_model()
    estimator = FisherInformationEstimator(model, n_samples=10)
    criterion = nn.MSELoss()
    fisher = estimator.compute_diagonal_fisher(_dataloader_fn, criterion)

    expected_names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(fisher.keys()) == expected_names


def test_fisher_values_non_negative():
    """All Fisher diagonal entries must be >= 0."""
    model = _make_model()
    estimator = FisherInformationEstimator(model, n_samples=20)
    criterion = nn.MSELoss()
    fisher = estimator.compute_diagonal_fisher(_dataloader_fn, criterion)

    for name, f in fisher.items():
        assert (f >= 0).all(), f"Negative Fisher value for param '{name}'"


def test_fisher_shapes_match_params():
    """Each Fisher tensor has the same shape as its parameter."""
    model = _make_model()
    estimator = FisherInformationEstimator(model, n_samples=10)
    criterion = nn.MSELoss()
    fisher = estimator.compute_diagonal_fisher(_dataloader_fn, criterion)

    param_shapes = {name: p.shape for name, p in model.named_parameters()}
    for name, f in fisher.items():
        assert f.shape == param_shapes[name], (
            f"Shape mismatch for '{name}': fisher={f.shape} param={param_shapes[name]}"
        )


# ---------------------------------------------------------------------------
# EWCRegularizer tests
# ---------------------------------------------------------------------------


def test_ewc_consolidate_stores_theta_star():
    """After consolidate, _anchors contains one entry with theta_star."""
    model = _make_model()
    ewc = EWCRegularizer(model, lambda_ewc=100.0)
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters() if p.requires_grad}
    assert len(ewc._anchors) == 0
    ewc.consolidate(fisher)
    assert len(ewc._anchors) == 1

    theta_star, _ = ewc._anchors[0]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert name in theta_star
        assert theta_star[name].shape == p.shape


def test_ewc_penalty_zero_when_params_unchanged():
    """Penalty should be 0 immediately after consolidate (params unchanged)."""
    model = _make_model()
    ewc = EWCRegularizer(model, lambda_ewc=1000.0)
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters() if p.requires_grad}
    ewc.consolidate(fisher)
    p = ewc.penalty(model)
    assert p.item() < 1e-6, f"Expected ~0 penalty, got {p.item()}"


def test_ewc_penalty_positive_after_param_update():
    """Penalty should be > 0 after model parameters are modified."""
    model = _make_model()
    ewc = EWCRegularizer(model, lambda_ewc=1000.0)
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters() if p.requires_grad}
    ewc.consolidate(fisher)

    # Perturb all parameters.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))

    penalty_val = ewc.penalty(model)
    assert penalty_val.item() > 0, "Penalty should be positive after param update"


def test_ewc_loss_geq_task_loss():
    """ewc_loss >= task_loss (penalty is non-negative)."""
    model = _make_model()
    ewc = EWCRegularizer(model, lambda_ewc=1.0)
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters() if p.requires_grad}
    ewc.consolidate(fisher)

    # Perturb to ensure non-zero penalty.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))

    task_loss = torch.tensor(0.5)
    total = ewc.ewc_loss(task_loss, model)
    assert total.item() >= task_loss.item()


def test_ewc_penalty_zero_before_consolidate():
    """Before any consolidation, penalty returns 0."""
    model = _make_model()
    ewc = EWCRegularizer(model)
    p = ewc.penalty(model)
    assert p.item() == 0.0


# ---------------------------------------------------------------------------
# OnlineEWC tests
# ---------------------------------------------------------------------------


def _make_fisher(model: nn.Module, value: float = 1.0) -> dict:
    return {
        name: torch.full_like(p, value) for name, p in model.named_parameters() if p.requires_grad
    }


def test_online_ewc_consolidate_sets_running_fisher():
    """After consolidate_online, _running_fisher is populated."""
    model = _make_model()
    online = OnlineEWC(model, gamma=1.0)
    assert online._running_fisher is None

    fisher = _make_fisher(model, 2.0)
    online.consolidate_online(fisher, task_id=0)

    assert online._running_fisher is not None
    for name, f in online._running_fisher.items():
        assert (f == 2.0).all()


def test_online_ewc_gamma_decay():
    """Running Fisher blends old and new via gamma."""
    model = _make_model()
    gamma = 0.8
    online = OnlineEWC(model, gamma=gamma)

    f1 = _make_fisher(model, 4.0)
    online.consolidate_online(f1, task_id=0)

    f2 = _make_fisher(model, 2.0)
    online.consolidate_online(f2, task_id=1)

    expected = gamma * 4.0 + (1.0 - gamma) * 2.0  # 3.6
    for name, f in online._running_fisher.items():
        assert abs(f.mean().item() - expected) < 1e-5, (
            f"Expected {expected}, got {f.mean().item()} for '{name}'"
        )


def test_online_ewc_penalty_zero_before_consolidate():
    """OnlineEWC penalty returns 0 when no consolidation has been done."""
    model = _make_model()
    online = OnlineEWC(model)
    assert online.penalty(model).item() == 0.0


# ---------------------------------------------------------------------------
# PackNetManager tests
# ---------------------------------------------------------------------------


def test_packnet_prune_creates_mask():
    """prune_for_task creates a mask entry for the given task."""
    model = _make_model()
    pn = PackNetManager(model, prune_fraction=0.5)
    assert 0 not in pn.task_masks

    pn.prune_for_task(0)
    assert 0 in pn.task_masks

    # Mask should cover all trainable params.
    expected_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(pn.task_masks[0].keys()) == expected_names


def test_packnet_mask_coverage_approx_prune_fraction():
    """About (1 - prune_fraction) of weights should be kept per task."""
    torch.manual_seed(42)
    model = _make_model()
    prune_fraction = 0.5
    pn = PackNetManager(model, prune_fraction=prune_fraction)
    pn.prune_for_task(0)

    total_params = 0
    kept_params = 0
    for name, mask in pn.task_masks[0].items():
        total_params += mask.numel()
        kept_params += mask.sum().item()

    fraction_kept = kept_params / total_params
    # Expect ~50% kept, allow ±15% tolerance.
    assert abs(fraction_kept - (1.0 - prune_fraction)) < 0.15, (
        f"Fraction kept = {fraction_kept:.3f}, expected ~{1.0 - prune_fraction:.3f}"
    )


def test_packnet_apply_masks_zeroes_non_task_params():
    """apply_masks should zero all weights not in the task mask."""
    torch.manual_seed(7)
    model = _make_model()
    pn = PackNetManager(model, prune_fraction=0.5)
    pn.prune_for_task(0)

    # Apply masks — only task-0 weights should survive.
    pn.apply_masks(model, task_id=0)

    for name, param in model.named_parameters():
        mask = pn.task_masks[0].get(name)
        if mask is None:
            continue
        # Positions outside the mask should be zero.
        outside_mask_values = param.detach()[~mask]
        assert (outside_mask_values == 0).all(), (
            f"Non-zero values outside task mask for param '{name}'"
        )


def test_packnet_freeze_and_unfreeze():
    """freeze_task_weights adds hooks; unfreeze_all removes them."""
    model = _make_model()
    pn = PackNetManager(model, prune_fraction=0.5)
    pn.prune_for_task(0)

    assert len(pn._hook_handles) == 0
    pn.freeze_task_weights(0)
    assert len(pn._hook_handles) > 0

    pn.unfreeze_all()
    assert len(pn._hook_handles) == 0


# ---------------------------------------------------------------------------
# ProgressiveNNColumn tests
# ---------------------------------------------------------------------------


def test_progressive_output_shape():
    """forward returns [B, T, d_model]."""
    model = ProgressiveNNColumn(D_MODEL, N_TASKS)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = model(x, task_id=0)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
        f"Expected {(BATCH, SEQ_LEN, D_MODEL)}, got {out.shape}"
    )


def test_progressive_second_task_output_shape():
    """Task-1 forward also returns [B, T, d_model]."""
    model = ProgressiveNNColumn(D_MODEL, N_TASKS)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = model(x, task_id=1)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_progressive_different_tasks_different_outputs():
    """Different task_ids should produce different outputs for the same input."""
    torch.manual_seed(1)
    model = ProgressiveNNColumn(D_MODEL, N_TASKS)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    out0 = model(x, task_id=0)
    out1 = model(x, task_id=1)

    assert not torch.allclose(out0, out1), "Task-0 and task-1 should produce different outputs"


# ---------------------------------------------------------------------------
# ContinualTrainer tests
# ---------------------------------------------------------------------------


def test_trainer_train_task_returns_loss_list():
    """train_task should return a list of floats with length == n_steps."""
    model = _make_model()
    trainer = ContinualTrainer(model, strategy="naive", lr=1e-3)
    losses = trainer.train_task(task_id=0, data_fn=_data_fn, n_steps=N_STEPS)

    assert isinstance(losses, list)
    assert len(losses) == N_STEPS
    assert all(isinstance(v, float) for v in losses)


def test_trainer_naive_runs_without_ewc_packnet():
    """'naive' strategy should not initialise EWC or PackNet."""
    model = _make_model()
    trainer = ContinualTrainer(model, strategy="naive", lr=1e-3)

    assert trainer.ewc is None
    assert trainer.packnet is None

    losses = trainer.train_task(task_id=0, data_fn=_data_fn, n_steps=N_STEPS)
    assert len(losses) == N_STEPS


def test_trainer_ewc_strategy_runs():
    """'ewc' strategy should complete without errors across two tasks."""
    model = _make_model()
    trainer = ContinualTrainer(model, strategy="ewc", lr=1e-3)

    losses0 = trainer.train_task(task_id=0, data_fn=_data_fn, n_steps=N_STEPS)
    losses1 = trainer.train_task(task_id=1, data_fn=_data_fn, n_steps=N_STEPS)

    assert len(losses0) == N_STEPS
    assert len(losses1) == N_STEPS
    # After task 0, EWC should have one anchor.
    assert trainer.ewc is not None
    assert len(trainer.ewc._anchors) >= 1


def test_trainer_packnet_strategy_runs():
    """'packnet' strategy should complete without errors."""
    model = _make_model()
    trainer = ContinualTrainer(model, strategy="packnet", lr=1e-3)
    losses = trainer.train_task(task_id=0, data_fn=_data_fn, n_steps=N_STEPS)

    assert len(losses) == N_STEPS
    assert trainer.packnet is not None
    assert 0 in trainer.packnet.task_masks


def test_trainer_evaluate_forgetting_returns_float():
    """evaluate_forgetting returns a float."""
    model = _make_model()
    trainer = ContinualTrainer(model, strategy="naive", lr=1e-3)
    trainer.train_task(task_id=0, data_fn=_data_fn, n_steps=N_STEPS)

    def _metric(m: nn.Module) -> float:
        x, y = _make_batch()
        out = m(x)
        return -nn.MSELoss()(out, y).item()

    result = trainer.evaluate_forgetting(task_id=0, metric_fn=_metric)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# ContinualConfig tests
# ---------------------------------------------------------------------------


def test_continual_config_defaults():
    """ContinualConfig should have the correct default values."""
    cfg = ContinualConfig()

    assert cfg.lambda_ewc == 1000.0
    assert cfg.gamma == 0.9
    assert cfg.prune_fraction == 0.5
    assert cfg.n_tasks == 3
    assert cfg.strategy == "ewc"


def test_continual_config_custom_values():
    """ContinualConfig should accept custom values."""
    cfg = ContinualConfig(
        lambda_ewc=500.0, gamma=0.5, prune_fraction=0.3, n_tasks=5, strategy="packnet"
    )

    assert cfg.lambda_ewc == 500.0
    assert cfg.gamma == 0.5
    assert cfg.prune_fraction == 0.3
    assert cfg.n_tasks == 5
    assert cfg.strategy == "packnet"
