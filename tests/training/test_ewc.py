"""Tests for Elastic Weight Consolidation (EWC)."""
from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.ewc import (
    EWCConfig,
    EWCTrainer,
    TaskSequence,
    compute_fisher_diagonal,
    ewc_penalty,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 256
SEQ_LEN = 16
BATCH = 2


def _make_model() -> AureliusTransformer:
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def _data_iter(n_batches: int = 5):
    """Yield random input_ids tensors."""
    torch.manual_seed(0)
    for _ in range(n_batches):
        yield torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


def _make_trainer(model):
    cfg = EWCConfig(ewc_lambda=1000.0, n_fisher_samples=3)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    return EWCTrainer(model, cfg, opt)


# ---------------------------------------------------------------------------
# 1. EWCConfig defaults
# ---------------------------------------------------------------------------

def test_ewcconfig_default_lambda():
    cfg = EWCConfig()
    assert cfg.ewc_lambda == 1000.0


def test_ewcconfig_default_n_fisher_samples():
    cfg = EWCConfig()
    assert cfg.n_fisher_samples == 200


# ---------------------------------------------------------------------------
# 2-5. compute_fisher_diagonal
# ---------------------------------------------------------------------------

def test_compute_fisher_returns_dict():
    model = _make_model()
    fisher = compute_fisher_diagonal(model, _data_iter(), n_samples=3)
    assert isinstance(fisher, dict)
    assert len(fisher) > 0


def test_compute_fisher_keys_match_param_names():
    model = _make_model()
    fisher = compute_fisher_diagonal(model, _data_iter(), n_samples=3)
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(fisher.keys()) == expected


def test_compute_fisher_values_nonnegative():
    model = _make_model()
    fisher = compute_fisher_diagonal(model, _data_iter(), n_samples=3)
    for name, f in fisher.items():
        assert (f >= 0).all(), f"Negative Fisher values for {name}"


def test_compute_fisher_shapes_match_params():
    model = _make_model()
    fisher = compute_fisher_diagonal(model, _data_iter(), n_samples=3)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assert fisher[name].shape == param.shape, (
            f"Shape mismatch for {name}: fisher={fisher[name].shape}, param={param.shape}"
        )


# ---------------------------------------------------------------------------
# 6-9. ewc_penalty
# ---------------------------------------------------------------------------

def test_ewc_penalty_returns_scalar():
    model = _make_model()
    fisher = compute_fisher_diagonal(model, _data_iter(), n_samples=3)
    optimal = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    penalty = ewc_penalty(model, fisher, optimal, ewc_lambda=1000.0)
    assert penalty.shape == torch.Size([])  # scalar


def test_ewc_penalty_zero_when_params_unchanged():
    model = _make_model()
    fisher = compute_fisher_diagonal(model, _data_iter(), n_samples=3)
    # optimal == current params -> penalty should be ~0
    optimal = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    penalty = ewc_penalty(model, fisher, optimal, ewc_lambda=1000.0)
    assert abs(penalty.item()) < 1e-5


def test_ewc_penalty_positive_when_params_differ():
    model = _make_model()
    fisher = compute_fisher_diagonal(model, _data_iter(), n_samples=3)
    optimal = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    # Perturb model parameters
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    penalty = ewc_penalty(model, fisher, optimal, ewc_lambda=1000.0)
    assert penalty.item() > 0


def test_ewc_penalty_scales_with_lambda():
    model = _make_model()
    fisher = compute_fisher_diagonal(model, _data_iter(), n_samples=3)
    optimal = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    # Perturb so penalty is non-zero
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    p1 = ewc_penalty(model, fisher, optimal, ewc_lambda=100.0).item()
    p2 = ewc_penalty(model, fisher, optimal, ewc_lambda=200.0).item()
    assert abs(p2 - 2 * p1) < 1e-3 * abs(p2), (
        f"Penalty should scale linearly with lambda: p1={p1}, p2={p2}"
    )


# ---------------------------------------------------------------------------
# 10-14. EWCTrainer
# ---------------------------------------------------------------------------

def test_ewctrainer_starts_not_consolidated():
    model = _make_model()
    trainer = _make_trainer(model)
    assert not trainer.is_consolidated()


def test_ewctrainer_consolidate_sets_flag():
    model = _make_model()
    trainer = _make_trainer(model)
    trainer.consolidate(_data_iter(n_batches=3))
    assert trainer.is_consolidated()


def test_ewctrainer_train_step_before_consolidation_ewc_zero():
    model = _make_model()
    trainer = _make_trainer(model)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    result = trainer.train_step(input_ids)
    assert result["ewc_loss"] == 0.0


def test_ewctrainer_train_step_after_consolidation_has_all_keys():
    model = _make_model()
    trainer = _make_trainer(model)
    trainer.consolidate(_data_iter(n_batches=3))

    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    result = trainer.train_step(input_ids)

    assert "task_loss" in result
    assert "ewc_loss" in result
    assert "total_loss" in result


def test_ewctrainer_total_loss_equals_sum():
    model = _make_model()
    trainer = _make_trainer(model)
    trainer.consolidate(_data_iter(n_batches=3))

    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    result = trainer.train_step(input_ids)

    expected = result["task_loss"] + result["ewc_loss"]
    assert abs(result["total_loss"] - expected) < 1e-5, (
        f"total_loss={result['total_loss']} != task_loss + ewc_loss={expected}"
    )


# ---------------------------------------------------------------------------
# 15-16. TaskSequence
# ---------------------------------------------------------------------------

def test_task_sequence_starts_empty():
    ts = TaskSequence()
    assert len(ts) == 0


def test_task_sequence_add_get_roundtrip():
    ts = TaskSequence()
    data = [torch.randint(0, VOCAB, (BATCH, SEQ_LEN)) for _ in range(4)]
    ts.add_task("task_a", data)

    assert len(ts) == 1
    assert ts.task_names() == ["task_a"]

    retrieved = ts.get_task("task_a")
    assert len(retrieved) == len(data)
    for orig, ret in zip(data, retrieved):
        assert torch.equal(orig, ret)


def test_task_sequence_get_missing_raises():
    ts = TaskSequence()
    with pytest.raises(KeyError):
        ts.get_task("nonexistent")


def test_task_sequence_multiple_tasks():
    ts = TaskSequence()
    ts.add_task("a", [torch.zeros(2, 4)])
    ts.add_task("b", [torch.ones(2, 4)])
    ts.add_task("c", [torch.full((2, 4), 2)])

    assert len(ts) == 3
    assert ts.task_names() == ["a", "b", "c"]
    assert torch.equal(ts.get_task("b")[0], torch.ones(2, 4))
