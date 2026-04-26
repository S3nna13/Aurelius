"""Tests for the continual learning orchestrator."""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.continual import (
    ContinualConfig,
    ContinualReport,
    ContinualTrainer,
    TaskRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def _make_loader(n: int = 16, seq_len: int = 16, batch_size: int = 4) -> DataLoader:
    ids = torch.randint(0, 256, (n, seq_len))

    def collate(batch):
        b = torch.stack([x[0] for x in batch])
        return {"input_ids": b, "labels": b}

    return DataLoader(TensorDataset(ids), batch_size=batch_size, collate_fn=collate)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_continual_config_defaults():
    cfg = ContinualConfig()
    assert cfg.base_ewc_lambda == 5000.0
    assert cfg.lambda_growth == 1.0
    assert cfg.n_fisher_samples == 200
    assert cfg.steps_per_task == 100
    assert cfg.lr == 1e-4
    assert cfg.forgetting_threshold == 0.1


def test_ewc_lambda_for_task_growth(small_model):
    cfg = ContinualConfig(base_ewc_lambda=1000.0, lambda_growth=2.0)
    trainer = ContinualTrainer(small_model, cfg)

    assert trainer.ewc_lambda_for_task(0) == pytest.approx(1000.0)
    assert trainer.ewc_lambda_for_task(1) == pytest.approx(2000.0)
    assert trainer.ewc_lambda_for_task(2) == pytest.approx(4000.0)


def test_ewc_lambda_constant_growth_1(small_model):
    cfg = ContinualConfig(base_ewc_lambda=5000.0, lambda_growth=1.0)
    trainer = ContinualTrainer(small_model, cfg)

    for i in range(5):
        assert trainer.ewc_lambda_for_task(i) == pytest.approx(5000.0)


def test_train_first_task_returns_record(small_model):
    cfg = ContinualConfig(steps_per_task=2, n_fisher_samples=4)
    trainer = ContinualTrainer(small_model, cfg)
    loader = _make_loader()

    record = trainer.train_task("task_1", loader)

    assert isinstance(record, TaskRecord)
    assert record.task_id == "task_1"
    assert record.n_steps == 2


def test_train_task_fisher_computed(small_model):
    cfg = ContinualConfig(steps_per_task=2, n_fisher_samples=4)
    trainer = ContinualTrainer(small_model, cfg)
    loader = _make_loader()

    record = trainer.train_task("task_1", loader)
    assert record.fisher_computed is True


def test_train_two_tasks_no_crash(small_model):
    cfg = ContinualConfig(steps_per_task=2, n_fisher_samples=4)
    trainer = ContinualTrainer(small_model, cfg)
    loader1 = _make_loader()
    loader2 = _make_loader()

    trainer.train_task("task_1", loader1)
    trainer.train_task("task_2", loader2)  # should not raise


def test_get_report_task_count(small_model):
    cfg = ContinualConfig(steps_per_task=2, n_fisher_samples=4)
    trainer = ContinualTrainer(small_model, cfg)
    loader1 = _make_loader()
    loader2 = _make_loader()

    trainer.train_task("task_1", loader1)
    trainer.train_task("task_2", loader2)

    report = trainer.get_report()
    assert isinstance(report, ContinualReport)
    assert report.n_tasks_completed == 2


def test_evaluate_returns_finite(small_model):
    cfg = ContinualConfig(steps_per_task=2, n_fisher_samples=4)
    trainer = ContinualTrainer(small_model, cfg)
    loader = _make_loader()

    result = trainer.evaluate(loader)
    assert math.isfinite(result)
    assert result > 0.0


def test_continual_report_any_forgetting_false(small_model):
    """With random data and tiny training, forgetting should not be flagged
    (both tasks see similar random-data loss)."""
    torch.manual_seed(0)
    cfg = ContinualConfig(steps_per_task=2, n_fisher_samples=4)
    trainer = ContinualTrainer(small_model, cfg)
    loader1 = _make_loader()
    loader2 = _make_loader()

    trainer.train_task("task_1", loader1, loader1)
    trainer.train_task("task_2", loader2, loader2)

    report = trainer.get_report()
    # With symmetric random data and 2 steps the eval losses should be similar
    assert not report.any_forgetting
