"""Tests for structured pruning."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.pruning import (
    PruningConfig,
    compute_ffn_importance,
    prune_model,
)


@pytest.fixture
def small_model():
    torch.manual_seed(0)
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


def _make_loader(n=8, seq_len=16):
    ids = torch.randint(0, 256, (n, seq_len))

    def collate(batch):
        b = torch.stack([x[0] for x in batch])
        return {"input_ids": b, "labels": b}

    return DataLoader(TensorDataset(ids), batch_size=4, collate_fn=collate)


def test_compute_importance_shape(small_model):
    """compute_ffn_importance must return per-neuron scores of correct shape."""
    loader = _make_loader()
    importances = compute_ffn_importance(small_model, loader, n_steps=2)

    assert len(importances) > 0
    for name, scores in importances.items():
        # Scores should have d_ff=128 dimensions
        assert scores.shape == (128,)
        assert torch.isfinite(scores).all()
        assert (scores >= 0).all()  # squared gradients


def test_prune_reduces_params(small_model):
    """prune_model must reduce total parameter count."""
    original_params = sum(p.numel() for p in small_model.parameters())
    loader = _make_loader()
    result = prune_model(
        small_model, loader, PruningConfig(pruning_ratio=0.25, n_calibration_steps=2)
    )

    pruned_params = sum(p.numel() for p in small_model.parameters())
    assert pruned_params < original_params
    assert result.compression_ratio > 1.0


def test_pruned_model_forward(small_model):
    """Pruned model must still produce valid outputs."""
    loader = _make_loader()
    prune_model(small_model, loader, PruningConfig(pruning_ratio=0.2, n_calibration_steps=2))

    ids = torch.randint(0, 256, (1, 16))
    _, logits, _ = small_model(ids)
    assert logits.shape == (1, 16, 256)
    assert torch.isfinite(logits).all()


def test_prune_ratio_proportional(small_model):
    """Pruning 50% should remove approximately half the FFN neurons."""
    # Get original d_ff
    small_model.layers[0].ffn.gate_proj.out_features  # 128

    loader = _make_loader()
    prune_model(small_model, loader, PruningConfig(pruning_ratio=0.5, n_calibration_steps=2))

    pruned_d_ff = small_model.layers[0].ffn.gate_proj.out_features
    expected = int(128 * 0.5)  # 64
    assert pruned_d_ff == expected


def test_pruning_result_fields(small_model):
    """PruningResult must have valid fields."""
    loader = _make_loader()
    result = prune_model(
        small_model, loader, PruningConfig(pruning_ratio=0.3, n_calibration_steps=2)
    )

    assert result.original_params > 0
    assert result.pruned_params > 0
    assert result.pruned_params <= result.original_params
    assert len(result.layers_pruned) > 0
