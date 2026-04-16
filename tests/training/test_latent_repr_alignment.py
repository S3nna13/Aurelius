"""Tests for latent representation alignment (src/training/latent_repr_alignment.py)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.latent_repr_alignment import (
    ReprAlignmentConfig,
    RepresentationAligner,
    LayerwiseAlignmentTrainer,
    centered_kernel_alignment,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(n_layers: int = 2, d_model: int = 64) -> AureliusTransformer:
    """Small AureliusTransformer for fast tests."""
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=2,
        n_kv_heads=2,
        head_dim=d_model // 2,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    return AureliusTransformer(cfg)


# ---------------------------------------------------------------------------
# Test 1: ReprAlignmentConfig defaults
# ---------------------------------------------------------------------------

def test_repr_alignment_config_defaults():
    """ReprAlignmentConfig must expose the correct default values."""
    cfg = ReprAlignmentConfig()
    assert cfg.align_type == 'linear'
    assert cfg.normalize is True
    assert cfg.align_weight == 0.1
    assert cfg.layer_pairs == [] or cfg.layer_pairs is None or isinstance(cfg.layer_pairs, list)


# ---------------------------------------------------------------------------
# Test 2: RepresentationAligner forward returns (loss, dict) tuple
# ---------------------------------------------------------------------------

def test_repr_aligner_forward_returns_tuple():
    """RepresentationAligner.forward must return a (Tensor, dict) pair."""
    aligner = RepresentationAligner(student_dim=64, teacher_dim=64, align_type='linear')
    B, S, D = 2, 8, 64
    s = torch.randn(B, S, D)
    t = torch.randn(B, S, D)

    result = aligner(s, t)

    assert isinstance(result, tuple), "forward should return a tuple"
    assert len(result) == 2, "tuple must have 2 elements: (loss, dict)"
    loss, metrics = result
    assert isinstance(loss, torch.Tensor), "first element must be a Tensor"
    assert isinstance(metrics, dict), "second element must be a dict"
    assert 'cosine_sim' in metrics
    assert 'mse' in metrics
    assert 'projection_norm' in metrics


# ---------------------------------------------------------------------------
# Test 3: align_loss is scalar
# ---------------------------------------------------------------------------

def test_align_loss_is_scalar():
    """align_loss must return a 0-d scalar tensor."""
    for align_type in ('linear', 'cosine', 'mse'):
        aligner = RepresentationAligner(student_dim=32, teacher_dim=32, align_type=align_type)
        B, S, D = 2, 4, 32
        s = torch.randn(B, S, D)
        t = torch.randn(B, S, D)
        loss = aligner.align_loss(s, t)
        assert loss.ndim == 0, f"align_loss for '{align_type}' must be a scalar"
        assert torch.isfinite(loss), f"align_loss for '{align_type}' must be finite"


# ---------------------------------------------------------------------------
# Test 4: Identical representations have loss ~ 0 in cosine mode
# ---------------------------------------------------------------------------

def test_cosine_loss_identical_representations():
    """Cosine alignment loss should be ~0 when student == teacher."""
    aligner = RepresentationAligner(student_dim=64, teacher_dim=64, align_type='cosine', normalize=True)
    B, S, D = 2, 8, 64
    x = torch.randn(B, S, D)

    loss = aligner.align_loss(x, x.clone())
    assert loss.item() < 1e-5, f"Expected cosine loss ~0 for identical reprs, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 5: Linear projection aligns different dimensions
# ---------------------------------------------------------------------------

def test_linear_projection_different_dims():
    """Linear aligner must handle student_dim != teacher_dim via projection."""
    student_dim, teacher_dim = 32, 64
    aligner = RepresentationAligner(student_dim=student_dim, teacher_dim=teacher_dim, align_type='linear')

    # Verify projection exists and has correct shape
    assert aligner.projection is not None
    assert aligner.projection.weight.shape == (teacher_dim, student_dim)

    B, S = 2, 8
    s = torch.randn(B, S, student_dim)
    t = torch.randn(B, S, teacher_dim)

    loss, metrics = aligner(s, t)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# Test 6: LayerwiseAlignmentTrainer instantiates with a model pair
# ---------------------------------------------------------------------------

def test_layerwise_trainer_instantiation():
    """LayerwiseAlignmentTrainer should instantiate without errors."""
    student = _make_model(n_layers=2, d_model=64)
    teacher = _make_model(n_layers=2, d_model=64)
    layer_pairs = [(0, 0), (1, 1)]

    trainer = LayerwiseAlignmentTrainer(student, teacher, layer_pairs, align_weight=0.1)

    assert trainer.align_weight == 0.1
    assert len(trainer.layer_pairs) == 2
    assert len(trainer.aligners) == 2

    # Teacher should be frozen
    for p in teacher.parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# Test 7: extract_layer_representations returns correct count
# ---------------------------------------------------------------------------

def test_extract_layer_representations_count():
    """extract_layer_representations should return one tensor per requested layer."""
    model = _make_model(n_layers=2, d_model=64)
    trainer = LayerwiseAlignmentTrainer(
        model, _make_model(n_layers=2, d_model=64), layer_pairs=[(0, 0)]
    )

    B, S = 2, 8
    input_ids = torch.randint(0, 256, (B, S))

    reprs = trainer.extract_layer_representations(model, input_ids, layer_indices=[0, 1])

    assert len(reprs) == 2, "Should return 2 tensors for 2 layer indices"
    for r in reprs:
        assert r.shape == (B, S, 64), f"Each repr should be (B, S, d_model), got {r.shape}"


# ---------------------------------------------------------------------------
# Test 8: compute_alignment_loss returns dict with 'mean_cosine_sim'
# ---------------------------------------------------------------------------

def test_compute_alignment_loss_has_mean_cosine_sim():
    """compute_alignment_loss must return a dict with 'mean_cosine_sim' key."""
    student = _make_model(n_layers=2, d_model=64)
    teacher = _make_model(n_layers=2, d_model=64)
    trainer = LayerwiseAlignmentTrainer(student, teacher, layer_pairs=[(0, 0), (1, 1)])

    input_ids = torch.randint(0, 256, (2, 8))
    loss, metrics = trainer.compute_alignment_loss(input_ids, input_ids)

    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
    assert 'mean_cosine_sim' in metrics, "'mean_cosine_sim' must be in the metrics dict"
    assert 'layer_losses' in metrics
    assert len(metrics['layer_losses']) == 2


# ---------------------------------------------------------------------------
# Test 9: train_step returns dict with all required keys
# ---------------------------------------------------------------------------

def test_train_step_returns_required_keys():
    """train_step must return a dict containing 'loss', 'task_loss', 'alignment_loss', 'cosine_sim'."""
    student = _make_model(n_layers=2, d_model=64)
    teacher = _make_model(n_layers=2, d_model=64)
    trainer = LayerwiseAlignmentTrainer(student, teacher, layer_pairs=[(0, 0)])

    B, S = 2, 8
    input_ids = torch.randint(0, 256, (B, S))
    labels = torch.randint(0, 256, (B, S))

    result = trainer.train_step(input_ids, labels)

    required_keys = {'loss', 'task_loss', 'alignment_loss', 'cosine_sim'}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )
    for k, v in result.items():
        assert isinstance(v, float), f"result['{k}'] should be a float, got {type(v)}"


# ---------------------------------------------------------------------------
# Test 10: centered_kernel_alignment returns float in [0, 1]
# ---------------------------------------------------------------------------

def test_cka_returns_float_in_range():
    """centered_kernel_alignment must return a float in [0, 1]."""
    torch.manual_seed(0)
    n, d1, d2 = 32, 16, 24
    X = torch.randn(n, d1)
    Y = torch.randn(n, d2)

    score = centered_kernel_alignment(X, Y)

    assert isinstance(score, float), f"CKA should return float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"CKA must be in [0, 1], got {score}"


# ---------------------------------------------------------------------------
# Test 11: CKA of identical matrices = 1.0
# ---------------------------------------------------------------------------

def test_cka_identical_matrices():
    """CKA of a matrix with itself must be 1.0."""
    torch.manual_seed(1)
    n, d = 32, 16
    X = torch.randn(n, d)

    score = centered_kernel_alignment(X, X)

    assert abs(score - 1.0) < 1e-4, f"CKA(X, X) should be 1.0, got {score}"


# ---------------------------------------------------------------------------
# Test 12: Gradient flows through alignment loss
# ---------------------------------------------------------------------------

def test_gradient_flows_through_alignment_loss():
    """Gradients must flow back through the alignment loss to student parameters."""
    torch.manual_seed(42)
    aligner = RepresentationAligner(student_dim=32, teacher_dim=32, align_type='linear', normalize=True)

    B, S, D = 2, 4, 32
    s = torch.randn(B, S, D, requires_grad=True)
    t = torch.randn(B, S, D)

    loss, _ = aligner(s, t)
    loss.backward()

    assert s.grad is not None, "Gradient must flow to student input"
    assert s.grad.shape == s.shape
    assert not torch.all(s.grad == 0), "Gradients should not all be zero"

    # Also check projection weight gets a gradient
    assert aligner.projection is not None
    assert aligner.projection.weight.grad is not None
    assert not torch.all(aligner.projection.weight.grad == 0)
