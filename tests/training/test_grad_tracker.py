import pytest
import torch
import torch.nn as nn
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.grad_tracker import GradTracker, GradSnapshot

@pytest.fixture
def small_model():
    cfg = AureliusConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
                         head_dim=32, d_ff=128, vocab_size=256, max_seq_len=64)
    torch.manual_seed(0)
    return AureliusTransformer(cfg)

def _run_backward(model):
    input_ids = torch.randint(0, 256, (1, 8))
    loss, _, _ = model(input_ids, labels=input_ids)
    loss.backward()
    return loss

def test_snapshot_after_backward(small_model):
    tracker = GradTracker(small_model)
    _run_backward(small_model)
    snap = tracker.snapshot(step=0)
    assert isinstance(snap, GradSnapshot)
    assert len(snap.param_norms) > 0
    assert len(snap.layer_norms) > 0
    assert snap.global_norm > 0.0

def test_snapshot_norms_are_finite(small_model):
    import math
    tracker = GradTracker(small_model)
    _run_backward(small_model)
    snap = tracker.snapshot(step=1)
    for norm in snap.param_norms.values():
        assert math.isfinite(norm)
    assert math.isfinite(snap.global_norm)

def test_history_accumulates(small_model):
    tracker = GradTracker(small_model)
    for step in range(3):
        small_model.zero_grad()
        _run_backward(small_model)
        tracker.snapshot(step=step)
    assert len(tracker.history) == 3
    assert tracker.history[0].step == 0
    assert tracker.history[2].step == 2

def test_summary_is_string(small_model):
    tracker = GradTracker(small_model)
    _run_backward(small_model)
    tracker.snapshot(step=0)
    s = tracker.summary()
    assert isinstance(s, str)
    assert len(s) > 0

def test_global_norm_matches_manual(small_model):
    """global_norm should equal sqrt(sum of all param_norm^2)."""
    import math
    tracker = GradTracker(small_model)
    _run_backward(small_model)
    snap = tracker.snapshot(step=0)
    manual = math.sqrt(sum(v**2 for v in snap.param_norms.values()))
    assert abs(snap.global_norm - manual) < 1e-4

def test_no_grad_params_skipped(small_model):
    """Parameters without .grad should not appear in param_norms."""
    # Freeze all params, run backward on just one
    for p in small_model.parameters():
        p.requires_grad_(False)
    # Unfreeze just embed
    small_model.embed.weight.requires_grad_(True)

    tracker = GradTracker(small_model)
    input_ids = torch.randint(0, 256, (1, 8))
    loss, _, _ = small_model(input_ids, labels=input_ids)
    loss.backward()
    snap = tracker.snapshot(step=0)

    # Only embed.weight should appear
    assert all("embed" in k for k in snap.param_norms.keys())
