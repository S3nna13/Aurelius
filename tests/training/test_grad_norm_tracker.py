import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.grad_norm_tracker import (
    GradNormConfig,
    GradNormTracker,
    clip_by_adaptive_norm,
    compute_global_grad_norm,
)


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    torch.manual_seed(42)
    return AureliusTransformer(cfg)


def _run_backward(model):
    model.zero_grad()
    input_ids = torch.randint(0, 256, (1, 8))
    loss, _, _ = model(input_ids, labels=input_ids)
    loss.backward()
    return loss


# ---------------------------------------------------------------------------
# compute_global_grad_norm
# ---------------------------------------------------------------------------


def test_compute_global_grad_norm_zero_no_grads(small_model):
    """Returns 0.0 when no backward pass has been run (no .grad tensors)."""
    small_model.zero_grad()
    result = compute_global_grad_norm(small_model)
    assert result == 0.0


def test_compute_global_grad_norm_positive(small_model):
    """Returns a positive value after backward."""
    _run_backward(small_model)
    result = compute_global_grad_norm(small_model)
    assert result > 0.0


# ---------------------------------------------------------------------------
# GradNormTracker.step
# ---------------------------------------------------------------------------


def test_grad_norm_tracker_step_returns_metrics(small_model):
    """step() dict has all required keys with correct types."""
    tracker = GradNormTracker(small_model, GradNormConfig())
    _run_backward(small_model)
    metrics = tracker.step()
    required_keys = {"global_norm", "ema_norm", "is_spike", "step", "recommended_clip"}
    assert required_keys == set(metrics.keys())
    assert isinstance(metrics["global_norm"], float)
    assert isinstance(metrics["ema_norm"], float)
    assert isinstance(metrics["is_spike"], bool)
    assert isinstance(metrics["step"], int)
    assert isinstance(metrics["recommended_clip"], float)


def test_grad_norm_tracker_ema_updates(small_model):
    """ema_norm changes across multiple steps."""
    tracker = GradNormTracker(small_model, GradNormConfig())
    _run_backward(small_model)
    m1 = tracker.step()
    _run_backward(small_model)
    m2 = tracker.step()
    _run_backward(small_model)
    m3 = tracker.step()
    # EMA should update — collect all ema values and check they are not all identical
    emas = [m1["ema_norm"], m2["ema_norm"], m3["ema_norm"]]
    assert len(set(emas)) > 1, "ema_norm should change across steps"


def test_grad_norm_tracker_spike_detection(small_model):
    """is_spike=True when a very large gradient is injected."""
    cfg = GradNormConfig(ema_alpha=0.9, spike_threshold=2.0)
    tracker = GradNormTracker(small_model, cfg)

    # Warm up EMA with normal gradients
    for _ in range(10):
        _run_backward(small_model)
        tracker.step()

    # Inject a massive gradient
    with torch.no_grad():
        for p in small_model.parameters():
            if p.grad is not None:
                p.grad.fill_(1e4)

    metrics = tracker.step()
    assert metrics["is_spike"] is True


def test_grad_norm_tracker_no_spike_normal(small_model):
    """is_spike=False for consistently normal gradients after warm-up."""
    cfg = GradNormConfig(ema_alpha=0.99, spike_threshold=3.0)
    tracker = GradNormTracker(small_model, cfg)

    # Run many steps with normal backward passes to stabilise EMA
    for _ in range(30):
        _run_backward(small_model)
        metrics = tracker.step()

    # Final step: with stable EMA the norm should not be a spike
    assert metrics["is_spike"] is False


def test_grad_norm_tracker_per_layer_norms(small_model):
    """get_per_layer_norms() returns a non-empty dict after a step."""
    cfg = GradNormConfig(track_per_layer=True)
    tracker = GradNormTracker(small_model, cfg)
    _run_backward(small_model)
    tracker.step()
    layer_norms = tracker.get_per_layer_norms()
    assert isinstance(layer_norms, dict)
    assert len(layer_norms) > 0
    for name, norm in layer_norms.items():
        assert isinstance(name, str)
        assert isinstance(norm, float)
        assert norm >= 0.0


def test_grad_norm_tracker_stats(small_model):
    """get_stats() returns the correct set of keys."""
    tracker = GradNormTracker(small_model, GradNormConfig())
    for _ in range(5):
        _run_backward(small_model)
        tracker.step()
    stats = tracker.get_stats()
    required = {"mean_norm", "std_norm", "max_norm", "min_norm", "n_spikes", "spike_rate"}
    assert required == set(stats.keys())
    assert stats["max_norm"] >= stats["min_norm"]
    assert stats["mean_norm"] >= 0.0
    assert 0.0 <= stats["spike_rate"] <= 1.0


def test_grad_norm_tracker_step_counter(small_model):
    """step increments by 1 each call."""
    tracker = GradNormTracker(small_model, GradNormConfig())
    for expected in range(1, 6):
        _run_backward(small_model)
        metrics = tracker.step()
        assert metrics["step"] == expected


def test_clip_by_adaptive_norm(small_model):
    """clip_by_adaptive_norm clips and returns a float."""
    cfg = GradNormConfig(ema_alpha=0.9)
    tracker = GradNormTracker(small_model, cfg)

    # Warm up
    for _ in range(5):
        _run_backward(small_model)
        tracker.step()

    _run_backward(small_model)
    pre_clip_norm = clip_by_adaptive_norm(small_model, tracker, multiplier=2.0)
    assert isinstance(pre_clip_norm, float)
    assert pre_clip_norm >= 0.0


def test_grad_norm_tracker_reset(small_model):
    """reset() clears history and resets step counter to 0."""
    tracker = GradNormTracker(small_model, GradNormConfig())
    for _ in range(5):
        _run_backward(small_model)
        tracker.step()

    tracker.reset()

    # After reset, step should restart at 1
    _run_backward(small_model)
    metrics = tracker.step()
    assert metrics["step"] == 1

    # Stats window should have only the one post-reset step
    stats = tracker.get_stats()
    assert stats["max_norm"] == stats["min_norm"]  # single value in window
