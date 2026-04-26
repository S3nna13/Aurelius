import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.activation_stats import ActivationProfiler, LayerStats


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
    torch.manual_seed(0)
    return AureliusTransformer(cfg)


def test_profiler_captures_stats(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    with ActivationProfiler(small_model) as profiler:
        small_model(input_ids)
    assert len(profiler.stats) > 0
    for name, stat in profiler.stats.items():
        assert isinstance(stat, LayerStats)
        assert isinstance(stat.mean, float)


def test_profiler_sparsity_in_range(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    with ActivationProfiler(small_model) as profiler:
        small_model(input_ids)
    for stat in profiler.stats.values():
        assert 0.0 <= stat.sparsity <= 1.0


def test_profiler_hooks_removed_after_exit(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    with ActivationProfiler(small_model) as profiler:
        small_model(input_ids)
    # Run again outside context — should not accumulate more stats
    stats_count = len(profiler.stats)
    small_model(input_ids)
    assert len(profiler.stats) == stats_count  # unchanged


def test_profiler_summary_is_string(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    with ActivationProfiler(small_model) as profiler:
        small_model(input_ids)
    s = profiler.summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_profiler_abs_max_positive(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    with ActivationProfiler(small_model) as profiler:
        small_model(input_ids)
    for stat in profiler.stats.values():
        assert stat.abs_max >= 0.0


def test_profiler_custom_module_types(small_model):
    """Profiler with module_types=(nn.Linear,) should only hook Linear layers."""
    import torch.nn as nn

    input_ids = torch.randint(0, 256, (1, 8))
    with ActivationProfiler(small_model, module_types=(nn.Linear,)) as profiler:
        small_model(input_ids)
    assert len(profiler.stats) > 0
