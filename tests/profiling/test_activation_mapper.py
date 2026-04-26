import torch
import torch.nn as nn

from src.profiling.activation_mapper import (
    ACTIVATION_MAPPER_REGISTRY,
    ActivationMapper,
    ActivationStats,
)


def make_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def run_forward(model, mapper):
    handles = mapper.register_hooks(model)
    x = torch.randn(2, 8)
    model(x)
    return handles


def test_register_hooks_returns_handles():
    model = make_model()
    mapper = ActivationMapper()
    handles = mapper.register_hooks(model)
    assert len(handles) > 0


def test_forward_populates_stats():
    model = make_model()
    mapper = ActivationMapper()
    handles = run_forward(model, mapper)
    stats = mapper.get_stats()
    assert len(stats) > 0
    mapper.remove_hooks(handles)


def test_stats_are_activation_stats_instances():
    model = make_model()
    mapper = ActivationMapper()
    handles = run_forward(model, mapper)
    for s in mapper.get_stats():
        assert isinstance(s, ActivationStats)
    mapper.remove_hooks(handles)


def test_shape_correct():
    model = nn.Linear(8, 16)
    mapper = ActivationMapper()
    handles = mapper.register_hooks(model)
    x = torch.randn(2, 8)
    model(x)
    stats = mapper.get_stats()
    shapes = [s.shape for s in stats]
    assert (2, 16) in shapes
    mapper.remove_hooks(handles)


def test_has_nan_false_for_normal_weights():
    model = make_model()
    mapper = ActivationMapper()
    handles = run_forward(model, mapper)
    for s in mapper.get_stats():
        assert s.has_nan is False
    mapper.remove_hooks(handles)


def test_has_inf_false_for_normal_weights():
    model = make_model()
    mapper = ActivationMapper()
    handles = run_forward(model, mapper)
    for s in mapper.get_stats():
        assert s.has_inf is False
    mapper.remove_hooks(handles)


def test_detect_anomalies_empty_for_clean_model():
    model = make_model()
    mapper = ActivationMapper()
    handles = run_forward(model, mapper)
    anomalies = mapper.detect_anomalies()
    assert anomalies == []
    mapper.remove_hooks(handles)


def test_detect_anomalies_finds_nan():
    class NaNModule(nn.Module):
        def forward(self, x):
            return torch.full_like(x, float("nan"))

    model = NaNModule()
    mapper = ActivationMapper()
    handles = mapper.register_hooks(model)
    x = torch.randn(1, 4)
    model(x)
    anomalies = mapper.detect_anomalies()
    assert any("NaN" in a for a in anomalies)
    mapper.remove_hooks(handles)


def test_clear_resets_stats():
    model = make_model()
    mapper = ActivationMapper()
    handles = run_forward(model, mapper)
    mapper.clear()
    assert mapper.get_stats() == []
    mapper.remove_hooks(handles)


def test_registry_key():
    assert "default" in ACTIVATION_MAPPER_REGISTRY
    assert ACTIVATION_MAPPER_REGISTRY["default"] is ActivationMapper
