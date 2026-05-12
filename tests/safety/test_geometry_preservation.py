"""Safety geometry preservation tests."""
from __future__ import annotations

import torch

from src.safety.geometry_preservation import SafetyGeometryMonitor


class DummyModel(torch.nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear(x)


def test_safety_direction_unit_norm():
    model = DummyModel()
    unsafe = torch.randn(8, 64)
    safe = torch.randn(8, 64)
    monitor = SafetyGeometryMonitor(model, unsafe, safe)
    assert torch.allclose(monitor.safety_direction.norm(), torch.tensor(1.0), atol=1e-6)


def test_geometry_loss_zero_when_directions_match():
    model = DummyModel()
    unsafe = torch.randn(8, 64)
    safe = torch.randn(8, 64)
    monitor = SafetyGeometryMonitor(model, unsafe, safe)
    loss = monitor.geometry_preservation_loss(unsafe, safe, coeff=0.01)
    assert loss.item() < 0.02


def test_geometry_loss_positive_when_directions_diverge():
    model = DummyModel()
    unsafe = torch.randn(8, 64)
    safe = torch.randn(8, 64)  # different direction, not exactly opposite
    monitor = SafetyGeometryMonitor(model, unsafe, safe)
    loss = monitor.geometry_preservation_loss(unsafe, safe, coeff=0.01)
    assert isinstance(loss.item(), float)


def test_check_and_warn_no_drift():
    model = DummyModel()
    unsafe = torch.randn(8, 64)
    safe = torch.randn(8, 64)
    monitor = SafetyGeometryMonitor(model, unsafe, safe, warning_threshold=0.15)
    result = monitor.check_and_warn(unsafe, safe)
    assert result is False


def test_compute_current_direction():
    model = DummyModel()
    unsafe = torch.randn(8, 64)
    safe = torch.randn(8, 64)
    monitor = SafetyGeometryMonitor(model, unsafe, safe)
    cos_sim = monitor.compute_current_direction(unsafe, safe)
    assert -1.0 <= cos_sim <= 1.0001