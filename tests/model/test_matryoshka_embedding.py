"""Unit tests for :mod:`src.model.matryoshka_embedding`."""

from __future__ import annotations

import pytest
import torch

import src.model as model_pkg
from src.model.matryoshka_embedding import MatryoshkaConfig, MatryoshkaEmbedding


def test_config_validates_dimensions() -> None:
    with pytest.raises(ValueError):
        MatryoshkaConfig(full_dim=0)
    with pytest.raises(ValueError):
        MatryoshkaConfig(full_dim=8, nested_dims=[4, 2, 8])
    with pytest.raises(ValueError):
        MatryoshkaConfig(full_dim=8, nested_dims=[2, 4, 8], scale_weights=[1.0])


def test_forward_nested_and_loss_are_shape_safe() -> None:
    torch.manual_seed(0)
    cfg = MatryoshkaConfig(full_dim=8, nested_dims=[2, 4, 8])
    model = MatryoshkaEmbedding(cfg, in_features=4)

    x = torch.randn(3, 4, requires_grad=True)
    z = model(x)

    assert z.shape == (3, 8)
    norms = torch.linalg.vector_norm(z, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    nested = model.get_nested(z, 4)
    assert nested.shape == (3, 4)

    losses = model.matryoshka_loss(z, z)
    assert set(losses) == {"loss_2", "loss_4", "loss_8", "loss"}
    losses["loss"].backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_anisotropy_singleton_is_zero() -> None:
    cfg = MatryoshkaConfig(full_dim=8, nested_dims=[2, 4, 8])
    model = MatryoshkaEmbedding(cfg, in_features=4)
    x = torch.randn(1, 4)
    z = model(x)
    assert model.anisotropy(z) == 0.0


def test_registry_entry_present_in_model_package() -> None:
    assert "matryoshka_embedding" in model_pkg.MODEL_COMPONENT_REGISTRY
    assert (
        model_pkg.MODEL_COMPONENT_REGISTRY["matryoshka_embedding"]
        is MatryoshkaEmbedding
    )
