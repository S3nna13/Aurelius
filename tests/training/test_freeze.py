import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.freeze import (
    freeze_except,
    freeze_layers,
    get_trainable_param_count,
    unfreeze_layers,
)


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=4,
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


def test_freeze_layers_by_pattern(small_model):
    n = freeze_layers(small_model, ["layers.0.*"])
    assert n > 0
    for name, p in small_model.named_parameters():
        if name.startswith("layers.0."):
            assert not p.requires_grad, f"{name} should be frozen"
        else:
            assert p.requires_grad, f"{name} should still be trainable"


def test_unfreeze_restores_grad(small_model):
    freeze_layers(small_model, ["layers.0.*", "layers.1.*"])
    unfreeze_layers(small_model, ["layers.0.*"])
    for name, p in small_model.named_parameters():
        if name.startswith("layers.0."):
            assert p.requires_grad, f"{name} should be unfrozen"
        if name.startswith("layers.1."):
            assert not p.requires_grad, f"{name} should still be frozen"


def test_freeze_except_keeps_head(small_model):
    n = freeze_except(small_model, ["lm_head.*"])
    assert n > 0
    for name, p in small_model.named_parameters():
        if name.startswith("lm_head"):
            assert p.requires_grad
        else:
            assert not p.requires_grad


def test_get_trainable_param_count(small_model):
    freeze_layers(small_model, ["embed.*"])
    counts = get_trainable_param_count(small_model)
    assert counts["frozen"] > 0
    assert counts["trainable"] > 0
    assert counts["total"] == counts["trainable"] + counts["frozen"]


def test_freeze_returns_correct_count(small_model):
    n = freeze_layers(small_model, ["layers.0.*"])
    frozen_in_layer0 = sum(
        1 for name, _ in small_model.named_parameters() if name.startswith("layers.0.")
    )
    assert n == frozen_in_layer0
