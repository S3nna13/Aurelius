"""Tests for src/model/surgery.py — post-hoc model architecture modification."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.surgery import (
    ScalingResult,
    insert_layer,
    remove_layer,
    scale_model_depth,
    swap_ffn,
)
from src.model.transformer import AureliusTransformer, TransformerBlock

# ---------------------------------------------------------------------------
# Fixture: tiny model for fast tests
# ---------------------------------------------------------------------------


def make_tiny_config() -> AureliusConfig:
    """Return a tiny config that satisfies all AureliusConfig assertions."""
    return AureliusConfig(
        d_model=64,
        n_layers=4,
        n_heads=2,
        n_kv_heads=1,
        head_dim=32,  # 2 * 32 == 64 ✓
        d_ff=128,
        vocab_size=256,
        max_seq_len=128,
    )


@pytest.fixture()
def tiny_model() -> AureliusTransformer:
    return AureliusTransformer(make_tiny_config())


@pytest.fixture()
def input_ids() -> torch.Tensor:
    return torch.randint(0, 256, (1, 8))


# ---------------------------------------------------------------------------
# insert_layer tests
# ---------------------------------------------------------------------------


def test_insert_layer_increases_count(tiny_model):
    """n_layers should increase by 1 after insert."""
    original_count = tiny_model.config.n_layers
    insert_layer(tiny_model, layer_idx=1)
    assert tiny_model.config.n_layers == original_count + 1
    assert len(tiny_model.layers) == original_count + 1


def test_insert_layer_copy_init(tiny_model):
    """Copied layer should have same parameter structure as adjacent."""
    insert_layer(tiny_model, layer_idx=1, init_from="copy")
    # The inserted layer at index 1 is a copy of the original layer 1
    inserted = tiny_model.layers[1]
    assert isinstance(inserted, TransformerBlock)
    # Check same parameter shapes
    orig_layer = tiny_model.layers[2]  # original layer 1 is now at index 2
    for (n1, p1), (n2, p2) in zip(inserted.named_parameters(), orig_layer.named_parameters()):
        assert p1.shape == p2.shape, f"Shape mismatch for {n1} vs {n2}"


def test_insert_layer_random_init(tiny_model):
    """Random init should work without raising any errors."""
    original_count = tiny_model.config.n_layers
    insert_layer(tiny_model, layer_idx=2, init_from="random")
    assert tiny_model.config.n_layers == original_count + 1


def test_insert_layer_at_start(tiny_model):
    """Inserting at layer_idx=0 should prepend a layer."""
    # Grab a reference to the old first layer
    old_first_id = id(tiny_model.layers[0])
    insert_layer(tiny_model, layer_idx=0, init_from="copy")
    # The new layer is at index 0; the old first is now at index 1
    assert id(tiny_model.layers[1]) == old_first_id
    assert len(tiny_model.layers) == 5


def test_insert_layer_at_end(tiny_model):
    """Inserting at layer_idx=n should append a layer."""
    n = tiny_model.config.n_layers  # 4
    old_last_id = id(tiny_model.layers[-1])
    insert_layer(tiny_model, layer_idx=n, init_from="copy")
    # Old last layer is still at index n-1 == 3
    assert id(tiny_model.layers[n - 1]) == old_last_id
    assert len(tiny_model.layers) == 5


# ---------------------------------------------------------------------------
# remove_layer tests
# ---------------------------------------------------------------------------


def test_remove_layer_decreases_count(tiny_model):
    """n_layers should decrease by 1 after remove."""
    original_count = tiny_model.config.n_layers
    remove_layer(tiny_model, layer_idx=1)
    assert tiny_model.config.n_layers == original_count - 1
    assert len(tiny_model.layers) == original_count - 1


def test_remove_layer_returns_module(tiny_model):
    """remove_layer should return the detached TransformerBlock."""
    removed = remove_layer(tiny_model, layer_idx=2)
    assert isinstance(removed, nn.Module)
    assert isinstance(removed, TransformerBlock)


# ---------------------------------------------------------------------------
# swap_ffn tests
# ---------------------------------------------------------------------------


def test_swap_ffn_returns_old(tiny_model):
    """swap_ffn should return the old FFN module."""
    old_ffn = tiny_model.layers[0].ffn
    new_ffn = nn.Linear(64, 64)  # arbitrary replacement
    returned = swap_ffn(tiny_model, layer_idx=0, new_ffn=new_ffn)
    assert returned is old_ffn


def test_swap_ffn_model_uses_new(tiny_model, input_ids):
    """After swapping FFN, forward pass should produce different outputs."""
    with torch.no_grad():
        _, logits_before, _ = tiny_model(input_ids)

    # Replace the FFN in layer 0 with one that always outputs zeros
    class ZeroFFN(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    swap_ffn(tiny_model, layer_idx=0, new_ffn=ZeroFFN())

    with torch.no_grad():
        _, logits_after, _ = tiny_model(input_ids)

    # Outputs must differ because FFN contribution is zeroed out
    assert not torch.allclose(logits_before, logits_after), (
        "Expected logits to change after FFN swap"
    )


# ---------------------------------------------------------------------------
# scale_model_depth tests
# ---------------------------------------------------------------------------


def test_scale_model_depth_grow(tiny_model):
    """Growing: target > current → n_layers increases."""
    result = scale_model_depth(tiny_model, target_n_layers=6)
    assert tiny_model.config.n_layers == 6
    assert len(tiny_model.layers) == 6
    assert result.n_layers_new == 6
    assert result.layers_added > 0
    assert result.layers_removed == 0


def test_scale_model_depth_shrink(tiny_model):
    """Shrinking: target < current → n_layers decreases."""
    result = scale_model_depth(tiny_model, target_n_layers=2)
    assert tiny_model.config.n_layers == 2
    assert len(tiny_model.layers) == 2
    assert result.n_layers_new == 2
    assert result.layers_removed > 0
    assert result.layers_added == 0


def test_scale_result_fields(tiny_model):
    """ScalingResult should have all required fields with correct types."""
    result = scale_model_depth(tiny_model, target_n_layers=6)
    assert isinstance(result, ScalingResult)
    assert isinstance(result.original_params, int)
    assert isinstance(result.new_params, int)
    assert isinstance(result.n_layers_original, int)
    assert isinstance(result.n_layers_new, int)
    assert isinstance(result.layers_added, int)
    assert isinstance(result.layers_removed, int)
    assert result.n_layers_original == 4
    assert result.n_layers_new == 6
    assert result.layers_added == 2
    assert result.layers_removed == 0


# ---------------------------------------------------------------------------
# End-to-end: model still runs after surgery
# ---------------------------------------------------------------------------


def test_model_still_runs_after_surgery(tiny_model, input_ids):
    """Model forward pass should work correctly after insert + remove."""
    # Insert a layer
    insert_layer(tiny_model, layer_idx=2, init_from="copy")
    assert tiny_model.config.n_layers == 5

    # Remove a different layer
    remove_layer(tiny_model, layer_idx=0)
    assert tiny_model.config.n_layers == 4

    # Forward pass should succeed and produce correct shapes
    with torch.no_grad():
        loss, logits, pkv = tiny_model(input_ids)

    assert logits.shape == (1, 8, 256)
    assert len(pkv) == 4
    assert loss is None
