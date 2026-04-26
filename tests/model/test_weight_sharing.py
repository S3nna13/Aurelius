"""Tests for ALBERT-style weight reduction: factorized embeddings + cross-layer sharing."""

from __future__ import annotations

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.model.weight_sharing import (
    FactorizedEmbedding,
    WeightSharingConfig,
    apply_cross_layer_sharing,
    apply_factorized_embedding,
    count_unique_parameters,
)

# Small config for fast CPU tests
SMALL_CONFIG = AureliusConfig(
    n_layers=4,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
    tie_embeddings=False,
)

WS_CONFIG = WeightSharingConfig(factorized_embed_dim=16, n_shared_groups=1)


def make_model() -> AureliusTransformer:
    return AureliusTransformer(SMALL_CONFIG)


# ---------------------------------------------------------------------------
# FactorizedEmbedding tests
# ---------------------------------------------------------------------------


def test_factorized_embedding_forward_shape():
    """Output shape should be (B, S, d_model)."""
    B, S = 2, 8
    embed = FactorizedEmbedding(vocab_size=256, embed_dim=16, d_model=64)
    x = torch.randint(0, 256, (B, S))
    out = embed(x)
    assert out.shape == (B, S, 64), f"Expected (2, 8, 64), got {out.shape}"


def test_factorized_embedding_param_count():
    """Factorized embedding should have fewer params than standard embedding."""
    vocab_size, embed_dim, d_model = 256, 16, 64

    factorized = FactorizedEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, d_model=d_model)
    factorized_params = sum(p.numel() for p in factorized.parameters())

    standard_params = vocab_size * d_model  # standard nn.Embedding

    assert factorized_params < standard_params, (
        f"Factorized ({factorized_params}) should be < standard ({standard_params})"
    )


def test_factorized_embedding_parameter_savings_positive():
    """parameter_savings() should be > 0 when embed_dim < d_model."""
    embed = FactorizedEmbedding(vocab_size=256, embed_dim=16, d_model=64)
    savings = embed.parameter_savings(vocab_size=256, embed_dim=16, d_model=64)
    assert savings > 0, f"Expected positive savings, got {savings}"


# ---------------------------------------------------------------------------
# apply_factorized_embedding tests
# ---------------------------------------------------------------------------


def test_apply_factorized_embedding_changes_embed():
    """model.embed should become a FactorizedEmbedding after apply."""
    model = make_model()
    apply_factorized_embedding(model, embed_dim=16, config=SMALL_CONFIG)
    assert isinstance(model.embed, FactorizedEmbedding), (
        f"Expected FactorizedEmbedding, got {type(model.embed)}"
    )


def test_apply_factorized_embedding_forward_works():
    """Model forward pass should still work after replacing embed."""
    model = make_model()
    apply_factorized_embedding(model, embed_dim=16, config=SMALL_CONFIG)
    model.eval()
    input_ids = torch.randint(0, SMALL_CONFIG.vocab_size, (1, 8))
    with torch.no_grad():
        loss, logits, _ = model(input_ids)
    assert logits.shape == (1, 8, SMALL_CONFIG.vocab_size), (
        f"Unexpected logits shape: {logits.shape}"
    )


# ---------------------------------------------------------------------------
# apply_cross_layer_sharing tests
# ---------------------------------------------------------------------------


def test_apply_cross_layer_sharing_all_same():
    """With n_groups=1, all layers[i] should be the same Python object."""
    model = make_model()
    apply_cross_layer_sharing(model, n_shared_groups=1)
    layer0 = model.layers[0]
    for i, layer in enumerate(model.layers):
        assert layer is layer0, f"layers[{i}] is not layers[0]"


def test_apply_cross_layer_sharing_two_groups():
    """With n_groups=2 and n_layers=4: layers[0]==layers[2], layers[1]==layers[3]."""
    model = make_model()
    apply_cross_layer_sharing(model, n_shared_groups=2)
    layers = model.layers
    assert layers[0] is layers[2], "layers[0] and layers[2] should be the same object"
    assert layers[1] is layers[3], "layers[1] and layers[3] should be the same object"
    assert layers[0] is not layers[1], "layers[0] and layers[1] should be different objects"


# ---------------------------------------------------------------------------
# count_unique_parameters tests
# ---------------------------------------------------------------------------


def test_count_unique_parameters_shared():
    """With all layers sharing, unique_params should be much less than total_params."""
    model = make_model()
    apply_cross_layer_sharing(model, n_shared_groups=1)
    counts = count_unique_parameters(model)
    assert counts["unique_params"] < counts["total_params"], (
        f"Expected unique ({counts['unique_params']}) < total ({counts['total_params']})"
    )


def test_count_unique_parameters_ratio():
    """shared_ratio should be between 0 and 1."""
    model = make_model()
    apply_cross_layer_sharing(model, n_shared_groups=1)
    counts = count_unique_parameters(model)
    ratio = counts["shared_ratio"]
    assert 0.0 <= ratio <= 1.0, f"shared_ratio {ratio} out of [0, 1]"


# ---------------------------------------------------------------------------
# Forward pass after cross-layer sharing
# ---------------------------------------------------------------------------


def test_cross_layer_sharing_forward_works():
    """Forward pass should work correctly after cross-layer sharing."""
    model = make_model()
    apply_cross_layer_sharing(model, n_shared_groups=1)
    model.eval()
    input_ids = torch.randint(0, SMALL_CONFIG.vocab_size, (2, 8))
    with torch.no_grad():
        loss, logits, _ = model(input_ids)
    assert logits.shape == (2, 8, SMALL_CONFIG.vocab_size), (
        f"Unexpected logits shape after sharing: {logits.shape}"
    )
