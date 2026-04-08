"""Tests for Universal Transformer (Dehghani et al. 2018)."""

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.universal_transformer import (
    UniversalTransformer,
    UniversalTransformerBlock,
)


@pytest.fixture
def cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=4,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def model(cfg: AureliusConfig) -> UniversalTransformer:
    return UniversalTransformer(cfg)


def test_universal_transformer_forward_shape(model, cfg):
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    _loss, logits, _cache = model(input_ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_universal_transformer_returns_tuple(model, cfg):
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    result = model(input_ids)
    assert isinstance(result, tuple)
    assert len(result) == 3
    loss, logits, cache = result
    assert loss is None
    assert cache is None


def test_universal_transformer_loss_with_labels(model, cfg):
    import math
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = torch.randint(0, cfg.vocab_size, (2, 16))
    loss, logits, _ = model(input_ids, labels=labels)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0
    assert math.isfinite(loss.item())


def test_shared_weights(model):
    block = model.shared_block
    w0 = block.attn.q_proj.weight
    w_again = block.attn.q_proj.weight
    assert w0 is w_again

    assert not hasattr(model, "layers"), \
        "UniversalTransformer must not have a 'layers' ModuleList"

    for name, module in model.named_modules():
        if name == "":
            continue
        assert not isinstance(module, nn.ModuleList), \
            f"Found unexpected ModuleList at '{name}'"


def test_parameter_count_less_than_standard(cfg):
    from src.model.transformer import AureliusTransformer

    std_cfg = AureliusConfig(
        n_layers=6,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        tie_embeddings=False,
    )
    ut_cfg = AureliusConfig(
        n_layers=6,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )

    standard = AureliusTransformer(std_cfg)
    universal = UniversalTransformer(ut_cfg, n_steps=6)

    std_params = sum(p.numel() for p in standard.parameters())
    ut_params = universal.parameter_count()

    assert ut_params < std_params, \
        f"UT ({ut_params:,}) should have fewer params than standard ({std_params:,})"


def test_step_embedding_effect(cfg):
    model1 = UniversalTransformer(cfg, n_steps=1)
    model3 = UniversalTransformer(cfg, n_steps=3)

    model3.shared_block.load_state_dict(model1.shared_block.state_dict())
    model3.embed.load_state_dict(model1.embed.state_dict())
    model3.norm.load_state_dict(model1.norm.state_dict())
    model3.lm_head.load_state_dict(model1.lm_head.state_dict())

    model1.train(False)
    model3.train(False)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        _, logits1, _ = model1(input_ids)
        _, logits3, _ = model3(input_ids)

    assert not torch.allclose(logits1, logits3), \
        "n_steps=1 and n_steps=3 should produce different outputs"


def test_effective_depth(cfg):
    for n in (1, 4, 8):
        model = UniversalTransformer(cfg, n_steps=n)
        assert model.effective_depth() == n
