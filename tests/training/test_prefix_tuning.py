"""Tests for src/training/prefix_tuning.py.

Uses tiny dimensions so tests run quickly on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.prefix_tuning import (
    PrefixConfig,
    PrefixEmbedding,
    PrefixTuner,
    PromptTuningConfig,
    SoftPrompt,
    count_trainable_params,
    freeze_model,
    prepend_prefix,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------

D_MODEL = 16
PREFIX_LEN = 4
N_LAYERS = 2
VOCAB = 32
B = 2
T = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embed() -> nn.Embedding:
    """Return a small backbone embedding layer."""
    return nn.Embedding(VOCAB, D_MODEL)


def _make_config(**kwargs) -> PrefixConfig:
    return PrefixConfig(
        prefix_length=PREFIX_LEN,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. PrefixConfig defaults
# ---------------------------------------------------------------------------


def test_prefix_config_defaults():
    cfg = PrefixConfig()
    assert cfg.prefix_length == 10
    assert cfg.d_model == 512
    assert cfg.n_layers == 12
    assert cfg.init_from_vocab is False
    assert cfg.reparam_hidden_dim is None


# ---------------------------------------------------------------------------
# 2. PrefixEmbedding output shape (direct parameter)
# ---------------------------------------------------------------------------


def test_prefix_embedding_direct_shape():
    cfg = _make_config()
    pe = PrefixEmbedding(cfg)
    out = pe()
    assert out.shape == (PREFIX_LEN, D_MODEL), (
        f"Expected ({PREFIX_LEN}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 3. PrefixEmbedding with reparameterization output shape
# ---------------------------------------------------------------------------


def test_prefix_embedding_reparam_shape():
    cfg = _make_config(reparam_hidden_dim=8)
    pe = PrefixEmbedding(cfg)
    out = pe()
    assert out.shape == (PREFIX_LEN, D_MODEL), (
        f"Expected ({PREFIX_LEN}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 4. prepend_prefix output shape
# ---------------------------------------------------------------------------


def test_prepend_prefix_output_shape():
    prefix = torch.randn(PREFIX_LEN, D_MODEL)
    input_embeds = torch.randn(B, T, D_MODEL)
    out = prepend_prefix(prefix, input_embeds)
    assert out.shape == (B, PREFIX_LEN + T, D_MODEL), (
        f"Expected ({B}, {PREFIX_LEN + T}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 5. prepend_prefix preserves input at the end
# ---------------------------------------------------------------------------


def test_prepend_prefix_preserves_input():
    prefix = torch.zeros(PREFIX_LEN, D_MODEL)
    input_embeds = torch.randn(B, T, D_MODEL)
    out = prepend_prefix(prefix, input_embeds)
    # Suffix should exactly match input_embeds
    assert torch.allclose(out[:, PREFIX_LEN:, :], input_embeds), (
        "Input embeddings were modified by prepend_prefix"
    )


# ---------------------------------------------------------------------------
# 6. PrefixTuner output shape (B, P+T, D)
# ---------------------------------------------------------------------------


def test_prefix_tuner_output_shape():
    embed = _make_embed()
    cfg = _make_config()
    tuner = PrefixTuner(embed, cfg)
    token_ids = torch.randint(0, VOCAB, (B, T))
    out = tuner(token_ids)
    assert out.shape == (B, PREFIX_LEN + T, D_MODEL), (
        f"Expected ({B}, {PREFIX_LEN + T}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 7. Backbone embedding is frozen after PrefixTuner init
# ---------------------------------------------------------------------------


def test_backbone_embed_frozen_after_init():
    embed = _make_embed()
    cfg = _make_config()
    tuner = PrefixTuner(embed, cfg)
    for name, param in tuner.backbone_embed.named_parameters():
        assert not param.requires_grad, (
            f"Backbone param '{name}' should be frozen but requires_grad=True"
        )


# ---------------------------------------------------------------------------
# 8. get_trainable_params returns only prefix params (not backbone)
# ---------------------------------------------------------------------------


def test_get_trainable_params_only_prefix():
    embed = _make_embed()
    cfg = _make_config()
    tuner = PrefixTuner(embed, cfg)
    trainable = tuner.get_trainable_params()

    # All returned params must require grad
    for p in trainable:
        assert p.requires_grad, "get_trainable_params returned a frozen parameter"

    # Backbone weight must NOT be in the list
    backbone_ids = {id(p) for p in tuner.backbone_embed.parameters()}
    for p in trainable:
        assert id(p) not in backbone_ids, (
            "get_trainable_params included a backbone embedding parameter"
        )


# ---------------------------------------------------------------------------
# 9. freeze_model returns correct frozen count
# ---------------------------------------------------------------------------


def test_freeze_model_returns_correct_count():
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 2)

    net = TinyNet()
    # TinyNet has 4 parameters: a.weight, a.bias, b.weight, b.bias
    n_frozen = freeze_model(net)
    assert n_frozen == 4, f"Expected 4 frozen params, got {n_frozen}"
    for param in net.parameters():
        assert not param.requires_grad


# ---------------------------------------------------------------------------
# 10. count_trainable_params == 0 after freeze_model
# ---------------------------------------------------------------------------


def test_count_trainable_params_after_freeze_is_zero():
    net = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    freeze_model(net)
    assert count_trainable_params(net) == 0, "Expected 0 trainable params after freeze_model"


# ---------------------------------------------------------------------------
# 11. SoftPrompt output shape
# ---------------------------------------------------------------------------


def test_soft_prompt_output_shape():
    cfg = PromptTuningConfig(n_tokens=PREFIX_LEN, d_model=D_MODEL)
    sp = SoftPrompt(cfg)
    out = sp()
    assert out.shape == (PREFIX_LEN, D_MODEL), (
        f"Expected ({PREFIX_LEN}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 12. SoftPrompt requires grad
# ---------------------------------------------------------------------------


def test_soft_prompt_requires_grad():
    cfg = PromptTuningConfig(n_tokens=PREFIX_LEN, d_model=D_MODEL)
    sp = SoftPrompt(cfg)
    assert sp.embeddings.requires_grad, "SoftPrompt.embeddings should require grad"


# ---------------------------------------------------------------------------
# 13. Reparam MLP params differ from direct embedding params
# ---------------------------------------------------------------------------


def test_reparam_mlp_params_differ_from_direct():
    """Reparameterized PrefixEmbedding has different (MLP) parameters than the
    direct-parameter version, and no 'prefix' attribute."""
    cfg_direct = _make_config()
    cfg_reparam = _make_config(reparam_hidden_dim=8)

    pe_direct = PrefixEmbedding(cfg_direct)
    pe_reparam = PrefixEmbedding(cfg_reparam)

    # Direct variant has a 'prefix' parameter attribute
    assert hasattr(pe_direct, "prefix"), "Direct PrefixEmbedding should have 'prefix' attribute"
    # Reparam variant has an 'mlp' attribute instead
    assert hasattr(pe_reparam, "mlp"), "Reparam PrefixEmbedding should have 'mlp' attribute"
    assert not hasattr(pe_reparam, "prefix"), (
        "Reparam PrefixEmbedding should NOT have 'prefix' attribute"
    )

    # Both should produce the same output shape but via different mechanisms
    assert pe_direct().shape == pe_reparam().shape


# ---------------------------------------------------------------------------
# 14. PromptTuningConfig defaults
# ---------------------------------------------------------------------------


def test_prompt_tuning_config_defaults():
    cfg = PromptTuningConfig()
    assert cfg.n_tokens == 20
    assert cfg.d_model == 512
    assert cfg.init_text is None


# ---------------------------------------------------------------------------
# 15. PrefixTuner with reparam MLP produces correct shape
# ---------------------------------------------------------------------------


def test_prefix_tuner_reparam_output_shape():
    embed = _make_embed()
    cfg = _make_config(reparam_hidden_dim=8)
    tuner = PrefixTuner(embed, cfg)
    token_ids = torch.randint(0, VOCAB, (B, T))
    out = tuner(token_ids)
    assert out.shape == (B, PREFIX_LEN + T, D_MODEL), (
        f"Reparam PrefixTuner: expected ({B}, {PREFIX_LEN + T}, {D_MODEL}), got {out.shape}"
    )
