"""Tests for soft prompt tuning and Gist-style prompt compression."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.alignment.prompt_tuning import (
    PromptTuningConfig,
    SoftPrompt,
    SoftPromptModel,
    GistCompressor,
    get_prompt_tuning_optimizer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

MODEL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

N_SOFT = 4
COMPRESS_RATIO = 4


def make_backbone() -> AureliusTransformer:
    return AureliusTransformer(MODEL_CFG)


def make_config(**kwargs) -> PromptTuningConfig:
    defaults = dict(
        n_soft_tokens=N_SOFT,
        init_strategy="random",
        lr_multiplier=100.0,
        freeze_backbone=True,
        compress_ratio=COMPRESS_RATIO,
    )
    defaults.update(kwargs)
    return PromptTuningConfig(**defaults)


def make_soft_prompt_model(freeze: bool = True) -> SoftPromptModel:
    backbone = make_backbone()
    cfg = make_config(freeze_backbone=freeze)
    return SoftPromptModel(backbone, cfg, backbone.embed)


# ---------------------------------------------------------------------------
# 1. PromptTuningConfig defaults
# ---------------------------------------------------------------------------

def test_prompt_tuning_config_defaults():
    cfg = PromptTuningConfig()
    assert cfg.n_soft_tokens == 20
    assert cfg.init_strategy == "random"
    assert cfg.lr_multiplier == 100.0
    assert cfg.freeze_backbone is True
    assert cfg.compress_ratio == 4


# ---------------------------------------------------------------------------
# 2. SoftPrompt forward output shape (B, n_tokens, d_model)
# ---------------------------------------------------------------------------

def test_soft_prompt_forward_shape():
    B, n_tokens, d_model = 3, N_SOFT, MODEL_CFG.d_model
    sp = SoftPrompt(n_tokens, d_model, init_strategy="random")
    out = sp(B)
    assert out.shape == (B, n_tokens, d_model), (
        f"Expected ({B}, {n_tokens}, {d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 3. SoftPrompt embeddings require grad
# ---------------------------------------------------------------------------

def test_soft_prompt_embeddings_require_grad():
    sp = SoftPrompt(N_SOFT, MODEL_CFG.d_model, init_strategy="random")
    assert sp.embeddings.requires_grad, "SoftPrompt.embeddings should require grad"


# ---------------------------------------------------------------------------
# 4. SoftPrompt vocab_sample init shape matches
# ---------------------------------------------------------------------------

def test_soft_prompt_vocab_sample_init_shape():
    vocab_emb = nn.Embedding(MODEL_CFG.vocab_size, MODEL_CFG.d_model)
    sp = SoftPrompt(N_SOFT, MODEL_CFG.d_model, init_strategy="vocab_sample", vocab_embeddings=vocab_emb)
    assert sp.embeddings.shape == (N_SOFT, MODEL_CFG.d_model), (
        f"Expected ({N_SOFT}, {MODEL_CFG.d_model}), got {sp.embeddings.shape}"
    )


# ---------------------------------------------------------------------------
# 5. SoftPromptModel forward output is valid (logits not None)
# ---------------------------------------------------------------------------

def test_soft_prompt_model_forward_valid():
    model = make_soft_prompt_model()
    input_ids = torch.randint(0, MODEL_CFG.vocab_size, (2, 8))
    loss, logits, pkv = model(input_ids)
    assert logits is not None, "logits should not be None"
    assert logits.ndim == 3, f"logits should be 3D, got {logits.ndim}D"
    # Sequence dim = n_soft + T = 4 + 8 = 12
    assert logits.shape[0] == 2
    assert logits.shape[1] == N_SOFT + 8
    assert logits.shape[2] == MODEL_CFG.vocab_size


# ---------------------------------------------------------------------------
# 6. SoftPromptModel freeze_backbone freezes backbone params
# ---------------------------------------------------------------------------

def test_soft_prompt_model_freeze_backbone():
    model = make_soft_prompt_model(freeze=True)
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, (
            f"Backbone param {name!r} should be frozen but requires_grad=True"
        )


# ---------------------------------------------------------------------------
# 7. SoftPromptModel get_trainable_params returns only soft prompt params
# ---------------------------------------------------------------------------

def test_soft_prompt_model_get_trainable_params():
    model = make_soft_prompt_model(freeze=True)
    trainable = model.get_trainable_params()

    # Should contain exactly one parameter: the soft prompt embeddings
    assert len(trainable) == 1, f"Expected 1 trainable param, got {len(trainable)}"
    assert trainable[0] is model.soft_prompt.embeddings, (
        "Trainable param should be soft_prompt.embeddings"
    )


# ---------------------------------------------------------------------------
# 8. GistCompressor.compress output shape (B, T//compress_ratio, d_model)
# ---------------------------------------------------------------------------

def test_gist_compressor_output_shape():
    B, T, d_model = 2, 16, MODEL_CFG.d_model
    compressor = GistCompressor(d_model, compress_ratio=COMPRESS_RATIO)
    hidden = torch.randn(B, T, d_model)
    out = compressor.compress(hidden)
    expected = (B, T // COMPRESS_RATIO, d_model)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"


# ---------------------------------------------------------------------------
# 9. GistCompressor.compress with T=8, ratio=4 gives T//4=2 output tokens
# ---------------------------------------------------------------------------

def test_gist_compressor_t8_ratio4():
    B, T, d_model = 3, 8, MODEL_CFG.d_model
    compressor = GistCompressor(d_model, compress_ratio=4)
    hidden = torch.randn(B, T, d_model)
    out = compressor.compress(hidden)
    assert out.shape == (B, 2, d_model), (
        f"Expected (B=3, 2, d_model={d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 10. GistCompressor.compress handles T not divisible by ratio (truncate)
# ---------------------------------------------------------------------------

def test_gist_compressor_truncate():
    B, d_model = 2, MODEL_CFG.d_model
    T = 9  # not divisible by 4; should give 9//4 = 2 chunks
    compressor = GistCompressor(d_model, compress_ratio=4)
    hidden = torch.randn(B, T, d_model)
    out = compressor.compress(hidden)
    expected_chunks = T // 4  # 2
    assert out.shape == (B, expected_chunks, d_model), (
        f"Expected ({B}, {expected_chunks}, {d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 11. get_prompt_tuning_optimizer creates optimizer
# ---------------------------------------------------------------------------

def test_get_prompt_tuning_optimizer_creates():
    model = make_soft_prompt_model()
    cfg = make_config()
    optimizer = get_prompt_tuning_optimizer(model, base_lr=1e-4, config=cfg)
    assert isinstance(optimizer, torch.optim.Optimizer), (
        f"Expected Optimizer, got {type(optimizer)}"
    )


# ---------------------------------------------------------------------------
# 12. get_prompt_tuning_optimizer has correct param count (soft prompt only)
# ---------------------------------------------------------------------------

def test_get_prompt_tuning_optimizer_param_count():
    model = make_soft_prompt_model()
    cfg = make_config()
    optimizer = get_prompt_tuning_optimizer(model, base_lr=1e-4, config=cfg)

    # Collect all parameter tensors in the optimizer
    opt_params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(opt_params) == 1, f"Expected 1 param group tensor, got {len(opt_params)}"
    assert opt_params[0] is model.soft_prompt.embeddings, (
        "Optimizer should contain only soft_prompt.embeddings"
    )


# ---------------------------------------------------------------------------
# 13. SoftPrompt random init has non-zero variance
# ---------------------------------------------------------------------------

def test_soft_prompt_random_init_variance():
    sp = SoftPrompt(N_SOFT, MODEL_CFG.d_model, init_strategy="random")
    var = sp.embeddings.var().item()
    assert var > 0.0, f"Random init should have non-zero variance, got var={var}"
