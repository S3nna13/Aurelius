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
    # New gradient-based soft prompt API
    PromptConfig,
    SoftPromptV2,
    PromptTuner,
    initialize_from_vocab,
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


# ===========================================================================
# New gradient-based Prompt Tuning / P-Tuning tests (14 tests)
# ===========================================================================

N_PROMPT = 4
SEQ_LEN = 8
BATCH = 2


def make_pt_model() -> AureliusTransformer:
    return AureliusTransformer(MODEL_CFG)


def make_prompt_config(**kwargs) -> PromptConfig:
    defaults = dict(n_prompt_tokens=N_PROMPT, init_method="random", init_vocab_ids=None)
    defaults.update(kwargs)
    return PromptConfig(**defaults)


def make_tuner(freeze: bool = True) -> tuple[PromptTuner, AureliusTransformer]:
    model = make_pt_model()
    cfg = make_prompt_config()
    opt = torch.optim.Adam([torch.zeros(1)], lr=1e-3)  # placeholder; setup() replaces params
    tuner = PromptTuner(model, cfg, opt)
    if freeze:
        tuner.setup()
    return tuner, model


# ---------------------------------------------------------------------------
# PT-1. PromptConfig defaults
# ---------------------------------------------------------------------------

def test_prompt_config_defaults():
    cfg = PromptConfig()
    assert cfg.n_prompt_tokens == 10
    assert cfg.init_method == "random"
    assert cfg.init_vocab_ids is None


# ---------------------------------------------------------------------------
# PT-2. SoftPromptV2 forward output shape (1, n_prompt_tokens, d_model)
# ---------------------------------------------------------------------------

def test_softpromptv2_forward_shape():
    cfg = make_prompt_config(n_prompt_tokens=N_PROMPT)
    sp = SoftPromptV2(cfg, MODEL_CFG.d_model)
    out = sp()
    assert out.shape == (1, N_PROMPT, MODEL_CFG.d_model), (
        f"Expected (1, {N_PROMPT}, {MODEL_CFG.d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# PT-3. SoftPromptV2 embeddings require grad
# ---------------------------------------------------------------------------

def test_softpromptv2_embeddings_require_grad():
    cfg = make_prompt_config()
    sp = SoftPromptV2(cfg, MODEL_CFG.d_model)
    assert sp.embeddings.requires_grad, "SoftPromptV2.embeddings should require grad"


# ---------------------------------------------------------------------------
# PT-4. SoftPromptV2 init_method='vocab' initializes from vocab embeddings
# ---------------------------------------------------------------------------

def test_softpromptv2_vocab_init():
    model = make_pt_model()
    vocab_ids = list(range(N_PROMPT))
    cfg = make_prompt_config(init_method="vocab", init_vocab_ids=vocab_ids)
    sp = SoftPromptV2(cfg, MODEL_CFG.d_model, embed_layer=model.embed)
    # Shape must match
    assert sp.embeddings.shape == (N_PROMPT, MODEL_CFG.d_model)
    # Values should match the frozen embedding weights at those ids
    expected = model.embed.weight[vocab_ids].detach()
    assert torch.allclose(sp.embeddings.data, expected), (
        "vocab init should copy embedding weights for the given ids"
    )


# ---------------------------------------------------------------------------
# PT-5. initialize_from_vocab returns correct shape
# ---------------------------------------------------------------------------

def test_initialize_from_vocab_shape():
    model = make_pt_model()
    vocab_ids = [0, 5, 10, 15]
    result = initialize_from_vocab(model.embed, vocab_ids)
    assert result.shape == (len(vocab_ids), MODEL_CFG.d_model), (
        f"Expected ({len(vocab_ids)}, {MODEL_CFG.d_model}), got {result.shape}"
    )


# ---------------------------------------------------------------------------
# PT-6. PromptTuner.setup freezes model params
# ---------------------------------------------------------------------------

def test_prompt_tuner_setup_freezes_model():
    tuner, model = make_tuner(freeze=True)
    for name, param in model.named_parameters():
        assert not param.requires_grad, (
            f"Model param {name!r} should be frozen after setup(), got requires_grad=True"
        )


# ---------------------------------------------------------------------------
# PT-7. PromptTuner.setup makes prompt params trainable
# ---------------------------------------------------------------------------

def test_prompt_tuner_setup_prompt_trainable():
    tuner, _ = make_tuner(freeze=True)
    assert tuner.soft_prompt is not None, "soft_prompt should be initialized by setup()"
    assert tuner.soft_prompt.embeddings.requires_grad, (
        "soft_prompt.embeddings should require grad after setup()"
    )


# ---------------------------------------------------------------------------
# PT-8. forward_with_prompt output shape (B, n_prompt_tokens + T, vocab_size)
# ---------------------------------------------------------------------------

def test_forward_with_prompt_shape():
    tuner, _ = make_tuner()
    input_ids = torch.randint(0, MODEL_CFG.vocab_size, (BATCH, SEQ_LEN))
    logits = tuner.forward_with_prompt(input_ids)
    expected = (BATCH, N_PROMPT + SEQ_LEN, MODEL_CFG.vocab_size)
    assert logits.shape == expected, (
        f"Expected {expected}, got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# PT-9. forward_with_prompt is differentiable through prompt
# ---------------------------------------------------------------------------

def test_forward_with_prompt_differentiable():
    tuner, _ = make_tuner()
    input_ids = torch.randint(0, MODEL_CFG.vocab_size, (BATCH, SEQ_LEN))
    logits = tuner.forward_with_prompt(input_ids)
    loss = logits.mean()
    loss.backward()
    assert tuner.soft_prompt.embeddings.grad is not None, (
        "Gradient should flow back to soft_prompt.embeddings"
    )
    assert tuner.soft_prompt.embeddings.grad.shape == tuner.soft_prompt.embeddings.shape


# ---------------------------------------------------------------------------
# PT-10. train_step returns required keys
# ---------------------------------------------------------------------------

def test_train_step_returns_required_keys():
    tuner, _ = make_tuner()
    input_ids = torch.randint(0, MODEL_CFG.vocab_size, (BATCH, SEQ_LEN))
    result = tuner.train_step(input_ids)
    assert "loss" in result, "train_step result must contain 'loss'"
    assert "n_prompt_tokens" in result, "train_step result must contain 'n_prompt_tokens'"


# ---------------------------------------------------------------------------
# PT-11. train_step loss is finite
# ---------------------------------------------------------------------------

def test_train_step_loss_finite():
    tuner, _ = make_tuner()
    input_ids = torch.randint(0, MODEL_CFG.vocab_size, (BATCH, SEQ_LEN))
    result = tuner.train_step(input_ids)
    assert torch.isfinite(torch.tensor(result["loss"])), (
        f"train_step loss should be finite, got {result['loss']}"
    )


# ---------------------------------------------------------------------------
# PT-12. train_step n_prompt_tokens matches config
# ---------------------------------------------------------------------------

def test_train_step_n_prompt_tokens_matches_config():
    tuner, _ = make_tuner()
    input_ids = torch.randint(0, MODEL_CFG.vocab_size, (BATCH, SEQ_LEN))
    result = tuner.train_step(input_ids)
    assert result["n_prompt_tokens"] == N_PROMPT, (
        f"Expected n_prompt_tokens={N_PROMPT}, got {result['n_prompt_tokens']}"
    )


# ---------------------------------------------------------------------------
# PT-13. Gradient flows to prompt but NOT to frozen model params
# ---------------------------------------------------------------------------

def test_gradient_flows_to_prompt_not_model():
    tuner, model = make_tuner()
    input_ids = torch.randint(0, MODEL_CFG.vocab_size, (BATCH, SEQ_LEN))
    logits = tuner.forward_with_prompt(input_ids)
    loss = logits.mean()
    loss.backward()

    # Prompt grad must exist
    assert tuner.soft_prompt.embeddings.grad is not None, (
        "Gradient must exist for soft_prompt.embeddings"
    )

    # No model param should have a gradient (they are frozen / detached)
    for name, param in model.named_parameters():
        assert param.grad is None, (
            f"Frozen model param {name!r} should have no gradient, but got grad != None"
        )


# ---------------------------------------------------------------------------
# PT-14. train_step updates prompt embeddings but not model weights
# ---------------------------------------------------------------------------

def test_train_step_updates_prompt_not_model():
    tuner, model = make_tuner()

    # Snapshot initial values
    prompt_before = tuner.soft_prompt.embeddings.data.clone()
    model_weights_before = {
        name: param.data.clone()
        for name, param in model.named_parameters()
    }

    input_ids = torch.randint(0, MODEL_CFG.vocab_size, (BATCH, SEQ_LEN))
    tuner.train_step(input_ids)

    # Prompt should change
    assert not torch.allclose(tuner.soft_prompt.embeddings.data, prompt_before), (
        "train_step should update soft_prompt.embeddings"
    )

    # Model weights should be unchanged
    for name, param in model.named_parameters():
        assert torch.allclose(param.data, model_weights_before[name]), (
            f"Model param {name!r} should not change during train_step"
        )
