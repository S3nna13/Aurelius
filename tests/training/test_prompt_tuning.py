"""Tests for src/training/prompt_tuning.py."""

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.prompt_tuning import (
    PromptTuningConfig,
    SoftPromptEmbedding,
    PromptTunedModel,
    PrefixTuningModel,
    optimize_prompt,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SMALL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

N_PROMPT = 4
BATCH = 1
SEQ_LEN = 8


@pytest.fixture
def base_model():
    torch.manual_seed(0)
    return AureliusTransformer(SMALL_CFG)


@pytest.fixture
def pt_config():
    return PromptTuningConfig(n_prompt_tokens=N_PROMPT, prompt_init="random")


@pytest.fixture
def vocab_pt_config():
    return PromptTuningConfig(n_prompt_tokens=N_PROMPT, prompt_init="vocab_sample")


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, SMALL_CFG.vocab_size, (BATCH, SEQ_LEN))


@pytest.fixture
def labels(input_ids):
    return input_ids.clone()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_prompt_tuning_config_defaults():
    cfg = PromptTuningConfig()
    assert cfg.n_prompt_tokens == 20
    assert cfg.prompt_init == "random"
    assert cfg.prefix_tuning is False


def test_soft_prompt_embedding_shape(base_model, pt_config, input_ids):
    """Output shape should be (B, n_prompt_tokens + T, D)."""
    embed_layer = base_model.embed
    spe = SoftPromptEmbedding(pt_config, embed_layer)
    out = spe(input_ids)
    B, T, D = out.shape
    assert B == BATCH
    assert T == N_PROMPT + SEQ_LEN
    assert D == SMALL_CFG.d_model


def test_soft_prompt_embedding_random_init(base_model, pt_config):
    """soft_prompt should have shape (n_prompt_tokens, d_model)."""
    embed_layer = base_model.embed
    spe = SoftPromptEmbedding(pt_config, embed_layer)
    assert spe.soft_prompt.shape == (N_PROMPT, SMALL_CFG.d_model)


def test_soft_prompt_embedding_vocab_init(base_model, vocab_pt_config):
    """vocab_sample init should work without error and produce correct shape."""
    torch.manual_seed(0)
    embed_layer = base_model.embed
    spe = SoftPromptEmbedding(vocab_pt_config, embed_layer)
    assert spe.soft_prompt.shape == (N_PROMPT, SMALL_CFG.d_model)
    # soft_prompt must be a Parameter requiring grad
    assert spe.soft_prompt.requires_grad


def test_prompt_tuned_model_base_frozen(base_model, pt_config):
    """All base model parameters should be frozen after wrapping."""
    model = PromptTunedModel(base_model, pt_config)
    for name, param in model.base_model.named_parameters():
        assert not param.requires_grad, f"Base param {name} should be frozen"


def test_prompt_tuned_model_prompt_trainable(base_model, pt_config):
    """The soft prompt parameter should require grad."""
    model = PromptTunedModel(base_model, pt_config)
    assert model.soft_prompt_embedding.soft_prompt.requires_grad


def test_prompt_tuned_model_num_trainable(base_model, pt_config):
    """num_trainable_params should equal n_prompt_tokens * d_model."""
    model = PromptTunedModel(base_model, pt_config)
    expected = N_PROMPT * SMALL_CFG.d_model
    assert model.num_trainable_params() == expected


def test_prompt_tuned_model_forward_shape(base_model, pt_config, input_ids):
    """logits shape should be (B, S, V) where S >= T (S = T + n_prompt_tokens)."""
    model = PromptTunedModel(base_model, pt_config)
    with torch.no_grad():
        loss, logits, pkv = model(input_ids)
    B, S, V = logits.shape
    assert B == BATCH
    assert S >= SEQ_LEN, f"Expected S >= {SEQ_LEN}, got {S}"
    assert V == SMALL_CFG.vocab_size


def test_prefix_tuning_model_param_count(base_model):
    """prefix_keys should have n_layers tensors of shape (n_prefix, n_kv_heads, head_dim)."""
    n_prefix = 6
    n_layers = SMALL_CFG.n_layers
    n_kv_heads = SMALL_CFG.n_kv_heads
    head_dim = SMALL_CFG.head_dim

    model = PrefixTuningModel(
        base_model,
        n_prefix_tokens=n_prefix,
        d_model=SMALL_CFG.d_model,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )
    assert len(model.prefix_keys) == n_layers
    assert len(model.prefix_values) == n_layers
    for i in range(n_layers):
        k, v = model.get_prefix_kv(i)
        assert k.shape == (n_prefix, n_kv_heads, head_dim), f"Layer {i} keys shape wrong"
        assert v.shape == (n_prefix, n_kv_heads, head_dim), f"Layer {i} values shape wrong"


def test_prefix_tuning_model_base_frozen(base_model):
    """All base model parameters should be frozen in PrefixTuningModel."""
    model = PrefixTuningModel(
        base_model,
        n_prefix_tokens=4,
        d_model=SMALL_CFG.d_model,
        n_layers=SMALL_CFG.n_layers,
        n_kv_heads=SMALL_CFG.n_kv_heads,
        head_dim=SMALL_CFG.head_dim,
    )
    for name, param in model.base_model.named_parameters():
        assert not param.requires_grad, f"Base param {name} should be frozen"


def test_optimize_prompt_returns_losses(base_model, pt_config, input_ids, labels):
    """optimize_prompt should return a list of length n_steps."""
    torch.manual_seed(0)
    model = PromptTunedModel(base_model, pt_config)
    n_steps = 3
    losses = optimize_prompt(model, input_ids, labels, n_steps=n_steps, lr=0.01)
    assert isinstance(losses, list)
    assert len(losses) == n_steps
    assert all(isinstance(v, float) for v in losses)
