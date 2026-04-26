"""Tests for src/model/vocab_expansion.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.model.vocab_expansion import (
    VocabExpander,
    VocabExpansionConfig,
    expand_embedding,
    expand_lm_head,
    verify_expansion,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
N_NEW = 16
D_MODEL = 64


@pytest.fixture
def small_config() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=512,
    )


@pytest.fixture
def model(small_config: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(small_config)


@pytest.fixture
def base_embedding() -> nn.Embedding:
    torch.manual_seed(0)
    emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
    nn.init.normal_(emb.weight, mean=0.0, std=0.02)
    return emb


@pytest.fixture
def base_lm_head() -> nn.Linear:
    torch.manual_seed(1)
    head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
    nn.init.normal_(head.weight, mean=0.0, std=0.02)
    return head


# ---------------------------------------------------------------------------
# 1. VocabExpansionConfig defaults
# ---------------------------------------------------------------------------


def test_vocab_expansion_config_defaults():
    cfg = VocabExpansionConfig()
    assert cfg.init_strategy == "mean"
    assert cfg.freeze_existing is True
    assert cfg.new_token_lr_scale == 10.0
    assert cfg.n_init_similar_tokens == 5


# ---------------------------------------------------------------------------
# 2. expand_embedding output has correct vocab size
# ---------------------------------------------------------------------------


def test_expand_embedding_correct_vocab_size(base_embedding: nn.Embedding):
    cfg = VocabExpansionConfig(init_strategy="mean")
    new_emb = expand_embedding(base_embedding, N_NEW, cfg)
    assert new_emb.weight.shape[0] == VOCAB_SIZE + N_NEW
    assert new_emb.weight.shape[1] == D_MODEL


# ---------------------------------------------------------------------------
# 3. expand_embedding preserves old weights exactly
# ---------------------------------------------------------------------------


def test_expand_embedding_preserves_old_weights(base_embedding: nn.Embedding):
    cfg = VocabExpansionConfig(init_strategy="mean")
    old_weight = base_embedding.weight.data.clone()
    new_emb = expand_embedding(base_embedding, N_NEW, cfg)
    assert torch.allclose(new_emb.weight[:VOCAB_SIZE].data, old_weight), (
        "Old embedding rows must be preserved exactly after expansion."
    )


# ---------------------------------------------------------------------------
# 4. expand_embedding strategy="mean" — new rows ≈ mean of old rows
# ---------------------------------------------------------------------------


def test_expand_embedding_mean_strategy(base_embedding: nn.Embedding):
    cfg = VocabExpansionConfig(init_strategy="mean")
    expected_mean = base_embedding.weight.data.mean(dim=0)
    new_emb = expand_embedding(base_embedding, N_NEW, cfg)
    new_rows = new_emb.weight[VOCAB_SIZE:].data  # (N_NEW, D_MODEL)
    for i in range(N_NEW):
        assert torch.allclose(new_rows[i], expected_mean, atol=1e-6), (
            f"New row {i} should equal the mean of existing embeddings."
        )


# ---------------------------------------------------------------------------
# 5. expand_embedding strategy="zeros" — new rows = 0
# ---------------------------------------------------------------------------


def test_expand_embedding_zeros_strategy(base_embedding: nn.Embedding):
    cfg = VocabExpansionConfig(init_strategy="zeros")
    new_emb = expand_embedding(base_embedding, N_NEW, cfg)
    new_rows = new_emb.weight[VOCAB_SIZE:].data
    assert torch.all(new_rows == 0.0), "New rows should be zeros with 'zeros' strategy."


# ---------------------------------------------------------------------------
# 6. expand_embedding strategy="random" — new rows have non-zero variance
# ---------------------------------------------------------------------------


def test_expand_embedding_random_strategy_has_variance(base_embedding: nn.Embedding):
    torch.manual_seed(99)
    cfg = VocabExpansionConfig(init_strategy="random")
    new_emb = expand_embedding(base_embedding, N_NEW, cfg)
    new_rows = new_emb.weight[VOCAB_SIZE:].data  # (N_NEW, D_MODEL)
    # Variance across rows should be non-trivial
    var = new_rows.var().item()
    assert var > 0.0, "Random strategy should produce non-zero variance across new rows."


# ---------------------------------------------------------------------------
# 7. expand_lm_head output has correct output dim
# ---------------------------------------------------------------------------


def test_expand_lm_head_correct_output_dim(base_lm_head: nn.Linear):
    cfg = VocabExpansionConfig(init_strategy="mean")
    new_head = expand_lm_head(base_lm_head, N_NEW, cfg)
    assert new_head.weight.shape[0] == VOCAB_SIZE + N_NEW
    assert new_head.weight.shape[1] == D_MODEL


# ---------------------------------------------------------------------------
# 8. expand_lm_head preserves old output rows exactly
# ---------------------------------------------------------------------------


def test_expand_lm_head_preserves_old_rows(base_lm_head: nn.Linear):
    cfg = VocabExpansionConfig(init_strategy="mean")
    old_weight = base_lm_head.weight.data.clone()
    new_head = expand_lm_head(base_lm_head, N_NEW, cfg)
    assert torch.allclose(new_head.weight[:VOCAB_SIZE].data, old_weight), (
        "Old LM head output rows must be preserved exactly after expansion."
    )


# ---------------------------------------------------------------------------
# 9. verify_expansion returns True for old_weights_preserved when correct
# ---------------------------------------------------------------------------


def test_verify_expansion_old_weights_preserved_true(base_embedding: nn.Embedding):
    cfg = VocabExpansionConfig(init_strategy="mean")
    new_emb = expand_embedding(base_embedding, N_NEW, cfg)
    results = verify_expansion(base_embedding, new_emb)
    assert results["old_weights_preserved"] is True
    assert results["size_correct"] is True
    assert results["no_nan"] is True


# ---------------------------------------------------------------------------
# 10. verify_expansion returns False for old_weights_preserved when weights differ
# ---------------------------------------------------------------------------


def test_verify_expansion_old_weights_preserved_false(base_embedding: nn.Embedding):
    cfg = VocabExpansionConfig(init_strategy="mean")
    new_emb = expand_embedding(base_embedding, N_NEW, cfg)
    # Corrupt a row in new_emb
    with torch.no_grad():
        new_emb.weight[0] = new_emb.weight[0] + 999.0
    results = verify_expansion(base_embedding, new_emb)
    assert results["old_weights_preserved"] is False


# ---------------------------------------------------------------------------
# 11. VocabExpander.expand increases embedding vocab size
# ---------------------------------------------------------------------------


def test_vocab_expander_increases_embedding_size(model: AureliusTransformer):
    old_vocab = model.embed.weight.shape[0]
    cfg = VocabExpansionConfig(init_strategy="mean", freeze_existing=False)
    expander = VocabExpander(model, cfg)
    expander.expand(N_NEW)
    new_vocab = model.embed.weight.shape[0]
    assert new_vocab == old_vocab + N_NEW, (
        f"Expected vocab size {old_vocab + N_NEW}, got {new_vocab}."
    )


# ---------------------------------------------------------------------------
# 12. VocabExpander.expand — model runs forward successfully after expansion
# ---------------------------------------------------------------------------


def test_vocab_expander_forward_after_expansion(model: AureliusTransformer):
    cfg = VocabExpansionConfig(init_strategy="mean", freeze_existing=False)
    expander = VocabExpander(model, cfg)
    expander.expand(N_NEW)

    new_vocab = model.embed.weight.shape[0]
    # Test with tokens spanning the new vocabulary (including new token ids)
    input_ids = torch.randint(0, new_vocab, (2, 8))
    with torch.no_grad():
        loss, logits, pkv = model(input_ids)

    assert logits.shape == (2, 8, new_vocab), (
        f"Expected logits shape (2, 8, {new_vocab}), got {logits.shape}."
    )
    assert loss is None  # no labels provided
    assert len(pkv) == model.config.n_layers


# ---------------------------------------------------------------------------
# 13. VocabExpander.get_param_groups returns two groups with different LRs
# ---------------------------------------------------------------------------


def test_vocab_expander_get_param_groups(model: AureliusTransformer):
    base_lr = 1e-4
    cfg = VocabExpansionConfig(
        init_strategy="mean",
        freeze_existing=False,
        new_token_lr_scale=10.0,
    )
    expander = VocabExpander(model, cfg)
    expander.expand(N_NEW)

    groups = expander.get_param_groups(base_lr)
    assert len(groups) == 2, "Should return exactly two param groups."

    lrs = {g["lr"] for g in groups}
    assert len(lrs) == 2, "The two groups should have different learning rates."
    assert base_lr in lrs, f"base_lr {base_lr} should be one of the LRs."
    assert base_lr * cfg.new_token_lr_scale in lrs, (
        f"scaled LR {base_lr * cfg.new_token_lr_scale} should be in param groups."
    )

    # Both groups must have at least one parameter
    for g in groups:
        assert len(g["params"]) > 0, "Each param group should have at least one parameter."
