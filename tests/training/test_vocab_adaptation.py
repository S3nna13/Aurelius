"""Tests for vocab_adaptation: vocabulary expansion and domain adaptation."""

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.vocab_adaptation import (
    VocabAdaptationTrainer,
    VocabAdaptConfig,
    analyze_token_usage,
    compute_token_similarity,
    expand_vocabulary,
    initialize_new_token_embedding,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(0)
    model = AureliusTransformer(small_cfg)
    # Untie embeddings so expand_vocabulary can replace them independently
    model.config.tie_embeddings = False
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())
    return model


@pytest.fixture
def embed_weight():
    torch.manual_seed(7)
    return torch.randn(256, 64)


# ---------------------------------------------------------------------------
# 1. VocabAdaptConfig defaults
# ---------------------------------------------------------------------------


def test_vocab_adapt_config_defaults():
    cfg = VocabAdaptConfig()
    assert cfg.new_tokens == []
    assert cfg.init_method == "mean"
    assert cfg.freeze_old_embeddings is False
    assert cfg.scale_lr_new == 10.0


# ---------------------------------------------------------------------------
# 2. compute_token_similarity — shapes
# ---------------------------------------------------------------------------


def test_compute_token_similarity_shapes(embed_weight):
    query = embed_weight[0]
    top_k = 5
    sims, indices = compute_token_similarity(embed_weight, query, top_k=top_k)
    assert sims.shape == (top_k,)
    assert indices.shape == (top_k,)


# ---------------------------------------------------------------------------
# 3. compute_token_similarity — values in [-1, 1]
# ---------------------------------------------------------------------------


def test_compute_token_similarity_range(embed_weight):
    query = embed_weight[10]
    sims, _ = compute_token_similarity(embed_weight, query, top_k=5)
    assert (sims >= -1.0 - 1e-5).all()
    assert (sims <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# 4. compute_token_similarity — self-similarity ≈ 1
# ---------------------------------------------------------------------------


def test_compute_token_similarity_self(embed_weight):
    query = embed_weight[3]
    sims, indices = compute_token_similarity(embed_weight, query, top_k=1)
    assert sims[0].item() == pytest.approx(1.0, abs=1e-5)
    assert indices[0].item() == 3


# ---------------------------------------------------------------------------
# 5. initialize_new_token_embedding — mean method returns shape (D,)
# ---------------------------------------------------------------------------


def test_initialize_new_token_embedding_mean_shape(embed_weight):
    result = initialize_new_token_embedding(embed_weight, method="mean")
    assert result.shape == (embed_weight.shape[1],)


# ---------------------------------------------------------------------------
# 6. initialize_new_token_embedding — random method returns different values
# ---------------------------------------------------------------------------


def test_initialize_new_token_embedding_random_differs(embed_weight):
    torch.manual_seed(0)
    r1 = initialize_new_token_embedding(embed_weight, method="random")
    r2 = initialize_new_token_embedding(embed_weight, method="random")
    assert not torch.allclose(r1, r2)


# ---------------------------------------------------------------------------
# 7. expand_vocabulary increases embed weight rows
# ---------------------------------------------------------------------------


def test_expand_vocabulary_embed_rows(small_model):
    old_vocab = small_model.embed.weight.shape[0]
    n_new = 8
    expand_vocabulary(small_model, n_new_tokens=n_new, method="mean")
    assert small_model.embed.weight.shape[0] == old_vocab + n_new


# ---------------------------------------------------------------------------
# 8. expand_vocabulary increases lm_head weight rows
# ---------------------------------------------------------------------------


def test_expand_vocabulary_lm_head_rows(small_model):
    old_vocab = small_model.lm_head.weight.shape[0]
    n_new = 8
    expand_vocabulary(small_model, n_new_tokens=n_new, method="mean")
    assert small_model.lm_head.weight.shape[0] == old_vocab + n_new


# ---------------------------------------------------------------------------
# 9. VocabAdaptationTrainer.build_optimizer returns Optimizer
# ---------------------------------------------------------------------------


def test_build_optimizer_type(small_model):
    base_vocab = 256
    n_new = 8
    expand_vocabulary(small_model, n_new_tokens=n_new, method="mean")
    cfg = VocabAdaptConfig()
    trainer = VocabAdaptationTrainer(small_model, cfg, base_vocab_size=base_vocab)
    opt = trainer.build_optimizer(base_lr=1e-4)
    assert isinstance(opt, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# 10. VocabAdaptationTrainer.build_optimizer has 2 param groups
# ---------------------------------------------------------------------------


def test_build_optimizer_param_groups(small_model):
    base_vocab = 256
    n_new = 8
    expand_vocabulary(small_model, n_new_tokens=n_new, method="mean")
    cfg = VocabAdaptConfig()
    trainer = VocabAdaptationTrainer(small_model, cfg, base_vocab_size=base_vocab)
    opt = trainer.build_optimizer(base_lr=1e-4)
    assert len(opt.param_groups) == 2


# ---------------------------------------------------------------------------
# 11. VocabAdaptationTrainer.train_step returns required keys
# ---------------------------------------------------------------------------


def test_train_step_keys(small_model):
    base_vocab = 256
    n_new = 4
    expand_vocabulary(small_model, n_new_tokens=n_new, method="mean")
    cfg = VocabAdaptConfig()
    trainer = VocabAdaptationTrainer(small_model, cfg, base_vocab_size=base_vocab)
    opt = trainer.build_optimizer()
    input_ids = torch.randint(0, base_vocab + n_new, (2, 16))
    opt.zero_grad()
    result = trainer.train_step(input_ids)
    opt.step()
    assert "loss" in result
    assert "n_new_tokens" in result


# ---------------------------------------------------------------------------
# 12. VocabAdaptationTrainer.train_step loss is finite
# ---------------------------------------------------------------------------


def test_train_step_loss_finite(small_model):
    base_vocab = 256
    n_new = 4
    expand_vocabulary(small_model, n_new_tokens=n_new, method="mean")
    cfg = VocabAdaptConfig()
    trainer = VocabAdaptationTrainer(small_model, cfg, base_vocab_size=base_vocab)
    opt = trainer.build_optimizer()
    input_ids = torch.randint(0, base_vocab + n_new, (2, 16))
    opt.zero_grad()
    result = trainer.train_step(input_ids)
    opt.step()
    assert isinstance(result["loss"], float)
    assert torch.isfinite(torch.tensor(result["loss"]))


# ---------------------------------------------------------------------------
# 13. analyze_token_usage returns required keys
# ---------------------------------------------------------------------------


def test_analyze_token_usage_keys():
    input_ids = torch.randint(0, 100, (4, 32))
    result = analyze_token_usage(input_ids, vocab_size=100, top_k=10)
    assert "token_freq" in result
    assert "coverage" in result
    assert "top_tokens" in result


# ---------------------------------------------------------------------------
# 14. analyze_token_usage coverage in [0, 1]
# ---------------------------------------------------------------------------


def test_analyze_token_usage_coverage_range():
    input_ids = torch.randint(0, 50, (2, 64))
    result = analyze_token_usage(input_ids, vocab_size=100, top_k=10)
    assert 0.0 <= result["coverage"] <= 1.0


# ---------------------------------------------------------------------------
# 15. analyze_token_usage top_tokens length <= top_k
# ---------------------------------------------------------------------------


def test_analyze_token_usage_top_tokens_length():
    # Use a small vocab so fewer than top_k tokens exist
    input_ids = torch.randint(0, 5, (2, 32))
    top_k = 20
    result = analyze_token_usage(input_ids, vocab_size=10, top_k=top_k)
    assert len(result["top_tokens"]) <= top_k
