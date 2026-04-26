"""Tests for ThoughtTokenTrainer — pause token / latent reasoning training."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.thought_token_trainer import (
    PauseTokenEmbedding,
    ThoughtTokenConfig,
    ThoughtTokenInserter,
    ThoughtTokenTrainer,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 32
N_THOUGHT_TOKENS = 3
B = 2
T = 5  # prompt length
T_R = 4  # response length


# ---------------------------------------------------------------------------
# Minimal stub model: (B, T) -> (B, T, V)
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Single linear layer used as a stub language model."""

    def __init__(self, d_model: int = D_MODEL, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return ThoughtTokenConfig(
        n_thought_tokens=N_THOUGHT_TOKENS,
        thought_token_id=1,
        loss_on_thoughts=False,
    )


@pytest.fixture
def config_with_loss():
    return ThoughtTokenConfig(
        n_thought_tokens=N_THOUGHT_TOKENS,
        thought_token_id=1,
        loss_on_thoughts=True,
    )


@pytest.fixture
def inserter(config):
    return ThoughtTokenInserter(config)


@pytest.fixture
def inserter_with_loss(config_with_loss):
    return ThoughtTokenInserter(config_with_loss)


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(2, VOCAB_SIZE, (B, T))


@pytest.fixture
def response_ids():
    torch.manual_seed(1)
    return torch.randint(2, VOCAB_SIZE, (B, T_R))


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    return _TinyModel(d_model=D_MODEL, vocab_size=VOCAB_SIZE)


@pytest.fixture
def trainer(tiny_model, config, inserter):
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    return ThoughtTokenTrainer(
        model_fn=tiny_model,
        optimizer=optimizer,
        config=config,
        inserter=inserter,
    )


# ---------------------------------------------------------------------------
# 1. ThoughtTokenConfig stores fields correctly
# ---------------------------------------------------------------------------


def test_config_stores_fields():
    cfg = ThoughtTokenConfig(n_thought_tokens=7, thought_token_id=5, loss_on_thoughts=True)
    assert cfg.n_thought_tokens == 7
    assert cfg.thought_token_id == 5
    assert cfg.loss_on_thoughts is True


# ---------------------------------------------------------------------------
# 2. ThoughtTokenInserter.insert_thoughts output shape (B, T + n_thought_tokens)
# ---------------------------------------------------------------------------


def test_insert_thoughts_output_shape(inserter, input_ids):
    augmented_ids, thought_mask = inserter.insert_thoughts(input_ids)
    assert augmented_ids.shape == (B, T + N_THOUGHT_TOKENS)
    assert thought_mask.shape == (B, T + N_THOUGHT_TOKENS)


# ---------------------------------------------------------------------------
# 3. Thought mask has exactly n_thought_tokens True values per sequence
# ---------------------------------------------------------------------------


def test_thought_mask_true_count(inserter, input_ids):
    _, thought_mask = inserter.insert_thoughts(input_ids)
    for b in range(B):
        assert thought_mask[b].sum().item() == N_THOUGHT_TOKENS


# ---------------------------------------------------------------------------
# 4. Thought token positions contain correct token_id
# ---------------------------------------------------------------------------


def test_thought_positions_have_correct_token_id(inserter, input_ids, config):
    augmented_ids, thought_mask = inserter.insert_thoughts(input_ids)
    thought_tokens_in_aug = augmented_ids[thought_mask]
    assert (thought_tokens_in_aug == config.thought_token_id).all()


# ---------------------------------------------------------------------------
# 5. create_labels length = T + n_thought_tokens + T_r
# ---------------------------------------------------------------------------


def test_create_labels_length(inserter, input_ids, response_ids):
    augmented_ids, thought_mask = inserter.insert_thoughts(input_ids)
    labels = inserter.create_labels(augmented_ids, thought_mask, response_ids)
    assert labels.shape == (B, T + N_THOUGHT_TOKENS + T_R)


# ---------------------------------------------------------------------------
# 6. Prompt + thought positions have -100 labels (when loss_on_thoughts=False)
# ---------------------------------------------------------------------------


def test_prompt_and_thought_masked_when_no_loss(inserter, input_ids, response_ids):
    augmented_ids, thought_mask = inserter.insert_thoughts(input_ids)
    labels = inserter.create_labels(augmented_ids, thought_mask, response_ids)
    # Positions 0 .. T+K-1 (prompt + thoughts) should all be -100
    prefix_labels = labels[:, : T + N_THOUGHT_TOKENS]
    assert (prefix_labels == -100).all(), f"Expected all -100 in prefix, got: {prefix_labels}"


# ---------------------------------------------------------------------------
# 7. Response positions have correct labels
# ---------------------------------------------------------------------------


def test_response_positions_correct(inserter, input_ids, response_ids):
    augmented_ids, thought_mask = inserter.insert_thoughts(input_ids)
    labels = inserter.create_labels(augmented_ids, thought_mask, response_ids)
    response_labels = labels[:, T + N_THOUGHT_TOKENS :]
    assert (response_labels == response_ids).all()


# ---------------------------------------------------------------------------
# 8. loss_on_thoughts=True: thought positions NOT masked
# ---------------------------------------------------------------------------


def test_thought_positions_unmasked_when_loss_on_thoughts(
    inserter_with_loss, input_ids, response_ids
):
    augmented_ids, thought_mask = inserter_with_loss.insert_thoughts(input_ids)
    labels = inserter_with_loss.create_labels(augmented_ids, thought_mask, response_ids)
    # Thought positions are T .. T+K-1
    thought_labels = labels[:, T : T + N_THOUGHT_TOKENS]
    # None of the thought labels should be -100 when loss_on_thoughts=True
    assert (thought_labels != -100).all(), (
        f"Thought labels should not be -100 when loss_on_thoughts=True, got: {thought_labels}"
    )


# ---------------------------------------------------------------------------
# 9. PauseTokenEmbedding.forward output shape (n_thought_tokens, d_model)
# ---------------------------------------------------------------------------


def test_pause_embedding_forward_shape():
    emb = PauseTokenEmbedding(n_thought_tokens=N_THOUGHT_TOKENS, d_model=D_MODEL)
    out = emb()
    assert out.shape == (N_THOUGHT_TOKENS, D_MODEL)


# ---------------------------------------------------------------------------
# 10. get_token_embedding output shape (d_model,)
# ---------------------------------------------------------------------------


def test_pause_embedding_single_shape():
    emb = PauseTokenEmbedding(n_thought_tokens=N_THOUGHT_TOKENS, d_model=D_MODEL)
    out = emb.get_token_embedding(0)
    assert out.shape == (D_MODEL,)


# ---------------------------------------------------------------------------
# 11. Different positions give different embeddings (not all same)
# ---------------------------------------------------------------------------


def test_pause_embedding_different_positions():
    torch.manual_seed(7)
    emb = PauseTokenEmbedding(n_thought_tokens=N_THOUGHT_TOKENS, d_model=D_MODEL)
    all_embs = emb()  # (K, D)
    # Check that not all rows are identical
    first = all_embs[0]
    any_different = any(not torch.equal(all_embs[i], first) for i in range(1, N_THOUGHT_TOKENS))
    assert any_different, "Expected different embeddings for different thought positions"


# ---------------------------------------------------------------------------
# 12. ThoughtTokenTrainer.train_step returns expected keys
# ---------------------------------------------------------------------------


def test_train_step_returns_keys(trainer, input_ids, response_ids):
    result = trainer.train_step(input_ids, response_ids)
    assert "loss" in result
    assert "n_thought_tokens" in result
    assert "response_loss" in result


# ---------------------------------------------------------------------------
# 13. train_step loss is finite and positive
# ---------------------------------------------------------------------------


def test_train_step_loss_finite_positive(trainer, input_ids, response_ids):
    result = trainer.train_step(input_ids, response_ids)
    assert isinstance(result["loss"], float)
    assert result["loss"] > 0.0
    assert result["loss"] == result["loss"]  # not NaN
    import math

    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# 14. generate_with_thoughts output shape (max_new_tokens,)
# ---------------------------------------------------------------------------


def test_generate_with_thoughts_output_shape(trainer):
    torch.manual_seed(0)
    single_input = torch.randint(2, VOCAB_SIZE, (1, T))
    max_new = 6
    out = trainer.generate_with_thoughts(single_input, max_new_tokens=max_new)
    assert out.shape == (max_new,)


# ---------------------------------------------------------------------------
# 15. generate_with_thoughts token ids are non-negative
# ---------------------------------------------------------------------------


def test_generate_with_thoughts_nonnegative(trainer):
    torch.manual_seed(0)
    single_input = torch.randint(2, VOCAB_SIZE, (1, T))
    out = trainer.generate_with_thoughts(single_input, max_new_tokens=4)
    assert (out >= 0).all(), f"Generated token IDs must be non-negative, got: {out}"
