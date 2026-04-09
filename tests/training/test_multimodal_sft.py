"""Tests for multimodal supervised fine-tuning module."""
from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.multimodal_sft import (
    MultimodalSFTConfig,
    MultimodalExample,
    expand_image_tokens,
    build_multimodal_labels,
    inject_image_features,
    MultimodalSFTCollator,
    MultimodalSFTTrainer,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

IMAGE_TOKEN_ID = 1
N_IMAGE_TOKENS = 4
D_MODEL = 64


@pytest.fixture
def sft_config():
    return MultimodalSFTConfig(
        image_token_id=IMAGE_TOKEN_ID,
        n_image_tokens=N_IMAGE_TOKENS,
        max_seq_len=512,
    )


@pytest.fixture
def small_model_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=D_MODEL,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def small_model(small_model_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_model_cfg)


def _make_image_features(n_images: int, n_image_tokens: int = N_IMAGE_TOKENS, d_model: int = D_MODEL):
    return [torch.randn(n_image_tokens, d_model) for _ in range(n_images)]


def _make_example(n_images: int = 1) -> tuple[list[int], list[int]]:
    """Return (text_ids, labels) with n_images image placeholders."""
    text_ids = [2, IMAGE_TOKEN_ID, 3, 4]  # 1 image placeholder by default
    labels   = [-100, -100,          3, 4]
    if n_images == 0:
        text_ids = [2, 3, 4]
        labels = [-100, 3, 4]
    return text_ids, labels


# ---------------------------------------------------------------------------
# 1. test_multimodal_sft_config_defaults
# ---------------------------------------------------------------------------

def test_multimodal_sft_config_defaults():
    cfg = MultimodalSFTConfig()
    assert cfg.image_token_id == 1
    assert cfg.n_image_tokens == 16
    assert cfg.max_seq_len == 512
    assert cfg.loss_on_image_tokens is False
    assert cfg.learning_rate == 2e-5
    assert cfg.image_loss_weight == 0.0


# ---------------------------------------------------------------------------
# 2. test_multimodal_example_fields
# ---------------------------------------------------------------------------

def test_multimodal_example_fields():
    text_ids = [2, IMAGE_TOKEN_ID, 3]
    labels = [-100, -100, 3]
    feats = _make_image_features(1)
    ex = MultimodalExample(
        text_ids=text_ids,
        image_features=feats,
        labels=labels,
        n_images=1,
    )
    assert ex.text_ids == text_ids
    assert ex.labels == labels
    assert ex.n_images == 1
    assert ex.image_features is feats


# ---------------------------------------------------------------------------
# 3. test_expand_image_tokens_positions
# ---------------------------------------------------------------------------

def test_expand_image_tokens_positions():
    text_ids = [2, IMAGE_TOKEN_ID, 3, IMAGE_TOKEN_ID, 5]
    feats = _make_image_features(2)
    _, image_positions = expand_image_tokens(
        text_ids, feats, IMAGE_TOKEN_ID, N_IMAGE_TOKENS
    )
    assert len(image_positions) == 2 * N_IMAGE_TOKENS


# ---------------------------------------------------------------------------
# 4. test_expand_image_tokens_length
# ---------------------------------------------------------------------------

def test_expand_image_tokens_length():
    n_images = 2
    text_ids = [2, IMAGE_TOKEN_ID, 3, IMAGE_TOKEN_ID, 5]  # 5 tokens, 2 images
    feats = _make_image_features(n_images)
    expanded_ids, _ = expand_image_tokens(
        text_ids, feats, IMAGE_TOKEN_ID, N_IMAGE_TOKENS
    )
    expected_len = len(text_ids) + (N_IMAGE_TOKENS - 1) * n_images
    assert expanded_ids.shape[0] == expected_len


# ---------------------------------------------------------------------------
# 5. test_build_multimodal_labels_masked
# ---------------------------------------------------------------------------

def test_build_multimodal_labels_masked():
    text_ids = [2, IMAGE_TOKEN_ID, 3]
    labels = [-100, 5, 3]
    expanded_labels = build_multimodal_labels(
        text_ids, labels, IMAGE_TOKEN_ID, N_IMAGE_TOKENS, loss_on_image=False
    )
    # Positions 1..N_IMAGE_TOKENS (expanded image positions) should all be -100
    img_slice = expanded_labels[1 : 1 + N_IMAGE_TOKENS]
    assert (img_slice == -100).all()


# ---------------------------------------------------------------------------
# 6. test_build_multimodal_labels_with_loss
# ---------------------------------------------------------------------------

def test_build_multimodal_labels_with_loss():
    text_ids = [2, IMAGE_TOKEN_ID, 3]
    labels = [-100, 5, 3]  # image placeholder has real label 5
    expanded_labels = build_multimodal_labels(
        text_ids, labels, IMAGE_TOKEN_ID, N_IMAGE_TOKENS, loss_on_image=True
    )
    img_slice = expanded_labels[1 : 1 + N_IMAGE_TOKENS]
    # With loss_on_image=True and label != -100, should NOT be all -100
    assert (img_slice != -100).any()


# ---------------------------------------------------------------------------
# 7. test_inject_image_features_shape
# ---------------------------------------------------------------------------

def test_inject_image_features_shape():
    B, T, D = 2, 10, D_MODEL
    embeddings = torch.zeros(B, T, D)
    feats = _make_image_features(1, n_image_tokens=N_IMAGE_TOKENS, d_model=D)
    positions = list(range(N_IMAGE_TOKENS))  # first N positions

    result = inject_image_features(embeddings, feats, positions)
    assert result.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 8. test_inject_image_features_no_inplace
# ---------------------------------------------------------------------------

def test_inject_image_features_no_inplace():
    B, T, D = 1, 8, D_MODEL
    embeddings = torch.zeros(B, T, D)
    original = embeddings.clone()
    feats = _make_image_features(1, n_image_tokens=N_IMAGE_TOKENS, d_model=D)
    positions = list(range(N_IMAGE_TOKENS))

    inject_image_features(embeddings, feats, positions)
    # Original should be unchanged
    assert torch.equal(embeddings, original)


# ---------------------------------------------------------------------------
# 9. test_collator_batch_shapes
# ---------------------------------------------------------------------------

def test_collator_batch_shapes(sft_config):
    collator = MultimodalSFTCollator(sft_config, pad_id=0)
    examples = [
        MultimodalExample(
            text_ids=[2, IMAGE_TOKEN_ID, 3, 4],
            image_features=_make_image_features(1),
            labels=[-100, -100, 3, 4],
            n_images=1,
        ),
        MultimodalExample(
            text_ids=[5, 6, IMAGE_TOKEN_ID, 7],
            image_features=_make_image_features(1),
            labels=[-100, 6, -100, 7],
            n_images=1,
        ),
    ]
    batch = collator(examples)
    assert batch["input_ids"].shape == (2, sft_config.max_seq_len)
    assert batch["labels"].shape == (2, sft_config.max_seq_len)


# ---------------------------------------------------------------------------
# 10. test_collator_padding
# ---------------------------------------------------------------------------

def test_collator_padding(sft_config):
    collator = MultimodalSFTCollator(sft_config, pad_id=0)
    short_text = [2, 3]  # No images, 2 tokens
    ex = MultimodalExample(
        text_ids=short_text,
        image_features=None,
        labels=[-100, 3],
        n_images=0,
    )
    batch = collator([ex])
    input_ids = batch["input_ids"][0]
    # Should be padded to max_seq_len
    assert input_ids.shape[0] == sft_config.max_seq_len
    # The padding positions should be 0
    assert (input_ids[2:] == 0).all()


# ---------------------------------------------------------------------------
# 11. test_trainer_train_step_keys
# ---------------------------------------------------------------------------

def test_trainer_train_step_keys(small_model, sft_config):
    optimizer = torch.optim.SGD(small_model.parameters(), lr=1e-3)
    trainer = MultimodalSFTTrainer(
        model=small_model,
        embed_layer=small_model.embed,
        config=sft_config,
        optimizer=optimizer,
    )
    # Create a small batch
    B, T = 2, 16
    input_ids = torch.randint(2, 256, (B, T))
    labels = torch.randint(2, 256, (B, T))
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "image_positions": [[], []],
    }
    result = trainer.train_step(batch)
    assert "loss" in result
    assert "n_tokens" in result
    assert isinstance(result["loss"], float)
    assert isinstance(result["n_tokens"], int)


# ---------------------------------------------------------------------------
# 12. test_trainer_evaluate_step_no_grad
# ---------------------------------------------------------------------------

def test_trainer_evaluate_step_no_grad(small_model, sft_config):
    optimizer = torch.optim.SGD(small_model.parameters(), lr=1e-3)
    trainer = MultimodalSFTTrainer(
        model=small_model,
        embed_layer=small_model.embed,
        config=sft_config,
        optimizer=optimizer,
    )
    B, T = 2, 16
    input_ids = torch.randint(2, 256, (B, T))
    labels = torch.randint(2, 256, (B, T))
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "image_positions": [[], []],
    }
    result = trainer.evaluate_step(batch)
    assert isinstance(result, dict)
    assert "loss" in result
    assert "n_tokens" in result
