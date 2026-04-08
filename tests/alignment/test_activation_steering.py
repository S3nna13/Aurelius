"""Tests for activation steering / representation engineering."""
from __future__ import annotations

import torch
import pytest

from src.alignment.activation_steering import (
    SteeringVector,
    SteeringVectorExtractor,
    ActivationSteerer,
    ContrastiveActivationDataset,
    compute_steering_effect,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


def _make_tokenizer_encode(vocab_size: int = 256):
    """Minimal char-level tokenizer encode."""
    def encode(text: str) -> list:
        return [ord(c) % vocab_size for c in text]
    return encode


def _make_tokenizer_decode():
    """Minimal tokenizer decode."""
    def decode(ids: list) -> str:
        return "".join(chr(i % 128) for i in ids)
    return decode


POSITIVE_TEXTS = ["I'd be happy to help with that!", "Sure, here's a clear explanation."]
NEGATIVE_TEXTS = ["I cannot and will not help.", "I refuse to assist."]


# ---------------------------------------------------------------------------
# Test 1: mean_diff direction has unit norm
# ---------------------------------------------------------------------------

def test_steering_vector_direction_unit_norm(small_model, small_cfg):
    extractor = SteeringVectorExtractor(
        model=small_model,
        tokenizer_encode=_make_tokenizer_encode(small_cfg.vocab_size),
        layer_idx=0,
        max_seq_len=small_cfg.max_seq_len,
    )
    sv = extractor.extract_mean_diff(POSITIVE_TEXTS, NEGATIVE_TEXTS)
    norm = sv.direction.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


# ---------------------------------------------------------------------------
# Test 2: direction shape matches d_model
# ---------------------------------------------------------------------------

def test_steering_vector_shape(small_model, small_cfg):
    extractor = SteeringVectorExtractor(
        model=small_model,
        tokenizer_encode=_make_tokenizer_encode(small_cfg.vocab_size),
        layer_idx=0,
        max_seq_len=small_cfg.max_seq_len,
    )
    sv = extractor.extract_mean_diff(POSITIVE_TEXTS, NEGATIVE_TEXTS)
    assert sv.direction.shape == (small_cfg.d_model,), (
        f"Expected ({small_cfg.d_model},), got {sv.direction.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: ActivationSteerer works as context manager
# ---------------------------------------------------------------------------

def test_activation_steerer_context_manager(small_model, small_cfg):
    extractor = SteeringVectorExtractor(
        model=small_model,
        tokenizer_encode=_make_tokenizer_encode(small_cfg.vocab_size),
        layer_idx=0,
        max_seq_len=small_cfg.max_seq_len,
    )
    sv = extractor.extract_mean_diff(POSITIVE_TEXTS, NEGATIVE_TEXTS)
    # Should not raise
    with ActivationSteerer(small_model, [sv], alpha=10.0) as steerer:
        input_ids = torch.randint(0, small_cfg.vocab_size, (1, 4))
        with torch.no_grad():
            _, logits, _ = small_model(input_ids)
        assert logits.shape == (1, 4, small_cfg.vocab_size)


# ---------------------------------------------------------------------------
# Test 4: alpha=10 changes model output vs no steering
# ---------------------------------------------------------------------------

def test_activation_steerer_changes_output(small_model, small_cfg):
    torch.manual_seed(7)
    extractor = SteeringVectorExtractor(
        model=small_model,
        tokenizer_encode=_make_tokenizer_encode(small_cfg.vocab_size),
        layer_idx=0,
        max_seq_len=small_cfg.max_seq_len,
    )
    sv = extractor.extract_mean_diff(POSITIVE_TEXTS, NEGATIVE_TEXTS)

    input_ids = torch.randint(0, small_cfg.vocab_size, (1, 4))

    # Baseline output
    small_model.eval()
    with torch.no_grad():
        _, baseline_logits, _ = small_model(input_ids)

    # Steered output
    with ActivationSteerer(small_model, [sv], alpha=10.0):
        with torch.no_grad():
            _, steered_logits, _ = small_model(input_ids)

    assert not torch.allclose(baseline_logits, steered_logits), (
        "Steered output should differ from baseline"
    )


# ---------------------------------------------------------------------------
# Test 5: hooks are removed after context exit
# ---------------------------------------------------------------------------

def test_activation_steerer_hooks_removed_after_exit(small_model, small_cfg):
    extractor = SteeringVectorExtractor(
        model=small_model,
        tokenizer_encode=_make_tokenizer_encode(small_cfg.vocab_size),
        layer_idx=0,
        max_seq_len=small_cfg.max_seq_len,
    )
    sv = extractor.extract_mean_diff(POSITIVE_TEXTS, NEGATIVE_TEXTS)

    steerer = ActivationSteerer(small_model, [sv], alpha=10.0)
    with steerer:
        pass  # enter and immediately exit

    # After exit, no forward hooks should remain on any layer
    for layer in small_model.layers:
        assert len(layer._forward_hooks) == 0, (
            f"Expected no hooks after context exit, found {len(layer._forward_hooks)}"
        )


# ---------------------------------------------------------------------------
# Test 6: extract_multiple_layers returns correct count
# ---------------------------------------------------------------------------

def test_extract_multiple_layers_count(small_model, small_cfg):
    extractor = SteeringVectorExtractor(
        model=small_model,
        tokenizer_encode=_make_tokenizer_encode(small_cfg.vocab_size),
        layer_idx=0,
        max_seq_len=small_cfg.max_seq_len,
    )
    layer_indices = [0, 1]
    vectors = extractor.extract_multiple_layers(
        POSITIVE_TEXTS, NEGATIVE_TEXTS, layer_indices
    )
    assert len(vectors) == len(layer_indices), (
        f"Expected {len(layer_indices)} vectors, got {len(vectors)}"
    )
    for i, sv in enumerate(vectors):
        assert sv.layer_idx == layer_indices[i], (
            f"Expected layer_idx={layer_indices[i]}, got {sv.layer_idx}"
        )


# ---------------------------------------------------------------------------
# Test 7: PCA extraction produces unit norm direction
# ---------------------------------------------------------------------------

def test_pca_extraction_unit_norm(small_model, small_cfg):
    extractor = SteeringVectorExtractor(
        model=small_model,
        tokenizer_encode=_make_tokenizer_encode(small_cfg.vocab_size),
        layer_idx=0,
        max_seq_len=small_cfg.max_seq_len,
    )
    sv = extractor.extract_pca(POSITIVE_TEXTS, NEGATIVE_TEXTS)
    norm = sv.direction.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"PCA direction should have unit norm, got {norm}"


# ---------------------------------------------------------------------------
# Test 8: ContrastiveActivationDataset __len__
# ---------------------------------------------------------------------------

def test_contrastive_dataset_len():
    positive = ["I am helpful.", "This is great."]
    negative = ["I refuse.", "This is bad."]
    dataset = ContrastiveActivationDataset(positive, negative)
    assert len(dataset) == 2


# ---------------------------------------------------------------------------
# Test 9: from_preset returns ContrastiveActivationDataset
# ---------------------------------------------------------------------------

def test_contrastive_dataset_from_preset():
    dataset = ContrastiveActivationDataset.from_preset("helpful")
    assert isinstance(dataset, ContrastiveActivationDataset)
    assert len(dataset) > 0


# ---------------------------------------------------------------------------
# Test 10: alpha=0 gives same output as baseline
# ---------------------------------------------------------------------------

def test_compute_steering_effect_alpha_0_is_baseline(small_model, small_cfg):
    torch.manual_seed(0)
    encode = _make_tokenizer_encode(small_cfg.vocab_size)
    decode = _make_tokenizer_decode()

    extractor = SteeringVectorExtractor(
        model=small_model,
        tokenizer_encode=encode,
        layer_idx=0,
        max_seq_len=small_cfg.max_seq_len,
    )
    sv = extractor.extract_mean_diff(POSITIVE_TEXTS, NEGATIVE_TEXTS)

    prompt = "Hello"
    token_ids = encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    # Generate baseline with fixed seed
    torch.manual_seed(99)
    small_model.eval()
    with torch.no_grad():
        out_baseline = small_model.generate(input_ids, max_new_tokens=5)
    baseline_text = decode(out_baseline[0, len(token_ids):].tolist())

    # compute_steering_effect with same seed for alpha=0
    torch.manual_seed(99)
    results = compute_steering_effect(
        model=small_model,
        tokenizer_encode=encode,
        tokenizer_decode=decode,
        prompt=prompt,
        steering_vector=sv,
        alphas=[0.0],
        max_new_tokens=5,
    )

    assert 0.0 in results, "alpha=0.0 must be in results"
    assert results[0.0] == baseline_text, (
        f"alpha=0 result '{results[0.0]}' != baseline '{baseline_text}'"
    )
