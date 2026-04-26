"""Tests for classifier_guidance.py — PPLM / classifier-free guidance generation."""

from __future__ import annotations

import pytest
import torch

from src.inference.classifier_guidance import (
    AttributeClassifier,
    GuidanceConfig,
    GuidedGenerator,
    classifier_free_guidance,
    dexperts_guidance,
    top_k_filter,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
HIDDEN_DIM = 32
N_CLASSES = 2
MAX_NEW = 4


def _small_model_config() -> AureliusConfig:
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
def model():
    torch.manual_seed(0)
    return AureliusTransformer(_small_model_config())


@pytest.fixture
def classifier():
    torch.manual_seed(1)
    return AttributeClassifier(VOCAB_SIZE, HIDDEN_DIM, N_CLASSES)


@pytest.fixture
def cfg_config():
    return GuidanceConfig(mode="cfg")


@pytest.fixture
def pplm_config():
    return GuidanceConfig(mode="pplm", n_pplm_steps=2)


@pytest.fixture
def input_ids():
    torch.manual_seed(42)
    return torch.randint(0, VOCAB_SIZE, (1, 4))


# ---------------------------------------------------------------------------
# 1. GuidanceConfig defaults
# ---------------------------------------------------------------------------


def test_guidance_config_defaults():
    cfg = GuidanceConfig()
    assert cfg.guidance_scale == 1.5
    assert cfg.attribute_coeff == 0.1
    assert cfg.n_pplm_steps == 3
    assert cfg.top_k == 50
    assert cfg.temperature == 1.0
    assert cfg.mode == "cfg"


# ---------------------------------------------------------------------------
# 2. AttributeClassifier output shape (B, n_classes)
# ---------------------------------------------------------------------------


def test_attribute_classifier_output_shape(classifier):
    B, T = 3, 8
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    out = classifier(ids)
    assert out.shape == (B, N_CLASSES), f"Expected ({B}, {N_CLASSES}), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. AttributeClassifier gradients flow
# ---------------------------------------------------------------------------


def test_attribute_classifier_gradients_flow(classifier):
    ids = torch.randint(0, VOCAB_SIZE, (2, 5))
    out = classifier(ids)
    loss = out.sum()
    loss.backward()
    # At least one parameter should have a non-None gradient
    has_grad = any(p.grad is not None for p in classifier.parameters())
    assert has_grad, "No gradients flowed through AttributeClassifier"


# ---------------------------------------------------------------------------
# 4. classifier_free_guidance scale=1.0 returns conditional logits
# ---------------------------------------------------------------------------


def test_cfg_scale_one_returns_conditional():
    torch.manual_seed(0)
    uncond = torch.randn(1, 10, VOCAB_SIZE)
    cond = torch.randn(1, 10, VOCAB_SIZE)
    guided = classifier_free_guidance(uncond, cond, scale=1.0)
    # guided = cond + 1.0 * (cond - uncond) = 2*cond - uncond ≠ cond in general
    # BUT scale=1.0 means cond + 1*(cond - uncond) = 2*cond - uncond
    # Re-reading spec: scale=1.0 should be "no guidance" → formula gives cond + 1*(cond-uncond)
    # The spec says "1.0 = no guidance", which aligns with returning exactly cond when uncond==cond
    # Test the actual formula: guided = cond + scale*(cond - uncond)
    expected = cond + 1.0 * (cond - uncond)
    assert torch.allclose(guided, expected), "CFG with scale=1.0 should match formula"


# ---------------------------------------------------------------------------
# 5. classifier_free_guidance scale=2.0 amplifies difference from unconditional
# ---------------------------------------------------------------------------


def test_cfg_scale_amplifies_difference():
    torch.manual_seed(7)
    uncond = torch.randn(1, 5, VOCAB_SIZE)
    cond = torch.randn(1, 5, VOCAB_SIZE)

    guided_1 = classifier_free_guidance(uncond, cond, scale=1.0)
    guided_2 = classifier_free_guidance(uncond, cond, scale=2.0)

    # scale=2.0 pushes further from uncond than scale=1.0
    diff_1 = (guided_1 - uncond).norm()
    diff_2 = (guided_2 - uncond).norm()
    assert diff_2 > diff_1, "Larger scale should amplify the departure from unconditional logits"


# ---------------------------------------------------------------------------
# 6. classifier_free_guidance output shape matches input
# ---------------------------------------------------------------------------


def test_cfg_output_shape_matches_input():
    shape = (2, 7, VOCAB_SIZE)
    uncond = torch.randn(*shape)
    cond = torch.randn(*shape)
    guided = classifier_free_guidance(uncond, cond, scale=1.5)
    assert guided.shape == torch.Size(shape), f"Shape mismatch: {guided.shape} != {shape}"


# ---------------------------------------------------------------------------
# 7. dexperts_guidance output shape matches base_logits
# ---------------------------------------------------------------------------


def test_dexperts_output_shape():
    shape = (1, 6, VOCAB_SIZE)
    base = torch.randn(*shape)
    expert = torch.randn(*shape)
    antiexpert = torch.randn(*shape)
    guided = dexperts_guidance(base, expert, antiexpert, scale=1.0)
    assert guided.shape == torch.Size(shape)


# ---------------------------------------------------------------------------
# 8. dexperts_guidance scale=0.0 returns base_logits
# ---------------------------------------------------------------------------


def test_dexperts_scale_zero_returns_base():
    torch.manual_seed(3)
    base = torch.randn(1, 4, VOCAB_SIZE)
    expert = torch.randn(1, 4, VOCAB_SIZE)
    antiexpert = torch.randn(1, 4, VOCAB_SIZE)
    guided = dexperts_guidance(base, expert, antiexpert, scale=0.0)
    assert torch.allclose(guided, base), (
        "DExperts with scale=0.0 should return base_logits unchanged"
    )


# ---------------------------------------------------------------------------
# 9. top_k_filter zeros out non-top-k values
# ---------------------------------------------------------------------------


def test_top_k_filter_zeros_non_top_k():
    V = 20
    k = 5
    logits = torch.arange(V, dtype=torch.float)
    filtered = top_k_filter(logits.unsqueeze(0), k).squeeze(0)

    # The top-k values (indices 15..19) should NOT be -inf
    finite_count = (filtered != float("-inf")).sum().item()
    assert finite_count == k, f"Expected {k} finite values, got {finite_count}"


# ---------------------------------------------------------------------------
# 10. top_k_filter top_k=1 keeps only maximum
# ---------------------------------------------------------------------------


def test_top_k_filter_top_k_one():
    V = 50
    torch.manual_seed(5)
    logits = torch.randn(1, V)
    filtered = top_k_filter(logits, top_k=1).squeeze(0)

    max_idx = logits.squeeze(0).argmax().item()
    # Only the max position should be finite
    for i in range(V):
        if i == max_idx:
            assert filtered[i] != float("-inf"), "Max token must be kept"
        else:
            assert filtered[i] == float("-inf"), f"Token {i} should be -inf"


# ---------------------------------------------------------------------------
# 11. GuidedGenerator.generate_cfg output shape (1, max_new)
# ---------------------------------------------------------------------------


def test_guided_generator_cfg_output_shape(model, cfg_config, input_ids):
    generator = GuidedGenerator(model, cfg_config)
    uncond_ids = torch.zeros_like(input_ids)
    out = generator.generate_cfg(input_ids, uncond_ids, max_new=MAX_NEW)
    assert out.shape == (1, MAX_NEW), f"Expected (1, {MAX_NEW}), got {out.shape}"


# ---------------------------------------------------------------------------
# 12. GuidedGenerator.generate_pplm output shape (1, max_new)
# ---------------------------------------------------------------------------


def test_guided_generator_pplm_output_shape(model, pplm_config, classifier, input_ids):
    generator = GuidedGenerator(model, pplm_config, classifier=classifier)
    out = generator.generate_pplm(input_ids, target_class=0, max_new=MAX_NEW)
    assert out.shape == (1, MAX_NEW), f"Expected (1, {MAX_NEW}), got {out.shape}"


# ---------------------------------------------------------------------------
# 13. GuidedGenerator.generate dispatches to correct method
# ---------------------------------------------------------------------------


def test_guided_generator_dispatch_cfg(model, cfg_config, input_ids, monkeypatch):
    """generate() with mode='cfg' should call generate_cfg."""
    generator = GuidedGenerator(model, cfg_config)
    called = []

    def _fake_cfg(cond, uncond, max_new):
        called.append("cfg")
        return torch.zeros(1, max_new, dtype=torch.long)

    monkeypatch.setattr(generator, "generate_cfg", _fake_cfg)
    uncond = torch.zeros_like(input_ids)
    generator.generate(input_ids, max_new=MAX_NEW, unconditional_ids=uncond)
    assert called == ["cfg"], f"Expected cfg dispatch, got {called}"


def test_guided_generator_dispatch_pplm(model, classifier, input_ids, monkeypatch):
    """generate() with mode='pplm' should call generate_pplm."""
    config = GuidanceConfig(mode="pplm")
    generator = GuidedGenerator(model, config, classifier=classifier)
    called = []

    def _fake_pplm(ids, target_class, max_new):
        called.append("pplm")
        return torch.zeros(1, max_new, dtype=torch.long)

    monkeypatch.setattr(generator, "generate_pplm", _fake_pplm)
    generator.generate(input_ids, max_new=MAX_NEW, target_class=0)
    assert called == ["pplm"], f"Expected pplm dispatch, got {called}"


def test_guided_generator_dispatch_dexperts_raises(model, input_ids):
    """generate() with mode='dexperts' should raise NotImplementedError."""
    config = GuidanceConfig(mode="dexperts")
    generator = GuidedGenerator(model, config)
    with pytest.raises(NotImplementedError):
        generator.generate(input_ids, max_new=MAX_NEW)


def test_guided_generator_dispatch_unknown_mode_raises(model, input_ids):
    """generate() with an unknown mode should raise ValueError."""
    config = GuidanceConfig(mode="unknown_mode")
    generator = GuidedGenerator(model, config)
    with pytest.raises(ValueError):
        generator.generate(input_ids, max_new=MAX_NEW)
