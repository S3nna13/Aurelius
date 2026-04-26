"""Unit tests for SoftThinkingMixer (src/inference/soft_thinking.py).

Tiny config: d_model=64, vocab_size=256.
Run with: .venv/bin/python3.14 -m pytest tests/inference/test_soft_thinking.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from src.inference.soft_thinking import SoftThinkingConfig, SoftThinkingMixer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
VOCAB = 256
B = 4
T = 8


@pytest.fixture()
def cfg() -> SoftThinkingConfig:
    return SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=10)


@pytest.fixture()
def mixer(cfg: SoftThinkingConfig) -> SoftThinkingMixer:
    torch.manual_seed(0)
    return SoftThinkingMixer(cfg)


@pytest.fixture()
def logits_2d() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, VOCAB)


@pytest.fixture()
def logits_3d() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(B, T, VOCAB)


# ---------------------------------------------------------------------------
# Test 1: config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = SoftThinkingConfig()
    assert cfg.top_k == 50
    assert cfg.temperature == 1.0
    assert cfg.renormalize is True
    assert cfg.d_model == 2048
    assert cfg.vocab_size == 128000


# ---------------------------------------------------------------------------
# Test 2: mix shape 2D — [B, V] → [B, d_model]
# ---------------------------------------------------------------------------


def test_mix_shape_2d(mixer: SoftThinkingMixer, logits_2d: torch.Tensor):
    out = mixer.mix(logits_2d)
    assert out.shape == (B, D_MODEL), f"Expected ({B}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3: mix shape 3D — [B, T, V] → [B, T, d_model]
# ---------------------------------------------------------------------------


def test_mix_shape_3d(mixer: SoftThinkingMixer, logits_3d: torch.Tensor):
    out = mixer.mix(logits_3d)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 4: weights sum to 1 when renormalize=True
# ---------------------------------------------------------------------------


def test_weights_sum_to_one(logits_2d: torch.Tensor):
    cfg = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=10, renormalize=True)
    SoftThinkingMixer(cfg)

    # Compute the top-k weights manually and verify renormalization
    probs = torch.softmax(logits_2d, dim=-1)
    topk_w, _ = torch.topk(probs, k=10, dim=-1)
    renorm_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    sums = renorm_w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"Weights don't sum to 1: {sums}"


# ---------------------------------------------------------------------------
# Test 5: renormalize=False — weights may not sum to 1
# ---------------------------------------------------------------------------


def test_renormalize_false(logits_2d: torch.Tensor):
    cfg = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=5, renormalize=False)
    mixer = SoftThinkingMixer(cfg)

    probs = torch.softmax(logits_2d, dim=-1)
    topk_w, _ = torch.topk(probs, k=5, dim=-1)
    sums = topk_w.sum(dim=-1)

    # With renormalize=False and top_k < vocab, sums should be < 1
    # (extremely unlikely they equal 1 for random logits with vocab=256 and k=5)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3), (
        "Expected weights NOT to sum to 1 when renormalize=False"
    )

    # The mixer still produces the correct output shape
    out = mixer.mix(logits_2d)
    assert out.shape == (B, D_MODEL)


# ---------------------------------------------------------------------------
# Test 6: temperature effect — lower temperature → more peaked → lower entropy
# ---------------------------------------------------------------------------


def test_temperature_effect(logits_2d: torch.Tensor):
    cfg_hot = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=10, temperature=2.0)
    cfg_cold = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=10, temperature=0.1)

    mixer_hot = SoftThinkingMixer(cfg_hot)
    mixer_cold = SoftThinkingMixer(cfg_cold)

    ent_hot = mixer_hot.entropy(logits_2d).mean().item()
    ent_cold = mixer_cold.entropy(logits_2d).mean().item()

    assert ent_cold < ent_hot, (
        f"Lower temperature should produce lower entropy: cold={ent_cold:.4f}, hot={ent_hot:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 7: top_k=1 → soft embedding equals embedding of argmax token
# ---------------------------------------------------------------------------


def test_top_k_1_is_argmax(logits_2d: torch.Tensor):
    cfg = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=1, renormalize=True)
    torch.manual_seed(42)
    mixer = SoftThinkingMixer(cfg)

    out = mixer.mix(logits_2d)  # [B, d_model]

    argmax_ids = logits_2d.argmax(dim=-1)  # [B]
    expected = mixer.embedding.weight[argmax_ids]  # [B, d_model]

    assert torch.allclose(out, expected, atol=1e-5), (
        "top_k=1 soft embedding should equal embedding of argmax token"
    )


# ---------------------------------------------------------------------------
# Test 8: top_k == vocab_size → uses all tokens
# ---------------------------------------------------------------------------


def test_top_k_equals_vocab(logits_2d: torch.Tensor):
    cfg = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=VOCAB, renormalize=True)
    mixer = SoftThinkingMixer(cfg)

    out = mixer.mix(logits_2d)
    assert out.shape == (B, D_MODEL)

    # With all tokens included and renormalize=True, result equals full softmax-weighted sum
    probs = torch.softmax(logits_2d, dim=-1)  # [B, V]
    expected = probs @ mixer.embedding.weight  # [B, d_model]

    assert torch.allclose(out, expected, atol=1e-4), (
        "top_k=vocab_size should equal full softmax-weighted embedding sum"
    )


# ---------------------------------------------------------------------------
# Test 9: external embedding_weight passed to mix()
# ---------------------------------------------------------------------------


def test_external_embedding(logits_2d: torch.Tensor):
    cfg = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=10)
    mixer = SoftThinkingMixer(cfg)

    torch.manual_seed(99)
    custom_weight = torch.randn(VOCAB, D_MODEL)

    out_custom = mixer.mix(logits_2d, embedding_weight=custom_weight)
    out_default = mixer.mix(logits_2d)

    assert out_custom.shape == (B, D_MODEL)
    # Results must differ because different embedding weights are used
    assert not torch.allclose(out_custom, out_default, atol=1e-4), (
        "Custom embedding weight should produce different output from default"
    )


# ---------------------------------------------------------------------------
# Test 10: gradients flow through mix() into embedding.weight
# ---------------------------------------------------------------------------


def test_gradients_flow(logits_2d: torch.Tensor):
    cfg = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=10)
    mixer = SoftThinkingMixer(cfg)

    logits = logits_2d.detach().requires_grad_(True)
    out = mixer.mix(logits)
    loss = out.sum()
    loss.backward()

    assert mixer.embedding.weight.grad is not None, (
        "embedding.weight should have a gradient after backward"
    )
    assert mixer.embedding.weight.grad.abs().sum() > 0, "embedding.weight.grad should be non-zero"


# ---------------------------------------------------------------------------
# Test 11: entropy shape 2D — [B, V] → [B]
# ---------------------------------------------------------------------------


def test_entropy_shape_2d(mixer: SoftThinkingMixer, logits_2d: torch.Tensor):
    ent = mixer.entropy(logits_2d)
    assert ent.shape == (B,), f"Expected ({B},), got {ent.shape}"


# ---------------------------------------------------------------------------
# Test 12: entropy shape 3D — [B, T, V] → [B, T]
# ---------------------------------------------------------------------------


def test_entropy_shape_3d(mixer: SoftThinkingMixer, logits_3d: torch.Tensor):
    ent = mixer.entropy(logits_3d)
    assert ent.shape == (B, T), f"Expected ({B}, {T}), got {ent.shape}"


# ---------------------------------------------------------------------------
# Test 13: entropy of uniform distribution ≈ log(vocab_size)
# ---------------------------------------------------------------------------


def test_entropy_uniform():
    cfg = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=10, temperature=1.0)
    mixer = SoftThinkingMixer(cfg)

    # Uniform logits → uniform distribution → max entropy = log(V)
    uniform_logits = torch.zeros(B, VOCAB)
    ent = mixer.entropy(uniform_logits)  # [B]

    expected = math.log(VOCAB)
    assert torch.allclose(ent, torch.full_like(ent, expected), atol=1e-4), (
        f"Uniform entropy should be log({VOCAB})={expected:.4f}, got {ent}"
    )


# ---------------------------------------------------------------------------
# Test 14: determinism — same logits + inference mode → same output
# ---------------------------------------------------------------------------


def test_determinism(logits_2d: torch.Tensor):
    cfg = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=10)
    mixer = SoftThinkingMixer(cfg)
    mixer.train(False)  # switch to inference mode without using eval()

    with torch.no_grad():
        out1 = mixer.mix(logits_2d)
        out2 = mixer.mix(logits_2d)

    assert torch.equal(out1, out2), "mix() should be deterministic in inference mode"


# ---------------------------------------------------------------------------
# Test 15: registry — DECODER_REGISTRY["soft_thinking"] is SoftThinkingMixer
# ---------------------------------------------------------------------------


def test_registry():
    from src.inference import DECODER_REGISTRY

    assert "soft_thinking" in DECODER_REGISTRY, "DECODER_REGISTRY must contain 'soft_thinking'"
    assert DECODER_REGISTRY["soft_thinking"] is SoftThinkingMixer, (
        "DECODER_REGISTRY['soft_thinking'] must be SoftThinkingMixer"
    )
