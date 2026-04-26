"""Tests for Contrastive Decoding (Li et al., 2022)."""

from __future__ import annotations

import pytest
import torch

from src.inference.contrastive_decoding import (
    ContrastiveDecodeConfig,
    ContrastiveDecoder,
    adaptive_alpha,
    compute_plausibility_mask,
    contrastive_score,
    contrastive_step,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _small_config() -> AureliusConfig:
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
def expert_model():
    torch.manual_seed(0)
    m = AureliusTransformer(_small_config())
    m.eval()
    return m


@pytest.fixture
def amateur_model():
    torch.manual_seed(1)
    m = AureliusTransformer(_small_config())
    m.eval()
    return m


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (1, 4))


# ---------------------------------------------------------------------------
# 1. ContrastiveDecodeConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = ContrastiveDecodeConfig()
    assert cfg.alpha == 0.1
    assert cfg.temperature == 1.0
    assert cfg.max_new_tokens == 64


# ---------------------------------------------------------------------------
# 2. ContrastiveDecodeConfig custom values
# ---------------------------------------------------------------------------


def test_config_custom():
    cfg = ContrastiveDecodeConfig(alpha=0.3, temperature=0.7, max_new_tokens=128)
    assert cfg.alpha == 0.3
    assert cfg.temperature == 0.7
    assert cfg.max_new_tokens == 128


# ---------------------------------------------------------------------------
# 3. compute_plausibility_mask shape
# ---------------------------------------------------------------------------


def test_plausibility_mask_shape():
    logits = torch.randn(2, 256)
    mask = compute_plausibility_mask(logits, alpha=0.1)
    assert mask.shape == (2, 256)
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 4. compute_plausibility_mask keeps top token
# ---------------------------------------------------------------------------


def test_plausibility_mask_keeps_top():
    logits = torch.full((1, 16), -10.0)
    logits[0, 5] = 10.0  # dominant token
    mask = compute_plausibility_mask(logits, alpha=0.1)
    assert mask[0, 5].item() is True


# ---------------------------------------------------------------------------
# 5. compute_plausibility_mask filters low-prob tokens
# ---------------------------------------------------------------------------


def test_plausibility_mask_filters_low():
    logits = torch.full((1, 16), -100.0)
    logits[0, 0] = 10.0
    mask = compute_plausibility_mask(logits, alpha=0.1)
    # Only token 0 should pass; others have negligible probability
    assert mask[0, 0].item() is True
    assert mask[0, 1:].sum().item() == 0


# ---------------------------------------------------------------------------
# 6. compute_plausibility_mask with alpha=0 keeps all
# ---------------------------------------------------------------------------


def test_plausibility_mask_alpha_zero():
    logits = torch.randn(1, 32)
    mask = compute_plausibility_mask(logits, alpha=0.0)
    assert mask.all()


# ---------------------------------------------------------------------------
# 7. contrastive_score shape
# ---------------------------------------------------------------------------


def test_contrastive_score_shape():
    B, V = 2, 256
    e = torch.randn(B, V)
    a = torch.randn(B, V)
    mask = torch.ones(B, V, dtype=torch.bool)
    s = contrastive_score(e, a, mask)
    assert s.shape == (B, V)


# ---------------------------------------------------------------------------
# 8. contrastive_score masked positions are -inf
# ---------------------------------------------------------------------------


def test_contrastive_score_masked_inf():
    B, V = 1, 8
    e = torch.randn(B, V)
    a = torch.randn(B, V)
    mask = torch.zeros(B, V, dtype=torch.bool)
    mask[0, 3] = True
    s = contrastive_score(e, a, mask)
    # Only position 3 should be finite
    assert torch.isfinite(s[0, 3])
    for i in range(V):
        if i != 3:
            assert s[0, i] == float("-inf")


# ---------------------------------------------------------------------------
# 9. contrastive_score favors expert-preferred tokens
# ---------------------------------------------------------------------------


def test_contrastive_score_expert_preference():
    B, V = 1, 4
    # Expert strongly prefers token 0, amateur prefers token 1
    e = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
    a = torch.tensor([[-10.0, 10.0, -10.0, -10.0]])
    mask = torch.ones(B, V, dtype=torch.bool)
    s = contrastive_score(e, a, mask)
    assert s.argmax(dim=-1).item() == 0


# ---------------------------------------------------------------------------
# 10. adaptive_alpha at step 0
# ---------------------------------------------------------------------------


def test_adaptive_alpha_start():
    assert adaptive_alpha(0, 10) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 11. adaptive_alpha at last step
# ---------------------------------------------------------------------------


def test_adaptive_alpha_end():
    assert adaptive_alpha(9, 10) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 12. adaptive_alpha mid-point
# ---------------------------------------------------------------------------


def test_adaptive_alpha_mid():
    mid = adaptive_alpha(5, 11)  # t = 5/10 = 0.5
    expected = 0.05 + 0.5 * (0.5 - 0.05)
    assert mid == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 13. adaptive_alpha with total_steps=1
# ---------------------------------------------------------------------------


def test_adaptive_alpha_single_step():
    assert adaptive_alpha(0, 1) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 14. contrastive_step returns correct shapes
# ---------------------------------------------------------------------------


def test_contrastive_step_shapes(expert_model, amateur_model, input_ids):
    cfg = ContrastiveDecodeConfig()
    next_token, scores = contrastive_step(expert_model, amateur_model, input_ids, cfg)
    assert next_token.shape == (1, 1)
    assert scores.shape == (1, 256)


# ---------------------------------------------------------------------------
# 15. contrastive_step token is valid
# ---------------------------------------------------------------------------


def test_contrastive_step_valid_token(expert_model, amateur_model, input_ids):
    cfg = ContrastiveDecodeConfig()
    next_token, _ = contrastive_step(expert_model, amateur_model, input_ids, cfg)
    assert 0 <= next_token.item() < 256


# ---------------------------------------------------------------------------
# 16. ContrastiveDecoder generate output shape
# ---------------------------------------------------------------------------


def test_decoder_generate_shape(expert_model, amateur_model, input_ids):
    cfg = ContrastiveDecodeConfig(max_new_tokens=3)
    decoder = ContrastiveDecoder(expert_model, amateur_model, cfg)
    output = decoder.generate(input_ids)
    assert output.shape == (1, input_ids.shape[1] + 3)


# ---------------------------------------------------------------------------
# 17. ContrastiveDecoder preserves prompt prefix
# ---------------------------------------------------------------------------


def test_decoder_preserves_prompt(expert_model, amateur_model, input_ids):
    cfg = ContrastiveDecodeConfig(max_new_tokens=2)
    decoder = ContrastiveDecoder(expert_model, amateur_model, cfg)
    output = decoder.generate(input_ids)
    assert torch.equal(output[:, : input_ids.shape[1]], input_ids)


# ---------------------------------------------------------------------------
# 18. ContrastiveDecoder generate tokens are valid vocab indices
# ---------------------------------------------------------------------------


def test_decoder_generate_valid_tokens(expert_model, amateur_model, input_ids):
    cfg = ContrastiveDecodeConfig(max_new_tokens=4)
    decoder = ContrastiveDecoder(expert_model, amateur_model, cfg)
    output = decoder.generate(input_ids)
    generated = output[:, input_ids.shape[1] :]
    assert (generated >= 0).all()
    assert (generated < 256).all()


# ---------------------------------------------------------------------------
# 19. adaptive_alpha is monotonically increasing
# ---------------------------------------------------------------------------


def test_adaptive_alpha_monotonic():
    total = 20
    values = [adaptive_alpha(s, total) for s in range(total)]
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1]


# ---------------------------------------------------------------------------
# 20. contrastive_score all masked yields all -inf
# ---------------------------------------------------------------------------


def test_contrastive_score_all_masked():
    B, V = 1, 8
    e = torch.randn(B, V)
    a = torch.randn(B, V)
    mask = torch.zeros(B, V, dtype=torch.bool)
    s = contrastive_score(e, a, mask)
    assert (s == float("-inf")).all()
