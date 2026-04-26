"""Tests for DoLa Decoding (Chuang et al., 2023).

Covers:
    Unit tests  (1-14): config defaults, shape, semantic properties, JSD,
                         sampling, decode_step keys/validity, alpha edge-case,
                         batch decode.
    Integration test (15): end-to-end decode_step with registry-retrieved class.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.inference import DECODER_REGISTRY
from src.inference.dola_decoding import (
    DoLaConfig,
    DoLaDecoder,
    DoLaLayerOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 128
BATCH = 4


def _make_decoder(**kwargs) -> DoLaDecoder:
    return DoLaDecoder(DoLaConfig(**kwargs))


def _random_logits(*shape) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = DoLaConfig()
    assert cfg.early_exit_layer == 12
    assert cfg.alpha == 0.5
    assert cfg.contrast_mode == "subtract"
    assert cfg.temperature == 1.0
    assert cfg.top_k == 0
    assert cfg.top_p == 1.0


# ---------------------------------------------------------------------------
# 2. test_contrast_subtract_shape
# ---------------------------------------------------------------------------


def test_contrast_subtract_shape():
    dec = _make_decoder(contrast_mode="subtract")
    late = _random_logits(VOCAB)
    early = _random_logits(VOCAB)
    out = dec.contrast_logits(late, early)
    assert out.shape == late.shape


def test_contrast_subtract_shape_batched():
    dec = _make_decoder(contrast_mode="subtract")
    late = _random_logits(BATCH, VOCAB)
    early = _random_logits(BATCH, VOCAB)
    out = dec.contrast_logits(late, early)
    assert out.shape == late.shape


# ---------------------------------------------------------------------------
# 3. test_contrast_subtract_late_preferred
# ---------------------------------------------------------------------------


def test_contrast_subtract_late_preferred():
    """When the late layer strongly prefers token 0, contrast should also prefer 0."""
    dec = _make_decoder(contrast_mode="subtract", alpha=0.5)
    late = torch.zeros(VOCAB)
    late[0] = 20.0  # strongly prefer token 0
    early = torch.zeros(VOCAB)  # uniform early layer
    out = dec.contrast_logits(late, early)
    assert out.argmax().item() == 0


# ---------------------------------------------------------------------------
# 4. test_contrast_jsd_shape
# ---------------------------------------------------------------------------


def test_contrast_jsd_shape():
    dec = _make_decoder(contrast_mode="jsd")
    late = _random_logits(VOCAB)
    early = _random_logits(VOCAB)
    out = dec.contrast_logits(late, early)
    assert out.shape == late.shape


def test_contrast_jsd_shape_batched():
    dec = _make_decoder(contrast_mode="jsd")
    late = _random_logits(BATCH, VOCAB)
    early = _random_logits(BATCH, VOCAB)
    out = dec.contrast_logits(late, early)
    assert out.shape == late.shape


# ---------------------------------------------------------------------------
# 5. test_contrast_jsd_amplifies_late
# ---------------------------------------------------------------------------


def test_contrast_jsd_amplifies_late():
    """JSD contrast should give a higher score to tokens favoured by the late layer."""
    dec = _make_decoder(contrast_mode="jsd", alpha=0.5)
    late = torch.zeros(VOCAB)
    late[7] = 15.0  # strongly prefer token 7
    early = torch.zeros(VOCAB)  # uniform early layer
    out = dec.contrast_logits(late, early)
    assert out.argmax().item() == 7


# ---------------------------------------------------------------------------
# 6. test_jsd_symmetric
# ---------------------------------------------------------------------------


def test_jsd_symmetric():
    dec = _make_decoder()
    torch.manual_seed(0)
    raw_p = torch.randn(VOCAB).softmax(dim=-1)
    raw_q = torch.randn(VOCAB).softmax(dim=-1)
    assert torch.allclose(dec.jsd(raw_p, raw_q), dec.jsd(raw_q, raw_p), atol=1e-5)


# ---------------------------------------------------------------------------
# 7. test_jsd_zero_identical
# ---------------------------------------------------------------------------


def test_jsd_zero_identical():
    dec = _make_decoder()
    p = torch.ones(VOCAB) / VOCAB
    assert torch.allclose(dec.jsd(p, p), torch.tensor(0.0), atol=1e-5)


# ---------------------------------------------------------------------------
# 8. test_jsd_positive
# ---------------------------------------------------------------------------


def test_jsd_positive():
    dec = _make_decoder()
    p = torch.zeros(VOCAB)
    p[0] = 1.0
    q = torch.zeros(VOCAB)
    q[-1] = 1.0
    val = dec.jsd(p, q)
    assert val.item() > 0.0


# ---------------------------------------------------------------------------
# 9. test_sample_temperature
# ---------------------------------------------------------------------------


def test_sample_temperature():
    """Higher temperature → higher entropy over many samples."""
    torch.manual_seed(123)
    logits = torch.randn(VOCAB)

    def _entropy(t: float, n: int = 2000) -> float:
        dec = DoLaDecoder(DoLaConfig(temperature=t))
        counts = torch.zeros(VOCAB)
        for _ in range(n):
            tok = dec.sample(logits.clone())
            counts[tok.item()] += 1
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -(probs * probs.log()).sum().item()

    h_low = _entropy(0.3)
    h_high = _entropy(2.0)
    assert h_high > h_low, f"Expected h_high({h_high:.3f}) > h_low({h_low:.3f})"


# ---------------------------------------------------------------------------
# 10. test_sample_top_k_equals_1
# ---------------------------------------------------------------------------


def test_sample_top_k_equals_1():
    """top_k=1 must always return the argmax token."""
    dec = _make_decoder(top_k=1)
    logits = _random_logits(VOCAB)
    expected = logits.argmax().item()
    for _ in range(20):
        tok = dec.sample(logits.clone())
        assert tok.item() == expected


# ---------------------------------------------------------------------------
# 11. test_decode_step_keys
# ---------------------------------------------------------------------------


def test_decode_step_keys():
    dec = _make_decoder()
    late_out = DoLaLayerOutput(logits=_random_logits(VOCAB), layer_idx=23)
    early_out = DoLaLayerOutput(logits=_random_logits(VOCAB), layer_idx=12)
    result = dec.decode_step(late_out, early_out)
    assert set(result.keys()) == {"token_ids", "contrast_logits", "late_logits", "early_logits"}


# ---------------------------------------------------------------------------
# 12. test_decode_step_token_valid
# ---------------------------------------------------------------------------


def test_decode_step_token_valid():
    dec = _make_decoder(vocab_size=VOCAB)
    late_out = DoLaLayerOutput(logits=_random_logits(VOCAB), layer_idx=23)
    early_out = DoLaLayerOutput(logits=_random_logits(VOCAB), layer_idx=12)
    result = dec.decode_step(late_out, early_out)
    tok = result["token_ids"].item()
    assert 0 <= tok < VOCAB


# ---------------------------------------------------------------------------
# 13. test_alpha_zero_equals_late
# ---------------------------------------------------------------------------


def test_alpha_zero_equals_late():
    """alpha=0 in subtract mode → contrast = log_softmax(late_logits / T)."""
    dec = _make_decoder(contrast_mode="subtract", alpha=0.0, temperature=1.0)
    late = _random_logits(VOCAB)
    early = _random_logits(VOCAB)
    out = dec.contrast_logits(late, early)
    expected = F.log_softmax(late, dim=-1)
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 14. test_batch_decode
# ---------------------------------------------------------------------------


def test_batch_decode():
    """decode_step must work correctly for B > 1."""
    dec = _make_decoder(vocab_size=VOCAB)
    late_out = DoLaLayerOutput(logits=_random_logits(BATCH, VOCAB), layer_idx=23)
    early_out = DoLaLayerOutput(logits=_random_logits(BATCH, VOCAB), layer_idx=12)
    result = dec.decode_step(late_out, early_out)

    # token_ids shape
    assert result["token_ids"].shape == (BATCH,)
    # all tokens in valid range
    assert (result["token_ids"] >= 0).all()
    assert (result["token_ids"] < VOCAB).all()
    # contrast_logits shape preserved
    assert result["contrast_logits"].shape == (BATCH, VOCAB)


# ---------------------------------------------------------------------------
# 15. Integration test
# ---------------------------------------------------------------------------


def test_integration_full_decode_step():
    """End-to-end integration: retrieve from DECODER_REGISTRY, run decode_step."""
    # Retrieve from registry
    assert "dola" in DECODER_REGISTRY, "DoLaDecoder must be registered under 'dola'"
    DecoderCls = DECODER_REGISTRY["dola"]

    # Build decoder with JSD contrast
    cfg = DoLaConfig(
        early_exit_layer=8,
        alpha=0.4,
        contrast_mode="jsd",
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        vocab_size=VOCAB,
    )
    dec = DecoderCls(cfg)

    torch.manual_seed(7)
    late_out = DoLaLayerOutput(logits=torch.randn(BATCH, VOCAB), layer_idx=23)
    early_out = DoLaLayerOutput(logits=torch.randn(BATCH, VOCAB), layer_idx=8)

    result = dec.decode_step(late_out, early_out)

    # Dict completeness
    assert set(result.keys()) == {"token_ids", "contrast_logits", "late_logits", "early_logits"}

    # Shape checks
    assert result["token_ids"].shape == (BATCH,)
    assert result["contrast_logits"].shape == (BATCH, VOCAB)
    assert result["late_logits"].shape == (BATCH, VOCAB)
    assert result["early_logits"].shape == (BATCH, VOCAB)

    # Validity
    assert (result["token_ids"] >= 0).all()
    assert (result["token_ids"] < VOCAB).all()

    # JSD is finite everywhere
    assert result["contrast_logits"].isfinite().all()
