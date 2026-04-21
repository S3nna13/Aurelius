"""Integration test for Eagle3 speculative decoding (Cycle 130-A).

Verifies end-to-end behaviour:
- Eagle3Decoder instantiation with a small config
- Mock target model returning random probability vectors
- decode_step output structure and numeric invariants
- Registry wiring (DECODER_REGISTRY["eagle3"] == Eagle3Decoder)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.inference.eagle3_decoding import (
    Eagle3Config,
    Eagle3Decoder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D_MODEL = 64
VOCAB_SIZE = 256


def make_config(**kwargs) -> Eagle3Config:
    defaults = dict(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        max_draft_len=4,
        min_draft_len=1,
        confidence_threshold=0.5,
        temperature=1.0,
    )
    defaults.update(kwargs)
    return Eagle3Config(**defaults)


def make_target_fn(batch: int, vocab: int):
    """Returns a mock target_model_fn that yields random softmax probs."""
    def target_model_fn(draft_tokens):
        return [
            F.softmax(torch.randn(batch, vocab), dim=-1)
            for _ in draft_tokens
        ]
    return target_model_fn


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_eagle3_full_decode_step():
    """End-to-end: build decoder, run decode_step, check all invariants."""
    torch.manual_seed(7)
    batch = 1
    config = make_config()
    decoder = Eagle3Decoder(config)
    initial_hidden = torch.randn(batch, D_MODEL)
    target_fn = make_target_fn(batch, VOCAB_SIZE)

    result = decoder.decode_step(initial_hidden, target_fn)

    # Output dict structure
    assert set(result.keys()) >= {"accepted_tokens", "n_accepted", "n_drafted", "acceptance_rate"}, (
        f"Missing keys in result: {result.keys()}"
    )

    # Type checks
    assert isinstance(result["accepted_tokens"], list)
    assert isinstance(result["n_accepted"], int)
    assert isinstance(result["n_drafted"], int)
    assert isinstance(result["acceptance_rate"], float)

    # Numeric invariants
    assert 0 <= result["n_accepted"] <= result["n_drafted"], (
        "n_accepted must be <= n_drafted"
    )
    assert 1 <= result["n_drafted"] <= config.max_draft_len, (
        f"n_drafted={result['n_drafted']} outside [1, {config.max_draft_len}]"
    )
    assert 0.0 <= result["acceptance_rate"] <= 1.0, (
        f"acceptance_rate={result['acceptance_rate']} out of [0, 1]"
    )
    assert len(result["accepted_tokens"]) == result["n_accepted"]


def test_eagle3_multiple_steps_stable():
    """Run several decode steps in sequence; ensure no crashes or NaNs."""
    torch.manual_seed(99)
    batch = 1
    config = make_config(max_draft_len=6, confidence_threshold=0.3)
    decoder = Eagle3Decoder(config)
    target_fn = make_target_fn(batch, VOCAB_SIZE)

    for step in range(5):
        hidden = torch.randn(batch, D_MODEL)
        result = decoder.decode_step(hidden, target_fn)
        assert result["n_drafted"] >= 1, f"Step {step}: n_drafted must be >= 1"
        assert not (result["acceptance_rate"] != result["acceptance_rate"]), (
            f"Step {step}: NaN acceptance_rate"
        )


def test_eagle3_low_threshold_drafts_max():
    """With threshold=0.0, drafter should always draft max_draft_len tokens."""
    torch.manual_seed(3)
    batch = 1
    max_len = 5
    config = make_config(max_draft_len=max_len, confidence_threshold=0.0)
    decoder = Eagle3Decoder(config)
    hidden = torch.randn(batch, D_MODEL)
    target_fn = make_target_fn(batch, VOCAB_SIZE)

    result = decoder.decode_step(hidden, target_fn)
    assert result["n_drafted"] == max_len, (
        f"Expected n_drafted={max_len} with threshold=0.0, got {result['n_drafted']}"
    )


def test_eagle3_high_threshold_drafts_min():
    """With threshold=1.0 (unreachable), only min_draft_len tokens drafted."""
    torch.manual_seed(5)
    batch = 1
    config = make_config(max_draft_len=8, min_draft_len=1, confidence_threshold=1.0)
    decoder = Eagle3Decoder(config)
    hidden = torch.randn(batch, D_MODEL)
    target_fn = make_target_fn(batch, VOCAB_SIZE)

    result = decoder.decode_step(hidden, target_fn)
    assert result["n_drafted"] == config.min_draft_len, (
        f"Expected n_drafted={config.min_draft_len} with threshold=1.0, "
        f"got {result['n_drafted']}"
    )


def test_eagle3_registry_wired():
    """DECODER_REGISTRY['eagle3'] must point to Eagle3Decoder."""
    from src.inference import DECODER_REGISTRY
    assert "eagle3" in DECODER_REGISTRY, (
        "eagle3 not found in DECODER_REGISTRY"
    )
    assert DECODER_REGISTRY["eagle3"] is Eagle3Decoder, (
        "DECODER_REGISTRY['eagle3'] is not Eagle3Decoder"
    )


def test_eagle3_batch_size_2():
    """Decoder should work with batch_size > 1."""
    torch.manual_seed(11)
    batch = 2
    config = make_config()
    decoder = Eagle3Decoder(config)
    hidden = torch.randn(batch, D_MODEL)

    def target_fn_b2(draft_tokens):
        return [F.softmax(torch.randn(batch, VOCAB_SIZE), dim=-1) for _ in draft_tokens]

    result = decoder.decode_step(hidden, target_fn_b2)
    assert result["n_drafted"] >= 1
    assert 0.0 <= result["acceptance_rate"] <= 1.0
