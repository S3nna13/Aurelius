"""Tests for src/inference/self_speculative_decoding.py"""

from __future__ import annotations

import torch
from aurelius.inference.self_speculative_decoding import (
    DraftQualityMonitor,
    EarlyExitHead,
    LayeredModel,
    SelfSpeculativeDecoder,
)

# ---------------------------------------------------------------------------
# Shared test fixtures / constants
# ---------------------------------------------------------------------------

N_LAYERS = 4
D_MODEL = 16
VOCAB_SIZE = 32
N_DRAFT_TOKENS = 3
DRAFT_LAYER = 1  # exit after layer 1 of 4
B, T = 2, 6


def make_model() -> LayeredModel:
    return LayeredModel(n_layers=N_LAYERS, d_model=D_MODEL, vocab_size=VOCAB_SIZE)


def make_draft_head() -> EarlyExitHead:
    return EarlyExitHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE, draft_layer=DRAFT_LAYER)


def make_decoder() -> SelfSpeculativeDecoder:
    model = make_model()
    draft_head = make_draft_head()
    return SelfSpeculativeDecoder(
        model=model,
        draft_head=draft_head,
        draft_layer=DRAFT_LAYER,
        n_draft_tokens=N_DRAFT_TOKENS,
    )


# ---------------------------------------------------------------------------
# 1. EarlyExitHead output shape (B, T, vocab_size)
# ---------------------------------------------------------------------------


def test_early_exit_head_output_shape():
    head = make_draft_head()
    x = torch.randn(B, T, D_MODEL)
    out = head(x)
    assert out.shape == (B, T, VOCAB_SIZE), f"Expected ({B}, {T}, {VOCAB_SIZE}), got {out.shape}"


# ---------------------------------------------------------------------------
# 2. EarlyExitHead output is finite
# ---------------------------------------------------------------------------


def test_early_exit_head_output_finite():
    head = make_draft_head()
    x = torch.randn(B, T, D_MODEL)
    out = head(x)
    assert torch.isfinite(out).all(), "EarlyExitHead produced non-finite values"


# ---------------------------------------------------------------------------
# 3. LayeredModel.forward_to_layer returns hidden shape (B, T, d_model)
# ---------------------------------------------------------------------------


def test_forward_to_layer_hidden_shape():
    model = make_model()
    x = torch.randn(B, T, D_MODEL)
    hidden, _ = model.forward_to_layer(x, up_to_layer=DRAFT_LAYER)
    assert hidden.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {hidden.shape}"


# ---------------------------------------------------------------------------
# 4. LayeredModel.forward_to_layer returns logits shape (B, T, vocab_size)
# ---------------------------------------------------------------------------


def test_forward_to_layer_logits_shape():
    model = make_model()
    x = torch.randn(B, T, D_MODEL)
    _, logits = model.forward_to_layer(x, up_to_layer=DRAFT_LAYER)
    assert logits.shape == (B, T, VOCAB_SIZE), (
        f"Expected ({B}, {T}, {VOCAB_SIZE}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# 5. LayeredModel.forward_remaining output shape (B, T, vocab_size)
# ---------------------------------------------------------------------------


def test_forward_remaining_output_shape():
    model = make_model()
    x = torch.randn(B, T, D_MODEL)
    hidden, _ = model.forward_to_layer(x, up_to_layer=DRAFT_LAYER)
    out = model.forward_remaining(hidden, from_layer=DRAFT_LAYER)
    assert out.shape == (B, T, VOCAB_SIZE), f"Expected ({B}, {T}, {VOCAB_SIZE}), got {out.shape}"


# ---------------------------------------------------------------------------
# 6. Both forward_to_layer and forward_remaining produce finite output
# ---------------------------------------------------------------------------


def test_forward_paths_finite():
    model = make_model()
    x = torch.randn(B, T, D_MODEL)
    hidden, logits_early = model.forward_to_layer(x, up_to_layer=DRAFT_LAYER)
    logits_full = model.forward_remaining(hidden, from_layer=DRAFT_LAYER)
    assert torch.isfinite(logits_early).all(), "forward_to_layer logits not finite"
    assert torch.isfinite(logits_full).all(), "forward_remaining logits not finite"


# ---------------------------------------------------------------------------
# 7. SelfSpeculativeDecoder._draft returns (draft_tokens, draft_probs)
# ---------------------------------------------------------------------------


def test_draft_returns_tuple():
    dec = make_decoder()
    hidden = torch.randn(1, T, D_MODEL)
    result = dec._draft(hidden)
    assert isinstance(result, tuple) and len(result) == 2, "_draft must return a 2-tuple"


# ---------------------------------------------------------------------------
# 8. draft_tokens shape is (n_draft_tokens,) and dtype long
# ---------------------------------------------------------------------------


def test_draft_tokens_shape_and_dtype():
    dec = make_decoder()
    hidden = torch.randn(1, T, D_MODEL)
    draft_tokens, _ = dec._draft(hidden)
    assert draft_tokens.shape == (N_DRAFT_TOKENS,), (
        f"Expected ({N_DRAFT_TOKENS},), got {draft_tokens.shape}"
    )
    assert draft_tokens.dtype == torch.long, f"Expected long, got {draft_tokens.dtype}"


# ---------------------------------------------------------------------------
# 9. draft_probs shape is (n_draft_tokens, vocab_size)
# ---------------------------------------------------------------------------


def test_draft_probs_shape():
    dec = make_decoder()
    hidden = torch.randn(1, T, D_MODEL)
    _, draft_probs = dec._draft(hidden)
    assert draft_probs.shape == (N_DRAFT_TOKENS, VOCAB_SIZE), (
        f"Expected ({N_DRAFT_TOKENS}, {VOCAB_SIZE}), got {draft_probs.shape}"
    )


# ---------------------------------------------------------------------------
# 10. _verify returns (accepted_tokens, int)
# ---------------------------------------------------------------------------


def test_verify_returns_tuple():
    dec = make_decoder()
    input_ids = torch.randint(0, VOCAB_SIZE, (T,))
    draft_tokens = torch.randint(0, VOCAB_SIZE, (N_DRAFT_TOKENS,))
    result = dec._verify(input_ids, draft_tokens)
    assert isinstance(result, tuple) and len(result) == 2
    accepted_tokens, n_accepted = result
    assert isinstance(n_accepted, int), f"n_accepted must be int, got {type(n_accepted)}"
    assert accepted_tokens.dtype == torch.long


# ---------------------------------------------------------------------------
# 11. n_accepted <= n_draft_tokens
# ---------------------------------------------------------------------------


def test_verify_n_accepted_bounded():
    dec = make_decoder()
    input_ids = torch.randint(0, VOCAB_SIZE, (T,))
    draft_tokens = torch.randint(0, VOCAB_SIZE, (N_DRAFT_TOKENS,))
    _, n_accepted = dec._verify(input_ids, draft_tokens)
    assert 0 <= n_accepted <= N_DRAFT_TOKENS, (
        f"n_accepted={n_accepted} out of range [0, {N_DRAFT_TOKENS}]"
    )


# ---------------------------------------------------------------------------
# 12. generate output shape (max_new_tokens,)
# ---------------------------------------------------------------------------


def test_generate_output_shape():
    dec = make_decoder()
    prompt = torch.randint(0, VOCAB_SIZE, (5,))
    max_new = 8
    out = dec.generate(prompt, max_new_tokens=max_new)
    assert out.shape == (max_new,), f"Expected ({max_new},), got {out.shape}"


# ---------------------------------------------------------------------------
# 13. generate output dtype is long
# ---------------------------------------------------------------------------


def test_generate_output_dtype():
    dec = make_decoder()
    prompt = torch.randint(0, VOCAB_SIZE, (5,))
    out = dec.generate(prompt, max_new_tokens=6)
    assert out.dtype == torch.long, f"Expected torch.long, got {out.dtype}"


# ---------------------------------------------------------------------------
# 14. DraftQualityMonitor.record increments totals
# ---------------------------------------------------------------------------


def test_monitor_record_increments():
    mon = DraftQualityMonitor()
    assert mon.total_drafted == 0
    assert mon.total_accepted == 0
    mon.record(n_drafted=N_DRAFT_TOKENS, n_accepted=2)
    assert mon.total_drafted == N_DRAFT_TOKENS
    assert mon.total_accepted == 2
    mon.record(n_drafted=N_DRAFT_TOKENS, n_accepted=1)
    assert mon.total_drafted == 2 * N_DRAFT_TOKENS
    assert mon.total_accepted == 3


# ---------------------------------------------------------------------------
# 15. acceptance_rate in [0, 1]
# ---------------------------------------------------------------------------


def test_monitor_acceptance_rate_range():
    mon = DraftQualityMonitor()
    # Empty monitor should return 0 without error
    rate = mon.acceptance_rate()
    assert 0.0 <= rate <= 1.0

    mon.record(n_drafted=4, n_accepted=3)
    rate = mon.acceptance_rate()
    assert 0.0 <= rate <= 1.0, f"acceptance_rate={rate} out of [0, 1]"

    mon.record(n_drafted=4, n_accepted=0)
    rate = mon.acceptance_rate()
    assert 0.0 <= rate <= 1.0
