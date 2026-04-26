"""Tests for src/inference/chain_of_draft.py — 12 focused tests.

Paper: Chain of Draft: Thinking Faster by Writing Less (arXiv:2502.18600).
"""

from __future__ import annotations

import pytest
import torch

from src.inference.chain_of_draft import ChainOfDraftConfig, ChainOfDraftDecoder
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures — tiny model matching the spec:
#   n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
#   d_ff=128, vocab_size=256, max_seq_len=64
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cfg() -> AureliusConfig:
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


@pytest.fixture(scope="module")
def tiny_model(tiny_cfg: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(0)
    model = AureliusTransformer(tiny_cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def cod_cfg() -> ChainOfDraftConfig:
    return ChainOfDraftConfig(
        max_draft_steps=3,
        draft_budget=5,
        step_separator_id=10,
        answer_prefix_ids=[],
    )


@pytest.fixture(scope="module")
def decoder(tiny_model: AureliusTransformer, cod_cfg: ChainOfDraftConfig) -> ChainOfDraftDecoder:
    return ChainOfDraftDecoder(model=tiny_model, config=cod_cfg)


def _make_input(seq_len: int = 4, batch: int = 1, vocab: int = 256) -> torch.Tensor:
    """Create a deterministic integer input tensor of shape (batch, seq_len)."""
    torch.manual_seed(7)
    return torch.randint(0, vocab, (batch, seq_len))


# ---------------------------------------------------------------------------
# Test 1: generate_draft_step returns (B, k) tokens where k <= max_tokens
# ---------------------------------------------------------------------------


def test_generate_draft_step_shape(decoder: ChainOfDraftDecoder) -> None:
    input_ids = _make_input(seq_len=4)
    max_tokens = 5
    new_tokens, _hit_stop = decoder.generate_draft_step(
        input_ids, max_tokens=max_tokens, stop_ids=[]
    )
    assert new_tokens.ndim == 2, "new_tokens must be 2-D"
    assert new_tokens.size(0) == input_ids.size(0), "batch dimension must match"
    assert new_tokens.size(1) <= max_tokens, (
        f"k={new_tokens.size(1)} must be <= max_tokens={max_tokens}"
    )


# ---------------------------------------------------------------------------
# Test 2: generate_draft_step stops at stop_id before budget is exhausted
# ---------------------------------------------------------------------------


def test_generate_draft_step_stops_at_stop_id(
    tiny_model: AureliusTransformer,
    tiny_cfg: AureliusConfig,
) -> None:
    """Force a stop_id hit by using the token the model will produce next."""
    input_ids = _make_input(seq_len=4)

    decoder_probe = ChainOfDraftDecoder(
        model=tiny_model,
        config=ChainOfDraftConfig(max_draft_steps=1, draft_budget=10),
    )
    next_tok, _ = decoder_probe._greedy_next_token(input_ids)
    forced_stop_id = int(next_tok[0, 0])

    decoder_with_stop = ChainOfDraftDecoder(
        model=tiny_model,
        config=ChainOfDraftConfig(max_draft_steps=1, draft_budget=10),
    )
    new_tokens, hit_stop = decoder_with_stop.generate_draft_step(
        input_ids, max_tokens=10, stop_ids=[forced_stop_id]
    )
    assert hit_stop is True, "hit_stop must be True when stop_id is the first generated token"
    assert new_tokens.size(1) == 1, "Only the stop token should have been appended"


# ---------------------------------------------------------------------------
# Test 3: decode returns an output_ids Tensor
# ---------------------------------------------------------------------------


def test_decode_returns_tensor(decoder: ChainOfDraftDecoder) -> None:
    input_ids = _make_input(seq_len=4)
    output_ids, _meta = decoder.decode(input_ids, max_answer_tokens=5)
    assert isinstance(output_ids, torch.Tensor), "output_ids must be a Tensor"


# ---------------------------------------------------------------------------
# Test 4: metadata dict has required keys
# ---------------------------------------------------------------------------


def test_decode_metadata_keys(decoder: ChainOfDraftDecoder) -> None:
    input_ids = _make_input(seq_len=4)
    _output_ids, meta = decoder.decode(input_ids, max_answer_tokens=5)
    assert "n_draft_steps" in meta
    assert "draft_token_count" in meta
    assert "answer_token_count" in meta


# ---------------------------------------------------------------------------
# Test 5: draft_token_count <= max_draft_steps * draft_budget
# ---------------------------------------------------------------------------


def test_draft_token_count_bounded(decoder: ChainOfDraftDecoder) -> None:
    cfg = decoder.config
    input_ids = _make_input(seq_len=4)
    _output_ids, meta = decoder.decode(input_ids, max_answer_tokens=5)
    upper_bound = cfg.max_draft_steps * cfg.draft_budget
    assert meta["draft_token_count"] <= upper_bound, (
        f"draft_token_count={meta['draft_token_count']} > {upper_bound}"
    )


# ---------------------------------------------------------------------------
# Test 6: n_draft_steps <= max_draft_steps
# ---------------------------------------------------------------------------


def test_n_draft_steps_bounded(decoder: ChainOfDraftDecoder) -> None:
    input_ids = _make_input(seq_len=4)
    _output_ids, meta = decoder.decode(input_ids, max_answer_tokens=5)
    assert meta["n_draft_steps"] <= decoder.config.max_draft_steps


# ---------------------------------------------------------------------------
# Test 7: output_ids is longer than input_ids (generation happened)
# ---------------------------------------------------------------------------


def test_output_longer_than_input(decoder: ChainOfDraftDecoder) -> None:
    input_ids = _make_input(seq_len=4)
    output_ids, _meta = decoder.decode(input_ids, max_answer_tokens=5)
    assert output_ids.size(1) > input_ids.size(1), "output must be longer than the input prompt"


# ---------------------------------------------------------------------------
# Test 8: compute_draft_efficiency < 1.0 when draft < cot
# ---------------------------------------------------------------------------


def test_draft_efficiency_less_than_one() -> None:
    ratio = ChainOfDraftDecoder.compute_draft_efficiency(
        draft_token_count=5,
        cot_token_count=50,
    )
    assert ratio < 1.0, f"Expected ratio < 1.0, got {ratio}"


# ---------------------------------------------------------------------------
# Test 9: compute_draft_efficiency == 1.0 when equal
# ---------------------------------------------------------------------------


def test_draft_efficiency_equal_to_one() -> None:
    ratio = ChainOfDraftDecoder.compute_draft_efficiency(
        draft_token_count=20,
        cot_token_count=20,
    )
    assert ratio == pytest.approx(1.0), f"Expected ratio == 1.0, got {ratio}"


# ---------------------------------------------------------------------------
# Test 10: No NaN or Inf in generated logits
# ---------------------------------------------------------------------------


def test_no_nan_inf_in_logits(tiny_model: AureliusTransformer) -> None:
    decoder_local = ChainOfDraftDecoder(
        model=tiny_model,
        config=ChainOfDraftConfig(max_draft_steps=2, draft_budget=3),
    )
    input_ids = _make_input(seq_len=4)
    _next_tok, last_logits = decoder_local._greedy_next_token(input_ids)
    assert not torch.isnan(last_logits).any(), "NaN found in logits"
    assert not torch.isinf(last_logits).any(), "Inf found in logits"


# ---------------------------------------------------------------------------
# Test 11: Determinism under torch.manual_seed (greedy decode)
# ---------------------------------------------------------------------------


def test_determinism_under_seed(tiny_model: AureliusTransformer) -> None:
    decoder_local = ChainOfDraftDecoder(
        model=tiny_model,
        config=ChainOfDraftConfig(max_draft_steps=2, draft_budget=3),
    )
    input_ids = _make_input(seq_len=4)

    torch.manual_seed(42)
    out1, meta1 = decoder_local.decode(input_ids, max_answer_tokens=4)

    torch.manual_seed(42)
    out2, meta2 = decoder_local.decode(input_ids, max_answer_tokens=4)

    assert torch.equal(out1, out2), "Greedy decode must be deterministic"
    assert meta1 == meta2, "Metadata must match across identical runs"


# ---------------------------------------------------------------------------
# Test 12: T=1 input (single token prompt) works
# ---------------------------------------------------------------------------


def test_single_token_input(decoder: ChainOfDraftDecoder) -> None:
    input_ids = _make_input(seq_len=1)
    output_ids, meta = decoder.decode(input_ids, max_answer_tokens=3)
    assert isinstance(output_ids, torch.Tensor)
    assert output_ids.size(1) > 1, "Output must be longer than the single-token input"
    assert "n_draft_steps" in meta
