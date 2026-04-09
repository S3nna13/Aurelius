"""Tests for cot_with_prm: CoT beam search with process reward model scoring."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.inference.cot_with_prm import (
    CoTPRMConfig,
    CoTPRMDecoder,
    StepBeam,
    sample_next_step,
    score_step_with_prm,
    split_into_steps,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
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
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


class SimplePRM(nn.Module):
    """Minimal PRM for testing: Linear over embeddings, returns (1, T, 2) logits."""

    def __init__(self, vocab_size: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 16)
        self.linear = nn.Linear(16, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (1, T)
        x = self.embed(input_ids)       # (1, T, 16)
        logits = self.linear(x)         # (1, T, 2)
        return logits


@pytest.fixture
def prm():
    torch.manual_seed(0)
    return SimplePRM(vocab_size=256)


@pytest.fixture
def cot_cfg():
    return CoTPRMConfig(
        n_samples=2,
        max_steps=2,
        beam_width=2,
        max_tokens_per_step=4,
        temperature=1.0,
    )


def encode_fn(text: str) -> list[int]:
    """Simple encoder: use byte values mod 256."""
    return [b % 256 for b in text.encode("utf-8")]


def decode_fn(token_ids: list[int]) -> str:
    """Simple decoder: bytes to latin-1."""
    return bytes(token_ids).decode("latin-1")


@pytest.fixture
def decoder(small_model, prm, cot_cfg):
    return CoTPRMDecoder(
        model=small_model,
        prm=prm,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        config=cot_cfg,
    )


# ---------------------------------------------------------------------------
# Tests: CoTPRMConfig
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = CoTPRMConfig()
    assert cfg.n_samples == 4
    assert cfg.max_steps == 8
    assert cfg.step_delimiter == "\n"
    assert cfg.temperature == 0.8
    assert cfg.max_tokens_per_step == 32
    assert cfg.beam_width == 2


# ---------------------------------------------------------------------------
# Tests: split_into_steps
# ---------------------------------------------------------------------------

def test_split_into_steps_basic():
    steps = split_into_steps("step one\nstep two\nstep three", "\n")
    assert steps == ["step one", "step two", "step three"]


def test_split_into_steps_filters_empty():
    steps = split_into_steps("step one\n\n\nstep two", "\n")
    assert steps == ["step one", "step two"]


def test_split_into_steps_custom_delimiter():
    steps = split_into_steps("a|b|c", "|")
    assert steps == ["a", "b", "c"]


def test_split_into_steps_strips_whitespace():
    steps = split_into_steps("  hello  \n  world  ", "\n")
    assert steps == ["hello", "world"]


# ---------------------------------------------------------------------------
# Tests: StepBeam
# ---------------------------------------------------------------------------

def test_step_beam_init():
    ids = torch.zeros(1, 4, dtype=torch.long)
    beam = StepBeam(step_texts=["step 1"], score=0.5, token_ids=ids)
    assert beam.step_texts == ["step 1"]
    assert beam.score == 0.5
    assert beam.token_ids.shape == (1, 4)


def test_step_beam_add_step_returns_new_beam():
    ids = torch.zeros(1, 4, dtype=torch.long)
    beam = StepBeam(step_texts=["step 1"], score=0.5, token_ids=ids)
    new_ids = torch.ones(3, dtype=torch.long)
    new_beam = beam.add_step("step 2", step_score=0.3, new_ids=new_ids)

    # Original beam unchanged
    assert beam.score == 0.5
    assert len(beam.step_texts) == 1

    # New beam has higher score and extra step
    assert new_beam.score == pytest.approx(0.8)
    assert new_beam.step_texts == ["step 1", "step 2"]


def test_step_beam_text_joins_steps():
    ids = torch.zeros(1, 2, dtype=torch.long)
    beam = StepBeam(step_texts=["alpha", "beta", "gamma"], score=1.0, token_ids=ids)
    assert beam.text() == "alpha\nbeta\ngamma"


# ---------------------------------------------------------------------------
# Tests: score_step_with_prm
# ---------------------------------------------------------------------------

def test_score_step_with_prm_returns_float(prm):
    context_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    step_ids = torch.tensor([4, 5], dtype=torch.long)
    score = score_step_with_prm(prm, context_ids, step_ids)
    assert isinstance(score, float)


def test_score_step_with_prm_in_range(prm):
    context_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    step_ids = torch.tensor([40, 50, 60], dtype=torch.long)
    score = score_step_with_prm(prm, context_ids, step_ids)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests: sample_next_step
# ---------------------------------------------------------------------------

def test_sample_next_step_output_shapes(small_model, cot_cfg):
    torch.manual_seed(7)
    context_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    generated_ids, decoded_text = sample_next_step(small_model, context_ids, cot_cfg)
    # Should return at most max_tokens_per_step tokens
    assert generated_ids.ndim == 1
    assert 1 <= len(generated_ids) <= cot_cfg.max_tokens_per_step
    assert isinstance(decoded_text, str)


# ---------------------------------------------------------------------------
# Tests: CoTPRMDecoder.decode
# ---------------------------------------------------------------------------

def test_decoder_returns_correct_keys(decoder):
    result = decoder.decode("What is 2+2?")
    assert "answer" in result
    assert "steps" in result
    assert "score" in result
    assert "n_beams" in result


def test_decoder_answer_is_string(decoder):
    result = decoder.decode("Solve x squared equals 4.")
    assert isinstance(result["answer"], str)


def test_decoder_n_beams_equals_config(decoder, cot_cfg):
    result = decoder.decode("Explain photosynthesis.")
    assert result["n_beams"] == cot_cfg.beam_width
