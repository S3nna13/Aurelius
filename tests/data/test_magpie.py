"""Tests for Magpie self-instruct synthesizer."""
import torch
import pytest
from unittest.mock import MagicMock, patch
from src.data.magpie import MagpieSynthesizer, MagpieConfig, MagpieSample
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


class MockTokenizer:
    """Minimal tokenizer stub for testing."""
    def encode(self, text: str) -> list[int]:
        # Simple char-level encoding for testing
        return [ord(c) % 100 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids)


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2, d_model=256, n_heads=4, n_kv_heads=2,
        head_dim=64, d_ff=512, vocab_size=256, max_seq_len=128,
    )
    return AureliusTransformer(cfg)


@pytest.fixture
def synthesizer(small_model):
    tok = MockTokenizer()
    cfg = MagpieConfig(
        max_instruction_tokens=8,
        max_response_tokens=8,
        eos_token_id=2,
        pre_query_prefix="<|user|>",
    )
    return MagpieSynthesizer(small_model, tok, cfg)


def test_generate_sample_returns_magpie_sample(synthesizer):
    sample = synthesizer.generate_sample()
    assert isinstance(sample, MagpieSample)


def test_generate_sample_has_text(synthesizer):
    sample = synthesizer.generate_sample()
    assert isinstance(sample.instruction, str)
    assert isinstance(sample.response, str)


def test_generate_sample_has_ids(synthesizer):
    sample = synthesizer.generate_sample()
    assert isinstance(sample.instruction_ids, list)
    assert isinstance(sample.response_ids, list)


def test_generate_batch_length(synthesizer):
    samples = synthesizer.generate_batch(3)
    assert len(samples) == 3
    assert all(isinstance(s, MagpieSample) for s in samples)


def test_model_is_in_eval_mode(synthesizer):
    assert not synthesizer.model.training
