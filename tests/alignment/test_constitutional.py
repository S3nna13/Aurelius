"""Tests for Constitutional AI self-critique."""

from unittest.mock import MagicMock

import pytest
import torch

from src.alignment.constitutional import (
    ConstitutionalConfig,
    ConstitutionalReviser,
    _build_critique_prompt,
    _build_revision_prompt,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def _fake_tokenizer():
    tok = MagicMock()
    tok.encode = lambda text: [ord(c) % 256 for c in text[:20]]
    tok.decode = lambda ids: "revised response"
    return tok


def test_critique_prompt_contains_principle():
    """_build_critique_prompt must include the principle text."""
    p = _build_critique_prompt("What is 2+2?", "The answer is 5.", "Be accurate.")
    assert "Be accurate." in p
    assert "The answer is 5." in p


def test_revision_prompt_contains_critique():
    """_build_revision_prompt must include both critique and principle."""
    p = _build_revision_prompt("Q?", "A.", "Math is wrong.", "Be accurate.")
    assert "Math is wrong." in p
    assert "Be accurate." in p


def test_constitutional_reviser_returns_response(small_model):
    """ConstitutionalReviser.apply must return a dict with final_response."""
    cfg = ConstitutionalConfig(
        principles=["Be helpful."],
        num_rounds=1,
        max_critique_tokens=8,
        max_revision_tokens=8,
    )
    reviser = ConstitutionalReviser(small_model, _fake_tokenizer(), cfg)
    result = reviser.apply("Hello", "Hi there!")
    assert "final_response" in result
    assert "initial_response" in result
    assert "rounds" in result
    assert isinstance(result["final_response"], str)


def test_constitutional_no_revision_on_no_issues(small_model):
    """When critique says 'No issues', response should not be revised."""
    tok = MagicMock()
    tok.encode = lambda text: [1, 2, 3]
    tok.decode = lambda ids: "No issues found"  # critique says no issues

    cfg = ConstitutionalConfig(
        principles=["Be accurate."],
        num_rounds=1,
        max_critique_tokens=4,
        max_revision_tokens=4,
    )
    reviser = ConstitutionalReviser(small_model, tok, cfg)
    result = reviser.apply("Q?", "original response")

    # With "No issues" critique, response should be unchanged
    round_record = result["rounds"][0]["revisions"][0]
    assert not round_record["revised"]
