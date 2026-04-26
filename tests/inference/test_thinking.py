"""Tests for think-before-answer generation (thinking.py)."""

import pytest
import torch

from src.inference.thinking import (
    ThinkingConfig,
    ThinkingReranker,
    ThinkingResult,
    generate_answer,
    generate_thinking,
    think_and_answer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


@pytest.fixture
def prompt(small_cfg):
    torch.manual_seed(0)
    return torch.randint(0, small_cfg.vocab_size, (1, 4))


@pytest.fixture
def fast_thinking_cfg():
    """ThinkingConfig with tiny budgets for fast tests."""
    return ThinkingConfig(
        think_token_budget=8,
        answer_token_budget=8,
        temperature=1.0,
        top_p=0.9,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_thinking_config_defaults():
    cfg = ThinkingConfig()
    assert cfg.think_token_budget == 512
    assert cfg.answer_token_budget == 256


def test_generate_thinking_returns_tensor(small_model, prompt, fast_thinking_cfg):
    thinking = generate_thinking(small_model, prompt, fast_thinking_cfg)
    assert isinstance(thinking, torch.Tensor)


def test_generate_thinking_max_budget(small_model, prompt, fast_thinking_cfg):
    thinking = generate_thinking(small_model, prompt, fast_thinking_cfg)
    assert thinking.numel() <= fast_thinking_cfg.think_token_budget


def test_generate_answer_returns_tensor(small_model, prompt, fast_thinking_cfg):
    thinking = generate_thinking(small_model, prompt, fast_thinking_cfg)
    answer = generate_answer(small_model, prompt, thinking, fast_thinking_cfg)
    assert isinstance(answer, torch.Tensor)


def test_generate_answer_max_budget(small_model, prompt, fast_thinking_cfg):
    thinking = generate_thinking(small_model, prompt, fast_thinking_cfg)
    answer = generate_answer(small_model, prompt, thinking, fast_thinking_cfg)
    assert answer.numel() <= fast_thinking_cfg.answer_token_budget


def test_think_and_answer_returns_result(small_model, prompt, fast_thinking_cfg):
    result = think_and_answer(small_model, prompt, fast_thinking_cfg)
    assert isinstance(result, ThinkingResult)


def test_thinking_result_fields(small_model, prompt, fast_thinking_cfg):
    result = think_and_answer(small_model, prompt, fast_thinking_cfg)
    assert hasattr(result, "thinking_ids")
    assert hasattr(result, "answer_ids")
    assert hasattr(result, "full_ids")
    assert isinstance(result.thinking_ids, torch.Tensor)
    assert isinstance(result.answer_ids, torch.Tensor)
    assert isinstance(result.full_ids, torch.Tensor)


def test_thinking_result_total_tokens(small_model, prompt, fast_thinking_cfg):
    result = think_and_answer(small_model, prompt, fast_thinking_cfg)
    assert result.total_tokens == result.thinking_tokens_used + result.answer_tokens_used


def test_full_ids_concatenated(small_model, prompt, fast_thinking_cfg):
    result = think_and_answer(small_model, prompt, fast_thinking_cfg)
    expected_len = result.thinking_ids.numel() + result.answer_ids.numel()
    assert result.full_ids.numel() == expected_len


def test_thinking_reranker_returns_result(small_model, prompt, fast_thinking_cfg):
    reranker = ThinkingReranker(
        model=small_model,
        n_candidates=2,
        cfg=fast_thinking_cfg,
    )
    result = reranker.generate_best(prompt)
    assert isinstance(result, ThinkingResult)


def test_thinking_reranker_n_candidates_2(small_model, prompt, fast_thinking_cfg):
    reranker = ThinkingReranker(
        model=small_model,
        n_candidates=2,
        cfg=fast_thinking_cfg,
    )
    result = reranker.generate_best(prompt)
    assert isinstance(result, ThinkingResult)
    assert result.total_tokens >= 0
