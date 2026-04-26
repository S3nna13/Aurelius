"""Tests for CoTTrainer — chain-of-thought scratchpad training."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.cot_trainer import (
    CoTConfig,
    CoTExample,
    CoTTrainer,
    build_cot_labels,
    cot_loss,
    extract_cot_answer,
    generate_arithmetic_dataset,
    generate_arithmetic_example,
    verify_answer_correct,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


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
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


def _simple_encode(text: str) -> list[int]:
    """Deterministic byte-level tokenizer that stays within vocab_size=256."""
    return list(text.encode("utf-8", errors="replace"))


def _simple_decode(ids: list[int]) -> str:
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


# ── CoTExample ───────────────────────────────────────────────────────────────


def test_cot_example_full_text():
    ex = CoTExample(
        question="How many apples?",
        scratchpad="Step 1: count them.",
        answer="5",
    )
    text = ex.full_text()
    assert "How many apples?" in text
    assert "####" in text
    assert "5" in text


# ── Synthetic data generation ────────────────────────────────────────────────


def test_generate_arithmetic_example_fields():
    ex = generate_arithmetic_example()
    assert isinstance(ex.question, str) and len(ex.question) > 0
    assert isinstance(ex.scratchpad, str) and len(ex.scratchpad) > 0
    assert isinstance(ex.answer, str) and len(ex.answer) > 0


def test_generate_arithmetic_answer_numeric():
    for _ in range(10):
        ex = generate_arithmetic_example()
        # answer must be parseable as a number
        float(ex.answer)


def test_generate_arithmetic_dataset_count():
    ds = generate_arithmetic_dataset(n_examples=7)
    assert len(ds) == 7


def test_generate_arithmetic_dataset_reproducible():
    ds1 = generate_arithmetic_dataset(n_examples=5, seed=99)
    ds2 = generate_arithmetic_dataset(n_examples=5, seed=99)
    for a, b in zip(ds1, ds2):
        assert a.question == b.question
        assert a.answer == b.answer


# ── build_cot_labels ─────────────────────────────────────────────────────────


def _make_simple_sequence(cfg: CoTConfig) -> tuple[torch.Tensor, list[int]]:
    """Return (input_ids, delimiter_token_ids) with a known structure."""
    # Construct: question_tokens + delimiter_tokens + answer_tokens
    question_ids = [10, 11, 12]
    delimiter_ids = [ord("#"), ord("#"), ord("#"), ord("#")]  # "####"
    answer_ids = [20, 21]
    all_ids = question_ids + delimiter_ids + answer_ids
    return torch.tensor(all_ids, dtype=torch.long), delimiter_ids


def test_build_cot_labels_ignores_prefix():
    cfg = CoTConfig()
    input_ids, delim_ids = _make_simple_sequence(cfg)
    labels, weights = build_cot_labels(input_ids, delim_ids, cfg)
    assert labels.shape == input_ids.shape
    # All positions before the delimiter should be -100
    delim_start = 3  # after 3 question tokens
    assert (labels[:delim_start] == -100).all(), "Prefix tokens should be ignored"


def test_build_cot_labels_weight_after_delimiter():
    cfg = CoTConfig(scratchpad_weight=1.0, answer_weight=5.0)
    input_ids, delim_ids = _make_simple_sequence(cfg)
    labels, weights = build_cot_labels(input_ids, delim_ids, cfg)
    # Tokens after the delimiter should have higher weight than scratchpad
    delim_start = 3
    delim_end = delim_start + len(delim_ids)
    answer_weights = weights[delim_end:]
    # Answer tokens must have weight >= answer_weight
    assert (answer_weights >= cfg.answer_weight).all(), (
        f"Answer tokens should have weight={cfg.answer_weight}, got {answer_weights}"
    )


# ── cot_loss ─────────────────────────────────────────────────────────────────


def test_cot_loss_scalar(small_model):
    B, T = 2, 16
    vocab = 256
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab, (B, T))
    labels = input_ids.clone()
    labels[:, :4] = -100
    weights = torch.ones(B, T)
    weights[:, 8:] = 5.0
    CoTConfig()
    loss = cot_loss(small_model, input_ids, labels, weights)
    assert loss.shape == (), "cot_loss must return a scalar tensor"
    assert loss.item() > 0
    assert torch.isfinite(loss)


# ── CoTTrainer ───────────────────────────────────────────────────────────────


def test_cot_trainer_tokenize_shape(small_model):
    cfg = CoTConfig(max_seq_len=512)
    trainer = CoTTrainer(small_model, None, cfg, _simple_encode)
    ex = CoTExample(
        question="Alice has 3 apples.",
        scratchpad="She has 3.",
        answer="3",
    )
    input_ids, labels, weights = trainer.tokenize_example(ex)
    T = input_ids.shape[0]
    assert input_ids.shape == (T,), "input_ids must be 1D"
    assert labels.shape == (T,), "labels must be 1D"
    assert weights.shape == (T,), "weights must be 1D"
    assert T > 0


def test_cot_trainer_train_step_metrics(small_model, small_cfg):
    cfg = CoTConfig(max_seq_len=512)
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
    trainer = CoTTrainer(small_model, optimizer, cfg, _simple_encode)
    examples = generate_arithmetic_dataset(n_examples=4, seed=0)
    result = trainer.train_step(examples)
    assert "loss" in result
    assert isinstance(result["loss"], float)
    assert result["loss"] > 0


# ── extract_cot_answer ────────────────────────────────────────────────────────


def test_extract_cot_answer_basic():
    text = "Let me think.\n#### 42"
    ans = extract_cot_answer(text)
    assert ans == "42"


def test_extract_cot_answer_none():
    text = "No delimiter here"
    ans = extract_cot_answer(text)
    assert ans is None


# ── verify_answer_correct ─────────────────────────────────────────────────────


def test_verify_answer_correct_exact():
    assert verify_answer_correct("42", "42") is True


def test_verify_answer_correct_float():
    assert verify_answer_correct("3.14", "3.14") is True


def test_verify_answer_wrong():
    assert verify_answer_correct("42", "43") is False
