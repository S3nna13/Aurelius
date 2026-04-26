"""Tests for src/alignment/step_labeler.py"""

import pytest
import torch

from src.alignment.step_labeler import (
    LabeledChain,
    StepLabel,
    StepLabeler,
    StepLabelerConfig,
    answers_match,
    extract_answer,
    normalize_answer,
    parse_steps,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

STEP_DELIMITER_ID = 5
VOCAB_SIZE = 256


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=64,
    )
    torch.manual_seed(42)
    return AureliusTransformer(cfg)


@pytest.fixture
def labeler_cfg():
    return StepLabelerConfig(
        step_delimiter=STEP_DELIMITER_ID,
        n_completions=2,
        max_completion_tokens=8,
        temperature=1.0,
    )


@pytest.fixture
def labeler(small_model, labeler_cfg):
    return StepLabeler(small_model, labeler_cfg)


# ── parse_steps ──────────────────────────────────────────────────────────────


def test_parse_steps_basic():
    """Splits correctly at delimiter token."""
    chain = [1, 2, STEP_DELIMITER_ID, 3, 4, STEP_DELIMITER_ID, 6, 7]
    steps = parse_steps(chain, STEP_DELIMITER_ID)
    assert len(steps) == 3
    assert steps[0] == [1, 2]
    assert steps[1] == [3, 4]
    assert steps[2] == [6, 7]


def test_parse_steps_no_delimiter():
    """Returns a single step when no delimiter is present."""
    chain = [10, 20, 30]
    steps = parse_steps(chain, STEP_DELIMITER_ID)
    assert len(steps) == 1
    assert steps[0] == [10, 20, 30]


# ── extract_answer ────────────────────────────────────────────────────────────


def test_extract_answer_found():
    """Extracts answer text after #### delimiter."""
    text = "some text #### 42"
    result = extract_answer(text, "####")
    assert result == "42"


def test_extract_answer_not_found():
    """Returns None when delimiter is not in text."""
    text = "no delimiter here"
    result = extract_answer(text, "####")
    assert result is None


# ── normalize_answer ──────────────────────────────────────────────────────────


def test_normalize_answer():
    """Lowercases, strips punctuation, and collapses whitespace."""
    result = normalize_answer("  Hello, World!  ")
    assert result == "hello world"


# ── answers_match ─────────────────────────────────────────────────────────────


def test_answers_match_identical():
    """Identical strings match."""
    assert answers_match("42", "42") is True


def test_answers_match_normalized():
    """'42.' and '42' match after normalization."""
    assert answers_match("42.", "42", normalize=True) is True


# ── StepLabel enum ────────────────────────────────────────────────────────────


def test_step_label_enum():
    """CORRECT=1, INCORRECT=-1, NEUTRAL=0."""
    assert StepLabel.CORRECT == 1
    assert StepLabel.INCORRECT == -1
    assert StepLabel.NEUTRAL == 0


# ── StepLabeler.label_step ────────────────────────────────────────────────────


def test_label_step_returns_tuple(labeler):
    """label_step returns a (StepLabel, float) tuple."""
    prefix = torch.randint(1, VOCAB_SIZE, (6,))
    result = labeler.label_step(prefix, correct_answer="42")
    assert isinstance(result, tuple)
    assert len(result) == 2
    label, fraction = result
    assert isinstance(label, StepLabel)
    assert isinstance(fraction, float)


def test_label_step_fraction_in_range(labeler):
    """Fraction returned by label_step is in [0, 1]."""
    prefix = torch.randint(1, VOCAB_SIZE, (6,))
    _, fraction = labeler.label_step(prefix, correct_answer="42")
    assert 0.0 <= fraction <= 1.0


# ── StepLabeler.label_chain ───────────────────────────────────────────────────


def test_label_chain_returns_labeled_chain(labeler):
    """label_chain returns a LabeledChain instance."""
    prompt_ids = [1, 2, 3]
    # chain: step_a [5] step_b (delimiter=5)
    chain_ids = [10, 11, STEP_DELIMITER_ID, 12, 13]
    result = labeler.label_chain(prompt_ids, chain_ids, correct_answer="42")
    assert isinstance(result, LabeledChain)


def test_label_chain_step_count(labeler):
    """Number of steps equals number of delimiters + 1."""
    prompt_ids = [1, 2]
    # 2 delimiters → 3 steps
    chain_ids = [
        10,
        11,
        STEP_DELIMITER_ID,
        20,
        21,
        STEP_DELIMITER_ID,
        30,
        31,
    ]
    result = labeler.label_chain(prompt_ids, chain_ids, correct_answer="99")
    assert len(result.steps) == 3
