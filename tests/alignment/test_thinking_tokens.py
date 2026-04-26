"""Tests for src/alignment/thinking_tokens.py"""

import pytest
import torch

from src.alignment.thinking_tokens import (
    THINK_END,
    THINK_END_TOKEN_ID,
    THINK_START,
    THINK_START_TOKEN_ID,
    ThinkingFormat,
    ThinkingInferenceHelper,
    ThinkingLossWeights,
    ThinkingSFTDataset,
)

# ---------------------------------------------------------------------------
# ThinkingFormat
# ---------------------------------------------------------------------------


def test_wrap_and_extract():
    fmt = ThinkingFormat()
    text = fmt.wrap_thinking("Let me think", "42")
    assert f"{THINK_START}Let me think{THINK_END}42" == text
    thought, response = fmt.extract_thinking(text)
    assert thought == "Let me think"
    assert response == "42"


def test_extract_no_thinking():
    fmt = ThinkingFormat()
    thought, response = fmt.extract_thinking("just a response")
    assert thought == ""
    assert response == "just a response"


# ---------------------------------------------------------------------------
# ThinkingLossWeights
# ---------------------------------------------------------------------------


def _make_token_ids():
    """Helper: [prompt_tok, THINK_START, think_tok, THINK_END, answer_tok]"""
    return [1, THINK_START_TOKEN_ID, 2, THINK_END_TOKEN_ID, 3]


def test_loss_weights_inside_think():
    lw = ThinkingLossWeights(think_weight=0.5, answer_weight=1.0)
    ids = _make_token_ids()
    w = lw.compute_weights(ids)
    # index 2 is the token inside <think>
    assert w[2].item() == pytest.approx(0.5)


def test_loss_weights_answer():
    lw = ThinkingLossWeights(think_weight=0.5, answer_weight=1.0)
    ids = _make_token_ids()
    w = lw.compute_weights(ids)
    # index 0 (before think) and index 4 (after think) are answer tokens
    assert w[0].item() == pytest.approx(1.0)
    assert w[4].item() == pytest.approx(1.0)


def test_loss_weights_structure_tokens():
    lw = ThinkingLossWeights(think_weight=0.5, answer_weight=1.0)
    ids = _make_token_ids()
    w = lw.compute_weights(ids)
    # THINK_START (index 1) and THINK_END (index 3) should be 0.0
    assert w[1].item() == pytest.approx(0.0)
    assert w[3].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ThinkingSFTDataset
# ---------------------------------------------------------------------------


def test_create_example_shapes():
    ds = ThinkingSFTDataset()
    prompt_ids = [10, 11, 12]
    thought_ids = [20, 21]
    response_ids = [30, 31, 32]
    ex = ds.create_example(prompt_ids, thought_ids, response_ids)

    assert "input_ids" in ex and "labels" in ex and "weights" in ex
    L = len(ex["input_ids"])
    assert len(ex["labels"]) == L
    assert len(ex["weights"]) == L


def test_create_example_prompt_masked():
    ds = ThinkingSFTDataset()
    prompt_ids = [10, 11, 12]
    thought_ids = [20]
    response_ids = [30]
    ex = ds.create_example(prompt_ids, thought_ids, response_ids)

    labels = ex["labels"].tolist()
    # First len(prompt_ids) positions must be -100
    for i in range(len(prompt_ids)):
        assert labels[i] == -100, f"label at position {i} should be -100, got {labels[i]}"


def test_thinking_loss_weighted():
    """With think_weight=0.0, thinking tokens contribute 0 to the loss."""
    ds_no_think = ThinkingSFTDataset(think_weight=0.0, answer_weight=1.0)
    ds_with_think = ThinkingSFTDataset(think_weight=1.0, answer_weight=1.0)

    prompt_ids = [10]
    thought_ids = [20, 21, 22]
    response_ids = [30]

    ex = ds_no_think.create_example(prompt_ids, thought_ids, response_ids)
    L = len(ex["input_ids"])
    vocab = 200010  # must exceed special token IDs (200001, 200002)

    torch.manual_seed(0)
    logits = torch.randn(1, L, vocab)
    labels = ex["labels"].unsqueeze(0)
    weights_no_think = ex["weights"].unsqueeze(0)

    ex2 = ds_with_think.create_example(prompt_ids, thought_ids, response_ids)
    weights_with_think = ex2["weights"].unsqueeze(0)

    loss_no = ds_no_think.thinking_loss(logits, labels, weights_no_think)
    loss_with = ds_with_think.thinking_loss(logits, labels, weights_with_think)

    # When think_weight=0 the thinking tokens are excluded; losses should differ
    assert loss_no.item() != pytest.approx(loss_with.item(), rel=1e-3)


# ---------------------------------------------------------------------------
# ThinkingInferenceHelper
# ---------------------------------------------------------------------------


def test_strip_thinking():
    helper = ThinkingInferenceHelper()
    result = helper.strip_thinking(f"{THINK_START}internal{THINK_END}answer")
    assert result == "answer"


def test_should_stop_thinking():
    helper = ThinkingInferenceHelper()
    assert not helper.should_stop_thinking([1, 2, 3])
    assert helper.should_stop_thinking([1, THINK_END_TOKEN_ID, 3])
