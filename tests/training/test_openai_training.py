"""Tests for src/training/openai_training.py — OpenAI-inspired training algorithms.

All tests use synthetic data only; no network requests, no external APIs.
"""

from __future__ import annotations

import pytest
import torch

from src.training.openai_training import (
    # MMMLU-inspired
    aggregate_to_dpo_pairs,
    answer_verification_reward,
    borda_count,
    extract_final_answer,
    extract_graph_answer_nodes,
    format_multiple_choice_prompt,
    # GraphWalks-inspired
    graphwalks_f1,
    majority_vote_preference,
    # CoVal-inspired
    parse_ranking_string,
    split_scratchpad_answer_mask,
    verify_numeric_answer,
    weighted_lm_loss,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

VOCAB_SIZE = 64
D_MODEL = 32
MAX_SEQ_LEN = 32


def _make_tiny_model():
    """Return a trivial nn.Module that produces random logits."""
    import torch.nn as nn

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
            self.proj = nn.Linear(D_MODEL, VOCAB_SIZE)

        def forward(self, input_ids):
            return self.proj(self.embed(input_ids))

        def generate(self, input_ids, max_new_tokens=16):
            # Greedy generation stub
            out = input_ids.clone()
            for _ in range(max_new_tokens):
                logits = self.forward(out)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                out = torch.cat([out, next_tok], dim=1)
            return out

    return TinyLM()


def _dummy_encode(text: str) -> list[int]:
    """Encode text to token ids using ord % VOCAB_SIZE."""
    return [ord(c) % VOCAB_SIZE for c in text] or [0]


def _dummy_decode(ids) -> str:
    """Decode token ids back to characters."""
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return "".join(chr(i % 128) for i in ids)


# ── 1. GSM8K: extract_final_answer ───────────────────────────────────────────


def test_extract_final_answer_basic():
    result = extract_final_answer("Let me think... #### 72")
    assert result == "72"


def test_extract_final_answer_none():
    result = extract_final_answer("No delimiter here at all")
    assert result is None


# ── 2. GSM8K: verify_numeric_answer ──────────────────────────────────────────


def test_verify_numeric_answer_correct():
    assert verify_numeric_answer("72", "72") is True


def test_verify_numeric_answer_comma_format():
    assert verify_numeric_answer("1,234", "1234") is True


def test_verify_numeric_answer_tolerance():
    assert verify_numeric_answer("0.1", "0.10000001") is True


# ── 3. GSM8K: answer_verification_reward ─────────────────────────────────────


def test_answer_verification_reward_correct():
    reward = answer_verification_reward("Step 1... #### 42", "42", answer_reward=1.0)
    assert reward == 1.0


def test_answer_verification_reward_format_only():
    reward = answer_verification_reward(
        "Step 1... #### 99", "42", format_reward=0.1, answer_reward=1.0
    )
    assert reward == pytest.approx(0.1)


# ── 4. GSM8K: split_scratchpad_answer_mask ───────────────────────────────────


def test_split_scratchpad_answer_mask_shape():
    B, T = 3, 20
    delimiter_id = 5
    input_ids = torch.randint(0, 10, (B, T))
    mask = split_scratchpad_answer_mask(input_ids, delimiter_id=delimiter_id)
    assert mask.shape == (B, T)


def test_split_scratchpad_answer_mask_weights():
    B, T = 1, 10
    delimiter_id = 7
    # Place delimiter at position 4
    input_ids = torch.zeros(B, T, dtype=torch.long)
    input_ids[0, 4] = delimiter_id
    mask = split_scratchpad_answer_mask(input_ids, delimiter_id=delimiter_id, answer_weight=5.0)
    # Positions before delimiter (0..3) → 1.0
    assert (mask[0, :4] == 1.0).all(), f"Before delimiter: {mask[0, :4]}"
    # Positions at/after delimiter (4..9) → answer_weight
    assert (mask[0, 4:] == 5.0).all(), f"At/after delimiter: {mask[0, 4:]}"


# ── 5. GSM8K: weighted_lm_loss ───────────────────────────────────────────────


def test_weighted_lm_loss_shape():
    B, T, V = 2, 8, 64
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    weights = torch.ones(B, T)
    loss = weighted_lm_loss(logits, labels, weights)
    assert loss.shape == ()  # scalar


def test_weighted_lm_loss_ignores_minus100():
    B, T, V = 2, 8, 64
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :4] = -100  # mask first half
    weights = torch.ones(B, T)
    # Should not raise and should return a finite scalar
    loss = weighted_lm_loss(logits, labels, weights)
    assert torch.isfinite(loss)


# ── 6. MMMLU: format_multiple_choice_prompt ──────────────────────────────────


def test_format_multiple_choice_prompt_contains_choices():
    prompt = format_multiple_choice_prompt(
        question="What is 2+2?",
        choices={"A": "3", "B": "4", "C": "5", "D": "6"},
    )
    for letter in ["A", "B", "C", "D"]:
        assert letter in prompt
    assert "Answer:" in prompt


# ── 7. CoVal: parse_ranking_string ───────────────────────────────────────────


def test_parse_ranking_string_simple():
    result = parse_ranking_string("B>A>C=D")
    assert result == [["B"], ["A"], ["C", "D"]]


# ── 8. CoVal: borda_count ────────────────────────────────────────────────────


def test_borda_count_clear_winner():
    # A is always ranked first
    rankings = [
        [["A"], ["B"], ["C"]],
        [["A"], ["C"], ["B"]],
    ]
    scores = borda_count(rankings, candidates=["A", "B", "C"])
    assert scores["A"] > scores["B"]
    assert scores["A"] > scores["C"]


# ── 9. CoVal: majority_vote_preference ───────────────────────────────────────


def test_majority_vote_preference_clear():
    # All annotators prefer A over B
    rankings = [
        [["A"], ["B"]],
        [["A"], ["B"]],
        [["A"], ["B"]],
    ]
    winner, confidence = majority_vote_preference(rankings, "A", "B")
    assert winner == "A"
    assert confidence == pytest.approx(1.0)


# ── 10. CoVal: aggregate_to_dpo_pairs ────────────────────────────────────────


def test_aggregate_to_dpo_pairs_returns_list():
    responses = {"A": "great answer", "B": "bad answer", "C": "ok answer", "D": "meh"}
    rankings = [
        [["A"], ["C"], ["D"], ["B"]],
        [["A"], ["D"], ["C"], ["B"]],
    ]
    pairs = aggregate_to_dpo_pairs(responses, rankings, min_confidence=0.5)
    assert isinstance(pairs, list)
    # At least one pair should be produced given clear winner/loser
    assert len(pairs) >= 1


# ── 11. GraphWalks: graphwalks_f1 ────────────────────────────────────────────


def test_graphwalks_f1_perfect():
    assert graphwalks_f1(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_graphwalks_f1_empty_both():
    assert graphwalks_f1([], []) == pytest.approx(1.0)


# ── 12. GraphWalks: extract_graph_answer_nodes ───────────────────────────────


def test_extract_graph_answer_nodes_valid():
    text = "Some reasoning... Final Answer: [node1, node2]"
    nodes = extract_graph_answer_nodes(text)
    assert nodes == ["node1", "node2"]


def test_extract_graph_answer_nodes_empty():
    nodes = extract_graph_answer_nodes("No pattern here")
    assert nodes == []
