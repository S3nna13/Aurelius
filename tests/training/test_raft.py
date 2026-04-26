"""Tests for RAFT: Retrieval-Augmented Fine-Tuning."""

from __future__ import annotations

import random

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.raft import (
    RAFTConfig,
    RAFTDataCollator,
    RAFTExample,
    RAFTTrainer,
    format_raft_prompt,
    format_raft_target,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg() -> AureliusConfig:
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
def small_model(small_cfg: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


def _tokenize(text: str) -> list[int]:
    """Byte-level tokenizer — always within vocab_size=256."""
    return list(text.encode("utf-8", errors="replace"))


def _make_example(
    question: str = "Sky color?",
    oracle_doc: str = "Sky is blue.",
    distractor_docs: list[str] | None = None,
    answer: str = "Blue",
    chain_of_thought: str = "",
) -> RAFTExample:
    if distractor_docs is None:
        distractor_docs = [
            "Grass green.",
            "Roses red.",
            "Snow white.",
        ]
    return RAFTExample(
        question=question,
        oracle_doc=oracle_doc,
        distractor_docs=distractor_docs,
        answer=answer,
        chain_of_thought=chain_of_thought,
    )


def _make_raft_config(n_distractors: int = 2) -> RAFTConfig:
    return RAFTConfig(
        p_oracle=0.8,
        n_distractors=n_distractors,
        max_seq_len=128,
        cot_training=True,
        loss_only_on_answer=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_raft_config_defaults():
    cfg = RAFTConfig()
    assert cfg.p_oracle == 0.8
    assert cfg.n_distractors == 3
    assert cfg.max_seq_len == 512
    assert cfg.cot_training is True
    assert cfg.loss_only_on_answer is True


def test_raft_example_creation():
    ex = _make_example()
    assert ex.question == "Sky color?"
    assert ex.oracle_doc == "Sky is blue."
    assert len(ex.distractor_docs) == 3
    assert ex.answer == "Blue"
    assert ex.chain_of_thought == ""


def test_format_raft_prompt_with_oracle():
    """Oracle doc must appear in prompt when include_oracle=True."""
    ex = _make_example()
    rng = random.Random(7)
    prompt = format_raft_prompt(ex, include_oracle=True, n_distractors=2, rng=rng)
    assert ex.oracle_doc in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt


def test_format_raft_prompt_without_oracle():
    """Prompt without oracle should be shorter than prompt with oracle."""
    ex = _make_example()
    rng_with = random.Random(99)
    rng_without = random.Random(99)
    prompt_with = format_raft_prompt(ex, include_oracle=True, n_distractors=2, rng=rng_with)
    prompt_without = format_raft_prompt(ex, include_oracle=False, n_distractors=2, rng=rng_without)
    assert ex.oracle_doc not in prompt_without
    assert len(prompt_with) > len(prompt_without)


def test_format_raft_target_cot():
    """Target should include chain_of_thought when cot=True."""
    ex = _make_example(chain_of_thought="The sky contains Rayleigh-scattered light.")
    target = format_raft_target(ex, cot=True)
    assert ex.chain_of_thought in target
    assert ex.answer in target


def test_format_raft_target_no_cot():
    """Target should be just the answer when cot=False or no chain_of_thought."""
    ex = _make_example(chain_of_thought="Some reasoning here.")
    target_no_cot = format_raft_target(ex, cot=False)
    assert target_no_cot == ex.answer
    assert ex.chain_of_thought not in target_no_cot

    ex_no_cot = _make_example(chain_of_thought="")
    target_empty_cot = format_raft_target(ex_no_cot, cot=True)
    assert target_empty_cot == ex_no_cot.answer


def test_raft_data_collator_output_count():
    """Collator should return one (input_ids, labels) pair per example."""
    cfg = _make_raft_config()
    collator = RAFTDataCollator(cfg, _tokenize)
    examples = [_make_example() for _ in range(4)]
    pairs = collator.collate(examples, seed=0)
    assert len(pairs) == 4
    for input_ids, labels in pairs:
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert input_ids.shape == labels.shape


def test_raft_data_collator_label_masking():
    """With loss_only_on_answer=True, some labels must be -100 (the prompt portion)."""
    cfg = _make_raft_config()
    cfg.loss_only_on_answer = True
    collator = RAFTDataCollator(cfg, _tokenize)
    examples = [_make_example()]
    pairs = collator.collate(examples, seed=0)
    _, labels = pairs[0]
    assert (labels == -100).any(), "Some labels should be masked (-100) for prompt tokens"


def test_raft_trainer_step_keys(small_model):
    """train_step must return dict with loss, n_examples, oracle_rate."""
    cfg = _make_raft_config()
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
    trainer = RAFTTrainer(small_model, optimizer, cfg, _tokenize)
    examples = [_make_example() for _ in range(3)]
    result = trainer.train_step(examples)
    assert "loss" in result
    assert "n_examples" in result
    assert "oracle_rate" in result


def test_raft_trainer_loss_positive(small_model):
    """Loss returned by train_step must be a positive finite float."""
    cfg = _make_raft_config()
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
    trainer = RAFTTrainer(small_model, optimizer, cfg, _tokenize)
    examples = [_make_example() for _ in range(3)]
    result = trainer.train_step(examples)
    assert isinstance(result["loss"], float)
    assert result["loss"] > 0
    assert result["loss"] == result["loss"]  # not NaN


def test_format_raft_prompt_n_distractors():
    """Prompt should contain the correct number of document entries."""
    ex = _make_example(
        distractor_docs=[
            "Doc A.",
            "Doc B.",
            "Doc C.",
        ]
    )
    rng = random.Random(42)
    # With oracle + 2 distractors => 3 docs total
    prompt = format_raft_prompt(ex, include_oracle=True, n_distractors=2, rng=rng)
    # Count document markers "[" that start a doc entry
    doc_count = prompt.count("[")
    assert doc_count == 3, f"Expected 3 docs (1 oracle + 2 distractors), got {doc_count}"

    rng2 = random.Random(42)
    # Without oracle + 2 distractors => 2 docs total
    prompt2 = format_raft_prompt(ex, include_oracle=False, n_distractors=2, rng=rng2)
    doc_count2 = prompt2.count("[")
    assert doc_count2 == 2, f"Expected 2 docs (0 oracle + 2 distractors), got {doc_count2}"
