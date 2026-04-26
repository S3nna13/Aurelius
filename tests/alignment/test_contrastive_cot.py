"""Tests for contrastive chain-of-thought (CCoT) alignment module."""

from __future__ import annotations

import copy

import pytest
import torch

from src.alignment.contrastive_cot import (
    CCoTConfig,
    CCoTExample,
    CCoTTrainer,
    CoTQualityScorer,
    compute_ccot_loss,
    compute_sequence_log_prob,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
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
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


@pytest.fixture(scope="module")
def ref_model(small_model):
    ref = copy.deepcopy(small_model)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _tokenize(s: str, max_len: int = 64) -> list[int]:
    return [ord(c) % 256 for c in s[:max_len]]


def _detokenize(tokens: list[int]) -> str:
    return "".join(chr(max(32, t % 128)) for t in tokens)


@pytest.fixture(scope="module")
def ccot_cfg():
    return CCoTConfig()


@pytest.fixture(scope="module")
def example():
    return CCoTExample(
        question="What is 2 + 2?",
        correct_cot="We add 2 and 2 to get 4.",
        correct_answer="4",
        wrong_cot="We subtract 2 from 2 to get 0.",
        wrong_answer="0",
    )


@pytest.fixture(scope="module")
def examples(example):
    return [
        example,
        CCoTExample(
            question="What color is the sky?",
            correct_cot="The sky appears blue due to Rayleigh scattering.",
            correct_answer="Blue",
            wrong_cot="The sky is green because of plants.",
            wrong_answer="Green",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. CCoTConfig defaults
# ---------------------------------------------------------------------------


def test_ccot_config_defaults():
    cfg = CCoTConfig()
    assert cfg.beta == 0.1
    assert cfg.margin == 0.5
    assert cfg.max_seq_len == 512
    assert cfg.cot_weight == 1.0
    assert cfg.use_sft_loss is True


# ---------------------------------------------------------------------------
# 2. CCoTExample fields
# ---------------------------------------------------------------------------


def test_ccot_example_fields(example):
    assert hasattr(example, "question")
    assert hasattr(example, "correct_cot")
    assert hasattr(example, "correct_answer")
    assert hasattr(example, "wrong_cot")
    assert hasattr(example, "wrong_answer")
    assert isinstance(example.question, str)
    assert isinstance(example.correct_cot, str)
    assert isinstance(example.correct_answer, str)
    assert isinstance(example.wrong_cot, str)
    assert isinstance(example.wrong_answer, str)


# ---------------------------------------------------------------------------
# 3. compute_sequence_log_prob shape (B,)
# ---------------------------------------------------------------------------


def test_compute_sequence_log_prob_shape(small_model):
    torch.manual_seed(1)
    B, T = 1, 16
    input_ids = torch.randint(0, 256, (B, T))
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, T // 2 :] = True

    lp = compute_sequence_log_prob(small_model, input_ids, mask)
    assert lp.shape == (B,), f"Expected shape ({B},), got {lp.shape}"


# ---------------------------------------------------------------------------
# 4. compute_sequence_log_prob all-masked -> returns 0 or handles gracefully
# ---------------------------------------------------------------------------


def test_compute_sequence_log_prob_all_zero_mask(small_model):
    torch.manual_seed(2)
    B, T = 1, 16
    input_ids = torch.randint(0, 256, (B, T))
    mask = torch.zeros(B, T, dtype=torch.bool)  # all False

    lp = compute_sequence_log_prob(small_model, input_ids, mask)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all(), "Output must be finite even with zero mask"


# ---------------------------------------------------------------------------
# 5. compute_sequence_log_prob values <= 0 (log probs)
# ---------------------------------------------------------------------------


def test_compute_sequence_log_prob_values_nonpositive(small_model):
    torch.manual_seed(3)
    B, T = 1, 16
    input_ids = torch.randint(0, 256, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)

    lp = compute_sequence_log_prob(small_model, input_ids, mask)
    # Log probs should be <= 0
    assert (lp <= 0).all(), f"Log probs should be <= 0, got {lp}"


# ---------------------------------------------------------------------------
# 6. compute_ccot_loss returns (Tensor, dict)
# ---------------------------------------------------------------------------


def test_compute_ccot_loss_return_types(small_model, ref_model, examples, ccot_cfg):
    loss, metrics = compute_ccot_loss(small_model, ref_model, examples, _tokenize, ccot_cfg)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# 7. compute_ccot_loss loss is finite
# ---------------------------------------------------------------------------


def test_compute_ccot_loss_finite(small_model, ref_model, examples, ccot_cfg):
    loss, _ = compute_ccot_loss(small_model, ref_model, examples, _tokenize, ccot_cfg)
    assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"


# ---------------------------------------------------------------------------
# 8. compute_ccot_loss dict has required keys
# ---------------------------------------------------------------------------


def test_compute_ccot_loss_dict_keys(small_model, ref_model, examples, ccot_cfg):
    _, metrics = compute_ccot_loss(small_model, ref_model, examples, _tokenize, ccot_cfg)
    required_keys = {"contrastive_loss", "sft_loss", "margin_achieved", "accuracy"}
    assert required_keys.issubset(metrics.keys()), f"Missing keys: {required_keys - metrics.keys()}"


# ---------------------------------------------------------------------------
# 9. compute_ccot_loss accuracy in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_ccot_loss_accuracy_range(small_model, ref_model, examples, ccot_cfg):
    _, metrics = compute_ccot_loss(small_model, ref_model, examples, _tokenize, ccot_cfg)
    acc = metrics["accuracy"]
    assert 0.0 <= acc <= 1.0, f"Accuracy must be in [0, 1], got {acc}"


# ---------------------------------------------------------------------------
# 10. CoTQualityScorer.score_cot returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_scorer_score_cot_range(small_model):
    scorer = CoTQualityScorer(small_model, _tokenize, _detokenize)
    score = scorer.score_cot("What is 2+2?", "We add them.", "4")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"Score must be in [0, 1], got {score}"


# ---------------------------------------------------------------------------
# 11. CoTQualityScorer.rank_cots returns sorted list
# ---------------------------------------------------------------------------


def test_scorer_rank_cots_sorted(small_model):
    scorer = CoTQualityScorer(small_model, _tokenize, _detokenize)
    cots = [
        ("Step 1: add. Step 2: done.", "4"),
        ("Just guess.", "maybe 5"),
        ("Two plus two equals four.", "4"),
    ]
    ranked = scorer.rank_cots("What is 2+2?", cots)
    scores = [r[0] for r in ranked]
    assert scores == sorted(scores, reverse=True), "Ranked list must be sorted descending"


# ---------------------------------------------------------------------------
# 12. CoTQualityScorer.rank_cots length matches input
# ---------------------------------------------------------------------------


def test_scorer_rank_cots_length(small_model):
    scorer = CoTQualityScorer(small_model, _tokenize, _detokenize)
    cots = [
        ("reasoning A", "answer A"),
        ("reasoning B", "answer B"),
        ("reasoning C", "answer C"),
        ("reasoning D", "answer D"),
    ]
    ranked = scorer.rank_cots("Question?", cots)
    assert len(ranked) == len(cots), f"Expected {len(cots)} results, got {len(ranked)}"


# ---------------------------------------------------------------------------
# 13. CCoTTrainer.train_step returns dict with "loss" key
# ---------------------------------------------------------------------------


def test_trainer_train_step_loss_key(small_model, ref_model, examples, ccot_cfg):
    model_copy = copy.deepcopy(small_model)
    ref_copy = copy.deepcopy(small_model)
    for p in ref_copy.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model_copy.parameters(), lr=1e-4)
    trainer = CCoTTrainer(
        model=model_copy,
        ref_model=ref_copy,
        config=ccot_cfg,
        optimizer=optimizer,
        tokenizer_encode=_tokenize,
        tokenizer_decode=_detokenize,
    )

    metrics = trainer.train_step(examples)
    assert "loss" in metrics, f"train_step must return dict with 'loss' key, got {metrics.keys()}"
    assert isinstance(metrics["loss"], float)
    assert torch.isfinite(torch.tensor(metrics["loss"]))


# ---------------------------------------------------------------------------
# 14. CCoTTrainer.evaluate returns dict with "accuracy" key
# ---------------------------------------------------------------------------


def test_trainer_evaluate_accuracy_key(small_model, ref_model, examples, ccot_cfg):
    model_copy = copy.deepcopy(small_model)
    ref_copy = copy.deepcopy(small_model)
    for p in ref_copy.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model_copy.parameters(), lr=1e-4)
    trainer = CCoTTrainer(
        model=model_copy,
        ref_model=ref_copy,
        config=ccot_cfg,
        optimizer=optimizer,
        tokenizer_encode=_tokenize,
        tokenizer_decode=_detokenize,
    )

    metrics = trainer.evaluate(examples)
    assert "accuracy" in metrics, (
        f"evaluate must return dict with 'accuracy' key, got {metrics.keys()}"
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# 15. CoTQualityScorer.generate_contrastive_pairs returns CCoTExample list
# ---------------------------------------------------------------------------


def test_scorer_generate_contrastive_pairs(small_model):
    scorer = CoTQualityScorer(small_model, _tokenize, _detokenize)
    pairs = scorer.generate_contrastive_pairs("What is 1+1?", n_pairs=2)
    assert isinstance(pairs, list)
    assert len(pairs) > 0, "Should return at least one contrastive pair"
    for pair in pairs:
        assert isinstance(pair, CCoTExample)
        assert isinstance(pair.question, str)
        assert isinstance(pair.correct_cot, str)
        assert isinstance(pair.wrong_cot, str)
