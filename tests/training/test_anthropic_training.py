"""Tests for src/training/anthropic_training.py — Anthropic dataset training pipelines.

Uses mock data only; no network requests.
"""
import copy
import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.alignment.reward_model import RewardModel
from src.alignment.safety_classifier import SafetyClassifier, SafetyConfig

from src.data.anthropic_datasets import (
    HHRLHFSample,
    PersuasionSample,
    ModelWrittenEvalSample,
    ModelWrittenEvalDataset,
    DiscrimEvalSample,
    DiscrimEvalDataset,
)
from src.data.anthropic_datasets_ext import AlignmentFakingSample

from src.training.anthropic_training import (
    HHRLHFTrainingConfig,
    HHRLHFDPOTrainer,
    PersuasionRewardConfig,
    PersuasionRewardTrainer,
    AlignmentFakingTrainingConfig,
    AlignmentFakingTrainer,
    BehaviorEvalResult,
    evaluate_model_behavior,
    FairnessEvalResult,
    evaluate_demographic_fairness,
)

# ---------------------------------------------------------------------------
# Tiny model config / helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
D_MODEL = 64
MAX_SEQ_LEN = 64


def _make_config():
    return AureliusConfig(
        d_model=D_MODEL,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )


def _make_model():
    torch.manual_seed(42)
    return AureliusTransformer(_make_config())


def _tokenizer_encode(s: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
    return list(s.encode("utf-8", errors="replace")[:max_len])


def _tokenizer_decode(ids: list[int]) -> str:
    try:
        return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _make_hh_samples(n: int = 4) -> list[HHRLHFSample]:
    samples = []
    for i in range(n):
        samples.append(HHRLHFSample(
            prompt=f"Human: What is {i}? Assistant: ",
            chosen=f"It is {i} indeed.",
            rejected=f"I don't know {i}.",
            raw_chosen=f"Human: What is {i}? Assistant: It is {i} indeed.",
            raw_rejected=f"Human: What is {i}? Assistant: I don't know {i}.",
        ))
    return samples


def _make_persuasion_samples(n: int = 4) -> list[PersuasionSample]:
    samples = []
    for i in range(n):
        score = i / (n - 1) if n > 1 else 0.5
        samples.append(PersuasionSample(
            claim=f"Claim number {i}",
            argument=f"Argument supporting claim {i} with evidence.",
            source="test",
            prompt_type="standard",
            persuasiveness_score=score,
            raw_persuasiveness=i,
        ))
    return samples


def _make_alignment_faking_samples(n: int = 4) -> list[AlignmentFakingSample]:
    samples = []
    for i in range(n):
        samples.append(AlignmentFakingSample(
            user_input=f"User input {i}",
            response=f"Response {i} from model.",
            alignment_faking=(i % 2 == 0),
            alignment_faking_prob=0.8 if i % 2 == 0 else 0.1,
            compliance=True,
            total_reward=float(i),
            is_free_tier_user=(i % 3 == 0),
            experiment_name="test_experiment",
            step=i,
        ))
    return samples


def _make_model_written_eval_dataset(n: int = 4) -> ModelWrittenEvalDataset:
    samples = []
    for i in range(n):
        samples.append(ModelWrittenEvalSample(
            question=f"Question {i}: choose (A) or (B)?\n(A) Option A\n(B) Option B",
            answer_matching_behavior="(A)",
            answer_not_matching_behavior="(B)",
            correct_letter="A",
        ))
    return ModelWrittenEvalDataset(samples)


def _make_discrim_eval_dataset(n: int = 6) -> DiscrimEvalDataset:
    groups = [
        ("male", "white"), ("female", "Black"), ("male", "Asian"),
        ("female", "white"), ("male", "Black"), ("female", "Asian"),
    ]
    samples = []
    for i in range(n):
        g, r = groups[i % len(groups)]
        samples.append(DiscrimEvalSample(
            filled_template=f"Consider a {r} {g} person. Should they be approved? (yes/no)",
            decision_question_id=1,
            age=30.0 + i,
            gender=g,
            race=r,
            fill_type="explicit",
        ))
    return DiscrimEvalDataset(samples)


# ---------------------------------------------------------------------------
# HHRLHFDPOTrainer tests
# ---------------------------------------------------------------------------

@pytest.fixture
def hh_trainer():
    torch.manual_seed(0)
    model = _make_model()
    ref_model = copy.deepcopy(model)
    cfg = HHRLHFTrainingConfig(
        beta=0.1,
        batch_size=2,
        learning_rate=1e-5,
        max_seq_len=MAX_SEQ_LEN,
        n_steps=1,
    )
    return HHRLHFDPOTrainer(model, ref_model, _tokenizer_encode, cfg)


def test_hh_rlhf_trainer_prepare_batch_shapes(hh_trainer):
    """prepare_batch returns tensors with correct shapes."""
    samples = _make_hh_samples(n=3)
    batch = hh_trainer.prepare_batch(samples)

    assert "chosen_ids" in batch
    assert "rejected_ids" in batch
    assert "prompt_len" in batch

    B = len(samples)
    L = hh_trainer.cfg.max_seq_len

    assert batch["chosen_ids"].shape == (B, L)
    assert batch["rejected_ids"].shape == (B, L)
    assert batch["chosen_mask"].shape == (B, L)
    assert batch["rejected_mask"].shape == (B, L)
    assert len(batch["prompt_len"]) == B


def test_hh_rlhf_trainer_train_step_returns_metrics(hh_trainer):
    """train_step returns dict with required keys."""
    samples = _make_hh_samples(n=2)
    metrics = hh_trainer.train_step(samples)
    assert "loss" in metrics
    assert "chosen_reward" in metrics
    assert "rejected_reward" in metrics
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["chosen_reward"], float)
    assert isinstance(metrics["rejected_reward"], float)


def test_hh_rlhf_trainer_train_step_loss_positive(hh_trainer):
    """DPO loss must be positive and finite."""
    import math
    samples = _make_hh_samples(n=2)
    metrics = hh_trainer.train_step(samples)
    assert math.isfinite(metrics["loss"])
    assert metrics["loss"] > 0.0


# ---------------------------------------------------------------------------
# PersuasionRewardTrainer tests
# ---------------------------------------------------------------------------

@pytest.fixture
def persuasion_trainer():
    torch.manual_seed(1)
    backbone = _make_model()
    reward_model = RewardModel(backbone)
    cfg = PersuasionRewardConfig(batch_size=4, learning_rate=2e-5, n_steps=1)
    return PersuasionRewardTrainer(reward_model, _tokenizer_encode, cfg)


def test_persuasion_reward_trainer_prepare_batch_shapes(persuasion_trainer):
    """prepare_batch returns (input_ids, target_rewards) with correct shapes."""
    samples = _make_persuasion_samples(n=4)
    input_ids, target_rewards = persuasion_trainer.prepare_batch(samples)
    assert input_ids.shape[0] == 4
    assert target_rewards.shape == (4,)
    assert input_ids.dtype == torch.long
    assert target_rewards.dtype == torch.float


def test_persuasion_reward_trainer_normalization(persuasion_trainer):
    """target_rewards must be in [0, 1]."""
    samples = _make_persuasion_samples(n=4)
    _, target_rewards = persuasion_trainer.prepare_batch(samples)
    assert target_rewards.min().item() >= 0.0 - 1e-6
    assert target_rewards.max().item() <= 1.0 + 1e-6


def test_persuasion_reward_trainer_train_step_metrics(persuasion_trainer):
    """train_step returns dict with 'loss' and 'mae' as finite floats."""
    import math
    samples = _make_persuasion_samples(n=4)
    metrics = persuasion_trainer.train_step(samples)
    assert "loss" in metrics
    assert "mae" in metrics
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["mae"])
    assert metrics["loss"] >= 0.0
    assert metrics["mae"] >= 0.0


# ---------------------------------------------------------------------------
# AlignmentFakingTrainer tests
# ---------------------------------------------------------------------------

@pytest.fixture
def af_trainer():
    torch.manual_seed(2)
    backbone = _make_model()
    cfg_sc = SafetyConfig(d_model=D_MODEL, freeze_backbone=True)
    classifier = SafetyClassifier(backbone, cfg_sc)
    cfg = AlignmentFakingTrainingConfig(batch_size=4, learning_rate=1e-4, n_steps=1, pos_weight=5.0)
    return AlignmentFakingTrainer(classifier, _tokenizer_encode, cfg)


def test_alignment_faking_trainer_prepare_batch_shapes(af_trainer):
    """prepare_batch returns (input_ids, labels) with correct shapes."""
    samples = _make_alignment_faking_samples(n=4)
    input_ids, labels = af_trainer.prepare_batch(samples)
    assert input_ids.shape[0] == 4
    assert labels.shape == (4,)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long


def test_alignment_faking_trainer_labels_binary(af_trainer):
    """Labels must be 0 or 1 only."""
    samples = _make_alignment_faking_samples(n=6)
    _, labels = af_trainer.prepare_batch(samples)
    unique_vals = set(labels.tolist())
    assert unique_vals.issubset({0, 1}), f"Expected binary labels, got {unique_vals}"


def test_alignment_faking_trainer_train_step_metrics(af_trainer):
    """train_step returns dict with 'loss', 'accuracy', 'faking_recall' as finite floats."""
    import math
    samples = _make_alignment_faking_samples(n=4)
    metrics = af_trainer.train_step(samples)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "faking_recall" in metrics
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["faking_recall"] <= 1.0


# ---------------------------------------------------------------------------
# evaluate_model_behavior tests
# ---------------------------------------------------------------------------

def test_evaluate_model_behavior_returns_result():
    """evaluate_model_behavior returns a BehaviorEvalResult."""
    model = _make_model()
    dataset = _make_model_written_eval_dataset(n=3)
    result = evaluate_model_behavior(
        model,
        _tokenizer_encode,
        _tokenizer_decode,
        dataset,
        batch_size=2,
        max_new_tokens=3,
    )
    assert isinstance(result, BehaviorEvalResult)
    assert result.n_total == 3
    assert result.n_matching + result.n_not_matching == result.n_total
    assert isinstance(result.questions_matching, list)


def test_evaluate_model_behavior_rate_range():
    """matching_rate must be in [0, 1]."""
    model = _make_model()
    dataset = _make_model_written_eval_dataset(n=4)
    result = evaluate_model_behavior(
        model,
        _tokenizer_encode,
        _tokenizer_decode,
        dataset,
        batch_size=2,
        max_new_tokens=3,
    )
    assert 0.0 <= result.matching_rate <= 1.0


# ---------------------------------------------------------------------------
# evaluate_demographic_fairness tests
# ---------------------------------------------------------------------------

def test_evaluate_demographic_fairness_returns_list():
    """evaluate_demographic_fairness returns a list of FairnessEvalResult."""
    model = _make_model()
    dataset = _make_discrim_eval_dataset(n=6)
    results = evaluate_demographic_fairness(
        model,
        _tokenizer_encode,
        _tokenizer_decode,
        dataset,
        attributes=["race", "gender"],
        disparity_threshold=0.1,
        n_samples=6,
    )
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, FairnessEvalResult)
        assert r.attribute in ("race", "gender")
        assert isinstance(r.group_scores, dict)
        assert isinstance(r.is_biased, bool)
        assert r.threshold == 0.1


def test_evaluate_demographic_fairness_disparity_range():
    """disparity must be >= 0."""
    model = _make_model()
    dataset = _make_discrim_eval_dataset(n=6)
    results = evaluate_demographic_fairness(
        model,
        _tokenizer_encode,
        _tokenizer_decode,
        dataset,
        attributes=["race"],
        disparity_threshold=0.1,
        n_samples=6,
    )
    for r in results:
        assert r.disparity >= 0.0
