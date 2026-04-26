"""Tests for combined data/curriculum mix helpers."""

import pytest

from src.training.curriculum_sampling import DifficultyBucket
from src.training.data_curriculum_mix import CurriculumSource, combined_mix_weights, dominant_source
from src.training.pretraining_mix import CorpusSource


def make_sources():
    return [
        CurriculumSource(
            CorpusSource("web", weight=3.0), DifficultyBucket("easy", difficulty=0.1, weight=1.0)
        ),
        CurriculumSource(
            CorpusSource("math", weight=1.0), DifficultyBucket("hard", difficulty=1.0, weight=1.0)
        ),
    ]


def test_combined_mix_weights_sum_to_one():
    weights = combined_mix_weights(make_sources(), step=5, total_steps=10)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_combined_mix_weights_shift_with_curriculum():
    early = combined_mix_weights(make_sources(), step=0, total_steps=10)
    late = combined_mix_weights(make_sources(), step=10, total_steps=10)
    assert late["math"] > early["math"]


def test_dominant_source_returns_max_weight_source():
    assert dominant_source(make_sources(), step=0, total_steps=10) == "web"


def test_dominant_source_can_switch_late_in_training():
    assert dominant_source(make_sources(), step=10, total_steps=10) in {"web", "math"}


def test_combined_mix_weights_handles_empty_input():
    assert combined_mix_weights([], step=0, total_steps=10) == {}


def test_dominant_source_rejects_empty_input():
    with pytest.raises(ValueError):
        dominant_source([], step=0, total_steps=10)


def test_combined_mix_weights_respects_total_steps_validation():
    with pytest.raises(ValueError):
        combined_mix_weights(make_sources(), step=0, total_steps=0)
