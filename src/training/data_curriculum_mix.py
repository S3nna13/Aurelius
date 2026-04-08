"""Combine source mixing with curriculum difficulty weighting."""

from __future__ import annotations

from dataclasses import dataclass

from src.training.curriculum_sampling import DifficultyBucket, curriculum_weights
from src.training.pretraining_mix import CorpusSource, source_probabilities


@dataclass(frozen=True)
class CurriculumSource:
    corpus: CorpusSource
    bucket: DifficultyBucket


def combined_mix_weights(
    sources: list[CurriculumSource],
    step: int,
    total_steps: int,
) -> dict[str, float]:
    """Blend corpus-level and curriculum-level probabilities."""
    if not sources:
        return {}
    corpus_probs = source_probabilities([source.corpus for source in sources])
    bucket_probs = curriculum_weights([source.bucket for source in sources], step, total_steps)
    raw = {}
    for source in sources:
        raw[source.corpus.name] = corpus_probs[source.corpus.name] * bucket_probs[source.bucket.name]
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def dominant_source(
    sources: list[CurriculumSource],
    step: int,
    total_steps: int,
) -> str:
    """Return the source with the highest combined weight."""
    weights = combined_mix_weights(sources, step, total_steps)
    if not weights:
        raise ValueError("sources must be non-empty")
    return max(weights, key=weights.get)

