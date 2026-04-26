"""Tests for src/data/dataset_mixer.py — advanced dataset mixing and curriculum."""

from __future__ import annotations

import statistics

from aurelius.data.dataset_mixer import (
    AdaptiveMixer,
    CurriculumScheduler,
    DatasetMixer,
    DataSource,
    MixerConfig,
    compute_domain_weights,
    compute_proportional_weights,
    normalize_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_source(name: str, n: int = 15, weight: float = 1.0, domain: str = "general") -> DataSource:
    """Create a DataSource with n simple example dicts."""
    data = [{"input_ids": list(range(i, i + 5)), "labels": [i] * 5, "id": i} for i in range(n)]
    return DataSource(name=name, data=data, weight=weight, domain=domain)


def make_sources() -> list[DataSource]:
    return [
        make_source("src_a", n=10, weight=2.0, domain="code"),
        make_source("src_b", n=20, weight=1.0, domain="text"),
        make_source("src_c", n=15, weight=1.0, domain="code"),
    ]


# ---------------------------------------------------------------------------
# 1. normalize_weights sums to 1.0
# ---------------------------------------------------------------------------


def test_normalize_weights_sums_to_one():
    weights = [1.0, 2.0, 3.0]
    result = normalize_weights(weights)
    assert abs(sum(result) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. normalize_weights high temperature → more uniform (lower std)
# ---------------------------------------------------------------------------


def test_normalize_weights_temperature_effect():
    weights = [0.1, 1.0, 5.0]
    high_temp = normalize_weights(weights, temperature=5.0)
    low_temp = normalize_weights(weights, temperature=0.5)
    assert statistics.stdev(high_temp) < statistics.stdev(low_temp)


# ---------------------------------------------------------------------------
# 3. compute_proportional_weights: larger source gets higher weight
# ---------------------------------------------------------------------------


def test_proportional_weights_larger_gets_more():
    sizes = [100, 1000, 500]
    weights = compute_proportional_weights(sizes, alpha=0.7)
    # index 1 (size=1000) should have the highest weight
    assert weights[1] == max(weights)
    assert abs(sum(weights) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 4. compute_proportional_weights alpha=0 → uniform weights
# ---------------------------------------------------------------------------


def test_proportional_weights_alpha_zero_uniform():
    sizes = [50, 200, 900]
    weights = compute_proportional_weights(sizes, alpha=0.0)
    # All sizes^0 == 1, so weights should be equal
    expected = 1.0 / len(sizes)
    for w in weights:
        assert abs(w - expected) < 1e-9


# ---------------------------------------------------------------------------
# 5. compute_domain_weights: sources in same domain share domain weight
# ---------------------------------------------------------------------------


def test_domain_weights_shared_equally():
    sources = [
        make_source("a", domain="code"),
        make_source("b", domain="code"),
        make_source("c", domain="text"),
    ]
    domain_config = {"code": 0.6, "text": 0.4}
    weights = compute_domain_weights(sources, domain_config)

    # Both code sources share 0.6 → each gets 0.3 (before normalization)
    # text source gets 0.4
    # Since they already sum to 1.0, normalization is identity
    assert abs(weights[0] - weights[1]) < 1e-9  # same domain share
    assert weights[2] > weights[0]  # text has larger share
    assert abs(sum(weights) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 6. CurriculumScheduler.get_weights returns list of length == n_sources
# ---------------------------------------------------------------------------


def test_curriculum_scheduler_weights_length():
    sources = make_sources()
    scheduler = CurriculumScheduler(sources, easy_domains=["code"], n_steps=500)
    weights = scheduler.get_weights(step=0)
    assert len(weights) == len(sources)
    assert abs(sum(weights) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 7. CurriculumScheduler.is_warmed_up: False before warmup, True after
# ---------------------------------------------------------------------------


def test_curriculum_scheduler_warmup():
    sources = make_sources()
    scheduler = CurriculumScheduler(
        sources, easy_domains=["code"], n_steps=1000, warmup_fraction=0.3
    )
    warmup_steps = int(1000 * 0.3)  # 300

    assert not scheduler.is_warmed_up(0)
    assert not scheduler.is_warmed_up(warmup_steps - 1)
    assert scheduler.is_warmed_up(warmup_steps)
    assert scheduler.is_warmed_up(warmup_steps + 1)
    assert scheduler.is_warmed_up(1000)


# ---------------------------------------------------------------------------
# 8. DatasetMixer.sample_batch returns (examples, source_names) with correct length
# ---------------------------------------------------------------------------


def test_sample_batch_length():
    sources = make_sources()
    mixer = DatasetMixer(sources, MixerConfig(seed=0))
    examples, names = mixer.sample_batch(batch_size=16, step=0)
    assert len(examples) == 16
    assert len(names) == 16


# ---------------------------------------------------------------------------
# 9. source_names all in source name set
# ---------------------------------------------------------------------------


def test_sample_batch_source_names_valid():
    sources = make_sources()
    mixer = DatasetMixer(sources, MixerConfig(seed=1))
    valid_names = {src.name for src in sources}
    _, names = mixer.sample_batch(batch_size=50, step=0)
    for name in names:
        assert name in valid_names


# ---------------------------------------------------------------------------
# 10. DatasetMixer iterator yields dict examples
# ---------------------------------------------------------------------------


def test_iterator_yields_dicts():
    sources = make_sources()
    mixer = DatasetMixer(sources, MixerConfig(seed=2))
    it = iter(mixer)
    for _ in range(20):
        example = next(it)
        assert isinstance(example, dict)
        assert "input_ids" in example
        assert "labels" in example


# ---------------------------------------------------------------------------
# 11. DatasetMixer over many samples: each source sampled proportional to weight
# ---------------------------------------------------------------------------


def test_sampling_proportional_to_weight():
    sources = [
        make_source("heavy", n=15, weight=8.0),
        make_source("light", n=15, weight=2.0),
    ]
    config = MixerConfig(strategy="weighted", temperature=1.0, min_weight=0.0, seed=99)
    mixer = DatasetMixer(sources, config)

    n_samples = 5000
    examples, names = mixer.sample_batch(batch_size=n_samples, step=0)
    heavy_frac = names.count("heavy") / n_samples
    # Expected ~0.8; allow generous tolerance
    assert 0.70 < heavy_frac < 0.90


# ---------------------------------------------------------------------------
# 12. DatasetMixer.update_weights changes sampling distribution
# ---------------------------------------------------------------------------


def test_update_weights_changes_distribution():
    sources = [
        make_source("alpha", n=15, weight=1.0),
        make_source("beta", n=15, weight=1.0),
    ]
    mixer = DatasetMixer(sources, MixerConfig(strategy="weighted", seed=7))

    _, names_before = mixer.sample_batch(batch_size=500, step=0)
    alpha_before = names_before.count("alpha") / 500

    # Strongly upweight alpha
    mixer.update_weights("alpha", 9.0)
    _, names_after = mixer.sample_batch(batch_size=500, step=0)
    alpha_after = names_after.count("alpha") / 500

    assert alpha_after > alpha_before


# ---------------------------------------------------------------------------
# 13. DatasetMixer.get_stats returns dict with source name keys
# ---------------------------------------------------------------------------


def test_get_stats_has_all_source_keys():
    sources = make_sources()
    mixer = DatasetMixer(sources, MixerConfig(seed=3))
    mixer.sample_batch(batch_size=20, step=0)
    stats = mixer.get_stats()
    assert isinstance(stats, dict)
    for src in sources:
        assert src.name in stats
    assert sum(stats.values()) == 20


# ---------------------------------------------------------------------------
# 14. AdaptiveMixer.update_from_loss increases weight for high-loss source
# ---------------------------------------------------------------------------


def test_adaptive_mixer_update_from_loss_increases_weight():
    sources = [
        make_source("easy", n=15, weight=1.0),
        make_source("hard", n=15, weight=1.0),
    ]
    mixer = AdaptiveMixer(sources, MixerConfig(seed=5), adaptation_rate=0.5)

    # Record a high loss for "hard"
    mixer.update_from_loss("hard", loss=5.0)
    weights = mixer.get_adapted_weights()

    assert weights["hard"] > weights["easy"]


# ---------------------------------------------------------------------------
# 15. AdaptiveMixer.get_adapted_weights returns dict with all source names
# ---------------------------------------------------------------------------


def test_adaptive_mixer_get_adapted_weights_all_sources():
    sources = make_sources()
    mixer = AdaptiveMixer(sources, MixerConfig(seed=6))
    weights = mixer.get_adapted_weights()
    assert isinstance(weights, dict)
    for src in sources:
        assert src.name in weights


# ---------------------------------------------------------------------------
# 16. DatasetMixer with curriculum strategy: weights change over steps
# ---------------------------------------------------------------------------


def test_curriculum_weights_change_over_steps():
    sources = [
        make_source("easy_src", n=15, weight=1.0, domain="easy_domain"),
        make_source("hard_src", n=15, weight=1.0, domain="hard_domain"),
    ]
    config = MixerConfig(
        strategy="curriculum",
        curriculum_steps=1000,
        seed=10,
    )
    mixer = DatasetMixer(sources, config)

    weights_early = mixer._get_current_weights(step=0)
    weights_late = mixer._get_current_weights(step=999)

    # Weights should differ between early and late steps
    assert weights_early != weights_late


# ---------------------------------------------------------------------------
# Bonus: weight_history records updates
# ---------------------------------------------------------------------------


def test_weight_history_records_updates():
    sources = make_sources()
    mixer = DatasetMixer(sources, MixerConfig(seed=11))

    assert mixer.get_weight_history() == []
    mixer.update_weights("src_a", 3.0)
    mixer.update_weights("src_b", 0.5)
    history = mixer.get_weight_history()
    assert len(history) == 2
    assert history[0]["source"] == "src_a"
    assert history[1]["source"] == "src_b"
