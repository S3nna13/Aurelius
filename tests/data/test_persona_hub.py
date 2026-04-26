"""Tests for src/data/persona_hub.py — PersonaHub synthetic data generation."""

from __future__ import annotations

from src.data.persona_hub import (
    Persona,
    PersonaConfig,
    PersonaHub,
    create_default_personas,
    estimate_coverage,
    persona_weighted_sample,
)

# ---------------------------------------------------------------------------
# Test 1: PersonaConfig defaults
# ---------------------------------------------------------------------------


def test_persona_config_defaults():
    """PersonaConfig should have the correct default values."""
    cfg = PersonaConfig()
    assert cfg.n_default_personas == 20
    assert cfg.allow_domain_filter is True
    expected_domains = {
        "math",
        "science",
        "programming",
        "writing",
        "history",
        "business",
        "education",
        "art",
    }
    assert set(cfg.domains) == expected_domains


# ---------------------------------------------------------------------------
# Test 2: create_default_personas returns 20 personas
# ---------------------------------------------------------------------------


def test_create_default_personas_count():
    """create_default_personas should return exactly 20 Persona objects."""
    personas = create_default_personas()
    assert len(personas) == 20


# ---------------------------------------------------------------------------
# Test 3: All default personas have required fields
# ---------------------------------------------------------------------------


def test_default_personas_required_fields():
    """Every default persona must have non-empty name, description, domain, and traits."""
    for p in create_default_personas():
        assert isinstance(p, Persona)
        assert p.name and isinstance(p.name, str)
        assert p.description and isinstance(p.description, str)
        assert p.domain and isinstance(p.domain, str)
        assert isinstance(p.traits, list) and len(p.traits) >= 1


# ---------------------------------------------------------------------------
# Test 4: sample_persona returns correct count
# ---------------------------------------------------------------------------


def test_sample_persona_count():
    """sample_persona(n) should return exactly n personas."""
    hub = PersonaHub(seed=0)
    for n in (1, 3, 5, 10):
        result = hub.sample_persona(n=n)
        assert len(result) == n
        assert all(isinstance(p, Persona) for p in result)


# ---------------------------------------------------------------------------
# Test 5: sample_persona with domain filter
# ---------------------------------------------------------------------------


def test_sample_persona_domain_filter():
    """sample_persona with domain filter should only return personas from that domain."""
    hub = PersonaHub(seed=0)
    for domain in ("math", "science", "programming", "art"):
        result = hub.sample_persona(n=1, domain=domain)
        assert all(p.domain == domain for p in result)


# ---------------------------------------------------------------------------
# Test 6: generate_instruction_prompt contains persona prefix
# ---------------------------------------------------------------------------


def test_generate_instruction_prompt_contains_prefix():
    """generate_instruction_prompt should contain the persona's to_prompt_prefix() string."""
    hub = PersonaHub(seed=0)
    for task_type in ("qa", "creative", "reasoning", "coding", "analysis"):
        persona = hub.sample_persona(n=1)[0]
        prompt = hub.generate_instruction_prompt(persona, task_type=task_type)
        assert persona.to_prompt_prefix() in prompt, (
            f"Prompt missing persona prefix for task_type='{task_type}'"
        )


# ---------------------------------------------------------------------------
# Test 7: diversify_dataset returns n_personas_per * len(instructions) examples
# ---------------------------------------------------------------------------


def test_diversify_dataset_output_size():
    """diversify_dataset should return exactly n_personas_per * len(instructions) dicts."""
    hub = PersonaHub(seed=0)
    base = ["Explain gradient descent.", "What is a linked list?", "Write a haiku about rain."]
    n_per = 3
    results = hub.diversify_dataset(base, n_personas_per=n_per)
    assert len(results) == n_per * len(base)


def test_diversify_dataset_dict_keys():
    """Each dict in diversify_dataset output should have the required keys."""
    hub = PersonaHub(seed=0)
    results = hub.diversify_dataset(["Test instruction"], n_personas_per=2)
    for item in results:
        assert "instruction" in item
        assert "persona_name" in item
        assert "domain" in item
        assert "persona_prefix" in item


# ---------------------------------------------------------------------------
# Test 8: compute_diversity_score returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_diversity_score_range():
    """compute_diversity_score should return a float in [0.0, 1.0]."""
    hub = PersonaHub(seed=0)
    instructions = [
        "Explain the quadratic formula.",
        "Write a short poem about autumn leaves.",
        "Describe the TCP/IP model.",
        "What are the causes of World War I?",
    ]
    score = hub.compute_diversity_score(instructions)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Test 9: More diverse instructions → higher diversity score
# ---------------------------------------------------------------------------


def test_compute_diversity_score_ordering():
    """Highly diverse instructions should score higher than near-identical ones."""
    hub = PersonaHub(seed=0)
    diverse = [
        "Solve the integral of x^2.",
        "Write a haiku about cherry blossoms.",
        "Describe the French Revolution.",
        "How does TCP handle packet loss?",
        "Analyse a balance sheet.",
    ]
    repetitive = [
        "What is 2 + 2?",
        "What is 3 + 3?",
        "What is 4 + 4?",
        "What is 5 + 5?",
        "What is 6 + 6?",
    ]
    score_diverse = hub.compute_diversity_score(diverse)
    score_repetitive = hub.compute_diversity_score(repetitive)
    assert score_diverse > score_repetitive, (
        f"Expected diverse ({score_diverse:.4f}) > repetitive ({score_repetitive:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 10: get_persona_distribution returns dict with all domains
# ---------------------------------------------------------------------------


def test_get_persona_distribution_all_domains():
    """get_persona_distribution should include every domain from PersonaConfig."""
    hub = PersonaHub(seed=0)
    dist = hub.get_persona_distribution()
    assert isinstance(dist, dict)
    cfg_domains = set(PersonaConfig().domains)
    assert cfg_domains.issubset(set(dist.keys())), (
        f"Missing domains: {cfg_domains - set(dist.keys())}"
    )
    assert all(v > 0 for v in dist.values())


# ---------------------------------------------------------------------------
# Test 11: persona_weighted_sample respects domain weights
# ---------------------------------------------------------------------------


def test_persona_weighted_sample_domain_weight():
    """Heavily weighting one domain should make it appear more often in samples."""
    personas = create_default_personas()
    # Give math a very high weight, all others near zero
    weights = {d: 0.001 for d in PersonaConfig().domains}
    weights["math"] = 1000.0
    sampled = persona_weighted_sample(personas, weights, n=50)
    assert len(sampled) == 50
    math_count = sum(1 for p in sampled if p.domain == "math")
    # Expect the vast majority to be math personas
    assert math_count > 40, f"Expected >40 math personas but got {math_count}"


# ---------------------------------------------------------------------------
# Test 12: estimate_coverage returns 1.0 when all domains covered
# ---------------------------------------------------------------------------


def test_estimate_coverage_full():
    """estimate_coverage should return 1.0 when sampled personas cover all domains."""
    all_personas = create_default_personas()
    # sample_persona without domain filter guarantees coverage if we take all
    coverage = estimate_coverage(all_personas, all_personas)
    assert coverage == 1.0


def test_estimate_coverage_partial():
    """estimate_coverage should return < 1.0 when only a subset of domains are sampled."""
    all_personas = create_default_personas()
    math_only = [p for p in all_personas if p.domain == "math"]
    coverage = estimate_coverage(math_only, all_personas)
    assert 0.0 < coverage < 1.0


def test_estimate_coverage_empty_all():
    """estimate_coverage should return 1.0 when all_personas is empty (vacuously true)."""
    assert estimate_coverage([], []) == 1.0
