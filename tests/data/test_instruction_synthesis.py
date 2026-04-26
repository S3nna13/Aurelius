"""
tests/data/test_instruction_synthesis.py

≥10 tests for instruction_synthesis.py using only the Python stdlib.
Tiny configs are used throughout to keep tests fast.
"""

import pytest

from src.data.instruction_synthesis import (
    InstructionSynthesizer,
    SynthConfig,
    classify_instruction_type,
    compute_rouge1_similarity,
    deduplicate_instructions,
    estimate_difficulty,
    fill_template,
)

# ---------------------------------------------------------------------------
# SynthConfig defaults
# ---------------------------------------------------------------------------


class TestSynthConfigDefaults:
    def test_default_n_instructions_per_seed(self):
        cfg = SynthConfig()
        assert cfg.n_instructions_per_seed == 10

    def test_default_template_count(self):
        cfg = SynthConfig()
        assert len(cfg.templates) == 4

    def test_default_similarity_threshold(self):
        cfg = SynthConfig()
        assert cfg.similarity_threshold == pytest.approx(0.7)

    def test_default_max_min_length(self):
        cfg = SynthConfig()
        assert cfg.max_length == 512
        assert cfg.min_length == 20


# ---------------------------------------------------------------------------
# fill_template
# ---------------------------------------------------------------------------


class TestFillTemplate:
    def test_fills_known_slots(self):
        result = fill_template("Explain {topic}", {"topic": "gravity"})
        assert result == "Explain gravity"

    def test_leaves_unknown_slots_as_is(self):
        result = fill_template("Write a {adjective} essay about {topic}", {"topic": "space"})
        assert "{adjective}" in result
        assert "space" in result

    def test_fills_multiple_slots(self):
        result = fill_template(
            "Compare {a} and {b}",
            {"a": "Python", "b": "Rust"},
        )
        assert result == "Compare Python and Rust"

    def test_empty_slots_dict(self):
        template = "List {n} ways to {action}"
        result = fill_template(template, {})
        assert result == template  # nothing changed


# ---------------------------------------------------------------------------
# compute_rouge1_similarity
# ---------------------------------------------------------------------------


class TestRouge1Similarity:
    def test_identical_strings_return_one(self):
        assert compute_rouge1_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different_strings_less_than_one(self):
        score = compute_rouge1_similarity("the cat sat on the mat", "dogs fly in the sky")
        assert score < 1.0

    def test_empty_strings_return_one(self):
        assert compute_rouge1_similarity("", "") == pytest.approx(1.0)

    def test_one_empty_string_returns_zero(self):
        assert compute_rouge1_similarity("hello", "") == pytest.approx(0.0)

    def test_partial_overlap_between_zero_and_one(self):
        score = compute_rouge1_similarity("the quick brown fox", "the slow brown dog")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# deduplicate_instructions
# ---------------------------------------------------------------------------


class TestDeduplicateInstructions:
    def test_removes_near_duplicates(self):
        instructions = [
            "Explain machine learning",
            "Explain machine learning concepts",  # highly similar
        ]
        result = deduplicate_instructions(instructions, threshold=0.7)
        # The second is very similar to the first; only one should survive
        assert len(result) == 1

    def test_keeps_distinct_instructions(self):
        instructions = [
            "Explain gravity",
            "Write a poem about the ocean",
            "List 5 ways to improve memory",
        ]
        result = deduplicate_instructions(instructions, threshold=0.7)
        assert len(result) == 3

    def test_preserves_order(self):
        instructions = ["alpha beta", "gamma delta", "epsilon zeta"]
        result = deduplicate_instructions(instructions, threshold=0.7)
        assert result == instructions

    def test_identical_strings_collapsed(self):
        instructions = ["same text here", "same text here", "same text here"]
        result = deduplicate_instructions(instructions, threshold=0.7)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# classify_instruction_type
# ---------------------------------------------------------------------------


class TestClassifyInstructionType:
    VALID_TYPES = {"open_qa", "classification", "generation", "analysis", "other"}

    def test_open_qa_what(self):
        assert classify_instruction_type("What is the speed of light?") == "open_qa"

    def test_open_qa_how(self):
        assert classify_instruction_type("How does a neural network work?") == "open_qa"

    def test_classification_type(self):
        assert (
            classify_instruction_type("Classify the following text into categories")
            == "classification"
        )

    def test_generation_type(self):
        assert classify_instruction_type("Write a short story about dragons") == "generation"

    def test_analysis_type(self):
        assert (
            classify_instruction_type("Compare Python and Java for web development") == "analysis"
        )

    def test_other_type(self):
        result = classify_instruction_type("Recite the alphabet backwards")
        assert result == "other"

    def test_returns_valid_type_string(self):
        instructions = [
            "What is AI?",
            "Classify this sentence",
            "Generate a haiku",
            "Explain recursion",
            "Do something random",
        ]
        for instr in instructions:
            assert classify_instruction_type(instr) in self.VALID_TYPES


# ---------------------------------------------------------------------------
# estimate_difficulty
# ---------------------------------------------------------------------------


class TestEstimateDifficulty:
    def test_returns_value_in_unit_interval(self):
        for text in ["short", "a" * 300, "x" * 600]:
            d = estimate_difficulty(text)
            assert 0.0 <= d <= 1.0, f"difficulty out of range for text of len {len(text)}"

    def test_short_instruction_easier_than_long(self):
        short = "Explain AI"
        long_text = "Explain the historical, philosophical, and technical foundations of artificial intelligence, including symbolic AI, connectionism, and modern deep learning, with examples from computer vision, natural language processing, robotics, and reinforcement learning, and discuss ethical implications."  # noqa: E501
        assert estimate_difficulty(short) < estimate_difficulty(long_text)

    def test_max_length_clips_to_one(self):
        very_long = "a " * 300  # 600 chars, many commas if we add them
        d = estimate_difficulty(very_long, max_length=50)
        assert d == pytest.approx(1.0)

    def test_empty_instruction_gives_zero(self):
        assert estimate_difficulty("") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# InstructionSynthesizer.from_template
# ---------------------------------------------------------------------------


class TestFromTemplate:
    def setup_method(self):
        self.cfg = SynthConfig(min_length=5, max_length=100)
        self.synth = InstructionSynthesizer(self.cfg)

    def test_returns_list(self):
        result = self.synth.from_template(
            "Explain {topic}",
            [{"topic": "gravity"}, {"topic": "photosynthesis"}],
        )
        assert isinstance(result, list)

    def test_filters_too_short(self):
        # min_length=5, "Hi" has length 2 → filtered
        cfg = SynthConfig(min_length=5, max_length=200)
        synth = InstructionSynthesizer(cfg)
        result = synth.from_template(
            "{x}", [{"x": "Hi"}, {"x": "Hello world, this is long enough"}]
        )
        # "Hi" (2 chars) should be filtered out
        assert all(len(r) >= 5 for r in result)

    def test_filters_too_long(self):
        cfg = SynthConfig(min_length=1, max_length=10)
        synth = InstructionSynthesizer(cfg)
        result = synth.from_template(
            "{x}",
            [{"x": "short"}, {"x": "this is definitely way too long for the limit"}],
        )
        assert all(len(r) <= 10 for r in result)

    def test_unfilled_template_may_be_filtered(self):
        # Template with no slots, below min_length
        cfg = SynthConfig(min_length=50, max_length=200)
        synth = InstructionSynthesizer(cfg)
        result = synth.from_template("Hi {name}", [{}])
        # "Hi {name}" is 9 chars, below min_length=50 → filtered
        assert result == []


# ---------------------------------------------------------------------------
# InstructionSynthesizer.build_dataset
# ---------------------------------------------------------------------------


class TestBuildDataset:
    def setup_method(self):
        self.cfg = SynthConfig(n_instructions_per_seed=3, similarity_threshold=0.95)
        self.synth = InstructionSynthesizer(self.cfg)

    def test_returns_list_of_dicts(self):
        seeds = ["Explain neural networks"]
        slots = [{"neural": "deep"}, {"networks": "architectures"}]
        dataset = self.synth.build_dataset(seeds, slots)
        assert isinstance(dataset, list)
        assert len(dataset) > 0
        assert isinstance(dataset[0], dict)

    def test_required_keys_present(self):
        seeds = ["Write a story about space exploration"]
        slots = [{"space": "ocean"}, {"story": "poem"}]
        dataset = self.synth.build_dataset(seeds, slots)
        for item in dataset:
            assert "instruction" in item
            assert "type" in item
            assert "difficulty" in item

    def test_difficulty_values_in_range(self):
        seeds = ["Compare apples and oranges"]
        slots = [{"apples": "cats"}, {"oranges": "dogs"}]
        dataset = self.synth.build_dataset(seeds, slots)
        for item in dataset:
            assert 0.0 <= item["difficulty"] <= 1.0


# ---------------------------------------------------------------------------
# InstructionSynthesizer.get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def setup_method(self):
        self.cfg = SynthConfig(n_instructions_per_seed=5, similarity_threshold=0.95)
        self.synth = InstructionSynthesizer(self.cfg)

    def _make_dataset(self):
        seeds = [
            "What is machine learning?",
            "Write a poem about autumn leaves",
            "Compare REST and GraphQL",
            "Classify this sentence as positive or negative",
        ]
        slots = [{"machine": "deep"}, {"poem": "story"}, {"REST": "SOAP"}]
        return self.synth.build_dataset(seeds, slots)

    def test_has_all_four_keys(self):
        dataset = self._make_dataset()
        stats = self.synth.get_stats(dataset)
        assert "n_total" in stats
        assert "type_distribution" in stats
        assert "mean_difficulty" in stats
        assert "mean_length" in stats

    def test_n_total_matches_dataset_length(self):
        dataset = self._make_dataset()
        stats = self.synth.get_stats(dataset)
        assert stats["n_total"] == len(dataset)

    def test_type_distribution_sums_to_n_total(self):
        dataset = self._make_dataset()
        stats = self.synth.get_stats(dataset)
        dist_sum = sum(stats["type_distribution"].values())
        assert dist_sum == stats["n_total"]

    def test_mean_difficulty_in_unit_interval(self):
        dataset = self._make_dataset()
        stats = self.synth.get_stats(dataset)
        assert 0.0 <= stats["mean_difficulty"] <= 1.0

    def test_mean_length_positive(self):
        dataset = self._make_dataset()
        stats = self.synth.get_stats(dataset)
        assert stats["mean_length"] > 0

    def test_empty_dataset_returns_zeros(self):
        stats = self.synth.get_stats([])
        assert stats["n_total"] == 0
        assert stats["mean_difficulty"] == pytest.approx(0.0)
        assert stats["mean_length"] == pytest.approx(0.0)
        assert stats["type_distribution"] == {}
