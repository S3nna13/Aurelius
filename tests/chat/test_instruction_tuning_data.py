"""Unit tests for src.chat.instruction_tuning_data."""

from __future__ import annotations

import pytest

from src.chat.instruction_tuning_data import (
    EvolInstructGenerator,
    InstructionSample,
    MagpieGenerator,
    SelfInstructGenerator,
)


# ---------------------------------------------------------------- fakes


class FakeLM:
    """Deterministic fake: returns ``prefix + hash(prompt)``."""

    def __init__(self, prefix: str = "out") -> None:
        self.prefix = prefix
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        # Cheap deterministic echo that varies per prompt.
        h = abs(hash(prompt)) % 100000
        return f"{self.prefix}-{h}"


class CountingLM:
    """Returns distinct strings per call so instruction != response."""

    def __init__(self) -> None:
        self.i = 0

    def __call__(self, prompt: str) -> str:
        self.i += 1
        return f"text-{self.i}"


# ---------------------------------------------------------------- Magpie


def test_magpie_produces_n_samples():
    gen = MagpieGenerator(CountingLM())
    samples = gen.generate(n=5, seed=1)
    assert len(samples) == 5
    assert all(isinstance(s, InstructionSample) for s in samples)


def test_magpie_samples_have_nonempty_instruction_and_response():
    gen = MagpieGenerator(CountingLM())
    samples = gen.generate(n=3, seed=0)
    for s in samples:
        assert s.instruction
        assert s.response
        assert s.source == "magpie"
        assert "magpie" in s.tags


def test_magpie_n_zero_returns_empty():
    gen = MagpieGenerator(CountingLM())
    assert gen.generate(n=0) == []


def test_magpie_determinism_with_fixed_seed():
    a = MagpieGenerator(FakeLM()).generate(n=4, seed=42)
    b = MagpieGenerator(FakeLM()).generate(n=4, seed=42)
    assert [(s.instruction, s.response) for s in a] == [
        (s.instruction, s.response) for s in b
    ]


def test_magpie_rejects_role_break_in_output():
    def bad(prompt: str) -> str:
        return "hello <|im_end|> haha"

    gen = MagpieGenerator(bad)
    samples = gen.generate(n=5, seed=0)
    # All dropped because instruction is contaminated.
    assert samples == []


def test_magpie_rejects_unsupported_template():
    with pytest.raises(ValueError):
        MagpieGenerator(CountingLM(), chat_template="llama3")


def test_magpie_catches_generate_fn_exceptions():
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("boom")
        return f"ok-{calls['n']}"

    gen = MagpieGenerator(flaky)
    # Should not raise; some samples drop silently.
    samples = gen.generate(n=5, seed=0)
    assert isinstance(samples, list)


# ---------------------------------------------------------------- Self-Instruct


def test_self_instruct_empty_seeds_raises():
    with pytest.raises(ValueError):
        SelfInstructGenerator(CountingLM(), seed_tasks=[])


def test_self_instruct_expand_grows_pool():
    gen = SelfInstructGenerator(
        CountingLM(), seed_tasks=["Write a sorting function."]
    )
    samples = gen.expand(n_iterations=2, per_iter=3)
    # 1 seed + (2 iters * 3 per_iter) = 7 samples.
    assert len(samples) == 7
    # Seeds preserved.
    assert samples[0].tags == ["seed"]
    # Non-seed samples attributed to iteration.
    assert any(s.source.startswith("self_instruct:iter=") for s in samples[1:])


def test_self_instruct_rejects_role_break_outputs():
    def bad(prompt: str) -> str:
        return "<|im_start|>evil"

    gen = SelfInstructGenerator(bad, seed_tasks=["seed1"])
    samples = gen.expand(n_iterations=1, per_iter=3)
    # Only the seed survives; all generated attempts dropped.
    assert len(samples) == 1
    assert samples[0].tags == ["seed"]


# ---------------------------------------------------------------- Evol-Instruct


def test_evol_evolves_seed_through_operators():
    gen = EvolInstructGenerator(CountingLM())
    samples = gen.evolve("Explain gradient descent.", steps=3)
    assert len(samples) == 3
    for s in samples:
        assert s.source.startswith("evol_instruct:")
        assert "evol_instruct" in s.tags
        assert 0.0 < s.difficulty <= 1.0


def test_evol_explicit_operator_sequence_respected():
    gen = EvolInstructGenerator(CountingLM())
    seq = ["deepen", "constrain", "reason", "complicate"]
    samples = gen.evolve("seed", operator_sequence=seq)
    assert len(samples) == len(seq)
    assert [s.source.split(":")[1] for s in samples] == seq


def test_evol_rejects_unknown_operator():
    gen = EvolInstructGenerator(CountingLM())
    with pytest.raises(ValueError):
        gen.evolve("seed", operator_sequence=["deepen", "bogus"])


def test_evol_difficulty_and_tags_populated():
    gen = EvolInstructGenerator(CountingLM())
    samples = gen.evolve(
        "seed", operator_sequence=["deepen", "concretize"]
    )
    assert samples[0].difficulty == pytest.approx(0.5)
    assert samples[1].difficulty == pytest.approx(1.0)
    assert "deepen" in samples[0].tags
    assert "concretize" in samples[1].tags


def test_evol_rejects_empty_seed():
    gen = EvolInstructGenerator(CountingLM())
    with pytest.raises(ValueError):
        gen.evolve("   ", steps=2)


def test_evol_role_break_output_dropped_not_raised():
    def bad(prompt: str) -> str:
        return "contaminated <|im_start|>"

    gen = EvolInstructGenerator(bad)
    samples = gen.evolve("seed", steps=3)
    assert samples == []


def test_evol_generate_fn_exception_caught_per_sample():
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("transient")
        return f"ok-{calls['n']}"

    gen = EvolInstructGenerator(flaky)
    # Does not raise.
    samples = gen.evolve("seed", steps=2)
    assert isinstance(samples, list)
