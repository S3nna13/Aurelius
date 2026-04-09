"""Tests for synthetic instruction data generation."""

from __future__ import annotations

import random

import pytest

from src.data.synthetic_instruct import (
    InstructConfig,
    InstructionGenerator,
    InstructionSample,
    evolve_instruction,
    generate_coding_instruction,
    generate_factual_instruction,
    generate_math_instruction,
    generate_reasoning_instruction,
    generate_writing_instruction,
)


RNG = random.Random(42)


# ---- 1. InstructConfig defaults ----

def test_instruct_config_defaults():
    cfg = InstructConfig()
    assert cfg.n_instructions == 100
    assert cfg.max_instruction_len == 200
    assert cfg.complexity_levels == 3
    assert cfg.domains == ["math", "coding", "reasoning", "writing", "factual"]
    assert cfg.seed == 42
    assert cfg.include_cot is True


# ---- 2. generate_math_instruction returns InstructionSample with domain="math" ----

def test_math_instruction_domain():
    sample = generate_math_instruction(random.Random(42), complexity=1)
    assert isinstance(sample, InstructionSample)
    assert sample.domain == "math"


# ---- 3. generate_math_instruction output contains a number ----

def test_math_instruction_output_has_number():
    for c in [1, 2, 3]:
        sample = generate_math_instruction(random.Random(42), complexity=c)
        assert any(ch.isdigit() for ch in sample.output), (
            f"Math output at complexity={c} should contain a number"
        )


# ---- 4. generate_coding_instruction returns InstructionSample with domain="coding" ----

def test_coding_instruction_domain():
    sample = generate_coding_instruction(random.Random(42), complexity=1)
    assert isinstance(sample, InstructionSample)
    assert sample.domain == "coding"


# ---- 5. generate_reasoning_instruction returns valid InstructionSample ----

def test_reasoning_instruction_valid():
    sample = generate_reasoning_instruction(random.Random(42), complexity=2)
    assert isinstance(sample, InstructionSample)
    assert sample.domain == "reasoning"
    assert len(sample.instruction) > 0
    assert len(sample.output) > 0


# ---- 6. generate_writing_instruction returns valid InstructionSample ----

def test_writing_instruction_valid():
    sample = generate_writing_instruction(random.Random(42), complexity=1)
    assert isinstance(sample, InstructionSample)
    assert sample.domain == "writing"
    assert len(sample.instruction) > 0
    assert len(sample.output) > 0


# ---- 7. generate_factual_instruction returns valid InstructionSample ----

def test_factual_instruction_valid():
    sample = generate_factual_instruction(random.Random(42), complexity=1)
    assert isinstance(sample, InstructionSample)
    assert sample.domain == "factual"
    assert len(sample.instruction) > 0
    assert len(sample.output) > 0


# ---- 8. evolve_instruction increases complexity (capped at max) ----

def test_evolve_increases_complexity():
    base = generate_math_instruction(random.Random(42), complexity=1)
    assert base.complexity == 1
    evolved = evolve_instruction(random.Random(42), base)
    assert evolved.complexity == 2

    # Already at max complexity=3 -> stays at 3
    base3 = generate_math_instruction(random.Random(42), complexity=3)
    evolved3 = evolve_instruction(random.Random(42), base3)
    assert evolved3.complexity == 3


# ---- 9. InstructionGenerator.generate returns correct count ----

def test_generator_generate_count():
    cfg = InstructConfig(n_instructions=25, seed=42)
    gen = InstructionGenerator(cfg)
    samples = gen.generate()
    assert len(samples) == 25


# ---- 10. InstructionGenerator.generate covers multiple domains ----

def test_generator_covers_domains():
    cfg = InstructConfig(n_instructions=50, seed=42)
    gen = InstructionGenerator(cfg)
    samples = gen.generate()
    domains_seen = {s.domain for s in samples}
    assert len(domains_seen) >= 3, f"Expected at least 3 domains, got {domains_seen}"


# ---- 11. generate_batch with domain filter only returns that domain ----

def test_generate_batch_domain_filter():
    gen = InstructionGenerator(InstructConfig(seed=42))
    batch = gen.generate_batch(10, domain="math")
    assert len(batch) == 10
    assert all(s.domain == "math" for s in batch)


# ---- 12. to_sft_format returns list of dicts with prompt/completion keys ----

def test_to_sft_format():
    gen = InstructionGenerator(InstructConfig(seed=42))
    samples = gen.generate_batch(5)
    sft = gen.to_sft_format(samples)
    assert isinstance(sft, list)
    assert len(sft) == 5
    for item in sft:
        assert isinstance(item, dict)
        assert "prompt" in item
        assert "completion" in item
        assert isinstance(item["prompt"], str)
        assert isinstance(item["completion"], str)


# ---- 13. InstructionGenerator reproducible with same seed ----

def test_generator_reproducible():
    gen1 = InstructionGenerator(InstructConfig(n_instructions=20, seed=42))
    gen2 = InstructionGenerator(InstructConfig(n_instructions=20, seed=42))
    samples1 = gen1.generate()
    samples2 = gen2.generate()
    for s1, s2 in zip(samples1, samples2):
        assert s1.instruction == s2.instruction
        assert s1.output == s2.output
        assert s1.domain == s2.domain
        assert s1.complexity == s2.complexity


# ---- 14. generate_math_instruction with include_cot=True has step-by-step ----

def test_math_instruction_cot():
    for c in [1, 2, 3]:
        sample = generate_math_instruction(random.Random(42), complexity=c, include_cot=True)
        assert sample.has_cot is True
        assert "step-by-step" in sample.output.lower(), (
            f"CoT math output at complexity={c} should contain 'step-by-step'"
        )
