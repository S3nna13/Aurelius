"""Tests for src/data/synthetic_cot_dataset.py"""

import json
import pathlib
import tempfile

from src.data.synthetic_cot_dataset import (
    COT_DATASET_REGISTRY,
    CoTDatasetConfig,
    CoTExample,
    SyntheticCoTDataset,
)


def test_generate_math_example_returns_cot_example():
    ds = SyntheticCoTDataset()
    ex = ds.generate_math_example(seed=0)
    assert isinstance(ex, CoTExample)
    assert ex.domain == "math"


def test_generate_math_example_chain_has_multiple_lines():
    ds = SyntheticCoTDataset()
    ex = ds.generate_math_example(seed=0)
    lines = [line for line in ex.chain_of_thought.split("\n") if line.strip()]
    assert len(lines) >= 2


def test_generate_math_example_answer_is_numeric():
    ds = SyntheticCoTDataset()
    ex = ds.generate_math_example(seed=0)
    int(ex.answer)


def test_generate_logic_example_returns_cot_example():
    ds = SyntheticCoTDataset()
    ex = ds.generate_logic_example(seed=1)
    assert isinstance(ex, CoTExample)
    assert ex.domain == "logic"


def test_generate_logic_example_chain_has_multiple_lines():
    ds = SyntheticCoTDataset()
    ex = ds.generate_logic_example(seed=1)
    lines = [line for line in ex.chain_of_thought.split("\n") if line.strip()]
    assert len(lines) >= 2


def test_generate_example_dispatch_math():
    ds = SyntheticCoTDataset()
    ex = ds.generate_example("math", seed=0)
    assert ex.domain == "math"


def test_generate_example_dispatch_logic():
    ds = SyntheticCoTDataset()
    ex = ds.generate_example("logic", seed=0)
    assert ex.domain == "logic"


def test_generate_example_dispatch_coding():
    ds = SyntheticCoTDataset()
    ex = ds.generate_example("coding", seed=0)
    assert ex.domain == "coding"


def test_generate_example_dispatch_general():
    ds = SyntheticCoTDataset()
    ex = ds.generate_example("general", seed=0)
    assert ex.domain == "general"


def test_generate_returns_n_examples():
    config = CoTDatasetConfig(n_examples=8)
    ds = SyntheticCoTDataset(config)
    examples = ds.generate()
    assert len(examples) == 8


def test_generate_with_explicit_n():
    ds = SyntheticCoTDataset()
    examples = ds.generate(n=5)
    assert len(examples) == 5


def test_generate_all_cot_examples():
    ds = SyntheticCoTDataset()
    examples = ds.generate(n=4)
    assert all(isinstance(e, CoTExample) for e in examples)


def test_to_chatml_contains_question():
    ds = SyntheticCoTDataset()
    ex = ds.generate_math_example(seed=0)
    chatml = ds.to_chatml(ex)
    assert ex.question in chatml


def test_to_chatml_contains_answer():
    ds = SyntheticCoTDataset()
    ex = ds.generate_math_example(seed=0)
    chatml = ds.to_chatml(ex)
    assert f"Answer: {ex.answer}" in chatml


def test_to_chatml_format():
    ds = SyntheticCoTDataset()
    ex = ds.generate_math_example(seed=0)
    chatml = ds.to_chatml(ex)
    assert "<|user|>" in chatml
    assert "<|assistant|>" in chatml
    assert "<|end|>" in chatml


def test_export_jsonl_writes_valid_jsonl():
    ds = SyntheticCoTDataset()
    examples = ds.generate(n=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "out.jsonl"
        ds.export_jsonl(examples, path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert "question" in record
            assert "chain_of_thought" in record
            assert "answer" in record
            assert "domain" in record


def test_registry_has_default():
    assert "default" in COT_DATASET_REGISTRY
    assert COT_DATASET_REGISTRY["default"] is SyntheticCoTDataset
