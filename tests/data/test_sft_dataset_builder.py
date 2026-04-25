"""Tests for src/data/sft_dataset_builder.py"""
import json

import pytest

from src.data.sft_dataset_builder import (
    SFTExample,
    SFTDatasetConfig,
    SFTDatasetBuilder,
    SFT_BUILDER_REGISTRY,
)


def make_example(prompt: str = "What is AI?", response: str = "AI is artificial intelligence.") -> SFTExample:
    return SFTExample(prompt=prompt, response=response)


def test_add_single_example():
    builder = SFTDatasetBuilder()
    builder.add(make_example())
    assert len(builder._examples) == 1


def test_add_batch():
    builder = SFTDatasetBuilder()
    builder.add_batch([make_example("Q1", "A1"), make_example("Q2", "A2")])
    assert len(builder._examples) == 2


def test_filter_by_length_drops_long():
    config = SFTDatasetConfig(max_length=10)
    builder = SFTDatasetBuilder(config)
    short = SFTExample(prompt="Hi", response="Ok")
    long = SFTExample(prompt="x" * 100, response="y" * 100)
    result = builder.filter_by_length([short, long])
    assert short in result
    assert long not in result


def test_filter_by_length_keeps_within_limit():
    config = SFTDatasetConfig(max_length=1000)
    builder = SFTDatasetBuilder(config)
    ex = make_example()
    result = builder.filter_by_length([ex])
    assert ex in result


def test_deduplicate_removes_exact_prompt_duplicates():
    builder = SFTDatasetBuilder()
    ex1 = SFTExample(prompt="same", response="response 1")
    ex2 = SFTExample(prompt="same", response="response 2")
    result = builder.deduplicate([ex1, ex2])
    assert len(result) == 1
    assert result[0] is ex1


def test_deduplicate_keeps_unique():
    builder = SFTDatasetBuilder()
    ex1 = SFTExample(prompt="A", response="R1")
    ex2 = SFTExample(prompt="B", response="R2")
    result = builder.deduplicate([ex1, ex2])
    assert len(result) == 2


def test_build_splits_correctly():
    config = SFTDatasetConfig(split_ratio=0.8, shuffle=False, dedup=False)
    builder = SFTDatasetBuilder(config)
    for i in range(10):
        builder.add(SFTExample(prompt=f"Q{i}", response=f"A{i}"))
    splits = builder.build()
    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 2


def test_build_returns_train_val_keys():
    builder = SFTDatasetBuilder()
    builder.add(make_example())
    splits = builder.build()
    assert "train" in splits
    assert "val" in splits


def test_build_dedup_applied():
    config = SFTDatasetConfig(dedup=True, shuffle=False)
    builder = SFTDatasetBuilder(config)
    builder.add(SFTExample(prompt="dup", response="r1"))
    builder.add(SFTExample(prompt="dup", response="r2"))
    builder.add(SFTExample(prompt="unique", response="r3"))
    splits = builder.build()
    total = len(splits["train"]) + len(splits["val"])
    assert total == 2


def test_to_jsonl_lines_valid_json():
    builder = SFTDatasetBuilder()
    examples = [make_example("Q1", "A1"), make_example("Q2", "A2")]
    lines = builder.to_jsonl_lines(examples)
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "prompt" in record
        assert "response" in record


def test_stats_has_all_keys():
    builder = SFTDatasetBuilder()
    for i in range(5):
        builder.add(SFTExample(prompt=f"Q{i}", response=f"A{i}"))
    s = builder.stats()
    for key in ("total", "after_filter", "after_dedup", "train", "val"):
        assert key in s


def test_stats_total_correct():
    builder = SFTDatasetBuilder()
    for i in range(7):
        builder.add(make_example(f"Q{i}", f"A{i}"))
    s = builder.stats()
    assert s["total"] == 7


def test_registry_has_default():
    assert "default" in SFT_BUILDER_REGISTRY
    assert SFT_BUILDER_REGISTRY["default"] is SFTDatasetBuilder
