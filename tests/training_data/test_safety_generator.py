from __future__ import annotations

import json
import tempfile
from pathlib import Path

from training_data.safety_generator import SafetyDataGenerator


def test_default_config() -> None:
    g = SafetyDataGenerator({})
    assert len(g.CATEGORIES) == 5
    assert "refusals" in g.CATEGORIES
    assert "jailbreak_resistant" in g.CATEGORIES
    assert "ethical_reasoning" in g.CATEGORIES
    assert "content_moderation" in g.CATEGORIES
    assert "privacy_aware" in g.CATEGORIES


def test_generate_example_refusals() -> None:
    g = SafetyDataGenerator({})
    ex = g.generate_example("refusals")
    assert ex["category"] == "refusals"
    assert ex["is_refusal"] is True
    assert isinstance(ex["instruction"], str)
    assert isinstance(ex["response"], str)
    assert len(ex["instruction"]) > 0
    assert len(ex["response"]) > 0


def test_generate_example_jailbreak() -> None:
    g = SafetyDataGenerator({})
    ex = g.generate_example("jailbreak_resistant")
    assert ex["category"] == "jailbreak_resistant"
    assert ex["is_refusal"] is False
    keywords = ["recognize", "notice", "jailbreak", "safety", "understand", "aware", "see that"]
    assert any(k in ex["response"].lower() for k in keywords)


def test_generate_example_ethical() -> None:
    g = SafetyDataGenerator({})
    ex = g.generate_example("ethical_reasoning")
    assert ex["category"] == "ethical_reasoning"
    assert ex["is_refusal"] is False
    assert len(ex["response"]) > 50


def test_generate_example_moderation() -> None:
    g = SafetyDataGenerator({})
    ex = g.generate_example("content_moderation")
    assert ex["category"] == "content_moderation"
    assert "remove" in ex["response"].lower() or "action" in ex["response"].lower()


def test_generate_example_privacy() -> None:
    g = SafetyDataGenerator({})
    ex = g.generate_example("privacy_aware")
    assert ex["category"] == "privacy_aware"
    keywords = [
        "cannot",
        "can't",
        "private",
        "privacy",
        "personally",
        "won't",
        "should not",
        "confidential",
    ]
    assert any(k in ex["response"].lower() for k in keywords)


def test_generate_example_unknown_category() -> None:
    g = SafetyDataGenerator({})
    try:
        g.generate_example("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_all_categories_have_10_templates() -> None:
    g = SafetyDataGenerator({})
    for cat in g.CATEGORIES:
        method_name = g._METHOD_MAP[cat]
        templates = getattr(g.__class__, method_name)()
        assert len(templates) >= 10, f"{cat} has {len(templates)} templates, need >= 10"


def test_run_creates_jsonl_files() -> None:
    g = SafetyDataGenerator({})
    with tempfile.TemporaryDirectory() as tmpdir:
        g.run(100, tmpdir)
        safety_dir = Path(tmpdir) / "safety"
        assert (safety_dir / "train.jsonl").exists()
        assert (safety_dir / "val.jsonl").exists()


def test_run_train_val_split() -> None:
    g = SafetyDataGenerator({})
    with tempfile.TemporaryDirectory() as tmpdir:
        g.run(100, tmpdir)
        safety_dir = Path(tmpdir) / "safety"
        train_lines = (safety_dir / "train.jsonl").read_text().strip().split("\n")
        val_lines = (safety_dir / "val.jsonl").read_text().strip().split("\n")
        assert len(train_lines) == 90
        assert len(val_lines) == 10


def test_run_output_schema() -> None:
    g = SafetyDataGenerator({})
    with tempfile.TemporaryDirectory() as tmpdir:
        g.run(10, tmpdir)
        safety_dir = Path(tmpdir) / "safety"
        with open(safety_dir / "train.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                assert "instruction" in obj
                assert "response" in obj
                assert "category" in obj
                assert "is_refusal" in obj
                assert isinstance(obj["is_refusal"], bool)


def test_refusal_is_polite() -> None:
    g = SafetyDataGenerator({})
    templates = g.__class__._refusal_templates()
    for instruction, response in templates:
        assert (
            "cannot" in response
            or "can't" in response
            or "won't" in response
            or "sorry" in response.lower()
            or "unable" in response.lower()
        )


def test_jailbreak_detects_manipulation() -> None:
    g = SafetyDataGenerator({})
    templates = g.__class__._jailbreak_templates()
    for instruction, response in templates:
        keywords = [
            "recognize",
            "jailbreak",
            "attempt",
            "manipulation",
            "understand",
            "aware",
            "see that",
            "framing",
            "fictional",
            "won't",
            "don't respond",
            "don't participate",
            "remain bound",
            "doesn't change",
            "pressure tactic",
        ]
    assert any(k in response.lower() for k in keywords)


def test_output_is_json_serializable() -> None:
    g = SafetyDataGenerator({})
    ex = g.generate_example("refusals")
    json.dumps(ex)
