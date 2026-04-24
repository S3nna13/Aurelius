"""Tests for src.data.magpie_generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.domain_templates import get_template
from src.data.magpie_generator import (
    MAGPIE_GENERATOR_REGISTRY,
    MagpieConfig,
    MagpieExample,
    MagpieGenerator,
)


def _fn_user_assistant(**_kwargs) -> str:
    return "User: Explain recursion briefly.\nAssistant: Recursion is a function calling itself."


def _fn_blank_line(**_kwargs) -> str:
    return "What is 2+2?\n\nIt is 4."


def _fn_empty(**_kwargs) -> str:
    return ""


def _fn_lowercase_markers(**_kwargs) -> str:
    return "user: hello\nassistant: hi there"


def test_config_defaults():
    cfg = MagpieConfig()
    assert cfg.model_id == "dummy"
    assert cfg.output_dir == "data/synthetic"
    assert cfg.batch_size == 32
    assert cfg.max_new_tokens == 2048
    assert cfg.temperature == 0.9
    assert cfg.seed == 42


def test_config_overrides():
    cfg = MagpieConfig(model_id="m", batch_size=4, temperature=0.2)
    assert cfg.model_id == "m"
    assert cfg.batch_size == 4
    assert cfg.temperature == 0.2


def test_registry_has_default():
    assert MAGPIE_GENERATOR_REGISTRY["default"] is MagpieGenerator


def test_example_is_frozen():
    ex = MagpieExample(instruction="i", response="r", domain="d")
    with pytest.raises(Exception):
        ex.instruction = "x"  # type: ignore[misc]


def test_generate_from_template_user_assistant():
    gen = MagpieGenerator(MagpieConfig())
    t = get_template("coding")
    out = gen.generate_from_template(t, 3, _fn_user_assistant)
    assert len(out) == 3
    assert out[0].instruction == "Explain recursion briefly."
    assert out[0].response.startswith("Recursion")


def test_generate_from_template_domain_set():
    gen = MagpieGenerator(MagpieConfig())
    t = get_template("reasoning")
    out = gen.generate_from_template(t, 1, _fn_user_assistant)
    assert out[0].domain == "reasoning"


def test_generate_stores_prefix():
    gen = MagpieGenerator(MagpieConfig())
    t = get_template("general")
    out = gen.generate_from_template(t, 1, _fn_user_assistant)
    assert out[0].template_prefix == t.prefix


def test_generate_fallback_split():
    gen = MagpieGenerator(MagpieConfig())
    t = get_template("general")
    out = gen.generate_from_template(t, 2, _fn_blank_line)
    assert len(out) == 2
    assert out[0].instruction == "What is 2+2?"
    assert out[0].response == "It is 4."


def test_generate_lowercase_markers():
    gen = MagpieGenerator(MagpieConfig())
    t = get_template("general")
    out = gen.generate_from_template(t, 1, _fn_lowercase_markers)
    assert out[0].instruction == "hello"
    assert out[0].response == "hi there"


def test_generate_empty_output_skipped():
    gen = MagpieGenerator(MagpieConfig())
    t = get_template("general")
    out = gen.generate_from_template(t, 5, _fn_empty)
    assert out == []


def test_generate_zero_examples():
    gen = MagpieGenerator(MagpieConfig())
    t = get_template("general")
    out = gen.generate_from_template(t, 0, _fn_user_assistant)
    assert out == []


def test_generate_negative_raises():
    gen = MagpieGenerator(MagpieConfig())
    t = get_template("general")
    with pytest.raises(ValueError):
        gen.generate_from_template(t, -1, _fn_user_assistant)


def test_generate_fn_receives_prompt_and_params():
    seen: dict = {}

    def fn(**kwargs) -> str:
        seen.update(kwargs)
        return "User: q\nAssistant: a"

    gen = MagpieGenerator(MagpieConfig(max_new_tokens=128, temperature=0.3))
    t = get_template("coding")
    gen.generate_from_template(t, 1, fn)
    assert seen["prompt"] == t.prefix
    assert seen["max_tokens"] == 128
    assert seen["temperature"] == 0.3


def test_export_jsonl_returns_count(tmp_path: Path):
    gen = MagpieGenerator(MagpieConfig())
    examples = [MagpieExample("i1", "r1", "general"), MagpieExample("i2", "r2", "coding")]
    path = tmp_path / "out.jsonl"
    n = gen.export_jsonl(examples, str(path))
    assert n == 2


def test_export_jsonl_writes_file(tmp_path: Path):
    gen = MagpieGenerator(MagpieConfig())
    examples = [MagpieExample("i", "r", "general")]
    path = tmp_path / "out.jsonl"
    gen.export_jsonl(examples, str(path))
    assert path.exists()


def test_export_jsonl_creates_parent(tmp_path: Path):
    gen = MagpieGenerator(MagpieConfig())
    path = tmp_path / "nested" / "dir" / "out.jsonl"
    gen.export_jsonl([MagpieExample("i", "r", "d")], str(path))
    assert path.exists()


def test_export_jsonl_valid_lines(tmp_path: Path):
    gen = MagpieGenerator(MagpieConfig())
    path = tmp_path / "out.jsonl"
    gen.export_jsonl([MagpieExample("i", "r", "d", "pfx")], str(path))
    line = path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["instruction"] == "i"
    assert record["response"] == "r"
    assert record["domain"] == "d"
    assert record["template_prefix"] == "pfx"


def test_load_jsonl_round_trip(tmp_path: Path):
    gen = MagpieGenerator(MagpieConfig())
    originals = [
        MagpieExample("a", "b", "coding"),
        MagpieExample("c", "d", "reasoning"),
    ]
    path = tmp_path / "out.jsonl"
    gen.export_jsonl(originals, str(path))
    loaded = gen.load_jsonl(str(path))
    assert loaded == originals


def test_load_jsonl_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps({"instruction": "i", "response": "r", "domain": "d"}) + "\n\n\n",
        encoding="utf-8",
    )
    gen = MagpieGenerator(MagpieConfig())
    assert len(gen.load_jsonl(str(path))) == 1


def test_stats_empty():
    gen = MagpieGenerator(MagpieConfig())
    s = gen.stats([])
    assert s["total"] == 0
    assert s["by_domain"] == {}
    assert s["avg_instruction_len"] == 0.0
    assert s["avg_response_len"] == 0.0


def test_stats_total():
    gen = MagpieGenerator(MagpieConfig())
    examples = [MagpieExample("xx", "yyy", "general"), MagpieExample("a", "b", "coding")]
    s = gen.stats(examples)
    assert s["total"] == 2


def test_stats_by_domain():
    gen = MagpieGenerator(MagpieConfig())
    examples = [
        MagpieExample("i", "r", "general"),
        MagpieExample("i", "r", "general"),
        MagpieExample("i", "r", "coding"),
    ]
    s = gen.stats(examples)
    assert s["by_domain"] == {"general": 2, "coding": 1}


def test_stats_avg_instruction_len():
    gen = MagpieGenerator(MagpieConfig())
    examples = [MagpieExample("ab", "x", "d"), MagpieExample("abcd", "x", "d")]
    s = gen.stats(examples)
    assert s["avg_instruction_len"] == 3.0


def test_stats_avg_response_len():
    gen = MagpieGenerator(MagpieConfig())
    examples = [MagpieExample("i", "x", "d"), MagpieExample("i", "xxxxx", "d")]
    s = gen.stats(examples)
    assert s["avg_response_len"] == 3.0


def test_split_handles_human_prefix():
    gen = MagpieGenerator(MagpieConfig())

    def fn(**_):
        return "Human: hi\nAssistant: hello"

    out = gen.generate_from_template(get_template("general"), 1, fn)
    assert out[0].instruction == "hi"
    assert out[0].response == "hello"


def test_unparseable_text_skipped():
    gen = MagpieGenerator(MagpieConfig())

    def fn(**_):
        return "no markers and no double newline"

    out = gen.generate_from_template(get_template("general"), 3, fn)
    assert out == []


def test_mixed_outputs_partial():
    calls = {"i": 0}

    def fn(**_):
        calls["i"] += 1
        if calls["i"] % 2 == 0:
            return ""
        return "User: q\nAssistant: a"

    gen = MagpieGenerator(MagpieConfig())
    out = gen.generate_from_template(get_template("general"), 4, fn)
    assert len(out) == 2
