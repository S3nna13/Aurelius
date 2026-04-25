"""Integration tests for the Toolformer data generation pipeline.

Validates:
  - symbols exported from src.agent surface
  - TOOLFORMER_DATA_REGISTRY registration
  - config keys wired correctly
  - end-to-end pipeline: text → annotated training examples
"""

from __future__ import annotations

import src.agent as agent_surface
from src.agent.toolformer_data_gen import (
    Tool,
    ToolCallAnnotation,
    ToolformerConfig,
    ToolformerDataGenerator,
)
from src.model.config import AureliusConfig
from src.runtime.feature_flags import FeatureFlag, FEATURE_FLAG_REGISTRY


# ---------------------------------------------------------------------------
# Registry / export tests
# ---------------------------------------------------------------------------


def test_toolformer_symbols_exposed_on_agent_surface():
    """ToolformerDataGenerator and ToolCallAnnotation must be importable from surface."""
    assert hasattr(agent_surface, "ToolformerDataGenerator")
    assert hasattr(agent_surface, "ToolCallAnnotation")


def test_toolformer_data_registry_exists_and_has_entry():
    assert hasattr(agent_surface, "TOOLFORMER_DATA_REGISTRY")
    registry = agent_surface.TOOLFORMER_DATA_REGISTRY
    assert "toolformer" in registry
    assert registry["toolformer"] is ToolformerDataGenerator


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def test_aurelius_config_has_toolformer_keys():
    """AureliusConfig must expose the new toolformer config fields."""
    cfg = AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    assert hasattr(cfg, "toolformer_utility_threshold")
    assert hasattr(cfg, "enable_toolformer_data_gen")
    # New features default to OFF
    assert cfg.enable_toolformer_data_gen is False
    assert cfg.toolformer_utility_threshold == 0.1


def test_toolformer_config_constructed_from_aurelius_config():
    """ToolformerConfig can be constructed from AureliusConfig values."""
    FEATURE_FLAG_REGISTRY.register(
        FeatureFlag(
            name="model.toolformer_data_gen",
            enabled=False,
            metadata={"utility_threshold": 0.25},
        )
    )
    aurelius_cfg = AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    tf_cfg = ToolformerConfig(utility_threshold=aurelius_cfg.toolformer_utility_threshold)
    assert tf_cfg.utility_threshold == 0.25


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def _make_word_count_tool() -> Tool:
    """Tool that returns the word count of its input as a string."""
    def word_count(input: str = "") -> str:
        return str(len(input.split()))
    return Tool(name="word_count", description="Counts words", fn=word_count)


def _make_upper_tool() -> Tool:
    """Tool that uppercases its input."""
    return Tool(name="upper", description="Uppercases text", fn=lambda input="": input.upper())


def test_end_to_end_pipeline_produces_annotated_output():
    """Full pipeline: text -> annotate -> filter -> format."""
    cfg = ToolformerConfig(utility_threshold=-999.0)  # keep all
    gen = ToolformerDataGenerator(config=cfg)
    tools = [_make_word_count_tool(), _make_upper_tool()]

    text = "The quick brown fox jumps"
    anns = gen.annotate(text, tools)
    assert isinstance(anns, list)
    assert len(anns) > 0
    for ann in anns:
        assert isinstance(ann, ToolCallAnnotation)

    # filter (with threshold = -999, all should pass)
    kept = gen.filter_by_utility(anns)
    assert len(kept) == len(anns)

    formatted = gen.format_training_example(text, kept)
    assert isinstance(formatted, str)
    assert len(formatted) > len(text)  # annotations add content
    assert "[API(" in formatted


def test_end_to_end_pipeline_filters_correctly():
    """Pipeline with real threshold filters low-utility calls."""
    cfg = ToolformerConfig(utility_threshold=0.0)
    gen = ToolformerDataGenerator(config=cfg)
    text = "hello world"
    tools = [_make_upper_tool()]

    anns = gen.annotate(text, tools)
    kept = gen.filter_by_utility(anns)
    # All kept annotations must exceed threshold
    for ann in kept:
        assert ann.utility_gain > cfg.utility_threshold


def test_end_to_end_batch_pipeline():
    """Batch pipeline produces per-text annotation lists."""
    cfg = ToolformerConfig(utility_threshold=-999.0)
    gen = ToolformerDataGenerator(config=cfg)
    texts = [
        "first sentence here",
        "second sentence here with more words",
        "third",
    ]
    tools = [_make_word_count_tool()]
    batch_anns = gen.batch_annotate(texts, tools)
    assert len(batch_anns) == 3
    for i, anns in enumerate(batch_anns):
        assert isinstance(anns, list), f"text {i} result is not a list"
        formatted = gen.format_training_example(texts[i], anns)
        assert isinstance(formatted, str)


def test_end_to_end_crashing_tool_graceful():
    """Pipeline with a crashing tool does not crash; other tools still work."""

    def _crash(input=""):
        raise ValueError("simulated tool failure")

    crash_tool = Tool(name="crash", description="crashes", fn=_crash)
    good_tool = _make_upper_tool()

    cfg = ToolformerConfig(utility_threshold=-999.0)
    gen = ToolformerDataGenerator(config=cfg)
    anns = gen.annotate("test sentence", [crash_tool, good_tool])
    for ann in anns:
        assert ann.tool_name == "upper"


def test_end_to_end_registry_callable():
    """ToolformerDataGenerator can be instantiated via the registry."""
    registry = agent_surface.TOOLFORMER_DATA_REGISTRY
    GenClass = registry["toolformer"]
    gen = GenClass()
    assert isinstance(gen, ToolformerDataGenerator)
    # Smoke test
    anns = gen.annotate("quick test", [_make_upper_tool()])
    assert isinstance(anns, list)


def test_end_to_end_format_notation_structure():
    """Formatted output contains proper [API(...) -> result] tokens."""
    cfg = ToolformerConfig(utility_threshold=-999.0)
    gen = ToolformerDataGenerator(config=cfg)
    ann = ToolCallAnnotation(
        position=5,
        tool_name="upper",
        args={"input": "hello"},
        result="HELLO",
        utility_gain=0.5,
    )
    text = "greet world"
    formatted = gen.format_training_example(text, [ann])
    assert "[API(" in formatted
    assert "upper" in formatted
    assert "HELLO" in formatted
    # Result separator present
    assert " -> " in formatted or "->" in formatted
