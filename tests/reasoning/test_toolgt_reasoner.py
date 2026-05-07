"""Tests for src/reasoning/toolgt_reasoner.py"""

from __future__ import annotations

from src.reasoning.toolgt_reasoner import (
    TOOLGT_REASONER,
    TOOLGT_REGISTRY,
    TemplateSlot,
    ToolGTReasoner,
    ToolGTSchema,
    ToolGTTemplate,
    classify_intent,
)


# ---------- classify_intent ----------


class TestClassifyIntent:
    def test_compare_keywords(self):
        assert classify_intent("Compare Python vs Java") == ToolGTTemplate.COMPARE_THEN_SELECT
        assert classify_intent("Which is better, A or B?") == ToolGTTemplate.COMPARE_THEN_SELECT
        assert classify_intent("Pros and cons of X") == ToolGTTemplate.COMPARE_THEN_SELECT

    def test_multi_tool_keywords(self):
        assert classify_intent("First search then calculate") == ToolGTTemplate.MULTI_TOOL_SEQUENCE
        assert classify_intent("Chain multiple tools") == ToolGTTemplate.MULTI_TOOL_SEQUENCE
        assert classify_intent("Step by step plan") == ToolGTTemplate.MULTI_TOOL_SEQUENCE

    def test_search_keywords(self):
        assert classify_intent("Search for the capital of France") == ToolGTTemplate.SEARCH_THEN_ANSWER
        assert classify_intent("Look up the weather") == ToolGTTemplate.SEARCH_THEN_ANSWER
        assert classify_intent("Retrieve user data") == ToolGTTemplate.SEARCH_THEN_ANSWER

    def test_fallback(self):
        assert classify_intent("Hello world") == ToolGTTemplate.FREEFORM
        assert classify_intent("Call a tool") == ToolGTTemplate.SEARCH_THEN_ANSWER


# ---------- ToolGTSchema / TemplateSlot ----------


class TestToolGTSchema:
    def test_search_schema_slots(self):
        schema = ToolGTReasoner().select_schema("Search for X")
        assert schema.template == ToolGTTemplate.SEARCH_THEN_ANSWER
        slot_names = [s.name for s in schema.slots]
        assert slot_names == ["think", "tool_call", "observation", "answer"]

    def test_multi_schema_slots(self):
        schema = ToolGTReasoner().select_schema("Chain tools A and B")
        assert schema.template == ToolGTTemplate.MULTI_TOOL_SEQUENCE
        slot_names = [s.name for s in schema.slots]
        assert "plan" in slot_names
        assert "tool_call_1" in slot_names
        assert "tool_call_2" in slot_names
        assert "answer" in slot_names

    def test_compare_schema_slots(self):
        schema = ToolGTReasoner().select_schema("Compare X and Y")
        assert schema.template == ToolGTTemplate.COMPARE_THEN_SELECT
        slot_names = [s.name for s in schema.slots]
        assert "options" in slot_names
        assert "criteria" in slot_names
        assert "comparison" in slot_names
        assert "selection" in slot_names


# ---------- ToolGTReasoner ----------


class TestToolGTReasoner:
    def setup_method(self):
        self.reasoner = ToolGTReasoner()

    def test_select_schema_returns_schema(self):
        schema = self.reasoner.select_schema("Search for X")
        assert isinstance(schema, ToolGTSchema)

    def test_format_prompt_contains_question(self):
        schema = self.reasoner.select_schema("Search for X")
        prompt = self.reasoner.format_prompt(schema, "Search for X")
        assert "Search for X" in prompt
        assert "<think>" in prompt
        assert "<tool_call>" in prompt

    def test_format_full_example(self):
        schema = self.reasoner.select_schema("Search for X")
        fillings = {
            "think": "plan",
            "tool_call": '{"name":"search"}',
            "observation": "result",
            "answer": "Paris",
        }
        text = self.reasoner.format_full_example(schema, fillings)
        assert "<think>plan</think>" in text
        assert '<tool_call>{"name":"search"}</tool_call>' in text
        assert "<answer>Paris</answer>" in text

    def test_parse_fillings(self):
        schema = self.reasoner.select_schema("Search for X")
        text = '<think>plan</think><tool_call>{"name":"search"}</tool_call><observation>ok</observation><answer>Paris</answer>'
        fillings = ToolGTReasoner.parse_fillings(text, schema)
        assert fillings["think"] == "plan"
        assert fillings["tool_call"] == '{"name":"search"}'
        assert fillings["observation"] == "ok"
        assert fillings["answer"] == "Paris"

    def test_parse_fillings_missing_slot(self):
        schema = self.reasoner.select_schema("Search for X")
        text = "<think>plan</think><answer>Paris</answer>"
        fillings = ToolGTReasoner.parse_fillings(text, schema)
        assert fillings["think"] == "plan"
        assert fillings["tool_call"] == ""
        assert fillings["answer"] == "Paris"

    def test_extract_tool_calls(self):
        slot_text = '{"name":"search","arguments":{"q":"hi"}}'
        calls = ToolGTReasoner.extract_tool_calls(slot_text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"

    def test_validate_completion_all_present(self):
        schema = self.reasoner.select_schema("Search for X")
        text = "<think>plan</think><tool_call>{}</tool_call><observation>ok</observation><answer>Paris</answer>"
        result = ToolGTReasoner.validate_completion(text, schema)
        assert all(result.values())

    def test_validate_completion_missing(self):
        schema = self.reasoner.select_schema("Search for X")
        text = "<think>plan</think><answer>Paris</answer>"
        result = ToolGTReasoner.validate_completion(text, schema)
        assert result["think"] is True
        assert result["tool_call"] is False
        assert result["answer"] is True

    def test_constraint_for_slot_no_encode_fn(self):
        schema = self.reasoner.select_schema("Search for X")
        cfg = self.reasoner.constraint_for_slot(schema, 0)
        # ConstraintConfig may be None if torch is unavailable in test env
        assert cfg is None or cfg.prefix_tokens is None

    def test_constraint_for_slot_with_encode_fn(self):
        schema = self.reasoner.select_schema("Search for X")
        cfg = self.reasoner.constraint_for_slot(schema, 0, encode_fn=lambda s: [1, 2, 3])
        assert cfg is None or cfg.prefix_tokens == [1, 2, 3]

    def test_slot_constraints_sequence(self):
        schema = self.reasoner.select_schema("Search for X")
        seq = self.reasoner.slot_constraints_sequence(schema, encode_fn=lambda s: [0])
        assert len(seq) == len(schema.slots)


# ---------- Registries ----------


class TestRegistries:
    def test_toolgt_registry_has_default(self):
        assert "default" in TOOLGT_REGISTRY
        assert isinstance(TOOLGT_REGISTRY["default"], ToolGTReasoner)

    def test_reasoning_registry_has_toolgt(self):
        from src.reasoning import REASONING_REGISTRY

        assert "toolgt" in REASONING_REGISTRY
