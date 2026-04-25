"""Tests for src/reasoning/chain_of_thought.py"""
from __future__ import annotations

import pytest
from src.reasoning.chain_of_thought import (
    ChainOfThought,
    CoTFormat,
    CoTStep,
    COT_REGISTRY,
)


# ---------- CoTFormat enum ----------

class TestCoTFormat:
    def test_numbered_value(self):
        assert CoTFormat.NUMBERED == "numbered"

    def test_bullet_value(self):
        assert CoTFormat.BULLET == "bullet"

    def test_xml_tags_value(self):
        assert CoTFormat.XML_TAGS == "xml_tags"

    def test_freeform_value(self):
        assert CoTFormat.FREEFORM == "freeform"


# ---------- CoTStep dataclass ----------

class TestCoTStep:
    def test_fields_present(self):
        s = CoTStep(index=0, content="hello")
        assert s.index == 0
        assert s.content == "hello"

    def test_default_confidence(self):
        s = CoTStep(index=1, content="x")
        assert s.confidence == 1.0

    def test_custom_confidence(self):
        s = CoTStep(index=2, content="y", confidence=0.5)
        assert s.confidence == 0.5


# ---------- parse_steps: NUMBERED ----------

class TestParseStepsNumbered:
    def setup_method(self):
        self.cot = ChainOfThought(CoTFormat.NUMBERED)

    def test_two_steps(self):
        steps = self.cot.parse_steps("1. foo\n2. bar")
        assert len(steps) == 2

    def test_step_content(self):
        steps = self.cot.parse_steps("1. foo\n2. bar")
        assert steps[0].content == "foo"
        assert steps[1].content == "bar"

    def test_step_index(self):
        steps = self.cot.parse_steps("1. foo\n2. bar")
        assert steps[0].index == 1
        assert steps[1].index == 2

    def test_paren_delimiter(self):
        steps = self.cot.parse_steps("1) alpha\n2) beta")
        assert len(steps) == 2

    def test_empty_text(self):
        assert self.cot.parse_steps("") == []

    def test_no_match(self):
        assert self.cot.parse_steps("no numbered steps here") == []


# ---------- parse_steps: BULLET ----------

class TestParseStepsBullet:
    def setup_method(self):
        self.cot = ChainOfThought(CoTFormat.BULLET)

    def test_two_steps(self):
        steps = self.cot.parse_steps("- foo\n* bar")
        assert len(steps) == 2

    def test_content(self):
        steps = self.cot.parse_steps("- foo\n* bar")
        assert steps[0].content == "foo"
        assert steps[1].content == "bar"

    def test_bullet_symbol(self):
        steps = self.cot.parse_steps("• item")
        assert len(steps) == 1
        assert steps[0].content == "item"

    def test_empty(self):
        assert self.cot.parse_steps("") == []


# ---------- parse_steps: XML_TAGS ----------

class TestParseStepsXml:
    def setup_method(self):
        self.cot = ChainOfThought(CoTFormat.XML_TAGS)

    def test_two_steps(self):
        steps = self.cot.parse_steps("<step>foo</step><step>bar</step>")
        assert len(steps) == 2

    def test_content(self):
        steps = self.cot.parse_steps("<step>foo</step><step>bar</step>")
        assert steps[0].content == "foo"
        assert steps[1].content == "bar"

    def test_multiline_step(self):
        steps = self.cot.parse_steps("<step>line1\nline2</step>")
        assert len(steps) == 1
        assert "line1" in steps[0].content

    def test_empty(self):
        assert self.cot.parse_steps("") == []


# ---------- parse_steps: FREEFORM ----------

class TestParseStepsFreeform:
    def setup_method(self):
        self.cot = ChainOfThought(CoTFormat.FREEFORM)

    def test_two_paragraphs(self):
        steps = self.cot.parse_steps("para1\n\npara2")
        assert len(steps) == 2

    def test_content(self):
        steps = self.cot.parse_steps("hello world\n\nsecond para")
        assert steps[0].content == "hello world"
        assert steps[1].content == "second para"

    def test_single_paragraph(self):
        steps = self.cot.parse_steps("only one")
        assert len(steps) == 1

    def test_empty(self):
        assert self.cot.parse_steps("") == []


# ---------- format_prompt ----------

class TestFormatPrompt:
    def test_returns_non_empty(self):
        cot = ChainOfThought()
        result = cot.format_prompt("What is 2+2?")
        assert len(result) > 0

    def test_contains_question(self):
        cot = ChainOfThought()
        result = cot.format_prompt("What is 2+2?")
        assert "What is 2+2?" in result

    def test_all_formats_include_question(self):
        question = "Test question"
        for fmt in CoTFormat:
            cot = ChainOfThought(fmt)
            assert question in cot.format_prompt(question)


# ---------- consistency_score ----------

class TestConsistencyScore:
    def test_empty_returns_one(self):
        cot = ChainOfThought()
        assert cot.consistency_score([]) == 1.0

    def test_mean_confidence(self):
        cot = ChainOfThought()
        steps = [CoTStep(0, "a", 0.8), CoTStep(1, "b", 0.6)]
        assert abs(cot.consistency_score(steps) - 0.7) < 1e-9

    def test_all_ones(self):
        cot = ChainOfThought()
        steps = [CoTStep(i, "x") for i in range(5)]
        assert cot.consistency_score(steps) == 1.0

    def test_single_step(self):
        cot = ChainOfThought()
        steps = [CoTStep(0, "a", 0.5)]
        assert cot.consistency_score(steps) == 0.5


# ---------- to_text ----------

class TestToText:
    def test_numbered_format(self):
        cot = ChainOfThought(CoTFormat.NUMBERED)
        steps = [CoTStep(1, "first"), CoTStep(2, "second")]
        text = cot.to_text(steps)
        assert "1." in text
        assert "first" in text
        assert "2." in text

    def test_bullet_format(self):
        cot = ChainOfThought(CoTFormat.BULLET)
        steps = [CoTStep(0, "item one"), CoTStep(1, "item two")]
        text = cot.to_text(steps)
        assert text.count("- ") == 2
        assert "item one" in text

    def test_xml_format(self):
        cot = ChainOfThought(CoTFormat.XML_TAGS)
        steps = [CoTStep(0, "content")]
        text = cot.to_text(steps)
        assert "<step>" in text
        assert "content" in text

    def test_freeform_format(self):
        cot = ChainOfThought(CoTFormat.FREEFORM)
        steps = [CoTStep(0, "para one"), CoTStep(1, "para two")]
        text = cot.to_text(steps)
        assert "para one" in text
        assert "para two" in text
        assert "\n\n" in text

    def test_empty_steps(self):
        cot = ChainOfThought(CoTFormat.NUMBERED)
        assert cot.to_text([]) == ""


# ---------- COT_REGISTRY ----------

class TestCotRegistry:
    def test_has_numbered(self):
        assert "numbered" in COT_REGISTRY

    def test_has_bullet(self):
        assert "bullet" in COT_REGISTRY

    def test_has_xml(self):
        assert "xml" in COT_REGISTRY

    def test_numbered_instance(self):
        assert isinstance(COT_REGISTRY["numbered"], ChainOfThought)

    def test_bullet_instance(self):
        assert isinstance(COT_REGISTRY["bullet"], ChainOfThought)

    def test_xml_instance(self):
        assert isinstance(COT_REGISTRY["xml"], ChainOfThought)

    def test_numbered_format(self):
        assert COT_REGISTRY["numbered"].format == CoTFormat.NUMBERED

    def test_bullet_format(self):
        assert COT_REGISTRY["bullet"].format == CoTFormat.BULLET

    def test_xml_format(self):
        assert COT_REGISTRY["xml"].format == CoTFormat.XML_TAGS
