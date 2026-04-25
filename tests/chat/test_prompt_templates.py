"""Tests for prompt templates."""
from __future__ import annotations

import pytest

from src.chat.prompt_templates import PromptTemplate, PromptRegistry


class TestPromptTemplate:
    def test_render_simple(self):
        tmpl = PromptTemplate(template="Hello {{name}}!")
        assert tmpl.render(name="World") == "Hello World!"

    def test_missing_variable_preserved(self):
        tmpl = PromptTemplate(template="{{a}}-{{b}}")
        assert tmpl.render(a="1") == "1-{{b}}"

    def test_variables_returns_keys(self):
        tmpl = PromptTemplate(template="{{x}} and {{y}}")
        assert sorted(tmpl.variables()) == ["x", "y"]


class TestPromptRegistry:
    def test_register_and_render(self):
        reg = PromptRegistry()
        reg.register("greet", "Hi {{name}}!")
        assert reg.render("greet", name="Alice") == "Hi Alice!"

    def test_unknown_template_raises(self):
        reg = PromptRegistry()
        with pytest.raises(KeyError):
            reg.render("unknown")