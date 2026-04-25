"""Prompt template engine for dynamic prompt construction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptTemplate:
    """Template with {{variable}} substitution."""

    template: str = ""
    _pattern: re.Pattern = field(default_factory=lambda: re.compile(r"\{\{(\w+)\}\}"), repr=False)

    def render(self, **kwargs: Any) -> str:
        def replacer(match: re.Match) -> str:
            key = match.group(1)
            return str(kwargs.get(key, match.group(0)))
        return self._pattern.sub(replacer, self.template)

    def variables(self) -> list[str]:
        return list(set(self._pattern.findall(self.template)))


@dataclass
class PromptRegistry:
    templates: dict[str, PromptTemplate] = field(default_factory=dict, repr=False)

    def register(self, name: str, template: str) -> None:
        self.templates[name] = PromptTemplate(template=template)

    def render(self, template_name: str, **kwargs: Any) -> str:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            raise KeyError(f"unknown template: {template_name}")
        return tmpl.render(**kwargs)


PROMPT_REGISTRY = PromptRegistry()