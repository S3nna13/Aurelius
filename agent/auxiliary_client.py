"""Auxiliary LLM client for side-task model calls.

Wraps a ``generate_fn`` with lightweight convenience methods for
classification, summarization, keyword extraction, and routing.
Uses a constrained configuration (low max_tokens, temperature=0)
suitable for fast, deterministic side-tasks that don't need the
full agent loop.

Stdlib-only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class AuxiliaryConfig:
    """Configuration for side-task LLM calls."""

    max_tokens: int = 128
    temperature: float = 0.0
    system_prefix: str = (
        "You are a concise assistant. Respond with only the requested output. No explanation."
    )


_CLASSIFICATION_SYSTEM = (
    "Classify the following text into exactly one of the provided categories. "
    "Respond with only the category name, nothing else."
)

_SUMMARIZE_SYSTEM = (
    "Summarize the following text in at most {max_words} words. "
    "Respond with only the summary."
)

_EXTRACT_TAGS_SYSTEM = (
    "Extract up to {max_tags} relevant tags from the following text. "
    "Respond with a JSON array of lowercase strings, e.g. [\"tag1\", \"tag2\"]. "
    "No other output."
)

_ROUTING_SYSTEM = (
    "Given the following task, which agent personality is best suited? "
    "Respond with exactly one of: {personalities}. No other output."
)

_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


class AuxiliaryClient:
    """Lightweight LLM client for side-tasks.

    Parameters
    ----------
    generate_fn:
        A callable matching the ``generate_fn`` signature
        (``list[dict] -> str``).
    config:
        Auxiliary call configuration.
    """

    def __init__(
        self,
        generate_fn: Callable[[list[dict]], str],
        config: AuxiliaryConfig | None = None,
    ) -> None:
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        self._generate = generate_fn
        self._config: AuxiliaryConfig = config if config is not None else AuxiliaryConfig()

    def _call(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": self._config.system_prefix},
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._generate(messages).strip()

    def classify(self, text: str, categories: list[str]) -> str:
        """Classify *text* into one of *categories*.

        Returns the raw model response (expected to be a category name).
        Falls back to the first category if the response is invalid.
        """
        cats = ", ".join(categories)
        prompt = f"Categories: [{cats}]\n\nText:\n{text}"
        result = self._call(_CLASSIFICATION_SYSTEM, prompt)
        for cat in categories:
            if cat.lower() == result.lower():
                return cat
        return categories[0] if categories else result

    def summarize(self, text: str, max_words: int = 50) -> str:
        """Summarize *text* in at most *max_words* words."""
        system = _SUMMARIZE_SYSTEM.format(max_words=max_words)
        return self._call(system, text)

    def extract_tags(self, text: str, max_tags: int = 5) -> list[str]:
        """Extract up to *max_tags* tags from *text*."""
        system = _EXTRACT_TAGS_SYSTEM.format(max_tags=max_tags)
        raw = self._call(system, text)
        m = _JSON_ARRAY_RE.search(raw)
        if m is None:
            return []
        try:
            tags = json.loads(m.group())
        except json.JSONDecodeError:
            return []
        if not isinstance(tags, list):
            return []
        return [str(t).lower() for t in tags if isinstance(t, (str, int, float))][:max_tags]

    def route_personality(self, task: str, personalities: list[str]) -> str:
        """Route *task* to one of *personalities*."""
        plist = ", ".join(personalities)
        system = _ROUTING_SYSTEM.format(personalities=plist)
        result = self._call(system, task)
        for p in personalities:
            if p.lower() == result.lower():
                return p
        return personalities[0] if personalities else result


AUXILIARY_CLIENT_REGISTRY: dict[str, type[AuxiliaryClient]] = {"default": AuxiliaryClient}
