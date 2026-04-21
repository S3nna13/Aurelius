"""Reasoning Level Controller — GPT-OSS-120B (arXiv:2508.10925).

Maps system-prompt "Reasoning: low/medium/high" prefix to generation
hyperparameters.

SWE-bench Verified performance gradient:
  low=47.9%  medium=52.6%  high=62.4%
"""
from __future__ import annotations

import re
from typing import Literal

ReasoningLevel = Literal["low", "medium", "high"]

LEVEL_CONFIGS: dict[str, dict] = {
    "low": {
        "temperature": 0.3,
        "max_tokens": 512,
        "top_p": 0.9,
        "reasoning_level": "low",
    },
    "medium": {
        "temperature": 0.6,
        "max_tokens": 2048,
        "top_p": 0.95,
        "reasoning_level": "medium",
    },
    "high": {
        "temperature": 1.0,
        "max_tokens": 8192,
        "top_p": 0.95,
        "reasoning_level": "high",
    },
}

_PATTERN = re.compile(r"reasoning:\s*(low|medium|high)", re.IGNORECASE)
DEFAULT_LEVEL: ReasoningLevel = "medium"


def parse_reasoning_level(system_prompt: str | None) -> dict:
    """Parse reasoning level from system prompt and return generation config dict.

    Searches for "Reasoning: low/medium/high" (case-insensitive) anywhere in
    the system prompt.  Defaults to "medium" if no match is found or if
    *system_prompt* is ``None`` / empty.

    Returns a fresh copy of the matching :data:`LEVEL_CONFIGS` entry so callers
    may mutate it without affecting shared state.
    """
    m = _PATTERN.search(system_prompt or "")
    level: str = m.group(1).lower() if m else DEFAULT_LEVEL
    return dict(LEVEL_CONFIGS[level])  # return a copy


def apply_reasoning_level(
    system_prompt: str | None,
    generation_kwargs: dict,
) -> dict:
    """Merge reasoning level config into existing generation kwargs (non-destructive).

    Keys already present in *generation_kwargs* are left unchanged so that
    explicitly-set caller values always take precedence.  Missing keys are
    populated from :func:`parse_reasoning_level`.
    """
    config = parse_reasoning_level(system_prompt)
    merged = dict(generation_kwargs)
    for k, v in config.items():
        if k not in merged:  # don't override explicitly set values
            merged[k] = v
    return merged
