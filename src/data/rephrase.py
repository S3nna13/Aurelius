"""Synthetic rephrasing pipeline utility.

Builds prompts for rephrasing web text with a small model. Inspired by
Nemotron-Synth: mixing rephrased + original data improves benchmark diversity.

At pipeline time no model runs here — this module only builds prompts.
"""

from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_TEMPLATE = (
    "Rephrase the following text in different words while keeping the same meaning:"
    "\n\n{text}\n\nRephrased version:"
)


@dataclass
class RephraseConfig:
    template: str = _DEFAULT_TEMPLATE
    max_input_chars: int = 2048
    min_input_chars: int = 50


def build_rephrase_prompt(
    text: str,
    config: RephraseConfig | None = None,
) -> str:
    """Return the rephrasing prompt for *text*, or "" if text is out of range.

    Args:
        text: The source text to rephrase.
        config: Optional :class:`RephraseConfig`; defaults are used if None.

    Returns:
        Filled-in prompt string, or "" when ``text`` falls outside the
        configured character-length window.
    """
    if config is None:
        config = RephraseConfig()

    if len(text) < config.min_input_chars or len(text) > config.max_input_chars:
        return ""

    return config.template.format(text=text)
