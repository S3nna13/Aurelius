"""Synthetic rephrasing pipeline utility.

Builds prompts for rephrasing web text with a small model. Inspired by
Nemotron-Synth: mixing rephrased + original data improves benchmark diversity.

At pipeline time no model runs here — this module only builds prompts and
pairs results once a model has filled in the rephrased text externally.
"""

from __future__ import annotations

from dataclasses import dataclass, field


_DEFAULT_TEMPLATE = (
    "Rephrase the following text in different words while keeping the same meaning:"
    "\n\n{text}\n\nRephrased version:"
)


@dataclass
class RephraseConfig:
    template: str = _DEFAULT_TEMPLATE
    max_input_chars: int = 2048
    min_input_chars: int = 50


@dataclass
class RephrasedExample:
    original: str
    prompt: str
    rephrased: str = ""
    was_rephrased: bool = False


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


def rephrase_batch(
    texts: list[str],
    rephrased: list[str],
    config: RephraseConfig | None = None,
) -> list[RephrasedExample]:
    """Pair original texts with their model-generated rephrased outputs.

    ``texts`` and ``rephrased`` must have the same length.  For each pair the
    function builds the prompt (to record it) and marks ``was_rephrased=True``
    when a non-empty rephrased string is provided.

    Args:
        texts: Original source texts.
        rephrased: Model outputs, one per source text (may be empty strings).
        config: Optional :class:`RephraseConfig`.

    Returns:
        List of :class:`RephrasedExample` instances.
    """
    if config is None:
        config = RephraseConfig()

    if len(texts) != len(rephrased):
        raise ValueError(
            f"texts and rephrased must have the same length "
            f"({len(texts)} vs {len(rephrased)})"
        )

    examples: list[RephrasedExample] = []
    for original, reph in zip(texts, rephrased):
        prompt = build_rephrase_prompt(original, config)
        examples.append(
            RephrasedExample(
                original=original,
                prompt=prompt,
                rephrased=reph,
                was_rephrased=bool(reph),
            )
        )
    return examples
