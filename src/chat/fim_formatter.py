"""Fill-in-the-middle (FIM) formatter for Aurelius coding models.

Implements the Bavarian 2022 (arXiv:2207.14255) FIM transform used by
StarCoder, CodeLlama, and DeepSeek-Coder. A (prefix, middle, suffix)
triple is serialised to a single string using sentinel control tokens
so a plain left-to-right autoregressive LM learns to infill.

Two sentinel orderings are supported:

    PSM (prefix-suffix-middle, default):
        <fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}

    SPM (suffix-prefix-middle):
        <fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>{middle}

At training time ``middle`` is appended and the loss mask zeros out all
positions up to and including the ``<fim_middle>`` sentinel so the
model is only graded on generating the middle span. At inference time
``middle`` is omitted and the model decodes freely after
``<fim_middle>``.

The encoder is strict: prefix/middle/suffix MUST NOT embed any of the
FIM control tokens. Attempts to do so raise ``FIMFormatError`` with no
silent fallback, matching the role-confusion hygiene of the sibling
ChatML / Harmony templates.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"
FIM_PAD = "<fim_pad>"

_CONTROL_TOKENS = (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE, FIM_PAD)
_VALID_MODES = frozenset({"psm", "spm"})

# Regex used by .parse(): DOTALL so spans may contain newlines; lazy so
# the spans do not swallow later sentinels.
_PSM_RE = re.compile(
    r"^<fim_prefix>(?P<prefix>.*?)<fim_suffix>(?P<suffix>.*?)<fim_middle>(?P<middle>.*)$",
    re.DOTALL,
)
_SPM_RE = re.compile(
    r"^<fim_suffix>(?P<suffix>.*?)<fim_prefix>(?P<prefix>.*?)<fim_middle>(?P<middle>.*)$",
    re.DOTALL,
)


class FIMFormatError(ValueError):
    """Raised on invalid FIM input or malformed parse target."""


@dataclass(frozen=True)
class FIMExample:
    """An (prefix, middle, suffix) training / inference triple."""

    prefix: str
    middle: str
    suffix: str


def _reject_control_tokens(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise FIMFormatError(f"{name} must be str, got {type(value).__name__}")
    for tok in _CONTROL_TOKENS:
        if tok in value:
            raise FIMFormatError(f"{name} contains reserved FIM control token {tok!r}")


class FIMFormatter:
    """Format / parse FIM triples.

    Parameters
    ----------
    mode:
        ``"psm"`` (default) or ``"spm"``. Selects sentinel ordering.
    include_middle:
        If False, ``.format`` behaves like ``.format_for_inference`` and
        emits the trailing ``<fim_middle>`` sentinel with no middle span.
    random_mode:
        If True, each ``.format`` call independently samples PSM or SPM
        (uniform) for training-time diversity. The ``mode`` argument is
        ignored when ``random_mode`` is True.
    rng:
        Optional ``random.Random`` used when ``random_mode`` is True. A
        fresh ``random.Random()`` is constructed if not supplied.
    """

    def __init__(
        self,
        mode: str = "psm",
        include_middle: bool = True,
        random_mode: bool = False,
        rng: random.Random | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise FIMFormatError(
                f"unknown FIM mode {mode!r}; expected one of {sorted(_VALID_MODES)}"
            )
        self.mode = mode
        self.include_middle = include_middle
        self.random_mode = random_mode
        self._rng = rng if rng is not None else random.Random()

    # ------------------------------------------------------------------ format
    def _render(
        self, prefix: str, middle: str, suffix: str, mode: str, include_middle: bool
    ) -> str:
        _reject_control_tokens("prefix", prefix)
        _reject_control_tokens("suffix", suffix)
        if include_middle:
            _reject_control_tokens("middle", middle)
        if mode == "psm":
            head = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
        elif mode == "spm":
            head = f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}"
        else:  # pragma: no cover — validated in __init__
            raise FIMFormatError(f"unknown FIM mode {mode!r}")
        return head + middle if include_middle else head

    def format(self, example: FIMExample) -> str:
        if not isinstance(example, FIMExample):
            raise FIMFormatError(f"format() requires FIMExample, got {type(example).__name__}")
        mode = self._rng.choice(("psm", "spm")) if self.random_mode else self.mode
        return self._render(
            example.prefix,
            example.middle,
            example.suffix,
            mode=mode,
            include_middle=self.include_middle,
        )

    def format_for_inference(self, prefix: str, suffix: str) -> str:
        mode = self._rng.choice(("psm", "spm")) if self.random_mode else self.mode
        return self._render(prefix, "", suffix, mode=mode, include_middle=False)

    # ------------------------------------------------------------------- parse
    def parse(self, text: str) -> FIMExample:
        if not isinstance(text, str):
            raise FIMFormatError(f"parse() requires str, got {type(text).__name__}")
        # Try whichever ordering matches the first sentinel observed. This
        # makes parse() robust to random_mode-produced training data.
        if text.startswith(FIM_PREFIX):
            match = _PSM_RE.match(text)
        elif text.startswith(FIM_SUFFIX):
            match = _SPM_RE.match(text)
        else:
            raise FIMFormatError("FIM text must begin with <fim_prefix> or <fim_suffix>")
        if match is None:
            raise FIMFormatError("malformed FIM text: missing sentinels")
        prefix = match.group("prefix")
        suffix = match.group("suffix")
        middle = match.group("middle")
        # Reject nested sentinels — a correctly-formatted triple cannot
        # contain them because the encoder rejects them on the way in.
        for name, span in (("prefix", prefix), ("suffix", suffix), ("middle", middle)):
            for tok in _CONTROL_TOKENS:
                if tok in span:
                    raise FIMFormatError(f"malformed FIM text: {name} contains nested {tok!r}")
        return FIMExample(prefix=prefix, middle=middle, suffix=suffix)

    # --------------------------------------------------------------- loss mask
    @staticmethod
    def make_loss_mask(tokens: list[int], middle_token_id: int) -> list[bool]:
        """Return a per-token boolean mask.

        True at position ``i`` means token ``i`` should contribute to the
        training loss. Loss is enabled only for positions strictly after
        the first occurrence of ``middle_token_id`` (the ``<fim_middle>``
        sentinel). If the sentinel is absent, the entire mask is False.
        """
        if not isinstance(tokens, list):
            raise FIMFormatError(f"tokens must be list[int], got {type(tokens).__name__}")
        mask = [False] * len(tokens)
        found = False
        for i, tok in enumerate(tokens):
            if found:
                mask[i] = True
            elif tok == middle_token_id:
                found = True
        return mask


__all__ = [
    "FIM_PREFIX",
    "FIM_SUFFIX",
    "FIM_MIDDLE",
    "FIM_PAD",
    "FIMExample",
    "FIMFormatError",
    "FIMFormatter",
]
