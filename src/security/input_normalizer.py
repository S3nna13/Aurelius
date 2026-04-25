"""Input normalizer for evasion-resistant text processing.

Strips zero-width characters, maps common homoglyphs to ASCII,
and applies NFC unicode normalization so that detectors downstream
see a canonical form.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass


# Common homoglyph map: non-ASCII → ASCII look-alikes
_HOMOGLYPH_MAP: dict[str, str] = {
    "а": "a",  # Cyrillic а (U+0430)
    "е": "e",  # Cyrillic е (U+0435)
    "о": "o",  # Cyrillic о (U+043E)
    "р": "p",  # Cyrillic р (U+0440)
    "ѕ": "s",  # Cyrillic ѕ (U+0455)
    "ս": "u",  # Armenian u (U+057D)
    "ⅰ": "i",  # Roman numeral one (U+2170)
    "ⅼ": "l",  # Roman numeral fifty (U+217C)
    "𝟶": "0",  # Mathematical monospace 0 (U+1D7F6)
    "𝟷": "1",  # Mathematical monospace 1 (U+1D7F7)
}

_ZERO_WIDTH_CATEGORIES: set[str] = {"Mn", "Cf"}
_ZERO_WIDTH_EXPLICIT: set[str] = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # zero width no-break space (BOM)
    "\u2060",  # word joiner
    "\u180e",  # Mongolian vowel separator
}


@dataclass(frozen=True)
class NormalizationResult:
    original: str
    normalized: str
    changes: list[str]
    zero_width_removed: int
    homoglyphs_replaced: int


class InputNormalizer:
    """Normalizes text to prevent visual-evasion attacks."""

    def normalize(self, text: str) -> NormalizationResult:
        changes: list[str] = []
        zw_removed = 0
        hg_replaced = 0

        # Step 1: NFC unicode normalization
        step1 = unicodedata.normalize("NFC", text)
        if step1 != text:
            changes.append("nfc_normalize")

        # Step 2: strip zero-width characters
        step2_chars: list[str] = []
        for ch in step1:
            if ch in _ZERO_WIDTH_EXPLICIT or unicodedata.category(ch) in _ZERO_WIDTH_CATEGORIES:
                zw_removed += 1
                continue
            step2_chars.append(ch)
        step2 = "".join(step2_chars)
        if zw_removed > 0:
            changes.append(f"strip_zero_width:{zw_removed}")

        # Step 3: homoglyph replacement
        step3_chars: list[str] = []
        for ch in step2:
            replacement = _HOMOGLYPH_MAP.get(ch)
            if replacement is not None:
                hg_replaced += 1
                step3_chars.append(replacement)
            else:
                step3_chars.append(ch)
        step3 = "".join(step3_chars)
        if hg_replaced > 0:
            changes.append(f"homoglyph_replace:{hg_replaced}")

        return NormalizationResult(
            original=text,
            normalized=step3,
            changes=changes,
            zero_width_removed=zw_removed,
            homoglyphs_replaced=hg_replaced,
        )

    def normalize_text(self, text: str) -> str:
        """Convenience: return only the normalized string."""
        return self.normalize(text).normalized


# Module-level registry
NORMALIZER_REGISTRY: dict[str, InputNormalizer] = {}
DEFAULT_INPUT_NORMALIZER = InputNormalizer()
NORMALIZER_REGISTRY["default"] = DEFAULT_INPUT_NORMALIZER
