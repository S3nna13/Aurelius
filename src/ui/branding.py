"""Aurelius branding constants for the terminal/IDE UI surface.

All branding — wordmark, tagline, mascot ASCII, colour tokens, and motion
language label — is original to Aurelius. No third-party vendor, product,
or model name appears anywhere in this module. ``AureliusBranding`` is a
frozen dataclass so callers can safely pass it through registries without
risking accidental mutation, and the defaults are exposed via
``DEFAULT_BRANDING`` for direct use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AureliusBranding:
    """Immutable brand token-set for the Aurelius UI surface.

    Attributes:
        product_name: Canonical product name used in conversational copy.
        wordmark: All-caps brand wordmark for banner/title rendering.
        tagline: Short one-line descriptor rendered near the wordmark.
        mascot_name: Friendly name for the ASCII mascot companion.
        primary_glyph: Primary decorative glyph. A solid rhombus by
            default — the Aurelius "measured coin" motif.
        secondary_glyph: Lower-emphasis glyph for bullets and dividers.
        color_primary_rgb: 24-bit RGB tuple for the primary brand hue
            (muted amethyst by default).
        color_accent_rgb: 24-bit RGB tuple for the accent hue (warm
            aurum by default).
        color_dim_rgb: 24-bit RGB tuple for dim/disabled text.
        motion_language: Stable identifier for the Aurelius motion
            vocabulary. Downstream renderers can key off this string
            to select matching easings/frame rates.
    """

    product_name: str = "Aurelius"
    wordmark: str = "AURELIUS"
    tagline: str = "the stoic coder's companion"
    mascot_name: str = "Marcus"
    primary_glyph: str = "\u25c8"
    secondary_glyph: str = "\u25aa"
    color_primary_rgb: tuple[int, int, int] = (176, 140, 220)
    color_accent_rgb: tuple[int, int, int] = (220, 180, 80)
    color_dim_rgb: tuple[int, int, int] = (64, 64, 80)
    motion_language: str = "measured_stoic"


DEFAULT_BRANDING = AureliusBranding()


MASCOT_ASCII: str = (
    "  ,___,\n"
    "  (o,o)\n"
    "  /)_)\n"
    '  "" ""'
)
"""A small owl-shaped ASCII mascot for Aurelius.

Original to Aurelius — hand-drawn from generic punctuation. Four lines,
at most 8 columns wide, safe to embed in a welcome panel without extra
dependencies. The trailing quote-pair represents the owl's feet on a
branch.
"""


__all__ = [
    "AureliusBranding",
    "DEFAULT_BRANDING",
    "MASCOT_ASCII",
]
