"""Shared exception type for the Aurelius UI surface.

All UI-side modules (``panel_layout``, ``motion``, ``ui_surface``,
``welcome``) raise :class:`UIError` — a single typed failure mode — so
consumers can write one ``except UIError`` handler around any UI
operation without chasing a zoo of per-module subclasses. This also
keeps the rule "no silent fallbacks" easy to enforce: malformed inputs
always become a ``UIError``, never a ``ValueError`` or ``TypeError``.
"""

from __future__ import annotations


class UIError(ValueError):
    """Raised for any malformed UI input or unmet rendering precondition.

    Subclass of :class:`ValueError` so pre-existing ``except ValueError``
    code paths continue to catch it, while specific handlers can still
    switch on ``UIError`` alone.
    """


__all__ = ["UIError"]
