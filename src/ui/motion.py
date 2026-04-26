"""Motion/animation primitive for the Aurelius terminal UI surface.

A :class:`MotionSpec` describes a short, fixed-length sequence of text
frames — a blinking cursor, a thinking-dots indicator, a one-shot intro
fade. :func:`play` is a **pure function** from ``(spec, reduced_motion,
t_ms) → str`` so the UI can be unit-tested deterministically without
real time or a render loop: the caller is responsible for deciding when
to re-render.

Reduced-motion behaviour (WCAG 2.3.3 ``prefers-reduced-motion``) is a
first-class parameter: when ``reduced_motion=True`` the motion spec's
``reduced_motion_frame`` is always returned, never the animated
sequence, regardless of ``t_ms``.

The module registers three original Aurelius motions at import:
``stoic-cursor``, ``thinking-dots``, and ``welcome-fade``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ui.errors import UIError

_NAME_RE = re.compile(r"^[a-z0-9_\-]+$")

_DURATION_MIN_MS = 0
_DURATION_MAX_MS = 10_000


@dataclass(frozen=True)
class MotionSpec:
    """Immutable animation spec for a terminal UI motion.

    Attributes:
        name: Stable registry key; must match ``[a-z0-9_\\-]+``.
        duration_ms: Total playback length in milliseconds, in
            ``[0, 10000]``. A duration of zero means the motion is a
            single still frame.
        frames: Non-empty tuple of non-empty strings; each frame is
            rendered once per time-slot of ``duration_ms / len(frames)``
            ms.
        loop: If True, ``play`` wraps ``t_ms`` modulo ``duration_ms``.
            If False, ``play`` clamps to the final frame once ``t_ms``
            passes ``duration_ms``.
        reduced_motion_frame: Frame returned by :func:`play` when
            ``reduced_motion=True``. May be the empty string **only**
            if ``len(frames) == 1``; otherwise it must be a non-empty
            string (not required to be identical to one of ``frames``
            so UIs can substitute a static glyph).
    """

    name: str
    duration_ms: int
    frames: tuple[str, ...]
    loop: bool = False
    reduced_motion_frame: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise UIError("name must be a non-empty str")
        if not _NAME_RE.match(self.name):
            raise UIError(f"name={self.name!r} must match [a-z0-9_-]+")

        if not isinstance(self.duration_ms, int) or isinstance(self.duration_ms, bool):
            raise UIError("duration_ms must be an int")
        if not (_DURATION_MIN_MS <= self.duration_ms <= _DURATION_MAX_MS):
            raise UIError(
                f"duration_ms={self.duration_ms} out of range "
                f"[{_DURATION_MIN_MS}, {_DURATION_MAX_MS}]"
            )

        if not isinstance(self.frames, tuple) or len(self.frames) == 0:
            raise UIError("frames must be a non-empty tuple of str")
        for idx, f in enumerate(self.frames):
            if not isinstance(f, str) or f == "":
                raise UIError(f"frames[{idx}] must be a non-empty str, got {f!r}")

        if not isinstance(self.loop, bool):
            raise UIError("loop must be a bool")

        if not isinstance(self.reduced_motion_frame, str):
            raise UIError("reduced_motion_frame must be a str")
        if self.reduced_motion_frame == "" and len(self.frames) != 1:
            raise UIError(
                "reduced_motion_frame may be empty only when frames has "
                "exactly one entry; otherwise provide a non-empty reduced "
                "frame"
            )


def play(spec: MotionSpec, *, reduced_motion: bool = False, t_ms: int = 0) -> str:
    """Return the frame of ``spec`` to render at time ``t_ms``.

    Pure function: no sleeps, no side effects, no randomness. The
    caller drives the wall-clock.

    Args:
        spec: The :class:`MotionSpec` to sample.
        reduced_motion: If True, always return
            ``spec.reduced_motion_frame`` — or ``spec.frames[0]`` if
            ``reduced_motion_frame`` is empty and the spec has exactly
            one frame.
        t_ms: Requested time offset into the motion, in milliseconds.
            Must be a non-negative int.

    Raises:
        UIError: If any argument has the wrong type or ``t_ms`` is
            negative.
    """
    if not isinstance(spec, MotionSpec):
        raise UIError("spec must be a MotionSpec instance")
    if not isinstance(reduced_motion, bool):
        raise UIError("reduced_motion must be a bool")
    if not isinstance(t_ms, int) or isinstance(t_ms, bool):
        raise UIError("t_ms must be an int")
    if t_ms < 0:
        raise UIError(f"t_ms must be >= 0, got {t_ms}")

    if reduced_motion:
        if spec.reduced_motion_frame:
            return spec.reduced_motion_frame
        return spec.frames[0]

    n = len(spec.frames)
    if spec.duration_ms == 0:
        return spec.frames[-1]

    if spec.loop:
        eff_t = t_ms % spec.duration_ms
    else:
        if t_ms >= spec.duration_ms:
            return spec.frames[-1]
        eff_t = t_ms

    slot = spec.duration_ms / n
    idx = int(eff_t // slot)
    if idx >= n:
        idx = n - 1
    return spec.frames[idx]


MOTION_REGISTRY: dict[str, MotionSpec] = {}


def register_motion(spec: MotionSpec) -> None:
    """Register ``spec`` in :data:`MOTION_REGISTRY`.

    Raises:
        UIError: If ``spec`` is not a :class:`MotionSpec` or if its
            ``name`` is already present in the registry.
    """
    if not isinstance(spec, MotionSpec):
        raise UIError("register_motion requires a MotionSpec instance")
    if spec.name in MOTION_REGISTRY:
        raise UIError(f"motion {spec.name!r} is already registered")
    MOTION_REGISTRY[spec.name] = spec


def get_motion(name: str) -> MotionSpec:
    """Return the registered :class:`MotionSpec` for ``name``.

    Raises:
        UIError: If ``name`` is not a str or not registered.
    """
    if not isinstance(name, str):
        raise UIError("name must be a str")
    try:
        return MOTION_REGISTRY[name]
    except KeyError:
        raise UIError(f"no motion registered under {name!r}") from None


def list_motions() -> list[str]:
    """Return a sorted list of registered motion names."""
    return sorted(MOTION_REGISTRY)


_STOIC_CURSOR = MotionSpec(
    name="stoic-cursor",
    duration_ms=1200,
    frames=("\u25c8", "\u25c7", "\u25c8", "\u25c7"),
    loop=True,
    reduced_motion_frame="\u25c8",
)

_THINKING_DOTS = MotionSpec(
    name="thinking-dots",
    duration_ms=1600,
    frames=(
        ".   ",
        "..  ",
        "... ",
        " ...",
    ),
    loop=True,
    reduced_motion_frame="\u2026",
)

_WELCOME_FADE = MotionSpec(
    name="welcome-fade",
    duration_ms=2000,
    frames=("AURELIUS", "AURELIUS", "AURELIUS \u25c8"),
    loop=False,
    reduced_motion_frame="AURELIUS \u25c8",
)

register_motion(_STOIC_CURSOR)
register_motion(_THINKING_DOTS)
register_motion(_WELCOME_FADE)


__all__ = [
    "MotionSpec",
    "MOTION_REGISTRY",
    "play",
    "register_motion",
    "get_motion",
    "list_motions",
]
